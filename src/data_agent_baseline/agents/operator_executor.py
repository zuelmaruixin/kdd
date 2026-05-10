"""Unified operator executor — tool-first execution pipeline.

Architecture (each stage is now a dedicated object):

    ExecutionContext          — schema computed once, passed everywhere
    SemanticConsistencyPipeline.plan()         — 审题官: semantic plan before codegen
    CodegenDirectAgent.run()                   — generate + execute Python
    RepairCoordinator.run()                    — local repair → schema-retry
    _run_structured_doc_synthesis()            — fallback: LLM prose extraction
    SemanticConsistencyPipeline.judge_and_repair() — 执行官: judge + semantic repair

OperatorExecutor.run() is now a thin ~50-line orchestrator.
All implementation details live in the dedicated stage modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.execution_context import ExecutionContext
from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.record_text_classifier import (
    VERDICT_READ,
    ClassificationMeta,
    classify_record_text_query,
)
from data_agent_baseline.agents.repair_coordinator import RepairCoordinator
from data_agent_baseline.agents.semantic_guard import assess_cheap_semantic_risk
from data_agent_baseline.agents.semantic_consistency import SemanticConsistencyPipeline
from data_agent_baseline.agents.tablellm_direct import (
    CodegenDirectAgent,
    CodegenRunResult,
)
from data_agent_baseline.agents.task_compiler import CompiledTask, SourceCapability
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.budget import BudgetExceeded


@dataclass(slots=True)
class OperatorExecutor:
    model: OpenAIModelAdapter
    compiled_task: CompiledTask
    max_table_rows: int = 50
    max_input_chars: int = 12000
    python_timeout: int = 30
    sample_temperature: float | None = None
    sample_seed: int | None = None
    rag_kwargs: dict[str, Any] | None = None
    max_local_repairs: int = 3
    semantic_consistency_enabled: bool = True
    semantic_consistency_max_repairs: int = 3
    # Per-task transient state for the record_text query_type classifier.
    # ``_query_type`` is "aggregate" | "read" | None (None == classifier
    # never ran because there are no record_text files in this task).
    # ``_classifier_meta`` is the audit record surfaced to trace.json.
    _query_type: str | None = None
    _classifier_meta: ClassificationMeta | None = None

    def run(self, task: PublicTask) -> CodegenRunResult:
        """Execute the tool-first pipeline.

        Stages:
          1. Build ExecutionContext (schema once)
          2. Semantic planning (审题官, no code)
          3. Initial codegen  or  structured-doc pre-extraction
          4. Local repair → schema-retry (RepairCoordinator)
          5. Structured-doc synthesis fallback (if record_text codegen failed)
          6. Semantic consistency judge + repair when semantic plan is active
             or when the fast path needs escalation
        """
        ctx = ExecutionContext(self.compiled_task)

        # --- Phase 0: record_text query_type classification (once per task) ---
        # Decide up-front whether a table-synthesis pass is even desirable
        # for this question. For "read" (reading-comprehension) questions,
        # materializing a CSV drops the narrative context the answer
        # depends on, so we want to skip the extractor in both the
        # pre-extract (Phase 2) and post-codegen fallback (Phase 3b)
        # branches. A single classifier call feeds both decisions and is
        # persisted in trace.json via result.manifest below.
        self._query_type = None
        self._classifier_meta = None
        if self._has_record_text_docs():
            model_id = getattr(self.model, "model", "") or ""
            verdict, classifier_meta = classify_record_text_query(
                task.question,
                model=self.model,
                model_id=model_id,
                stream_label="record_text_classifier",
            )
            self._query_type = verdict
            self._classifier_meta = classifier_meta

        sc_pipeline = SemanticConsistencyPipeline(
            self.model, ctx,
            enabled=self.semantic_consistency_enabled,
            max_repairs=self.semantic_consistency_max_repairs,
        )
        coordinator = RepairCoordinator(
            self.model, ctx,
            python_timeout=self.python_timeout,
            max_local_repairs=self.max_local_repairs,
            sample_temperature=self.sample_temperature,
            sample_seed=self.sample_seed,
        )

        # --- Phase 1: semantic planning ---
        semantic_plan = sc_pipeline.plan(task)

        # --- Phase 2: initial codegen or structured-doc pre-extraction ---
        if self._should_preextract_record_text():
            result = self._run_structured_doc_synthesis(
                task=task,
                prior=CodegenRunResult(
                    answer=None,
                    succeeded=False,
                    failure_reason="structured_doc_preextract",
                ),
            ) or CodegenRunResult(
                answer=None,
                succeeded=False,
                failure_reason="structured_doc_preextract_failed",
            )
        else:
            agent = CodegenDirectAgent(
                model=self.model,
                max_table_rows=self.max_table_rows,
                max_input_chars=self.max_input_chars,
                python_timeout=self.python_timeout,
                sample_temperature=self.sample_temperature,
                sample_seed=self.sample_seed,
                rag_kwargs=self.rag_kwargs,
                compiled=self.compiled_task,
                semantic_plan=semantic_plan,
                progress_label="operator-codegen",
            )
            result = agent.run(task)

        # Surface the classifier's verdict + source branch (llm / cache /
        # fallback_keyword) into trace.json as soon as we have a result
        # object to attach it to. Attached once per task, regardless of
        # which pre-extract branch ran above.
        if self._classifier_meta is not None:
            result.manifest = list(result.manifest or []) + [
                {"record_text_classifier": self._classifier_meta.to_dict()}
            ]

        if semantic_plan is not None:
            result.manifest = list(result.manifest or []) + [{
                "semantic_consistency": "enabled",
                "plan_source": sc_pipeline.last_plan_source or "llm",
                "semantic_plan": semantic_plan,
            }]

        # --- Phase 3a: deterministic local repair ---
        result, repair_log = coordinator.local_repair_loop(
            task, result, semantic_plan=semantic_plan
        )
        if repair_log:
            result.manifest = list(result.manifest or []) + [{"local_repair_log": repair_log}]

        # --- Phase 3b: structured-doc synthesis (before schema-retry so that
        #     the synthesized CSV columns are available to the retry LLM) ---
        # Skip entirely if the query_type classifier labeled this a
        # reading-comprehension question — synthesizing a table would
        # drop the narrative context the answer depends on.
        needs_synth_fallback = (
            not result.succeeded
            and self._has_record_text_docs()
            and not self._should_preextract_record_text()
            and self._query_type != VERDICT_READ
        )
        if needs_synth_fallback:
            try:
                synth = self._run_structured_doc_synthesis(task=task, prior=result)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                result.failure_reason = f"{result.failure_reason} | structured_doc_synth_error:{exc}"
                synth = None
            if synth is not None:
                result = synth

        # --- Phase 3c: schema-guided LLM retry (after synthesis so schema is complete) ---
        if not result.succeeded:
            result = coordinator.schema_retry(task, result, semantic_plan=semantic_plan)
            result, post_schema_repair_log = coordinator.local_repair_loop(
                task, result, semantic_plan=semantic_plan
            )
            if post_schema_repair_log:
                result.manifest = list(result.manifest or []) + [
                    {"post_schema_retry_local_repair_log": post_schema_repair_log}
                ]

        # --- Phase 3d: recover semantic plan before judge runs ---
        # Two independent recovery paths both end up ensuring that judge
        # has a plan to judge against. Without one of these firing, a
        # None semantic_plan causes judge_and_repair() to silently
        # return, bypassing the entire consistency check.

        # Path 1 (analyst-exception escalation). The analyst raised
        # during Phase 1 (likely a transient LLM/network hiccup). Unlike
        # fast_path_skip or gate_skip, an exception is NOT a signal that
        # the plan is unnecessary — the task may still need semantic
        # judging. We capture the original failure, try once more with
        # force=True, and if the retry also fails we synthesize a
        # low-confidence plan so judge is forced to run. This closes the
        # hole where "analyst throws -> code self-overrides -> judge
        # skipped" let unverified answers through.
        original_plan_failure = dict(sc_pipeline.last_plan_failure or {})
        analyst_exception_escalated = False
        if (
            semantic_plan is None
            and original_plan_failure.get("stage") == "analyst_exception"
        ):
            retry_plan = sc_pipeline.plan(task, force=True)
            analyst_exception_escalated = True
            if retry_plan is not None:
                semantic_plan = retry_plan
                result.manifest = list(result.manifest or []) + [{
                    "semantic_consistency": "analyst_exception_retry_succeeded",
                    "plan_source": sc_pipeline.last_plan_source or "llm",
                    "original_plan_failure": original_plan_failure,
                    "semantic_plan": semantic_plan,
                }]
            else:
                # Retry also failed. Build a low-confidence plan so judge
                # is forced to run against something grounded (the
                # cheap-risk fallback shape already enforces
                # _force_semantic_consistency=True).
                synthetic_assessment = {
                    "should_escalate": True,
                    "score": 1.0,
                    "risks": [{
                        "code": "analyst_exception",
                        "severity": "error",
                        "message": (
                            "Semantic analyst raised an exception and a "
                            "forced retry also failed; judge must verify "
                            "the answer without an analyst-authored plan."
                        ),
                        "weight": 1.0,
                    }],
                }
                semantic_plan = self._fallback_semantic_plan(
                    task=task,
                    cheap_assessment=synthetic_assessment,
                    plan_failure=original_plan_failure,
                )
                # Overwrite the fallback tag so trace readers can tell
                # this fallback came from analyst exception, not from the
                # cheap guard's normal escalation.
                semantic_plan["_force_reason"] = "analyst_exception_fallback_plan"
                result.manifest = list(result.manifest or []) + [{
                    "semantic_consistency": "analyst_exception_fallback_plan",
                    "reason": (
                        "analyst raised during plan() and forced retry "
                        "also failed; using synthetic low-confidence plan "
                        "to keep judge in the loop"
                    ),
                    "original_plan_failure": original_plan_failure,
                    "retry_plan_failure": sc_pipeline.last_plan_failure,
                    "semantic_plan": semantic_plan,
                }]

        # Path 2 (cheap-guard lazy escalation). Unchanged from before,
        # except that if analyst exception already supplied a plan above
        # we skip this block (no point running two escalations).
        if semantic_plan is None:
            cheap_assessment = self._cheap_semantic_assessment(task, result)
        else:
            cheap_assessment = None

        if semantic_plan is None and cheap_assessment is not None:
            semantic_plan = sc_pipeline.plan(task, force=True)
            if semantic_plan is not None:
                result.manifest = list(result.manifest or []) + [{
                    "semantic_consistency": "lazy_escalation",
                    "plan_source": sc_pipeline.last_plan_source or "llm",
                    "semantic_plan": semantic_plan,
                }]
            else:
                semantic_plan = self._fallback_semantic_plan(
                    task=task,
                    cheap_assessment=cheap_assessment,
                    plan_failure=sc_pipeline.last_plan_failure,
                )
                result.manifest = list(result.manifest or []) + [{
                    "semantic_consistency": "fallback_plan_from_cheap_guard",
                    "reason": (
                        "cheap semantic guard requested escalation, but "
                        "semantic analyst did not produce a plan; using "
                        "cheap-risk fallback plan for judge"
                    ),
                    "plan_failure": sc_pipeline.last_plan_failure,
                    "semantic_plan": semantic_plan,
                }]

        # --- Phase 4: semantic consistency judge + repair ---
        # Let the judge own both successful answers and terminal error states
        # so static failures / invalid answers cannot be stamped as pass later.
        result = sc_pipeline.judge_and_repair(task, result, semantic_plan)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_record_text_docs(self) -> bool:
        return any(
            cap.kind in {"record_text", "structured_table"}
            for cap in self.compiled_task.source_capabilities
        )

    def _should_preextract_record_text(self) -> bool:
        if not self._has_record_text_docs():
            return False
        flags = set(self.compiled_task.ambiguity_flags)
        candidate = (
            self.compiled_task.task_type == "record_text_with_semantic_rule"
            or "record_extraction_required" in flags
        )
        if not candidate:
            return False
        # Reading-comprehension questions do not benefit from CSV
        # synthesis (it drops narrative context). Let those fall through
        # to the raw-codegen path; the operator prompt will then see the
        # record_text files directly.
        return self._query_type != VERDICT_READ

    def _cheap_semantic_assessment(
        self,
        task: PublicTask,
        result: CodegenRunResult,
    ) -> dict[str, Any] | None:
        if not self.semantic_consistency_enabled:
            return None
        assessment = assess_cheap_semantic_risk(
            task=task,
            compiled=self.compiled_task,
            result=result,
        )
        if assessment.should_escalate:
            payload = assessment.to_dict()
            result.manifest = list(result.manifest or []) + [
                {"cheap_semantic_assessment": payload}
            ]
            return payload
        return None

    def _fallback_semantic_plan(
        self,
        *,
        task: PublicTask,
        cheap_assessment: dict[str, Any],
        plan_failure: dict[str, Any] | None,
    ) -> dict[str, Any]:
        risks = cheap_assessment.get("risks") or []
        return {
            "schema_mapping": [],
            "join_plan": [],
            "filters": [],
            "aggregation": None,
            "output": {"question": task.question},
            "requires_rule_resolution": False,
            "unresolved_core_filters": [],
            "rule_resolution_queries": [],
            "rule_resolution": {},
            "applied_plan_overrides": [],
            "consistency_checks": [
                "Check the generated program against cheap_semantic_assessment risks.",
                "Verify schema_mapping, used_columns, join_keys, filters, and row counts.",
                (
                    "Do not pass if answer validity depends on suspicious "
                    "fallback or ambiguous grounding."
                ),
            ],
            "uncertainties": [
                risk.get("message") or risk.get("code")
                for risk in risks
                if isinstance(risk, dict)
            ],
            "confidence": "low",
            "cheap_semantic_assessment": cheap_assessment,
            "plan_failure": plan_failure or {},
            "_force_semantic_consistency": True,
            "_force_reason": "cheap_semantic_guard_fallback_plan",
        }

    def _run_structured_doc_synthesis(
        self,
        *,
        task: PublicTask,
        prior: CodegenRunResult,
    ) -> CodegenRunResult | None:
        """LLM-driven structured-prose extraction → synthesized CSVs → re-run codegen."""
        from data_agent_baseline.agents.structured_doc_executor import (
            StructuredDocExecutor,
            render_synthesis_block,
        )

        try:
            from data_agent_baseline.config import PROJECT_ROOT
            cache_dir = PROJECT_ROOT / "artifacts" / "cache" / "extraction"
        except Exception:  # noqa: BLE001
            cache_dir = None

        extractor = StructuredDocExecutor(
            model=self.model,
            compiled_task=self.compiled_task,
            cache_dir=cache_dir,
        )
        extraction = extractor.run(task)
        if not extraction.synthesized_csvs:
            prior.failure_reason = f"{prior.failure_reason} | structured_doc_synth_empty"
            return prior

        # Augment source_capabilities so static checker and next codegen prompt
        # see the synthesized CSVs alongside the original prose files.
        new_caps = list(self.compiled_task.source_capabilities)
        for original_path, synth_rel in extraction.synthesized_csvs.items():
            schema = list(extraction.schemas.get(original_path) or [])
            new_caps.append(SourceCapability(
                path=synth_rel,
                kind="csv",
                bytes=0,
                role="data",
                tool="pandas",
                columns=schema,
                row_count=int(extraction.record_counts.get(original_path, 0)),
            ))
        try:
            self.compiled_task.source_capabilities[:] = new_caps  # type: ignore[index]
        except Exception:  # noqa: BLE001
            pass

        rerun_agent = CodegenDirectAgent(
            model=self.model,
            max_table_rows=self.max_table_rows,
            max_input_chars=self.max_input_chars,
            python_timeout=self.python_timeout,
            sample_temperature=self.sample_temperature,
            sample_seed=self.sample_seed,
            rag_kwargs=None,
            compiled=self.compiled_task,
            progress_label="operator-codegen",
        )
        result = rerun_agent.run(task)
        if not result.succeeded:
            prior_fr = prior.failure_reason or ""
            result.failure_reason = (
                f"{prior_fr} | structured_doc_synth ok "
                f"({sum(extraction.record_counts.values())} records) "
                f"| post_synth: {result.failure_reason}"
            )
        result.manifest = list(result.manifest or []) + [{
            "structured_doc_synthesis": extraction.to_dict(),
            "synthesis_banner": render_synthesis_block(extraction),
        }]
        return result
