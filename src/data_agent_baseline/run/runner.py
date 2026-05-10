from __future__ import annotations

import csv
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

# ---------------------------------------------------------------------------
# trace.json schema (canonical fields written per task)
# ---------------------------------------------------------------------------
#
# Stable top-level keys (always present):
#
#   task_id                    str
#   succeeded                  bool          # combined of executor + answer_validation
#   answer                     {columns, rows} | null
#   failure_reason             str | null
#   agent_mode                 "react" | "multi_agent" | "router"
#   compiled_task              { task_type, modalities, operations,
#                                primary_tool, auxiliary_tools, source_capabilities,
#                                budget_level, max_llm_calls, max_tool_calls,
#                                ambiguity_flags, ... }
#   router_decision            { route_name, kind, model, difficulty,
#                                difficulty_source, fallback_used,
#                                cascade_attempts[], notes[] }
#   budget                     { llm_calls, tool_calls,
#                                local_repairs, reasoner_repairs,
#                                multiagent_fallbacks, events[] }
#   answer_validation          { valid, errors[], warnings[],
#                                column_count, row_count }
#   e2e_elapsed_seconds        float
#
# Conditionally-present keys:
#   tablellm_direct            { program, exec_stdout/stderr,
#                                context_manifest, raw_response,
#                                local_repair_log[] }
#   operator_executor          (alias of tablellm_direct, when invoked
#                                through OperatorExecutor)
#   multi_agent                { plan, findings[], synthesizer_steps[],
#                                refinement_attempts }
#   reasoner_repair            { attempted, succeeded, failure_reason,
#                                program, exec_stdout/stderr }
#   cross_model_verify         { applied, verifiers[], outcome,
#                                intersection.column_decisions[],
#                                primary_answer_before_intersection? }
#   self_consistency           { num_samples, samples[], decision }
#   local_score                { recall, penalty, score }   (only when gold known)
#   semantic_consistency_audit { plan_ran, plan_failure?, cheap_guard_triggered_escalation,
#                                analyst_exception_retry_succeeded,
#                                analyst_exception_fallback_used,
#                                lazy_escalation_plan_built,
#                                fallback_plan_from_cheap_guard,
#                                judge_ran, judge_attempts,
#                                judge_final_verdict, judge_repaired_code,
#                                final_gate }
#       — flat summary of the semantic-consistency machinery for this
#         task. ``final_gate`` is the quick read: "plan+judge",
#         "escalated+judge", "cheap_guard_only", or "bypassed". Written
#         by _run_operator_executor_pass (present only for routes whose
#         kind is operator_executor; other executors do not run semantic
#         consistency). On cascade, the audit reflects the LAST route's
#         behavior, matching how ``operator_executor`` itself is
#         reported.
#
# When adding new agent stages, ALWAYS write under a top-level key —
# never inline inside an existing block — so downstream tooling can
# parse trace.json without surprises.

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.orchestrator import (
    MultiAgentOrchestrator,
    MultiAgentRunResult,
    OrchestratorConfig,
)
from data_agent_baseline.agents.planner import PlannerConfig
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.router import run_router
from data_agent_baseline.agents.specialist import SpecialistConfig
from data_agent_baseline.agents.synthesizer import SynthesizerConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import AppConfig
from data_agent_baseline.eval.answer_validator import validate_answer_table
from data_agent_baseline.run.self_consistency import (  # noqa: F401  (kept for compat)
    SelfConsistencyResult,
    run_with_self_consistency,
)
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        max_tokens=config.agent.max_tokens,
        request_timeout=config.agent.request_timeout,
        max_retries=config.agent.max_retries,
        retry_backoff_seconds=config.agent.retry_backoff_seconds,
        retry_backoff_max_seconds=config.agent.retry_backoff_max_seconds,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _stamp_answer_validation(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the answer validator on a result payload and stamp the verdict.

    If validation fails, downgrade ``succeeded`` to ``False`` so callers
    (incl. the router cascade) can react. Always attaches an
    ``answer_validation`` block to the trace.
    """
    answer = payload.get("answer")
    result = validate_answer_table(answer)
    payload["answer_validation"] = result.to_dict()
    if not result.valid and payload.get("succeeded"):
        payload["succeeded"] = False
        existing_reason = payload.get("failure_reason")
        validation_msg = "; ".join(item.message for item in result.errors) or "answer_validation_failed"
        payload["failure_reason"] = (
            f"{existing_reason} | invalid_answer: {validation_msg}"
            if existing_reason
            else f"invalid_answer: {validation_msg}"
        )
    return payload


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _build_orchestrator_config(config: AppConfig) -> OrchestratorConfig:
    ma = config.agent.multi_agent
    return OrchestratorConfig(
        planner=PlannerConfig(sample_temperature=ma.planner_temperature),
        specialist=SpecialistConfig(
            max_steps=ma.specialist_max_steps,
            sample_temperature=ma.specialist_temperature,
        ),
        synthesizer=SynthesizerConfig(
            max_steps=ma.synthesizer_max_steps,
            sample_temperature=ma.synthesizer_temperature,
        ),
        max_specialist_workers=ma.max_specialist_workers,
        enable_iterative_refinement=ma.enable_iterative_refinement,
    )


def _run_react_pass(
    *,
    task: PublicTask,
    model,
    tools: ToolRegistry,
    config: AppConfig,
    sample_temperature: float | None = None,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    agent = ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(
            max_steps=config.agent.max_steps,
            sample_temperature=sample_temperature,
            sample_seed=sample_seed,
            cache_tool_results=config.agent.cache_tool_results,
        ),
    )
    payload = agent.run(task).to_dict()
    payload["agent_mode"] = "react"
    return payload


def _run_multi_agent_pass(
    *,
    task: PublicTask,
    model,
    config: AppConfig,
    sample_temperature: float | None = None,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    orch_config = _build_orchestrator_config(config)
    if sample_temperature is not None:
        orch_config.planner.sample_temperature = sample_temperature
        orch_config.specialist.sample_temperature = sample_temperature
        orch_config.synthesizer.sample_temperature = sample_temperature
    if sample_seed is not None:
        orch_config.planner.sample_seed = sample_seed
        orch_config.specialist.sample_seed = sample_seed
        orch_config.synthesizer.sample_seed = sample_seed

    orchestrator = MultiAgentOrchestrator(model=model, config=orch_config)
    result: MultiAgentRunResult = orchestrator.run(task)
    payload: dict[str, Any] = {
        "task_id": result.task_id,
        "answer": result.answer.to_dict() if result.answer is not None else None,
        "steps": list(result.synthesizer_steps),
        "failure_reason": result.failure_reason,
        "succeeded": result.succeeded,
        "agent_mode": "multi_agent",
        "multi_agent": result.to_dict(),
    }
    return payload


def _run_pass(
    *,
    task: PublicTask,
    model,
    tools: ToolRegistry,
    config: AppConfig,
    sample_temperature: float | None = None,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    if config.agent.mode.lower() == "multi_agent":
        return _run_multi_agent_pass(
            task=task,
            model=model,
            config=config,
            sample_temperature=sample_temperature,
            sample_seed=sample_seed,
        )
    return _run_react_pass(
        task=task,
        model=model,
        tools=tools,
        config=config,
        sample_temperature=sample_temperature,
        sample_seed=sample_seed,
    )


def _run_self_consistency_passes(
    *,
    task: PublicTask,
    model,
    tools: ToolRegistry,
    config: AppConfig,
) -> dict[str, Any]:
    """Run the chosen agent mode N times and column-vote the result."""
    from data_agent_baseline.benchmark.schema import AnswerTable
    from data_agent_baseline.eval.column_match import column_signature

    sc_config = config.agent.self_consistency
    num_samples = max(sc_config.num_samples, 1)
    samples: list[dict[str, Any]] = []
    for sample_index in range(num_samples):
        try:
            sample_payload = _run_pass(
                task=task,
                model=model,
                tools=tools,
                config=config,
                sample_temperature=sc_config.sample_temperature,
                sample_seed=sample_index + 1,
            )
            sample_payload["sample_index"] = sample_index
            samples.append(sample_payload)
        except Exception as exc:  # noqa: BLE001
            samples.append(
                {
                    "task_id": task.task_id,
                    "answer": None,
                    "succeeded": False,
                    "failure_reason": f"sample_runtime_error: {exc}",
                    "sample_index": sample_index,
                    "agent_mode": config.agent.mode,
                }
            )

    successful = [
        sample for sample in samples
        if sample.get("succeeded") and sample.get("answer") is not None
    ]
    if not successful:
        return {
            "task_id": task.task_id,
            "answer": None,
            "steps": [],
            "failure_reason": next(
                (sample.get("failure_reason") for sample in reversed(samples) if sample.get("failure_reason")),
                "All self-consistency samples failed.",
            ),
            "succeeded": False,
            "agent_mode": config.agent.mode,
            "self_consistency": {
                "num_samples": num_samples,
                "samples": samples,
                "decision": {"aggregator": sc_config.aggregator, "notes": ["no_successful_samples"]},
            },
        }
    if sc_config.aggregator.lower().strip() == "first_success":
        chosen = successful[0]
        return {
            "task_id": task.task_id,
            "answer": chosen["answer"],
            "steps": chosen.get("steps") or [],
            "failure_reason": None,
            "succeeded": True,
            "agent_mode": config.agent.mode,
            "self_consistency": {
                "num_samples": num_samples,
                "aggregator": "first_success",
                "chosen_sample_index": chosen.get("sample_index", 0),
                "samples": samples,
            },
        }

    # Column-signature voting (works for both react and multi_agent outputs).
    sample_columns: list[list[list[Any]]] = []
    for sample in successful:
        answer = sample["answer"]
        cols = answer.get("columns") or []
        rows = answer.get("rows") or []
        per_column: list[list[Any]] = [[] for _ in cols]
        for row in rows:
            for idx in range(len(cols)):
                per_column[idx].append(row[idx] if idx < len(row) else None)
        sample_columns.append(per_column)

    sample_signatures: list[list[tuple[str, ...]]] = []
    for cols in sample_columns:
        sigs = [
            column_signature(
                col,
                numeric_tolerance=config.scoring.numeric_tolerance,
                case_insensitive=config.scoring.case_insensitive,
                strip_whitespace=config.scoring.strip_whitespace,
            )
            for col in cols
        ]
        sample_signatures.append(sigs)

    votes: dict[tuple[str, ...], int] = {}
    for sigs in sample_signatures:
        for sig in set(sigs):
            votes[sig] = votes.get(sig, 0) + 1

    counts = [len(sigs) for sigs in sample_signatures]
    target_count = max(set(counts), key=counts.count) if counts else 0
    eligible = [
        (sig, count) for sig, count in votes.items() if count >= max(sc_config.min_votes, 1)
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    winning = [sig for sig, _ in eligible[:target_count]]
    if not winning and sample_signatures:
        winning = list(sample_signatures[0])
    winning_set = set(winning)

    best_sample_index = 0
    best_score: tuple[int, int] | None = None
    for idx, sigs in enumerate(sample_signatures):
        sig_set = set(sigs)
        coverage = len(sig_set & winning_set)
        extras = max(len(sig_set) - coverage, 0)
        score = (coverage, -extras)
        if best_score is None or score > best_score:
            best_score = score
            best_sample_index = idx

    chosen = successful[best_sample_index]
    chosen_answer = chosen["answer"]
    chosen_cols: list[str] = list(chosen_answer.get("columns") or [])
    chosen_rows: list[list[Any]] = [list(row) for row in chosen_answer.get("rows") or []]
    sigs_in_chosen = sample_signatures[best_sample_index]
    sig_to_idx: dict[tuple[str, ...], int] = {}
    for idx, sig in enumerate(sigs_in_chosen):
        sig_to_idx.setdefault(sig, idx)
    if all(sig in sig_to_idx for sig in winning):
        keep_indices = [sig_to_idx[sig] for sig in winning]
        projected_cols = [chosen_cols[i] for i in keep_indices]
        projected_rows = [[row[i] for i in keep_indices] for row in chosen_rows]
        final_answer = {"columns": projected_cols, "rows": projected_rows}
    else:
        final_answer = {"columns": chosen_cols, "rows": chosen_rows}

    return {
        "task_id": task.task_id,
        "answer": final_answer,
        "steps": chosen.get("steps") or [],
        "failure_reason": None,
        "succeeded": True,
        "agent_mode": config.agent.mode,
        "self_consistency": {
            "num_samples": num_samples,
            "aggregator": sc_config.aggregator,
            "voted_signatures": [
                {"signature_preview": list(sig)[:6], "votes": count}
                for sig, count in eligible
            ],
            "chosen_sample_index": best_sample_index,
            "samples": samples,
        },
    }


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    # Router mode: difficulty -> route -> per-route adapter + agent. The
    # router builds its own per-route OpenAIModelAdapters (so each route
    # may speak to a different endpoint), so the agent-level model + tools
    # passed in here are intentionally ignored.
    if config.agent.mode.lower() == "router":
        return run_router(
            task=task,
            agent_config=config.agent,
            numeric_tolerance=config.scoring.numeric_tolerance,
            case_insensitive=config.scoring.case_insensitive,
            strip_whitespace=config.scoring.strip_whitespace,
        )

    effective_model = model or build_model_adapter(config)
    effective_tools = tools or create_default_tool_registry()

    sc_config = config.agent.self_consistency
    if sc_config.num_samples > 1:
        sc_payload = _run_self_consistency_passes(
            task=task,
            model=effective_model,
            tools=effective_tools,
            config=config,
        )
        _stamp_answer_validation(sc_payload)
        return sc_payload

    payload = _run_pass(
        task=task,
        model=effective_model,
        tools=effective_tools,
        config=config,
    )
    _stamp_answer_validation(payload)
    return payload


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    queue: multiprocessing.Queue[Any],
    progress_events_enabled: bool = False,
    stream_enabled: bool = False,
    progress_lang: str = "en",
) -> None:
    try:
        if progress_events_enabled:
            from data_agent_baseline.progress import ProgressLogger, set_progress_logger

            set_progress_logger(ProgressLogger(enabled=True, lang=progress_lang))
        if stream_enabled:
            from data_agent_baseline.agents.model import StreamSink, set_stream_sink

            set_stream_sink(StreamSink(enabled=True))
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(task_id=task_id, config=config),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    from data_agent_baseline.agents.model import get_stream_sink
    from data_agent_baseline.progress import get_progress_logger

    progress_logger = get_progress_logger()
    progress_events_enabled = progress_logger is not None
    progress_lang = getattr(progress_logger, "lang", "en") if progress_logger is not None else "en"
    stream_sink = get_stream_sink()
    stream_enabled = stream_sink is not None and stream_sink.enabled
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue, progress_events_enabled, stream_enabled, progress_lang),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")

    if queue.empty():
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        return _failure_run_result_payload(task_id, "Task exited without returning a result.")

    result = queue.get()
    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    from data_agent_baseline.progress import get_progress_logger

    started_at = perf_counter()
    logger = get_progress_logger()
    if logger is not None:
        try:
            public_dataset = DABenchPublicDataset(config.dataset.root_path)
            task_obj = public_dataset.get_task(task_id)
            logger.task_start(
                task_id=task_id,
                difficulty=task_obj.difficulty,
                question=task_obj.question,
            )
        except Exception:  # noqa: BLE001
            pass

    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)

    if logger is not None:
        # Local score (only when gold is present, i.e. on the public split).
        try:
            from data_agent_baseline.eval.column_match import score_pair
            gold_csv = config.dataset.gold_root / task_id / "gold.csv"
            if gold_csv.exists():
                # Build a tiny in-memory pred manifest by reading the
                # answer payload directly (don't wait for prediction.csv).
                answer = run_result.get("answer") or {}
                pred_columns = list(answer.get("columns") or [])
                pred_rows = [list(r) for r in (answer.get("rows") or [])]
                if pred_columns:
                    from data_agent_baseline.eval.column_match import score_table
                    import csv as _csv
                    with gold_csv.open(newline="") as h:
                        rows = [list(r) for r in _csv.reader(h)]
                    if rows:
                        gold_header = rows[0]
                        gold_body = rows[1:]
                        gold_cols: list[list[str]] = [[] for _ in gold_header]
                        for row in gold_body:
                            for i in range(len(gold_header)):
                                gold_cols[i].append(row[i] if i < len(row) else "")
                        # Transpose pred too.
                        pred_cols_t: list[list[Any]] = [[] for _ in pred_columns]
                        for row in pred_rows:
                            for i in range(len(pred_columns)):
                                pred_cols_t[i].append(row[i] if i < len(row) else None)
                        _, recall, penalty, score = score_table(
                            pred_cols_t,
                            gold_cols,
                            redundancy_lambda=config.scoring.redundancy_lambda,
                            numeric_tolerance=config.scoring.numeric_tolerance,
                            case_insensitive=config.scoring.case_insensitive,
                            strip_whitespace=config.scoring.strip_whitespace,
                        )
                        logger.score(score=score, recall=recall, penalty=penalty)
        except Exception:  # noqa: BLE001 — never let logger crash a run
            pass

        logger.task_end(
            succeeded=bool(run_result.get("succeeded")),
            failure_reason=run_result.get("failure_reason"),
        )

    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
