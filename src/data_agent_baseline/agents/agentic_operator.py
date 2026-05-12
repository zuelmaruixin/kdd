"""Agentic wrapper around the tool-first operator.

The original operator executor is strong at reliable program execution,
but it can look like a fixed pipeline: compile -> codegen -> repair ->
judge. This module lifts that executor into an explicit agent loop:

    plan -> act with tools -> reflect -> optionally revise

The important distinction is that the LLM now optimizes the agent's
internal behavior, not only the input route or final output. PlannerAgent
creates a task-level decomposition, OperatorExecutor performs the tool
actions, and a reflection critic decides whether the executed program
actually follows the plan and the task evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.agents.operator_executor import OperatorExecutor
from data_agent_baseline.agents.planner import PlannerAgent, PlannerConfig
from data_agent_baseline.agents.planning import Plan
from data_agent_baseline.agents.tablellm_direct import CodegenRunResult
from data_agent_baseline.agents.task_compiler import CompiledTask
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.budget import BudgetExceeded


_REFLECTION_SYSTEM = """
You are the reflection critic inside a data-agent runtime.

You are not grading against hidden gold. Your job is to inspect the
agent's plan, generated program, debug trace, and answer preview, then
decide whether the agent should accept the answer or revise its tool
action.

Return exactly one JSON object in a ```json fenced block:
{
  "verdict": "accept|revise",
  "confidence": "high|medium|low",
  "issues": ["concrete issue, or empty if accepted"],
  "revision_instruction": "specific instruction for a retry, empty if accepted"
}

Choose "revise" only for concrete, evidence-backed problems such as:
- requested output shape is not respected,
- program uses a field not supported by SourceCapability/debug_steps,
- semantic plan or task wording requires a missing filter/join/rule,
- answer is empty despite non-empty evidence paths,
- execution trace shows suspicious fallback, zero intermediate count, or missing schema inspection.

If the result is plausible and no concrete issue is visible, choose
"accept" even if confidence is only medium.
""".strip()


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    fence = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence is not None:
        return fence.group(1).strip()
    generic = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic is not None:
        return generic.group(1).strip()
    return text


def _json_preview(value: Any, *, max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _compact_plan(plan: Plan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    payload = plan.to_dict()
    payload.pop("raw_response", None)
    return payload


def _compact_manifest(manifest: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in list(manifest or [])[-12:]:
        if not isinstance(item, dict):
            continue
        if "semantic_consistency" in item:
            compact.append({"semantic_consistency": item.get("semantic_consistency")})
        elif "cheap_semantic_assessment" in item:
            compact.append({"cheap_semantic_assessment": item.get("cheap_semantic_assessment")})
        elif "local_repair_log" in item:
            compact.append({"local_repair_log": item.get("local_repair_log")})
        else:
            compact.append({
                "path": item.get("path"),
                "kind": item.get("kind"),
                "columns": item.get("columns"),
                "row_count": item.get("row_count"),
            })
    return compact


@dataclass(slots=True)
class ReflectionDecision:
    verdict: str
    confidence: str = "medium"
    issues: list[str] = field(default_factory=list)
    revision_instruction: str = ""
    raw_response: str = ""
    parse_error: str | None = None

    @property
    def wants_revision(self) -> bool:
        return self.verdict == "revise" and bool(self.revision_instruction.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "issues": list(self.issues),
            "revision_instruction": self.revision_instruction,
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
        }


class _ReflectionScopedTask:
    """PublicTask-like wrapper that augments the question for a retry."""

    __slots__ = ("base", "scoped_question")

    def __init__(self, *, base: PublicTask, scoped_question: str) -> None:
        self.base = base
        self.scoped_question = scoped_question

    @property
    def task_id(self) -> str:
        return self.base.task_id

    @property
    def difficulty(self) -> str:
        return self.base.difficulty

    @property
    def question(self) -> str:
        return self.scoped_question

    @property
    def task_dir(self):
        return self.base.task_dir

    @property
    def context_dir(self):
        return self.base.context_dir

    @property
    def record(self):
        return self.base.record

    @property
    def assets(self):
        return self.base.assets


@dataclass(slots=True)
class AgenticOperatorExecutor:
    """Plan/act/reflect wrapper for data-agent tasks."""

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
    planner_temperature: float | None = None
    max_reflection_rounds: int = 1

    def run(self, task: PublicTask) -> CodegenRunResult:
        trace: dict[str, Any] = {
            "agent_kind": "agentic_operator",
            "loop": "plan_act_reflect",
            "planner": None,
            "reflection_rounds": [],
            "chosen": "initial",
        }

        plan = self._safe_plan(task, trace)
        initial = self._run_operator(task)
        logger = self._progress_logger()
        if logger is not None:
            logger.reflection_start()
        decision = self._safe_reflect(task=task, plan=plan, result=initial)
        if logger is not None:
            logger.reflection_done(
                verdict=decision.verdict,
                confidence=decision.confidence,
                issues=decision.issues,
                revision_instruction=decision.revision_instruction,
            )
        trace["reflection_rounds"].append({
            "round": 0,
            "decision": decision.to_dict(),
            "input_succeeded": initial.succeeded,
            "input_failure_reason": initial.failure_reason,
        })

        chosen = initial
        if (
            initial.succeeded
            and decision.wants_revision
            and self.max_reflection_rounds > 0
        ):
            retry_task = _ReflectionScopedTask(
                base=task,
                scoped_question=self._revision_question(
                    task=task,
                    plan=plan,
                    decision=decision,
                ),
            )
            try:
                revised = self._run_operator(retry_task)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                revised = CodegenRunResult(
                    answer=None,
                    succeeded=False,
                    failure_reason=f"reflection_retry_error:{exc}",
                )
            trace["reflection_rounds"].append({
                "round": 1,
                "revision_instruction": decision.revision_instruction,
                "revised_succeeded": revised.succeeded,
                "revised_failure_reason": revised.failure_reason,
            })
            if revised.succeeded:
                chosen = revised
                trace["chosen"] = "reflection_revision"
            else:
                trace["chosen"] = "initial_after_failed_revision"

        chosen.manifest = list(chosen.manifest or []) + [{"agentic_operator": trace}]
        return chosen

    @staticmethod
    def _progress_logger() -> Any:
        try:
            from data_agent_baseline.progress import get_progress_logger

            return get_progress_logger()
        except Exception:  # noqa: BLE001
            return None

    def _safe_plan(self, task: PublicTask, trace: dict[str, Any]) -> Plan | None:
        try:
            planner = PlannerAgent(
                model=self.model,
                config=PlannerConfig(sample_temperature=self.planner_temperature),
            )
            plan = planner.plan(task)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            trace["planner"] = {
                "succeeded": False,
                "failure_reason": f"planner_error:{exc}",
            }
            return None
        trace["planner"] = {
            "succeeded": True,
            "plan": _compact_plan(plan),
        }
        return plan

    def _run_operator(self, task: PublicTask) -> CodegenRunResult:
        executor = OperatorExecutor(
            model=self.model,
            compiled_task=self.compiled_task,
            max_table_rows=self.max_table_rows,
            max_input_chars=self.max_input_chars,
            python_timeout=self.python_timeout,
            sample_temperature=self.sample_temperature,
            sample_seed=self.sample_seed,
            rag_kwargs=self.rag_kwargs,
            max_local_repairs=self.max_local_repairs,
            semantic_consistency_enabled=self.semantic_consistency_enabled,
            semantic_consistency_max_repairs=self.semantic_consistency_max_repairs,
        )
        return executor.run(task)

    def _safe_reflect(
        self,
        *,
        task: PublicTask,
        plan: Plan | None,
        result: CodegenRunResult,
    ) -> ReflectionDecision:
        if not result.succeeded:
            return ReflectionDecision(
                verdict="revise",
                confidence="low",
                issues=[str(result.failure_reason or "operator failed")],
                revision_instruction=(
                    "The operator did not produce a valid answer. Repair the program "
                    "using the failure reason and execution trace."
                ),
                raw_response="",
            )
        try:
            raw = self.model.complete(
                [
                    ModelMessage(role="system", content=_REFLECTION_SYSTEM),
                    ModelMessage(
                        role="user",
                        content=self._reflection_prompt(
                            task=task,
                            plan=plan,
                            result=result,
                        ),
                    ),
                ],
                temperature=0.0,
                seed=self.sample_seed,
                stream_label="agentic-reflection",
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            return ReflectionDecision(
                verdict="accept",
                confidence="low",
                issues=[f"reflection_error:{exc}"],
                raw_response="",
                parse_error=str(exc),
            )

        try:
            payload, _ = json.JSONDecoder().raw_decode(_strip_json_fence(raw))
        except ValueError as exc:
            return ReflectionDecision(
                verdict="accept",
                confidence="low",
                issues=["reflection_parse_error"],
                raw_response=raw,
                parse_error=str(exc),
            )
        if not isinstance(payload, dict):
            return ReflectionDecision(
                verdict="accept",
                confidence="low",
                issues=["reflection_non_object"],
                raw_response=raw,
                parse_error="reflection JSON must be an object",
            )

        verdict = str(payload.get("verdict", "accept")).strip().lower()
        if verdict not in {"accept", "revise"}:
            verdict = "accept"
        issues_raw = payload.get("issues") or []
        issues = [str(item) for item in issues_raw] if isinstance(issues_raw, list) else []
        return ReflectionDecision(
            verdict=verdict,
            confidence=str(payload.get("confidence", "medium")).strip().lower() or "medium",
            issues=issues,
            revision_instruction=str(payload.get("revision_instruction", "")).strip(),
            raw_response=raw,
        )

    def _reflection_prompt(
        self,
        *,
        task: PublicTask,
        plan: Plan | None,
        result: CodegenRunResult,
    ) -> str:
        answer_payload = result.answer.to_dict() if result.answer is not None else None
        packet = {
            "task_id": task.task_id,
            "question": task.question,
            "compiled_task": self.compiled_task.to_dict(),
            "planner_plan": _compact_plan(plan),
            "operator": {
                "succeeded": result.succeeded,
                "failure_reason": result.failure_reason,
                "answer": answer_payload,
                "program": result.program,
                "exec_stdout_tail": str(result.exec_stdout)[-4000:],
                "exec_stderr_tail": str(result.exec_stderr)[-2000:],
                "manifest": _compact_manifest(result.manifest),
            },
        }
        return (
            "Inspect this data-agent run packet. Decide whether the agent "
            "should accept the answer or revise its tool action.\n\n"
            f"Run packet JSON:\n{_json_preview(packet, max_chars=18000)}"
        )

    def _revision_question(
        self,
        *,
        task: PublicTask,
        plan: Plan | None,
        decision: ReflectionDecision,
    ) -> str:
        return (
            f"{task.question}\n\n"
            "Agent reflection feedback for this retry:\n"
            f"- verdict: {decision.verdict}\n"
            f"- confidence: {decision.confidence}\n"
            f"- issues: {json.dumps(decision.issues, ensure_ascii=False)}\n"
            f"- required revision: {decision.revision_instruction}\n\n"
            "Use this feedback as an internal agent instruction. Produce the "
            "same final answer shape requested by the original question. "
            "Do not add explanatory columns.\n\n"
            f"Planner plan summary:\n{_json_preview(_compact_plan(plan), max_chars=4000)}"
        )
