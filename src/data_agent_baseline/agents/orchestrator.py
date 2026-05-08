"""Multi-agent orchestrator: Planner -> Specialist DAG -> Synthesizer.

Implements the three reasoning topologies the competition asks for:

- Sequential chain: a chain of dependent sub-tasks (linear `depends_on`).
- Branching parallel + merge: independent sub-tasks in the same layer
  run concurrently; their findings are merged at the synthesizer stage.
- Iterative loop refinement: when the synthesizer fails to produce an
  answer, we ask the planner for a refined plan (one retry).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.planner import PlannerAgent, PlannerConfig
from data_agent_baseline.agents.planning import (
    Finding,
    Plan,
    SpecialistKind,
    Subtask,
    topological_layers,
)
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.agents.specialist import SpecialistAgent, SpecialistConfig
from data_agent_baseline.agents.synthesizer import SynthesizerAgent, SynthesizerConfig
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.budget import BudgetExceeded


@dataclass(slots=True)
class OrchestratorConfig:
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    specialist: SpecialistConfig = field(default_factory=SpecialistConfig)
    synthesizer: SynthesizerConfig = field(default_factory=SynthesizerConfig)
    max_specialist_workers: int = 4
    enable_iterative_refinement: bool = True


@dataclass(slots=True)
class MultiAgentRunResult:
    """Trace of one full multi-agent run."""

    task_id: str
    answer: AnswerTable | None
    plan: Plan | None
    findings: dict[str, Finding]
    synthesizer_steps: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None
    refinement_attempts: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "findings": [finding.to_dict() for finding in self.findings.values()],
            "synthesizer_steps": list(self.synthesizer_steps),
            "failure_reason": self.failure_reason,
            "refinement_attempts": self.refinement_attempts,
            "succeeded": self.succeeded,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class MultiAgentOrchestrator:
    model: ModelAdapter
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def _planner(self) -> PlannerAgent:
        return PlannerAgent(model=self.model, config=self.config.planner)

    def _specialist(self, kind: SpecialistKind) -> SpecialistAgent:
        return SpecialistAgent(model=self.model, kind=kind, config=self.config.specialist)

    def _synthesizer(self) -> SynthesizerAgent:
        return SynthesizerAgent(model=self.model, config=self.config.synthesizer)

    # ---------------------------------------------------------------
    # Subtask execution
    # ---------------------------------------------------------------

    def _execute_subtask(
        self,
        *,
        task: PublicTask,
        subtask: Subtask,
        upstream_findings: dict[str, Finding],
    ) -> Finding:
        try:
            return self._specialist(subtask.specialist).execute(
                task=task,
                subtask=subtask,
                upstream_findings=upstream_findings,
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            return Finding(
                subtask_id=subtask.id,
                specialist=subtask.specialist,
                succeeded=False,
                summary=f"Specialist crashed: {exc}",
                failure_reason=f"specialist_runtime_error: {exc}",
                step_count=0,
            )

    def _blocked_finding(
        self,
        *,
        subtask: Subtask,
        blocked_by: list[str],
    ) -> Finding:
        deps = ", ".join(blocked_by)
        return Finding(
            subtask_id=subtask.id,
            specialist=subtask.specialist,
            succeeded=False,
            summary=f"Blocked by failed or missing dependency: {deps}",
            failure_reason=f"blocked_by_dependency:{deps}",
            step_count=0,
        )

    @staticmethod
    def _blocked_dependencies(
        subtask: Subtask,
        findings: dict[str, Finding],
    ) -> list[str]:
        blocked: list[str] = []
        for dep_id in subtask.depends_on:
            finding = findings.get(dep_id)
            if finding is None or not finding.succeeded:
                blocked.append(dep_id)
        return blocked

    def _execute_plan(self, *, task: PublicTask, plan: Plan) -> dict[str, Finding]:
        findings: dict[str, Finding] = {}
        try:
            layers = topological_layers(plan)
        except ValueError as exc:
            return {
                subtask.id: Finding(
                    subtask_id=subtask.id,
                    specialist=subtask.specialist,
                    succeeded=False,
                    summary=f"Plan DAG invalid: {exc}",
                    failure_reason=f"invalid_dag:{exc}",
                    step_count=0,
                )
                for subtask in plan.subtasks
            }

        for layer in layers:
            if not layer:
                continue
            if len(layer) == 1 or self.config.max_specialist_workers <= 1:
                for subtask in layer:
                    blocked = self._blocked_dependencies(subtask, findings)
                    if blocked:
                        findings[subtask.id] = self._blocked_finding(
                            subtask=subtask,
                            blocked_by=blocked,
                        )
                        continue
                    findings[subtask.id] = self._execute_subtask(
                        task=task,
                        subtask=subtask,
                        upstream_findings=dict(findings),
                    )
                continue

            workers = min(len(layer), self.config.max_specialist_workers)
            snapshot = dict(findings)
            runnable: list[Subtask] = []
            for subtask in layer:
                blocked = self._blocked_dependencies(subtask, snapshot)
                if blocked:
                    findings[subtask.id] = self._blocked_finding(
                        subtask=subtask,
                        blocked_by=blocked,
                    )
                else:
                    runnable.append(subtask)
            if not runnable:
                continue
            workers = min(len(runnable), self.config.max_specialist_workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_subtask = {
                    executor.submit(
                        self._execute_subtask,
                        task=task,
                        subtask=subtask,
                        upstream_findings=snapshot,
                    ): subtask
                    for subtask in runnable
                }
                for future in as_completed(future_to_subtask):
                    subtask = future_to_subtask[future]
                    findings[subtask.id] = future.result()
        return findings

    # ---------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------

    def run(self, task: PublicTask) -> MultiAgentRunResult:
        notes: list[str] = []
        try:
            plan = self._planner().plan(task)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            return MultiAgentRunResult(
                task_id=task.task_id,
                answer=None,
                plan=None,
                findings={},
                failure_reason=f"planner_error: {exc}",
                notes=[f"planner_failed: {exc}"],
            )

        findings = self._execute_plan(task=task, plan=plan)
        synth_run = self._safe_synthesize(task=task, plan=plan, findings=findings)
        synth_steps = [step.to_dict() for step in synth_run.steps]

        if synth_run.answer is not None:
            return MultiAgentRunResult(
                task_id=task.task_id,
                answer=synth_run.answer,
                plan=plan,
                findings=findings,
                synthesizer_steps=synth_steps,
                failure_reason=None,
                refinement_attempts=0,
                notes=notes,
            )

        # Iterative refinement (one shot): replan with the failure note + findings
        # surfaced as additional context.
        if not self.config.enable_iterative_refinement:
            return MultiAgentRunResult(
                task_id=task.task_id,
                answer=None,
                plan=plan,
                findings=findings,
                synthesizer_steps=synth_steps,
                failure_reason=synth_run.failure_reason
                or "Synthesizer produced no answer.",
                notes=notes,
            )

        notes.append("first_synthesis_failed_attempting_refinement")
        try:
            refined_plan = self._planner().plan(_RefinementTask(base=task, plan=plan, findings=findings))
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            return MultiAgentRunResult(
                task_id=task.task_id,
                answer=None,
                plan=plan,
                findings=findings,
                synthesizer_steps=synth_steps,
                failure_reason=f"refinement_planner_error: {exc}",
                refinement_attempts=1,
                notes=notes,
            )

        refined_findings = dict(findings)
        refined_findings.update(self._execute_plan(task=task, plan=refined_plan))
        refined_synth = self._safe_synthesize(
            task=task, plan=refined_plan, findings=refined_findings
        )
        return MultiAgentRunResult(
            task_id=task.task_id,
            answer=refined_synth.answer,
            plan=refined_plan,
            findings=refined_findings,
            synthesizer_steps=[step.to_dict() for step in refined_synth.steps],
            failure_reason=refined_synth.failure_reason
            if refined_synth.answer is None
            else None,
            refinement_attempts=1,
            notes=notes,
        )

    def _safe_synthesize(
        self, *, task: PublicTask, plan: Plan, findings: dict[str, Finding]
    ) -> AgentRunResult:
        try:
            return self._synthesizer().synthesize(task=task, plan=plan, findings=findings)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=[],
                failure_reason=f"synthesizer_error: {exc}",
            )


# ---------------------------------------------------------------------------
# A small task-proxy so the planner can see the previous plan + findings on
# the refinement pass without changing its public interface.
# ---------------------------------------------------------------------------


class _RefinementTask:
    __slots__ = ("base", "plan", "findings")

    def __init__(
        self,
        *,
        base: PublicTask,
        plan: Plan,
        findings: dict[str, Finding],
    ) -> None:
        self.base = base
        self.plan = plan
        self.findings = findings

    @property
    def task_id(self) -> str:
        return self.base.task_id

    @property
    def difficulty(self) -> str:
        return self.base.difficulty

    @property
    def question(self) -> str:
        previous_plan = "\n".join(
            f"- {subtask.id} [{subtask.specialist.value}]: {subtask.instruction}"
            for subtask in self.plan.subtasks
        )
        notes = []
        for finding in self.findings.values():
            status = "ok" if finding.succeeded else f"failed: {finding.failure_reason or 'unknown'}"
            notes.append(f"- {finding.subtask_id} [{status}]: {finding.summary[:200]}")
        notes_block = "\n".join(notes) if notes else "(no findings yet)"
        return (
            f"Original question:\n  {self.base.question}\n\n"
            "The previous plan was:\n"
            f"{previous_plan}\n\n"
            "Findings produced (some may be partial or wrong):\n"
            f"{notes_block}\n\n"
            "The synthesizer could NOT produce a final answer table from those "
            "findings. Re-plan: think about what was missing and emit a NEW JSON "
            "plan (you may reuse sub-task ids, but treat them as fresh). Keep "
            "the plan small and focused on closing the gap."
        )

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
