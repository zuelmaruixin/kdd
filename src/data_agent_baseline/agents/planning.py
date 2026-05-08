"""Shared data contracts for the multi-agent planner / executor pipeline.

The competition rubric explicitly asks for autonomous decomposition and
DAG-style reasoning, so we model the agent loop as:

    PlannerAgent          -> emits a `Plan` (list of `Subtask`s with deps)
    Specialist[s]          -> each consumes a `Subtask`, produces a `Finding`
    SynthesizerAgent       -> consumes the question + all findings, emits
                              the final `AnswerTable`.

This module just holds the dataclasses and a small DAG topological sort
helper so the orchestrator and the specialists agree on shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SpecialistKind(str, Enum):
    """The set of specialist worker agents the planner can dispatch to."""

    SCHEMA = "schema"
    SQL = "sql"
    PYTHON = "python"
    DOCUMENT = "document"
    GENERIC = "generic"


@dataclass(frozen=True, slots=True)
class Subtask:
    """A single node in the analysis plan."""

    id: str
    specialist: SpecialistKind
    instruction: str
    depends_on: tuple[str, ...] = ()
    expected_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "specialist": self.specialist.value,
            "instruction": self.instruction,
            "depends_on": list(self.depends_on),
            "expected_output": self.expected_output,
        }


@dataclass(frozen=True, slots=True)
class Plan:
    """Top-level analysis plan produced by the planner."""

    rationale: str
    subtasks: tuple[Subtask, ...]
    raw_response: str = ""

    @property
    def subtask_ids(self) -> list[str]:
        return [subtask.id for subtask in self.subtasks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "raw_response": self.raw_response,
        }


@dataclass(slots=True)
class Finding:
    """The output of one specialist run on one subtask."""

    subtask_id: str
    specialist: SpecialistKind
    succeeded: bool
    summary: str
    artifact_columns: list[str] = field(default_factory=list)
    artifact_rows: list[list[Any]] = field(default_factory=list)
    failure_reason: str | None = None
    step_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "specialist": self.specialist.value,
            "succeeded": self.succeeded,
            "summary": self.summary,
            "artifact_columns": list(self.artifact_columns),
            "artifact_rows": [list(row) for row in self.artifact_rows],
            "failure_reason": self.failure_reason,
            "step_count": self.step_count,
        }

    @property
    def has_artifact(self) -> bool:
        return bool(self.artifact_columns) and bool(self.artifact_rows)

    def render_for_dependency(self, max_rows: int = 20) -> str:
        """Compact textual representation of the finding for downstream agents."""
        head = (
            f"[finding {self.subtask_id} via {self.specialist.value}] "
            f"{'ok' if self.succeeded else 'failed'}: {self.summary.strip()}"
        )
        if not self.has_artifact:
            return head
        cols = ", ".join(self.artifact_columns)
        rows_preview = self.artifact_rows[:max_rows]
        rendered_rows = "\n".join(
            "  | " + " | ".join(str(cell) for cell in row) for row in rows_preview
        )
        truncated = "" if len(self.artifact_rows) <= max_rows else (
            f"\n  ... ({len(self.artifact_rows) - max_rows} more rows truncated)"
        )
        return f"{head}\n  columns: {cols}\n{rendered_rows}{truncated}"


# ---------------------------------------------------------------------------
# DAG helpers
# ---------------------------------------------------------------------------


def topological_layers(plan: Plan) -> list[list[Subtask]]:
    """Group plan subtasks into execution layers respecting dependencies.

    Each returned layer can run concurrently; layers run in order. Cycles
    raise ValueError so the orchestrator can fall back to a flat retry.
    """
    by_id: dict[str, Subtask] = {subtask.id: subtask for subtask in plan.subtasks}
    in_degree: dict[str, int] = {sid: 0 for sid in by_id}
    children: dict[str, list[str]] = {sid: [] for sid in by_id}
    for subtask in plan.subtasks:
        for dep in subtask.depends_on:
            if dep not in by_id:
                # Tolerate dangling deps by treating them as already-done.
                continue
            in_degree[subtask.id] += 1
            children[dep].append(subtask.id)

    layers: list[list[Subtask]] = []
    ready = [sid for sid, deg in in_degree.items() if deg == 0]
    placed: set[str] = set()
    while ready:
        layer = sorted({by_id[sid] for sid in ready}, key=lambda subtask: subtask.id)
        layers.append(list(layer))
        next_ready: list[str] = []
        for subtask in layer:
            placed.add(subtask.id)
            for child in children.get(subtask.id, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_ready.append(child)
        ready = next_ready

    if len(placed) != len(by_id):
        missing = sorted(set(by_id) - placed)
        raise ValueError(f"Plan contains a cycle or missing deps; unscheduled: {missing}")
    return layers


def serialize_findings(findings: dict[str, Finding]) -> list[dict[str, Any]]:
    return [finding.to_dict() for finding in findings.values()]


def asdict_plan(plan: Plan) -> dict[str, Any]:
    """Helper for trace serialization."""
    return asdict(plan)
