"""Single-shot planner call that runs BEFORE the React loop.

The React loop is one-tool-per-turn: the model sees the task and chooses
the next tool blind, without a structured layout of what it intends to
do over the next 10 steps. For multi-step tasks this leads to wasted
budget — the model spends three turns rediscovering what files exist
and which docs matter before it starts computing.

The planner asks the same model (or a configurable separate model) to
produce a short ordered list of subtasks. The plan is injected as a
prefix message to the React loop and persisted to the trace so a human
audit can see what the model intended vs. what it actually did.

This is intentionally cheap — one LLM call per task, capped at a small
output budget — and gracefully degrades to "no plan" if anything goes
wrong. The React harness still works without a plan; the plan is a
prior, not a constraint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    enabled: bool = True
    skip_for_easy: bool = True
    max_subtasks: int = 6
    temperature: float = 0.0
    max_tokens: int = 1024


# Planner prompt is small + explicit so it works on qwen-plus / qwen-max
# without hand-holding. We keep it task-shape-aware: the planner is told
# what file kinds are in context so it can map tools to file types.
_PLANNER_SYSTEM_PROMPT = """
You are a planning assistant for a Data Agent. The agent will solve a
KDD-Cup-style data analysis question by calling tools in a React loop.
Before the agent starts, your job is to produce a SHORT ordered plan of
subtasks so the agent does not flail.

Tools available to the agent:
- list_context: list files in context.
- read_csv: preview a CSV (supports offset / columns_only).
- read_doc / head_doc / grep_doc: preview / head / regex-search a text doc.
- read_json: preview a JSON file.
- inspect_sqlite_schema: list tables, columns, sample rows of a sqlite DB.
- execute_context_sql: read-only SQL against a sqlite file.
- execute_python: arbitrary Python (pandas, numpy, json, sqlite3) inside
  the task context directory, capped at 30 seconds.
- answer: submit the final answer table — exactly once at the end.

Rules:
1. Output a JSON object with two fields:
   - "rationale": a single short string explaining the approach.
   - "subtasks": an ordered list of at most 6 items.
2. Each subtask is an object with:
   - "id": integer (1-based).
   - "goal": one short sentence describing what this step accomplishes.
   - "suggested_tools": list of tool names from the menu above.
   - "reads": optional list of file paths the step expects to consult.
3. Start with a context-listing step if you don't already know the files.
4. If knowledge.md / *rule* / *glossary* docs are present, the SECOND step
   must be to read them — those docs override general domain knowledge.
5. Plan ends with an `answer` step.
6. Do NOT pre-commit to a numeric result. The agent will compute it.
7. Output ONLY the JSON object — no prose, no markdown fences.
""".strip()


@dataclass(frozen=True, slots=True)
class PlannedSubtask:
    id: int
    goal: str
    suggested_tools: list[str]
    reads: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "suggested_tools": list(self.suggested_tools),
            "reads": list(self.reads),
        }


@dataclass(frozen=True, slots=True)
class ReactPlan:
    rationale: str
    subtasks: list[PlannedSubtask]
    raw_response: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "raw_response": self.raw_response,
            "error": self.error,
        }

    def to_user_message(self) -> str:
        """Render the plan as a chat message prefix for the React loop."""
        if not self.subtasks:
            return ""
        lines = ["Planner sketch (advisory — adapt as you learn more):"]
        if self.rationale:
            lines.append(f"Rationale: {self.rationale}")
        lines.append("Subtasks:")
        for sub in self.subtasks:
            tools = ", ".join(sub.suggested_tools) if sub.suggested_tools else "n/a"
            reads = (
                f" reads={sub.reads}" if sub.reads else ""
            )
            lines.append(
                f"  {sub.id}. {sub.goal} [tools: {tools}]{reads}"
            )
        lines.append(
            "You are NOT required to follow the plan verbatim — deviate "
            "whenever the data tells you to. Use it as a starting outline "
            "only."
        )
        return "\n".join(lines)


def _strip_fence(text: str) -> str:
    """Strip a ```json ... ``` or plain ``` ... ``` fence if present."""
    candidate = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    return candidate


def _parse_plan(raw: str, *, max_subtasks: int) -> ReactPlan:
    cleaned = _strip_fence(raw)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("planner response had no JSON object")
    payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(payload, dict):
        raise ValueError("planner response was not a JSON object")

    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)

    raw_subtasks = payload.get("subtasks") or []
    if not isinstance(raw_subtasks, list):
        raise ValueError("planner subtasks must be a list")

    subtasks: list[PlannedSubtask] = []
    for index, item in enumerate(raw_subtasks[:max_subtasks], start=1):
        if not isinstance(item, dict):
            continue
        goal = item.get("goal") or ""
        if not isinstance(goal, str) or not goal.strip():
            continue
        sub_id = item.get("id", index)
        if not isinstance(sub_id, int):
            sub_id = index
        tools_raw = item.get("suggested_tools") or []
        if not isinstance(tools_raw, list):
            tools_raw = []
        tools = [str(t) for t in tools_raw if isinstance(t, (str, int))]
        reads_raw = item.get("reads") or []
        if not isinstance(reads_raw, list):
            reads_raw = []
        reads = [str(r) for r in reads_raw if isinstance(r, (str, int))]
        subtasks.append(PlannedSubtask(
            id=sub_id,
            goal=goal.strip(),
            suggested_tools=tools,
            reads=reads,
        ))

    return ReactPlan(
        rationale=rationale.strip(),
        subtasks=subtasks,
        raw_response=raw,
    )


def _context_summary(task: Any, *, max_entries: int = 25) -> str:
    """Cheap deterministic file listing fed to the planner.

    Uses ``list_context_tree`` directly so we don't burn a model turn on
    discovery before the planner even runs.
    """
    try:
        from data_agent_baseline.tools.filesystem import list_context_tree
        tree = list_context_tree(task, max_depth=3)
    except Exception as exc:  # noqa: BLE001
        return f"(context listing unavailable: {exc})"
    entries = tree.get("entries") or []
    files = [e for e in entries if e.get("kind") == "file"]
    files.sort(key=lambda e: e.get("path", ""))
    files = files[:max_entries]
    if not files:
        return "(context is empty)"
    lines: list[str] = []
    for entry in files:
        size = entry.get("size")
        size_label = f"{size}b" if isinstance(size, int) else "?"
        lines.append(f"- {entry.get('path')} ({size_label})")
    return "\n".join(lines)


def make_plan(
    *,
    config: PlannerConfig,
    model: ModelAdapter,
    task: Any,
    stream_label: str = "react.planner",
) -> ReactPlan | None:
    """Run the pre-loop planner.

    Returns ``None`` if planning is disabled, the task is easy and
    ``skip_for_easy`` is true, or the planner call hard-fails. A plan with
    zero parsed subtasks is treated the same as no plan.
    """
    if not config.enabled:
        return None
    difficulty = getattr(task, "difficulty", None) or ""
    if config.skip_for_easy and difficulty.strip().lower() == "easy":
        return None

    question = getattr(task, "question", "") or ""
    listing = _context_summary(task)

    user_content = (
        f"Question: {question}\n"
        f"Difficulty: {difficulty}\n\n"
        f"Files in context (truncated):\n{listing}\n\n"
        "Plan the subtasks in JSON now."
    )

    messages = [
        ModelMessage(role="system", content=_PLANNER_SYSTEM_PROMPT),
        ModelMessage(role="user", content=user_content),
    ]
    try:
        raw = model.complete(
            messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream_label=stream_label,
        )
    except Exception as exc:  # noqa: BLE001
        return ReactPlan(
            rationale="",
            subtasks=[],
            raw_response="",
            error=f"planner_call_failed: {exc}",
        )

    try:
        plan = _parse_plan(raw, max_subtasks=config.max_subtasks)
    except Exception as exc:  # noqa: BLE001
        return ReactPlan(
            rationale="",
            subtasks=[],
            raw_response=raw,
            error=f"planner_parse_failed: {exc}",
        )

    if not plan.subtasks:
        return ReactPlan(
            rationale=plan.rationale,
            subtasks=[],
            raw_response=raw,
            error="planner_returned_no_subtasks",
        )

    return plan
