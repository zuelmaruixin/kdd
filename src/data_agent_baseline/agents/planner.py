"""Planner agent: turns a natural-language question into a Plan DAG."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.planning import Plan, SpecialistKind, Subtask
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.progress import get_progress_logger


PLANNER_SYSTEM_PROMPT = """
You are the planner inside a multi-agent Data Agent. Your one job is to
read a question + a context overview and emit a JSON plan: a small DAG of
sub-tasks that worker agents will execute in order.

Worker specialists (you must dispatch every sub-task to one of these):
- "schema": peeks at files / DB schemas to ground later steps. Use it
  for schema discovery only; do NOT make this specialist compute final
  answers.
- "sql": runs read-only SQL against SQLite/DB files in `context/`.
- "python": runs pandas / pure-Python over CSV / JSON / DB exports.
- "document": reads Markdown / text knowledge files, extracts business
  rules or entity-mapping facts.
- "generic": fallback for things that need free-form ReAct.

Hard rules:
- Every sub-task MUST be assigned to exactly one specialist.
- Use `depends_on` to express data dependencies. Independent sub-tasks
  must share a layer (no false sequential chains).
- Keep the plan small (typically 2-5 sub-tasks). Do NOT plan a final
  "answer" sub-task; a separate synthesizer will produce the final
  table from the findings you collect.
- Output ONE JSON object inside a single ```json fenced block. No prose
  before or after.
- Before planning or answering, identify the requested answer type:
count, ratio, difference, sum, average, max/min, list, boolean, etc.
Then ensure the final computation matches that answer type.


JSON schema:
{
  "rationale": "1-3 sentence high level approach.",
  "subtasks": [
    {
      "id": "s1",
      "specialist": "schema|sql|python|document|generic",
      "instruction": "What this worker should produce.",
      "depends_on": ["s0", ...],
      "expected_output": "Optional: shape of the finding."
    }
  ]
}
""".strip()


PLANNER_RESPONSE_EXAMPLE = """
Example output:
```json
{
  "rationale": "Need to join races and results on raceId, filter to 2008, then sum points per driver.",
  "subtasks": [
    {"id":"s1","specialist":"schema","instruction":"Inspect csv/results.csv columns and db/races.db schema.","depends_on":[],"expected_output":"column lists"},
    {"id":"s2","specialist":"sql","instruction":"From db/races.db, select raceId where year=2008.","depends_on":["s1"],"expected_output":"raceIds for 2008"},
    {"id":"s3","specialist":"python","instruction":"Load csv/results.csv, keep rows whose raceId is in s2, group by driverId, sum points, sort desc, take top 3.","depends_on":["s2"],"expected_output":"top-3 driverIds with summed points"}
  ]
}
```
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


def _parse_specialist(value: str) -> SpecialistKind:
    normalized = value.strip().lower()
    for kind in SpecialistKind:
        if kind.value == normalized:
            return kind
    return SpecialistKind.GENERIC


def _build_context_overview(task: PublicTask, *, max_chars: int = 8000) -> str:
    """Compact schema overview for the planner (no data rows)."""
    from data_agent_baseline.agents.context_render import render_compact_schema

    manifest = render_compact_schema(task, max_chars=max_chars)
    return manifest.rendered


@dataclass(slots=True)
class PlannerConfig:
    sample_temperature: float | None = None
    sample_seed: int | None = None
    context_max_chars: int = 8000


@dataclass(slots=True)
class PlannerAgent:
    model: ModelAdapter
    config: PlannerConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = PlannerConfig()

    def _build_messages(self, task: PublicTask) -> list[ModelMessage]:
        context_overview = _build_context_overview(
            task, max_chars=self.config.context_max_chars
        )
        user = (
            f"Question: {task.question}\n"
            f"Difficulty: {task.difficulty}\n\n"
            f"Context overview (paths under `context/`):\n{context_overview}\n\n"
            "Plan the work. Remember: ignore column NAMES in the final answer "
            "(grader matches by content), and avoid extra columns. Output ONE "
            "fenced ```json``` plan."
        )
        return [
            ModelMessage(role="system", content=PLANNER_SYSTEM_PROMPT + "\n\n" + PLANNER_RESPONSE_EXAMPLE),
            ModelMessage(role="user", content=user),
        ]

    def plan(self, task: PublicTask) -> Plan:
        messages = self._build_messages(task)
        raw = self.model.complete(
            messages,
            temperature=self.config.sample_temperature,
            seed=self.config.sample_seed,
            stream_label="planner",
        )
        body = _strip_json_fence(raw)
        try:
            payload, _ = json.JSONDecoder().raw_decode(body)
        except ValueError as exc:
            raise ValueError(f"Planner returned non-JSON: {exc}; raw={raw[:500]}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Planner output must be a JSON object.")

        rationale = str(payload.get("rationale", "")).strip()
        raw_subtasks = payload.get("subtasks") or []
        if not isinstance(raw_subtasks, list) or not raw_subtasks:
            raise ValueError("Planner output must include a non-empty 'subtasks' list.")

        subtasks: list[Subtask] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(raw_subtasks):
            if not isinstance(item, dict):
                raise ValueError(f"Subtask #{index} must be an object.")
            sid = str(item.get("id", "")).strip() or f"s{index + 1}"
            if sid in seen_ids:
                sid = f"{sid}_{index}"
            seen_ids.add(sid)
            specialist = _parse_specialist(str(item.get("specialist", "generic")))
            instruction = str(item.get("instruction", "")).strip()
            if not instruction:
                raise ValueError(f"Subtask {sid} is missing an instruction.")
            depends_on_raw = item.get("depends_on") or []
            if not isinstance(depends_on_raw, list):
                raise ValueError(f"Subtask {sid} depends_on must be a list.")
            depends_on = tuple(str(dep) for dep in depends_on_raw if isinstance(dep, str))
            expected_output = str(item.get("expected_output", "")).strip()
            subtasks.append(
                Subtask(
                    id=sid,
                    specialist=specialist,
                    instruction=instruction,
                    depends_on=depends_on,
                    expected_output=expected_output,
                )
            )

        plan = Plan(rationale=rationale, subtasks=tuple(subtasks), raw_response=raw)

        logger = get_progress_logger()
        if logger is not None:
            logger.planner_done(
                rationale=rationale,
                subtasks=[st.to_dict() for st in subtasks],
            )
        return plan
