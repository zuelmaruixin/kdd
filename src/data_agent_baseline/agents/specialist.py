"""Specialist worker agents for the multi-agent pipeline.

Each specialist wraps the existing ReAct kernel with:
- A focused system prompt that frames the worker's role and the kind of
  finding it must produce.
- A limited tool subset so the model isn't tempted to wander.
- A custom answer-tool wrapper: the worker's terminal action is `report`,
  whose payload is a `Finding` (summary + optional small artifact table).

A specialist's `report` is intentionally NOT the final benchmark answer.
The synthesizer is the only agent that calls the official `answer` tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.planning import Finding, SpecialistKind, Subtask
from data_agent_baseline.agents.prompt import RESPONSE_EXAMPLES, build_observation_prompt
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import (
    ToolExecutionResult,
    ToolRegistry,
    ToolSpec,
    create_default_tool_registry,
)


# ---------------------------------------------------------------------------
# Per-specialist prompt + tool subset
# ---------------------------------------------------------------------------


_SCHEMA_PROMPT = """
You are the SCHEMA worker in a multi-agent Data Agent. Your only goal is
schema discovery: list available files under `context/`, peek at the
columns / fields of the candidates the planner pointed you at, and
report a compact description back.

Hard rules:
- Do NOT compute the final answer. You only report what files exist and
  what their schemas look like.
- Prefer `list_context`, `read_csv`, `read_json`, `read_doc`, and
  `inspect_sqlite_schema`. Do not run heavy SQL or Python here.
- Terminate with the `report` tool: pass a short `summary` string and an
  optional small `columns` / `rows` table (e.g. a tiny preview).
""".strip()


_SQL_PROMPT = """
You are the SQL worker. The planner gave you a sub-task that wants you
to query SQLite/DB files inside `context/`. Tools available to you:
`inspect_sqlite_schema`, `execute_context_sql`, `list_context`,
`read_doc` (for joining business-rule docs).

Hard rules:
- Use read-only SQL only (`SELECT` / `WITH` / `PRAGMA`).
- If the planner's sub-task asks for a small table, materialize it and
  pass it through `report.rows`.
- Always keep the returned table small (<= 1000 rows). Aggregate when
  possible.
- Terminate with the `report` tool.
""".strip()


_PYTHON_PROMPT = """
You are the PYTHON worker. You answer the planner's sub-task using
pandas / pure-Python over CSV / JSON / SQLite. Tools available:
`execute_python`, `read_csv`, `read_json`, `read_doc`, `list_context`,
`inspect_sqlite_schema`, `execute_context_sql` (helpful as a quick
data dump).

Hard rules:
- All file paths inside `execute_python` are relative to the task
  `context/` directory (the working directory is set for you).
- Print the table you intend to put into `report.rows` (e.g. via
  `df.to_dict(orient='split')`) before reporting, so you can re-read
  it from the observation.
- Keep output tables compact; don't dump megabytes back into the
  conversation.
- Terminate with the `report` tool.
""".strip()


_DOCUMENT_PROMPT = """
You are the DOCUMENT worker. Sub-tasks for you involve reading
Markdown / text documents (`knowledge.md`, doc/*.md, JSON product
specs) and extracting business rules, mappings, or definitions.

Hard rules:
- Prefer `read_doc`, `read_json`, and `read_csv` for previews. Do not
  do numeric aggregation here.
- Quote the sentences you used in your `summary`, then put any
  structured extraction (e.g. enum -> definition mappings) into the
  optional `columns` / `rows` table.
- Terminate with the `report` tool.
""".strip()


_GENERIC_PROMPT = """
You are a GENERIC worker for tasks that don't fit the specialist
specialties above. You have full ReAct access to all tools. Stay
focused on the planner's sub-task only; do NOT compute the final
benchmark answer.

Terminate with the `report` tool.
""".strip()


_PROMPT_BY_KIND: dict[SpecialistKind, str] = {
    SpecialistKind.SCHEMA: _SCHEMA_PROMPT,
    SpecialistKind.SQL: _SQL_PROMPT,
    SpecialistKind.PYTHON: _PYTHON_PROMPT,
    SpecialistKind.DOCUMENT: _DOCUMENT_PROMPT,
    SpecialistKind.GENERIC: _GENERIC_PROMPT,
}


_TOOL_SUBSET_BY_KIND: dict[SpecialistKind, set[str]] = {
    SpecialistKind.SCHEMA: {
        "list_context",
        "read_csv",
        "read_json",
        "read_doc",
        "inspect_sqlite_schema",
    },
    SpecialistKind.SQL: {
        "list_context",
        "read_doc",
        "inspect_sqlite_schema",
        "execute_context_sql",
    },
    SpecialistKind.PYTHON: {
        "list_context",
        "read_csv",
        "read_json",
        "read_doc",
        "inspect_sqlite_schema",
        "execute_context_sql",
        "execute_python",
    },
    SpecialistKind.DOCUMENT: {
        "list_context",
        "read_doc",
        "read_json",
        "read_csv",
    },
    SpecialistKind.GENERIC: {
        "list_context",
        "read_csv",
        "read_json",
        "read_doc",
        "inspect_sqlite_schema",
        "execute_context_sql",
        "execute_python",
    },
}


# ---------------------------------------------------------------------------
# Custom `report` terminal tool
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ReportPayload:
    summary: str
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)


def _report_handler(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    summary = str(action_input.get("summary", "")).strip()
    if not summary:
        raise ValueError("report.summary must be a non-empty string.")
    columns_raw = action_input.get("columns") or []
    rows_raw = action_input.get("rows") or []
    if not isinstance(columns_raw, list) or not all(isinstance(item, str) for item in columns_raw):
        raise ValueError("report.columns must be a list of strings (may be empty).")
    if not isinstance(rows_raw, list):
        raise ValueError("report.rows must be a list of lists (may be empty).")
    normalized_rows: list[list[Any]] = []
    for index, row in enumerate(rows_raw):
        if not isinstance(row, list):
            raise ValueError(f"report.rows[{index}] must be a list.")
        if columns_raw and len(row) != len(columns_raw):
            raise ValueError(
                f"report.rows[{index}] has {len(row)} cells but columns has {len(columns_raw)}."
            )
        normalized_rows.append(list(row))

    return ToolExecutionResult(
        ok=True,
        content={
            "status": "reported",
            "summary": summary,
            "column_count": len(columns_raw),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        # Reuse AnswerTable as a tidy carrier; the orchestrator unpacks it.
        answer=AnswerTable(columns=list(columns_raw), rows=normalized_rows),
    )


def _make_specialist_registry(
    kind: SpecialistKind,
    *,
    max_read_doc_calls: int | None = None,
) -> ToolRegistry:
    """Build a ToolRegistry whose specs/handlers are restricted to a kind."""
    base = create_default_tool_registry()
    allowed = _TOOL_SUBSET_BY_KIND[kind]
    specs = {name: spec for name, spec in base.specs.items() if name in allowed}
    handlers = {name: handler for name, handler in base.handlers.items() if name in allowed}
    if max_read_doc_calls is not None and "read_doc" in handlers:
        original_read_doc = handlers["read_doc"]
        counter = {"count": 0}

        def _limited_read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
            counter["count"] += 1
            if counter["count"] > max_read_doc_calls:
                return ToolExecutionResult(
                    ok=False,
                    content={
                        "error": (
                            f"read_doc limit exceeded: max {max_read_doc_calls}. "
                            "Report the current finding or report not_found/failure."
                        )
                    },
                )
            return original_read_doc(task, action_input)

        handlers["read_doc"] = _limited_read_doc
    # Replace the generic terminal `answer` tool with `report`.
    specs["report"] = ToolSpec(
        name="report",
        description=(
            "Submit your finding for the planner's sub-task. "
            "`summary` is required. Optionally include a small "
            "`columns` + `rows` artifact when your sub-task asks for a table."
        ),
        input_schema={
            "summary": "Short natural-language description of what you found.",
            "columns": ["optional_column_name"],
            "rows": [["optional_value_1"]],
        },
    )
    handlers["report"] = _report_handler
    return ToolRegistry(specs=specs, handlers=handlers)


# ---------------------------------------------------------------------------
# Per-specialist prompt builder
# ---------------------------------------------------------------------------


def _format_dependencies(
    subtask: Subtask,
    upstream_findings: dict[str, Finding],
    *,
    max_finding_chars: int = 1500,
) -> str:
    if not subtask.depends_on:
        return ""
    rendered_blocks: list[str] = []
    for dep_id in subtask.depends_on:
        finding = upstream_findings.get(dep_id)
        if finding is None:
            rendered_blocks.append(f"- {dep_id}: <missing upstream finding>")
            continue
        block = finding.render_for_dependency()
        if len(block) > max_finding_chars:
            block = block[:max_finding_chars] + "\n  ... (truncated)"
        rendered_blocks.append(block)
    return "Upstream findings (already computed):\n" + "\n\n".join(rendered_blocks)


def _build_subtask_prompt(
    *,
    task: PublicTask,
    subtask: Subtask,
    upstream_findings: dict[str, Finding],
) -> str:
    deps_block = _format_dependencies(subtask, upstream_findings)
    expected = (
        f"Expected output shape: {subtask.expected_output}\n"
        if subtask.expected_output
        else ""
    )
    return (
        f"Top-level question (for context only — do NOT answer it directly here):\n"
        f"  {task.question}\n\n"
        f"Your sub-task ({subtask.id}, specialist={subtask.specialist.value}):\n"
        f"  {subtask.instruction}\n\n"
        f"{expected}"
        f"{deps_block}\n\n"
        "When done, call the `report` tool with a `summary` and (if relevant) "
        "a small `columns` + `rows` table."
    ).strip()


# ---------------------------------------------------------------------------
# Public agent class
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SpecialistConfig:
    max_steps: int = 12
    sample_temperature: float | None = None
    sample_seed: int | None = None


@dataclass(slots=True)
class SpecialistAgent:
    """ReAct-backed worker that produces a `Finding` for a `Subtask`."""

    model: ModelAdapter
    kind: SpecialistKind
    config: SpecialistConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = SpecialistConfig()

    def _build_system_prompt(self) -> str:
        registry = _make_specialist_registry(self.kind)
        body = _PROMPT_BY_KIND[self.kind]
        tool_descriptions = registry.describe_for_prompt()
        return (
            f"{body}\n\n"
            "Global specialist hard rules:\n"
            "- You may call `read_doc` at most 2 times.\n"
            "- Once you find the requested evidence, immediately call `report`.\n"
            "- If the evidence is unavailable, call `report` with summary='not_found: ...'.\n"
            "- If blocked by missing upstream data or tool errors, call `report` with "
            "summary='failure: ...'.\n\n"
            "Available tools (this specialist):\n"
            f"{tool_descriptions}\n\n"
            f"{RESPONSE_EXAMPLES}\n\n"
            "Always emit one ```json fenced object per step with `thought`, "
            "`action`, `action_input`."
        )

    def execute(
        self,
        *,
        task: PublicTask,
        subtask: Subtask,
        upstream_findings: dict[str, Finding],
    ) -> Finding:
        from data_agent_baseline.progress import get_progress_logger
        logger = get_progress_logger()
        if logger is not None:
            logger.specialist_start(
                subtask_id=subtask.id,
                kind=self.kind.value,
                instruction=subtask.instruction,
            )

        registry = _make_specialist_registry(self.kind, max_read_doc_calls=2)
        system_prompt = self._build_system_prompt()

        agent = ReActAgent(
            model=self.model,
            tools=registry,
            config=ReActAgentConfig(
                max_steps=self.config.max_steps,
                sample_temperature=self.config.sample_temperature,
                sample_seed=self.config.sample_seed,
            ),
            system_prompt=system_prompt,
            stream_label_prefix=f"specialist:{self.kind.value}",
        )

        # We piggyback on ReAct.run, but the question prompt for the worker
        # is the sub-task brief, not the global task question. We do this by
        # constructing a thin PublicTask-like proxy that overrides .question.
        proxy_task = _SubtaskScopedTask(
            base=task,
            scoped_question=_build_subtask_prompt(
                task=task,
                subtask=subtask,
                upstream_findings=upstream_findings,
            ),
        )
        run_result: AgentRunResult = agent.run(proxy_task)

        # Unpack the AnswerTable carrier we used in the report handler.
        answer = run_result.answer
        if answer is None:
            finding = Finding(
                subtask_id=subtask.id,
                specialist=self.kind,
                succeeded=False,
                summary="Specialist did not call the `report` tool within max_steps.",
                failure_reason=run_result.failure_reason or "no_report_call",
                step_count=len(run_result.steps),
            )
            if logger is not None:
                logger.specialist_done(
                    subtask_id=subtask.id,
                    kind=self.kind.value,
                    succeeded=False,
                    summary=finding.summary,
                    artifact_columns=None,
                    artifact_rows=None,
                    step_count=finding.step_count,
                )
            return finding

        # The summary is the last step's action_input.summary string.
        summary = ""
        for step in reversed(run_result.steps):
            if step.action == "report":
                summary = str(step.action_input.get("summary", "")).strip()
                break
        finding = Finding(
            subtask_id=subtask.id,
            specialist=self.kind,
            succeeded=True,
            summary=summary or "(empty summary)",
            artifact_columns=list(answer.columns),
            artifact_rows=[list(row) for row in answer.rows],
            step_count=len(run_result.steps),
        )
        if logger is not None:
            logger.specialist_done(
                subtask_id=subtask.id,
                kind=self.kind.value,
                succeeded=True,
                summary=finding.summary,
                artifact_columns=finding.artifact_columns,
                artifact_rows=finding.artifact_rows,
                step_count=finding.step_count,
            )
        return finding


# ---------------------------------------------------------------------------
# PublicTask proxy that overrides `.question`
# ---------------------------------------------------------------------------


class _SubtaskScopedTask:
    """Lightweight wrapper that delegates to a real ``PublicTask`` but
    swaps out ``.question`` with the per-subtask prompt."""

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
