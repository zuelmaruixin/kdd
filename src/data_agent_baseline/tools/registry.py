from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    grep_doc,
    head_doc,
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.knowledge import (
    consult_knowledge,
    get_helper_runtime,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    # Human-readable example payload, embedded into the system prompt by
    # ``describe_for_prompt`` so the model can see a concrete shape.
    input_schema: dict[str, Any]
    # Strict JSON Schema for the OpenAI tools API. Used by
    # ``describe_for_tool_api`` to build the ``tools=[...]`` list when the
    # React loop runs in native tool-calling mode.
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    offset = int(action_input.get("offset", 0))
    columns_only = bool(action_input.get("columns_only", False))
    return ToolExecutionResult(
        ok=True,
        content=read_csv_preview(
            task,
            path,
            max_rows=max_rows,
            offset=offset,
            columns_only=columns_only,
        ),
    )


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", action_input.get("chunk_size", 4000)))
    offset = int(action_input.get("offset", 0))
    return ToolExecutionResult(
        ok=True,
        content=read_json_preview(task, path, max_chars=max_chars, offset=offset),
    )


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", action_input.get("chunk_size", 4000)))
    offset = int(action_input.get("offset", 0))
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(task, path, max_chars=max_chars, offset=offset),
    )


def _head_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_lines = int(action_input.get("max_lines", 40))
    return ToolExecutionResult(
        ok=True,
        content=head_doc(task, path, max_lines=max_lines),
    )


def _grep_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    pattern = str(action_input["pattern"])
    max_hits = int(action_input.get("max_hits", 20))
    context_lines = int(action_input.get("context_lines", 1))
    case_insensitive = bool(action_input.get("case_insensitive", True))
    return ToolExecutionResult(
        ok=True,
        content=grep_doc(
            task,
            path,
            pattern=pattern,
            max_hits=max_hits,
            context_lines=context_lines,
            case_insensitive=case_insensitive,
        ),
    )


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sample_rows = int(action_input.get("sample_rows", 3))
    return ToolExecutionResult(
        ok=True,
        content=inspect_sqlite_schema(path, sample_rows=sample_rows),
    )


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _consult_knowledge(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    question = action_input.get("question")
    if not isinstance(question, str):
        raise ValueError("consult_knowledge.question must be a string.")
    files_raw = action_input.get("files")
    files: list[str] | None
    if files_raw is None:
        files = None
    elif isinstance(files_raw, list):
        files = [str(item) for item in files_raw if isinstance(item, (str, int))]
    else:
        raise ValueError("consult_knowledge.files must be a list of strings.")
    content = consult_knowledge(task, question=question, files=files)
    return ToolExecutionResult(ok=bool(content.get("ok")), content=content)


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def describe_for_tool_api(self) -> list[dict[str, Any]]:
        """Return the registry's tools in OpenAI Chat Completions
        ``tools=[...]`` shape, ready to hand to ``complete_with_tools``.

        Tools missing a ``parameters`` schema are emitted with an open
        object schema so the API still accepts them — that lets older
        custom registrations work, even though the React loop won't get
        strict per-tool validation."""
        tools: list[dict[str, Any]] = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            params = spec.parameters or {
                "type": "object",
                "additionalProperties": True,
            }
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": params,
                    },
                }
            )
        return tools

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        from data_agent_baseline.budget import get_budget_controller

        budget = get_budget_controller()
        if budget is not None:
            budget.consume_tool(action)
        return self.handlers[action](task, action_input)


def create_default_tool_registry(*, include_helper_tool: bool | None = None) -> ToolRegistry:
    """Build the React tool registry.

    ``include_helper_tool`` toggles registration of ``consult_knowledge``.
    Default ``None`` resolves to "register if a helper runtime is
    installed at construction time" — which lets call sites simply call
    ``create_default_tool_registry()`` after setting the helper runtime
    and get the right shape automatically.
    """
    if include_helper_tool is None:
        include_helper_tool = get_helper_runtime() is not None
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
            parameters={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Final-answer column headers.",
                    },
                    "rows": {
                        "type": "array",
                        "items": {"type": "array"},
                        "description": "List of rows; each row's length must match columns.",
                    },
                },
                "required": ["columns", "rows"],
                "additionalProperties": False,
            },
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to a sqlite/db file inside context."},
                    "sql": {"type": "string", "description": "A single read-only SQL statement."},
                    "limit": {"type": "integer", "minimum": 1, "default": 200},
                },
                "required": ["path", "sql"],
                "additionalProperties": False,
            },
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to execute. Captures stdout; cwd is the task context dir.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description=(
                "Inspect tables in a sqlite/db file inside context. Returns "
                "CREATE TABLE statements, per-column metadata (name, type, "
                "nullable, primary-key index), row counts, and up to "
                "sample_rows example rows per table."
            ),
            input_schema={
                "path": "relative/path/to/file.sqlite",
                "sample_rows": 3,
            },
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to a sqlite/db file inside context."},
                    "sample_rows": {"type": "integer", "minimum": 0, "maximum": 50, "default": 3},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
            parameters={
                "type": "object",
                "properties": {
                    "max_depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4},
                },
                "additionalProperties": False,
            },
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description=(
                "Read a preview of a CSV file inside context. Use offset to "
                "page through the rows; columns_only=true returns just the "
                "header and total row count (cheap schema confirmation "
                "without paying for any data)."
            ),
            input_schema={
                "path": "relative/path/to/file.csv",
                "max_rows": 20,
                "offset": 0,
                "columns_only": False,
            },
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to a .csv inside context."},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 20},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "columns_only": {"type": "boolean", "default": False},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a text-like document inside context. Use offset to "
                "page through a long doc (chunks are capped at ~6KB each)."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "max_chars": 4000,
                "offset": 0,
            },
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": 6000, "default": 4000},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        "read_json": ToolSpec(
            name="read_json",
            description=(
                "Read a preview of a JSON file inside context. Use offset "
                "to page through a long pretty-printed payload."
            ),
            input_schema={
                "path": "relative/path/to/file.json",
                "max_chars": 4000,
                "offset": 0,
            },
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": 6000, "default": 4000},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        "head_doc": ToolSpec(
            name="head_doc",
            description=(
                "Read the first N lines of a text document. Useful as a "
                "quick peek at a log / CSV / markdown header."
            ),
            input_schema={
                "path": "relative/path/to/file.txt",
                "max_lines": 40,
            },
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 500, "default": 40},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        "grep_doc": ToolSpec(
            name="grep_doc",
            description=(
                "Regex / substring search inside a text document. Returns "
                "matching lines with surrounding context — the right way "
                "to locate a specific rule in knowledge.md or a value in "
                "a long log without paging through the whole file."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "pattern": "regex or substring",
                "max_hits": 20,
                "context_lines": 1,
                "case_insensitive": True,
            },
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string", "description": "Regex / substring to search for."},
                    "max_hits": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                    "context_lines": {"type": "integer", "minimum": 0, "maximum": 10, "default": 1},
                    "case_insensitive": {"type": "boolean", "default": True},
                },
                "required": ["path", "pattern"],
                "additionalProperties": False,
            },
        ),
    }
    handlers = {
        "answer": _answer,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "grep_doc": _grep_doc,
        "head_doc": _head_doc,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    if include_helper_tool:
        specs["consult_knowledge"] = ToolSpec(
            name="consult_knowledge",
            description=(
                "Ask a stronger helper LLM to interpret the task's "
                "knowledge / rule / glossary docs. Provide a focused "
                "natural-language question; optionally restrict to "
                "specific files. The helper returns a short factual "
                "summary you can cite. Use this when a rule is "
                "ambiguous or paraphrased differently from the column "
                "names — DO NOT use it as a general chatbot."
            ),
            input_schema={
                "question": "What counts as a 'qualifying driver'?",
                "files": ["knowledge.md"],
            },
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "One focused natural-language question about a doc rule.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of relative doc paths to restrict the read to.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        )
        handlers["consult_knowledge"] = _consult_knowledge
    return ToolRegistry(specs=specs, handlers=handlers)
