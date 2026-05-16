"""One-shot executable-code generation agent.

This module asks the configured Qwen model to emit one
Python program that computes the answer. The router uses it as the
tool-first codegen branch: skip ReAct, generate code in one shot, run it
locally, and lift the resulting table into an `AnswerTable`.

This file intentionally imports as little of the multi-agent stack as
possible so the codegen prompt stays separate from the ReAct prompt.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.context_render import (
    render_focused,
    render_with_rag,
    render_with_samples,
)
from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.agents.schema_grounding import (
    render_foreign_key_block,
    render_grounding_block,
)
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.budget import BudgetExceeded
from data_agent_baseline.tools.python_exec import execute_python_code


# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------


def render_context_for_codegen(
    task: PublicTask,
    *,
    max_table_rows: int,
    max_input_chars: int,
    question: str | None = None,
    rag_kwargs: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Build the question-time context block and a manifest of files used.

    Three modes:

    - ``rag_kwargs`` is set: route docs through full RAG (chunk → score
      with TF-IDF or embeddings → top-K). Best for long .docx / .md.
    - ``question`` is set: keyword-focused doc section selection.
    - else: schema + diverse samples per table only.
    """
    if rag_kwargs:
        manifest = render_with_rag(
            task,
            question=question or task.question,
            max_chars=max_input_chars,
            samples_per_table=max(1, min(max_table_rows, 6)),
            **rag_kwargs,
        )
    elif question:
        manifest = render_focused(
            task,
            question=question,
            max_chars=max_input_chars,
            samples_per_table=max(1, min(max_table_rows, 6)),
            doc_sections=3,
        )
    else:
        manifest = render_with_samples(
            task,
            max_chars=max_input_chars,
            samples_per_table=max(1, min(max_table_rows, 6)),
        )
    files_payload: list[dict[str, Any]] = [item.to_dict() for item in manifest.files]
    return manifest.rendered, files_payload


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


_CODEGEN_SYSTEM = (
    "You are an expert at writing short Python programs that answer questions "
    "about tabular and semi-structured data. You are given the question and "
    "previews of every relevant file. Write ONE Python "
    "program that computes the answer and assigns it to a variable named "
    "`answer`."
)

_CODEGEN_INSTRUCTION = """
Hard requirements for the program you write:
1. The working directory is the task's `context/` folder. All paths are
   relative to it. Use `pandas` for CSV/JSON, `sqlite3` for `.db` files.
   For JSON files whose preview/capability says top-level key `records`,
   do NOT use `pd.read_json(path)` directly; use
   `json.load(open(path))` then `pd.DataFrame(payload["records"])`.
2. Before using any dataframe column, make the schema concrete in code.
   Initialize `debug_steps = {..., "schema_inspection": {}, ...}` for
   every table/mixed task, and after each load/SQL query/JSON-to-DataFrame
   statement record the real columns, e.g.:
     `debug_steps["schema_inspection"]["patient_df"] = list(patient_df.columns)`
   This is the code equivalent of first checking `print(patient.columns)`.
   Do not write `df["SomeColumn"]` just because the question or
   `knowledge.md` uses that phrase. The final field must exist in the
   recorded real columns or in Source Capabilities.
3. The variable `answer` MUST be a `pandas.DataFrame` whose columns hold
   ONLY the columns the question asks for. Column names are ignored by
   the grader, so don't add bonus columns "for context".
   Preserve source-field granularity: if the source stores a concept in
   multiple fields (e.g. first_name + last_name), keep those as separate
   answer columns unless the question explicitly asks for one combined
   string.
   If the question says "For people/customers/records satisfying X, give
   their Y", output Y only unless it explicitly asks to list the person
   or customer identifier too.
   Do a semantic schema-linking step instead of guessing from keywords:
   identify each requested concept in the question, list plausible real
   fields from Source Capabilities / samples / knowledge.md, choose the
   best field(s), and make the code use only those chosen real fields.
   Put this mapping in `debug_steps["schema_mapping"]` when debug_steps
   is required. Each mapping entry should include:
   `concept`, `chosen_source`, `chosen_field`, `candidate_fields`, and
   `evidence`. `knowledge.md` rules are only hypotheses/semantics; they
   are never final schema proof. If several fields are plausible, inspect
   sample values and knowledge.md semantics; do not rely on a fixed
   keyword rule.
   Also record lightweight trace fields so cheap validation can verify the
   path without an LLM: `used_tables`, `used_columns`, `join_keys`,
   `filter_conditions`, `derived_fields`, and `unmapped_question_terms`.
   Use empty lists when a category does not apply.
   If the Semantic Analyst Plan contains a low-confidence filter, unresolved
   core filter, or `requires_rule_resolution=true`, resolve the rule from
   `knowledge.md` or real data before filtering. If that resolution changes
   any analyst value/source/field, record it in `debug_steps["plan_override"]`
   with `concept`, `field`, `old_value`, `new_value`, and `evidence`.
   When a semantic filter has alternatives, probe candidate row counts
   before declaring an empty result. For example, record counts for
   `Thrombosis == 2`, `Thrombosis == 3`, and `Thrombosis >= 2` when those
   are plausible alternatives. If the primary candidate gives zero rows
   and another evidence-backed candidate gives rows, use the evidence-backed
   candidate, and record the decision in `debug_steps["filters"]` and
   `debug_steps["plan_override"]`.
4. Do NOT print anything and do NOT write output files such as
   `answer.csv`, `prediction.csv`, or temporary result files. Do NOT
   call `display()` / `plt.show()` / network access. Put inspection/debug
   information into `debug_steps`; the execution harness will print it in
   a structured way and will write the final `answer` table itself.
5. Wrap the entire program in a single fenced ```python ... ``` block.
   Do NOT emit prose before or after the block.
6. The sample rows shown in the context above are ONLY for type / format
   hints. They are a tiny biased excerpt; never compute the final answer
   from them. Always re-load the full file in your program and apply
   the actual filter / aggregation / join.
7. If a Markdown/text file is labeled `record_text`, treat it as
   unstructured narrative text containing repeated record mentions:
   load the full file and extract records with regex/string parsing or
   the record-extraction pre-stage. Do not answer from the preview
   excerpt alone.
8. For tasks involving `knowledge.md`, use only the rules that are relevant
   to the question's requested concept. `knowledge.md` is authoritative for
   explicit domain rule definitions, but it is not schema authority.
   If `knowledge.md` does not define a numeric threshold, do NOT invent
   one; use explicit source-text labels/evidence instead. Treat these
   rules as semantic hypotheses that must be grounded onto real columns,
   tables, keys, or extracted record fields before final computation.
   Formulas in `knowledge.md` are not default transformations. Apply a
   formula only when the question explicitly asks for that formula's target
   metric, or when the requested output cannot be computed from a direct
   real column. If a direct real field answers the question, prefer it over
   deriving a value from a formula. Before using a formula, record
   `formula_name_or_rule`, `question_phrase_that_triggers_it`,
   `input_fields`, and `output_metric` in `debug_steps["knowledge_rules_used"]`.
   When knowledge.md provides a more specific rule than the Semantic
   Analyst Plan, the code may override the plan, but only by writing the
   override into `debug_steps["plan_override"]`; otherwise the consistency
   judge cannot distinguish a justified correction from a semantic drift.
9. Normalize common dtype mismatches before filtering/joining:
   - booleans: compare with `.astype(str).str.lower().eq("true")`
     instead of assuming the column stores the literal string "true".
   - ID joins across CSV/SQLite/JSON: normalize both join keys to the
     same type before `merge`. If you cast numeric-looking IDs to string,
     also strip the pandas float artifact `.0`, e.g.
     `.astype(str).str.replace(r"\\.0$", "", regex=True).str.strip()`;
     otherwise `163109.0` will not match `163109`.
   - if two merged tables share a non-key column name (for example
     `Diagnosis`), call `merge(..., suffixes=("_exam", "_patient"))`
     or use source-specific renames before merging; never project the
     unsuffixed column after merge.
     If the requested output field exists in both sources, decide which
     source owns the question concept before merging. For example,
     patient disease/diagnosis should come from the patient source; a
     filter field such as thrombosis should come from the examination
     source. Prefer renaming the chosen column before merge
     (`Diagnosis` -> `patient_Diagnosis`) so the final projection cannot
     accidentally ask for a non-existent unsuffixed column.
   - YYYYMM dates: treat integer `201306` and string "201306" as the
     same month via `.astype(str)`.
10. Metric conventions:
   - Schema Grounding is only a real-column candidate list, not a
     semantic authority. For requested metrics, inspect actual rows and
     simple numeric relationships before deciding whether a column is
     direct or must be transformed. Do not apply a knowledge.md formula
     merely because the formula exists; require a question phrase that
     asks for the formula's target metric. If data evidence contradicts a
     grounding hint, use the data-backed formula/field and record the
     override in `debug_steps["schema_mapping"]` or
     `debug_steps["plan_override"]`.
   - For "average monthly consumption" over customer consumption rows,
     compute the average consumption value for the filtered rows, then
     divide by 12 unless the question explicitly asks for total segment
     consumption.
   - For graph/connection tables with `atom_id`, `atom_id2`, `bond_id`,
     count distinct `bond_id` per atom; such tables often contain both
     directions of the same undirected bond.
11. For mixed, semantic-rule, or record_text tasks, define a `debug_steps` dict before
   `answer`. Include at least: `knowledge_rules_used`,
   `schema_inspection`, `schema_mapping`, `plan_override`,
   `input_counts`, `intermediate_counts`, `filters`, and a small
   `preview_rows` list. Use an empty dict/list for `plan_override` when
   no override was needed. This is for trace/debug only.

Source-boundary rules (HARD constraint, your code WILL be statically
checked against the listed Source Capabilities below):
- A `.db` / `.sqlite` file may only be queried with SQL via `sqlite3`,
  and only for tables that actually exist in that file. Do NOT JOIN to
  table names that aren't listed under that file's capability.
- A `.json` / `.jsonl` file must be loaded with `json` / `pandas.read_json`
  and used by Python lookup, NOT joined inside SQL.
- A `.md` / `.txt` / `.docx` file must be parsed with regex / string
  ops / RAG, NOT loaded into SQL.
- For mixed-context tasks (sqlite + json + doc), do NOT try to express
  the whole pipeline as one big SQL query. The default canonical shape
  is: doc-extract → sql query → json/pandas lookup → answer.
""".strip()


def _capabilities_block(compiled: Any | None) -> str:
    """Render a compact, machine-checkable list of source capabilities.

    The static checker reads the same field, so this is the single
    source of truth for "which file may be queried with which tool".
    Returns the empty string if no compiled task is available.
    """
    if compiled is None:
        return ""
    capabilities = getattr(compiled, "source_capabilities", None) or []
    if not capabilities:
        return ""
    lines: list[str] = ["Source Capabilities (use ONLY these paths/tables):"]
    for cap in capabilities:
        if cap.kind in {"db", "sqlite", "sqlite3"}:
            tables = []
            for table in cap.tables:
                cols = ", ".join(
                    str(c.get("name")) for c in (table.get("columns") or [])
                )
                tables.append(f"{table.get('table')}({cols})")
            lines.append(f"- {cap.path} [sqlite] tables: {' ; '.join(tables) or '(none)'}")
        elif cap.kind in {"csv", "tsv"}:
            cols = ", ".join(cap.columns)
            lines.append(f"- {cap.path} [{cap.kind}] columns: {cols} ({cap.row_count} rows) — use pandas.read_csv")
        elif cap.kind in {"json", "jsonl"}:
            fields = ", ".join(cap.json_record_fields) or ", ".join(cap.json_top_keys)
            lines.append(
                f"- {cap.path} [{cap.kind}] record_fields: {fields} "
                f"({cap.json_record_count} records) — use json/pandas, NOT SQL"
            )
        elif cap.kind in {"record_text", "structured_table"}:
            field_list = ", ".join(cap.structured_record_fields) or "(none auto-detected)"
            sample_block = ""
            if cap.structured_records:
                import json as _json
                sample_block = "\n  sample_records:\n    " + "\n    ".join(
                    _json.dumps(rec, ensure_ascii=False, default=str)[:240]
                    for rec in cap.structured_records[:3]
                )
            lines.append(
                f"- {cap.path} [record_text] unstructured long-form report, NOT a clean table.\n"
                f"  splitter_hint: {cap.structured_record_splitter or '(unknown)'}\n"
                f"  auto_detected_record_count: {cap.structured_record_count}\n"
                f"  auto_detected_fields: {field_list}{sample_block}"
            )
        else:
            lines.append(f"- {cap.path} [{cap.kind}] role={cap.role}")
    return "\n".join(lines)


_STRUCTURED_DOC_TEMPLATE = """
Record-text extraction template (this task includes a `record_text`
source — a multi-page narrative that mentions records inline, NOT a
clean per-paragraph list). Many paragraphs are
context / methodology / corrections, not records, so be defensive:

  ```python
  import re, pandas as pd

  text = open('<path>').read()
  rows = []

  # Iterate over every patient/record id mention; the surrounding ~600
  # chars typically contain the fields you need. Adjust the patterns
  # below to whichever fields the question actually asks for.
  ID_RE = re.compile(r'(?:patient|medical record number|file number|record number)\\D{0,40}?(\\d{3,8})', re.I)
  for match in ID_RE.finditer(text):
      pid = int(match.group(1))
      window = text[max(0, match.start()-50): match.end()+600]

      def grab_num(label):
          m = re.search(rf'\\b{label}\\b[^.\\n]{{0,80}}?\\b(\\d+(?:\\.\\d+)?)\\b', window, re.I)
          return float(m.group(1)) if m else None

      def grab_corrected(label):
          # "originally X; corrected to Y" → take Y
          m = re.search(
              rf'\\b{label}\\b[^.]*?(?:corrected|adjusted|amended|finalized|confirmed)\\s+to\\s+(\\d+(?:\\.\\d+)?)',
              window, re.I,
          )
          return float(m.group(1)) if m else grab_num(label)

      birthday = None
      m = re.search(
          r'born(?:[^.\\n]*?on)?[^.\\n]*?((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\w*\\s+\\d{1,2}\\w{0,2},?\\s*\\d{4})',
          window, re.I,
      )
      if m:
          birthday = m.group(1)

      rows.append({
          'patient_id': pid,
          'birthday':   birthday,
          'creatinine': grab_corrected('creatinine'),
          'got':        grab_corrected('GOT'),
          'gpt':        grab_corrected('GPT'),
          'ldh':        grab_corrected('LDH'),
          'alp':        grab_corrected('ALP'),
          't_bil':      grab_corrected('T-?BIL'),
      })

  df = pd.DataFrame(rows).drop_duplicates(subset=['patient_id'])
  # Now apply the question's filter (e.g. abnormal threshold from
  # knowledge.md, age-< 70, …) and assign the final result to `answer`.
  ```

Adapt the field list to whichever fields the question actually requires.
Always read the FULL file and iterate over every id mention — never
answer from the small sample shown above alone.
""".strip()


_MIXED_CONTEXT_TEMPLATE = """
Mixed-context default pipeline (this task type is `mixed_context`):
  1. doc/record_text → regex/string-parse to pull the few key
     identifiers you need (e.g. raceId for "2009 Singapore Grand Prix").
  2. sqlite → SELECT only against tables that actually exist in the
     listed Source Capabilities; use the identifiers from step 1 as
     literal WHERE values.
  3. json/csv → load with pandas/json and look up the final fields by
     the identifiers from step 2.
Do NOT JOIN sqlite tables that don't exist in the same .db file.
""".strip()


_SYNTHESIZED_CSV_TEMPLATE = """
Synthesized record_text CSVs are available in `.synthesized/`.
For any source that has a synthesized CSV, use `pandas.read_csv` on the
synthesized CSV and do NOT re-parse the original long `.md` file.
The synthesized CSVs already materialize the record-level fields needed
from the narrative documents; combine them with the remaining CSV/JSON/DB
sources and the semantic rules from `knowledge.md`.

Required debug contract for this path:
  - read or quote the relevant `knowledge.md` rule in `debug_steps["knowledge_rules_used"]`
  - record row counts after each join/filter in `debug_steps["intermediate_counts"]`
  - include 3-5 representative rows in `debug_steps["preview_rows"]`
""".strip()


def build_codegen_prompt(
    *,
    task: PublicTask,
    rendered_context: str,
    compiled: Any | None = None,
    semantic_plan: dict[str, Any] | None = None,
) -> str:
    capabilities_block = _capabilities_block(compiled)
    extras_parts: list[str] = []
    schema_grounding = ""
    foreign_keys = ""
    if compiled is not None:
        try:
            schema_grounding, _bindings = render_grounding_block(
                question=task.question,
                compiled=compiled,
            )
        except Exception:  # noqa: BLE001
            schema_grounding = ""
        try:
            foreign_keys = render_foreign_key_block(compiled)
        except Exception:  # noqa: BLE001
            foreign_keys = ""
        if getattr(compiled, "task_type", "") == "mixed_context":
            extras_parts.append(_MIXED_CONTEXT_TEMPLATE)
        capabilities = getattr(compiled, "source_capabilities", None) or []
        has_synthesized_csv = any(
            getattr(cap, "kind", "") in {"csv", "tsv"}
            and ".synthesized/" in str(getattr(cap, "path", ""))
            for cap in capabilities
        )
        if has_synthesized_csv:
            extras_parts.append(_SYNTHESIZED_CSV_TEMPLATE)
        elif any(getattr(cap, "kind", "") in {"record_text", "structured_table"} for cap in capabilities):
            extras_parts.append(_STRUCTURED_DOC_TEMPLATE)
    extras = ("\n\n" + "\n\n".join(extras_parts)) if extras_parts else ""
    capabilities_section = (
        f"\n{capabilities_block}\n" if capabilities_block else ""
    )
    grounding_section = (
        "\nSchema Grounding Evidence (hypotheses, verify against actual columns in code):\n"
        f"{schema_grounding}\n" if schema_grounding else ""
    )
    foreign_key_section = (
        "\nJoin-Key Evidence (hypotheses, verify with real columns/dtypes before merge):\n"
        f"{foreign_keys}\n" if foreign_keys else ""
    )
    semantic_section = ""
    if semantic_plan:
        compact_plan = {
            "schema_mapping": semantic_plan.get("schema_mapping"),
            "join_plan": semantic_plan.get("join_plan"),
            "filters": semantic_plan.get("filters"),
            "aggregation": semantic_plan.get("aggregation"),
            "output": semantic_plan.get("output"),
            "requires_rule_resolution": semantic_plan.get("requires_rule_resolution"),
            "unresolved_core_filters": semantic_plan.get("unresolved_core_filters"),
            "rule_resolution_queries": semantic_plan.get("rule_resolution_queries"),
            "rule_resolution": semantic_plan.get("rule_resolution"),
            "applied_plan_overrides": semantic_plan.get("applied_plan_overrides"),
            "consistency_checks": semantic_plan.get("consistency_checks"),
            "uncertainties": semantic_plan.get("uncertainties"),
            "confidence": semantic_plan.get("confidence"),
        }
        semantic_section = (
            "\nSemantic Analyst Plan (审题 baseline, not ground truth when "
            "low-confidence or rule resolution is pending. Align with it when "
            "supported; when knowledge.md/runtime evidence resolves or corrects "
            "it, record that in debug_steps['plan_override']):\n"
            f"{json.dumps(compact_plan, ensure_ascii=False, indent=2, default=str)}\n"
        )
    return (
        f"Question: {task.question}\n"
        f"Difficulty: {task.difficulty}\n"
        f"{capabilities_section}\n"
        f"{grounding_section}"
        f"{foreign_key_section}"
        f"{semantic_section}\n"
        f"Available context (paths are relative to the task's `context/` dir):\n\n"
        f"{rendered_context}\n"
        f"{_CODEGEN_INSTRUCTION}{extras}"
    )


_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_PYTHON_OPEN_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n", re.IGNORECASE)
_FENCE_MARKER_LINE_RE = re.compile(r"^\s*```(?:python|py)?\s*$", re.IGNORECASE)


def strip_markdown_code_fences(text: str) -> str:
    """Remove markdown fence marker lines from a Python program.

    Handles both well-formed fences and the common truncated-model case
    where the response starts with ```python but never emits the closing
    fence. Only exact fence-marker lines are removed, so ordinary code is
    left alone.
    """
    lines = text.strip().splitlines()
    while lines and _FENCE_MARKER_LINE_RE.match(lines[0]):
        lines.pop(0)
    while lines and _FENCE_MARKER_LINE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def extract_python_program(raw_response: str) -> str:
    text = raw_response.strip()
    match = _PYTHON_FENCE_RE.search(text)
    if match:
        return strip_markdown_code_fences(match.group(1))
    open_match = _PYTHON_OPEN_FENCE_RE.search(text)
    if open_match is not None:
        text = text[open_match.end():]
    # Some endpoints respond without fences; treat as raw program.
    return strip_markdown_code_fences(text)


def validate_python_syntax(program: str) -> str | None:
    """Return a compact syntax error string, or None when parseable."""
    try:
        ast.parse(program)
    except SyntaxError as exc:
        line = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
        return f"{exc.msg} ({line})"
    return None


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


_EXEC_HARNESS = """
__user_code__
import os, json, csv
import pandas as _pd

if 'answer' not in dir():
    raise RuntimeError("Operator codegen program did not define an `answer` variable.")

_a = answer
if isinstance(_a, _pd.DataFrame):
    _df = _a
elif isinstance(_a, _pd.Series):
    _df = _a.to_frame()
elif isinstance(_a, (list, tuple)):
    if _a and isinstance(_a[0], dict):
        _df = _pd.DataFrame(list(_a))
    else:
        _df = _pd.DataFrame({'value': list(_a)})
elif isinstance(_a, dict):
    _df = _pd.DataFrame(_a)
else:
    _df = _pd.DataFrame({'value': [_a]})

_df.to_csv(__RESULT_PATH__, index=False)
print('OPERATOR_CODEGEN_RESULT_OK')
print('OPERATOR_CODEGEN_SHAPE=' + str(_df.shape))
if 'debug_steps' in dir():
    try:
        print('OPERATOR_CODEGEN_DEBUG=' + json.dumps(debug_steps, ensure_ascii=False, default=str))
    except Exception as _debug_exc:
        print('OPERATOR_CODEGEN_DEBUG_ERROR=' + str(_debug_exc))
"""


def _wrap_for_execution(user_code: str, *, result_path: Path) -> str:
    return _EXEC_HARNESS.replace("__user_code__", user_code).replace(
        "__RESULT_PATH__", repr(str(result_path))
    )


def _read_result_csv(result_path: Path) -> AnswerTable | None:
    if not result_path.exists():
        return None
    try:
        with result_path.open(newline="") as handle:
            rows = list(csv.reader(handle))
    finally:
        try:
            result_path.unlink()
        except OSError:
            pass
    if not rows:
        return AnswerTable(columns=[], rows=[])
    header, body = rows[0], rows[1:]
    return AnswerTable(columns=list(header), rows=[list(row) for row in body])


def _extract_debug_steps(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        if line.startswith("OPERATOR_CODEGEN_DEBUG="):
            payload = line.removeprefix("OPERATOR_CODEGEN_DEBUG=").strip()
        elif line.startswith("TABLELLM_DEBUG="):
            # Backward compatibility for older traces.
            payload = line.removeprefix("TABLELLM_DEBUG=").strip()
        else:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {"raw": payload}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return None


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CodegenRunResult:
    answer: AnswerTable | None
    succeeded: bool
    failure_reason: str | None = None
    raw_response: str = ""
    program: str = ""
    manifest: list[dict[str, Any]] = field(default_factory=list)
    exec_stdout: str = ""
    exec_stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_kind": "operator_codegen",
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "raw_response": self.raw_response,
            "program": self.program,
            "context_manifest": list(self.manifest),
            "exec_stdout": self.exec_stdout,
            "exec_stderr": self.exec_stderr,
        }


@dataclass(slots=True)
class CodegenDirectAgent:
    """One-shot code-solution agent backed by the configured LLM endpoint."""

    model: OpenAIModelAdapter
    max_table_rows: int = 50
    max_input_chars: int = 12000
    python_timeout: int = 30
    sample_temperature: float | None = None
    sample_seed: int | None = None
    # When set, document files are routed through RAG instead of the
    # default keyword-section renderer. See
    # ``context_render.render_with_rag`` for the kwargs shape.
    rag_kwargs: dict[str, Any] | None = None
    # CompiledTask passed in by OperatorExecutor so the codegen prompt
    # can include source-capability hard constraints + the mixed_context
    # canonical pipeline. Optional for backwards compat — without it the
    # prompt simply omits the new sections.
    compiled: Any | None = None
    semantic_plan: dict[str, Any] | None = None
    progress_label: str = "operator-codegen"

    def run(self, task: PublicTask) -> CodegenRunResult:
        rendered_context, manifest = render_context_for_codegen(
            task,
            max_table_rows=self.max_table_rows,
            max_input_chars=self.max_input_chars,
            question=task.question,
            rag_kwargs=self.rag_kwargs,
        )
        user_prompt = build_codegen_prompt(
            task=task,
            rendered_context=rendered_context,
            compiled=self.compiled,
            semantic_plan=self.semantic_plan,
        )
        try:
            raw_response = self.model.complete(
                [
                    ModelMessage(role="system", content=_CODEGEN_SYSTEM),
                    ModelMessage(role="user", content=user_prompt),
                ],
                temperature=self.sample_temperature,
                seed=self.sample_seed,
                stream_label=f"{self.progress_label} code-solution",
                max_tokens=max(int(getattr(self.model, "max_tokens", 0) or 0), 2048),
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            return CodegenRunResult(
                answer=None,
                succeeded=False,
                failure_reason=f"operator_codegen_request_error: {exc}",
                manifest=manifest,
            )

        program = extract_python_program(raw_response)
        if not program:
            return CodegenRunResult(
                answer=None,
                succeeded=False,
                failure_reason="operator_codegen_empty_program",
                raw_response=raw_response,
                manifest=manifest,
            )
        syntax_error = validate_python_syntax(program)
        if syntax_error is not None:
            return CodegenRunResult(
                answer=None,
                succeeded=False,
                failure_reason=f"operator_codegen_syntax_error: {syntax_error}",
                raw_response=raw_response,
                program=program,
                manifest=manifest,
            )

        from data_agent_baseline.progress import get_progress_logger
        logger = get_progress_logger()
        if logger is not None:
            manifest_summary = ", ".join(
                f"{m.get('kind')}:{m.get('path', '?')}".split('/')[-1]
                for m in manifest
            )
            logger.codegen_program(
                program=program,
                manifest_summary=manifest_summary or "(no files)",
                label=self.progress_label,
            )

        if self.compiled is not None:
            try:
                from data_agent_baseline.agents.static_checker import (
                    check_program,
                    has_blocking,
                )

                capabilities = getattr(self.compiled, "source_capabilities", None) or []
                static_issues = check_program(program, capabilities=capabilities)
            except Exception:  # noqa: BLE001
                static_issues = []
            if static_issues and has_blocking(static_issues):
                if logger is not None:
                    logger.codegen_debug(
                        debug={
                            "static_check": "failed_before_execution",
                            "issues": [item.to_dict() for item in static_issues],
                        }
                    )
                    logger.codegen_executed(
                        succeeded=False,
                        shape=None,
                        failure_reason="static_check_failed",
                    )
                return CodegenRunResult(
                    answer=None,
                    succeeded=False,
                    failure_reason="operator_codegen_static_error",
                    raw_response=raw_response,
                    program=program,
                    manifest=list(manifest or []) + [{
                        "static_issues": [item.to_dict() for item in static_issues],
                    }],
                )

        with tempfile.TemporaryDirectory() as result_dir:
            result_path = Path(result_dir) / "operator_codegen_answer.csv"
            wrapped = _wrap_for_execution(program, result_path=result_path)
            from data_agent_baseline.budget import get_budget_controller

            budget = get_budget_controller()
            if budget is not None:
                budget.consume_tool("execute_python:operator")
            exec_result = execute_python_code(
                context_root=task.context_dir,
                code=wrapped,
                timeout_seconds=self.python_timeout,
            )
            stdout = str(exec_result.get("output", ""))
            stderr = str(exec_result.get("stderr", ""))
            if not exec_result.get("success"):
                error_msg = exec_result.get("error", "unknown")
                if logger is not None:
                    logger.codegen_executed(succeeded=False, shape=None, failure_reason=str(error_msg))
                return CodegenRunResult(
                    answer=None,
                    succeeded=False,
                    failure_reason=f"operator_codegen_exec_error: {error_msg}",
                    raw_response=raw_response,
                    program=program,
                    manifest=manifest,
                    exec_stdout=stdout,
                    exec_stderr=stderr,
                )
            answer = _read_result_csv(result_path)
            if logger is not None and answer is not None:
                logger.codegen_executed(
                    succeeded=True,
                    shape=(len(answer.rows), len(answer.columns)),
                    failure_reason=None,
                )
                debug_steps = _extract_debug_steps(stdout)
                if debug_steps is not None:
                    logger.codegen_debug(debug=debug_steps)
        if answer is None or not answer.columns:
            return CodegenRunResult(
                answer=None,
                succeeded=False,
                failure_reason="operator_codegen_no_result_table",
                raw_response=raw_response,
                program=program,
                manifest=manifest,
                exec_stdout=stdout,
                exec_stderr=stderr,
            )

        return CodegenRunResult(
            answer=answer,
            succeeded=True,
            raw_response=raw_response,
            program=program,
            manifest=manifest,
            exec_stdout=stdout,
            exec_stderr=stderr,
        )
