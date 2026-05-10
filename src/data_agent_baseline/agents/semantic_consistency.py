"""Dual-path semantic consistency for table-centric operator tasks.

This module adds the "审题官 + 执行官" split:

1. Semantic Analyst: reads the question + real schema/samples and emits a
   structured semantic plan. It does not write code.
2. Code Executor: existing operator-codegen writes and runs Python.
3. Consistency Judge: compares the analyst plan with the executed program,
   debug_steps, and answer preview.
4. Semantic Repair: if the judge finds a semantic mismatch, rewrite only
   the necessary Python.

The module deliberately avoids record_text / large-document tasks. Those
need extraction/RAG first; asking an LLM to "guess" from long prose would
re-introduce hallucination.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.agents.tablellm_direct import (
    CodegenRunResult,
    _extract_debug_steps,
    extract_python_program,
)
from data_agent_baseline.agents.task_compiler import CompiledTask
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.budget import BudgetExceeded
from data_agent_baseline.eval.answer_validator import validate_answer_table


@dataclass(slots=True)
class SemanticConsistencyResult:
    """One consistency round result for trace/debug."""

    semantic_plan: dict[str, Any] | None = None
    rule_resolution: dict[str, Any] | None = None
    effective_plan: dict[str, Any] | None = None
    judge_history: list[dict[str, Any]] = field(default_factory=list)
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    final_verdict: str = "skipped"
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_plan": self.semantic_plan,
            "rule_resolution": self.rule_resolution,
            "effective_plan": self.effective_plan,
            "judge_history": list(self.judge_history),
            "repair_history": list(self.repair_history),
            "final_verdict": self.final_verdict,
            "failure_reason": self.failure_reason,
        }


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.I | re.S)
_HARD_RUNTIME_RE = re.compile(
    r"\b(KeyError|IndexError|FileNotFoundError|JSONDecodeError)\b|"
    r"json\.decoder\.JSONDecodeError|single positional indexer is out-of-bounds",
    re.IGNORECASE,
)


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    match = _JSON_FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as first_exc:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "model did not return parseable JSON; raw preview="
                    + repr(raw[:500])
                ) from exc
        else:
            raise ValueError(
                "model did not return parseable JSON; raw preview="
                + repr(raw[:500])
            ) from first_exc
    if not isinstance(parsed, dict):
        raise ValueError("model did not return a JSON object")
    return parsed


def _source_summary(compiled_task: CompiledTask, *, max_sources: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cap in compiled_task.source_capabilities[:max_sources]:
        tables = []
        for table in cap.tables[:8]:
            tables.append({
                "table": table.get("table"),
                "columns": [
                    col.get("name") if isinstance(col, dict) else col
                    for col in (table.get("columns") or [])
                ],
                "row_count": table.get("row_count"),
            })
        out.append({
            "path": cap.path,
            "kind": cap.kind,
            "role": cap.role,
            "tool": cap.tool,
            "columns": list(cap.columns)[:80],
            "json_top_keys": list(cap.json_top_keys)[:20],
            "json_record_fields": list(cap.json_record_fields)[:80],
            "column_value_samples": {
                str(k): list(v)[:5]
                for k, v in list((cap.column_value_samples or {}).items())[:80]
            },
            "column_dtypes": dict(list((cap.column_dtypes or {}).items())[:80]),
            "column_cardinalities": dict(list((cap.column_cardinalities or {}).items())[:80]),
            "row_count": cap.row_count or cap.json_record_count or cap.structured_record_count,
            "tables": tables,
            "sample": list(cap.sample)[:3],
        })
    return out


def _semantic_rule_documents(
    task: PublicTask,
    compiled_task: CompiledTask,
    *,
    max_chars_per_doc: int = 7000,
) -> list[dict[str, Any]]:
    """Load semantic-rule docs such as knowledge.md for the analyst.

    SourceCapability only carries schema-ish metadata for docs, so without
    this the analyst sees that a rule document exists but cannot resolve
    phrases like "severe". That is exactly how a guess becomes a false
    baseline.
    """
    docs: list[dict[str, Any]] = []
    for cap in compiled_task.source_capabilities:
        if cap.role != "semantic_rule" and cap.path.lower() != "knowledge.md":
            continue
        path = task.context_dir / cap.path
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            docs.append({"path": cap.path, "error": str(exc)})
            continue
        docs.append({
            "path": cap.path,
            "chars": len(text),
            "text": text[:max_chars_per_doc],
            "truncated": len(text) > max_chars_per_doc,
        })
    return docs


def _answer_preview(answer: AnswerTable | None, *, max_rows: int = 8) -> dict[str, Any] | None:
    if answer is None:
        return None
    return {
        "columns": list(answer.columns),
        "rows": [list(row) for row in answer.rows[:max_rows]],
        "row_count": len(answer.rows),
        "column_count": len(answer.columns),
    }


def _has_hard_runtime_error(*parts: str | None) -> bool:
    return any(_HARD_RUNTIME_RE.search(str(part or "")) for part in parts)


def _has_schema_inspection(debug_steps: dict[str, Any] | None) -> bool:
    if not isinstance(debug_steps, dict):
        return False
    inspection = debug_steps.get("schema_inspection")
    if isinstance(inspection, dict) and inspection:
        return True
    # Backward-compatible aliases from earlier traces. These do not
    # satisfy the new prompt perfectly, but they prove the code inspected
    # real columns rather than only following knowledge.md wording.
    for key in ("actual_columns", "columns_seen", "inspected_columns"):
        value = debug_steps.get(key)
        if isinstance(value, dict) and value:
            return True
    return False


def _has_zero_row_probe(debug_steps: dict[str, Any] | None) -> bool:
    if not isinstance(debug_steps, dict):
        return False
    probe = debug_steps.get("zero_row_probe")
    return isinstance(probe, dict) and bool(probe)


def _preflight_judge_failure(
    *,
    compiled_task: CompiledTask,
    answer: AnswerTable | None,
    debug_steps: dict[str, Any] | None,
    exec_stdout: str,
    exec_stderr: str,
    failure_reason: str | None,
    execution_succeeded: bool = True,
    static_issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not execution_succeeded or answer is None:
        failure_types = ["execution_failed"]
        mismatches = [
            "Program did not produce an executable AnswerTable; repair must run before semantic pass."
        ]
        if static_issues:
            failure_types = ["static_check_failed"]
            issue_bits = [
                f"{item.get('code', 'issue')}: {item.get('message', '')}"
                for item in static_issues[:5]
            ]
            mismatches = [
                "Static/schema check failed before a valid answer was produced.",
                *issue_bits,
            ]
        if _has_hard_runtime_error(exec_stdout, exec_stderr, failure_reason):
            failure_types.append("runtime_exception")
        return {
            "verdict": "fail",
            "confidence": 1.0,
            "failure_types": list(dict.fromkeys(failure_types)),
            "mismatches": mismatches,
            "repair_hint": (
                "Repair the cited structural/runtime error using the previous "
                "program, static_issues, schema_diagnostics, and semantic_plan. "
                "Do not mark failed execution as semantically consistent."
            ),
            "must_fix": True,
        }

    validation = validate_answer_table(answer)
    if not validation.valid:
        codes = [item.code for item in validation.errors]
        if "no_rows" in codes:
            return {
                "verdict": "fail",
                "confidence": 1.0,
                "failure_types": ["zero_rows"],
                "mismatches": [
                    "Program executed but produced zero answer rows.",
                    *[item.message for item in validation.errors],
                ],
                "repair_hint": (
                    "Run a zero-row probe before changing final answer logic: "
                    "record debug_steps['zero_row_probe'] with base row counts, "
                    "row counts after each join/filter, candidate value counts "
                    "for every filter key, sample unique values, and the first "
                    "operation that drops rows to zero. Patch only the operation "
                    "identified by that probe."
                ),
                "must_fix": True,
            }
        return {
            "verdict": "fail",
            "confidence": 1.0,
            "failure_types": ["invalid_answer", *codes],
            "mismatches": [item.message for item in validation.errors],
            "repair_hint": (
                "Repair the final AnswerTable shape/projection before semantic pass; "
                "do not pass empty, ragged, or fully-null answer columns."
            ),
            "must_fix": True,
        }

    if _has_hard_runtime_error(exec_stdout, exec_stderr, failure_reason):
        return {
            "verdict": "fail",
            "confidence": 1.0,
            "failure_types": ["runtime_exception"],
            "mismatches": [
                "Hard runtime exception detected; repair must run before semantic judging."
            ],
            "repair_hint": (
                "Repair the program with schema-first inspection. Do not pass a "
                "KeyError/IndexError/FileNotFoundError/JSONDecodeError result."
            ),
            "must_fix": True,
        }
    if compiled_task.task_type in {
        "table_computation",
        "mixed_context",
        "table_with_semantic_rule",
    } and not _has_schema_inspection(debug_steps):
        return {
            "verdict": "fail",
            "confidence": 0.95,
            "failure_types": ["schema_inspection_missing"],
            "mismatches": [
                "debug_steps does not contain schema_inspection from real loaded columns."
            ],
            "repair_hint": (
                "Add debug_steps['schema_inspection'] after every dataframe/SQL/JSON "
                "load, and make schema_mapping choose only fields present there."
            ),
            "must_fix": True,
        }
    return None


def _normal_overrides(debug_steps: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(debug_steps, dict):
        return []
    raw = (
        debug_steps.get("plan_override")
        or debug_steps.get("plan_overrides")
        or debug_steps.get("semantic_plan_override")
    )
    if raw is None:
        return []
    if isinstance(raw, dict):
        # Accept either one override object or a mapping of names -> override.
        if any(key in raw for key in ("field", "concept", "old_value", "new_value", "evidence")):
            return [raw]
        return [
            {"name": str(name), **value}
            for name, value in raw.items()
            if isinstance(value, dict)
        ]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _apply_plan_overrides(
    plan: dict[str, Any],
    debug_steps: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the plan the judge should compare against.

    Code is allowed to resolve analyst uncertainty with stronger evidence
    from knowledge.md or runtime schema inspection, but it must record the
    change explicitly in debug_steps["plan_override"]. The judge then sees
    an effective plan instead of treating the initial analyst guess as law.
    """
    overrides = _normal_overrides(debug_steps)
    if not overrides:
        return plan, []

    effective = deepcopy(plan)
    effective["applied_plan_overrides"] = overrides
    filters = effective.get("filters")
    if not isinstance(filters, list):
        filters = []
        effective["filters"] = filters

    for override in overrides:
        field = str(
            override.get("field")
            or override.get("chosen_field")
            or override.get("filter_field")
            or ""
        ).strip()
        concept = str(override.get("concept") or "").strip().lower()
        has_new_value = "new_value" in override or "value" in override
        new_value = override.get("new_value", override.get("value"))
        matched = False
        for item in filters:
            if not isinstance(item, dict):
                continue
            item_field = str(item.get("field") or item.get("chosen_field") or "").strip()
            item_concept = str(item.get("concept") or item.get("name") or "").strip().lower()
            if (field and item_field == field) or (concept and concept == item_concept):
                if has_new_value:
                    item["value"] = new_value
                item["overridden_by_code"] = True
                item["override_evidence"] = override.get("evidence")
                matched = True
        if field and has_new_value and not matched:
            filters.append({
                "field": field,
                "operator": override.get("operator", "=="),
                "value": new_value,
                "concept": override.get("concept"),
                "overridden_by_code": True,
                "override_evidence": override.get("evidence"),
            })
    return effective, overrides


_RULE_VALUE_RE = re.compile(
    r"['\"]?(?P<value>-?\d+(?:\.\d+)?)['\"]?\s+"
    r"(?:indicates?|indicating|means|denotes|represents?|corresponds?\s+to)\s+"
    r"(?P<label>[^.;\n]{0,120})",
    re.IGNORECASE,
)
_RULE_LABEL_VALUE_RE = re.compile(
    r"(?P<label>[^.;\n]{0,80}?\b(?:severe|abnormal|normal|active|eligible|open|closed)\b[^.;\n]{0,80}?)"
    r"(?:=|:|is|value)\s*['\"]?(?P<value>-?\d+(?:\.\d+)?)['\"]?",
    re.IGNORECASE,
)


def _resolve_semantic_rules(
    *,
    task: PublicTask,
    compiled_task: CompiledTask,
    plan: dict[str, Any],
) -> dict[str, Any]:
    docs = _semantic_rule_documents(task, compiled_task)
    filters = plan.get("filters") if isinstance(plan.get("filters"), list) else []
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    plan_confidence = _as_float(plan.get("confidence"), default=1.0)
    requires_global = bool(plan.get("requires_rule_resolution")) or plan_confidence <= 0.65

    for item in filters:
        if not isinstance(item, dict):
            continue
        concept = str(item.get("concept") or item.get("name") or "").strip()
        field = str(item.get("field") or item.get("chosen_field") or "").strip()
        needs_resolution = (
            bool(item.get("requires_rule_resolution"))
            or item.get("value") is None
            or requires_global
        )
        if not needs_resolution:
            continue
        match = _find_rule_value(
            concept=concept,
            field=field,
            docs=docs,
        )
        if match is None:
            unresolved.append({
                "concept": concept,
                "field": field,
                "reason": "no explicit rule value found in semantic rule documents",
                "alternatives": item.get("alternatives"),
            })
            continue
        resolved.append({
            "concept": concept,
            "field": field,
            "operator": item.get("operator", "=="),
            "value": match["value"],
            "old_value": item.get("value"),
            "evidence": match["evidence"],
            "source": match["path"],
        })

    return {
        "resolved_filters": resolved,
        "unresolved_filters": unresolved,
        "documents": [
            {"path": doc.get("path"), "chars": doc.get("chars"), "truncated": doc.get("truncated")}
            for doc in docs
        ],
    }


def _find_rule_value(
    *,
    concept: str,
    field: str,
    docs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    concept_lower = concept.lower()
    field_lower = field.lower()
    wants_severe = "severe" in concept_lower
    for doc in docs:
        text = str(doc.get("text") or "")
        path = str(doc.get("path") or "")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            lower = line.lower()
            if not line:
                continue
            if field_lower and field_lower not in lower and not any(tok in lower for tok in concept_lower.split() if len(tok) > 3):
                continue
            for match in _RULE_VALUE_RE.finditer(line):
                label = match.group("label").lower()
                if wants_severe and "severe" not in label:
                    continue
                if wants_severe and "most severe" in label:
                    continue
                value = _coerce_rule_value(match.group("value"))
                return {"value": value, "evidence": line, "path": path}
            for match in _RULE_LABEL_VALUE_RE.finditer(line):
                label = match.group("label").lower()
                if wants_severe and "severe" not in label:
                    continue
                if wants_severe and "most severe" in label:
                    continue
                value = _coerce_rule_value(match.group("value"))
                return {"value": value, "evidence": line, "path": path}
    return None


def _coerce_rule_value(raw: str) -> Any:
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number.is_integer():
        return int(number)
    return number


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_rule_resolution(
    plan: dict[str, Any],
    rule_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    effective = deepcopy(plan)
    if not rule_resolution:
        return effective
    resolved = [
        item for item in (rule_resolution.get("resolved_filters") or [])
        if isinstance(item, dict)
    ]
    if not resolved:
        effective["rule_resolution"] = rule_resolution
        return effective
    filters = effective.get("filters")
    if not isinstance(filters, list):
        filters = []
        effective["filters"] = filters
    for resolved_filter in resolved:
        field = str(resolved_filter.get("field") or "").strip()
        concept = str(resolved_filter.get("concept") or "").strip().lower()
        matched = False
        for item in filters:
            if not isinstance(item, dict):
                continue
            item_field = str(item.get("field") or item.get("chosen_field") or "").strip()
            item_concept = str(item.get("concept") or item.get("name") or "").strip().lower()
            if (field and field == item_field) or (concept and concept == item_concept):
                item["value"] = resolved_filter.get("value")
                item["operator"] = resolved_filter.get("operator", item.get("operator", "=="))
                item["requires_rule_resolution"] = False
                item["rule_resolution_evidence"] = resolved_filter.get("evidence")
                matched = True
        if not matched:
            filters.append({
                "concept": resolved_filter.get("concept"),
                "field": field,
                "operator": resolved_filter.get("operator", "=="),
                "value": resolved_filter.get("value"),
                "requires_rule_resolution": False,
                "rule_resolution_evidence": resolved_filter.get("evidence"),
            })
    effective["requires_rule_resolution"] = bool(rule_resolution.get("unresolved_filters"))
    effective["rule_resolution"] = rule_resolution
    return effective


def should_run_semantic_consistency(compiled_task: CompiledTask) -> bool:
    """Gate dual-path consistency to table-centric tasks.

    Skip long prose / image / pure document tasks because their robust
    path is extraction/RAG first, not parallel LLM guessing.
    """
    flags = set(compiled_task.ambiguity_flags)
    if compiled_task.task_type not in {
        "table_computation",
        "mixed_context",
        "table_with_semantic_rule",
    }:
        return False
    if flags & {
        "record_text_context",
        "record_extraction_required",
        "large_document_context",
        "needs_vision",
        "unsupported_file_type",
    }:
        return False
    if compiled_task.file_count > 8:
        return False
    return True


_ANALYST_SYSTEM = """You are the Semantic Analyst for a table/data QA agent.
Do not write code and do not solve by guessing.
Map the natural-language question to real schema fields, joins, filters,
aggregation grain, and output semantics using only the provided schema,
samples, and knowledge notes.

Important uncertainty policy:
- If a question concept can plausibly map to multiple real fields, list
  all plausible fields in `alternative_mappings`.
- Treat knowledge.md as rules/semantics/hypotheses. It cannot prove that
  a dataframe column exists; the selected schema must be one of the real
  fields in the provided sources/schema diagnostics.
- knowledge.md formulas are not default transformations. Use a formula only
  when the question explicitly asks for that formula's target metric, or when
  the requested output cannot be computed from a direct real column. If a
  direct real field answers the question, prefer the direct field.
- For every selected mapping, include evidence from real column names,
  sample values, or schema diagnostics. If evidence is only from
  knowledge.md wording, mark it uncertain.
- Do not assign high confidence to an unresolved mapping ambiguity.
  If unresolved alternatives remain, confidence must be <= 0.65.
- A selected field is only high-confidence when schema names, sample
  values, and knowledge semantics all support it.
- For semantic-rule filters (for example severe/abnormal/active/eligible),
  do not guess a concrete value when the rule document does not prove it.
  In that case set `requires_rule_resolution=true`, put the concept in
  `unresolved_core_filters`, and leave the filter value null or expressed
  as alternatives. Never turn a low-confidence rule guess into a fixed
  filter baseline.
- If the context is small enough to infer a tentative answer from the
  shown samples/schema, include `tentative_answer`; otherwise use null.

Return exactly one JSON object. No prose.
"""


def run_semantic_analyst(
    *,
    task: PublicTask,
    model: OpenAIModelAdapter,
    compiled_task: CompiledTask,
    schema_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packet = {
        "task_id": task.task_id,
        "question": task.question,
        "task_profile": {
            "task_type": compiled_task.task_type,
            "answer_type": compiled_task.answer_type,
            "operations": list(compiled_task.operations),
            "modalities": list(compiled_task.modalities),
            "ambiguity_flags": list(compiled_task.ambiguity_flags),
        },
        "sources": _source_summary(compiled_task),
        "semantic_rule_documents": _semantic_rule_documents(task, compiled_task),
        "schema_diagnostics": schema_diagnostics or {},
        "output_contract": {
            "schema_mapping": (
                "map question concepts to real fields; each item should include "
                "concept, chosen_source, chosen_field, candidate_fields, and "
                "real_schema_evidence"
            ),
            "alternative_mappings": "plausible competing fields for each ambiguous concept",
            "join_plan": "list source.field -> source.field joins",
            "filters": "semantic filters and exact fields/values",
            "requires_rule_resolution": "true when a core semantic filter needs knowledge.md/runtime evidence before fixing a value",
            "unresolved_core_filters": "semantic filters whose concrete values are not proven yet",
            "rule_resolution_queries": "specific rules codegen must resolve from semantic_rule_documents",
            "aggregation": "metric, group_by, grain",
            "output": "requested output semantics and likely columns",
            "tentative_answer": "optional small-context natural-language answer or null",
            "uncertainties": "ambiguous points that code must inspect",
            "confidence": "0.0-1.0",
        },
    }
    raw = model.complete(
        [
            ModelMessage(role="system", content=_ANALYST_SYSTEM),
            ModelMessage(
                role="user",
                content=(
                    "Build the semantic execution plan. Return JSON with keys: "
                    "schema_mapping, alternative_mappings, join_plan, filters, "
                    "requires_rule_resolution, unresolved_core_filters, "
                    "rule_resolution_queries, aggregation, output, "
                    "tentative_answer, consistency_checks, uncertainties, "
                    "confidence.\n\n"
                    f"{json.dumps(packet, ensure_ascii=False, indent=2, default=str)}"
                ),
            ),
        ],
        temperature=0.0,
        stream_label="semantic analyst",
        max_tokens=1200,
    )
    plan = _parse_json_object(raw)
    return plan


def semantic_plan_prompt_block(plan: dict[str, Any] | None) -> str:
    if not plan:
        return ""
    compact = {
        "schema_mapping": plan.get("schema_mapping"),
        "join_plan": plan.get("join_plan"),
        "filters": plan.get("filters"),
        "aggregation": plan.get("aggregation"),
        "output": plan.get("output"),
        "alternative_mappings": plan.get("alternative_mappings"),
        "requires_rule_resolution": plan.get("requires_rule_resolution"),
        "unresolved_core_filters": plan.get("unresolved_core_filters"),
        "rule_resolution_queries": plan.get("rule_resolution_queries"),
        "tentative_answer": plan.get("tentative_answer"),
        "consistency_checks": plan.get("consistency_checks"),
        "uncertainties": plan.get("uncertainties"),
        "confidence": plan.get("confidence"),
    }
    return (
        "\n\nSemantic Analyst Plan (审题 baseline, not ground truth when "
        "low-confidence or rule resolution is pending. If code resolves a "
        "rule from knowledge.md or runtime schema evidence, record it in "
        "debug_steps['plan_override']):\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2, default=str)}\n"
    )


_JUDGE_SYSTEM = """You are the Consistency Judge for a table/data QA agent.
Compare the Semantic Analyst plan with the executed Python program,
debug_steps, and answer preview. You are not grading against hidden gold.
The Semantic Analyst plan is a baseline hypothesis, not an authority when
it explicitly has low confidence or requires rule resolution.
You are checking semantic consistency:
- schema field mapping
- unresolved mapping ambiguity
- join path
- filters
- aggregation grain
- output projection
- whether a non-empty answer plausibly answers the question
- whether debug_steps contains schema_inspection from real loaded columns
- whether schema_mapping chose fields from real schema rather than from
  knowledge.md wording alone
- whether metric formulas and field choices are supported by row-level
  data evidence rather than only by column-name grounding hints.
- prior static/repair issues, especially no_such_column and closest_matches,
  as evidence of what the repair was supposed to fix.

You may challenge the Semantic Analyst plan if the plan picked one field
while leaving plausible alternatives unresolved. If selected mapping is
ambiguous and the code does not show evidence resolving it, return
`low_confidence` or `fail`, not `pass`.
If code_debug_steps includes `plan_override` with concrete evidence from
knowledge.md or runtime schema inspection, judge against the
effective_semantic_plan after that override. Do not fail merely because
the code disagrees with a low-confidence analyst guess that the override
resolved.
If row-level evidence contradicts a Schema Grounding hint, prefer the
data-backed interpretation when the code records the evidence in
schema_mapping, filters, or plan_override. Do not treat grounding hints
as semantic truth.
If a prior no_such_column issue had a closest match like `link_to_event`,
do not treat that as proof that `event_name` should be replaced by the
foreign key; it usually means the code must join the referenced source to
materialize the requested semantic attribute.
If debug_steps lacks schema_inspection for a table/mixed task, return
`fail` with failure_type `schema_inspection_missing`.
If execution output contains KeyError, IndexError, FileNotFoundError, or
JSONDecodeError, return `fail`; such runs must be repaired before pass.
If the answer has zero rows, zero columns, ragged rows, or a fully empty
column, return `fail`; invalid answer tables are never semantically pass.
For zero rows specifically, require a `zero_row_probe` in debug_steps that
shows row counts after each join/filter and identifies the first operation
that drops rows to zero.

Return exactly one JSON object. No prose.
"""


def judge_consistency(
    *,
    task: PublicTask,
    model: OpenAIModelAdapter,
    compiled_task: CompiledTask,
    semantic_plan: dict[str, Any],
    program: str,
    answer: AnswerTable | None,
    debug_steps: dict[str, Any] | None,
    exec_stdout: str = "",
    schema_diagnostics: dict[str, Any] | None = None,
    static_issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    packet = {
        "question": task.question,
        "task_profile": {
            "task_type": compiled_task.task_type,
            "answer_type": compiled_task.answer_type,
            "operations": list(compiled_task.operations),
        },
        "effective_semantic_plan": semantic_plan,
        "code_debug_steps": debug_steps or {},
        "answer_preview": _answer_preview(answer),
        "program_excerpt": program[:5000],
        "exec_stdout_tail": exec_stdout[-1200:],
        "schema_diagnostics": schema_diagnostics or {},
        "prior_static_or_repair_issues": static_issues or [],
    }
    raw = model.complete(
        [
            ModelMessage(role="system", content=_JUDGE_SYSTEM),
            ModelMessage(
                role="user",
                content=(
                    "Judge consistency. Return JSON with keys: verdict "
                    "('pass'|'fail'|'low_confidence'), confidence, failure_types, "
                    "mismatches, repair_hint, must_fix. Use failure_types such as "
                    "semantic_field_mapping, unresolved_mapping_ambiguity, "
                    "aggregation_grain, output_projection, filter_semantics, "
                    "join_semantics, metric_formula.\n\n"
                    f"{json.dumps(packet, ensure_ascii=False, indent=2, default=str)}"
                ),
            ),
        ],
        temperature=0.0,
        stream_label="consistency judge",
        max_tokens=1000,
    )
    verdict = _parse_json_object(raw)
    return verdict


_REPAIR_SYSTEM = """You are the Semantic Repair module for a table/data QA agent.
Repair the previous Python program so it becomes consistent with the
Semantic Analyst plan and Consistency Judge feedback.

Rules:
- Output exactly one fenced ```python``` block.
- Do not include prose.
- Do not solve from scratch unless necessary; preserve correct loading and
  joins already present.
- Use only real context paths and real schema fields.
- Define `debug_steps` with schema_mapping, joins, filters, aggregation,
  output_columns, preview_rows when applicable.
- Also define lightweight trace fields: used_tables, used_columns,
  join_keys, filter_conditions, derived_fields, and unmapped_question_terms.
  Use empty lists when a category does not apply.
- Always define `debug_steps["schema_inspection"]` and fill it with
  `list(df.columns)` after each dataframe/SQL/JSON load. Then make
  `debug_steps["schema_mapping"]` choose only fields present in that
  inspection or in Source Capabilities.
- Treat knowledge.md as semantic hypotheses/rules only. Never use a
  column/key/table solely because knowledge.md mentions that word.
- Do not apply a knowledge.md formula merely because it exists. Apply it
  only when the question explicitly asks for the formula's target metric or
  when no direct real field can answer the requested output.
- If semantic_plan contains `plan_failure`, treat it as analyst failure
  context, not as evidence that there is no semantic risk. Use
  `cheap_semantic_assessment` and its risks as the fallback source of
  truth for what must be checked or repaired.
- Assign the final result to `answer`.
- If the judge says the analyst mapping is ambiguous, re-evaluate the
  competing real fields using schema names, sample values, and
  knowledge.md semantics; do not blindly follow the previous code.
- If judge_result includes `zero_rows`, do not simply rewrite the whole
  solution. Add a targeted zero-row probe in `debug_steps['zero_row_probe']`
  with base row counts, row counts after each join/filter, candidate value
  counts for every filter key, sample unique values, and the first operation
  that drops rows to zero. Patch only that identified operation.
"""


def run_semantic_repair(  # noqa: PLR0913
    *,
    task: PublicTask,
    model: OpenAIModelAdapter,
    compiled_task: CompiledTask,
    semantic_plan: dict[str, Any],
    judge_result: dict[str, Any],
    previous_program: str,
    answer: AnswerTable | None,
    debug_steps: dict[str, Any] | None,
    schema_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packet = {
        "question": task.question,
        "compiled_task": {
            "task_type": compiled_task.task_type,
            "answer_type": compiled_task.answer_type,
            "operations": list(compiled_task.operations),
            "sources": _source_summary(compiled_task),
        },
        "schema_diagnostics": schema_diagnostics or {},
        "semantic_plan": semantic_plan,
        "judge_result": judge_result,
        "previous_debug_steps": debug_steps or {},
        "previous_answer_preview": _answer_preview(answer),
        "previous_program": previous_program,
    }
    raw = model.complete(
        [
            ModelMessage(role="system", content=_REPAIR_SYSTEM),
            ModelMessage(
                role="user",
                content=(
                    "Repair the program according to the semantic mismatch. "
                    "Return only one fenced python block.\n\n"
                    f"{json.dumps(packet, ensure_ascii=False, indent=2, default=str)}"
                ),
            ),
        ],
        temperature=0.0,
        stream_label="semantic repair",
        max_tokens=1800,
    )
    program = extract_python_program(raw)
    return {
        "raw_response": raw,
        "program": program,
        "succeeded": bool(program.strip()),
        "failure_reason": None if program.strip() else "semantic_repair_empty_program",
    }


# ---------------------------------------------------------------------------
# SemanticConsistencyPipeline — first-class pipeline object
# ---------------------------------------------------------------------------

class SemanticConsistencyPipeline:
    """Orchestrates the 审题官 (Analyst) + 执行官 (Judge + Repair) halves.

    Usage in OperatorExecutor::

        pipeline = SemanticConsistencyPipeline(model, context, enabled=True)
        semantic_plan = pipeline.plan(task)             # before codegen
        result = pipeline.judge_and_repair(task, result, semantic_plan)

    Separating plan() from judge_and_repair() lets the semantic plan flow
    into the codegen prompt without coupling planning to post-execution logic.
    """

    def __init__(
        self,
        model: OpenAIModelAdapter,
        context: Any,   # ExecutionContext — typed as Any to avoid circular import
        *,
        enabled: bool = True,
        max_repairs: int = 3,
    ) -> None:
        self.model = model
        self.context = context
        self.enabled = enabled
        self.max_repairs = max_repairs
        self.last_plan_failure: dict[str, Any] | None = None
        # "llm" when the most recent plan() call hit the analyst, "cache"
        # when it was served from the on-disk plan cache, "" when plan()
        # did not run or was skipped. Consumed by OperatorExecutor's
        # manifest and the router's semantic_consistency_audit so the
        # cost path is visible in trace.json.
        self.last_plan_source: str = ""

    # ------------------------------------------------------------------
    # Phase 1: semantic planning (no code)
    # ------------------------------------------------------------------

    def plan(self, task: PublicTask, *, force: bool = False) -> dict[str, Any] | None:
        """Run the semantic analyst.  Returns plan dict, or None if skipped/failed."""
        self.last_plan_failure = None
        self.last_plan_source = ""
        if not self.enabled:
            self.last_plan_failure = {"stage": "disabled", "force": force}
            return None
        compiled = self.context.compiled_task
        profile = compiled.execution_profile or {}
        low_complexity = profile.get("operation_complexity") in {
            "direct_lookup",
            "single_filter",
        }
        # Low-complexity table questions should stay fast and tool-first
        # regardless of the dataset difficulty label. If the direct path
        # fails or produces an invalid/suspicious answer, OperatorExecutor
        # calls plan(..., force=True) and escalates into semantic repair.
        if (
            not force
            and compiled.task_type in {"table_computation", "table_with_semantic_rule"}
            and low_complexity
            and not profile.get("semantic_rule_required")
        ):
            self.last_plan_failure = {
                "stage": "fast_path_skip",
                "force": force,
                "reason": "low_complexity_programmatic_task",
            }
            return None
        forced_but_normally_gated = force and not should_run_semantic_consistency(compiled)
        force_allowed = (
            compiled.task_type in {
                "table_computation",
                "mixed_context",
                "table_with_semantic_rule",
            }
            and "needs_vision" not in set(compiled.ambiguity_flags)
            and "unsupported_file_type" not in set(compiled.ambiguity_flags)
        )
        if forced_but_normally_gated and not force_allowed:
            self.last_plan_failure = {
                "stage": "gate_blocked",
                "force": force,
                "task_type": compiled.task_type,
                "ambiguity_flags": list(compiled.ambiguity_flags),
                "reason": "forced semantic consistency is not allowed for this task shape",
            }
            return None
        if not force and not should_run_semantic_consistency(compiled):
            self.last_plan_failure = {
                "stage": "gate_skip",
                "force": force,
                "task_type": compiled.task_type,
                "ambiguity_flags": list(compiled.ambiguity_flags),
            }
            return None
        try:
            plan = self._plan_with_cache(task=task, compiled=compiled)
        except BudgetExceeded:
            self.last_plan_failure = {
                "stage": "budget_exceeded",
                "force": force,
            }
            raise
        except Exception as exc:  # noqa: BLE001
            self.last_plan_failure = {
                "stage": "analyst_exception",
                "force": force,
                "exception_type": type(exc).__name__,
                "message": str(exc)[:1000],
            }
            return None

        rule_resolution = _resolve_semantic_rules(
            task=task,
            compiled_task=compiled,
            plan=plan,
        )
        effective_plan = _apply_rule_resolution(plan, rule_resolution)
        if rule_resolution.get("resolved_filters") or rule_resolution.get("unresolved_filters"):
            effective_plan["original_semantic_plan"] = plan
        if forced_but_normally_gated:
            effective_plan["_force_semantic_consistency"] = True
            effective_plan["_force_reason"] = "cheap_semantic_guard_escalation"

        try:
            from data_agent_baseline.progress import get_progress_logger
            logger = get_progress_logger()
            if logger is not None:
                logger.codegen_debug(debug={"semantic_plan": {
                    k: effective_plan.get(k) for k in (
                        "schema_mapping", "join_plan", "filters",
                        "aggregation", "output", "confidence",
                    )
                }})
                if rule_resolution.get("resolved_filters") or rule_resolution.get("unresolved_filters"):
                    logger.codegen_debug(debug={"rule_resolution": rule_resolution})
        except Exception:  # noqa: BLE001
            pass

        return effective_plan

    # ------------------------------------------------------------------
    # Phase 1 helper: on-disk cache for the analyst call
    # ------------------------------------------------------------------

    def _plan_with_cache(
        self,
        *,
        task: PublicTask,
        compiled: CompiledTask,
    ) -> dict[str, Any]:
        """Read analyst plan from disk cache, or call the analyst and cache.

        Cache invalidation is automatic: the key mixes model id, task id,
        the question text, and a fingerprint of the source capabilities.
        Any of: changing the model, rewording the question, or adding /
        removing a context file (including the synthesized CSVs written
        by the record_text extractor) produces a fresh key and a fresh
        analyst call. The cached payload is the RAW analyst plan before
        rule resolution, so knowledge.md re-reads still happen every run
        and a stale plan can never leak into _apply_rule_resolution.

        On any read error the cache is silently bypassed; on any write
        error we log-and-ignore. The analyst call itself is the source
        of truth — the cache is an accelerator, not a dependency.
        """
        from data_agent_baseline.agents.semantic_plan_cache import (
            PlanCacheKeyInputs,
            build_source_fingerprint,
            cache_read,
            cache_write,
            compute_plan_cache_key,
            default_cache_dir,
        )

        cache_dir = default_cache_dir()
        model_id = getattr(self.model, "model", "") or ""
        key_inputs = PlanCacheKeyInputs(
            model_id=model_id,
            task_id=task.task_id,
            question=task.question or "",
            source_fingerprint=build_source_fingerprint(compiled.source_capabilities),
        )
        key = compute_plan_cache_key(key_inputs)
        cached = cache_read(cache_dir, key)
        if cached is not None:
            self.last_plan_source = "cache"
            return cached

        plan = run_semantic_analyst(
            task=task,
            model=self.model,
            compiled_task=compiled,
            schema_diagnostics=self.context.schema_diagnostics,
        )
        self.last_plan_source = "llm"
        cache_write(cache_dir, key, plan=plan)
        return plan

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 2: consistency judge + bounded semantic repair
    # ------------------------------------------------------------------

    def judge_and_repair(
        self,
        task: PublicTask,
        result: CodegenRunResult,
        semantic_plan: dict[str, Any] | None,
    ) -> CodegenRunResult:
        """Compare plan vs code vs answer; repair up to max_repairs times."""
        if not self.enabled or not semantic_plan:
            return result
        force_enabled = bool(semantic_plan.get("_force_semantic_consistency"))
        if (
            not force_enabled
            and not should_run_semantic_consistency(self.context.compiled_task)
        ):
            return result

        from data_agent_baseline.agents.local_repair import issues_from_exec_error, try_program_repair
        from data_agent_baseline.agents.repair_coordinator import exec_program
        from data_agent_baseline.agents.static_checker import check_program, has_blocking
        from data_agent_baseline.progress import get_progress_logger

        compiled = self.context.compiled_task
        trace = SemanticConsistencyResult(
            semantic_plan=semantic_plan.get("original_semantic_plan", semantic_plan),
            rule_resolution=semantic_plan.get("rule_resolution"),
            effective_plan=semantic_plan,
            final_verdict="running",
        )
        current = result
        logger = get_progress_logger()
        zero_row_probe_required = False

        for attempt in range(self.max_repairs + 1):
            debug_steps = _extract_debug_steps(current.exec_stdout) or {}
            effective_plan, applied_overrides = _apply_plan_overrides(
                semantic_plan, debug_steps
            )
            trace.effective_plan = effective_plan
            static_issues: list[dict[str, Any]] = []
            for item in current.manifest or []:
                if isinstance(item, dict) and isinstance(item.get("static_issues"), list):
                    static_issues.extend(item["static_issues"])
                if isinstance(item, dict):
                    for key in ("local_repair_log", "post_schema_retry_local_repair_log"):
                        if not isinstance(item.get(key), list):
                            continue
                        for entry in item[key]:
                            if isinstance(entry, dict) and isinstance(entry.get("issues"), list):
                                static_issues.extend(entry["issues"])
            preflight_judge = _preflight_judge_failure(
                compiled_task=compiled,
                answer=current.answer,
                debug_steps=debug_steps,
                exec_stdout=current.exec_stdout,
                exec_stderr=current.exec_stderr,
                failure_reason=current.failure_reason,
                execution_succeeded=current.succeeded,
                static_issues=static_issues[-12:],
            )
            if (
                preflight_judge is None
                and zero_row_probe_required
                and not _has_zero_row_probe(debug_steps)
            ):
                preflight_judge = {
                    "verdict": "fail",
                    "confidence": 1.0,
                    "failure_types": ["zero_row_probe_missing"],
                    "mismatches": [
                        "Previous repair round hit zero_rows, but the repaired program did not record debug_steps['zero_row_probe']."
                    ],
                    "repair_hint": (
                        "Add debug_steps['zero_row_probe'] with base row counts, "
                        "row counts after each join/filter, candidate value counts, "
                        "sample unique values, and the first operation that drops "
                        "rows to zero before attempting another semantic pass."
                    ),
                    "must_fix": True,
                }
            try:
                if preflight_judge is not None:
                    judge = preflight_judge
                else:
                    judge = judge_consistency(
                        task=task,
                        model=self.model,
                        compiled_task=compiled,
                        semantic_plan=effective_plan,
                        program=current.program,
                        answer=current.answer,
                        debug_steps=debug_steps,
                        exec_stdout=current.exec_stdout,
                        schema_diagnostics=self.context.schema_diagnostics,
                        static_issues=static_issues[-12:],
                    )
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                trace.final_verdict = "judge_failed"
                trace.failure_reason = str(exc)
                break

            trace.judge_history.append({
                "attempt": attempt,
                "applied_plan_overrides": applied_overrides,
                **judge,
            })
            if logger is not None:
                logger.codegen_debug(debug={"consistency_judge": {
                    "attempt": attempt,
                    "verdict": judge.get("verdict"),
                    "confidence": judge.get("confidence"),
                    "failure_types": judge.get("failure_types"),
                    "mismatches": judge.get("mismatches"),
                    "repair_hint": judge.get("repair_hint"),
                }})

            verdict = str(judge.get("verdict", "")).strip().lower()
            failure_types = [str(x) for x in (judge.get("failure_types") or [])]
            if "zero_rows" in failure_types:
                zero_row_probe_required = True
            try:
                confidence = float(judge.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

            if verdict == "pass" and confidence >= 0.55:
                trace.final_verdict = "pass"
                current.manifest = list(current.manifest or []) + [
                    {"semantic_consistency": trace.to_dict()}
                ]
                return current

            if attempt >= self.max_repairs:
                trace.final_verdict = verdict or "fail"
                trace.failure_reason = (
                    "semantic_consistency_failed:"
                    + ",".join(failure_types)
                )
                current.succeeded = False
                current.failure_reason = trace.failure_reason
                break

            try:
                repair = run_semantic_repair(
                    task=task,
                    model=self.model,
                    compiled_task=compiled,
                    semantic_plan=effective_plan,
                    judge_result=judge,
                    previous_program=current.program,
                    answer=current.answer,
                    debug_steps=debug_steps,
                    schema_diagnostics=self.context.schema_diagnostics,
                )
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                trace.repair_history.append({
                    "attempt": attempt,
                    "succeeded": False,
                    "failure_reason": f"semantic_repair_runtime_error:{exc}",
                })
                trace.final_verdict = "repair_failed"
                trace.failure_reason = f"semantic_repair_runtime_error:{exc}"
                current.succeeded = False
                current.failure_reason = trace.failure_reason
                break

            program = str(repair.get("program") or "")
            if not program.strip():
                trace.repair_history.append({
                    "attempt": attempt,
                    "succeeded": False,
                    "failure_reason": repair.get("failure_reason") or "empty_program",
                })
                current.succeeded = False
                current.failure_reason = str(
                    repair.get("failure_reason") or "semantic_repair_empty_program"
                )
                trace.final_verdict = "repair_failed"
                trace.failure_reason = current.failure_reason
                break

            # Static-check the repaired program; try a local patch if blocked.
            static_issues = check_program(program, capabilities=compiled.source_capabilities)
            if static_issues and has_blocking(static_issues):
                local_outcome = try_program_repair(
                    program=program,
                    issues=static_issues,
                    compiled=compiled,
                    semantic_plan=effective_plan,
                )
                if local_outcome is not None and local_outcome.succeeded:
                    program = local_outcome.patched_program
                    trace.repair_history.append({
                        "attempt": attempt,
                        "stage": "semantic_repair_static_local_patch",
                        "issues": [it.to_dict() for it in static_issues],
                        "local_outcome": local_outcome.to_dict(),
                    })
                else:
                    trace.repair_history.append({
                        "attempt": attempt,
                        "succeeded": False,
                        "failure_reason": "semantic_repair_static_error",
                        "issues": [it.to_dict() for it in static_issues],
                    })
                    current.succeeded = False
                    current.failure_reason = "semantic_repair_static_error"
                    trace.final_verdict = "repair_failed"
                    trace.failure_reason = current.failure_reason
                    break

            new_result = exec_program(
                task=task,
                program=program,
                raw_response=str(repair.get("raw_response") or ""),
                manifest=list(current.manifest or []),
                label="semantic_repair",
            )
            if not new_result.succeeded:
                runtime_issues = issues_from_exec_error(new_result.exec_stderr)
                runtime_issues.extend(issues_from_exec_error(str(new_result.failure_reason or "")))
                if runtime_issues and has_blocking(runtime_issues):
                    local_outcome = try_program_repair(
                        program=program,
                        issues=runtime_issues,
                        compiled=compiled,
                        semantic_plan=effective_plan,
                    )
                    if local_outcome is not None and local_outcome.succeeded:
                        new_result = exec_program(
                            task=task,
                            program=local_outcome.patched_program,
                            raw_response=str(repair.get("raw_response") or ""),
                            manifest=list(current.manifest or []),
                            label="semantic_repair_local_patch",
                        )
                        trace.repair_history.append({
                            "attempt": attempt,
                            "stage": "semantic_repair_exec_local_patch",
                            "issues": [it.to_dict() for it in runtime_issues],
                            "local_outcome": local_outcome.to_dict(),
                            "succeeded": bool(new_result.succeeded),
                            "failure_reason": new_result.failure_reason,
                        })
            trace.repair_history.append({
                "attempt": attempt,
                "succeeded": bool(new_result.succeeded),
                "failure_reason": new_result.failure_reason,
            })
            current = new_result

        current.manifest = list(current.manifest or []) + [
            {"semantic_consistency": trace.to_dict()}
        ]
        return current
