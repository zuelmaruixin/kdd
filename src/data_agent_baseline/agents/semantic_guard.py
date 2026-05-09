"""Cheap semantic-risk guard for fast-path operator answers.

This module is deliberately non-LLM. It decides whether a valid-looking
programmatic answer is safe enough to skip the heavy Semantic Analyst/Judge
path, using only:

- execution/answer validation status,
- debug trace emitted by codegen,
- deterministic schema grounding candidates,
- known source schema from CompiledTask,
- fallback/repair history.

It is not a semantic judge. It only says "low risk, return fast" or "there is
enough cheap evidence of risk; escalate to the expensive semantic path".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from data_agent_baseline.agents.schema_grounding import (
    ConceptBinding,
    extract_question_concepts,
    match_concept,
)
from data_agent_baseline.agents.tablellm_direct import _extract_debug_steps
from data_agent_baseline.eval.answer_validator import validate_answer_table

if TYPE_CHECKING:
    from data_agent_baseline.agents.tablellm_direct import CodegenRunResult
    from data_agent_baseline.agents.task_compiler import CompiledTask
    from data_agent_baseline.benchmark.schema import PublicTask


@dataclass(frozen=True, slots=True)
class SemanticRisk:
    code: str
    severity: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    weight: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "evidence": dict(self.evidence),
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class SemanticRiskAssessment:
    risks: tuple[SemanticRisk, ...]
    grounding: tuple[dict[str, Any], ...] = ()
    trace_present: bool = False
    score: float = 0.0

    @property
    def should_escalate(self) -> bool:
        if any(risk.severity == "error" for risk in self.risks):
            return True
        return self.score >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_escalate": self.should_escalate,
            "score": round(self.score, 3),
            "trace_present": self.trace_present,
            "risks": [risk.to_dict() for risk in self.risks],
            "grounding": list(self.grounding),
        }


_RISK_TERMS = {
    "abnormal", "normal", "severe", "severity", "diagnosis", "disease",
    "status", "format", "ratio", "percentage", "percent", "formula",
    "eligible", "active", "thrombosis", "admission", "inpatient",
    "outpatient",
}

_FORMULA_TRIGGER_RE = re.compile(
    r"\b("
    r"formula|derive|derived|ratio|percentage|percent|"
    r"rate|score|index|normalized|weighted|average|mean"
    r")\b",
    re.I,
)

_FORMULA_TRACE_RE = re.compile(
    r"\b(formula|equation|derive|derived|computed?|ratio|percentage|percent|"
    r"rate|score|index|weighted)\b|"
    r"(?:[A-Za-z_]\w*|\d+(?:\.\d+)?)\s*[*/]\s*(?:[A-Za-z_]\w*|\d+(?:\.\d+)?)",
    re.I,
)


def assess_cheap_semantic_risk(
    *,
    task: PublicTask,
    compiled: CompiledTask,
    result: CodegenRunResult,
) -> SemanticRiskAssessment:
    risks: list[SemanticRisk] = []

    if not result.succeeded or result.answer is None:
        risks.append(SemanticRisk(
            code="execution_failed_or_missing_answer",
            severity="error",
            message="Program did not produce a successful AnswerTable.",
            evidence={"failure_reason": result.failure_reason},
            weight=1.0,
        ))
        return _assessment(risks, grounding=[], trace_present=False)

    if not result.answer.rows:
        risks.append(SemanticRisk(
            code="empty_answer_rows",
            severity="error",
            message="Program succeeded and produced a structurally valid but empty AnswerTable.",
            evidence={"columns": list(result.answer.columns)},
            weight=0.75,
        ))
    validation = validate_answer_table(result.answer)
    if not validation.valid:
        risks.append(SemanticRisk(
            code="invalid_answer",
            severity="error",
            message="AnswerTable failed cheap structural validation.",
            evidence=validation.to_dict(),
            weight=1.0,
        ))
        return _assessment(risks, grounding=[], trace_present=False)

    debug_steps = _extract_debug_steps(result.exec_stdout) or {}
    trace_present = isinstance(debug_steps, dict) and bool(debug_steps)

    risks.extend(_fallback_risks(result.manifest or []))
    risks.extend(_trace_risks(compiled=compiled, debug_steps=debug_steps))
    risks.extend(_formula_risks(task=task, debug_steps=debug_steps))
    grounding, grounding_risks = _grounding_risks(task=task, compiled=compiled)
    risks.extend(grounding_risks)
    risks.extend(_field_coverage_risks(
        task=task,
        compiled=compiled,
        debug_steps=debug_steps,
    ))

    return _assessment(risks, grounding=grounding, trace_present=trace_present)


def _assessment(
    risks: list[SemanticRisk],
    *,
    grounding: list[dict[str, Any]],
    trace_present: bool,
) -> SemanticRiskAssessment:
    deduped: list[SemanticRisk] = []
    seen: set[tuple[str, str]] = set()
    for risk in risks:
        key = (risk.code, str(risk.evidence))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(risk)
    score = min(1.0, sum(risk.weight for risk in deduped))
    return SemanticRiskAssessment(
        risks=tuple(deduped),
        grounding=tuple(grounding),
        trace_present=trace_present,
        score=score,
    )


def _fallback_risks(manifest: list[dict[str, Any]]) -> list[SemanticRisk]:
    risks: list[SemanticRisk] = []
    for item in manifest:
        if not isinstance(item, dict):
            continue
        if item.get("schema_retry") == "applied":
            risks.append(SemanticRisk(
                code="suspicious_fallback:schema_retry",
                severity="error",
                message="Schema retry was needed before producing the answer.",
                evidence={"stage": "schema_retry"},
                weight=0.8,
            ))
        for key, weight in (
            ("local_repair_log", 0.35),
            ("post_schema_retry_local_repair_log", 0.65),
        ):
            log = item.get(key)
            if isinstance(log, list) and log:
                risks.append(SemanticRisk(
                    code=f"suspicious_fallback:{key}",
                    severity="warning" if weight < 0.5 else "error",
                    message=f"Program answer required {key}.",
                    evidence={"entries": _compact_repair_log(log)},
                    weight=weight,
                ))
    return risks


def _compact_repair_log(log: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for entry in log[:5]:
        if not isinstance(entry, dict):
            continue
        compact.append({
            "stage": entry.get("stage"),
            "action": (entry.get("outcome") or {}).get("action")
            if isinstance(entry.get("outcome"), dict) else None,
            "exec_succeeded": entry.get("exec_succeeded"),
            "post_failure_reason": entry.get("post_failure_reason"),
        })
    return compact


def _trace_risks(
    *,
    compiled: CompiledTask,
    debug_steps: dict[str, Any],
) -> list[SemanticRisk]:
    risks: list[SemanticRisk] = []
    profile = compiled.execution_profile or {}
    operations = set(compiled.operations)
    table_like = compiled.task_type in {
        "table_computation",
        "table_with_semantic_rule",
        "mixed_context",
    }

    if table_like and not _has_nonempty_dict(debug_steps.get("schema_inspection")):
        risks.append(SemanticRisk(
            code="trace_missing:schema_inspection",
            severity="error",
            message="debug_steps lacks schema_inspection from real loaded columns.",
            evidence={"task_type": compiled.task_type},
            weight=0.8,
        ))

    used_columns = _trace_columns(debug_steps)
    if table_like and not used_columns:
        risks.append(SemanticRisk(
            code="trace_missing:used_columns",
            severity="error",
            message="debug_steps lacks used_columns/schema_mapping/filter field trace.",
            evidence={"task_type": compiled.task_type},
            weight=0.7,
        ))

    if _nonempty_list(debug_steps.get("unmapped_question_terms")):
        risks.append(SemanticRisk(
            code="trace_unmapped_question_terms",
            severity="error",
            message="Code reported unmapped question terms.",
            evidence={"terms": _as_strings(debug_steps.get("unmapped_question_terms"))[:8]},
            weight=0.75,
        ))

    if "filter" in operations and not (
        _nonempty(debug_steps.get("filters"))
        or _nonempty(debug_steps.get("filter_conditions"))
    ):
        risks.append(SemanticRisk(
            code="trace_missing:filter_conditions",
            severity="error",
            message="Question requires filtering but trace lacks filter conditions.",
            evidence={"operations": list(compiled.operations)},
            weight=0.65,
        ))

    if (
        "join" in operations
        or profile.get("source_shape") == "multi_table"
    ) and not (
        _nonempty(debug_steps.get("joins"))
        or _nonempty(debug_steps.get("join_keys"))
    ):
        risks.append(SemanticRisk(
            code="trace_missing:join_keys",
            severity="error",
            message="Multi-table/join task lacks join key trace.",
            evidence={"source_shape": profile.get("source_shape")},
            weight=0.65,
        ))

    if used_columns:
        unknown = _unknown_trace_columns(
            used_columns=used_columns,
            derived_columns=_as_strings(debug_steps.get("derived_fields")),
            compiled=compiled,
            debug_steps=debug_steps,
        )
        if unknown:
            risks.append(SemanticRisk(
                code="trace_unknown_used_columns",
                severity="error",
                message=(
                    "Trace claims columns not found in schema_inspection "
                    "or SourceCapabilities."
                ),
                evidence={"columns": unknown[:8]},
                weight=0.8,
            ))

    zero_points = _zero_count_paths(debug_steps.get("intermediate_counts"))
    zero_points.extend(
        f"filters.{path}" for path in _zero_count_paths(debug_steps.get("filters"))
    )
    if zero_points:
        risks.append(SemanticRisk(
            code="suspicious_trace:zero_intermediate_count",
            severity="error",
            message="Intermediate trace contains a zero row count.",
            evidence={"zero_count_paths": zero_points[:8]},
            weight=0.8,
        ))

    return risks


def _field_coverage_risks(
    *,
    task: PublicTask,
    compiled: CompiledTask,
    debug_steps: dict[str, Any],
) -> list[SemanticRisk]:
    """Check that risky question concepts are covered by actual traced fields.

    Schema grounding by itself only says "the schema has plausible fields".
    The fast path is safe only when the generated program's trace shows that
    those fields, or defensible alternatives, were actually used. This catches
    valid-shaped answers that filtered on a nearby but wrong field.
    """
    used_columns = _trace_columns(debug_steps)
    if not used_columns:
        return []

    used_norms = _normalized_trace_column_set(used_columns)
    risks: list[SemanticRisk] = []
    for concept in extract_question_concepts(task.question, max_concepts=10):
        if concept.lower() not in _RISK_TERMS:
            continue
        matches = match_concept(concept, compiled.source_capabilities, top_k=3)
        if not matches:
            continue
        plausible = [match for match in matches if match.score >= 0.68]
        if not plausible:
            continue
        if any(_column_covered(match.column, used_norms) for match in plausible):
            continue
        risks.append(SemanticRisk(
            code="field_coverage_mismatch",
            severity="error",
            message=(
                "Risk-bearing question concept has grounded schema candidates, "
                "but the program trace does not show any of those fields being used."
            ),
            evidence={
                "concept": concept,
                "expected_candidate_fields": [
                    {
                        "path": match.file_path,
                        "column": match.column,
                        "score": round(match.score, 3),
                    }
                    for match in plausible
                ],
                "used_columns": used_columns[:12],
            },
            weight=0.75,
        ))
    return risks


def _formula_risks(*, task: PublicTask, debug_steps: dict[str, Any]) -> list[SemanticRisk]:
    used_rules = debug_steps.get("knowledge_rules_used")
    if not _nonempty(used_rules):
        return []
    trace_text = " ".join(_as_strings(used_rules))
    derived_text = " ".join(_as_strings(debug_steps.get("derived_fields")))
    formula_like = bool(_FORMULA_TRACE_RE.search(f"{trace_text} {derived_text}"))
    if not formula_like:
        return []
    if _FORMULA_TRIGGER_RE.search(task.question):
        return []
    return [SemanticRisk(
        code="knowledge_formula_without_question_trigger",
        severity="error",
        message=(
            "Trace indicates a knowledge formula/derived metric, but the "
            "question does not explicitly request that formula target."
        ),
        evidence={"knowledge_rules_used": _as_strings(used_rules)[:5]},
        weight=0.75,
    )]


def _grounding_risks(
    *,
    task: PublicTask,
    compiled: CompiledTask,
) -> tuple[list[dict[str, Any]], list[SemanticRisk]]:
    grounding: list[dict[str, Any]] = []
    risks: list[SemanticRisk] = []
    for concept in extract_question_concepts(task.question, max_concepts=10):
        matches = match_concept(concept, compiled.source_capabilities, top_k=3)
        summary = _grounding_summary(concept, matches)
        if summary is not None:
            grounding.append(summary)
        concept_is_risky = concept.lower() in _RISK_TERMS
        if not matches:
            if concept_is_risky:
                risks.append(SemanticRisk(
                    code="grounding_unmapped",
                    severity="error",
                    message="Risk-bearing question concept has no schema grounding candidate.",
                    evidence={"concept": concept},
                    weight=0.7,
                ))
            continue

        best = matches[0]
        second = matches[1] if len(matches) > 1 else None
        if concept_is_risky and best.score < 0.68:
            risks.append(SemanticRisk(
                code="grounding_low_confidence",
                severity="error",
                message="Risk-bearing concept has low schema grounding confidence.",
                evidence={"concept": concept, "best": best.to_dict()},
                weight=0.7,
            ))
        if (
            second is not None
            and best.score >= 0.55
            and second.score >= 0.55
            and best.score - second.score < 0.08
            and (best.file_path, best.column) != (second.file_path, second.column)
        ):
            risks.append(SemanticRisk(
                code="grounding_ambiguous",
                severity="error" if concept_is_risky else "warning",
                message="Question concept has competing close schema candidates.",
                evidence={
                    "concept": concept,
                    "best": best.to_dict(),
                    "second": second.to_dict(),
                    "margin": round(best.score - second.score, 3),
                },
                weight=0.7 if concept_is_risky else 0.3,
            ))
    return grounding, risks


def _grounding_summary(
    concept: str,
    matches: list[ConceptBinding],
) -> dict[str, Any] | None:
    if not matches:
        return {"concept": concept, "status": "unmapped"}
    best = matches[0]
    second = matches[1] if len(matches) > 1 else None
    return {
        "concept": concept,
        "best": {
            "path": best.file_path,
            "column": best.column,
            "score": round(best.score, 3),
        },
        "second": None if second is None else {
            "path": second.file_path,
            "column": second.column,
            "score": round(second.score, 3),
        },
        "margin": None if second is None else round(best.score - second.score, 3),
    }


def _trace_columns(debug_steps: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    columns.extend(_as_strings(debug_steps.get("used_columns")))
    columns.extend(_mapping_fields(debug_steps.get("schema_mapping")))
    columns.extend(_condition_fields(debug_steps.get("filter_conditions")))
    columns.extend(_condition_fields(debug_steps.get("filters")))
    return list(dict.fromkeys(_clean_column(c) for c in columns if _clean_column(c)))


def _mapping_fields(value: Any) -> list[str]:
    fields: list[str] = []
    if isinstance(value, dict):
        if any(k in value for k in ("chosen_field", "field", "column")):
            fields.extend(_as_strings(
                value.get("chosen_field") or value.get("field") or value.get("column")
            ))
        else:
            for item in value.values():
                fields.extend(_mapping_fields(item))
    elif isinstance(value, list):
        for item in value:
            fields.extend(_mapping_fields(item))
    return fields


def _condition_fields(value: Any) -> list[str]:
    fields: list[str] = []
    if isinstance(value, dict):
        fields.extend(_as_strings(
            value.get("field") or value.get("column") or value.get("chosen_field")
        ))
        for item in value.values():
            if isinstance(item, (dict, list)):
                fields.extend(_condition_fields(item))
    elif isinstance(value, list):
        for item in value:
            fields.extend(_condition_fields(item))
    return fields


def _unknown_trace_columns(
    *,
    used_columns: list[str],
    derived_columns: list[str],
    compiled: CompiledTask,
    debug_steps: dict[str, Any],
) -> list[str]:
    known = {_norm_column(col) for col in _known_columns(compiled)}
    known.update(_norm_column(col) for col in derived_columns)
    for col in _inspection_columns(debug_steps):
        known.add(_norm_column(col))
    unknown: list[str] = []
    for column in used_columns:
        norm = _norm_column(column)
        bare = _norm_column(column.split(".")[-1])
        if norm not in known and bare not in known:
            unknown.append(column)
    return list(dict.fromkeys(unknown))


def _known_columns(compiled: CompiledTask) -> list[str]:
    columns: list[str] = []
    for cap in compiled.source_capabilities:
        columns.extend(str(col) for col in cap.columns or [])
        columns.extend(str(col) for col in cap.json_record_fields or [])
        columns.extend(str(col) for col in cap.structured_record_fields or [])
        for table in cap.tables or []:
            table_name = str(table.get("table") or "")
            for col in table.get("columns") or []:
                name = ""
                if isinstance(col, dict):
                    name = str(col.get("name") or "")
                elif isinstance(col, str):
                    name = col
                if not name:
                    continue
                columns.append(name)
                if table_name:
                    columns.append(f"{table_name}.{name}")
    return list(dict.fromkeys(columns))


def _inspection_columns(debug_steps: dict[str, Any]) -> list[str]:
    inspection = debug_steps.get("schema_inspection")
    if not isinstance(inspection, dict):
        return []
    return _as_strings(inspection)


def _zero_count_paths(value: Any, *, prefix: str = "") -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [prefix or "$"] if value == 0 else []
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_zero_count_paths(item, prefix=next_prefix))
        return out
    if isinstance(value, list):
        out = []
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            out.extend(_zero_count_paths(item, prefix=next_prefix))
        return out
    return []


def _has_nonempty_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict, str)):
        return bool(value)
    return True


def _as_strings(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple, set)):
                out.extend(_as_strings(item))
            elif item is not None:
                out.append(str(item))
            elif key:
                out.append(str(key))
        return out
    if isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(_as_strings(item))
        return out
    return [str(value)]


def _clean_column(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    # Strip common pandas snippets such as df['Column'].
    match = re.search(r"\[['\"]([^'\"]+)['\"]\]", value)
    if match:
        return match.group(1)
    return value.strip("'\"` ")


def _norm_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _normalized_trace_column_set(columns: list[str]) -> set[str]:
    norms: set[str] = set()
    for column in columns:
        clean = _clean_column(column)
        if not clean:
            continue
        norms.add(_norm_column(clean))
        norms.add(_norm_column(clean.split(".")[-1]))
    return norms


def _column_covered(column: str, used_norms: set[str]) -> bool:
    norm = _norm_column(column)
    bare = _norm_column(column.split(".")[-1])
    return norm in used_norms or bare in used_norms
