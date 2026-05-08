"""Local validation of an :class:`AnswerTable` before submission.

The official scorer is unforgiving: it matches column content signatures
exactly, so a table with 0 rows or a column full of empty strings will
score 0 even if the agent reports ``succeeded=true``. We catch the most
obvious failure modes BEFORE the runner writes a prediction.csv:

- ``column_count == 0`` : nothing to score.
- ``row_count == 0``    : empty table (often from a buggy filter).
- entirely-empty column : a column whose every cell is empty / null.
- ragged rows           : rows with a different cell count from the header.
- mixed-type column     : warning only — sometimes legitimate.

The validator returns a structured ``ValidationResult`` so the trace can
record exactly what went wrong, and the cascade fallback in the router
can decide to retry on a heavier route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


# ---------------------------------------------------------------------------
# Issue / Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One thing that's wrong (or suspicious) about a candidate answer."""

    code: str            # e.g. "empty_columns", "ragged_row", "mixed_types"
    severity: str        # "error" | "warning"
    message: str
    location: dict[str, Any] = field(default_factory=dict)
    repair_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "location": dict(self.location),
            "repair_hint": self.repair_hint,
        }


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    column_count: int = 0
    row_count: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [item for item in self.issues if item.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [item for item in self.issues if item.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "column_count": self.column_count,
            "row_count": self.row_count,
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
        }


# ---------------------------------------------------------------------------
# Cell-level helpers
# ---------------------------------------------------------------------------


_EMPTYISH = {"", "null", "none", "nan", "n/a", "na"}


def _is_emptyish(cell: Any) -> bool:
    if cell is None:
        return True
    text = str(cell).strip().lower()
    return text in _EMPTYISH


def _python_type_label(cell: Any) -> str:
    if cell is None:
        return "null"
    if isinstance(cell, bool):
        return "bool"
    if isinstance(cell, (int, float)):
        return "number"
    text = str(cell).strip()
    if not text:
        return "empty"
    # Soft-typed: numeric-looking strings count as number for type-stability check.
    try:
        float(text)
        return "number"
    except ValueError:
        return "string"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_answer_table(
    answer: AnswerTable | dict[str, Any] | None,
    *,
    allow_empty_rows: bool = False,
    warn_on_mixed_types: bool = True,
) -> ValidationResult:
    """Run all checks on a candidate answer.

    Accepts an :class:`AnswerTable`, the dict form produced by
    ``AnswerTable.to_dict()``, or ``None``. Returns a structured result.
    """
    if answer is None:
        return ValidationResult(
            valid=False,
            issues=[ValidationIssue(
                code="missing_answer",
                severity="error",
                message="No AnswerTable was produced.",
                repair_hint="rerun the selected executor or switch to a fallback executor that can materialize an AnswerTable",
            )],
        )

    if isinstance(answer, AnswerTable):
        columns = list(answer.columns)
        rows = [list(row) for row in answer.rows]
    elif isinstance(answer, dict):
        columns = list(answer.get("columns") or [])
        rows = [list(row) for row in (answer.get("rows") or [])]
    else:
        return ValidationResult(
            valid=False,
            issues=[ValidationIssue(
                code="invalid_answer_type",
                severity="error",
                message=f"Answer must be AnswerTable / dict, got {type(answer).__name__}.",
                repair_hint="convert the executor output into {'columns': [...], 'rows': [[...]]}",
            )],
        )

    issues: list[ValidationIssue] = []
    column_count = len(columns)
    row_count = len(rows)

    if column_count == 0:
        issues.append(ValidationIssue(
            code="no_columns",
            severity="error",
            message="Answer has zero columns.",
            repair_hint="inspect the final projection and include exactly the requested answer columns",
        ))

    if row_count == 0 and not allow_empty_rows:
        issues.append(ValidationIssue(
            code="no_rows",
            severity="error",
            message="Answer has zero rows.",
            repair_hint="check filters, joins, and key normalization; rerun only the failing data operation",
        ))

    # Ragged rows + per-cell stats.
    column_buckets: list[list[Any]] = [[] for _ in range(column_count)]
    for row_index, row in enumerate(rows):
        if column_count and len(row) != column_count:
            issues.append(ValidationIssue(
                code="ragged_row",
                severity="error",
                message=(
                    f"Row {row_index} has {len(row)} cells but the header "
                    f"has {column_count} columns."
                ),
                location={"row_index": row_index},
                repair_hint="normalize every row to the same width as the columns list",
            ))
        for col_index in range(column_count):
            cell = row[col_index] if col_index < len(row) else None
            column_buckets[col_index].append(cell)

    # Whole-empty columns.
    for col_index, bucket in enumerate(column_buckets):
        if not bucket:
            continue
        if all(_is_emptyish(cell) for cell in bucket):
            issues.append(ValidationIssue(
                code="empty_column",
                severity="error",
                message=(
                    f"Column #{col_index} ('{columns[col_index]}') is "
                    f"entirely empty / null."
                ),
                location={"column_index": col_index, "column_name": columns[col_index]},
                repair_hint="remove the empty projected column or repair the field mapping that produced it",
            ))

    # Mixed-type warning (don't fail on it).
    if warn_on_mixed_types:
        for col_index, bucket in enumerate(column_buckets):
            type_set = {
                _python_type_label(cell)
                for cell in bucket
                if not _is_emptyish(cell)
            }
            type_set.discard("null")
            type_set.discard("empty")
            if len(type_set) > 1:
                issues.append(ValidationIssue(
                    code="mixed_types",
                    severity="warning",
                    message=(
                        f"Column #{col_index} ('{columns[col_index]}') has "
                        f"mixed value types: {sorted(type_set)}."
                    ),
                    location={"column_index": col_index, "column_name": columns[col_index]},
                    repair_hint="coerce values to a stable type if the answer column is supposed to be numeric or categorical",
                ))

    has_error = any(item.severity == "error" for item in issues)
    return ValidationResult(
        valid=not has_error,
        issues=issues,
        column_count=column_count,
        row_count=row_count,
    )
