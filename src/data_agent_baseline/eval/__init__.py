"""Scoring utilities for DABench predictions.

Implements the official rubric described in the task:

    score = max(0, recall - lambda * (extra_cols / pred_cols))

where two columns "match" iff their content signatures are equal,
ignoring column names and row order.
"""

from data_agent_baseline.eval.answer_validator import (
    ValidationIssue,
    ValidationResult,
    validate_answer_table,
)
from data_agent_baseline.eval.column_match import (
    ColumnMatchResult,
    PerTaskScore,
    column_signature,
    match_columns,
    score_run,
    score_table,
)

__all__ = [
    "ColumnMatchResult",
    "PerTaskScore",
    "ValidationIssue",
    "ValidationResult",
    "column_signature",
    "match_columns",
    "score_run",
    "score_table",
    "validate_answer_table",
]
