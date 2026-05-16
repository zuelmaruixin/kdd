"""Column content-signature matching scorer.

The official KDD-Cup DataAgent rubric scores answers by *column content*:
- Column names are ignored.
- Row order is ignored.
- Two columns match if and only if their normalized cell multisets are equal.
- recall = matched_cols / gold_cols
- penalty = lambda * extra_pred_cols / max(pred_cols, 1)
- score = max(0, recall - penalty)

Cell normalization rules (mirroring the official spec at our best
interpretation):
- ``None`` / NaN / empty string collapse to a single sentinel.
- Numeric values are rounded with ``numeric_tolerance``; integers stay integers.
- Strings are stripped and (optionally) lowercased.

The pred-to-gold column matching is solved as a maximum bipartite matching
on the "signatures equal" graph. With at most a few dozen columns, the
greedy / DFS-based augmenting-path implementation here is plenty fast.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Cell normalization
# ---------------------------------------------------------------------------


_EMPTY_SENTINEL = "__EMPTY__"


def _looks_int(value: float) -> bool:
    if math.isnan(value) or math.isinf(value):
        return False
    return abs(value - round(value)) < 1e-9


def normalize_cell(
    raw_value: Any,
    *,
    numeric_tolerance: float = 1e-6,
    case_insensitive: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """Normalize a single cell value into a canonical string token."""
    if raw_value is None:
        return _EMPTY_SENTINEL

    # Pandas-style NaN floats.
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return _EMPTY_SENTINEL

    # Booleans: map to lower-case true/false.
    if isinstance(raw_value, bool):
        return "true" if raw_value else "false"

    # Numeric: round to tolerance.
    if isinstance(raw_value, (int, float)):
        numeric = float(raw_value)
        if numeric_tolerance > 0:
            quantum = numeric_tolerance
            rounded = round(numeric / quantum) * quantum
        else:
            rounded = numeric
        if _looks_int(rounded):
            return f"{int(round(rounded))}"
        return f"{rounded:.12g}"

    text = str(raw_value)
    if strip_whitespace:
        text = text.strip()
    if not text:
        return _EMPTY_SENTINEL

    # Try to coerce numeric-looking strings so '1' and '1.0' collapse.
    try:
        as_float = float(text)
    except ValueError:
        as_float = None
    if as_float is not None and not math.isnan(as_float):
        if numeric_tolerance > 0:
            rounded = round(as_float / numeric_tolerance) * numeric_tolerance
        else:
            rounded = as_float
        if _looks_int(rounded):
            return f"{int(round(rounded))}"
        return f"{rounded:.12g}"

    if case_insensitive:
        text = text.lower()
    return text


def column_signature(
    column_values: Iterable[Any],
    *,
    numeric_tolerance: float = 1e-6,
    case_insensitive: bool = True,
    strip_whitespace: bool = True,
) -> tuple[str, ...]:
    """Return the canonical content signature of a column.

    A signature is the sorted tuple of normalized cell tokens; two columns
    match iff their signatures are identical.
    """
    normalized = [
        normalize_cell(
            cell,
            numeric_tolerance=numeric_tolerance,
            case_insensitive=case_insensitive,
            strip_whitespace=strip_whitespace,
        )
        for cell in column_values
    ]
    normalized.sort()
    return tuple(normalized)


# ---------------------------------------------------------------------------
# Bipartite matching on equal signatures
# ---------------------------------------------------------------------------


def _max_bipartite_match(adjacency: list[list[int]], num_right: int) -> list[int]:
    """Hopcroft-Karp-flavored DFS augmenting paths.

    Returns ``match`` where ``match[right_index] = left_index or -1``.
    """
    match_right = [-1] * num_right

    def try_assign(left: int, visited: list[bool]) -> bool:
        for right in adjacency[left]:
            if visited[right]:
                continue
            visited[right] = True
            if match_right[right] == -1 or try_assign(match_right[right], visited):
                match_right[right] = left
                return True
        return False

    for left_index in range(len(adjacency)):
        visited = [False] * num_right
        try_assign(left_index, visited)
    return match_right


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColumnMatchResult:
    matched_pairs: list[tuple[int, int]]  # (gold_col_index, pred_col_index)
    matched_count: int
    pred_col_count: int
    gold_col_count: int

    @property
    def extra_pred_cols(self) -> int:
        return max(self.pred_col_count - self.matched_count, 0)


@dataclass(frozen=True, slots=True)
class PerTaskScore:
    task_id: str
    matched_count: int
    pred_col_count: int
    gold_col_count: int
    recall: float
    penalty: float
    score: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "matched_count": self.matched_count,
            "pred_col_count": self.pred_col_count,
            "gold_col_count": self.gold_col_count,
            "recall": round(self.recall, 6),
            "penalty": round(self.penalty, 6),
            "score": round(self.score, 6),
            "error": self.error,
        }


def match_columns(
    pred_columns: Sequence[Sequence[Any]],
    gold_columns: Sequence[Sequence[Any]],
    *,
    numeric_tolerance: float = 1e-6,
    case_insensitive: bool = True,
    strip_whitespace: bool = True,
) -> ColumnMatchResult:
    """Compute the column-signature bipartite match.

    Args:
        pred_columns: list of columns (each column is a list of cell values).
        gold_columns: list of columns from the reference answer.
    """
    pred_signatures = [
        column_signature(
            col,
            numeric_tolerance=numeric_tolerance,
            case_insensitive=case_insensitive,
            strip_whitespace=strip_whitespace,
        )
        for col in pred_columns
    ]
    gold_signatures = [
        column_signature(
            col,
            numeric_tolerance=numeric_tolerance,
            case_insensitive=case_insensitive,
            strip_whitespace=strip_whitespace,
        )
        for col in gold_columns
    ]

    # adjacency[gold_idx] = list of pred_idx whose signatures match.
    adjacency: list[list[int]] = []
    for gold_idx, gold_sig in enumerate(gold_signatures):
        candidates = [
            pred_idx
            for pred_idx, pred_sig in enumerate(pred_signatures)
            if pred_sig == gold_sig
        ]
        adjacency.append(candidates)

    pred_match = _max_bipartite_match(adjacency, num_right=len(pred_signatures))
    matched_pairs: list[tuple[int, int]] = []
    for pred_idx, gold_idx in enumerate(pred_match):
        if gold_idx != -1:
            matched_pairs.append((gold_idx, pred_idx))
    matched_pairs.sort()

    return ColumnMatchResult(
        matched_pairs=matched_pairs,
        matched_count=len(matched_pairs),
        pred_col_count=len(pred_signatures),
        gold_col_count=len(gold_signatures),
    )


def score_table(
    pred_columns: Sequence[Sequence[Any]],
    gold_columns: Sequence[Sequence[Any]],
    *,
    redundancy_lambda: float = 0.5,
    numeric_tolerance: float = 1e-6,
    case_insensitive: bool = True,
    strip_whitespace: bool = True,
) -> tuple[ColumnMatchResult, float, float, float]:
    """Compute (match, recall, penalty, score)."""
    match = match_columns(
        pred_columns,
        gold_columns,
        numeric_tolerance=numeric_tolerance,
        case_insensitive=case_insensitive,
        strip_whitespace=strip_whitespace,
    )
    if match.gold_col_count == 0:
        return match, 0.0, 0.0, 0.0
    recall = match.matched_count / match.gold_col_count
    if match.pred_col_count == 0:
        penalty = 0.0
    else:
        penalty = redundancy_lambda * (match.extra_pred_cols / match.pred_col_count)
    score = max(0.0, recall - penalty)
    return match, recall, penalty, score


# ---------------------------------------------------------------------------
# CSV helpers and run-level scoring
# ---------------------------------------------------------------------------


def _read_csv_columns(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = [list(row) for row in reader]
    if not rows:
        return []
    header = rows[0]
    data_rows = rows[1:]
    column_count = len(header)
    columns: list[list[str]] = [[] for _ in range(column_count)]
    for row in data_rows:
        for col_idx in range(column_count):
            cell = row[col_idx] if col_idx < len(row) else ""
            columns[col_idx].append(cell)
    return columns


def score_pair(
    *,
    task_id: str,
    pred_csv: Path,
    gold_csv: Path,
    redundancy_lambda: float = 0.5,
    numeric_tolerance: float = 1e-6,
    case_insensitive: bool = True,
    strip_whitespace: bool = True,
) -> PerTaskScore:
    if not gold_csv.exists():
        return PerTaskScore(
            task_id=task_id,
            matched_count=0,
            pred_col_count=0,
            gold_col_count=0,
            recall=0.0,
            penalty=0.0,
            score=0.0,
            error=f"missing_gold:{gold_csv}",
        )
    if not pred_csv.exists():
        gold_columns = _read_csv_columns(gold_csv)
        return PerTaskScore(
            task_id=task_id,
            matched_count=0,
            pred_col_count=0,
            gold_col_count=len(gold_columns),
            recall=0.0,
            penalty=0.0,
            score=0.0,
            error="missing_prediction",
        )

    pred_columns = _read_csv_columns(pred_csv)
    gold_columns = _read_csv_columns(gold_csv)
    match, recall, penalty, score = score_table(
        pred_columns,
        gold_columns,
        redundancy_lambda=redundancy_lambda,
        numeric_tolerance=numeric_tolerance,
        case_insensitive=case_insensitive,
        strip_whitespace=strip_whitespace,
    )
    return PerTaskScore(
        task_id=task_id,
        matched_count=match.matched_count,
        pred_col_count=match.pred_col_count,
        gold_col_count=match.gold_col_count,
        recall=recall,
        penalty=penalty,
        score=score,
    )


@dataclass(slots=True)
class RunScoreSummary:
    run_dir: Path
    gold_root: Path
    redundancy_lambda: float
    per_task: list[PerTaskScore] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.per_task)

    @property
    def scored_task_count(self) -> int:
        return sum(1 for item in self.per_task if item.error is None)

    @property
    def total_score(self) -> float:
        return sum(item.score for item in self.per_task)

    @property
    def mean_score(self) -> float:
        if not self.per_task:
            return 0.0
        return self.total_score / len(self.per_task)

    @property
    def mean_recall(self) -> float:
        if not self.per_task:
            return 0.0
        return sum(item.recall for item in self.per_task) / len(self.per_task)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "gold_root": str(self.gold_root),
            "redundancy_lambda": self.redundancy_lambda,
            "task_count": self.task_count,
            "scored_task_count": self.scored_task_count,
            "total_score": round(self.total_score, 6),
            "mean_score": round(self.mean_score, 6),
            "mean_recall": round(self.mean_recall, 6),
            "per_task": [item.to_dict() for item in self.per_task],
        }


def score_run(
    *,
    run_dir: Path,
    gold_root: Path,
    redundancy_lambda: float = 0.5,
    numeric_tolerance: float = 1e-6,
    case_insensitive: bool = True,
    strip_whitespace: bool = True,
) -> RunScoreSummary:
    """Walk ``run_dir/<task_id>/prediction.csv`` and score against gold."""
    summary = RunScoreSummary(
        run_dir=run_dir,
        gold_root=gold_root,
        redundancy_lambda=redundancy_lambda,
    )
    if not run_dir.is_dir():
        return summary

    task_dirs = [
        path for path in sorted(run_dir.iterdir())
        if path.is_dir() and path.name.startswith("task_")
    ]
    for task_dir in task_dirs:
        task_id = task_dir.name
        pred_csv = task_dir / "prediction.csv"
        gold_csv = gold_root / task_id / "gold.csv"
        per_task = score_pair(
            task_id=task_id,
            pred_csv=pred_csv,
            gold_csv=gold_csv,
            redundancy_lambda=redundancy_lambda,
            numeric_tolerance=numeric_tolerance,
            case_insensitive=case_insensitive,
            strip_whitespace=strip_whitespace,
        )
        summary.per_task.append(per_task)
    return summary
