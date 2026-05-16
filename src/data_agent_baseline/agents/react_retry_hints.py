"""Deterministic retry hints for the React loop.

When a tool call fails, the bare error message often isn't enough for the
model to fix the next step — especially for Python tracebacks or SQL
errors that name a missing column / file. This module classifies the
failure and produces a structured ``retry_hint`` that the React harness
inlines into the observation, so the model sees both the raw error AND a
concrete suggestion (call ``list_context`` first, switch to
``inspect_sqlite_schema``, etc.).

It also tracks the recent error signatures so that "same error twice"
gets an escalated "switch approach" nudge — repeated identical failures
are the strongest local signal that the current approach is wrong.

This is deterministic on purpose: no LLM call, no API budget. Matches
the project's "cheap deterministic guard first, model round only when
warranted" policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


# ── error pattern → suggestion ────────────────────────────────────────
# Each pattern is matched (case-insensitive) against the combined
# error+traceback+stderr text. First match wins.

_PYTHON_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"FileNotFoundError|No such file or directory", re.IGNORECASE),
        "missing_file",
        "The code referenced a file that does not exist. Call list_context "
        "(or read_csv on a known-good path) to confirm the exact filename "
        "before retrying, and use the path verbatim.",
    ),
    (
        re.compile(r"ModuleNotFoundError|No module named", re.IGNORECASE),
        "missing_module",
        "That import is unavailable. Only pandas, numpy, json, csv, "
        "sqlite3, re, math, statistics, datetime, pathlib and Python "
        "stdlib are reliably present. Rewrite without the optional "
        "dependency.",
    ),
    (
        re.compile(r"KeyError", re.IGNORECASE),
        "key_error",
        "A dict / DataFrame lookup hit a key that does not exist. Call "
        "read_csv (max_rows=1) or inspect_sqlite_schema to print the real "
        "column names, then use them exactly (case + spaces matter).",
    ),
    (
        re.compile(r"AttributeError", re.IGNORECASE),
        "attribute_error",
        "An attribute lookup failed. Print type(obj) and dir(obj) in a "
        "quick execute_python probe before retrying.",
    ),
    (
        re.compile(r"ValueError.*could not convert|invalid literal", re.IGNORECASE),
        "value_conversion",
        "Numeric conversion failed because of stray characters. Strip "
        "currency / percent / thousand separators with .str.replace before "
        "casting to float/int.",
    ),
    (
        re.compile(r"TypeError", re.IGNORECASE),
        "type_error",
        "Two incompatible types were combined. Use df.dtypes (pandas) or "
        "type(x) (Python) to confirm types, then cast explicitly with "
        "pd.to_numeric / pd.to_datetime / int / str.",
    ),
    (
        re.compile(r"SyntaxError|IndentationError|EOL while scanning", re.IGNORECASE),
        "syntax_error",
        "The code did not parse. Most likely cause: unterminated string, "
        "missing colon, or stray smart-quote. Rewrite the snippet cleanly.",
    ),
    (
        re.compile(r"timed out after \d+ seconds", re.IGNORECASE),
        "python_timeout",
        "The code took longer than 30s. Avoid reading entire files into "
        "memory at once. Use pandas chunksize, sqlite, or a narrower "
        "filter, then aggregate.",
    ),
    (
        re.compile(r"ParserError|EmptyDataError|tokenizing data", re.IGNORECASE),
        "csv_parse_error",
        "pandas failed to parse the CSV. Try read_csv(... sep=None, "
        "engine='python') or inspect the raw bytes with read_doc first "
        "to pick the right delimiter / encoding.",
    ),
    (
        re.compile(r"UnicodeDecodeError|codec can't decode", re.IGNORECASE),
        "encoding_error",
        "The file isn't UTF-8. Re-open with encoding='latin-1' or "
        "encoding='gbk' depending on the data origin.",
    ),
]

_SQL_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"no such table", re.IGNORECASE),
        "sql_unknown_table",
        "That table does not exist in the database. Call "
        "inspect_sqlite_schema first to list the real table names.",
    ),
    (
        re.compile(r"no such column", re.IGNORECASE),
        "sql_unknown_column",
        "That column does not exist. Call inspect_sqlite_schema to see "
        "the actual column names, then quote identifiers with double "
        "quotes if they contain spaces.",
    ),
    (
        re.compile(r"syntax error|near \".*\": syntax error", re.IGNORECASE),
        "sql_syntax_error",
        "SQL did not parse. Re-check quoting (single quotes for strings, "
        "double quotes for identifiers with spaces) and missing commas.",
    ),
    (
        re.compile(r"only read-only queries|forbidden|not allowed", re.IGNORECASE),
        "sql_write_blocked",
        "Only SELECT queries are allowed. Rewrite without INSERT / "
        "UPDATE / CREATE / ATTACH.",
    ),
]


_PATH_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"(?:no such|cannot find|does not exist|outside the task context)", re.IGNORECASE),
        "bad_context_path",
        "The path is not under this task's context_dir. Call list_context "
        "to see the exact relative paths available and use one of those.",
    ),
]


_ANSWER_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"answer\.columns must be", re.IGNORECASE),
        "answer_columns_invalid",
        "answer.columns must be a non-empty list of strings — strip any "
        "None / numeric headers and resubmit.",
    ),
    (
        re.compile(r"answer\.rows must be", re.IGNORECASE),
        "answer_rows_invalid",
        "answer.rows must be a list of lists; flatten any nested objects "
        "and ensure every row has the same length as columns.",
    ),
    (
        re.compile(r"row must match the number of columns", re.IGNORECASE),
        "answer_row_arity",
        "At least one row's length didn't match len(columns). Pad / trim "
        "every row to match before calling answer again.",
    ),
]


def _extract_error_text(action: str, observation: dict[str, Any]) -> str:
    """Pull the most informative free-text slice out of an observation."""
    content = observation.get("content")
    pieces: list[str] = []
    if isinstance(content, dict):
        for key in ("error", "traceback", "stderr"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value)
    elif isinstance(content, str):
        pieces.append(content)
    direct = observation.get("error")
    if isinstance(direct, str) and direct.strip():
        pieces.append(direct)
    return "\n".join(pieces).strip()


def _match(patterns: Iterable[tuple[re.Pattern[str], str, str]], text: str) -> tuple[str, str] | None:
    for pattern, code, hint in patterns:
        if pattern.search(text):
            return code, hint
    return None


@dataclass
class RetryHint:
    code: str
    suggestion: str
    repeated: bool = False
    occurrence_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "suggestion": self.suggestion,
            "repeated": self.repeated,
        }
        if self.occurrence_count > 1:
            out["occurrence_count"] = self.occurrence_count
        return out


def _signature(action: str, error_text: str) -> str:
    """Coarse fingerprint for "this is the same error as last time".

    We strip absolute paths, line numbers, and quoted identifiers so that
    e.g. "no such column: foo" and "no such column: bar" still collide,
    which is the signal we actually want.
    """
    normalized = error_text.strip()
    normalized = re.sub(r"\bline\s+\d+", "line N", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"/[^\s'\"]+", "<path>", normalized)
    normalized = re.sub(r"'[^']*'", "<q>", normalized)
    normalized = re.sub(r'"[^"]*"', "<q>", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return f"{action}::{normalized[:240]}"


@dataclass
class ErrorHistory:
    """Lightweight bookkeeping for repeated-error detection.

    One instance per React loop. Pushes a signature for every failed
    tool call and reports the count back so the harness can decide
    whether to escalate the hint.
    """

    seen: dict[str, int] = field(default_factory=dict)

    def record(self, action: str, observation: dict[str, Any]) -> tuple[str, int]:
        error_text = _extract_error_text(action, observation)
        signature = _signature(action, error_text)
        count = self.seen.get(signature, 0) + 1
        self.seen[signature] = count
        return signature, count


def build_retry_hint(
    *,
    action: str,
    action_input: dict[str, Any],
    observation: dict[str, Any],
    history: ErrorHistory,
) -> RetryHint | None:
    """Build a retry hint for a failed tool observation.

    Returns ``None`` when there's no error text to interpret. Mutates
    ``history`` so successive identical errors get flagged as repeats.
    """
    error_text = _extract_error_text(action, observation)
    if not error_text:
        return None

    signature, count = history.record(action, observation)

    matched: tuple[str, str] | None = None
    if action == "execute_python":
        matched = _match(_PYTHON_PATTERNS, error_text)
    elif action == "execute_context_sql":
        matched = _match(_SQL_PATTERNS, error_text)
    elif action in {"inspect_sqlite_schema", "read_csv", "read_doc", "read_json"}:
        matched = _match(_PATH_PATTERNS, error_text)
    elif action == "answer":
        matched = _match(_ANSWER_PATTERNS, error_text)

    if matched is None:
        # Fall through: a generic "read the error message carefully and
        # switch tools if needed" suggestion is still better than silence.
        code = f"{action}_failed"
        suggestion = (
            "Re-read the error message. If it mentions a column / table / "
            "file, verify it exists with inspect_sqlite_schema / read_csv "
            "/ list_context first, then retry with the corrected name."
        )
    else:
        code, suggestion = matched

    repeated = count >= 2
    if repeated:
        suggestion = (
            f"You hit the same error pattern {count} times in a row. "
            "Stop retrying this approach — switch tools or read the data "
            "first. " + suggestion
        )

    return RetryHint(
        code=code,
        suggestion=suggestion,
        repeated=repeated,
        occurrence_count=count,
    )


# ── verify-round instructions targeted at the guard's risk codes ──────


_VERIFY_HINTS: dict[str, str] = {
    "execution_failed_or_missing_answer": (
        "No answer table was produced. Build one with concrete rows."
    ),
    "empty_answer_rows": (
        "Your answer has columns but zero rows. Re-run the computation; "
        "if the filter is too strict it is probably wrong."
    ),
    "invalid_answer": (
        "Answer failed structural validation. Re-check columns and rows: "
        "every row must be a flat list with the same arity as columns."
    ),
    "answer_without_computation": (
        "You answered without ever running execute_python or "
        "execute_context_sql against the full data. Sample previews from "
        "read_csv are NOT safe to derive the final answer from. Run "
        "actual code against the file before re-answering."
    ),
    "knowledge_doc_not_read": (
        "There is a knowledge / rule / glossary doc in this task that you "
        "never read. Per the project contract, this doc overrides general "
        "domain knowledge. Call read_doc on it, apply its rules, then "
        "re-answer."
    ),
    "shape_mismatch_scalar_question": (
        "The question asks for a single scalar value but your draft has "
        "multiple rows. Aggregate / filter down to one row."
    ),
    "shape_mismatch_list_question": (
        "The question asks for a list but your draft is a single scalar. "
        "Produce one row per matching item."
    ),
}


def build_verify_instructions(guard_summary: dict[str, Any]) -> str:
    """Targeted verify-round instructions derived from guard risk codes.

    When the cheap deterministic guard flagged something, we don't just
    ask the model to "re-check" — we tell it WHICH specific risk we
    detected so the verification round has a real target.
    """
    risk_codes = guard_summary.get("risk_codes") or []
    if not risk_codes:
        return (
            "Re-check the draft against the question and the data: confirm "
            "the columns match exactly, verify numeric/string formats, and "
            "re-run any computation you are unsure about with "
            "execute_python or execute_context_sql. When you are "
            "satisfied, call `answer` again with the final table."
        )
    bullets = []
    for code in risk_codes:
        hint = _VERIFY_HINTS.get(str(code))
        if hint:
            bullets.append(f"- [{code}] {hint}")
        else:
            bullets.append(f"- [{code}] Re-check this aspect of the draft.")
    top_risk = guard_summary.get("top_risk") or {}
    top_msg = top_risk.get("message") if isinstance(top_risk, dict) else None
    header = (
        "The cheap deterministic guard flagged this draft as risky. "
        "Address EVERY item below before calling `answer` again; the "
        "next answer call will be final."
    )
    if top_msg:
        header += f"\nMost important: {top_msg}"
    return header + "\n" + "\n".join(bullets)
