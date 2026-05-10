"""Deterministic task profiler for tool-first routing.

This layer is intentionally *not* a solver. It only profiles:

- source modalities and scale,
- coarse task type,
- coarse answer shape,
- finite candidate operators,
- primary / auxiliary tool preferences,
- uncertainty flags and budgets.

Field mapping, joins, formulas, and output columns belong to the
operator executor or a future LLM TaskCompiler layer.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask


RECORD_TEXT_KINDS = {"record_text"}
TABLE_KINDS = {
    "csv", "tsv", "json", "jsonl", "db", "sqlite", "sqlite3",
}
DB_KINDS = {"db", "sqlite", "sqlite3"}
DOC_KINDS = {"md", "txt", "docx", "pdf"}
IMAGE_KINDS = {"png", "jpg", "jpeg", "webp"}
KNOWN_KINDS = TABLE_KINDS | DOC_KINDS | RECORD_TEXT_KINDS | IMAGE_KINDS

_COUNT_RE = re.compile(r"\b(how many|number of|count of|count)\b", re.I)
_AGG_RE = re.compile(r"\b(avg|average|mean|median|sum|total|minimum|maximum|min|max)\b", re.I)
_RATIO_RE = re.compile(r"\b(ratio|percentage|percent|how many times|times as|as many as)\b", re.I)
_TOPK_RE = re.compile(r"\b(top\s+\d+|bottom\s+\d+|highest|lowest|largest|smallest|rank)\b", re.I)
_COMPARE_RE = re.compile(r"\b(compare|difference|greater than|less than|more than|fewer than|between)\b", re.I)
_JOIN_RE = re.compile(r"\b(whose|corresponding|associated|linked|related|match(?:ing)?)\b", re.I)
_GROUP_RE = re.compile(r"\b(for each|per|group by|by each)\b", re.I)
_FILTER_RE = re.compile(r"\b(where|whose|that|which|among|before|after|during|in \d{4})\b", re.I)
_BOOLEAN_RE = re.compile(r"^\s*(is|are|was|were|does|do|did|has|have|can)\b", re.I)

_MD_TABLE_LINE_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_MD_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)
# Generic "record id mention" regex: any paragraph that attaches a
# 2+ digit number to an id-like noun. Kept domain-agnostic.
_RECORD_ID_RE = re.compile(
    r"\b(?:id|number|record|file|ref(?:erence)?|case|entry)\b"
    r"[^.\n]{0,80}?\b\d{2,}\b",
    re.I,
)
# Generic "looks like a field name" regex: any word commonly used
# in key: value metadata blocks (date, value, level, description, url).
_STRUCTURED_FIELD_RE = re.compile(
    r"\b("
    r"name|id|ref|reference|date|year|month|day|"
    r"value|level|amount|count|score|rank|"
    r"type|status|category|class|kind|group|"
    r"description|note|comment|remark|"
    r"url|link|email|address|phone"
    r")\b",
    re.I,
)
_SEMANTIC_RULE_QUESTION_RE = re.compile(
    r"\b("
    r"abnormal|normal range|outside normal|within normal|"
    r"severe|severity|threshold|rule|defined as|definition|"
    r"according to (?:the )?(?:knowledge|rules)|"
    r"per (?:the )?(?:knowledge|rules)|using (?:the )?(?:knowledge|rules)"
    r")\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class SourceCapability:
    """What a single context file actually contains.

    This is the **single source of truth** for codegen prompts and the
    static checker: which file may be queried with which tool, and what
    table / column names actually exist there. No drift between prompt
    text and runtime checks.
    """

    path: str                     # relative to context_dir
    kind: str                     # csv | json | sqlite | record_text | doc | image | unknown
    bytes: int = 0
    role: str = "data"            # data | semantic_rule
    tool: str = ""                # canonical tool: pandas / sql / json / doc_parser / vision / unknown
    tables: list[dict[str, Any]] = field(default_factory=list)   # for sqlite: [{table, columns: [{name,type}], row_count}]
    columns: list[str] = field(default_factory=list)             # for csv/table-like sources
    row_count: int = 0
    json_top_keys: list[str] = field(default_factory=list)
    json_record_fields: list[str] = field(default_factory=list)
    json_record_count: int = 0
    sample: list[Any] = field(default_factory=list)              # small preview rows / records / snippets
    # record_text prose: each record mention block + best-effort field extraction
    structured_records: list[dict[str, Any]] = field(default_factory=list)
    structured_record_fields: list[str] = field(default_factory=list)
    structured_record_count: int = 0
    structured_record_splitter: str = ""                          # human-readable description, e.g. "paragraph"
    structured_id_pattern: str = ""                               # regex hint for the canonical id field
    # Per-column distinct-value samples (column name → up to 5 values, only
    # for low-cardinality columns).  This is what tells the LLM that
    # 'Symptoms' takes values like ['headache', 'pain'] vs that
    # 'Thrombosis' takes values like [0, 1, 2] — i.e. *which one* of
    # multiple plausibly-named columns the natural-language concept
    # actually maps to. The single biggest preventer of column-name
    # hallucinations.
    column_value_samples: dict[str, list[Any]] = field(default_factory=dict)
    column_dtypes: dict[str, str] = field(default_factory=dict)
    column_cardinalities: dict[str, int] = field(default_factory=dict)
    scan_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "bytes": self.bytes,
            "role": self.role,
            "tool": self.tool,
            "tables": list(self.tables),
            "columns": list(self.columns),
            "row_count": self.row_count,
            "json_top_keys": list(self.json_top_keys),
            "json_record_fields": list(self.json_record_fields),
            "json_record_count": self.json_record_count,
            "sample": list(self.sample),
            "structured_records": list(self.structured_records),
            "structured_record_fields": list(self.structured_record_fields),
            "structured_record_count": self.structured_record_count,
            "structured_record_splitter": self.structured_record_splitter,
            "structured_id_pattern": self.structured_id_pattern,
            "column_value_samples": {k: list(v) for k, v in self.column_value_samples.items()},
            "column_dtypes": dict(self.column_dtypes),
            "column_cardinalities": dict(self.column_cardinalities),
            "scan_error": self.scan_error,
        }


@dataclass(frozen=True, slots=True)
class CompiledTask:
    task_type: str
    answer_type: str
    data_sources: list[dict[str, Any]]
    modalities: list[str]
    operations: list[str]
    primary_tool: str
    auxiliary_tools: list[str]
    preferred_tools: list[str]
    source_capabilities: list[SourceCapability] = field(default_factory=list)
    # Inferred candidate join keys: ``(file_a, col_a, file_b, col_b, jaccard)``.
    # Populated after every source is scanned by overlap of distinct values
    # across same-dtype columns. Top-K only.
    foreign_key_candidates: list[dict[str, Any]] = field(default_factory=list)
    needs_reasoner: bool = False
    needs_vision: bool = False
    task_type_confidence: float = 1.0
    operation_confidence: float = 1.0
    source_confidence: float = 1.0
    ambiguity_flags: list[str] = field(default_factory=list)
    budget_level: str = "medium"
    max_llm_calls: int = 10
    max_tool_calls: int = 18
    context_bytes: int = 0
    file_count: int = 0
    execution_profile: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "answer_type": self.answer_type,
            "data_sources": list(self.data_sources),
            "modalities": list(self.modalities),
            "operations": list(self.operations),
            "primary_tool": self.primary_tool,
            "auxiliary_tools": list(self.auxiliary_tools),
            "preferred_tools": list(self.preferred_tools),
            "source_capabilities": [item.to_dict() for item in self.source_capabilities],
            "foreign_key_candidates": list(self.foreign_key_candidates),
            "needs_reasoner": self.needs_reasoner,
            "needs_vision": self.needs_vision,
            "task_type_confidence": self.task_type_confidence,
            "operation_confidence": self.operation_confidence,
            "source_confidence": self.source_confidence,
            "ambiguity_flags": list(self.ambiguity_flags),
            "budget_level": self.budget_level,
            "max_llm_calls": self.max_llm_calls,
            "max_tool_calls": self.max_tool_calls,
            "context_bytes": self.context_bytes,
            "file_count": self.file_count,
            "execution_profile": dict(self.execution_profile),
            "notes": list(self.notes),
        }


def _read_text_probe(path: Path, *, max_chars: int = 80_000) -> str:
    try:
        with path.open(errors="replace") as handle:
            return handle.read(max_chars)
    except OSError:
        return ""


def _looks_like_record_text(path: Path) -> bool:
    text = _read_text_probe(path)
    if not text:
        return False
    table_lines = _MD_TABLE_LINE_RE.findall(text)
    if len(table_lines) >= 2 and _MD_TABLE_SEPARATOR_RE.search(text):
        return True

    record_hits = len(_RECORD_ID_RE.findall(text))
    field_hits = len(_STRUCTURED_FIELD_RE.findall(text))
    date_hits = len(re.findall(
        r"\b\d{4}-\d{2}-\d{2}\b|"
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+of\s+[A-Z][a-z]+|"
        r"\b[A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?",
        text,
    ))
    # Count numeric literals, optionally with a short alphabetic unit suffix
    # such as "kg", "m/s", "U/L". Kept domain-agnostic — we only care about
    # how dense numeric content is, not what the unit actually means.
    numeric_hits = len(re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:[A-Za-z]{1,4}(?:/[A-Za-z]{1,4})?)?\b",
        text,
    ))
    if record_hits >= 5 and field_hits >= 8:
        return True
    if record_hits >= 3 and field_hits >= 5 and (date_hits + numeric_hits) >= 20:
        return True
    return False


def classify_context_kind(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"md", "txt"} and path.name.lower() != "knowledge.md":
        if _looks_like_record_text(path):
            return "record_text"
    if suffix == "sqlite":
        return "sqlite"
    return suffix or "unknown"


def _is_real_context_file(path: Path, *, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    if any(part.startswith(".") for part in rel.parts):
        return False
    # Guard against generated artifacts accidentally left inside a task's
    # context directory. They are answers, not inputs, and will poison the
    # schema-linking / codegen context if treated as source data.
    if path.name.lower() in {
        "answer.csv",
        "prediction.csv",
        "operator_codegen_answer.csv",
        "operator_local_repair_answer.csv",
        "operator_retry_answer.csv",
        "repaired_answer.csv",
    }:
        return False
    return True


def _role(path: Path, kind: str) -> str:
    if path.name.lower() == "knowledge.md":
        return "semantic_rule"
    return "data"


def _modalities_for_kinds(kinds: set[str]) -> list[str]:
    modalities: list[str] = []
    if kinds & TABLE_KINDS:
        modalities.append("table")
    if kinds & (DOC_KINDS | RECORD_TEXT_KINDS):
        modalities.append("document")
    if kinds & IMAGE_KINDS:
        modalities.append("image")
    if kinds - KNOWN_KINDS:
        modalities.append("unknown")
    return modalities


def _answer_shape(question: str) -> str:
    q = question.strip()
    q_lower = q.lower()
    if _BOOLEAN_RE.search(q):
        return "boolean"
    if _COUNT_RE.search(q_lower) or _AGG_RE.search(q_lower) or _RATIO_RE.search(q_lower):
        return "scalar"
    return "table"


def _operations(question: str, modalities: list[str], *, uses_semantic_rule: bool) -> tuple[list[str], float, list[str]]:
    q = question.lower()
    ops: set[str] = set()
    weak_hits: list[str] = []

    if "document" in modalities or uses_semantic_rule:
        ops.update({"retrieve", "extract"})
    if _FILTER_RE.search(q):
        ops.add("filter")
    if _JOIN_RE.search(q) and "table" in modalities:
        ops.add("join")
    if _GROUP_RE.search(q):
        ops.add("groupby")
    if _COUNT_RE.search(q) or _AGG_RE.search(q):
        ops.add("aggregate")
    if _TOPK_RE.search(q):
        ops.update({"sort", "topk"})
    if _COMPARE_RE.search(q) or _RATIO_RE.search(q):
        ops.update({"compare", "compute"})
    if uses_semantic_rule:
        ops.add("compute")

    for word in ("with", "by", "same", "times", "number", "after", "before"):
        if re.search(rf"\b{re.escape(word)}\b", q):
            weak_hits.append(word)

    if not ops:
        ops.add("extract" if "document" in modalities else "filter")
        confidence = 0.55
    elif weak_hits and len(ops) <= 1:
        confidence = 0.7
    else:
        confidence = 0.9

    ordered = [
        op for op in (
            "retrieve", "extract", "filter", "join", "groupby",
            "aggregate", "sort", "topk", "compare", "compute", "format",
        )
        if op in ops
    ]
    return ordered, confidence, weak_hits


def _task_type(
    *,
    modalities: list[str],
    has_tables: bool,
    has_non_doc_tables: bool,
    has_plain_docs: bool,
    has_record_text: bool,
    uses_semantic_rule: bool,
    has_semantic_doc: bool,
) -> tuple[str, float]:
    modality_set = set(modalities)
    if has_record_text and uses_semantic_rule:
        return "record_text_with_semantic_rule", 0.86
    if has_tables and uses_semantic_rule:
        return "table_with_semantic_rule", 0.86
    if has_record_text and has_non_doc_tables:
        return "mixed_context", 0.82
    if has_tables and has_plain_docs:
        return "mixed_context", 0.8
    if "unknown" in modality_set and len(modality_set) == 1:
        return "pure_reasoning", 0.45
    if modality_set == {"document"}:
        return "document_qa", 0.9
    if modality_set == {"table"}:
        return "table_computation", 0.9
    if modality_set == {"image"}:
        return "image_understanding", 0.85
    if "image" in modality_set and ("table" in modality_set or "document" in modality_set):
        return "mixed_context", 0.72
    if has_semantic_doc:
        return "document_qa", 0.62
    return "pure_reasoning", 0.55


def _tools(task_type: str, modalities: list[str], kinds: set[str]) -> tuple[str, list[str]]:
    if task_type == "record_text_with_semantic_rule":
        primary = "document_extractor"
        aux = ["pandas", "document_parser", "rag"]
    elif task_type == "table_with_semantic_rule":
        primary = "pandas"
        aux = ["document_parser", "rag"]
        if kinds & DB_KINDS:
            aux.insert(0, "sql")
    elif task_type == "table_computation":
        primary = "sql" if kinds & DB_KINDS else "pandas"
        aux = ["pandas" if primary == "sql" else "sql"]
    elif task_type == "document_qa":
        primary = "rag"
        aux = ["document_parser"]
    elif task_type == "mixed_context":
        primary = "pandas"
        aux = ["sql" if kinds & DB_KINDS else "document_parser", "rag"]
    elif task_type == "image_understanding":
        primary = "vision"
        aux = []
    else:
        primary = "react"
        aux = []
    deduped = []
    for tool in aux:
        if tool != primary and tool not in deduped:
            deduped.append(tool)
    return primary, deduped


def _budget(*, difficulty: str, task_type: str, flags: list[str]) -> tuple[str, int, int]:
    base_by_type = {
        "table_computation": (8, 18),
        "record_text_with_semantic_rule": (15, 36),
        "table_with_semantic_rule": (15, 34),
        "document_qa": (14, 28),
        "mixed_context": (14, 30),
        "image_understanding": (12, 22),
        "pure_reasoning": (8, 12),
    }
    llm, tools = base_by_type.get(task_type, (10, 18))
    diff_add = {"easy": 0, "medium": 1, "hard": 2, "extreme": 4}.get(
        (difficulty or "").lower(),
        1,
    )
    llm += diff_add
    tools += diff_add * 2

    if "large_document_context" in flags:
        llm += 6
        tools += 8
    if "large_table_context" in flags:
        llm += 1
        tools += 10
    if "many_files" in flags:
        llm += 1
        tools += 4
    if "multiple_modalities" in flags:
        llm += 1
        tools += 4
    if "semantic_rule_context" in flags:
        llm += 2
        tools += 4
    if "low_confidence" in flags or "ambiguous_schema" in flags:
        llm += 1
        tools += 2

    llm = min(llm, 30)
    tools = min(tools, 60)
    if llm <= 9:
        level = "small"
    elif llm <= 14:
        level = "medium"
    elif llm <= 22:
        level = "large"
    else:
        level = "xlarge"
    return level, llm, tools


# ---------------------------------------------------------------------------
# Source-capability scanners (deterministic, no LLM)
# ---------------------------------------------------------------------------


_KIND_TO_TOOL = {
    "csv": "pandas",
    "tsv": "pandas",
    "json": "json",
    "jsonl": "json",
    "db": "sql",
    "sqlite": "sql",
    "sqlite3": "sql",
    "record_text": "doc_parser",
    "md": "doc_parser",
    "txt": "doc_parser",
    "docx": "doc_parser",
    "pdf": "doc_parser",
    "png": "vision",
    "jpg": "vision",
    "jpeg": "vision",
    "webp": "vision",
}


def _scan_csv(path: Path) -> dict[str, Any]:
    """Schema-grounding scan: header + per-column distinct samples + dtypes.

    The per-column samples are what lets a downstream LLM verify that
    a natural-language concept like 'thrombosis' actually maps to a
    column whose values look like the concept (numeric severity codes,
    not free-text descriptions, etc).
    """
    with path.open(newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {"columns": [], "row_count": 0, "sample_rows": []}
        sample_rows: list[list[str]] = []
        # Track distinct values per column up to a small cap so
        # cardinality estimates stay cheap on huge files.
        per_col_values: list[dict[str, int]] = [{} for _ in header]
        per_col_dtype_votes: list[dict[str, int]] = [{} for _ in header]
        count = 0
        max_distinct_tracked = 60
        for row in reader:
            count += 1
            if len(sample_rows) < 3:
                sample_rows.append(row)
            for col_idx in range(len(header)):
                if col_idx >= len(row):
                    continue
                cell = row[col_idx]
                bucket = per_col_values[col_idx]
                if len(bucket) < max_distinct_tracked or cell in bucket:
                    bucket[cell] = bucket.get(cell, 0) + 1
                # Cheap dtype voting.
                votes = per_col_dtype_votes[col_idx]
                votes[_infer_cell_dtype(cell)] = votes.get(_infer_cell_dtype(cell), 0) + 1
    column_value_samples: dict[str, list[Any]] = {}
    column_cardinalities: dict[str, int] = {}
    column_dtypes: dict[str, str] = {}
    for col_idx, name in enumerate(header):
        bucket = per_col_values[col_idx]
        column_cardinalities[name] = len(bucket)
        # Top values by frequency.
        if bucket:
            top = sorted(bucket.items(), key=lambda item: -item[1])[:5]
            column_value_samples[name] = [value for value, _ in top]
        votes = per_col_dtype_votes[col_idx]
        if votes:
            column_dtypes[name] = max(votes.items(), key=lambda item: item[1])[0]
    return {
        "columns": list(header),
        "row_count": count,
        "sample_rows": sample_rows,
        "column_value_samples": column_value_samples,
        "column_cardinalities": column_cardinalities,
        "column_dtypes": column_dtypes,
    }


def _infer_cell_dtype(cell: str) -> str:
    text = (cell or "").strip()
    if not text:
        return "empty"
    # Boolean-ish.
    if text.lower() in {"true", "false", "t", "f", "yes", "no"}:
        return "bool"
    # Integer.
    try:
        int(text)
        return "int"
    except ValueError:
        pass
    # Float.
    try:
        float(text)
        return "float"
    except ValueError:
        pass
    # ISO-ish date.
    if re.match(r"^\d{4}-\d{2}-\d{2}", text) or re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", text):
        return "date"
    return "str"


def _scan_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)

    def _value_samples_for_records(records: list[Any]) -> tuple[dict[str, list[Any]], dict[str, int]]:
        """Top-5 distinct values + cardinality per field, only for the
        fields that look like flat scalars."""
        per_field: dict[str, dict[Any, int]] = {}
        for item in records:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if isinstance(v, (dict, list)):
                    continue
                bucket = per_field.setdefault(str(k), {})
                if len(bucket) < 60 or v in bucket:
                    bucket[v] = bucket.get(v, 0) + 1
        samples: dict[str, list[Any]] = {}
        cards: dict[str, int] = {}
        for key, bucket in per_field.items():
            cards[key] = len(bucket)
            top = sorted(bucket.items(), key=lambda kv: -kv[1])[:5]
            samples[key] = [v for v, _ in top]
        return samples, cards

    if isinstance(payload, dict):
        result: dict[str, Any] = {"top_level_type": "object", "top_level_keys": list(payload)[:30]}
        for key, value in payload.items():
            if isinstance(value, list):
                sample = value[:2]
                fields = sorted({k for item in sample if isinstance(item, dict) for k in item})
                samples, cards = _value_samples_for_records(value)
                result[f"{key}_count"] = len(value)
                result[f"{key}_sample_fields"] = fields
                result[f"{key}_sample"] = sample
                result[f"{key}_value_samples"] = samples
                result[f"{key}_cardinalities"] = cards
                break
        return result
    if isinstance(payload, list):
        sample = payload[:2]
        fields = sorted({k for item in sample if isinstance(item, dict) for k in item})
        samples, cards = _value_samples_for_records(payload)
        return {
            "top_level_type": "list",
            "count": len(payload),
            "sample_fields": fields,
            "sample": sample,
            "value_samples": samples,
            "cardinalities": cards,
        }
    return {"top_level_type": type(payload).__name__, "sample": payload}


def _scan_sqlite(path: Path) -> dict[str, Any]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables: list[dict[str, Any]] = []
        for (table_name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ):
            cols = [
                {"name": row[1], "type": row[2]}
                for row in conn.execute(f'PRAGMA table_info("{table_name}")')
            ]
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            sample = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 2').fetchall()
            # Per-column distinct-value samples, capped at 5 per column.
            column_value_samples: dict[str, list[Any]] = {}
            column_cardinalities: dict[str, int] = {}
            for col in cols:
                col_name = str(col.get("name") or "")
                if not col_name:
                    continue
                try:
                    distinct_rows = conn.execute(
                        f'SELECT "{col_name}", COUNT(*) FROM "{table_name}" '
                        f'GROUP BY "{col_name}" ORDER BY COUNT(*) DESC LIMIT 6'
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                cardinality = conn.execute(
                    f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table_name}"'
                ).fetchone()[0]
                column_cardinalities[col_name] = int(cardinality or 0)
                if cardinality and cardinality <= 60:
                    column_value_samples[col_name] = [r[0] for r in distinct_rows[:5]]
            tables.append({
                "table": table_name,
                "columns": cols,
                "row_count": row_count,
                "sample_rows": [list(row) for row in sample],
                "column_value_samples": column_value_samples,
                "column_cardinalities": column_cardinalities,
            })
        return {"tables": tables}
    finally:
        conn.close()


# --- Generic structured-prose scanner --------------------------------------
#
# We intentionally do NOT ship a hard-coded field list here. Domain-specific
# field extraction (e.g. "GPT U/L", "creatinine mg/dL", or sports metrics) is
# both brittle and a sign of over-fitting the profiler to a handful of
# benchmark tasks. The profiler's job is limited to:
#
#   1. splitting the document into per-record paragraphs, and
#   2. handing the first few snippets to downstream code so the actual field
#      extraction can be done by the LLM-based StructuredDocExecutor.
#
# Generic field-name heuristics (" key: value " style) are used only to
# populate the `record_fields` hint so the prompt shows *what kinds of
# keys* appear, without dictating specific ones.


_KEY_VALUE_LINE_RE = re.compile(
    r"^\s*(?P<k>[A-Za-z][A-Za-z0-9 _/\-]{1,40}?)\s*[:=]\s*(?P<v>\S.*?)\s*$",
)


def _split_records(text: str) -> list[str]:
    """Split a structured-prose doc into per-record paragraphs.

    Heuristic: cut on blank lines (the canonical paragraph splitter) and
    drop blocks that don't look like self-contained records — no
    multi-digit id mention and no key:value pair means it's probably
    abstract or narrative methodology, not a record.
    """
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    out: list[str] = []
    for block in blocks:
        has_id = bool(re.search(r"\b\d{2,}\b", block))
        has_kv = any(_KEY_VALUE_LINE_RE.match(line) for line in block.splitlines())
        if has_id or has_kv:
            out.append(block)
    return out


def _best_effort_record_fields(blocks: list[str], *, max_fields: int = 40) -> list[str]:
    """Collect a generic key list from "key: value" lines across blocks.

    Used only as a hint in the prompt: "records appear to use these keys".
    No values are extracted here; the actual values come from the LLM
    extractor downstream.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for block in blocks:
        for line in block.splitlines():
            match = _KEY_VALUE_LINE_RE.match(line)
            if match is None:
                continue
            key = match.group("k").strip().lower()
            key = re.sub(r"\s+", "_", key)
            if key and key not in seen_set:
                seen_set.add(key)
                seen.append(key)
            if len(seen) >= max_fields:
                return seen
    return seen


def _scan_structured_doc(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    blocks = _split_records(text)

    sample_snippets: list[str] = []
    for block in blocks[:3]:
        snippet = block.replace("\n", " ").strip()
        sample_snippets.append(snippet[:600])

    record_fields = _best_effort_record_fields(blocks)

    return {
        "chars": len(text),
        "record_count": len(blocks),
        "record_fields": record_fields,
        "sample_records": [],           # deterministic field extraction no longer runs here
        "sample_snippets": sample_snippets,
        "all_records": [],
    }


def _build_source_capability(
    *,
    rel: str,
    kind: str,
    role: str,
    size: int,
    abs_path: Path,
) -> SourceCapability:
    tool = _KIND_TO_TOOL.get(kind, "unknown")
    cap = SourceCapability(path=rel, kind=kind, bytes=size, role=role, tool=tool)
    try:
        if kind in {"csv", "tsv"}:
            scan = _scan_csv(abs_path)
            return SourceCapability(
                path=rel, kind=kind, bytes=size, role=role, tool=tool,
                columns=list(scan.get("columns") or []),
                row_count=int(scan.get("row_count") or 0),
                sample=list(scan.get("sample_rows") or []),
                column_value_samples=dict(scan.get("column_value_samples") or {}),
                column_cardinalities=dict(scan.get("column_cardinalities") or {}),
                column_dtypes=dict(scan.get("column_dtypes") or {}),
            )
        if kind in {"json", "jsonl"}:
            scan = _scan_json(abs_path)
            top_keys = list(scan.get("top_level_keys") or [])
            record_fields: list[str] = []
            record_count = 0
            sample_records: list[Any] = []
            value_samples: dict[str, list[Any]] = {}
            cardinalities: dict[str, int] = {}
            # Most DABench JSON has shape {records: [...]}; fish that out.
            for key in top_keys:
                if scan.get(f"{key}_count") is not None:
                    record_fields = list(scan.get(f"{key}_sample_fields") or [])
                    record_count = int(scan.get(f"{key}_count") or 0)
                    sample_records = list(scan.get(f"{key}_sample") or [])
                    value_samples = dict(scan.get(f"{key}_value_samples") or {})
                    cardinalities = dict(scan.get(f"{key}_cardinalities") or {})
                    break
            if scan.get("top_level_type") == "list":
                record_fields = list(scan.get("sample_fields") or [])
                record_count = int(scan.get("count") or 0)
                sample_records = list(scan.get("sample") or [])
                value_samples = dict(scan.get("value_samples") or {})
                cardinalities = dict(scan.get("cardinalities") or {})
            return SourceCapability(
                path=rel, kind=kind, bytes=size, role=role, tool=tool,
                json_top_keys=top_keys,
                json_record_fields=record_fields,
                json_record_count=record_count,
                sample=sample_records,
                column_value_samples=value_samples,
                column_cardinalities=cardinalities,
            )
        if kind in {"db", "sqlite", "sqlite3"}:
            scan = _scan_sqlite(abs_path)
            tables = list(scan.get("tables") or [])
            # Lift per-table per-column samples to the capability level
            # under "table.column" keys so schema_grounding / prompt can
            # use a single flat namespace.
            column_value_samples: dict[str, list[Any]] = {}
            column_cardinalities: dict[str, int] = {}
            column_dtypes: dict[str, str] = {}
            for table in tables:
                t_name = str(table.get("table") or "")
                samples = table.get("column_value_samples") or {}
                for col_name, values in samples.items():
                    column_value_samples[f"{t_name}.{col_name}"] = list(values)
                cards = table.get("column_cardinalities") or {}
                for col_name, card in cards.items():
                    column_cardinalities[f"{t_name}.{col_name}"] = int(card)
                for col in table.get("columns") or []:
                    if isinstance(col, dict) and col.get("name"):
                        column_dtypes[f"{t_name}.{col.get('name')}"] = str(col.get("type") or "")
            return SourceCapability(
                path=rel, kind=kind, bytes=size, role=role, tool=tool,
                tables=tables,
                column_value_samples=column_value_samples,
                column_cardinalities=column_cardinalities,
                column_dtypes=column_dtypes,
            )
        if kind == "record_text":
            scan = _scan_structured_doc(abs_path)
            return SourceCapability(
                path=rel, kind=kind, bytes=size, role=role, tool=tool,
                sample=list(scan.get("sample_snippets") or []),
                row_count=int(scan.get("record_count") or 0),
                structured_records=list(scan.get("sample_records") or []),
                structured_record_fields=list(scan.get("record_fields") or []),
                structured_record_count=int(scan.get("record_count") or 0),
                structured_record_splitter="paragraph (\\n\\n) + id-mention or key:value filter",
                structured_id_pattern=r"\b\d{2,}\b",
            )
    except Exception as exc:  # noqa: BLE001
        return SourceCapability(
            path=rel, kind=kind, bytes=size, role=role, tool=tool,
            scan_error=str(exc),
        )
    return cap


def compile_task(task: PublicTask) -> CompiledTask:
    files = [
        path
        for path in sorted(task.context_dir.rglob("*"))
        if path.is_file() and _is_real_context_file(path, root=task.context_dir)
    ]
    data_sources: list[dict[str, Any]] = []
    source_capabilities: list[SourceCapability] = []
    context_bytes = 0
    for path in files:
        kind = classify_context_kind(path)
        size = path.stat().st_size
        context_bytes += size
        rel = str(path.relative_to(task.context_dir))
        role = _role(path, kind)
        data_sources.append({
            "path": rel,
            "kind": kind,
            "bytes": size,
            "role": role,
        })
        source_capabilities.append(_build_source_capability(
            rel=rel, kind=kind, role=role, size=size, abs_path=path,
        ))

    all_kinds = {item["kind"] for item in data_sources}
    data_kinds = {item["kind"] for item in data_sources if item.get("role") == "data"}
    rule_kinds = {item["kind"] for item in data_sources if item.get("role") == "semantic_rule"}
    has_rule_doc = bool(rule_kinds & DOC_KINDS)
    uses_semantic_rule = bool(has_rule_doc and _SEMANTIC_RULE_QUESTION_RE.search(task.question))

    has_tables = bool(data_kinds & TABLE_KINDS)
    has_non_doc_tables = bool(data_kinds & TABLE_KINDS)
    has_record_text = bool(data_kinds & RECORD_TEXT_KINDS)
    has_plain_docs = bool(data_kinds & DOC_KINDS)
    has_semantic_doc = bool(rule_kinds & DOC_KINDS)

    kind_basis = set(data_kinds) or set(rule_kinds)
    modalities = _modalities_for_kinds(kind_basis)
    if uses_semantic_rule and "document" not in modalities:
        modalities.append("document")

    task_type, task_conf = _task_type(
        modalities=modalities,
        has_tables=has_tables,
        has_non_doc_tables=has_non_doc_tables,
        has_plain_docs=has_plain_docs,
        has_record_text=has_record_text,
        uses_semantic_rule=uses_semantic_rule,
        has_semantic_doc=has_semantic_doc,
    )
    operations, op_conf, weak_hits = _operations(
        task.question,
        modalities,
        uses_semantic_rule=uses_semantic_rule,
    )
    answer_type = _answer_shape(task.question)
    primary_tool, auxiliary_tools = _tools(task_type, modalities, all_kinds)
    preferred_tools = [primary_tool, *auxiliary_tools]

    flags: list[str] = []
    notes: list[str] = []
    if not files:
        flags.append("no_context_files")
    if len(modalities) >= 2:
        flags.append("multiple_modalities")
    if "image" in modalities:
        flags.append("needs_vision")
    if all_kinds - KNOWN_KINDS:
        flags.append("unsupported_file_type")
    if len(files) >= 8:
        flags.append("many_files")
    if has_record_text:
        flags.append("record_text_context")
        flags.append("record_extraction_required")
    if uses_semantic_rule:
        flags.append("semantic_rule_context")
    elif has_rule_doc and data_kinds:
        notes.append("semantic_rule_present")

    doc_bytes = sum(
        int(item["bytes"])
        for item in data_sources
        if item["kind"] in (DOC_KINDS | RECORD_TEXT_KINDS)
    )
    table_bytes = sum(int(item["bytes"]) for item in data_sources if item["kind"] in TABLE_KINDS)
    if doc_bytes >= 200_000:
        flags.append("large_document_context")
        flags.append("large_context")
    if table_bytes >= 1_000_000:
        flags.append("large_table_context")
        flags.append("large_context")
    if weak_hits and op_conf < 0.8:
        flags.append("weak_keyword_match")
        notes.append("weak_keywords:" + ",".join(sorted(set(weak_hits))))
    if task_conf < 0.7 or op_conf < 0.65:
        flags.append("low_confidence")
    if task_type in {"mixed_context", "table_with_semantic_rule", "record_text_with_semantic_rule"}:
        flags.append("ambiguous_schema")

    source_conf = 1.0
    if "unsupported_file_type" in flags:
        source_conf -= 0.25
    if "many_files" in flags:
        source_conf -= 0.1
    if "large_document_context" in flags or "large_table_context" in flags:
        source_conf -= 0.1
    if "record_text_context" in flags:
        source_conf -= 0.05
    source_conf = max(0.4, round(source_conf, 2))

    needs_reasoner = bool({
        "unsupported_file_type",
        "needs_vision",
    } & set(flags)) or (
        task_type == "document_qa"
        and "low_confidence" in flags
    )

    budget_level, max_llm_calls, max_tool_calls = _budget(
        difficulty=task.difficulty,
        task_type=task_type,
        flags=flags,
    )

    foreign_key_candidates = _infer_foreign_key_candidates(source_capabilities)
    execution_profile = _build_execution_profile(
        context_bytes=context_bytes,
        file_count=len(files),
        data_kinds=data_kinds,
        modalities=modalities,
        operations=operations,
        task_type=task_type,
        flags=flags,
        uses_semantic_rule=uses_semantic_rule,
        has_rule_doc=has_rule_doc,
        has_record_text=has_record_text,
        has_plain_docs=has_plain_docs,
        doc_bytes=doc_bytes,
        table_bytes=table_bytes,
        source_capabilities=source_capabilities,
    )

    return CompiledTask(
        task_type=task_type,
        answer_type=answer_type,
        data_sources=data_sources,
        modalities=modalities,
        operations=operations,
        primary_tool=primary_tool,
        auxiliary_tools=auxiliary_tools,
        preferred_tools=preferred_tools,
        source_capabilities=source_capabilities,
        foreign_key_candidates=foreign_key_candidates,
        needs_reasoner=needs_reasoner,
        needs_vision="needs_vision" in flags,
        task_type_confidence=round(task_conf, 2),
        operation_confidence=round(op_conf, 2),
        source_confidence=source_conf,
        ambiguity_flags=flags,
        budget_level=budget_level,
        max_llm_calls=max_llm_calls,
        max_tool_calls=max_tool_calls,
        context_bytes=context_bytes,
        file_count=len(files),
        execution_profile=execution_profile,
        notes=notes,
    )


def _build_execution_profile(
    *,
    context_bytes: int,
    file_count: int,
    data_kinds: set[str],
    modalities: list[str],
    operations: list[str],
    task_type: str,
    flags: list[str],
    uses_semantic_rule: bool,
    has_rule_doc: bool,
    has_record_text: bool,
    has_plain_docs: bool,
    doc_bytes: int,
    table_bytes: int,
    source_capabilities: list[SourceCapability],
) -> dict[str, Any]:
    """Finer execution recommendation used by the router.

    Difficulty labels intentionally do not appear here. They remain budget
    hints; this profile answers "what kind of executor can verify this?".
    """
    flag_set = set(flags)
    table_source_count = sum(1 for cap in source_capabilities if cap.role == "data" and cap.kind in TABLE_KINDS)
    doc_source_count = sum(1 for cap in source_capabilities if cap.role == "data" and cap.kind in (DOC_KINDS | RECORD_TEXT_KINDS))

    if context_bytes < 120_000 and file_count <= 4:
        context_size = "small"
    elif context_bytes < 1_200_000 and file_count <= 10:
        context_size = "medium"
    else:
        context_size = "large"

    if "needs_vision" in flag_set:
        source_shape = "image"
    elif has_record_text or "large_document_context" in flag_set or doc_bytes >= 200_000:
        source_shape = "long_docs"
    elif has_plain_docs and table_source_count:
        source_shape = "mixed_docs"
    elif table_source_count <= 1 and data_kinds & TABLE_KINDS:
        source_shape = "single_table"
    elif table_source_count > 1:
        source_shape = "multi_table"
    else:
        source_shape = "mixed_docs" if doc_source_count else "single_table"

    if "unsupported_file_type" in flag_set or "needs_vision" in flag_set:
        verifiability = "weak"
    elif data_kinds & TABLE_KINDS:
        verifiability = "programmatic"
    elif has_plain_docs or has_record_text:
        verifiability = "evidence_based"
    else:
        verifiability = "weak"

    op_set = set(operations)
    if {"aggregate", "compute"} & op_set and any(op in op_set for op in {"aggregate", "compare"}):
        operation_complexity = "aggregation"
    elif task_type in {"mixed_context", "record_text_with_semantic_rule"} or "multi_hop" in flag_set:
        operation_complexity = "multi_hop"
    elif table_source_count > 1 or "extract" in op_set:
        operation_complexity = "filter_join"
    elif "retrieve" in op_set or "filter" in op_set:
        operation_complexity = "single_filter"
    else:
        operation_complexity = "direct_lookup"

    semantic_rule_required = bool(uses_semantic_rule)
    if not has_rule_doc:
        semantic_rule_status = "none"
    elif semantic_rule_required:
        semantic_rule_status = "unresolved"
    else:
        semantic_rule_status = "candidate"
    semantic_rule_fields = _semantic_rule_fields(source_capabilities)

    low_complexity = operation_complexity in {"direct_lookup", "single_filter"}
    direct_llm_candidate_allowed = (
        context_size == "small"
        and source_shape in {"single_table", "mixed_docs"}
        and operation_complexity in {"direct_lookup", "single_filter"}
        and verifiability in {"programmatic", "evidence_based"}
        and "unsupported_file_type" not in flag_set
    )
    if direct_llm_candidate_allowed:
        direct_candidate_role = "submit_after_verify" if verifiability == "programmatic" else "proposal_only"
    else:
        direct_candidate_role = "disabled"

    if verifiability == "weak" or source_shape == "image":
        recommended_strategy = "multi_agent"
    elif source_shape == "long_docs":
        recommended_strategy = "rag_extract"
    elif semantic_rule_required:
        recommended_strategy = "operator_with_rule_resolution"
    elif direct_llm_candidate_allowed and low_complexity:
        recommended_strategy = "direct_candidate_verify"
    else:
        recommended_strategy = "operator"

    allowed_strategies = [recommended_strategy]
    if verifiability == "programmatic":
        allowed_strategies.append("operator")
        if semantic_rule_required:
            allowed_strategies.append("operator_with_rule_resolution")
    if source_shape in {"mixed_docs", "long_docs"}:
        allowed_strategies.append("rag_extract")
    if direct_llm_candidate_allowed:
        allowed_strategies.append("direct_candidate_verify")
    allowed_strategies.append("multi_agent")
    allowed_strategies = list(dict.fromkeys(allowed_strategies))

    blocked_strategies: list[str] = []
    if context_size == "small" and verifiability == "programmatic" and "unsupported_file_type" not in flag_set:
        blocked_strategies.append("direct_extreme_escalation")
    if semantic_rule_required and semantic_rule_status != "resolved":
        blocked_strategies.append("unverified_direct_submit")

    return {
        "context_size": context_size,
        "source_shape": source_shape,
        "verifiability": verifiability,
        "operation_complexity": operation_complexity,
        "semantic_rule_required": semantic_rule_required,
        "semantic_rule_status": semantic_rule_status,
        "semantic_rule_fields": semantic_rule_fields,
        "direct_llm_candidate_allowed": direct_llm_candidate_allowed,
        "direct_candidate_role": direct_candidate_role,
        "recommended_strategy": recommended_strategy,
        "allowed_strategies": allowed_strategies,
        "blocked_strategies": blocked_strategies,
    }


def _semantic_rule_fields(capabilities: list[SourceCapability]) -> list[str]:
    candidates: list[str] = []
    semantic_tokens = {
        "status", "type", "format", "diagnosis", "admission", "thrombosis",
        "severe", "abnormal", "normal", "age", "date", "eligible",
    }
    for cap in capabilities:
        for field in _columns_for_capability(cap):
            norm = field.lower()
            if any(tok in norm for tok in semantic_tokens):
                candidates.append(field)
    return list(dict.fromkeys(candidates))[:40]


def _columns_for_capability(cap: SourceCapability) -> list[str]:
    out: list[str] = []
    out.extend(cap.columns)
    out.extend(cap.json_record_fields)
    out.extend(cap.structured_record_fields)
    for table in cap.tables:
        for col in table.get("columns") or []:
            if isinstance(col, dict) and col.get("name"):
                out.append(str(col["name"]))
            elif isinstance(col, str):
                out.append(col)
    return list(dict.fromkeys(str(c) for c in out if str(c)))


def _infer_foreign_key_candidates(
    capabilities: list[SourceCapability],
    *,
    top_k: int = 8,
    min_jaccard: float = 0.15,
) -> list[dict[str, Any]]:
    """Suggest cross-source join keys via Jaccard overlap of value samples.

    Two columns become a candidate key when:
    - both have a non-empty distinct-value sample,
    - their value sets overlap by ≥ ``min_jaccard``,
    - they live in different files (cross-source) or different sqlite tables.

    The output is a small list of dicts; the codegen prompt renders the
    top-K so the model has explicit hints like
    ``examination.ID ↔ patient.ID  (jaccard=0.94, sample=[1,2,5,…])``.
    """
    candidates: list[tuple[float, dict[str, Any]]] = []

    flat: list[tuple[str, str, set[Any]]] = []  # (path, full_col_name, value_set)
    for cap in capabilities:
        for col_name, values in (cap.column_value_samples or {}).items():
            normalized = {_normalize_join_value(v) for v in values}
            normalized.discard(None)
            if not normalized:
                continue
            flat.append((cap.path, col_name, normalized))

    for i in range(len(flat)):
        path_a, col_a, set_a = flat[i]
        for j in range(i + 1, len(flat)):
            path_b, col_b, set_b = flat[j]
            # Skip same column on same file (sqlite tables-with-same-name handled by .table prefix)
            if path_a == path_b and col_a == col_b:
                continue
            inter = set_a & set_b
            if not inter:
                continue
            union = set_a | set_b
            jaccard = len(inter) / max(len(union), 1)
            if jaccard < min_jaccard:
                continue
            candidates.append((jaccard, {
                "left_path": path_a,
                "left_column": col_a,
                "right_path": path_b,
                "right_column": col_b,
                "jaccard": round(jaccard, 3),
                "shared_sample": sorted(list(inter), key=str)[:5],
            }))

    candidates.sort(key=lambda item: -item[0])
    return [item for _, item in candidates[:top_k]]


def _normalize_join_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # Coerce floats that are exactly integers so 14 == 14.0.
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    text = str(value).strip()
    if not text:
        return None
    # Try numeric coercion so "14" == 14.
    try:
        as_int = int(text)
        return as_int
    except ValueError:
        try:
            as_float = float(text)
            if as_float.is_integer():
                return int(as_float)
            return as_float
        except ValueError:
            return text.lower()
