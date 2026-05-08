"""Shared execution context for the operator pipeline.

Built once from CompiledTask, passed immutably through all stages.
Centralizes schema_diagnostics and schema_scan so they are never
recomputed per-call (previously called 4+ times per task).
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from data_agent_baseline.agents.task_compiler import CompiledTask, SourceCapability


# ---------------------------------------------------------------------------
# Schema helpers (moved from operator_executor; single authoritative copy)
# ---------------------------------------------------------------------------

def source_columns(cap: SourceCapability) -> list[str]:
    """Collect every column/field name from a SourceCapability."""
    cols: list[str] = []
    cols.extend(getattr(cap, "columns", []) or [])
    cols.extend(getattr(cap, "json_record_fields", []) or [])
    cols.extend(getattr(cap, "structured_record_fields", []) or [])
    for table in getattr(cap, "tables", []) or []:
        for col in table.get("columns") or []:
            if isinstance(col, dict) and col.get("name"):
                cols.append(str(col["name"]))
            elif isinstance(col, str):
                cols.append(col)
    return list(dict.fromkeys(str(c) for c in cols if str(c)))


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_schema_diagnostics(compiled_task: CompiledTask) -> dict[str, Any]:
    """Compact schema map for repair/analyst prompts — no file I/O.

    Reads only compiled_task.source_capabilities, which task_compiler
    already populated. Previously this was _schema_diagnostics() in
    operator_executor and was recalculated on every call.
    """
    sources: list[dict[str, Any]] = []
    caps = list(compiled_task.source_capabilities)

    for cap in caps:
        sources.append({
            "path": cap.path,
            "kind": cap.kind,
            "role": cap.role,
            "columns": source_columns(cap),
            "column_value_samples": {
                str(k): list(v)[:5]
                for k, v in list((cap.column_value_samples or {}).items())[:80]
            },
            "column_dtypes": dict(list((cap.column_dtypes or {}).items())[:80]),
            "column_cardinalities": dict(list((cap.column_cardinalities or {}).items())[:80]),
            "tables": [
                {
                    "table": t.get("table"),
                    "columns": [
                        c.get("name") if isinstance(c, dict) else c
                        for c in (t.get("columns") or [])
                    ],
                }
                for t in (cap.tables or [])
            ],
        })

    joins: list[dict[str, Any]] = []
    for i, left in enumerate(caps):
        left_cols = source_columns(left)
        if not left_cols:
            continue
        for right in caps[i + 1:]:
            right_cols = source_columns(right)
            if not right_cols:
                continue
            left_norm = {norm_name(c): c for c in left_cols}
            right_norm = {norm_name(c): c for c in right_cols}

            for n in sorted(set(left_norm) & set(right_norm)):
                joins.append({
                    "left_path": left.path,
                    "right_path": right.path,
                    "left_on": left_norm[n],
                    "right_on": right_norm[n],
                    "reason": "same_normalized_name",
                })

            already_joined = any(
                j["left_path"] == left.path and j["right_path"] == right.path
                for j in joins
            )
            if not already_joined:
                left_ids = [c for c in left_cols if norm_name(c).endswith("id")]
                right_ids = [c for c in right_cols if norm_name(c).endswith("id")]
                for lc in left_ids:
                    lb = norm_name(lc).removesuffix("id")
                    for rc in right_ids:
                        rb = norm_name(rc).removesuffix("id")
                        if lb and rb and (lb in rb or rb in lb):
                            joins.append({
                                "left_path": left.path,
                                "right_path": right.path,
                                "left_on": lc,
                                "right_on": rc,
                                "reason": "related_id_name",
                            })

            for lc in left_cols:
                ln = norm_name(lc)
                if not ln.startswith("linkto"):
                    continue
                target = ln.removeprefix("linkto")
                for rc in right_cols:
                    if norm_name(rc) == f"{target}id":
                        joins.append({
                            "left_path": left.path,
                            "right_path": right.path,
                            "left_on": lc,
                            "right_on": rc,
                            "reason": "link_to_foreign_key",
                        })

            for rc in right_cols:
                rn = norm_name(rc)
                if not rn.startswith("linkto"):
                    continue
                target = rn.removeprefix("linkto")
                for lc in left_cols:
                    if norm_name(lc) == f"{target}id":
                        joins.append({
                            "left_path": left.path,
                            "right_path": right.path,
                            "left_on": lc,
                            "right_on": rc,
                            "reason": "link_to_foreign_key",
                        })

    return {
        "sources": sources,
        "candidate_join_keys": joins[:40],
        "ambiguity_flags": list(compiled_task.ambiguity_flags),
    }


def build_schema_scan(compiled_task: CompiledTask) -> list[dict[str, Any]]:
    """Build schema_scan from already-compiled SourceCapabilities.

    Replaces the file-reading _schema_scan() + _scan_csv/json/sqlite/structured_doc
    quartet in operator_executor.  task_compiler already read the files once;
    there is no reason to read them again.
    """
    scanned: list[dict[str, Any]] = []
    for cap in compiled_task.source_capabilities:
        kind = cap.kind or ""
        if kind in {"csv", "tsv"}:
            detail: dict[str, Any] = {
                "columns": list(cap.columns or []),
                "row_count": cap.row_count or 0,
                "sample_rows": [list(r) if not isinstance(r, list) else r
                                for r in (cap.sample or [])[:3]],
            }
        elif kind in {"json", "jsonl"}:
            detail = {
                "top_level_type": "object" if cap.json_top_keys else "list",
                "top_level_keys": list(cap.json_top_keys or [])[:30],
                "count": cap.json_record_count or 0,
                "sample_fields": list(cap.json_record_fields or [])[:30],
                "sample": list(cap.sample or [])[:2],
            }
        elif kind in {"db", "sqlite", "sqlite3"}:
            detail = {"tables": list(cap.tables or [])}
        elif kind in {"record_text", "structured_table"}:
            detail = {
                "chars": cap.bytes or 0,
                "record_like_mentions": cap.structured_record_count or 0,
                "field_words": list(cap.structured_record_fields or []),
                "sample_snippets": [
                    r.get("text", "") if isinstance(r, dict) else str(r)
                    for r in (cap.structured_records or [])[:3]
                ],
                "parse_hint": "Read the full file and extract repeated records with regex/string parsing.",
            }
        else:
            continue  # skip unknown / image kinds — not schema-queryable

        scanned.append({
            "path": cap.path,
            "kind": kind,
            "bytes": cap.bytes,
            "detail": detail,
        })
    return scanned


# ---------------------------------------------------------------------------
# ExecutionContext
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    """Immutable context shared by all pipeline stages in one task execution.

    schema_diagnostics and schema_scan are computed exactly once here
    instead of being recalculated inside semantic analyst, judge, repair,
    and schema-retry separately.
    """

    compiled_task: CompiledTask
    schema_diagnostics: dict[str, Any] = field(init=False)
    schema_scan: list[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.schema_diagnostics = build_schema_diagnostics(self.compiled_task)
        self.schema_scan = build_schema_scan(self.compiled_task)
