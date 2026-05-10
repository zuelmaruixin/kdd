"""Persistent cache for the Semantic Analyst's plan output.

The analyst is a ``temperature=0`` LLM call whose answer is fully
determined by (question, task_id, model_id, context files). Caching its
raw JSON plan to disk removes the single most expensive re-runnable step
in the pipeline:

    - Re-running the same benchmark task against the same model skips
      the analyst call entirely (~20-40s saved per hit).
    - Changing the question, the model, or adding / removing a context
      file (e.g. a synthesized CSV produced by record_text extraction)
      all invalidate the key automatically, so a stale plan can never
      slip back in.

What we cache is the **raw analyst plan**, BEFORE rule resolution and
the ``_apply_plan_overrides`` merging. Rule resolution is a pure disk
read (knowledge.md), so re-running it every time is practically free
and keeps the cached object small + insensitive to plan-override
post-processing. This preserves the invariant that
``_apply_rule_resolution`` always observes fresh disk state.

Storage layout matches the existing ``_ExtractionCache`` pattern so
operators already have mental model + tooling for it:
``artifacts/cache/semantic_plan/{sha256_hex[:32]}.json`` with payload
``{"plan": <raw plan dict>}``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class PlanCacheKeyInputs:
    """The structural inputs that deterministically shape an analyst plan.

    Kept as a thin dataclass so call sites can construct it once and
    reuse the same fingerprint for both ``get`` and ``put`` without
    risking a key drift.
    """

    model_id: str
    task_id: str
    question: str
    source_fingerprint: str


def build_source_fingerprint(source_capabilities: Iterable[Any]) -> str:
    """Compact string summary of context sources, used inside the cache key.

    The fingerprint captures only the columns and structural identity
    the analyst actually reasons over. It intentionally excludes volatile
    fields (row_count, sample values, scan_error) so benign data shifts
    don't invalidate the cache, but it DOES include every file path +
    kind + columns so schema changes invalidate on the spot.
    """
    parts: list[str] = []
    for cap in source_capabilities:
        path = getattr(cap, "path", "") or ""
        kind = getattr(cap, "kind", "") or ""
        columns = list(getattr(cap, "columns", None) or [])
        tables = list(getattr(cap, "tables", None) or [])
        record_fields = list(getattr(cap, "structured_record_fields", None) or [])
        parts.append(
            f"{path}|{kind}|cols={','.join(columns)}|tables={','.join(tables)}|recfields={','.join(record_fields)}"
        )
    return "\x1f".join(parts)


def compute_plan_cache_key(inputs: PlanCacheKeyInputs) -> str:
    payload = "\x00".join([
        inputs.model_id,
        inputs.task_id,
        inputs.question,
        inputs.source_fingerprint,
    ]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def default_cache_dir() -> Path | None:
    try:
        from data_agent_baseline.config import PROJECT_ROOT
    except Exception:  # noqa: BLE001
        return None
    return PROJECT_ROOT / "artifacts" / "cache" / "semantic_plan"


def cache_read(cache_dir: Path | None, key: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    plan = blob.get("plan")
    if not isinstance(plan, dict):
        return None
    # Defensive copy so the caller can freely mutate the returned plan
    # without corrupting a later cache hit in the same worker process.
    return json.loads(json.dumps(plan))


def cache_write(cache_dir: Path | None, key: str, *, plan: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(
            json.dumps({"plan": plan}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        # Cache is best-effort; never let a cache miss/write failure
        # bubble up and abort an otherwise-working analyst call.
        return


__all__ = [
    "PlanCacheKeyInputs",
    "build_source_fingerprint",
    "cache_read",
    "cache_write",
    "compute_plan_cache_key",
    "default_cache_dir",
]
