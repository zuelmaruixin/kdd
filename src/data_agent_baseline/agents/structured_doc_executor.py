"""LLM-driven unstructured record-text extractor.

Some tasks ship long-form report-style documents where individual
records are buried in narrative prose. Regex-only extraction is too
brittle, and pure RAG cannot materialize a full per-record table. The
fix is a tightly-bounded LLM extraction pre-stage:

    unstructured_record_text
        ↓ split by paragraph + record-id mentions
    chunks (≈6KB each)
        ↓ LLM call per chunk with a strict JSON schema
    [record dicts]
        ↓ concatenate + dedupe by id-like column
    synthesized CSV in temp dir
        ↓
    normal codegen path can now `pd.read_csv` the synthesized file
    and just write the question's filter / count.

We persistently cache extraction results by ``(file_hash + schema_hash +
endpoint)`` so re-runs of the same task don't re-pay the LLM cost.

Not every ``record_text`` task benefits from table synthesis though.
When the question is reading-comprehension-style (summarize / infer /
why / compare narrative detail), forcing a fixed schema on top of prose
drops information. We classify the question up-front and skip extraction
for those cases; the operator executor then uses the raw text + RAG.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.agents.task_compiler import CompiledTask, SourceCapability
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.budget import get_budget_controller


# ---------------------------------------------------------------------------
# Question-type classifier (record_text query_type)
# ---------------------------------------------------------------------------
#
# The primary path is an LLM call (``temperature=0, max_tokens=10``) whose
# verdict is persisted to disk keyed on ``(question_hash, model_id)`` so
# repeat runs of the same task don't re-pay the classification cost.
# A deterministic keyword vote is kept as a safety net for offline / no-
# model runs and for LLM failures (network errors, bad JSON, etc.). Every
# path records ``source`` so trace.json can tell after the fact whether
# the verdict came from the LLM, the cache, or the fallback.
#
# The split is a direct response to a known design gap: the previous
# pipeline always synthesized a CSV, which is correct for count/filter-
# style questions but strictly loses information for reading-
# comprehension ones.


_AGGREGATE_KEYWORDS = (
    "how many", "count", "number of", "total", "sum",
    "average", "mean", "median", "ratio", "percentage", "percent",
    "list all", "which ... have", "whose", "filter",
    "greater than", "less than", "at least", "at most",
    "top ", "bottom ", "highest", "lowest", "largest", "smallest",
    "maximum", "minimum", "rank", "order by",
    "for each", "per ", "group by",
)

_READING_KEYWORDS = (
    "summarize", "summary", "describe", "explain", "why",
    "infer", "deduce", "conclude", "interpret",
    "what is mentioned", "what does the report say",
    "compare the narrative", "narrative", "overall",
    "opinion", "argument", "rationale", "reasoning behind",
)


_QUERY_TYPE_SYSTEM_PROMPT = (
    "You classify a user question about a long-form narrative report as "
    "one of exactly two labels:\n"
    "  aggregate  — the question asks for a count, filter, list, ranking, "
    "statistic, or tabular value over records (e.g. 'how many', "
    "'list all', 'which ... have', 'top N', 'average', 'sum', "
    "'per X group by Y').\n"
    "  read       — the question asks to summarize, describe, explain, "
    "infer, interpret, or reason about the narrative content "
    "(e.g. 'summarize', 'why', 'describe', 'compare the narrative', "
    "'overall argument').\n"
    "Reply with EXACTLY ONE lowercase word: either 'aggregate' or "
    "'read'. No punctuation, no explanation, no quotes."
)


@dataclass(slots=True)
class RecordTextQueryTypeVerdict:
    """Outcome of classifying a record_text question.

    Surfaced end-to-end into trace.json so auditors can distinguish an
    LLM-driven classification from a fallback after the fact.
    """

    verdict: str                       # "aggregate" | "read"
    source: str                        # llm | cache | fallback_keyword | fallback_error | fallback_no_model
    model_id: str | None = None
    cache_hit: bool = False
    llm_raw: str | None = None
    error: str | None = None
    question_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "source": self.source,
            "model_id": self.model_id,
            "cache_hit": self.cache_hit,
            "llm_raw": self.llm_raw,
            "error": self.error,
            "question_hash": self.question_hash,
        }


def _keyword_classify_record_text_query(question: str) -> str:
    """Safety-net keyword classifier. Must never raise."""
    q = (question or "").lower()
    agg_hits = sum(1 for kw in _AGGREGATE_KEYWORDS if kw in q)
    read_hits = sum(1 for kw in _READING_KEYWORDS if kw in q)
    if read_hits > agg_hits and read_hits >= 1:
        return "read"
    if agg_hits == 0 and read_hits == 0:
        # Truly ambiguous: default to aggregate so the CSV path can still
        # catch the common case.
        return "aggregate"
    return "aggregate"


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _question_hash(question: str) -> str:
    return hashlib.sha256(_normalize_question(question).encode("utf-8")).hexdigest()[:16]


def _model_identity(model: Any) -> str:
    api_base = getattr(model, "api_base", "scripted")
    model_name = getattr(model, "model", type(model).__name__)
    return f"{api_base}::{model_name}"


def _query_type_cache_key(question_hash: str, model_id: str) -> str:
    digest = hashlib.sha256(f"{model_id}\x00{question_hash}".encode("utf-8")).hexdigest()
    return digest[:32]


def _query_type_cache_path(cache_dir: Path, question_hash: str, model_id: str) -> Path:
    return cache_dir / f"{_query_type_cache_key(question_hash, model_id)}.json"


def _parse_query_type_response(raw: str) -> str | None:
    """Map a raw model response to 'aggregate' / 'read' or None if unclear."""
    if not raw:
        return None
    # Strip JSON fencing / quotes / whitespace. We asked for one word but
    # tolerant parsing avoids a fallback trip on trivial decoration.
    text = raw.strip().strip("`\"' \n\r\t.").lower()
    if not text:
        return None
    # Take first non-trivial token.
    match = re.search(r"[a-z]+", text)
    if match is None:
        return None
    token = match.group(0)
    if token.startswith("agg"):
        return "aggregate"
    if token.startswith("read"):
        return "read"
    return None


def classify_record_text_query_verdict(
    question: str,
    *,
    model: Any = None,
    cache_dir: Path | None = None,
) -> RecordTextQueryTypeVerdict:
    """Full classification entry point: LLM → cache → keyword fallback.

    Always returns a verdict, even if model is None or the LLM fails.
    Budget accounting happens inside ``model.complete()`` via the global
    ``BudgetController``; callers MUST NOT double-book. ``BudgetExceeded``
    is allowed to propagate so the runner can terminate the task, but any
    other exception falls through to the keyword safety net.
    """
    from data_agent_baseline.budget import BudgetExceeded  # local: avoid cycle

    q_hash = _question_hash(question)

    if model is None:
        return RecordTextQueryTypeVerdict(
            verdict=_keyword_classify_record_text_query(question),
            source="fallback_no_model",
            model_id=None,
            question_hash=q_hash,
        )

    model_id = _model_identity(model)

    # --- Cache lookup ---
    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = _query_type_cache_path(cache_dir, q_hash, model_id)
            if cache_path.exists():
                try:
                    payload = json.loads(cache_path.read_text())
                    cached_verdict = payload.get("verdict")
                    if cached_verdict in {"aggregate", "read"}:
                        return RecordTextQueryTypeVerdict(
                            verdict=cached_verdict,
                            source="cache",
                            model_id=model_id,
                            cache_hit=True,
                            llm_raw=payload.get("llm_raw"),
                            question_hash=q_hash,
                        )
                except (OSError, json.JSONDecodeError, TypeError):
                    pass
        except OSError:
            cache_path = None  # noqa: F841 — best-effort cache init
    # --- LLM call ---
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=_QUERY_TYPE_SYSTEM_PROMPT),
                ModelMessage(
                    role="user",
                    content=f"Question:\n{question}\n\nAnswer:",
                ),
            ],
            temperature=0.0,
            stream_label="record_text_query_type",
            max_tokens=10,
        )
    except BudgetExceeded:
        # Budget exhaustion is fatal at the task level; surface it.
        raise
    except Exception as exc:  # noqa: BLE001
        return RecordTextQueryTypeVerdict(
            verdict=_keyword_classify_record_text_query(question),
            source="fallback_error",
            model_id=model_id,
            error=str(exc),
            question_hash=q_hash,
        )

    parsed = _parse_query_type_response(raw)
    if parsed is None:
        return RecordTextQueryTypeVerdict(
            verdict=_keyword_classify_record_text_query(question),
            source="fallback_error",
            model_id=model_id,
            llm_raw=raw,
            error="unparseable_llm_response",
            question_hash=q_hash,
        )

    # --- Cache write (best effort) ---
    if cache_dir is not None:
        try:
            cache_path = _query_type_cache_path(cache_dir, q_hash, model_id)
            cache_path.write_text(
                json.dumps(
                    {
                        "verdict": parsed,
                        "model_id": model_id,
                        "question_hash": q_hash,
                        "llm_raw": raw,
                    },
                    ensure_ascii=False,
                )
            )
        except OSError:
            pass

    return RecordTextQueryTypeVerdict(
        verdict=parsed,
        source="llm",
        model_id=model_id,
        llm_raw=raw,
        question_hash=q_hash,
    )


def classify_record_text_query(
    question: str,
    *,
    model: Any = None,
    cache_dir: Path | None = None,
) -> str:
    """Back-compat wrapper returning just the verdict string.

    Prefer ``classify_record_text_query_verdict`` when you also want to
    record ``source`` (llm / cache / fallback) into trace.json.
    """
    return classify_record_text_query_verdict(
        question, model=model, cache_dir=cache_dir
    ).verdict


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
#
# Generic record-boundary detector. We no longer hard-code id vocabulary
# like "patient id" / "race id" here; any "<id-like noun> + number"
# combination is treated as a record mention.


_ID_MENTION_RE = re.compile(
    r"\b(?:id|number|record|file|case|entry|ref(?:erence)?)\b"
    r"[^.\n]{0,60}?\b\d{2,}\b",
    re.I,
)


def _split_into_record_chunks(text: str, *, target_chars: int = 6000) -> list[str]:
    """Greedy chunker that aligns chunk boundaries to paragraph breaks.

    Keeps related id-mentions in the same chunk by never splitting mid-
    paragraph. Chunks with zero id mentions are dropped — they are
    almost always abstract / methodology sections.
    """
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for para in paragraphs:
        if buffer_len + len(para) > target_chars and buffer:
            chunks.append("\n\n".join(buffer))
            buffer = [para]
            buffer_len = len(para)
        else:
            buffer.append(para)
            buffer_len += len(para) + 2
    if buffer:
        chunks.append("\n\n".join(buffer))

    return [chunk for chunk in chunks if _ID_MENTION_RE.search(chunk)]


_STOPWORDS = {
    "among", "whose", "level", "them", "they", "that", "with", "from",
    "please", "give", "many", "how", "what", "which", "then", "than",
    "not", "yet", "are", "the", "and", "for",
}


def _keyword_terms(*parts: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", part.lower()):
            token = token.strip("_-")
            if token in _STOPWORDS or token in seen:
                continue
            seen.add(token)
            terms.append(token)
    return terms[:80]


def _select_relevant_chunks(
    chunks: list[str],
    *,
    question: str,
    semantic_context: str,
    schema_fields: list[str],
    max_chunks: int,
) -> list[str]:
    """Keep only chunks whose text overlaps with question terms.

    This is a guardrail, not a solver: it prevents a 300KB narrative
    report from fanning into dozens of LLM extraction calls.
    """
    terms = _keyword_terms(question, semantic_context, " ".join(schema_fields))
    if not terms:
        return chunks[:max_chunks]

    scored: list[tuple[int, int, str]] = []
    for idx, chunk in enumerate(chunks):
        lowered = chunk.lower()
        score = sum(1 for term in terms if term in lowered)
        if score:
            scored.append((score, -idx, chunk))
    if not scored:
        return chunks[:max_chunks]
    scored.sort(reverse=True)
    selected = [chunk for _score, _neg_idx, chunk in scored[:max_chunks]]
    order = {id(chunk): idx for idx, chunk in enumerate(chunks)}
    selected.sort(key=lambda chunk: order.get(id(chunk), 0))
    return selected


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------
#
# We no longer ship domain-specific ("patient" / "race") field
# dictionaries. Instead, the schema is derived from whatever key:value
# pairs the profiler already observed in the file, with a tiny generic
# fallback. Anything more precise is the LLM extractor's job.


_GENERIC_DEFAULT_FIELDS = ["id", "name", "date", "value"]


def _guess_schema(cap: SourceCapability, *, question: str = "") -> list[str]:
    """Choose a target schema for a record_text file.

    Strategy, in order:
      1. Use fields the deterministic profiler already collected from the
         file's own "key: value" lines.
      2. Otherwise fall back to a very small generic schema.

    No hard-coded topic-based lists.
    """
    fields: list[str] = list(cap.structured_record_fields or [])
    if fields:
        return fields
    return list(_GENERIC_DEFAULT_FIELDS)


def _schema_hash(fields: list[str]) -> str:
    return hashlib.sha256(",".join(fields).encode("utf-8")).hexdigest()[:12]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You extract structured records from a chunk of a long-form narrative "
    "report. The report is prose, not a clean record list. Many sentences "
    "will be context, summary, methodology, or corrections (e.g. "
    "'originally X; corrected to Y'). For every record explicitly "
    "mentioned in the chunk, output ONE JSON object with exactly the "
    "fields requested. Use null for fields not present in this chunk. "
    "When both an original and a corrected value are mentioned for the "
    "same field, keep the corrected one. Output STRICT JSON: a single "
    "top-level array; no prose."
)


def _build_user_prompt(
    *,
    chunk: str,
    schema_fields: list[str],
    question: str,
    semantic_context: str,
) -> str:
    return (
        "Original question:\n"
        + question
        + "\n\nSemantic/context guide from knowledge files:\n"
        + (semantic_context[:4000] if semantic_context.strip() else "(none)")
        + "\n\n"
        "Schema fields (each output object must contain ALL of these keys, "
        "with null where missing):\n"
        + json.dumps(schema_fields, ensure_ascii=False)
        + "\n\nChunk:\n"
        + chunk
        + "\n\nReturn a JSON array of record objects only. No prose, no "
        "markdown fencing, no explanations."
    )


# ---------------------------------------------------------------------------
# Response parsing + caching
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _parse_records(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence is not None:
        text = fence.group(1).strip()
    if not text:
        return []
    start = text.find("[")
    if start < 0:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


@dataclass(slots=True)
class _ExtractionCache:
    cache_dir: Path

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, *, file_hash: str, schema_hash: str, model_id: str, chunk_hash: str) -> str:
        digest = hashlib.sha256(
            f"{model_id}\x00{file_hash}\x00{schema_hash}\x00{chunk_hash}".encode("utf-8")
        ).hexdigest()
        return digest[:32]

    def get(self, **kw: Any) -> list[dict[str, Any]] | None:
        path = self.cache_dir / f"{self._key(**kw)}.json"
        if not path.exists():
            return None
        try:
            return list(json.loads(path.read_text())["records"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    def put(self, *, records: list[dict[str, Any]], **kw: Any) -> None:
        path = self.cache_dir / f"{self._key(**kw)}.json"
        try:
            path.write_text(json.dumps({"records": records}, ensure_ascii=False))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Rule-hint surfacing
# ---------------------------------------------------------------------------


def _relevant_rule_lines(
    *,
    semantic_context: str,
    question: str,
    schema_fields: list[str],
) -> list[str]:
    """Pick a handful of semantic-rule lines that mention question terms.

    Used only as a prompt hint downstream; we never "apply" these rules
    ourselves in code.
    """
    terms = set(_keyword_terms(question, " ".join(schema_fields)))
    lines: list[str] = []
    for raw_line in semantic_context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(term in lowered for term in terms):
            lines.append(line)
        if len(lines) >= 8:
            break
    return lines


def _preview_records(records: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    return [dict(record) for record in records[:limit]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StructuredDocExtraction:
    """Result of running the extractor on one task."""

    synthesized_csvs: dict[str, str] = field(default_factory=dict)  # original_path → temp csv path
    schemas: dict[str, list[str]] = field(default_factory=dict)
    record_counts: dict[str, int] = field(default_factory=dict)
    chunk_counts: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    notes: list[str] = field(default_factory=list)
    rule_notes: dict[str, list[str]] = field(default_factory=dict)
    previews: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Retained for backward compatibility with trace viewers.
    positive_previews: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # How the question was classified; recorded for audit.
    query_type: str = "aggregate"
    # Full classifier verdict (LLM vs cache vs fallback). Retained
    # alongside ``query_type`` so trace.json can answer "how did we pick
    # this branch?" without re-running the classifier.
    query_type_verdict: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "synthesized_csvs": dict(self.synthesized_csvs),
            "schemas": {k: list(v) for k, v in self.schemas.items()},
            "record_counts": dict(self.record_counts),
            "chunk_counts": dict(self.chunk_counts),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "notes": list(self.notes),
            "rule_notes": {k: list(v) for k, v in self.rule_notes.items()},
            "previews": {k: list(v) for k, v in self.previews.items()},
            "positive_previews": {k: list(v) for k, v in self.positive_previews.items()},
            "query_type": self.query_type,
            "query_type_verdict": dict(self.query_type_verdict) if self.query_type_verdict else None,
        }


@dataclass(slots=True)
class StructuredDocExecutor:
    """LLM-extraction stage that synthesizes a CSV per record-text file."""

    model: OpenAIModelAdapter
    compiled_task: CompiledTask
    chunk_chars: int = 7000
    cache_dir: Path | None = None
    # Separate cache for the record_text query_type classifier. Kept on
    # its own directory (not the extraction cache) because the key shape
    # is different: ``(question_hash, model_id)`` vs extraction's
    # ``(file_hash, schema_hash, model_id, chunk_hash)``.
    query_type_cache_dir: Path | None = None
    max_chunks_per_file: int = 8
    extraction_max_tokens: int = 900
    output_dir: Path | None = None  # where synthesized csvs are placed; default = task.context_dir / .synthesized

    def has_structured_prose(self) -> bool:
        return any(
            cap.kind in {"record_text", "structured_table"}
            for cap in self.compiled_task.source_capabilities
        )

    def run(self, task: PublicTask) -> StructuredDocExtraction:
        """Extract records from every record_text file on this task.

        For ``aggregate``-style questions we synthesize a CSV per file so
        the downstream codegen can use ``pandas`` directly. For ``read``-
        style questions we skip extraction entirely: materializing a table
        would drop the narrative context that the question actually needs.
        """
        result = StructuredDocExtraction()
        if not self.has_structured_prose():
            result.notes.append("no_structured_prose_files")
            return result

        # Branch: read-comprehension vs aggregate. The classifier first
        # tries the LLM (with (question_hash, model_id) cache + budget
        # accounting inside model.complete); on failure it falls back to
        # the deterministic keyword vote defined above. The full verdict
        # (including ``source``) is persisted into trace.json so after-
        # the-fact debugging can tell LLM-driven from fallback decisions
        # apart.
        verdict = classify_record_text_query_verdict(
            task.question,
            model=self.model,
            cache_dir=self.query_type_cache_dir,
        )
        query_type = verdict.verdict
        result.query_type = query_type
        result.query_type_verdict = verdict.to_dict()
        result.notes.append(
            f"query_type_classifier:source={verdict.source}"
            + (":cache_hit" if verdict.cache_hit else "")
            + (f":error={verdict.error}" if verdict.error else "")
        )
        if query_type == "read":
            result.notes.append(
                "skip_synthesis:query_type=read "
                "(downstream path should use raw text / RAG, not synthesized CSVs)"
            )
            return result

        from data_agent_baseline.progress import get_progress_logger

        logger = get_progress_logger()
        structured_files = [
            cap.path
            for cap in self.compiled_task.source_capabilities
            if cap.kind in {"record_text", "structured_table"}
        ]
        if logger is not None:
            logger.structured_extract_start(files=structured_files)

        cache = _ExtractionCache(cache_dir=self.cache_dir) if self.cache_dir else None
        out_root = self.output_dir or (task.context_dir / ".synthesized")
        out_root.mkdir(parents=True, exist_ok=True)
        model_id = f"{getattr(self.model, 'api_base', 'scripted')}::{getattr(self.model, 'model', type(self.model).__name__)}"
        semantic_context = self._read_semantic_context(task)

        for cap in self.compiled_task.source_capabilities:
            if cap.kind not in {"record_text", "structured_table"}:
                continue
            abs_path = task.context_dir / cap.path
            try:
                text = abs_path.read_text(errors="replace")
            except OSError as exc:
                result.notes.append(f"could_not_read:{cap.path}:{exc}")
                continue

            schema_fields = _guess_schema(cap, question=task.question)
            chunks = _split_into_record_chunks(text, target_chars=self.chunk_chars)
            raw_chunk_count = len(chunks)
            chunks = _select_relevant_chunks(
                chunks,
                question=task.question,
                semantic_context=semantic_context,
                schema_fields=schema_fields,
                max_chunks=self.max_chunks_per_file,
            )
            result.chunk_counts[cap.path] = len(chunks)
            result.schemas[cap.path] = list(schema_fields)
            result.rule_notes[cap.path] = _relevant_rule_lines(
                semantic_context=semantic_context,
                question=task.question,
                schema_fields=schema_fields,
            )
            if len(chunks) < raw_chunk_count:
                result.notes.append(f"chunk_pruned:{cap.path}:{raw_chunk_count}->{len(chunks)}")
            if not chunks:
                result.notes.append(f"no_record_chunks:{cap.path}")
                continue

            file_h = _file_hash(abs_path)
            schema_h = _schema_hash(schema_fields)
            all_records: list[dict[str, Any]] = []
            for chunk in chunks:
                chunk_h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:12]
                cached = (
                    cache.get(
                        file_hash=file_h,
                        schema_hash=schema_h,
                        model_id=model_id,
                        chunk_hash=chunk_h,
                    )
                    if cache is not None
                    else None
                )
                if cached is not None:
                    result.cache_hits += 1
                    all_records.extend(cached)
                    continue
                result.cache_misses += 1

                budget = get_budget_controller()
                if budget is not None:
                    budget.consume_llm("structured_doc_extract")
                try:
                    raw = self.model.complete(
                        [
                            ModelMessage(role="system", content=_SYSTEM_PROMPT),
                            ModelMessage(
                                role="user",
                                content=_build_user_prompt(
                                    chunk=chunk,
                                    schema_fields=schema_fields,
                                    question=task.question,
                                    semantic_context=semantic_context,
                                ),
                            ),
                        ],
                        temperature=0.0,
                        stream_label=f"struct_extract:{cap.path}",
                        max_tokens=self.extraction_max_tokens,
                    )
                except Exception as exc:  # noqa: BLE001
                    result.notes.append(f"extract_error:{cap.path}:{exc}")
                    continue

                records = _parse_records(raw)
                if cache is not None:
                    cache.put(
                        records=records,
                        file_hash=file_h,
                        schema_hash=schema_h,
                        model_id=model_id,
                        chunk_hash=chunk_h,
                    )
                all_records.extend(records)

            # Dedupe on a generic id-like column if the schema has one.
            # We pick the first schema field whose name ends in "id" or
            # equals "id" / "ref"; no domain-specific preferences.
            id_field = next(
                (
                    f for f in schema_fields
                    if f.lower() == "id"
                    or f.lower().endswith("_id")
                    or f.lower() in {"ref", "reference"}
                ),
                None,
            )
            if id_field is not None:
                seen: dict[Any, dict[str, Any]] = {}
                for rec in all_records:
                    key = rec.get(id_field)
                    if key is None:
                        continue
                    seen[key] = {**(seen.get(key) or {}), **rec}
                merged_records = list(seen.values())
            else:
                merged_records = all_records

            result.record_counts[cap.path] = len(merged_records)
            result.previews[cap.path] = _preview_records(merged_records)
            if not merged_records:
                result.notes.append(f"empty_extraction:{cap.path}")
                continue

            csv_name = cap.path.replace("/", "__").replace(".md", "") + ".synth.csv"
            csv_path = out_root / csv_name
            self._write_csv(csv_path=csv_path, schema_fields=schema_fields, records=merged_records)
            try:
                rel = csv_path.relative_to(task.context_dir).as_posix()
            except ValueError:
                rel = str(csv_path)
            result.synthesized_csvs[cap.path] = rel
        if logger is not None:
            logger.structured_extract_done(
                synthesized_csvs=result.synthesized_csvs,
                record_counts=result.record_counts,
                cache_hits=result.cache_hits,
                cache_misses=result.cache_misses,
                notes=result.notes,
                rule_notes=result.rule_notes,
                previews=result.previews,
                positive_previews=result.positive_previews,
            )
        return result

    def _read_semantic_context(self, task: PublicTask) -> str:
        parts: list[str] = []
        for cap in self.compiled_task.source_capabilities:
            if cap.role != "semantic_rule":
                continue
            path = task.context_dir / cap.path
            try:
                text = path.read_text(errors="replace").strip()
            except OSError:
                continue
            if text:
                parts.append(f"# {cap.path}\n{text}")
        return "\n\n".join(parts)

    def _write_csv(
        self,
        *,
        csv_path: Path,
        schema_fields: list[str],
        records: list[dict[str, Any]],
    ) -> None:
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(schema_fields)
            for record in records:
                writer.writerow([record.get(f) for f in schema_fields])


def render_synthesis_block(extraction: StructuredDocExtraction) -> str:
    """Render extraction outcome for the codegen prompt as a small banner."""
    if not extraction.synthesized_csvs:
        return ""
    lines = [
        "Synthesized CSVs (already extracted from record_text docs):",
        f"  query_type={extraction.query_type}",
    ]
    for src, csv_rel in extraction.synthesized_csvs.items():
        schema = ", ".join(extraction.schemas.get(src) or [])
        record_count = extraction.record_counts.get(src, 0)
        lines.append(
            f"- {csv_rel}  (← {src})  records={record_count}  columns: {schema}"
        )
    lines.append(
        "Use `pandas.read_csv` on the synthesized CSVs above; do NOT re-parse "
        "the original .md files."
    )
    return "\n".join(lines)
