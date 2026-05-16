"""Record-text query-type classifier.

Given a natural-language question against a ``record_text`` context, decide
whether the downstream path should:

    "aggregate" — materialize a CSV (record extraction → pandas codegen).
                  Good for count / filter / groupby / topk / numerical
                  questions where a tabular view wins.

    "read"     — stay on the raw narrative text (RAG / codegen over prose).
                  Good for summarize / explain / interpret / "why" style
                  questions where forcing a schema loses context.

Design
------
The previous implementation was a keyword vote on the question. That is
both too blunt (misses paraphrased verbs) and too domain-specific (biased
toward medical-record wording). This module upgrades the decision to a
small, cached LLM call while keeping the keyword classifier as a safety
net, so failures never block the pipeline.

Pipeline for ``classify_record_text_query(question, model, model_id)``:

    1. Hash ``(question, model_id)`` → local JSON cache under
       ``artifacts/cache/record_text_classifier/``. On hit, return the
       cached verdict. No LLM call, no budget spend.

    2. Cache miss and an LLM ``model`` is available →
         - Account one LLM call via ``BudgetController.consume_llm`` (the
           ``OpenAIModelAdapter.complete`` path does the accounting, so we
           pass a descriptive ``stream_label`` and let it charge once).
         - Request: temperature=0, max_tokens=10, two-choice answer.
         - Parse the single-token verdict. Persist to cache on success.

    3. LLM unavailable, raises, returns garbage, or budget-exhausted →
         fall back to the keyword classifier. The result is **not**
         cached (so a retry with a working LLM can replace it).

The return type is always ``(verdict, ClassificationMeta)`` so callers
can record in trace.json exactly which branch produced the answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.budget import BudgetExceeded


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


VERDICT_AGGREGATE = "aggregate"
VERDICT_READ = "read"
_VALID_VERDICTS: frozenset[str] = frozenset({VERDICT_AGGREGATE, VERDICT_READ})

# Verdict returned when the question is empty / non-string. Treated as
# aggregate so the fast path remains the default.
_DEFAULT_VERDICT = VERDICT_AGGREGATE


# ---------------------------------------------------------------------------
# Keyword classifier (safety net)
# ---------------------------------------------------------------------------
#
# Deliberately kept even though we have an LLM classifier now: when the
# LLM call fails (network, timeout, budget exhausted, malformed output),
# we still need a deterministic decision. The keyword set is intentionally
# generic — no medical / sports vocabulary — so it does not bias the
# domain-agnostic pipeline.
#
# Matching is word-boundary-aware: single-word cues (e.g. "sum") must
# match a whole word, not a substring. This prevents accidents like
# "summarize" being counted as an aggregation cue because "sum" is a
# prefix of it. Multi-word cues ("how many", "group by") are matched
# as literal substrings since their spaces already anchor them.


_AGGREGATE_KEYWORDS: tuple[str, ...] = (
    "how many", "count", "number of", "total", "sum",
    "average", "mean", "median", "ratio", "percentage", "percent",
    "list all", "whose", "filter",
    "greater than", "less than", "at least", "at most",
    "top", "bottom", "highest", "lowest", "largest", "smallest",
    "maximum", "minimum", "rank", "order by",
    "for each", "per", "group by",
)

_READING_KEYWORDS: tuple[str, ...] = (
    "summarize", "summary", "describe", "explain", "why",
    "infer", "deduce", "conclude", "interpret",
    "what is mentioned", "what does the report say",
    "compare the narrative", "narrative", "overall",
    "opinion", "argument", "rationale", "reasoning behind",
)


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    hits = 0
    for kw in keywords:
        kw_lower = kw.lower()
        if " " in kw_lower:
            # Multi-word phrase: plain substring match is fine because
            # the internal space anchors it away from other words.
            if kw_lower in text:
                hits += 1
        else:
            # Single word: require word-boundary match so "sum" does
            # not fire inside "summarize".
            pattern = r"\b" + re.escape(kw_lower) + r"\b"
            if re.search(pattern, text):
                hits += 1
    return hits


def classify_by_keywords(question: str) -> str:
    """Return a verdict based solely on keyword hits.

    Tie-breaks toward ``aggregate`` so truly ambiguous questions still
    get the fast table-synthesis path; ``read`` requires a strictly
    higher count of reading-comprehension cues than aggregation cues.
    """
    q = (question or "").lower()
    agg_hits = _count_keyword_hits(q, _AGGREGATE_KEYWORDS)
    read_hits = _count_keyword_hits(q, _READING_KEYWORDS)
    if read_hits > agg_hits and read_hits >= 1:
        return VERDICT_READ
    return VERDICT_AGGREGATE


# ---------------------------------------------------------------------------
# Classification metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassificationMeta:
    """Audit trail for one classification call.

    ``source`` tells trace.json consumers which branch produced the verdict:
      - ``"cache"``            : hit on the on-disk cache.
      - ``"llm"``              : LLM call succeeded and result was used.
      - ``"fallback_keyword"`` : LLM failed / unavailable / invalid output.

    ``llm_error`` is populated only for fallback cases so the post-mortem
    can tell ``LLM timed out`` from ``LLM returned 'maybe'``.
    """

    source: str
    verdict: str
    model_id: str
    cache_hit: bool = False
    cache_key: str = ""
    raw_response: str = ""
    llm_error: str = ""
    keyword_vote: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "verdict": self.verdict,
            "model_id": self.model_id,
            "cache_hit": self.cache_hit,
            "cache_key": self.cache_key,
            "raw_response": self.raw_response,
            "llm_error": self.llm_error,
            "keyword_vote": dict(self.keyword_vote),
        }


def _keyword_vote(question: str) -> dict[str, int]:
    q = (question or "").lower()
    return {
        "aggregate_hits": _count_keyword_hits(q, _AGGREGATE_KEYWORDS),
        "read_hits": _count_keyword_hits(q, _READING_KEYWORDS),
    }


# ---------------------------------------------------------------------------
# On-disk cache  (JSON-per-key, same pattern as _ExtractionCache)
# ---------------------------------------------------------------------------
#
# Key is ``sha256(question || "\x00" || model_id)``. Value is a tiny JSON
# blob with just the verdict + raw LLM response (for later audits).
# Intentionally NOT putting cache entries behind sqlite or pickle — the
# codebase already standardizes on one-file-per-key JSON for
# ``_ExtractionCache`` / ``EmbeddingCache``; a classifier cache at a few
# hundred entries per run is a drop in the bucket.


def _default_cache_dir() -> Path | None:
    try:
        from data_agent_baseline.config import PROJECT_ROOT
    except Exception:  # noqa: BLE001
        return None
    return PROJECT_ROOT / "artifacts" / "cache" / "record_text_classifier"


def _cache_key(question: str, model_id: str) -> str:
    payload = f"{model_id}\x00{question}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _cache_read(cache_dir: Path | None, key: str) -> tuple[str, str] | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    verdict = str(blob.get("verdict", ""))
    if verdict not in _VALID_VERDICTS:
        return None
    return verdict, str(blob.get("raw_response", ""))


def _cache_write(cache_dir: Path | None, key: str, *, verdict: str, raw_response: str) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(
            json.dumps(
                {"verdict": verdict, "raw_response": raw_response},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        # Cache is best-effort; a failure here must never abort the run.
        return


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------
#
# Two-choice prompt, forced short answer. We take the first bare word
# ``aggregate|read`` we see; anything else is treated as a parse failure
# and surfaces as a keyword fallback. No markdown, no JSON, no
# explanation — staying close to a single token keeps max_tokens=10
# comfortably sufficient.


_SYSTEM_PROMPT = (
    "You classify a user question into exactly one label based on how a "
    "downstream data pipeline should handle it.\n"
    "\n"
    "Labels:\n"
    "  aggregate  — the question asks for counts, sums, averages, "
    "filters, rankings, group-bys, or other tabular computations over "
    "records. A structured table (CSV) view of the records will help.\n"
    "  read       — the question asks to summarize, explain, interpret, "
    "or reason about narrative content. Forcing a tabular view would "
    "drop the narrative context.\n"
    "\n"
    "Respond with the single lowercase word 'aggregate' or 'read'. No "
    "punctuation, no explanation."
)


def _build_user_prompt(question: str) -> str:
    snippet = (question or "").strip()
    # Defensive truncation: classifier never needs more than a few
    # hundred characters of the question to decide. Keeps token cost flat
    # even if a downstream caller passes a pathological input.
    if len(snippet) > 1500:
        snippet = snippet[:1500]
    return f"Question: {snippet}\nLabel:"


_PARSE_RE = re.compile(r"\b(aggregate|read)\b", re.IGNORECASE)


def _parse_verdict(raw: str) -> str | None:
    match = _PARSE_RE.search(raw or "")
    if not match:
        return None
    return match.group(1).lower()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def classify_record_text_query(
    question: str,
    *,
    model: ModelAdapter | None = None,
    model_id: str = "",
    cache_dir: Path | None = None,
    stream_label: str = "record_text_classifier",
) -> tuple[str, ClassificationMeta]:
    """Return ``(verdict, meta)`` for a record_text question.

    Parameters
    ----------
    question:
        The user question.
    model:
        Optional LLM adapter. When ``None``, the keyword classifier is
        used directly (meta.source == "fallback_keyword", llm_error ==
        "no_model_configured"). This keeps unit tests cheap.
    model_id:
        Stable identifier of the model (e.g. ``"qwen-plus"``,
        ``"qwen-max"``). Participates in the cache key so switching
        model tiers invalidates cached verdicts automatically.
    cache_dir:
        Optional override for the on-disk cache directory. ``None``
        resolves to ``<project>/artifacts/cache/record_text_classifier/``.
    stream_label:
        Passed through to ``model.complete`` so budget accounting and
        the streaming UI can distinguish classifier calls from codegen.
    """
    question_text = (question or "").strip()
    keyword_vote = _keyword_vote(question_text)

    # Empty question: short-circuit to the default without touching the
    # LLM. Callers should not hit this in practice but it is free defense.
    if not question_text:
        return _DEFAULT_VERDICT, ClassificationMeta(
            source="fallback_keyword",
            verdict=_DEFAULT_VERDICT,
            model_id=model_id,
            llm_error="empty_question",
            keyword_vote=keyword_vote,
        )

    resolved_cache_dir = cache_dir if cache_dir is not None else _default_cache_dir()
    key = _cache_key(question_text, model_id)

    cached = _cache_read(resolved_cache_dir, key)
    if cached is not None:
        cached_verdict, cached_raw = cached
        return cached_verdict, ClassificationMeta(
            source="cache",
            verdict=cached_verdict,
            model_id=model_id,
            cache_hit=True,
            cache_key=key,
            raw_response=cached_raw,
            keyword_vote=keyword_vote,
        )

    # No model wired in → keyword fallback.
    if model is None:
        verdict = classify_by_keywords(question_text)
        return verdict, ClassificationMeta(
            source="fallback_keyword",
            verdict=verdict,
            model_id=model_id,
            cache_key=key,
            llm_error="no_model_configured",
            keyword_vote=keyword_vote,
        )

    # Real LLM call. OpenAIModelAdapter.complete will charge
    # BudgetController.consume_llm(stream_label) internally.
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=_SYSTEM_PROMPT),
                ModelMessage(role="user", content=_build_user_prompt(question_text)),
            ],
            temperature=0.0,
            max_tokens=10,
            stream_label=stream_label,
        )
    except BudgetExceeded:
        # Budget exhausted must bubble up — the router treats it as a
        # terminal error, not a classifier miss. Letting it propagate
        # keeps behavior consistent with every other LLM call in the
        # pipeline.
        raise
    except Exception as exc:  # noqa: BLE001
        verdict = classify_by_keywords(question_text)
        return verdict, ClassificationMeta(
            source="fallback_keyword",
            verdict=verdict,
            model_id=model_id,
            cache_key=key,
            llm_error=f"{type(exc).__name__}: {exc}"[:300],
            keyword_vote=keyword_vote,
        )

    parsed = _parse_verdict(raw)
    if parsed is None:
        # LLM returned text we could not map — fall back, but record the
        # raw output so we can audit whether the prompt needs tightening.
        verdict = classify_by_keywords(question_text)
        return verdict, ClassificationMeta(
            source="fallback_keyword",
            verdict=verdict,
            model_id=model_id,
            cache_key=key,
            raw_response=(raw or "")[:200],
            llm_error="unparsable_verdict",
            keyword_vote=keyword_vote,
        )

    _cache_write(resolved_cache_dir, key, verdict=parsed, raw_response=(raw or "")[:200])

    return parsed, ClassificationMeta(
        source="llm",
        verdict=parsed,
        model_id=model_id,
        cache_hit=False,
        cache_key=key,
        raw_response=(raw or "")[:200],
        keyword_vote=keyword_vote,
    )


__all__ = [
    "VERDICT_AGGREGATE",
    "VERDICT_READ",
    "ClassificationMeta",
    "classify_by_keywords",
    "classify_record_text_query",
]
