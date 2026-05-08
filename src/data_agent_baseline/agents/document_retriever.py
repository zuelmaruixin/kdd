"""Production-grade RAG retrieval for long-document questions.

Pipeline (configurable):

    ┌──────────────────────────────────────────────────────────────────┐
    │  load → chunk → contextual prefix (heading path)                 │
    └──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
            ┌──────────────────────────────────────┐
            │  Query Transformation                │
            │   • original                         │
            │   • LLM paraphrases (Query Expansion)│
            │   • Hypothetical answer (HyDE)       │
            └──────────────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
       ┌────────────────┐               ┌──────────────────┐
       │ BM25Retriever  │               │ EmbeddingRetriever│
       │ (sparse)       │               │ (dense)          │
       └───────┬────────┘               └────────┬─────────┘
               └────────────┬──────────────────────┘
                            ▼
            ┌──────────────────────────────────────┐
            │  Reciprocal Rank Fusion (RRF)        │
            └──────────────────────────────────────┘
                            │
                            ▼
            ┌──────────────────────────────────────┐
            │  Cross-Encoder Reranker (optional)   │
            └──────────────────────────────────────┘
                            │
                            ▼
                       Top-K chunks

Why each component:

* BM25 (not TF-IDF): IDF saturation, length normalization. Industry
  standard for sparse retrieval.
* Hybrid + RRF: dense embeddings catch semantic matches, sparse catches
  exact entity / id matches; RRF fuses ranks (not raw scores) so we
  don't need to normalize across retrievers.
* Query expansion: a single question phrasing rarely covers all the
  relevant chunks. LLM-generated paraphrases multiply recall.
* HyDE: embedding the *imagined answer* often beats embedding the
  question, because the answer's vocabulary is closer to the document
  text we're trying to retrieve.
* Reranker: cross-encoders score (query, chunk) jointly and beat any
  single-tower retriever; first-stage retrieves 30, reranker keeps K.
* Contextual prefix: prepending the chunk's heading path injects parent
  section context into the retrieval signal, free improvement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence


# ===========================================================================
# 1. Loading: .md / .txt / .docx → flat text
# ===========================================================================


def _read_docx(path: Path) -> str:
    """Read a .docx into flat text. Requires ``python-docx``.

    We collect both paragraph text and table cells so business-rule
    tables embedded in Word docs aren't lost.
    """
    try:
        import docx  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading .docx requires `python-docx`. Install with: pip install python-docx"
        ) from exc

    document = docx.Document(str(path))
    parts: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx(path)
    if suffix in {".md", ".txt"}:
        return path.read_text(errors="replace")
    raise ValueError(f"Unsupported document type: {path}")


# ===========================================================================
# 2. Chunking with heading awareness
# ===========================================================================


@dataclass(slots=True)
class DocumentChunk:
    """One retrievable piece of a document.

    ``raw_text`` is the literal extracted text. ``indexable_text`` is what
    we feed retrievers — by default it's ``raw_text`` prefixed with the
    section heading path so retrieval signals include parent context.
    """

    source_path: str
    heading_path: tuple[str, ...]
    raw_text: str
    indexable_text: str = ""

    def __post_init__(self) -> None:
        if not self.indexable_text:
            self.indexable_text = self._build_indexable()

    def _build_indexable(self) -> str:
        if not self.heading_path:
            return self.raw_text
        prefix = " › ".join(self.heading_path)
        # Inject context, but don't double-bill the LLM's char budget;
        # the *displayed* chunk uses raw_text only.
        return f"[{prefix}]\n{self.raw_text}"

    @property
    def heading_label(self) -> str:
        if not self.heading_path:
            return "(top)"
        return " › ".join(self.heading_path)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.indexable_text.encode("utf-8")).hexdigest()[:16]


_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)


def _split_markdown_by_headers(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Return ``[(heading_path, body)]`` in document order, preserving header hierarchy."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [((), text)]

    sections: list[tuple[tuple[str, ...], str]] = []
    pre = text[: matches[0].start()].strip()
    if pre:
        sections.append(((), pre))

    heading_stack: list[tuple[int, str]] = []  # (level, heading)
    for idx, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading))

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            path = tuple(item[1] for item in heading_stack)
            sections.append((path, body))
    return sections


def _split_paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]


def _sliding_window(
    text: str,
    *,
    max_chunk_chars: int,
    overlap_chars: int,
) -> list[str]:
    if len(text) <= max_chunk_chars:
        return [text]
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return [text]

    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(para) > max_chunk_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            cursor = 0
            while cursor < len(para):
                end = min(cursor + max_chunk_chars, len(para))
                chunks.append(para[cursor:end])
                if end >= len(para):
                    break
                cursor = end - overlap_chars
                if cursor <= 0:
                    break
            continue
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if len(candidate) > max_chunk_chars:
            chunks.append(buffer.strip())
            tail = buffer[-overlap_chars:] if overlap_chars > 0 else ""
            buffer = (tail + "\n\n" + para).strip() if tail else para
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer.strip())
    return chunks


def chunk_document(
    *,
    text: str,
    source_path: str,
    max_chunk_chars: int = 1200,
    chunk_overlap_chars: int = 150,
) -> list[DocumentChunk]:
    sections = _split_markdown_by_headers(text)
    chunks: list[DocumentChunk] = []
    for heading_path, body in sections:
        for window in _sliding_window(
            body,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=chunk_overlap_chars,
        ):
            window_text = window.strip()
            if window_text:
                chunks.append(DocumentChunk(
                    source_path=source_path,
                    heading_path=heading_path,
                    raw_text=window_text,
                ))
    return chunks


# ===========================================================================
# JSON path chunking
# ===========================================================================
#
# Same retriever, different shredder: walk the JSON tree, dump each
# subtree under ``max_chunk_chars`` as one chunk, descend further when
# a subtree is too large to fit. The path components (``$.records[3].name``)
# are stored in ``heading_path`` so retrieval signals include the parent
# nesting structure — same trick we use for Markdown headings.


def _format_json_pointer(parts: tuple[Any, ...]) -> tuple[str, ...]:
    """Turn ('records', 7, 'name') into a tuple suitable for heading_path."""
    if not parts:
        return ("$",)
    rendered: list[str] = ["$"]
    for part in parts:
        if isinstance(part, int):
            rendered[-1] = rendered[-1] + f"[{part}]"
        else:
            rendered.append(str(part))
    return tuple(rendered)


def _dump_subtree(payload: Any, *, max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(json subtree truncated)…"
    return text


def chunk_json(
    *,
    payload: Any,
    source_path: str,
    max_chunk_chars: int = 1200,
    max_depth: int = 4,
) -> list[DocumentChunk]:
    """Recursive path-aware JSON chunker.

    Algorithm:
      1. If the current subtree fits in ``max_chunk_chars``, emit one chunk.
      2. Else if it's a list, emit each element as its own chunk
         (recursively if elements are still too big).
      3. Else if it's a dict, emit each (key, value) pair as a chunk.
      4. Bail out at ``max_depth`` to avoid pathological recursion.

    The whole tree always gets a parent "summary" chunk with top-level
    keys + counts so retrievers can also match against the high-level
    structure.
    """
    chunks: list[DocumentChunk] = []

    def _summary_chunk() -> None:
        if isinstance(payload, dict):
            summary = "top-level keys: " + json.dumps(list(payload.keys()), ensure_ascii=False)
            if "records" in payload and isinstance(payload["records"], list):
                summary += f"\nrecords count: {len(payload['records'])}"
        elif isinstance(payload, list):
            summary = f"top-level list with {len(payload)} items"
        else:
            summary = f"scalar json payload: {type(payload).__name__}"
        chunks.append(DocumentChunk(
            source_path=source_path,
            heading_path=("$",),
            raw_text=summary,
        ))

    _summary_chunk()

    def _walk(value: Any, parts: tuple[Any, ...], depth: int) -> None:
        rendered = _dump_subtree(value, max_chars=max_chunk_chars)
        if len(rendered) <= max_chunk_chars or depth >= max_depth:
            chunks.append(DocumentChunk(
                source_path=source_path,
                heading_path=_format_json_pointer(parts),
                raw_text=rendered,
            ))
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                _walk(item, parts + (index,), depth + 1)
        elif isinstance(value, dict):
            for key, sub in value.items():
                _walk(sub, parts + (str(key),), depth + 1)
        else:  # scalar that's somehow too large to fit; just emit it
            chunks.append(DocumentChunk(
                source_path=source_path,
                heading_path=_format_json_pointer(parts),
                raw_text=rendered,
            ))

    _walk(payload, parts=(), depth=0)
    return chunks


def chunk_json_file(
    *,
    path: Path,
    max_chunk_chars: int = 1200,
    max_depth: int = 4,
) -> list[DocumentChunk]:
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [DocumentChunk(
            source_path=str(path),
            heading_path=("<load_error>",),
            raw_text=f"(could not parse json {path}: {exc})",
        )]
    return chunk_json(
        payload=payload,
        source_path=str(path),
        max_chunk_chars=max_chunk_chars,
        max_depth=max_depth,
    )


# ===========================================================================
# 3. Tokenization (mixed English + Chinese)
# ===========================================================================


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CHINESE_RE = re.compile(r"[一-鿿]")
_STOPWORDS_EN = frozenset({
    "the", "a", "an", "of", "is", "are", "was", "were", "and", "or", "in",
    "on", "for", "to", "by", "with", "what", "which", "who", "how", "list",
    "show", "give", "name", "names", "all", "this", "that", "these", "those",
    "from", "many", "much", "do", "does", "be", "as", "at", "it", "its",
})


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text):
        lowered = match.lower()
        if len(lowered) > 1 and lowered not in _STOPWORDS_EN:
            tokens.append(lowered)
    tokens.extend(_CHINESE_RE.findall(text))
    return tokens


# ===========================================================================
# 4. Retriever protocol
# ===========================================================================


class Retriever(Protocol):
    """Anything that scores chunks for a query. Higher score = more relevant."""

    chunks: list[DocumentChunk]

    def score(self, query: str) -> list[tuple[float, DocumentChunk]]: ...


# ===========================================================================
# 5. BM25Retriever (Okapi BM25, the standard sparse baseline)
# ===========================================================================


@dataclass(slots=True)
class Bm25Retriever:
    """Okapi BM25 with the canonical (k1=1.5, b=0.75) defaults.

    BM25 fixes two well-known TF-IDF weaknesses: (a) term frequency
    saturation (one extra mention of a term shouldn't 10× the score),
    (b) document length normalization (long docs aren't penalized
    arbitrarily).

    Score formula:
        idf(t)  = log((N − df(t) + 0.5) / (df(t) + 0.5) + 1)
        tf'(t,d) = tf(t,d) * (k1+1) / (tf(t,d) + k1*(1 − b + b * |d|/avgdl))
        score(q,d) = sum_{t in q} idf(t) * tf'(t,d)
    """

    chunks: list[DocumentChunk]
    k1: float = 1.5
    b: float = 0.75
    _doc_term_freqs: list[dict[str, int]] = field(default_factory=list)
    _doc_lengths: list[int] = field(default_factory=list)
    _avgdl: float = 0.0
    _df: dict[str, int] = field(default_factory=dict)
    _idf_cache: dict[str, float] = field(default_factory=dict)
    _N: int = 0

    def __post_init__(self) -> None:
        self._N = len(self.chunks)
        for chunk in self.chunks:
            tokens = tokenize(chunk.indexable_text)
            self._doc_lengths.append(len(tokens))
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._doc_term_freqs.append(tf)
            for token in set(tokens):
                self._df[token] = self._df.get(token, 0) + 1
        if self._doc_lengths:
            self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths)

    def _idf(self, token: str) -> float:
        cached = self._idf_cache.get(token)
        if cached is not None:
            return cached
        df = self._df.get(token, 0)
        if self._N == 0:
            return 0.0
        # Lucene-flavored BM25: clamps idf to non-negative so common terms
        # don't pull us toward irrelevant chunks.
        value = math.log(((self._N - df + 0.5) / (df + 0.5)) + 1.0)
        self._idf_cache[token] = value
        return value

    def score(self, query: str) -> list[tuple[float, DocumentChunk]]:
        if self._N == 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return [(0.0, chunk) for chunk in self.chunks]

        query_token_set = set(query_tokens)
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk_idx, chunk in enumerate(self.chunks):
            tf = self._doc_term_freqs[chunk_idx]
            doc_len = self._doc_lengths[chunk_idx] or 1
            length_norm = (1.0 - self.b) + self.b * (doc_len / max(self._avgdl, 1.0))
            score = 0.0
            for token in query_token_set:
                f = tf.get(token, 0)
                if f == 0:
                    continue
                idf = self._idf(token)
                score += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * length_norm)
            scored.append((score, chunk))
        scored.sort(key=lambda item: -item[0])
        return scored


# Kept around so callers that explicitly want the pure TF-IDF baseline still work.
@dataclass(slots=True)
class TfidfRetriever:
    """TF-IDF baseline. Prefer :class:`Bm25Retriever` in production."""

    chunks: list[DocumentChunk]
    _df: dict[str, int] = field(default_factory=dict)
    _doc_vectors: list[dict[str, float]] = field(default_factory=list)
    _doc_norms: list[float] = field(default_factory=list)
    _N: int = 0

    def __post_init__(self) -> None:
        self._N = len(self.chunks)
        for chunk in self.chunks:
            for token in set(tokenize(chunk.indexable_text)):
                self._df[token] = self._df.get(token, 0) + 1
        for chunk in self.chunks:
            tf: dict[str, int] = {}
            for token in tokenize(chunk.indexable_text):
                tf[token] = tf.get(token, 0) + 1
            vec = self._tfidf(tf)
            norm = math.sqrt(sum(value * value for value in vec.values())) or 1.0
            self._doc_vectors.append(vec)
            self._doc_norms.append(norm)

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        if df == 0 or self._N == 0:
            return 0.0
        return math.log((1 + self._N) / (1 + df)) + 1.0

    def _tfidf(self, tf: dict[str, int]) -> dict[str, float]:
        if not tf:
            return {}
        max_tf = max(tf.values())
        return {
            token: (count / max_tf) * self._idf(token)
            for token, count in tf.items()
        }

    def score(self, query: str) -> list[tuple[float, DocumentChunk]]:
        if not self.chunks:
            return []
        tf: dict[str, int] = {}
        for token in tokenize(query):
            tf[token] = tf.get(token, 0) + 1
        qv = self._tfidf(tf)
        if not qv:
            return [(0.0, chunk) for chunk in self.chunks]
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk, vec, dn in zip(self.chunks, self._doc_vectors, self._doc_norms):
            common = set(qv).intersection(vec)
            if not common:
                scored.append((0.0, chunk))
                continue
            dot = sum(qv[token] * vec[token] for token in common)
            scored.append((dot / (qn * dn), chunk))
        scored.sort(key=lambda item: -item[0])
        return scored


# ===========================================================================
# 6. EmbeddingRetriever (dense)
# ===========================================================================


@dataclass(slots=True)
class EmbeddingClientConfig:
    api_base: str
    api_key: str
    model: str
    request_timeout: float = 60.0
    batch_size: int = 32


@dataclass(slots=True)
class EmbeddingCache:
    """Disk-backed cache so we never re-embed a chunk we've seen."""

    cache_dir: Path

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, *, model_id: str, text: str) -> str:
        digest = hashlib.sha256((model_id + "\x00" + text).encode("utf-8")).hexdigest()
        return digest[:32]

    def get(self, *, model_id: str, text: str) -> list[float] | None:
        path = self.cache_dir / f"{self._key(model_id=model_id, text=text)}.json"
        if not path.exists():
            return None
        try:
            return list(json.loads(path.read_text())["embedding"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    def put(self, *, model_id: str, text: str, embedding: Sequence[float]) -> None:
        path = self.cache_dir / f"{self._key(model_id=model_id, text=text)}.json"
        try:
            path.write_text(json.dumps({"embedding": list(embedding)}))
        except OSError:
            pass


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


@dataclass(slots=True)
class EmbeddingRetriever:
    chunks: list[DocumentChunk]
    config: EmbeddingClientConfig
    cache: EmbeddingCache | None = None
    _chunk_embeddings: list[list[float]] = field(default_factory=list)
    _ready: bool = False

    def _model_id(self) -> str:
        return f"{self.config.api_base}::{self.config.model}"

    def _client(self):
        from openai import OpenAI
        return OpenAI(
            api_key=self.config.api_key or "EMPTY",
            base_url=self.config.api_base.rstrip("/"),
            timeout=self.config.request_timeout,
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        response = client.embeddings.create(
            model=self.config.model,
            input=texts,
            encoding_format="float",
        )
        return [list(item.embedding) for item in response.data]

    def _embed_with_cache(self, texts: list[str]) -> list[list[float]]:
        model_id = self._model_id()
        out: list[list[float] | None] = [None] * len(texts)
        misses_idx: list[int] = []
        if self.cache is not None:
            for i, text in enumerate(texts):
                cached = self.cache.get(model_id=model_id, text=text)
                if cached is not None:
                    out[i] = cached
                else:
                    misses_idx.append(i)
        else:
            misses_idx = list(range(len(texts)))

        for batch_start in range(0, len(misses_idx), self.config.batch_size):
            batch_indices = misses_idx[batch_start: batch_start + self.config.batch_size]
            batch_texts = [texts[i] for i in batch_indices]
            embeddings = self._embed_batch(batch_texts)
            for slot_idx, embedding in zip(batch_indices, embeddings):
                out[slot_idx] = embedding
                if self.cache is not None:
                    self.cache.put(model_id=model_id, text=texts[slot_idx], embedding=embedding)
        return [item if item is not None else [] for item in out]

    def prepare(self) -> None:
        if self._ready:
            return
        if not self.chunks:
            self._ready = True
            return
        texts = [chunk.indexable_text for chunk in self.chunks]
        self._chunk_embeddings = self._embed_with_cache(texts)
        self._ready = True

    def score(self, query: str) -> list[tuple[float, DocumentChunk]]:
        self.prepare()
        if not self.chunks:
            return []
        query_emb = self._embed_with_cache([query])[0]
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk, emb in zip(self.chunks, self._chunk_embeddings):
            scored.append((_cosine(query_emb, emb), chunk))
        scored.sort(key=lambda item: -item[0])
        return scored


# ===========================================================================
# 7. HybridRetriever — Reciprocal Rank Fusion
# ===========================================================================


def reciprocal_rank_fusion(
    rankings: list[list[tuple[float, DocumentChunk]]],
    *,
    rrf_k: int = 60,
) -> list[tuple[float, DocumentChunk]]:
    """Combine multiple retriever rankings into one.

    Each retriever produces a sorted ``[(score, chunk)]``. RRF gives each
    chunk a score = sum_over_retrievers(1 / (rrf_k + rank)) where ``rank``
    is 1-indexed from that retriever. RRF only uses *ranks*, never raw
    scores — this is what makes fusing BM25 (raw scores in the tens) with
    cosine (raw scores in [-1, 1]) work without normalization.

    Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods", SIGIR 2009.
    """
    fused_score: dict[str, float] = {}
    chunk_by_id: dict[str, DocumentChunk] = {}

    def _identity(chunk: DocumentChunk) -> str:
        return f"{chunk.source_path}#{chunk.content_hash}"

    for ranking in rankings:
        for rank, (_, chunk) in enumerate(ranking, start=1):
            cid = _identity(chunk)
            chunk_by_id[cid] = chunk
            fused_score[cid] = fused_score.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    fused = [(score, chunk_by_id[cid]) for cid, score in fused_score.items()]
    fused.sort(key=lambda item: -item[0])
    return fused


@dataclass(slots=True)
class HybridRetriever:
    """Run multiple retrievers in parallel, fuse with RRF."""

    retrievers: list[Retriever]
    rrf_k: int = 60

    @property
    def chunks(self) -> list[DocumentChunk]:
        return self.retrievers[0].chunks if self.retrievers else []

    def score(self, query: str) -> list[tuple[float, DocumentChunk]]:
        if not self.retrievers:
            return []
        rankings = [retriever.score(query) for retriever in self.retrievers]
        return reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k)


# ===========================================================================
# 8. Query transformation (multi-query + HyDE)
# ===========================================================================


class QueryTransformer(Protocol):
    """Turn one question into a list of queries we want to retrieve for."""

    def transform(self, query: str) -> list[str]: ...


class IdentityQueryTransformer:
    def transform(self, query: str) -> list[str]:
        return [query]


@dataclass(slots=True)
class LlmQueryExpander:
    """Use an LLM to generate paraphrases + (optional) a HyDE answer.

    Produces, in order:
      1. The original query.
      2. ``num_paraphrases`` alternative phrasings.
      3. (if ``include_hyde``) one hypothetical answer paragraph.

    Each output is fed to the retriever separately and the rankings are
    fused via RRF in the pipeline. Calling the LLM here costs one extra
    request per question, which is cheap relative to the synthesis call.
    """

    model: Any  # OpenAIModelAdapter or compatible (has .complete(messages))
    num_paraphrases: int = 2
    include_hyde: bool = True
    max_extra_queries: int = 4

    _SYSTEM = (
        "You rewrite a user's data-analysis question into multiple search "
        "queries that surface different aspects, plus (optionally) a short "
        "hypothetical answer paragraph that uses domain vocabulary likely "
        "to appear in the source documents.\n"
        "Output STRICT JSON with two keys: \"paraphrases\" (list of strings) "
        "and \"hypothetical_answer\" (string, may be empty). Do not output "
        "anything else."
    )

    def transform(self, query: str) -> list[str]:
        from data_agent_baseline.agents.model import ModelMessage  # lazy

        user = (
            f"Question: {query}\n\n"
            f"Generate {self.num_paraphrases} alternative phrasings (different word "
            f"choices, but same intent). Then write a 1–3 sentence hypothetical "
            f"answer to the question using likely-source vocabulary. Output JSON "
            f"with keys 'paraphrases' (list) and 'hypothetical_answer' (string)."
        )
        try:
            raw = self.model.complete(
                [
                    ModelMessage(role="system", content=self._SYSTEM),
                    ModelMessage(role="user", content=user),
                ],
                temperature=0.4,
            )
        except Exception as exc:  # noqa: BLE001 — never fail the task on expansion
            from data_agent_baseline.budget import BudgetExceeded

            if isinstance(exc, BudgetExceeded):
                raise
            return [query]

        paraphrases, hypothetical = self._parse(raw)
        out: list[str] = [query]
        out.extend(paraphrases[: self.num_paraphrases])
        if self.include_hyde and hypothetical:
            out.append(hypothetical)
        return out[: self.max_extra_queries + 1]

    @staticmethod
    def _parse(raw: str) -> tuple[list[str], str]:
        text = raw.strip()
        # Strip ```json fences if present.
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        try:
            payload, _ = json.JSONDecoder().raw_decode(text)
        except ValueError:
            return [], ""
        paraphrases = payload.get("paraphrases") or []
        if not isinstance(paraphrases, list):
            paraphrases = []
        paraphrases = [str(item).strip() for item in paraphrases if str(item).strip()]
        hypothetical = str(payload.get("hypothetical_answer") or "").strip()
        return paraphrases, hypothetical


# ===========================================================================
# 9. Cross-encoder reranker (optional second stage)
# ===========================================================================


@dataclass(slots=True)
class RerankerConfig:
    api_base: str
    api_key: str
    model: str
    request_timeout: float = 60.0
    # Some providers expose rerank under /v1/rerank, others under /rerank;
    # we let the user pick the path explicitly.
    endpoint_path: str = "/rerank"


class CrossEncoderReranker:
    """Calls a remote cross-encoder reranker endpoint.

    Compatible with:
      * DashScope ``gte-rerank`` (POST {api_base}/services/rerank/text-rerank/text-rerank)
      * Cohere /v1/rerank (model: ``rerank-multilingual-v3.0``)
      * Locally-hosted BGE-reranker via vLLM /v1/rerank
      * Any service that accepts ``{model, query, documents: [str]}`` and
        returns ``{results: [{index, relevance_score}]}``.

    If the endpoint shape doesn't match the common one, override
    :meth:`_call_endpoint` in a subclass.
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config

    def _call_endpoint(self, *, query: str, documents: list[str]) -> list[float]:
        import httpx

        url = self.config.api_base.rstrip("/") + self.config.endpoint_path
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "query": query,
            "documents": documents,
        }
        with httpx.Client(timeout=self.config.request_timeout) as client:
            response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Try the two most common response shapes.
        results = data.get("results")
        if results is None and "data" in data:
            # DashScope variant.
            output = data.get("output") or {}
            results = output.get("results") or data.get("data")
        if results is None:
            return [0.0] * len(documents)

        scored: list[tuple[int, float]] = []
        for item in results:
            idx = int(item.get("index", item.get("document_index", -1)))
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            scored.append((idx, score))
        # Build a same-order score list.
        score_by_idx = {idx: score for idx, score in scored}
        return [score_by_idx.get(i, 0.0) for i in range(len(documents))]

    def rerank(
        self,
        *,
        query: str,
        candidates: list[tuple[float, DocumentChunk]],
    ) -> list[tuple[float, DocumentChunk]]:
        if not candidates:
            return []
        documents = [chunk.indexable_text for _, chunk in candidates]
        try:
            scores = self._call_endpoint(query=query, documents=documents)
        except Exception:  # noqa: BLE001 — fall back to first-stage order
            return candidates
        if len(scores) != len(candidates):
            return candidates
        merged = [(score, chunk) for score, (_, chunk) in zip(scores, candidates)]
        merged.sort(key=lambda item: -item[0])
        return merged


# ===========================================================================
# 10. Pipeline orchestrator
# ===========================================================================


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[DocumentChunk]
    scores: list[float]
    backend: str
    queried_chunks_total: int
    rendered: str
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "queried_chunks_total": self.queried_chunks_total,
            "selected_chunks": [
                {
                    "source": chunk.source_path,
                    "heading": chunk.heading_label,
                    "score": round(score, 4),
                    "char_count": len(chunk.raw_text),
                }
                for chunk, score in zip(self.chunks, self.scores)
            ],
            "debug": self.debug,
        }


def _render(result: RetrievalResult) -> str:
    lines: list[str] = []
    last_source: str | None = None
    for chunk, score in zip(result.chunks, result.scores):
        if chunk.source_path != last_source:
            lines.append(f"## {chunk.source_path}")
            last_source = chunk.source_path
        lines.append(f"### {chunk.heading_label}  _(score={score:.3f})_")
        lines.append(chunk.raw_text)
    return "\n\n".join(lines)


@dataclass(slots=True)
class RetrievalPipeline:
    """Orchestrates: query expansion → multi-query retrieval → RRF → rerank → top-K."""

    retriever: Retriever
    query_transformer: QueryTransformer | None = None
    reranker: CrossEncoderReranker | None = None
    first_stage_top_n: int = 30      # how many candidates to feed the reranker
    final_top_k: int = 6
    rrf_k: int = 60

    def retrieve(self, *, question: str) -> RetrievalResult:
        chunks_total = len(self.retriever.chunks)

        # --- query transformation
        transformer = self.query_transformer or IdentityQueryTransformer()
        queries = transformer.transform(question)

        # --- multi-query retrieval + RRF
        rankings = [self.retriever.score(q) for q in queries]
        if len(rankings) == 1:
            fused = rankings[0]
        else:
            fused = reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k)

        # --- first-stage candidate cap
        candidates = fused[: self.first_stage_top_n]

        # --- reranking (optional)
        if self.reranker is not None and candidates:
            reranked = self.reranker.rerank(query=question, candidates=candidates)
        else:
            reranked = candidates

        selected = reranked[: self.final_top_k]
        backend = self._backend_label()

        result = RetrievalResult(
            chunks=[chunk for _, chunk in selected],
            scores=[float(score) for score, _ in selected],
            backend=backend,
            queried_chunks_total=chunks_total,
            rendered="",
            debug={
                "expanded_queries": queries,
                "first_stage_size": len(candidates),
                "reranked": self.reranker is not None,
            },
        )
        result.rendered = _render(result)
        return result

    def _backend_label(self) -> str:
        if isinstance(self.retriever, HybridRetriever):
            inner = "+".join(type(r).__name__ for r in self.retriever.retrievers)
            base = f"hybrid({inner})"
        else:
            base = type(self.retriever).__name__

        suffix: list[str] = []
        if isinstance(self.query_transformer, LlmQueryExpander):
            suffix.append("expand")
            if self.query_transformer.include_hyde:
                suffix.append("hyde")
        if self.reranker is not None:
            suffix.append(f"rerank({self.reranker.config.model})")
        if not suffix:
            return base
        return f"{base}|{'+'.join(suffix)}"


# ===========================================================================
# 11. Public entry point — what context_render calls
# ===========================================================================


def _build_chunks(
    *,
    document_paths: Iterable[Path],
    max_chunk_chars: int,
    chunk_overlap_chars: int,
) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for path in document_paths:
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            all_chunks.extend(chunk_json_file(
                path=path,
                max_chunk_chars=max_chunk_chars,
            ))
            continue
        try:
            text = load_document_text(path)
        except Exception as exc:  # noqa: BLE001
            all_chunks.append(DocumentChunk(
                source_path=str(path),
                heading_path=("<load_error>",),
                raw_text=f"(could not load {path}: {exc})",
            ))
            continue
        chunks = chunk_document(
            text=text,
            source_path=str(path),
            max_chunk_chars=max_chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )
        all_chunks.extend(chunks)
    return all_chunks


def _truncate_to_budget(
    selected: list[tuple[float, DocumentChunk]],
    *,
    char_budget: int | None,
) -> list[tuple[float, DocumentChunk]]:
    if char_budget is None:
        return selected
    out: list[tuple[float, DocumentChunk]] = []
    used = 0
    for score, chunk in selected:
        if out and used + len(chunk.raw_text) > char_budget:
            break
        out.append((score, chunk))
        used += len(chunk.raw_text)
    return out


def retrieve_top_k(
    *,
    document_paths: Iterable[Path],
    question: str,
    k: int,
    max_chunk_chars: int = 1200,
    chunk_overlap_chars: int = 150,
    embedding_config: EmbeddingClientConfig | None = None,
    cache_dir: Path | None = None,
    char_budget: int | None = None,
    # Advanced knobs (default off so existing callers still work):
    use_hybrid: bool = True,
    query_expansion_model: Any | None = None,
    query_expansion_paraphrases: int = 2,
    query_expansion_include_hyde: bool = True,
    reranker_config: RerankerConfig | None = None,
    first_stage_top_n: int = 30,
    rrf_k: int = 60,
) -> RetrievalResult:
    """End-to-end RAG: load → chunk → expand → retrieve+fuse → rerank → top-K.

    Behavior matrix (the relevant flag wins):

    - ``embedding_config=None`` and ``use_hybrid=False`` → BM25 only.
    - ``embedding_config!=None`` and ``use_hybrid=False`` → embedding only.
    - ``embedding_config!=None`` and ``use_hybrid=True``  → hybrid (BM25 + embedding via RRF).
    - ``embedding_config=None`` and ``use_hybrid=True``   → BM25 only (no dense retriever to fuse with).
    - ``query_expansion_model!=None`` adds LLM paraphrase + HyDE queries.
    - ``reranker_config!=None`` adds a cross-encoder rerank stage.
    """
    chunks = _build_chunks(
        document_paths=document_paths,
        max_chunk_chars=max_chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )
    if not chunks:
        return RetrievalResult(chunks=[], scores=[], backend="empty", queried_chunks_total=0, rendered="")

    # --- assemble retriever. Dense embedding is useful, but it must never
    # make the whole task crash: provider/model compatibility mistakes
    # should degrade to BM25-only sparse retrieval.
    sparse: Retriever = Bm25Retriever(chunks=chunks)
    dense: Retriever | None = None
    dense_error: str | None = None
    if embedding_config is not None:
        try:
            cache_obj = EmbeddingCache(cache_dir=cache_dir) if cache_dir is not None else None
            candidate_dense = EmbeddingRetriever(
                chunks=chunks,
                config=embedding_config,
                cache=cache_obj,
            )
            # Fail fast here so unsupported embedding models fall back
            # before the retrieval pipeline is assembled.
            candidate_dense.prepare()
            dense = candidate_dense
        except Exception as exc:  # noqa: BLE001
            dense_error = str(exc)
            dense = None

    if use_hybrid and dense is not None:
        retriever: Retriever = HybridRetriever(retrievers=[sparse, dense], rrf_k=rrf_k)
    elif dense is not None:
        retriever = dense
    else:
        retriever = sparse

    # --- query transformer
    transformer: QueryTransformer | None = None
    if query_expansion_model is not None:
        transformer = LlmQueryExpander(
            model=query_expansion_model,
            num_paraphrases=query_expansion_paraphrases,
            include_hyde=query_expansion_include_hyde,
        )

    # --- reranker
    reranker = CrossEncoderReranker(reranker_config) if reranker_config is not None else None

    pipeline = RetrievalPipeline(
        retriever=retriever,
        query_transformer=transformer,
        reranker=reranker,
        first_stage_top_n=first_stage_top_n,
        final_top_k=k,
        rrf_k=rrf_k,
    )
    try:
        result = pipeline.retrieve(question=question)
    except Exception as exc:  # noqa: BLE001
        if dense is None:
            raise
        dense_error = dense_error or str(exc)
        pipeline = RetrievalPipeline(
            retriever=sparse,
            query_transformer=transformer,
            reranker=reranker,
            first_stage_top_n=first_stage_top_n,
            final_top_k=k,
            rrf_k=rrf_k,
        )
        result = pipeline.retrieve(question=question)
    if dense_error:
        result.debug["dense_fallback_error"] = dense_error
        result.backend = f"{result.backend}|dense_fallback"

    if char_budget is not None:
        trimmed = _truncate_to_budget(
            list(zip(result.scores, result.chunks)),
            char_budget=char_budget,
        )
        result.chunks = [chunk for _, chunk in trimmed]
        result.scores = [score for score, _ in trimmed]
        result.rendered = _render(result)
    return result
