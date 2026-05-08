"""Smart context rendering for prompts.

Replaces the naive "read everything and truncate at 12K chars" strategy
with a layered approach that preserves the *information* the agent needs
while staying within a token budget.

Three render modes:

1. ``render_compact_schema`` — for the planner. Just file inventory +
   schemas + row counts; *no* data rows. The planner only needs to know
   what's available, not the contents.
2. ``render_with_samples`` — for ``TableLLMDirectAgent``. Schema + a
   diverse handful of rows per table (head + tail + a couple of
   spread-out middles), so the LLM can write code that types-matches
   without needing the full data.
3. ``render_focused`` — for long-doc tasks. Schema + question-keyword
   matched excerpts from each Markdown / text document, with a small
   header context window around each match.

All three respect a global character budget and degrade gracefully:
once the budget is exhausted, remaining files get a one-line
schema-only stub instead of being silently dropped.

The renderer is intentionally LLM-free (cheap, deterministic, no extra
API calls). It is not a RAG embedder; for 128K+ doc tasks, you should
plug an embedding-based retriever in front of ``render_focused``.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from data_agent_baseline.benchmark.schema import PublicTask


# ---------------------------------------------------------------------------
# Manifests + budget bookkeeping
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FileSummary:
    relative_path: str
    kind: str                 # csv | db | json | record_text | doc | other
    size_bytes: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            **self.detail,
        }


@dataclass(slots=True)
class RenderManifest:
    rendered: str
    files: list[FileSummary] = field(default_factory=list)
    truncated: bool = False
    char_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "char_count": self.char_count,
            "truncated": self.truncated,
            "files": [item.to_dict() for item in self.files],
        }


# ---------------------------------------------------------------------------
# File classification + walking
# ---------------------------------------------------------------------------


def _classify(path: Path) -> str:
    from data_agent_baseline.agents.task_compiler import classify_context_kind

    kind = classify_context_kind(path)
    if kind == "record_text":
        return kind
    if kind in {"csv", "tsv"}:
        return "csv"
    if kind in {"db", "sqlite", "sqlite3"}:
        return "db"
    if kind in {"json", "jsonl"}:
        return "json"
    if kind in {"md", "txt", "docx", "pdf"}:
        return "doc"
    if kind in {"png", "jpg", "jpeg", "webp"}:
        return "image"
    return "other"


def _iter_context_files(task: PublicTask) -> list[Path]:
    return sorted(
        (
            path
            for path in task.context_dir.rglob("*")
            if path.is_file() and _should_render_context_file(path, root=task.context_dir)
        ),
        key=lambda p: (_kind_order(_classify(p)), p.name),
    )


def _should_render_context_file(path: Path, *, root: Path) -> bool:
    if path.name == ".DS_Store":
        return False
    if path.name.lower() in {
        "answer.csv",
        "prediction.csv",
        "operator_codegen_answer.csv",
        "operator_local_repair_answer.csv",
        "operator_retry_answer.csv",
        "repaired_answer.csv",
    }:
        return False
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    # Hidden paths are operating-system/editor/generated artifacts. Generated
    # `.synthesized` CSVs are exposed through SourceCapability after the
    # extractor runs, not by scanning stale files from a previous run.
    return not any(part.startswith(".") for part in rel.parts)


def _kind_order(kind: str) -> int:
    # csv / db are most useful for code-solution; doc is supporting; image last.
    return {
        "csv": 0,
        "db": 1,
        "json": 2,
        "record_text": 3,
        "doc": 4,
        "image": 5,
        "other": 6,
    }.get(kind, 9)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _csv_metadata(path: Path) -> tuple[list[str], int]:
    """Return (header, row_count). Read line-by-line to avoid loading huge files."""
    with path.open(newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        row_count = sum(1 for _ in reader)
    return list(header), row_count


def _csv_diverse_sample(path: Path, *, sample_count: int) -> list[list[str]]:
    """Return a small set of rows showing data variety: head, tail, and a couple spread-out picks."""
    with path.open(newline="", errors="replace") as handle:
        rows = list(csv.reader(handle))
    if len(rows) <= 1:
        return []
    body = rows[1:]
    if len(body) <= sample_count:
        return body
    if sample_count <= 1:
        return body[:1]
    indices: list[int] = []
    indices.append(0)
    for k in range(1, sample_count - 1):
        indices.append(int(round(k * (len(body) - 1) / (sample_count - 1))))
    indices.append(len(body) - 1)
    seen: set[int] = set()
    result: list[list[str]] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        result.append(body[idx])
    return result


def _render_csv_block(
    path: Path,
    rel: str,
    *,
    include_samples: bool,
    sample_count: int,
) -> tuple[str, dict[str, Any]]:
    header, row_count = _csv_metadata(path)
    detail: dict[str, Any] = {"columns": header, "row_count": row_count}
    if not header:
        return f"## {rel}\n_(empty csv)_\n", detail

    schema_line = "**columns**: " + ", ".join(header)
    block = f"## {rel}\n_csv · {row_count} rows · {len(header)} cols_\n{schema_line}\n"

    if include_samples:
        samples = _csv_diverse_sample(path, sample_count=sample_count)
        if samples:
            sep = "| " + " | ".join("---" for _ in header) + " |"
            head_line = "| " + " | ".join(str(cell) for cell in header) + " |"
            sample_lines = [
                "| " + " | ".join(str(cell) if idx < len(row) else "" for idx, cell in enumerate(row + [""] * (len(header) - len(row)))) + " |"
                for row in samples
            ]
            block += "\nsample rows (head + spread + tail):\n"
            block += "\n".join([head_line, sep, *sample_lines]) + "\n"
    return block, detail


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def _open_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _score_table_relevance(
    *, table_name: str, columns: list[str], question_tokens: set[str]
) -> int:
    if not question_tokens:
        return 0
    haystack_tokens: set[str] = set()
    for raw in (table_name, *columns):
        haystack_tokens.update(re.findall(r"[a-zA-Z0-9]+", raw.lower()))
    return len(haystack_tokens & question_tokens)


def _render_sqlite_block(
    path: Path,
    rel: str,
    *,
    include_samples: bool,
    sample_count: int,
    question_tokens: set[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    blocks: list[str] = [f"## {rel}\n_sqlite database_\n"]
    detail: dict[str, Any] = {"tables": []}
    try:
        conn = _open_ro(path)
    except sqlite3.OperationalError as exc:
        return f"## {rel}\n_(could not open: {exc})_\n", detail
    try:
        cur = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        all_tables = list(cur.fetchall())

        # Pre-fetch column lists so we can score relevance and reorder
        # tables before rendering. Without this the prompt budget is spent
        # alphabetically — irrelevant in a 30-table DB.
        ordered: list[tuple[str, str, list[str]]] = []
        for table_name, create_sql in all_tables:
            try:
                col_cur = conn.execute(f'PRAGMA table_info("{table_name}")')
                col_names = [row[1] for row in col_cur.fetchall()]
            except sqlite3.OperationalError:
                col_names = []
            ordered.append((table_name, create_sql, col_names))
        if question_tokens:
            ordered.sort(
                key=lambda item: (
                    -_score_table_relevance(
                        table_name=item[0],
                        columns=item[2],
                        question_tokens=question_tokens,
                    ),
                    item[0],
                ),
            )

        for table_name, create_sql, _precomputed_cols in ordered:
            try:
                col_cur = conn.execute(f'PRAGMA table_info("{table_name}")')
                col_info = col_cur.fetchall()  # (cid, name, type, notnull, dflt, pk)
                cols = [(row[1], row[2]) for row in col_info]
                count_row = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
                row_count = int(count_row[0]) if count_row else 0
            except sqlite3.OperationalError as exc:
                blocks.append(f"### {table_name}\n_(introspect failed: {exc})_")
                continue

            schema_line = "; ".join(f"{name}:{dtype}" for name, dtype in cols)
            block = (
                f"### table `{table_name}` ({row_count} rows)\n"
                f"```sql\n{create_sql.strip()}\n```\n"
                f"columns: {schema_line}\n"
            )
            detail["tables"].append({
                "table": table_name,
                "columns": [name for name, _ in cols],
                "row_count": row_count,
            })
            if include_samples and row_count > 0:
                limit = min(sample_count, row_count)
                try:
                    rows = conn.execute(
                        f'SELECT * FROM "{table_name}" LIMIT {limit}'
                    ).fetchall()
                    col_names = [name for name, _ in cols]
                    if col_names:
                        sep = "| " + " | ".join("---" for _ in col_names) + " |"
                        head = "| " + " | ".join(col_names) + " |"
                        sample_lines = [
                            "| " + " | ".join(str(cell) for cell in row) + " |"
                            for row in rows
                        ]
                        block += "sample rows:\n" + "\n".join([head, sep, *sample_lines]) + "\n"
                except sqlite3.OperationalError as exc:
                    block += f"_(sample failed: {exc})_\n"
            blocks.append(block)
    finally:
        conn.close()
    return "\n".join(blocks), detail


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _summarize_json(payload: Any, *, max_records: int) -> str:
    """Compact textual summary so we don't blast 50KB of JSON into the prompt."""
    if isinstance(payload, dict):
        keys = list(payload.keys())
        if "records" in payload and isinstance(payload["records"], list):
            records = payload["records"]
            preview = records[:max_records]
            sample_keys = sorted({k for r in preview if isinstance(r, dict) for k in r}) if preview else []
            text = (
                f"top-level keys: {keys}\n"
                f"records: {len(records)} items\n"
                f"sample item fields: {sample_keys}\n"
                f"first {len(preview)} record(s):\n"
            )
            return text + json.dumps(preview, ensure_ascii=False, indent=2)
        return "top-level keys: " + json.dumps(keys, ensure_ascii=False)
    if isinstance(payload, list):
        head = payload[:max_records]
        return f"list with {len(payload)} items\nfirst {len(head)}:\n" + json.dumps(head, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _render_json_block(
    path: Path,
    rel: str,
    *,
    include_samples: bool,
    sample_count: int,
    max_chars: int,
) -> tuple[str, dict[str, Any]]:
    detail: dict[str, Any] = {}
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return f"## {rel}\n_(could not parse json: {exc})_\n", detail

    if isinstance(payload, dict):
        detail["top_level_keys"] = list(payload.keys())
        if "records" in payload and isinstance(payload["records"], list):
            detail["record_count"] = len(payload["records"])

    body = _summarize_json(payload, max_records=sample_count if include_samples else 0)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n_(json summary truncated)_"

    block = f"## {rel}\n_json_\n```json\n{body}\n```\n"
    return block, detail


# ---------------------------------------------------------------------------
# Doc helpers (Markdown / plain text)
# ---------------------------------------------------------------------------


_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _split_doc_sections(text: str) -> list[tuple[str, str]]:
    """Split a Markdown doc into [(heading, body)] sections by top-level headers."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("(top)", text)]
    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))
    return sections


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _question_tokens(question: str) -> set[str]:
    raw = _TOKEN_RE.findall(question.lower())
    # Drop very short / too-common tokens.
    blocklist = {
        "the", "a", "an", "of", "is", "are", "was", "were", "and", "or", "in", "on",
        "for", "to", "by", "with", "what", "which", "who", "how", "list", "show",
        "give", "name", "names", "all", "this", "that", "these", "those", "from",
        "many", "much", "do", "does",
    }
    return {tok for tok in raw if len(tok) > 2 and tok not in blocklist}


def _score_section(section_text: str, tokens: set[str]) -> int:
    if not tokens:
        return 0
    section_lower = section_text.lower()
    return sum(1 for tok in tokens if tok in section_lower)


def _load_doc_text(path: Path) -> str:
    """Load .md/.txt/.docx as plain text (delegates to document_retriever)."""
    if path.suffix.lower() == ".docx":
        # Imported lazily so the rest of the renderer doesn't need python-docx.
        from data_agent_baseline.agents.document_retriever import load_document_text
        return load_document_text(path)
    return path.read_text(errors="replace")


def _render_doc_block(
    path: Path,
    rel: str,
    *,
    question_tokens: set[str] | None,
    section_count: int,
    max_chars_per_section: int,
) -> tuple[str, dict[str, Any]]:
    detail: dict[str, Any] = {}
    try:
        text = _load_doc_text(path)
    except Exception as exc:  # noqa: BLE001
        return f"## {rel}\n_(doc load failed: {exc})_\n", detail
    detail["chars"] = len(text)
    sections = _split_doc_sections(text)
    detail["section_count"] = len(sections)

    if question_tokens:
        scored = [
            (idx, heading, body, _score_section(body, question_tokens))
            for idx, (heading, body) in enumerate(sections)
        ]
        scored.sort(key=lambda item: (-item[3], item[0]))
        top = [item for item in scored if item[3] > 0][:section_count]
        if not top:
            top = scored[:section_count]
    else:
        top = [(i, h, b, 0) for i, (h, b) in enumerate(sections[:section_count])]

    blocks: list[str] = [f"## {rel}\n_doc · {len(text)} chars · {len(sections)} sections_\n"]
    for _, heading, body, score in top:
        chunk = body if len(body) <= max_chars_per_section else body[:max_chars_per_section] + "\n_(section truncated)_"
        marker = f" (matched score={score})" if score > 0 else ""
        blocks.append(f"### {heading}{marker}\n{chunk}\n")
    return "\n".join(blocks), detail


def _render_record_text_block(
    path: Path,
    rel: str,
    *,
    question_tokens: set[str] | None,
    section_count: int,
    max_chars_per_section: int,
) -> tuple[str, dict[str, Any]]:
    block, detail = _render_doc_block(
        path,
        rel,
        question_tokens=question_tokens,
        section_count=section_count,
        max_chars_per_section=max_chars_per_section,
    )
    block = block.replace(f"## {rel}\n_doc", f"## {rel}\n_record_text · unstructured narrative_", 1)
    detail["record_text"] = True
    detail["parse_hint"] = (
        "This prose/Markdown file is unstructured narrative with repeated record mentions. "
        "Load the full file or use record extraction; do not treat the preview as full data."
    )
    return block, detail


# ---------------------------------------------------------------------------
# Public render entry points
# ---------------------------------------------------------------------------


def render_compact_schema(
    task: PublicTask,
    *,
    max_chars: int = 8000,
) -> RenderManifest:
    """For the planner: file inventory + schemas + row counts, no data rows."""
    return _render(
        task,
        max_chars=max_chars,
        include_samples=False,
        samples_per_table=0,
        question=None,
        doc_sections=2,
        doc_chars=400,
        json_chars=600,
    )


def render_with_samples(
    task: PublicTask,
    *,
    max_chars: int = 12000,
    samples_per_table: int = 5,
) -> RenderManifest:
    """For the one-shot code-solution agent: schema + diverse samples per table."""
    return _render(
        task,
        max_chars=max_chars,
        include_samples=True,
        samples_per_table=samples_per_table,
        question=None,
        doc_sections=2,
        doc_chars=600,
        json_chars=1200,
    )


def render_focused(
    task: PublicTask,
    *,
    question: str,
    max_chars: int = 10000,
    samples_per_table: int = 4,
    doc_sections: int = 3,
) -> RenderManifest:
    """For long-doc / multi-source tasks: keyword-relevant doc sections."""
    return _render(
        task,
        max_chars=max_chars,
        include_samples=True,
        samples_per_table=samples_per_table,
        question=question,
        doc_sections=doc_sections,
        doc_chars=900,
        json_chars=1200,
    )


def render_with_rag(
    task: PublicTask,
    *,
    question: str,
    max_chars: int = 12000,
    samples_per_table: int = 4,
    rag_top_k: int = 6,
    rag_max_chunk_chars: int = 1200,
    rag_chunk_overlap_chars: int = 150,
    rag_doc_char_budget: int | None = 6000,
    embedding_config: Any | None = None,
    embedding_cache_dir: Path | None = None,
    use_hybrid: bool = True,
    query_expansion_model: Any | None = None,
    query_expansion_paraphrases: int = 2,
    query_expansion_include_hyde: bool = True,
    reranker_config: Any | None = None,
    first_stage_top_n: int = 30,
    rrf_k: int = 60,
) -> RenderManifest:
    """Render context with proper RAG for long documents.

    This is the caller-of-choice when context contains ``.docx`` or
    multi-thousand-character Markdown notes that exceed the prompt
    budget. Tables are rendered with diverse samples (as in
    ``render_with_samples``) but documents are routed through
    :func:`document_retriever.retrieve_top_k` and only the top-K most
    relevant chunks make it into the prompt.

    Pass ``embedding_config`` (an
    :class:`document_retriever.EmbeddingClientConfig`) to use a remote
    embeddings endpoint; omit it to use the offline TF-IDF fallback.
    """
    from data_agent_baseline.agents.document_retriever import retrieve_top_k

    files = _iter_context_files(task)
    table_files: list[Path] = []
    doc_files: list[Path] = []
    for path in files:
        kind = _classify(path)
        # JSON files are tabular at the surface but often contain deeply
        # nested business rules that flat schema preview misses entirely.
        # Route them through RAG along with the docs so the retriever can
        # surface the relevant subtree.
        if kind == "doc" or path.suffix.lower() in {".json", ".jsonl"}:
            doc_files.append(path)
        else:
            table_files.append(path)

    # Reserve budget: tables get the bulk; docs get rag_doc_char_budget if set.
    tables_budget = max_chars
    if rag_doc_char_budget is not None and doc_files:
        tables_budget = max(2000, max_chars - rag_doc_char_budget)

    # Build a faux PublicTask for table-only rendering using the existing
    # pipeline. Cheaper than refactoring _render to accept a file filter.
    manifest = RenderManifest(rendered="")
    chunks: list[str] = []
    used = 0
    json_budget = max(1000, tables_budget // 4)

    def _push(block: str, summary: FileSummary) -> None:
        nonlocal used
        if used + len(block) <= tables_budget:
            chunks.append(block)
            manifest.files.append(summary)
            used += len(block)
            return
        stub = f"## {summary.relative_path}\n_({summary.kind}, omitted to stay within budget)_\n"
        if used + len(stub) <= tables_budget:
            chunks.append(stub)
            manifest.files.append(summary)
            used += len(stub)
        manifest.truncated = True

    rag_question_tokens = _question_tokens(question) if question else None
    for path in table_files:
        rel = path.relative_to(task.context_dir).as_posix()
        kind = _classify(path)
        summary = FileSummary(relative_path=rel, kind=kind, size_bytes=path.stat().st_size)
        if kind == "csv":
            block, detail = _render_csv_block(
                path, rel,
                include_samples=True,
                sample_count=samples_per_table,
            )
        elif kind == "db":
            block, detail = _render_sqlite_block(
                path, rel,
                include_samples=True,
                sample_count=max(1, min(samples_per_table, 4)),
                question_tokens=rag_question_tokens,
            )
        elif kind == "json":
            block, detail = _render_json_block(
                path, rel,
                include_samples=True,
                sample_count=samples_per_table,
                max_chars=json_budget,
            )
        elif kind == "record_text":
            block, detail = _render_record_text_block(
                path, rel,
                question_tokens=rag_question_tokens,
                section_count=3,
                max_chars_per_section=900,
            )
        elif kind == "image":
            block, detail = f"## {rel}\n_image · {summary.size_bytes} bytes_\n", {}
        else:
            block, detail = f"## {rel}\n_{kind} · {summary.size_bytes} bytes_\n", {}
        summary.detail = detail
        _push(block, summary)

    # Now run RAG over the doc files (if any) and append the result.
    if doc_files:
        retrieval = retrieve_top_k(
            document_paths=doc_files,
            question=question,
            k=rag_top_k,
            max_chunk_chars=rag_max_chunk_chars,
            chunk_overlap_chars=rag_chunk_overlap_chars,
            embedding_config=embedding_config,
            cache_dir=embedding_cache_dir,
            use_hybrid=use_hybrid,
            query_expansion_model=query_expansion_model,
            query_expansion_paraphrases=query_expansion_paraphrases,
            query_expansion_include_hyde=query_expansion_include_hyde,
            reranker_config=reranker_config,
            first_stage_top_n=first_stage_top_n,
            rrf_k=rrf_k,
            char_budget=rag_doc_char_budget,
        )
        rag_block = (
            f"\n## (long-doc RAG · backend={retrieval.backend} · "
            f"chunks_total={retrieval.queried_chunks_total} · "
            f"selected={len(retrieval.chunks)})\n"
            f"{retrieval.rendered}\n"
        )
        if used + len(rag_block) <= max_chars:
            chunks.append(rag_block)
            used += len(rag_block)
        else:
            allowed = max_chars - used
            if allowed > 200:
                chunks.append(rag_block[:allowed] + "\n_(rag block truncated)_\n")
                used += allowed
                manifest.truncated = True
        # Add a manifest entry summarizing the retrieval.
        for path in doc_files:
            rel = path.relative_to(task.context_dir).as_posix()
            manifest.files.append(FileSummary(
                relative_path=rel,
                kind="doc",
                size_bytes=path.stat().st_size,
                detail={"rag": retrieval.to_dict()},
            ))

    manifest.rendered = ("\n".join(chunks)).rstrip() + "\n"
    manifest.char_count = used
    return manifest


def _render(
    task: PublicTask,
    *,
    max_chars: int,
    include_samples: bool,
    samples_per_table: int,
    question: str | None,
    doc_sections: int,
    doc_chars: int,
    json_chars: int,
) -> RenderManifest:
    files = _iter_context_files(task)
    manifest = RenderManifest(rendered="")
    chunks: list[str] = []
    used = 0
    question_tokens = _question_tokens(question) if question else None

    def _push(block: str, summary: FileSummary) -> bool:
        """Add a block if budget allows; otherwise downgrade to a 1-line stub."""
        nonlocal used
        if used + len(block) <= max_chars:
            chunks.append(block)
            manifest.files.append(summary)
            used += len(block)
            return True
        # Try a stub if the rich block doesn't fit.
        stub = f"## {summary.relative_path}\n_({summary.kind}, omitted to stay within budget)_\n"
        if used + len(stub) <= max_chars:
            chunks.append(stub)
            manifest.files.append(summary)
            used += len(stub)
        manifest.truncated = True
        return False

    for path in files:
        rel = path.relative_to(task.context_dir).as_posix()
        kind = _classify(path)
        summary = FileSummary(
            relative_path=rel,
            kind=kind,
            size_bytes=path.stat().st_size,
        )

        if kind == "csv":
            block, detail = _render_csv_block(
                path, rel,
                include_samples=include_samples,
                sample_count=samples_per_table,
            )
        elif kind == "db":
            block, detail = _render_sqlite_block(
                path, rel,
                include_samples=include_samples,
                sample_count=max(1, min(samples_per_table, 4)),
                question_tokens=question_tokens,
            )
        elif kind == "json":
            block, detail = _render_json_block(
                path, rel,
                include_samples=include_samples,
                sample_count=samples_per_table,
                max_chars=json_chars,
            )
        elif kind == "record_text":
            block, detail = _render_record_text_block(
                path,
                rel,
                question_tokens=question_tokens,
                section_count=doc_sections,
                max_chars_per_section=doc_chars,
            )
        elif kind == "doc":
            block, detail = _render_doc_block(
                path, rel,
                question_tokens=question_tokens,
                section_count=doc_sections,
                max_chars_per_section=doc_chars,
            )
        elif kind == "image":
            block, detail = f"## {rel}\n_image · {summary.size_bytes} bytes_\n", {}
        else:
            block, detail = f"## {rel}\n_{kind} · {summary.size_bytes} bytes_\n", {}

        summary.detail = detail
        _push(block, summary)

    manifest.rendered = ("\n".join(chunks)).rstrip() + "\n"
    manifest.char_count = used
    return manifest
