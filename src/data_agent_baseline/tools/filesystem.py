from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


# Default per-observation char budget for big files. This keeps the
# observation token count predictable so a single read_doc on a 200KB
# markdown doesn't blow the React context window. Callers (the model)
# can override via chunk_size, but never above this hard cap.
_MAX_OBSERVATION_CHARS: int = 6000


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(
    task: PublicTask,
    relative_path: str,
    *,
    max_rows: int = 20,
    offset: int = 0,
    columns_only: bool = False,
) -> dict[str, object]:
    """Preview a CSV file.

    ``columns_only=True`` skips the data entirely — returns just the
    header plus the total row count. Use this on large CSVs to confirm
    schema before issuing an execute_python call.

    ``offset`` skips that many DATA rows (after the header). Combined
    with ``max_rows`` it lets the model page through a CSV.
    """
    path = resolve_context_path(task, relative_path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return {
            "path": relative_path,
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    header = rows[0]
    data_rows = rows[1:]
    total = len(data_rows)

    if columns_only:
        return {
            "path": relative_path,
            "columns": header,
            "rows": [],
            "row_count": total,
            "columns_only": True,
        }

    offset = max(0, int(offset))
    sliced = data_rows[offset : offset + max_rows]
    return {
        "path": relative_path,
        "columns": header,
        "rows": sliced,
        "row_count": total,
        "offset": offset,
        "returned_rows": len(sliced),
        "has_more": offset + len(sliced) < total,
    }


def read_json_preview(
    task: PublicTask,
    relative_path: str,
    *,
    max_chars: int = 4000,
    offset: int = 0,
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    payload = json.loads(path.read_text())
    preview_full = json.dumps(payload, ensure_ascii=False, indent=2)
    max_chars = min(max(1, int(max_chars)), _MAX_OBSERVATION_CHARS)
    offset = max(0, int(offset))
    chunk = preview_full[offset : offset + max_chars]
    return {
        "path": relative_path,
        "preview": chunk,
        "offset": offset,
        "chunk_chars": len(chunk),
        "total_chars": len(preview_full),
        "has_more": offset + len(chunk) < len(preview_full),
        "truncated": offset + len(chunk) < len(preview_full),
    }


def read_doc_preview(
    task: PublicTask,
    relative_path: str,
    *,
    max_chars: int = 4000,
    offset: int = 0,
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    max_chars = min(max(1, int(max_chars)), _MAX_OBSERVATION_CHARS)
    offset = max(0, int(offset))
    chunk = text[offset : offset + max_chars]
    return {
        "path": relative_path,
        "preview": chunk,
        "offset": offset,
        "chunk_chars": len(chunk),
        "total_chars": len(text),
        "has_more": offset + len(chunk) < len(text),
        "truncated": offset + len(chunk) < len(text),
    }


def head_doc(
    task: PublicTask,
    relative_path: str,
    *,
    max_lines: int = 40,
) -> dict[str, object]:
    """First N lines of a text document — useful for quick log / csv peek.

    Capped at ``_MAX_OBSERVATION_CHARS`` of content even if max_lines
    allows more, so a single hot line doesn't blow the budget.
    """
    path = resolve_context_path(task, relative_path)
    max_lines = max(1, int(max_lines))
    head_chunks: list[str] = []
    chars_so_far = 0
    with path.open(errors="replace") as handle:
        for index, line in enumerate(handle):
            if index >= max_lines:
                break
            head_chunks.append(line)
            chars_so_far += len(line)
            if chars_so_far >= _MAX_OBSERVATION_CHARS:
                break
    head_text = "".join(head_chunks)
    total_lines = sum(1 for _ in path.open(errors="replace"))
    return {
        "path": relative_path,
        "lines": head_text.splitlines(),
        "returned_lines": len(head_chunks),
        "total_lines": total_lines,
        "char_count": len(head_text),
    }


def grep_doc(
    task: PublicTask,
    relative_path: str,
    *,
    pattern: str,
    max_hits: int = 20,
    context_lines: int = 1,
    case_insensitive: bool = True,
) -> dict[str, object]:
    """Regex / substring search in a text document.

    Returns up to ``max_hits`` matching line indexes with ``context_lines``
    of surrounding context, so the model can locate a rule inside
    knowledge.md without reading the whole file.
    """
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("grep_doc requires a non-empty pattern.")

    path = resolve_context_path(task, relative_path)
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc

    text = path.read_text(errors="replace")
    lines = text.splitlines()
    hits: list[dict[str, object]] = []
    chars_used = 0
    max_hits = max(1, int(max_hits))
    context_lines = max(0, int(context_lines))

    for line_index, line in enumerate(lines):
        if regex.search(line) is None:
            continue
        start = max(0, line_index - context_lines)
        end = min(len(lines), line_index + context_lines + 1)
        snippet_lines = lines[start:end]
        snippet = "\n".join(snippet_lines)
        chars_used += len(snippet) + 1
        hits.append(
            {
                "line_number": line_index + 1,
                "match_line": line,
                "context_start_line": start + 1,
                "context": snippet,
            }
        )
        if len(hits) >= max_hits:
            break
        if chars_used >= _MAX_OBSERVATION_CHARS:
            break

    return {
        "path": relative_path,
        "pattern": pattern,
        "hits": hits,
        "hit_count": len(hits),
        "total_lines": len(lines),
        "truncated": len(hits) >= max_hits or chars_used >= _MAX_OBSERVATION_CHARS,
    }
