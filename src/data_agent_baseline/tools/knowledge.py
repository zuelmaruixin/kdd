"""Knowledge consultation tool backed by a separate helper LLM.

The React agent's main model is often a mid-sized tier (qwen-plus). For
tasks where the canonical knowledge.md / rules.md needs careful natural-
language interpretation (e.g. translating an ambiguous business rule
into the right pandas filter), a stronger model can produce a more
accurate reading than the loop model would.

``consult_knowledge`` reads the knowledge / rule / glossary docs in
context, posts them + the user's question to a configurable helper
model, and returns a short answer. The React loop sees the answer as a
regular observation and can cite or paraphrase it.

The helper is OPTIONAL — when disabled in config, the tool is not
registered. When enabled but a helper model isn't configured, calls
fall back to the loop model.

Per-task call budget and per-question cache live on a singleton
``HelperRuntime`` set by the React harness for the duration of one task.
This mirrors the project's existing budget/progress singleton pattern.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import HelperModelConfig

if TYPE_CHECKING:
    from data_agent_baseline.agents.model import ModelAdapter


_KNOWLEDGE_FILE_RE = re.compile(
    r"(?:^|[/\\_-])(knowledge|rule|rules|definition|definitions|glossary|schema_notes)"
    r"\b[^/\\]*\.(md|markdown|txt)$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class HelperRuntime:
    """Per-task state for the helper-model knowledge tool.

    ``calls_remaining`` enforces the per-task budget; ``cache`` short-
    circuits repeated identical questions in the same task.
    """

    config: HelperModelConfig
    adapter: "ModelAdapter"
    calls_remaining: int
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def remaining(self) -> int:
        return self.calls_remaining


_HELPER: HelperRuntime | None = None


def set_helper_runtime(runtime: HelperRuntime | None) -> None:
    global _HELPER
    _HELPER = runtime


def get_helper_runtime() -> HelperRuntime | None:
    return _HELPER


def _list_knowledge_files(task: PublicTask) -> list[Path]:
    try:
        root = Path(task.context_dir)
    except Exception:  # noqa: BLE001
        return []
    if not root.exists() or not root.is_dir():
        return []
    matches: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        if _KNOWLEDGE_FILE_RE.search(rel):
            matches.append(path)
    return matches


def _resolve_files(task: PublicTask, files: list[str] | None) -> list[Path]:
    if not files:
        return _list_knowledge_files(task)
    root = Path(task.context_dir).resolve()
    resolved: list[Path] = []
    for raw in files:
        candidate = (Path(task.context_dir) / str(raw)).resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"consult_knowledge path escapes context: {raw}")
        if not candidate.exists():
            raise FileNotFoundError(f"consult_knowledge missing file: {raw}")
        resolved.append(candidate)
    return resolved


def _bundle_docs(paths: list[Path], *, max_chars: int) -> tuple[str, list[str]]:
    """Concatenate doc bodies (with file headers), truncated to ``max_chars``."""
    fragments: list[str] = []
    used: list[str] = []
    budget = max_chars
    for path in paths:
        if budget <= 0:
            break
        try:
            body = path.read_text(errors="replace")
        except Exception as exc:  # noqa: BLE001
            body = f"(failed to read: {exc})"
        snippet = body[:budget]
        fragments.append(f"--- {path.name} ---\n{snippet}")
        used.append(path.name)
        budget -= len(snippet)
    return "\n\n".join(fragments), used


def _cache_key(question: str, file_names: list[str]) -> str:
    payload = f"{question.strip()}::{'|'.join(sorted(file_names))}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


_SYSTEM_PROMPT = (
    "You are a domain knowledge consultant. The user is solving a "
    "data-analysis question against a dataset whose specialized "
    "definitions live in the attached knowledge / rule / glossary "
    "files. Read those files and answer the user's question as a "
    "short, direct, factual paragraph.\n"
    "\n"
    "Rules:\n"
    "1. Use ONLY information present in the attached files. If the "
    "answer is not stated, say `not_specified` explicitly.\n"
    "2. Do NOT invent thresholds, formulas, or cutoffs.\n"
    "3. Quote the relevant rule verbatim when helpful.\n"
    "4. Apply the rule semantically — describe which column / field / "
    "filter it corresponds to in plain language, but do not write code.\n"
    "5. Keep the answer under ~200 words. The caller will translate it "
    "into code."
)


def consult_knowledge(
    task: PublicTask,
    *,
    question: str,
    files: list[str] | None = None,
) -> dict[str, Any]:
    """Ask the helper LLM a focused question about the task's knowledge docs.

    Raises ``RuntimeError`` when no helper runtime is registered (i.e.
    the tool is enabled in config but the React harness forgot to wire
    it up — caller-side bug, not user error).
    """
    runtime = get_helper_runtime()
    if runtime is None:
        raise RuntimeError(
            "consult_knowledge is enabled but no helper runtime is "
            "registered. Verify agent.helper_model.enabled is true and "
            "the harness installed a helper adapter."
        )

    if not isinstance(question, str) or not question.strip():
        raise ValueError("consult_knowledge requires a non-empty question.")

    paths = _resolve_files(task, files)
    if not paths:
        return {
            "ok": False,
            "answer": None,
            "files_used": [],
            "error": (
                "No knowledge / rule / glossary docs were found in "
                "context. consult_knowledge needs a doc to read."
            ),
        }

    doc_text, file_names = _bundle_docs(paths, max_chars=runtime.config.max_doc_chars)
    key = _cache_key(question, file_names)
    cached = runtime.cache.get(key)
    if cached is not None:
        return {**cached, "cached": True}

    if runtime.calls_remaining <= 0:
        return {
            "ok": False,
            "answer": None,
            "files_used": file_names,
            "error": (
                "consult_knowledge call budget exhausted for this task; "
                "rely on read_doc / grep_doc to inspect the doc directly."
            ),
        }

    from data_agent_baseline.agents.model import ModelMessage
    user_content = (
        f"Question: {question.strip()}\n\n"
        "Knowledge documents:\n"
        f"{doc_text}"
    )
    messages = [
        ModelMessage(role="system", content=_SYSTEM_PROMPT),
        ModelMessage(role="user", content=user_content),
    ]
    try:
        raw = runtime.adapter.complete(
            messages,
            temperature=runtime.config.temperature,
            max_tokens=runtime.config.max_tokens,
            stream_label="react.helper",
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "answer": None,
            "files_used": file_names,
            "error": f"helper_model_call_failed: {exc}",
        }
    runtime.calls_remaining -= 1
    payload: dict[str, Any] = {
        "ok": True,
        "answer": raw.strip(),
        "files_used": file_names,
        "calls_remaining": runtime.calls_remaining,
    }
    runtime.cache[key] = payload
    return payload
