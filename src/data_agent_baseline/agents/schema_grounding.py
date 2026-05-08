"""Bind question-language concepts to real schema columns.

The single biggest source of wrong answers in tool-first codegen is
**column-name hallucination**: the LLM understands the question but
guesses a column that doesn't exist (e.g. ``examination_df['Thrombosis']``
when the table doesn't have that column).

This module uses the deterministic ``SourceCapability`` data that
``task_compiler.py`` already collects (column names + per-column
distinct-value samples + dtypes + cross-file foreign-key candidates),
and produces a compact "Schema Grounding" block that goes straight into
the codegen prompt.

It does three jobs:

1. ``extract_question_concepts`` — pulls noun-phrase-ish tokens out of
   the question, drops stop words and very short tokens.
2. ``match_concept`` — for each concept, scores every known column by:
     - Levenshtein-similarity of column name vs concept,
     - presence of the concept (or its plural / hyphenated form) in
       the column's distinct-value samples,
     - cardinality bonus (low-cardinality columns are usually the
       categorical ones we want for filters).
3. ``render_grounding_block`` — outputs a small Markdown block that
   the codegen prompt can paste verbatim.

The scoring is intentionally cheap and deterministic. It will not
perfectly resolve ambiguous concepts; it just shrinks the search
space for the LLM and gives it explicit value examples to verify
against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.task_compiler import CompiledTask, SourceCapability


# ---------------------------------------------------------------------------
# Question concept extraction
# ---------------------------------------------------------------------------


_STOPWORDS = frozenset({
    "the", "a", "an", "of", "is", "are", "was", "were", "and", "or", "in",
    "on", "for", "to", "by", "with", "with", "what", "which", "who", "how",
    "list", "show", "give", "name", "names", "all", "this", "that", "these",
    "those", "from", "many", "much", "do", "does", "did", "be", "as", "at",
    "it", "its", "their", "his", "her", "they", "them", "we", "you", "i",
    "me", "my", "our", "us", "your", "yours", "any", "some", "each", "every",
    "such", "than", "then", "so", "if", "but", "not", "no", "yes", "yet",
    "more", "less", "most", "least", "between", "amongst", "among", "have",
    "has", "had", "been", "being", "where", "when", "while", "during",
    "after", "before", "into", "onto", "out", "over", "under", "above",
    "below", "about", "around", "across", "through", "per", "via", "due",
    "find", "list", "calculate", "compute", "count", "tell", "specify",
    "given", "based", "using", "use", "please", "can", "could", "would",
    "should", "may", "might", "must", "will", "shall", "their",
})


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def extract_question_concepts(question: str, *, max_concepts: int = 16) -> list[str]:
    """Pull plausibly-content-bearing tokens out of a question."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _TOKEN_RE.findall(question):
        token = match.lower()
        if len(token) <= 2:
            continue
        if token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:max_concepts]


# ---------------------------------------------------------------------------
# Concept ↔ column matching
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


def _name_similarity(concept: str, name: str) -> float:
    """0..1 similarity score between a question concept and a column name."""
    a = concept.lower()
    b = name.lower().replace("_", " ").replace(".", " ")
    if not a or not b:
        return 0.0
    if a == b or a in b.split() or b in a.split():
        return 1.0
    if a in b or b in a:
        return 0.85
    distance = _levenshtein(a, b)
    return max(0.0, 1.0 - distance / max(len(a), len(b)))


def _value_match(concept: str, samples: list[Any]) -> bool:
    """True if the concept appears (case-insensitively) in any value sample."""
    a = concept.lower()
    for value in samples or []:
        if isinstance(value, (int, float)):
            continue  # numbers don't carry "concept" semantics
        text = str(value).strip().lower()
        if not text:
            continue
        if a == text or a in text or text in a:
            return True
    return False


@dataclass(frozen=True, slots=True)
class ConceptBinding:
    concept: str
    file_path: str
    column: str
    score: float
    cardinality: int
    sample_values: tuple[Any, ...]
    dtype: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "file_path": self.file_path,
            "column": self.column,
            "score": round(self.score, 3),
            "cardinality": self.cardinality,
            "sample_values": list(self.sample_values),
            "dtype": self.dtype,
        }


def match_concept(
    concept: str,
    capabilities: list[SourceCapability],
    *,
    top_k: int = 3,
) -> list[ConceptBinding]:
    """Return the top-K (path, column) pairs likely matching ``concept``."""
    bindings: list[ConceptBinding] = []
    for cap in capabilities:
        # All known columns: csv .columns, sqlite tables.column, json
        # record fields, structured-doc record fields. Assemble one flat
        # list with their associated value samples.
        column_candidates: list[tuple[str, list[Any], int, str]] = []  # (display_name, samples, card, dtype)
        for col_name in cap.columns or []:
            samples = list(cap.column_value_samples.get(col_name, []))
            card = int(cap.column_cardinalities.get(col_name, 0))
            dtype = str(cap.column_dtypes.get(col_name, ""))
            column_candidates.append((col_name, samples, card, dtype))
        for table in cap.tables or []:
            t_name = str(table.get("table") or "")
            for col in table.get("columns") or []:
                if not isinstance(col, dict):
                    continue
                col_name = str(col.get("name") or "")
                if not col_name:
                    continue
                full = f"{t_name}.{col_name}"
                samples = list(cap.column_value_samples.get(full, []))
                card = int(cap.column_cardinalities.get(full, 0))
                dtype = str(cap.column_dtypes.get(full, "") or col.get("type", ""))
                column_candidates.append((full, samples, card, dtype))
        # JSON record fields and structured prose record fields.
        for col_name in (cap.json_record_fields or []) + (cap.structured_record_fields or []):
            if any(col_name == name for name, *_ in column_candidates):
                continue
            samples = list(cap.column_value_samples.get(col_name, []))
            card = int(cap.column_cardinalities.get(col_name, 0))
            dtype = str(cap.column_dtypes.get(col_name, ""))
            column_candidates.append((col_name, samples, card, dtype))

        for col_name, samples, card, dtype in column_candidates:
            name_score = _name_similarity(concept, col_name)
            value_hit = _value_match(concept, samples)
            score = name_score
            if value_hit:
                score = max(score, 0.9)
            # Low-cardinality categorical columns are usually the right
            # filter target — give them a small bonus.
            if 0 < card <= 12 and score >= 0.4:
                score = min(1.0, score + 0.05)
            if score >= 0.4:
                bindings.append(ConceptBinding(
                    concept=concept,
                    file_path=cap.path,
                    column=col_name,
                    score=score,
                    cardinality=card,
                    sample_values=tuple(samples[:3]),
                    dtype=dtype,
                ))
    bindings.sort(key=lambda b: -b.score)
    return bindings[:top_k]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_grounding_block(
    *,
    question: str,
    compiled: CompiledTask,
    max_concepts: int = 8,
) -> tuple[str, list[dict[str, Any]]]:
    """Render the full Schema Grounding block (text + machine debug log).

    Returns ``(prompt_block_text, bindings_for_trace)``. The trace
    payload is dropped into ``compiled_task.schema_grounding`` so the
    answer post-mortem can show *why* a column was chosen.
    """
    if not compiled.source_capabilities:
        return "", []
    concepts = extract_question_concepts(question, max_concepts=max_concepts)
    if not concepts:
        return "", []

    bindings_payload: list[dict[str, Any]] = []
    rendered_lines: list[str] = [
        "Schema Grounding (question concept → real column candidates; hypotheses only):"
    ]
    seen_pairs: set[tuple[str, str]] = set()

    for concept in concepts:
        matches = match_concept(concept, compiled.source_capabilities, top_k=3)
        if not matches:
            continue
        # Trim the candidate list rendered in the prompt: only show the
        # best match per (file, column) pair so we don't repeat noise.
        kept: list[ConceptBinding] = []
        for match in matches:
            key = (match.file_path, match.column)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            kept.append(match)
            bindings_payload.append(match.to_dict())
        if not kept:
            continue
        for match in kept:
            sample_str = ", ".join(repr(v) for v in match.sample_values) or "(no samples)"
            dtype_str = f", dtype={match.dtype}" if match.dtype else ""
            card_str = f", card={match.cardinality}" if match.cardinality else ""
            rendered_lines.append(
                f"- '{concept}' → {match.file_path}::{match.column}  "
                f"(score={match.score:.2f}{dtype_str}{card_str}, "
                f"samples=[{sample_str}])"
            )

    if len(rendered_lines) == 1:
        return "", []

    rendered_lines.append(
        "These are real schema candidates, not semantic proof. Use the "
        "listed path::column names when a candidate is selected, but verify "
        "whether the column directly represents the requested concept or "
        "must be transformed. If row-level arithmetic or rule evidence "
        "contradicts this grounding hint, prefer the data/rule evidence and "
        "record the override in debug_steps['schema_mapping'] or "
        "debug_steps['plan_override']."
    )
    return "\n".join(rendered_lines), bindings_payload


def render_foreign_key_block(compiled: CompiledTask, *, max_pairs: int = 6) -> str:
    """Render the cross-file join-key candidate block."""
    pairs = compiled.foreign_key_candidates or []
    if not pairs:
        return ""
    lines: list[str] = [
        "Foreign-Key Candidates (cross-file join keys, validated by "
        "value-set overlap):"
    ]
    for entry in pairs[:max_pairs]:
        sample = ", ".join(repr(v) for v in (entry.get("shared_sample") or []))
        lines.append(
            f"- {entry.get('left_path')}::{entry.get('left_column')}  ↔  "
            f"{entry.get('right_path')}::{entry.get('right_column')}  "
            f"(jaccard={entry.get('jaccard')}, shared=[{sample}])"
        )
    lines.append(
        "When you need to JOIN across files, use ONE of these pairs as "
        "your join key. Do NOT invent new join column names."
    )
    return "\n".join(lines)
