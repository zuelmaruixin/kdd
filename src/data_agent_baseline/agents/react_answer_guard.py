"""Cheap, deterministic risk guard for ReAct draft answers.

This is the React-flavored sibling of ``semantic_guard.py``. It decides
whether the model's first ``answer`` call is safe enough to commit
directly, or whether the loop should burn one extra round forcing the
model to self-verify.

It is NON-LLM by design — the whole point is "本地修了就本地修, 实在跑不通才
再用 LLM": we only escalate to a model round when a deterministic local
check finds concrete evidence of risk. This matches Huang et al. ICLR
2024 ("LLMs Cannot Self-Correct Reasoning Yet") which shows that
unconditional self-correction degrades accuracy; CRITIC (Gou et al.
ICLR 2024) which shows tool-grounded checks are the useful escalation
signal; and the FrugalGPT/AutoMix cascade pattern.

Signals are reused from :class:`SemanticRisk` /
:class:`SemanticRiskAssessment` in ``semantic_guard`` so trace
consumers, scoring code, and the demo UI can treat both guards
uniformly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from data_agent_baseline.agents.semantic_guard import (
    SemanticRisk,
    SemanticRiskAssessment,
)
from data_agent_baseline.eval.answer_validator import validate_answer_table

if TYPE_CHECKING:
    from data_agent_baseline.agents.runtime import StepRecord
    from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask


# Tools that count as "real computation" against the full dataset. If
# none of these were called before `answer`, the model is answering from
# the sample preview alone — high risk.
_COMPUTE_ACTIONS: frozenset[str] = frozenset({
    "execute_python",
    "execute_context_sql",
})

# Tools that count as "I read the data, not just listed it". Used to
# detect the degenerate "answer without ever looking at any file" case.
_INSPECT_ACTIONS: frozenset[str] = frozenset({
    "read_csv",
    "read_json",
    "read_doc",
    "inspect_sqlite_schema",
})

# Filename patterns that signal an authoritative knowledge doc. If any of
# these are in context but the model never called read_doc on one, that
# is a hard risk — knowledge.md is a NON-NEGOTIABLE override per the
# project's prompt contract.
_KNOWLEDGE_FILE_RE = re.compile(
    r"(?:^|[/\\_-])(knowledge|rule|rules|definition|definitions|glossary|schema_notes)"
    r"\b[^/\\]*\.(md|markdown|txt)$",
    re.IGNORECASE,
)

# Question phrasing that strongly implies a single scalar answer (1 row).
_SCALAR_QUESTION_RE = re.compile(
    r"\b("
    r"how\s+many|how\s+much|"
    r"what\s+is\s+the|what's\s+the|"
    r"average|mean|median|sum|total|count\s+of|number\s+of|"
    r"ratio|percentage|percent|proportion|"
    r"maximum|max(?:imum)?|minimum|min(?:imum)?|"
    r"highest|lowest|largest|smallest|most|least"
    r")\b",
    re.IGNORECASE,
)

# Question phrasing that implies multiple rows (a list / ranking).
_LIST_QUESTION_RE = re.compile(
    r"\b("
    r"list|which\s+\w+|name\s+the|top\s+\d+|"
    r"all\s+the|every|each\s+of|"
    r"rank|order\s+by|sort"
    r")\b",
    re.IGNORECASE,
)


def assess_react_answer_risk(
    *,
    task: "PublicTask",
    draft: "AnswerTable | None",
    steps: list["StepRecord"],
) -> SemanticRiskAssessment:
    """Return a deterministic risk assessment for a React draft answer.

    ``draft`` is the answer the model just submitted with its first
    ``answer`` tool call. ``steps`` is every step taken so far (including
    that first answer call). ``task`` provides the question and context
    directory.

    The returned assessment is shaped exactly like the codegen guard's
    output: ``.should_escalate`` says yes/no, ``.risks`` lists the
    concrete reasons. Callers (``react.py``) only need ``should_escalate``;
    the rest is for trace audit.
    """
    risks: list[SemanticRisk] = []

    # ── R1: missing / structurally invalid answer ───────────────────────
    if draft is None:
        risks.append(SemanticRisk(
            code="execution_failed_or_missing_answer",
            severity="error",
            message="No AnswerTable was produced by the answer tool call.",
            weight=1.0,
        ))
        return _finalize(risks)

    if not draft.rows:
        risks.append(SemanticRisk(
            code="empty_answer_rows",
            severity="error",
            message="Draft answer has columns but zero rows.",
            evidence={"columns": list(draft.columns)},
            weight=0.75,
        ))

    validation = validate_answer_table(draft)
    if not validation.valid:
        risks.append(SemanticRisk(
            code="invalid_answer",
            severity="error",
            message="Draft answer failed structural validation.",
            evidence=validation.to_dict(),
            weight=1.0,
        ))
        return _finalize(risks)

    # ── R2: no real computation before answer ───────────────────────────
    successful_compute = sum(
        1 for s in steps
        if s.ok and s.action in _COMPUTE_ACTIONS and not _is_answer_step(s)
    )
    successful_inspect = sum(
        1 for s in steps
        if s.ok and s.action in _INSPECT_ACTIONS
    )
    if successful_compute == 0:
        # Strong signal: the model never executed code/sql against the
        # full dataset. Sample previews from read_csv/read_json are NOT
        # safe to derive the final answer from (the system prompt
        # literally says so).
        risks.append(SemanticRisk(
            code="answer_without_computation",
            severity="error",
            message=(
                "Model emitted answer without any successful "
                "execute_python / execute_context_sql call. Sample rows "
                "shown by read_csv are insufficient to derive a final "
                "answer."
            ),
            evidence={
                "successful_compute_calls": successful_compute,
                "successful_inspect_calls": successful_inspect,
            },
            weight=0.85,
        ))

    # ── R3: knowledge file present but never read ───────────────────────
    knowledge_files = _list_knowledge_files(task)
    if knowledge_files:
        read_docs = _read_doc_paths(steps)
        knowledge_read = any(
            _path_matches_any(read_path, knowledge_files)
            for read_path in read_docs
        )
        if not knowledge_read:
            risks.append(SemanticRisk(
                code="knowledge_doc_not_read",
                severity="error",
                message=(
                    "Task context contains a knowledge/rule/glossary doc, "
                    "but the agent never called read_doc on it before "
                    "answering. Per project contract, this doc is the "
                    "single source of truth for domain definitions and "
                    "must be read first."
                ),
                evidence={"knowledge_files": [str(p) for p in knowledge_files][:5]},
                weight=0.8,
            ))

    # ── R4: shape vs question type mismatch ─────────────────────────────
    row_count = len(draft.rows)
    column_count = len(draft.columns)
    question = task.question or ""
    scalar_hint = bool(_SCALAR_QUESTION_RE.search(question))
    list_hint = bool(_LIST_QUESTION_RE.search(question))
    if scalar_hint and not list_hint and row_count > 1:
        risks.append(SemanticRisk(
            code="shape_mismatch_scalar_question",
            severity="warning",
            message=(
                "Question phrasing suggests a single scalar answer, but "
                "draft has multiple rows."
            ),
            evidence={"row_count": row_count, "column_count": column_count},
            weight=0.35,
        ))
    elif list_hint and not scalar_hint and row_count == 1 and column_count == 1:
        risks.append(SemanticRisk(
            code="shape_mismatch_list_question",
            severity="warning",
            message=(
                "Question phrasing suggests a list answer, but draft has "
                "a single scalar cell."
            ),
            evidence={"row_count": row_count, "column_count": column_count},
            weight=0.35,
        ))

    return _finalize(risks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finalize(risks: list[SemanticRisk]) -> SemanticRiskAssessment:
    deduped: list[SemanticRisk] = []
    seen: set[tuple[str, str]] = set()
    for risk in risks:
        key = (risk.code, str(risk.evidence))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(risk)
    score = min(1.0, sum(risk.weight for risk in deduped))
    return SemanticRiskAssessment(
        risks=tuple(deduped),
        grounding=(),
        trace_present=bool(deduped),
        score=score,
    )


def _is_answer_step(step: "StepRecord") -> bool:
    return step.action == "answer"


def _list_knowledge_files(task: "PublicTask") -> list[Path]:
    try:
        root = Path(task.context_dir)
    except Exception:  # noqa: BLE001
        return []
    if not root.exists() or not root.is_dir():
        return []
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        if _KNOWLEDGE_FILE_RE.search(rel):
            matches.append(path)
    return matches


def _read_doc_paths(steps: Iterable["StepRecord"]) -> list[str]:
    paths: list[str] = []
    for step in steps:
        if step.action != "read_doc" or not step.ok:
            continue
        ai = step.action_input or {}
        candidate = ai.get("path") if isinstance(ai, dict) else None
        if isinstance(candidate, str) and candidate:
            paths.append(candidate)
    return paths


def _path_matches_any(read_path: str, knowledge_files: list[Path]) -> bool:
    read_norm = read_path.replace("\\", "/").strip().lstrip("./")
    read_basename = Path(read_norm).name.lower()
    for known in knowledge_files:
        if known.name.lower() == read_basename:
            return True
        # Also accept a relative-path equivalence (e.g. the model passes
        # "docs/knowledge.md" while we recorded the same suffix).
        try:
            if str(known).replace("\\", "/").endswith(read_norm):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


# ---------------------------------------------------------------------------
# Public helper used by react.py to summarize the decision for trace.
# ---------------------------------------------------------------------------


def summarize_assessment(assessment: SemanticRiskAssessment) -> dict[str, Any]:
    """Compact dict for progress events / trace records."""
    return {
        "should_escalate": assessment.should_escalate,
        "score": round(assessment.score, 3),
        "risk_codes": [risk.code for risk in assessment.risks],
        "top_risk": (
            {
                "code": assessment.risks[0].code,
                "severity": assessment.risks[0].severity,
                "message": assessment.risks[0].message,
            }
            if assessment.risks else None
        ),
    }
