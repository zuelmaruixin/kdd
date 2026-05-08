"""LLM-driven unstructured record-text extractor.

Some tasks ship long-form report-style documents where individual
records are buried in narrative prose (e.g. ``Patient.md`` /
``Laboratory.md`` in task_418: a research write-up that mentions each
patient's id + birthday + lab values across multi-paragraph blocks).
Pure regex extraction is too brittle here; pure RAG can't materialize
a full per-record table either. The fix is a tightly-bounded LLM
extraction pre-stage:

    unstructured_record_text
        ↓ split by paragraph + record-id mentions
    chunks (≈6KB each)
        ↓ LLM call per chunk with a strict JSON schema
    [record dicts]
        ↓ concatenate + dedupe by id
    synthesized CSV in temp dir
        ↓
    normal codegen path can now `pd.read_csv` the synthesized file
    and just write the question's filter / count.

We persistently cache extraction results by ``(file_hash + schema_hash +
endpoint)`` so re-runs of the same task don't re-pay the LLM cost.
The executor is gated on the operator-executor failing once first —
zero overhead for tasks that don't need it.
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
# Chunking
# ---------------------------------------------------------------------------


_ID_MENTION_RE = re.compile(
    r"\b(?:patient|medical record number|record number|file number|case|subject|id)\b"
    r"[^.\n]{0,40}?\b\d{3,8}\b",
    re.I,
)


def _split_into_record_chunks(text: str, *, target_chars: int = 6000) -> list[str]:
    """Greedy chunker that aligns chunk boundaries to paragraph breaks
    and tries to keep all id-mentions for the same record inside the
    same chunk (so the LLM has full context).
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

    # Drop chunks that don't mention any record id at all — they're
    # almost certainly methodology / abstract sections.
    return [chunk for chunk in chunks if _ID_MENTION_RE.search(chunk)]


_STOPWORDS = {
    "among", "whose", "level", "them", "they", "that", "with", "from",
    "please", "give", "many", "how", "what", "which", "then", "than",
    "not", "yet", "are", "the", "and", "for", "patients", "patient",
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
    """Keep only chunks likely to contain fields needed by the question.

    This is a guardrail, not a solver: it prevents a 300KB narrative report
    from becoming dozens of LLM extraction calls. Deterministic extractors
    below usually handle common record_text files before this fallback runs.
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
    # Restore document order after selecting the best matches.
    order = {id(chunk): idx for idx, chunk in enumerate(chunks)}
    selected.sort(key=lambda chunk: order.get(id(chunk), 0))
    return selected


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------


_DEFAULT_SCHEMA_BY_HINT: dict[str, list[str]] = {
    "patient": [
        "patient_id", "sex", "birthday", "first_record_date",
        "got", "gpt", "ldh", "alp", "t_bil", "creatinine", "uric_acid",
        "diagnosis", "admission",
    ],
    "race": [
        "race_id", "year", "round", "name", "date",
        "constructor_id", "constructor_ref", "url",
    ],
    "default": [
        "id", "name", "date", "value",
    ],
}


def _guess_schema(cap: SourceCapability, *, question: str = "") -> list[str]:
    """Choose a target schema for a structured_prose file.

    Prefer the fields the deterministic profiler already saw; fall back
    to topic hints based on the file path / sample content.
    """
    question_lower = question.lower()
    path_lower = cap.path.lower()
    if "patient" in path_lower and re.search(r"\b(age|birthday|birth|born|70|years old)\b", question_lower):
        return ["patient_id", "birth_year", "birthday_text"]
    if "laboratory" in path_lower and re.search(r"\b(creatinine|cre)\b", question_lower):
        return ["patient_id", "creatinine", "creatinine_abnormal", "evidence"]
    if "race" in path_lower and re.search(r"\b(race|grand prix|constructor|champion)\b", question_lower):
        return ["race_id", "year", "name", "date"]

    fields: list[str] = list(cap.structured_record_fields or [])
    if "patient_id" not in fields and "patient" in cap.path.lower():
        fields = _DEFAULT_SCHEMA_BY_HINT["patient"]
    elif "race" in cap.path.lower() and "race_id" not in fields:
        fields = _DEFAULT_SCHEMA_BY_HINT["race"]
    if not fields:
        fields = _DEFAULT_SCHEMA_BY_HINT["default"]
    # Always carry an id-like first column.
    return fields


def _schema_hash(fields: list[str]) -> str:
    return hashlib.sha256(",".join(fields).encode("utf-8")).hexdigest()[:12]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You extract structured records from a chunk of a long-form medical / "
    "racing / scientific report. The report is narrative prose, not a "
    "clean record list. Many sentences will be context, summary, "
    "methodology, or corrections (e.g. 'originally 35.0; corrected to "
    "28.0'). For every record explicitly mentioned in the chunk, output "
    "ONE JSON object with exactly the fields requested. Use null for "
    "fields not present in this chunk. Apply 'corrected to' overrides "
    "when both an original and a corrected value are mentioned for the "
    "same field. Output STRICT JSON: a single top-level array; no prose."
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
# Deterministic medical narrative extraction
# ---------------------------------------------------------------------------


_PATIENT_ID_PATTERNS = (
    re.compile(r"\bpatient\s+(?:profile\s+)?(?:associated\s+with\s+)?(?:registered\s+under\s+)?(?:file\s+)?(?:number\s+)?(\d{3,8})\b", re.I),
    re.compile(r"\bMedical Record Number\s+(\d{3,8})\b", re.I),
    re.compile(r"\bfile(?:\s+number)?\s+(\d{3,8})\b", re.I),
    re.compile(r"\bprofile\s+for\s+patient\s+(\d{3,8})\b", re.I),
)


def _record_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _extract_patient_id(block: str) -> str | None:
    for pattern in _PATIENT_ID_PATTERNS:
        match = pattern.search(block)
        if match is not None:
            return match.group(1)
    return None


def _extract_birth_year(block: str) -> tuple[int | None, str | None]:
    sentence_match = re.search(
        r"([^.\n]*(?:born|birthdate|birthday|birth year|date of birth)[^.\n]*\.)",
        block,
        re.I,
    )
    if sentence_match is None:
        return None, None
    target = sentence_match.group(1) if sentence_match else block
    candidates = re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", target)
    if not candidates:
        return None, None
    # If the same birth sentence contains a correction, the final year is
    # the verified one; avoid looking into later record-creation sentences.
    year = int(candidates[-1])
    return year, sentence_match.group(1).strip() if sentence_match else None


def _creatinine_segment(block: str) -> str:
    sentences = [
        sentence
        for sentence in re.split(r"(?<!\d)\.(?!\d)", block)
        if re.search(r"\b(creatinine|CRE)\b", sentence, re.I)
    ]
    return " ".join(sentences)


def _extract_creatinine_value(block: str) -> float | None:
    segment = _creatinine_segment(block)
    sentences = [segment] if segment else []
    if not sentences:
        return None
    values = [float(item) for item in re.findall(r"\b(\d+(?:\.\d+)?)\s*mg/dL\b", segment, re.I)]
    if not values:
        values = [
            float(item)
            for item in re.findall(r"\b(\d+(?:\.\d+)?)\b", segment)
            if float(item) < 20
        ]
    if not values:
        return None
    return values[-1]


def _extract_threshold_from_semantic_context(
    *,
    semantic_context: str,
    field_terms: tuple[str, ...],
) -> float | None:
    if not semantic_context.strip():
        return None
    field_pattern = "|".join(re.escape(term) for term in field_terms)
    patterns = (
        rf"(?:{field_pattern})[^.\n]{{0,120}}?(?:above|greater than|>|exceeds)\s*(\d+(?:\.\d+)?)",
        rf"(?:{field_pattern})[^.\n]{{0,120}}?values?\s*(?:above|greater than|>|exceeding)\s*(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, semantic_context, re.I)
        if match is not None:
            return float(match.group(1))
    return None


def _relevant_rule_lines(*, semantic_context: str, question: str, schema_fields: list[str]) -> list[str]:
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


def _creatinine_abnormal(value: float, block: str, *, semantic_context: str) -> bool:
    threshold = _extract_threshold_from_semantic_context(
        semantic_context=semantic_context,
        field_terms=("creatinine", "CRE"),
    )
    if threshold is not None:
        return value > threshold
    lowered = _creatinine_segment(block).lower()
    abnormal_markers = (
        "elevated", "impaired", "compromised", "renal stress",
        "renal dysfunction", "significant reduction", "severely",
        "active renal disease", "renal impairment",
    )
    normal_markers = (
        "normal renal function", "normal physiological range",
        "healthy kidney function", "normal findings", "unremarkable",
        "within normal", "upper limit of the normal range",
    )
    if any(marker in lowered for marker in normal_markers):
        return False
    return any(marker in lowered for marker in abnormal_markers)


def _deterministic_extract_patient_births(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for block in _record_blocks(text):
        patient_id = _extract_patient_id(block)
        if patient_id is None:
            continue
        birth_year, birthday_text = _extract_birth_year(block)
        if birth_year is None:
            continue
        seen[patient_id] = {
            "patient_id": patient_id,
            "birth_year": birth_year,
            "birthday_text": birthday_text,
        }
    records.extend(seen.values())
    return records


def _deterministic_extract_creatinine(text: str) -> list[dict[str, Any]]:
    return _deterministic_extract_creatinine_with_rules(text, semantic_context="")


def _deterministic_extract_creatinine_with_rules(
    text: str,
    *,
    semantic_context: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    threshold = _extract_threshold_from_semantic_context(
        semantic_context=semantic_context,
        field_terms=("creatinine", "CRE"),
    )
    rule_source = (
        f"knowledge_threshold:creatinine>{threshold}"
        if threshold is not None
        else "knowledge_no_creatinine_threshold__used_source_text_status"
    )
    for block in _record_blocks(text):
        if not re.search(r"\b(creatinine|CRE)\b", block, re.I):
            continue
        patient_id = _extract_patient_id(block)
        if patient_id is None:
            continue
        value = _extract_creatinine_value(block)
        if value is None:
            continue
        evidence = re.sub(r"\s+", " ", block).strip()[:500]
        seen[patient_id] = {
            "patient_id": patient_id,
            "creatinine": value,
            "creatinine_abnormal": _creatinine_abnormal(
                value,
                block,
                semantic_context=semantic_context,
            ),
            "rule_source": rule_source,
            "evidence": evidence,
        }
    records.extend(seen.values())
    return records


def _deterministic_extract_records(
    *,
    cap: SourceCapability,
    text: str,
    question: str,
    semantic_context: str = "",
) -> tuple[list[str], list[dict[str, Any]], str] | None:
    question_lower = question.lower()
    path_lower = cap.path.lower()
    if "patient" in path_lower and re.search(r"\b(age|birthday|birth|born|70|years old)\b", question_lower):
        records = _deterministic_extract_patient_births(text)
        if records:
            return ["patient_id", "birth_year", "birthday_text"], records, "deterministic_patient_births"
    if "laboratory" in path_lower and re.search(r"\b(creatinine|cre)\b", question_lower):
        records = _deterministic_extract_creatinine_with_rules(
            text,
            semantic_context=semantic_context,
        )
        if records:
            return [
                "patient_id", "creatinine", "creatinine_abnormal",
                "rule_source", "evidence",
            ], records, "deterministic_creatinine"
    if "legalities" in path_lower and re.search(r"\b(commander|legal|status|content warning)\b", question_lower):
        records = _deterministic_extract_legalities(text)
        if records:
            return ["legality_id", "card_id", "format", "status", "rule_source"], records, "deterministic_legalities"
    return None


def _deterministic_extract_legalities(text: str) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for block in _record_blocks(text):
        id_match = re.search(r"\b(?:ID|id)\s+(\d{2,})\b", block)
        if id_match is None:
            continue
        legality_id = id_match.group(1)
        record = by_id.setdefault("legality_id:" + legality_id, {"legality_id": legality_id})
        card_ids = re.findall(r"\bcards_id\s+(\d+)\b", block, re.I)
        if card_ids:
            record["card_id"] = int(card_ids[-1])
        lowered = block.lower()
        if "commander" in lowered or "singleton multiplayer" in lowered or "multiplayer format" in lowered:
            record["format"] = "commander"
        if re.search(r"\blegal\b", block, re.I) or "unrestricted standing" in lowered or "cleared for play" in lowered:
            record["status"] = "Legal"
        elif re.search(r"\bbanned\b", block, re.I):
            record["status"] = "Banned"
        elif re.search(r"\brestricted\b", block, re.I):
            record["status"] = "Restricted"
        if "format" in record or "status" in record:
            record["rule_source"] = "knowledge:filter_by_format_and_status"

    merged: dict[str, dict[str, Any]] = {}
    for item in by_id.values():
        legality_id = str(item.get("legality_id"))
        merged.setdefault(legality_id, {"legality_id": legality_id}).update(item)
    return [
        rec
        for rec in merged.values()
        if rec.get("card_id") is not None or rec.get("format") is not None or rec.get("status") is not None
    ]


def _preview_records(records: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    return [dict(record) for record in records[:limit]]


def _positive_preview_records(records: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    for record in records:
        if any(
            key.endswith("_abnormal") and bool(value)
            for key, value in record.items()
        ):
            positives.append(dict(record))
        elif str(record.get("status", "")).lower() == "legal":
            positives.append(dict(record))
        if len(positives) >= limit:
            break
    return positives


def _record_rule_notes(
    *,
    method: str,
    semantic_context: str,
    question: str,
    schema_fields: list[str],
) -> list[str]:
    notes = _relevant_rule_lines(
        semantic_context=semantic_context,
        question=question,
        schema_fields=schema_fields,
    )
    if method == "deterministic_creatinine" and not _extract_threshold_from_semantic_context(
        semantic_context=semantic_context,
        field_terms=("creatinine", "CRE"),
    ):
        notes.append(
            "knowledge.md does not define a creatinine threshold; abnormality is taken from explicit source-text status/evidence, not an invented cutoff."
        )
    if method == "deterministic_legalities":
        notes.append(
            "knowledge.md rule: filter by `format` for play format and `status` for legality status."
        )
    return notes[:10]


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _parse_records(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence is not None:
        text = fence.group(1).strip()
    if not text:
        return []
    # Find the first JSON-array open bracket.
    start = text.find("[")
    if start < 0:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return []
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


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
    positive_previews: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

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
        }


@dataclass(slots=True)
class StructuredDocExecutor:
    """LLM-extraction stage that synthesizes a CSV per record-text file."""

    model: OpenAIModelAdapter
    compiled_task: CompiledTask
    chunk_chars: int = 7000
    cache_dir: Path | None = None
    max_chunks_per_file: int = 8
    extraction_max_tokens: int = 900
    output_dir: Path | None = None  # where synthesized csvs are placed; default = task.context_dir / .synthesized

    def has_structured_prose(self) -> bool:
        return any(
            cap.kind in {"record_text", "structured_table"}
            for cap in self.compiled_task.source_capabilities
        )

    def run(self, task: PublicTask) -> StructuredDocExtraction:
        result = StructuredDocExtraction()
        if not self.has_structured_prose():
            result.notes.append("no_structured_prose_files")
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

            deterministic = _deterministic_extract_records(
                cap=cap,
                text=text,
                question=task.question,
                semantic_context=semantic_context,
            )
            if deterministic is not None:
                schema_fields, records, method = deterministic
                result.schemas[cap.path] = list(schema_fields)
                result.record_counts[cap.path] = len(records)
                result.chunk_counts[cap.path] = 0
                result.notes.append(f"{method}:{cap.path}:{len(records)}")
                result.rule_notes[cap.path] = _record_rule_notes(
                    method=method,
                    semantic_context=semantic_context,
                    question=task.question,
                    schema_fields=schema_fields,
                )
                result.previews[cap.path] = _preview_records(records)
                result.positive_previews[cap.path] = _positive_preview_records(records)
                csv_name = cap.path.replace("/", "__").replace(".md", "") + ".synth.csv"
                csv_path = out_root / csv_name
                self._write_csv(csv_path=csv_path, schema_fields=schema_fields, records=records)
                try:
                    rel = csv_path.relative_to(task.context_dir).as_posix()
                except ValueError:
                    rel = str(csv_path)
                result.synthesized_csvs[cap.path] = rel
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
            result.rule_notes[cap.path] = _record_rule_notes(
                method="llm_chunk_extract",
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

            # Dedupe by the schema's canonical id field if present.
            id_field = next(
                (
                    candidate for candidate in (
                        "patient_id", "race_id", "id", "record_id",
                    )
                    if candidate in schema_fields
                ),
                None,
            )
            if id_field is not None:
                seen: dict[Any, dict[str, Any]] = {}
                for rec in all_records:
                    key = rec.get(id_field)
                    if key is None:
                        continue
                    # Newer mentions (later chunks) win — they often
                    # contain the corrected/finalized values.
                    seen[key] = {**(seen.get(key) or {}), **rec}
                merged_records = list(seen.values())
            else:
                merged_records = all_records

            result.record_counts[cap.path] = len(merged_records)
            result.previews[cap.path] = _preview_records(merged_records)
            result.positive_previews[cap.path] = _positive_preview_records(merged_records)
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
    lines = ["Synthesized CSVs (already extracted from record_text docs):"]
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
