"""Cross-way self-consistency / column-level voting.

For each task we run the ReAct agent ``num_samples`` times with a non-zero
sampling temperature. Each run gives us an :class:`AnswerTable` (or a
failure). We then build the official column content signatures for every
candidate column, count votes, and emit a consensus table.

The aggregation strategy mirrors the official scoring rule:

- The official rubric matches columns by content signature (ignores names,
  ignores row order). So we vote at the column-signature level.
- We want a final table whose set of column signatures maximizes the
  expected recall while penalizing extra columns. Concretely, we keep
  every signature whose vote count >= ``min_votes``, capped to the modal
  per-sample column count when ``min_votes == 1``.
- For the actual ``rows`` payload we fall back to the candidate run that
  produced exactly the winning signature set (or has the largest overlap),
  to keep the row contents internally consistent.

The implementation is deliberately defensive: any single failed sample is
discarded; if every sample fails we surface the last failure reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.config import SelfConsistencyConfig
from data_agent_baseline.eval.column_match import column_signature
from data_agent_baseline.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SampleRunRecord:
    sample_index: int
    temperature: float
    seed: int | None
    succeeded: bool
    failure_reason: str | None
    answer: AnswerTable | None
    step_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "temperature": self.temperature,
            "seed": self.seed,
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "step_count": self.step_count,
        }


@dataclass(slots=True)
class VotingDecision:
    aggregator: str
    voted_signatures: list[tuple[str, int]] = field(default_factory=list)  # (sig_repr, vote)
    chosen_sample_index: int | None = None
    consensus_column_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregator": self.aggregator,
            "voted_signatures": [
                {"signature": sig, "votes": votes} for sig, votes in self.voted_signatures
            ],
            "chosen_sample_index": self.chosen_sample_index,
            "consensus_column_count": self.consensus_column_count,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class SelfConsistencyResult:
    task_id: str
    answer: AnswerTable | None
    samples: list[SampleRunRecord]
    decision: VotingDecision
    failure_reason: str | None
    chosen_steps: list[StepRecord]

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "samples": [sample.to_dict() for sample in self.samples],
            "decision": self.decision.to_dict(),
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
            "steps": [step.to_dict() for step in self.chosen_steps],
        }


# ---------------------------------------------------------------------------
# Voting logic
# ---------------------------------------------------------------------------


def _table_to_columns(table: AnswerTable) -> list[list[Any]]:
    if not table.columns:
        return []
    columns: list[list[Any]] = [[] for _ in table.columns]
    for row in table.rows:
        for idx in range(len(table.columns)):
            cell = row[idx] if idx < len(row) else None
            columns[idx].append(cell)
    return columns


def _columns_to_signatures(
    columns: Sequence[Sequence[Any]],
    *,
    numeric_tolerance: float,
    case_insensitive: bool,
    strip_whitespace: bool,
) -> list[tuple[str, ...]]:
    return [
        column_signature(
            col,
            numeric_tolerance=numeric_tolerance,
            case_insensitive=case_insensitive,
            strip_whitespace=strip_whitespace,
        )
        for col in columns
    ]


def _vote_signatures(
    sample_signatures: Sequence[Sequence[tuple[str, ...]]],
) -> dict[tuple[str, ...], int]:
    """Count how many samples produced each column signature.

    Each signature contributes at most ONE vote per sample, even if the
    sample emitted the same column twice.
    """
    votes: dict[tuple[str, ...], int] = {}
    for signatures in sample_signatures:
        seen_in_sample = set(signatures)
        for sig in seen_in_sample:
            votes[sig] = votes.get(sig, 0) + 1
    return votes


def _modal_column_count(sample_signatures: Sequence[Sequence[tuple[str, ...]]]) -> int:
    if not sample_signatures:
        return 0
    counts: dict[int, int] = {}
    for signatures in sample_signatures:
        counts[len(signatures)] = counts.get(len(signatures), 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _pick_winning_signatures(
    *,
    votes: dict[tuple[str, ...], int],
    target_count: int,
    min_votes: int,
) -> list[tuple[str, ...]]:
    """Return up to ``target_count`` signatures sorted by vote count desc."""
    eligible = [(sig, count) for sig, count in votes.items() if count >= min_votes]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [sig for sig, _ in eligible[:target_count]]


def _choose_best_sample(
    *,
    samples: Sequence[SampleRunRecord],
    sample_signatures: Sequence[Sequence[tuple[str, ...]]],
    winning_set: set[tuple[str, ...]],
) -> int | None:
    """Pick the sample index whose column signatures best cover the winning set."""
    best_index: int | None = None
    best_score: tuple[int, int] | None = None
    for sample_index, signatures in enumerate(sample_signatures):
        sample = samples[sample_index]
        if not sample.succeeded or sample.answer is None:
            continue
        sig_set = set(signatures)
        coverage = len(sig_set & winning_set)
        # Tie-break: prefer fewer extra columns.
        extras = max(len(sig_set) - coverage, 0)
        score = (coverage, -extras)
        if best_score is None or score > best_score:
            best_score = score
            best_index = sample_index
    return best_index


def _project_answer_to_winners(
    *,
    sample: SampleRunRecord,
    sample_signatures: Sequence[tuple[str, ...]],
    winning_signatures: list[tuple[str, ...]],
) -> AnswerTable | None:
    """Trim a sample's answer down to columns whose signatures are winners.

    If the sample is missing any winning signature, return ``None`` so the
    caller can fall back.
    """
    if sample.answer is None:
        return None
    sig_to_idx: dict[tuple[str, ...], int] = {}
    for idx, sig in enumerate(sample_signatures):
        sig_to_idx.setdefault(sig, idx)
    selected_indices: list[int] = []
    for sig in winning_signatures:
        if sig not in sig_to_idx:
            return None
        selected_indices.append(sig_to_idx[sig])

    answer = sample.answer
    new_columns = [answer.columns[idx] for idx in selected_indices]
    new_rows = [[row[idx] for idx in selected_indices] for row in answer.rows]
    return AnswerTable(columns=new_columns, rows=new_rows)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


SampleRunner = Callable[[ReActAgentConfig], AgentRunResult]


def _aggregate_column_vote(
    *,
    task_id: str,
    samples: list[SampleRunRecord],
    config: SelfConsistencyConfig,
    numeric_tolerance: float,
    case_insensitive: bool,
    strip_whitespace: bool,
    chosen_steps_by_index: dict[int, list[StepRecord]],
) -> SelfConsistencyResult:
    successful = [sample for sample in samples if sample.succeeded and sample.answer is not None]
    if not successful:
        last_failure = next(
            (sample.failure_reason for sample in reversed(samples) if sample.failure_reason),
            "All self-consistency samples failed.",
        )
        return SelfConsistencyResult(
            task_id=task_id,
            answer=None,
            samples=samples,
            decision=VotingDecision(aggregator="column_vote", notes=["no_successful_samples"]),
            failure_reason=last_failure,
            chosen_steps=[],
        )

    successful_indices = [sample.sample_index for sample in successful]
    sample_signatures: list[list[tuple[str, ...]]] = []
    for sample in successful:
        assert sample.answer is not None  # for the type checker
        columns = _table_to_columns(sample.answer)
        sample_signatures.append(
            _columns_to_signatures(
                columns,
                numeric_tolerance=numeric_tolerance,
                case_insensitive=case_insensitive,
                strip_whitespace=strip_whitespace,
            )
        )

    votes = _vote_signatures(sample_signatures)
    target_count = _modal_column_count(sample_signatures)
    winning_signatures = _pick_winning_signatures(
        votes=votes,
        target_count=target_count,
        min_votes=max(config.min_votes, 1),
    )
    if not winning_signatures and sample_signatures:
        # Fallback: take whatever the first successful sample produced.
        winning_signatures = list(sample_signatures[0])

    winning_set = set(winning_signatures)
    chosen_local_idx = _choose_best_sample(
        samples=successful,
        sample_signatures=sample_signatures,
        winning_set=winning_set,
    )

    answer: AnswerTable | None = None
    chosen_sample_index: int | None = None
    if chosen_local_idx is not None:
        chosen_sample = successful[chosen_local_idx]
        chosen_signatures = sample_signatures[chosen_local_idx]
        chosen_sample_index = chosen_sample.sample_index
        answer = _project_answer_to_winners(
            sample=chosen_sample,
            sample_signatures=chosen_signatures,
            winning_signatures=winning_signatures,
        )
        if answer is None:
            answer = chosen_sample.answer

    voted_signatures_payload = sorted(
        ((str(_signature_repr(sig)), count) for sig, count in votes.items()),
        key=lambda item: (-item[1], item[0]),
    )
    decision = VotingDecision(
        aggregator="column_vote",
        voted_signatures=list(voted_signatures_payload),
        chosen_sample_index=chosen_sample_index,
        consensus_column_count=len(winning_signatures),
    )
    if chosen_sample_index is None:
        decision.notes.append("voting_failed_falling_back_to_first_sample")

    chosen_steps: list[StepRecord] = []
    if chosen_sample_index is not None:
        chosen_steps = chosen_steps_by_index.get(chosen_sample_index, [])

    failure_reason: str | None = None if answer is not None else "Self-consistency could not pick an answer."
    return SelfConsistencyResult(
        task_id=task_id,
        answer=answer,
        samples=samples,
        decision=decision,
        failure_reason=failure_reason,
        chosen_steps=chosen_steps,
    )


def _signature_repr(signature: tuple[str, ...]) -> str:
    """Make the signature JSON-serializable and reasonably small in the trace."""
    if len(signature) <= 8:
        return json.dumps(list(signature), ensure_ascii=False)
    head = list(signature[:4])
    tail = list(signature[-2:])
    summary = head + [f"...{len(signature) - 6} more..."] + tail
    return json.dumps(summary, ensure_ascii=False)


def _aggregate_first_success(
    *,
    task_id: str,
    samples: list[SampleRunRecord],
    chosen_steps_by_index: dict[int, list[StepRecord]],
) -> SelfConsistencyResult:
    for sample in samples:
        if sample.succeeded and sample.answer is not None:
            return SelfConsistencyResult(
                task_id=task_id,
                answer=sample.answer,
                samples=samples,
                decision=VotingDecision(
                    aggregator="first_success",
                    chosen_sample_index=sample.sample_index,
                    consensus_column_count=len(sample.answer.columns),
                ),
                failure_reason=None,
                chosen_steps=chosen_steps_by_index.get(sample.sample_index, []),
            )

    last_failure = next(
        (sample.failure_reason for sample in reversed(samples) if sample.failure_reason),
        "All self-consistency samples failed.",
    )
    return SelfConsistencyResult(
        task_id=task_id,
        answer=None,
        samples=samples,
        decision=VotingDecision(aggregator="first_success", notes=["no_successful_samples"]),
        failure_reason=last_failure,
        chosen_steps=[],
    )


def run_with_self_consistency(
    *,
    task: PublicTask,
    model: ModelAdapter,
    tools: ToolRegistry,
    base_agent_config: ReActAgentConfig,
    config: SelfConsistencyConfig,
    numeric_tolerance: float = 1e-6,
    case_insensitive: bool = True,
    strip_whitespace: bool = True,
) -> SelfConsistencyResult:
    num_samples = max(config.num_samples, 1)
    samples: list[SampleRunRecord] = []
    chosen_steps_by_index: dict[int, list[StepRecord]] = {}

    for sample_index in range(num_samples):
        if num_samples == 1 and config.sample_temperature <= 0:
            sample_temperature = base_agent_config.sample_temperature
        else:
            sample_temperature = config.sample_temperature

        agent_config = ReActAgentConfig(
            max_steps=base_agent_config.max_steps,
            sample_temperature=sample_temperature,
            sample_seed=sample_index + 1 if num_samples > 1 else base_agent_config.sample_seed,
        )
        agent = ReActAgent(model=model, tools=tools, config=agent_config)
        try:
            run_result = agent.run(task)
        except Exception as exc:  # noqa: BLE001
            samples.append(
                SampleRunRecord(
                    sample_index=sample_index,
                    temperature=sample_temperature if sample_temperature is not None else 0.0,
                    seed=agent_config.sample_seed,
                    succeeded=False,
                    failure_reason=f"sample_runtime_error: {exc}",
                    answer=None,
                    step_count=0,
                )
            )
            continue

        record = SampleRunRecord(
            sample_index=sample_index,
            temperature=sample_temperature if sample_temperature is not None else 0.0,
            seed=agent_config.sample_seed,
            succeeded=run_result.succeeded,
            failure_reason=run_result.failure_reason,
            answer=run_result.answer,
            step_count=len(run_result.steps),
        )
        samples.append(record)
        chosen_steps_by_index[sample_index] = list(run_result.steps)

    aggregator = config.aggregator.lower().strip()
    if aggregator == "first_success":
        return _aggregate_first_success(
            task_id=task.task_id,
            samples=samples,
            chosen_steps_by_index=chosen_steps_by_index,
        )
    return _aggregate_column_vote(
        task_id=task.task_id,
        samples=samples,
        config=config,
        numeric_tolerance=numeric_tolerance,
        case_insensitive=case_insensitive,
        strip_whitespace=strip_whitespace,
        chosen_steps_by_index=chosen_steps_by_index,
    )


def write_trace_payload(result: SelfConsistencyResult, *, base: dict[str, Any]) -> dict[str, Any]:
    """Merge a self-consistency outcome into the task trace dict."""
    enriched = dict(base)
    enriched["self_consistency"] = result.to_dict()
    return enriched
