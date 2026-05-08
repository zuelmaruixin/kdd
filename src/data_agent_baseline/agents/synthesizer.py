"""Synthesizer agent: turns plan + findings into the final AnswerTable.

The synthesizer is the only agent that may call the official `answer`
tool. It runs a tightly-scoped ReAct loop with the full tool set so it
can re-verify a finding (e.g. re-execute a SQL query) before submitting,
but its prompt strongly biases it toward a *single-step* answer when the
findings already cover the question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.planning import Finding, Plan
from data_agent_baseline.agents.prompt import RESPONSE_EXAMPLES, build_observation_prompt
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import create_default_tool_registry


SYNTHESIZER_SYSTEM_PROMPT = """
You are the SYNTHESIZER inside a multi-agent Data Agent. The planner
produced a Plan, specialist workers executed every sub-task, and you
now have all their findings. Your only job is to emit the final answer
TABLE for the original question.

Critical scoring details (read carefully):
- The grader matches columns by content signature ONLY. Column names
  and row order are ignored. Do not pad with extra columns.
- score = max(0, recall - lambda * extra_cols / pred_cols). Penalty
  per redundant column is small but non-zero.
- Numeric values are matched up to a tolerance, strings are compared
  case-insensitively after trimming whitespace.

How to behave:
- If the upstream findings already contain the answer table, prefer to
  emit it immediately by calling the `answer` tool. Do NOT redo work.
- If a finding looks wrong or incomplete, you may use the available
  tools (read_csv, read_json, read_doc, inspect_sqlite_schema,
  execute_context_sql, execute_python) to verify or compute the
  missing piece. Stay focused; do not explore beyond what is needed.
- Output exactly one JSON action per step inside a single ```json
  fenced block, with keys `thought`, `action`, `action_input`.
- The task is complete only when you call `answer` with `columns` and
  `rows`. Output ONLY the columns the question asks for.
- Before planning or answering, identify the requested answer type:
count, ratio, difference, sum, average, max/min, list, boolean, etc.
Then ensure the final computation matches that answer type.

""".strip()


def _render_findings(findings: dict[str, Finding], *, max_chars_per_finding: int = 2000) -> str:
    if not findings:
        return "(no findings)"
    parts: list[str] = []
    for finding in findings.values():
        block = finding.render_for_dependency()
        if len(block) > max_chars_per_finding:
            block = block[:max_chars_per_finding] + "\n  ... (truncated)"
        parts.append(block)
    return "\n\n".join(parts)


@dataclass(slots=True)
class SynthesizerConfig:
    max_steps: int = 6
    sample_temperature: float | None = None
    sample_seed: int | None = None


@dataclass(slots=True)
class SynthesizerAgent:
    model: ModelAdapter
    config: SynthesizerConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = SynthesizerConfig()

    def synthesize(
        self,
        *,
        task: PublicTask,
        plan: Plan,
        findings: dict[str, Finding],
    ) -> AgentRunResult:
        registry = create_default_tool_registry()
        system_prompt = (
            f"{SYNTHESIZER_SYSTEM_PROMPT}\n\n"
            "Available tools:\n"
            f"{registry.describe_for_prompt()}\n\n"
            f"{RESPONSE_EXAMPLES}\n\n"
            "Always emit one ```json fenced object per step."
        )

        plan_text = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)
        findings_text = _render_findings(findings)
        question_block = (
            f"Original question: {task.question}\n"
            f"Difficulty: {task.difficulty}\n\n"
            f"Plan that was executed:\n{plan_text}\n\n"
            f"Findings collected from specialists:\n{findings_text}\n\n"
            "Submit the final answer TABLE via the `answer` tool. Output "
            "ONLY the columns the question asks for."
        )

        agent = ReActAgent(
            model=self.model,
            tools=registry,
            config=ReActAgentConfig(
                max_steps=self.config.max_steps,
                sample_temperature=self.config.sample_temperature,
                sample_seed=self.config.sample_seed,
            ),
            system_prompt=system_prompt,
            stream_label_prefix="synthesizer",
        )

        proxy = _ScopedTask(base=task, scoped_question=question_block)
        result = agent.run(proxy)

        from data_agent_baseline.progress import get_progress_logger
        logger = get_progress_logger()
        if logger is not None and result.answer is not None:
            logger.synthesizer_done(
                columns=list(result.answer.columns),
                row_count=len(result.answer.rows),
            )
        return result


class _ScopedTask:
    __slots__ = ("base", "scoped_question")

    def __init__(self, *, base: PublicTask, scoped_question: str) -> None:
        self.base = base
        self.scoped_question = scoped_question

    @property
    def task_id(self) -> str:
        return self.base.task_id

    @property
    def difficulty(self) -> str:
        return self.base.difficulty

    @property
    def question(self) -> str:
        return self.scoped_question

    @property
    def task_dir(self):
        return self.base.task_dir

    @property
    def context_dir(self):
        return self.base.context_dir

    @property
    def record(self):
        return self.base.record

    @property
    def assets(self):
        return self.base.assets
