from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    ModelStep,
    ToolCall,
)
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.react_planner import (
    PlannerConfig,
    ReactPlan,
    make_plan,
)
from data_agent_baseline.agents.react_retry_hints import (
    ErrorHistory,
    build_retry_hint,
    build_verify_instructions,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.budget import BudgetExceeded
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 40
    sample_temperature: float | None = None  # override for self-consistency sampling
    sample_seed: int | None = None
    # When True, repeated calls to the same read-only tool with the same
    # arguments inside one ReAct loop are served from an in-memory cache.
    # This stops the model from re-reading the same csv 5 times and burning
    # tokens on identical observations.
    cache_tool_results: bool = True
    # Harness-style self-verification. After the first `answer` call we
    # don't terminate — we feed back a verification prompt and require the
    # model to call `answer` again. `verification_rounds=1` means one
    # extra answer call (i.e. total 2). Set to 0 to terminate on first
    # answer (the cheap React guard still runs and may escalate one
    # conditional round when it sees concrete evidence of risk).
    verification_rounds: int = 0
    # When True, run the deterministic React answer guard on the first
    # `answer` and force one self-verify round if the guard flags risk.
    # Independent of ``verification_rounds`` — even with rounds=0 we
    # still benefit from a conditional check when it's warranted.
    use_answer_guard: bool = True
    # Pre-loop planner. When enabled, a single LLM call sketches an
    # ordered list of subtasks before the React loop runs. The plan is
    # injected as a user message prefix and persisted to trace.json.
    # Failures degrade gracefully — the loop runs without a plan.
    planner: PlannerConfig = PlannerConfig()
    # When True, the React loop talks to the model through the native
    # OpenAI tool-calling API (``tools=[...] + tool_choice="auto"``)
    # instead of asking the model to emit a JSON object in text. The
    # legacy text path stays as a fallback: if the model returns text
    # without any tool_calls we parse it the old way. Set False to force
    # the legacy path (useful for endpoints that don't support tools).
    use_native_tool_calls: bool = True


# Tools that are pure functions of (task.context_dir, action_input) and
# safe to cache. `execute_python` is excluded because the LLM may rely on
# side effects (printing different things) across calls. `answer` is the
# terminator and obviously must not be cached.
_CACHEABLE_TOOLS: frozenset[str] = frozenset({
    "list_context",
    "read_csv",
    "read_json",
    "read_doc",
    "head_doc",
    "grep_doc",
    "inspect_sqlite_schema",
    "execute_context_sql",
    "consult_knowledge",
})


def _cache_key(action: str, action_input: dict[str, object]) -> str:
    """Stable JSON-string key for a tool call (sorted keys handles dict reordering)."""
    try:
        return action + "::" + json.dumps(action_input, ensure_ascii=False, sort_keys=True)
    except TypeError:
        # Non-JSON-serializable args: fall through, key uses repr().
        return action + "::" + repr(sorted(action_input.items()))


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _escape_control_chars_inside_json_strings(text: str) -> str:
    """Repair common LLM JSON mistakes like literal newlines in code strings."""
    chars: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if not in_string:
            chars.append(char)
            if char == '"':
                in_string = True
            continue

        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\":
            chars.append(char)
            escaped = True
            continue
        if char == '"':
            chars.append(char)
            in_string = False
            continue
        if char == "\n":
            chars.append("\\n")
            continue
        if char == "\r":
            chars.append("\\r")
            continue
        if char == "\t":
            chars.append("\\t")
            continue
        chars.append(char)
    return "".join(chars)


def _append_missing_json_closers(text: str) -> str:
    """Append missing object/array closers when the model truncates final braces."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
    if not stack:
        return text
    return text + "".join(reversed(stack))


def _load_single_json_object(text: str) -> dict[str, object]:
    start = text.find("{")
    if start < 0:
        raise ValueError("Model response must contain a JSON object.")
    candidate = text[start:]
    try:
        payload, end = json.JSONDecoder().raw_decode(candidate)
        decoded_text = candidate
    except json.JSONDecodeError:
        decoded_text = _escape_control_chars_inside_json_strings(candidate)
        try:
            payload, end = json.JSONDecoder().raw_decode(decoded_text)
        except json.JSONDecodeError:
            decoded_text = _append_missing_json_closers(decoded_text)
            payload, end = json.JSONDecoder().raw_decode(decoded_text)
    remainder = decoded_text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


def _model_step_from_tool_call(call: ToolCall, *, text: str) -> ModelStep:
    """Lift a native tool_call into the ModelStep shape the rest of the
    loop expects. The optional free-form ``text`` becomes the thought."""
    return ModelStep(
        thought=text.strip(),
        action=call.name,
        action_input=dict(call.arguments),
        raw_response=text,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        stream_label_prefix: str = "react",
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.stream_label_prefix = stream_label_prefix

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
            use_native_tool_calls=self.config.use_native_tool_calls,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        # If the pre-loop planner produced a plan, surface it as a user
        # message right after the task prompt so the model can refer to
        # it on every subsequent turn (the harness re-builds the message
        # list each step).
        plan_message = (state.plan or {}).get("user_message") if state.plan else None
        if isinstance(plan_message, str) and plan_message.strip():
            messages.append(ModelMessage(role="user", content=plan_message))
        for step in state.steps:
            if step.tool_call_id is not None:
                # Native tool-call replay: assistant emits the call, the
                # tool message replies with the captured observation.
                assistant = ModelMessage(
                    role="assistant",
                    content=step.thought or "",
                    tool_calls=(
                        ToolCall(
                            id=step.tool_call_id,
                            name=step.action,
                            arguments=dict(step.action_input),
                        ),
                    ),
                )
                tool_content = build_observation_prompt(step.observation)
                messages.append(assistant)
                messages.append(
                    ModelMessage(
                        role="tool",
                        content=tool_content,
                        tool_call_id=step.tool_call_id,
                        name=step.action,
                    )
                )
            else:
                # Text-mode fallback: legacy assistant-JSON + user
                # observation pair. Also used when the model emitted
                # plain text in native mode (no tool_calls returned).
                messages.append(ModelMessage(role="assistant", content=step.raw_response))
                messages.append(
                    ModelMessage(role="user", content=build_observation_prompt(step.observation))
                )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        # One LLM call BEFORE the loop runs. The plan is advisory only —
        # if it fails to parse or the call errors, we log it on state
        # and the loop runs unchanged.
        plan = make_plan(
            config=self.config.planner,
            model=self.model,
            task=task,
            stream_label=f"{self.stream_label_prefix}.planner",
        )
        if plan is not None:
            user_message = plan.to_user_message()
            state.plan = plan.to_dict()
            if user_message:
                state.plan["user_message"] = user_message
            from data_agent_baseline.progress import get_progress_logger
            progress = get_progress_logger()
            if progress is not None and hasattr(progress, "react_plan"):
                try:
                    progress.react_plan(
                        prefix=self.stream_label_prefix,
                        subtask_count=len(plan.subtasks),
                        rationale=plan.rationale,
                        error=plan.error,
                    )
                except Exception:  # noqa: BLE001
                    pass

        # Per-loop tool result cache: maps a stable key derived from
        # (action, action_input) to the observation we emitted last time.
        # We additionally remember whether the cached call was terminal so
        # we never accidentally re-trigger termination on a cache hit.
        tool_cache: dict[str, dict[str, object]] = {}
        last_error: str | None = None
        # Self-verify accounting: how many `answer` calls have we let
        # through, and how many we need before we terminate.
        answers_seen = 0
        # Baseline rounds from config. The cheap guard may bump this
        # for THIS task only if the first draft answer looks risky.
        required_answers = max(0, self.config.verification_rounds) + 1
        guard_escalated = False
        # Most recent draft answer, retained so we can fall back to it if
        # the model fails to call answer again after self-verify reflection.
        pending_answer = None
        # Tracks failed-tool error signatures so the harness can label
        # repeated identical failures and nudge the model to switch
        # approach instead of retrying blindly.
        error_history = ErrorHistory()
        # Most recent guard verdict, surfaced into the verify observation
        # so the model knows WHICH risks to address (not just "re-check").
        last_guard_summary: dict[str, object] | None = None

        for step_index in range(1, self.config.max_steps + 1):
            raw_response = ""
            tool_call_id: str | None = None
            preparsed_step: ModelStep | None = None
            try:
                if self.config.use_native_tool_calls:
                    response = self.model.complete_with_tools(
                        self._build_messages(task, state),
                        tools=self.tools.describe_for_tool_api(),
                        tool_choice="auto",
                        temperature=self.config.sample_temperature,
                        seed=self.config.sample_seed,
                        stream_label=f"{self.stream_label_prefix} step {step_index}",
                    )
                    raw_response = response.text or ""
                    if response.tool_calls:
                        # One step per turn. If the model emitted multiple
                        # tool_calls we only honor the first; the surplus is
                        # silently dropped (no orphan: we never replay them
                        # because they're never recorded in state.steps).
                        call = response.tool_calls[0]
                        tool_call_id = call.id
                        preparsed_step = _model_step_from_tool_call(call, text=response.text)
                else:
                    raw_response = self.model.complete(
                        self._build_messages(task, state),
                        temperature=self.config.sample_temperature,
                        seed=self.config.sample_seed,
                        stream_label=f"{self.stream_label_prefix} step {step_index}",
                    )
            except BudgetExceeded as budget_exc:
                # If we already have a self-verify draft, the budget timeout
                # interrupted us mid-verification. Don't throw away the
                # draft — commit it and let the run finish gracefully.
                if pending_answer is not None:
                    state.answer = pending_answer
                    state.failure_reason = (
                        f"budget_exceeded_during_self_verify: {budget_exc}; "
                        "committed most recent draft answer."
                    )
                    break
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"model_error: {exc}"
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__model_error__",
                        action_input={},
                        raw_response="",
                        observation={"ok": False, "error": str(exc)},
                        ok=False,
                    )
                )
                state.failure_reason = last_error
                break
            try:
                if preparsed_step is not None:
                    model_step = preparsed_step
                else:
                    # Either text-mode, or native-mode that emitted text
                    # without tool_calls — both fall through to the legacy
                    # JSON parser. The model gets one shot; if its text
                    # isn't parseable the outer except records an error
                    # step and the retry-hint pipeline takes over.
                    model_step = parse_model_step(raw_response)

                cache_hit = False
                cache_key: str | None = None
                if (
                    self.config.cache_tool_results
                    and model_step.action in _CACHEABLE_TOOLS
                ):
                    cache_key = _cache_key(model_step.action, model_step.action_input)
                    cached = tool_cache.get(cache_key)
                    if cached is not None:
                        cache_hit = True
                        observation = {
                            "ok": True,
                            "tool": model_step.action,
                            "cached": True,
                            "content": cached["content"],
                        }
                        step_record = StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=True,
                            tool_call_id=tool_call_id,
                        )
                        state.steps.append(step_record)

                if not cache_hit:
                    tool_result = self.tools.execute(
                        task, model_step.action, model_step.action_input
                    )

                    is_answer_call = (
                        tool_result.is_terminal and tool_result.answer is not None
                    )
                    final_answer_call = False
                    if is_answer_call:
                        answers_seen += 1
                        pending_answer = tool_result.answer
                        # On the FIRST answer with rounds=0, consult the
                        # cheap deterministic guard. Only escalate when
                        # it finds concrete risk — otherwise commit now.
                        # See agents/react_answer_guard.py for signals.
                        if (
                            self.config.use_answer_guard
                            and answers_seen == 1
                            and not guard_escalated
                            and required_answers == 1
                        ):
                            from data_agent_baseline.agents.react_answer_guard import (
                                assess_react_answer_risk,
                                summarize_assessment,
                            )
                            # Build the step list as the guard will see it
                            # (the current answer step is recorded below,
                            # so include it explicitly).
                            preview_step = StepRecord(
                                step_index=step_index,
                                thought=model_step.thought,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                raw_response=raw_response,
                                observation={
                                    "ok": True,
                                    "tool": "answer",
                                    "content": {"draft": True},
                                },
                                ok=True,
                            )
                            assessment = assess_react_answer_risk(
                                task=task,
                                draft=tool_result.answer,
                                steps=list(state.steps) + [preview_step],
                            )
                            summary = summarize_assessment(assessment)
                            last_guard_summary = summary
                            from data_agent_baseline.progress import get_progress_logger
                            logger = get_progress_logger()
                            if logger is not None:
                                logger.react_answer_guard(
                                    prefix=self.stream_label_prefix,
                                    step_index=step_index,
                                    verdict="escalate" if assessment.should_escalate else "pass",
                                    score=summary["score"],
                                    risk_codes=summary["risk_codes"],
                                    top_risk=summary["top_risk"],
                                )
                            if assessment.should_escalate:
                                required_answers = 2
                                guard_escalated = True
                        final_answer_call = answers_seen >= required_answers

                    if is_answer_call and not final_answer_call:
                        # Replace the would-be terminal observation with a
                        # self-verification prompt: force the model to look
                        # at its own draft one more time before we commit.
                        # When the guard fired we surface its concrete
                        # findings (risk codes + top risk) so the verify
                        # round has a real target, not a generic re-check.
                        verify_content: dict[str, object] = {
                            "status": "draft_submitted",
                            "verification_round": answers_seen,
                            "remaining_rounds": required_answers - answers_seen,
                            "draft_columns": list(tool_result.answer.columns),
                            "draft_row_count": len(tool_result.answer.rows),
                        }
                        if last_guard_summary is not None:
                            verify_content["guard_score"] = last_guard_summary.get("score")
                            verify_content["guard_risk_codes"] = (
                                last_guard_summary.get("risk_codes") or []
                            )
                            top_risk = last_guard_summary.get("top_risk")
                            if top_risk:
                                verify_content["guard_top_risk"] = top_risk
                            verify_content["guard_escalated"] = guard_escalated
                        verify_content["instructions"] = build_verify_instructions(
                            last_guard_summary or {}
                        )
                        observation = {
                            "ok": True,
                            "tool": model_step.action,
                            "content": verify_content,
                        }
                        step_record = StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=True,
                            tool_call_id=tool_call_id,
                        )
                        state.steps.append(step_record)

                        from data_agent_baseline.progress import get_progress_logger
                        logger = get_progress_logger()
                        if logger is not None:
                            logger.react_step(
                                prefix=self.stream_label_prefix,
                                step_index=step_index,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                ok=True,
                                cached=False,
                            )
                        continue

                    observation = {
                        "ok": tool_result.ok,
                        "tool": model_step.action,
                        "content": tool_result.content,
                    }
                    if not tool_result.ok:
                        retry_hint = build_retry_hint(
                            action=model_step.action,
                            action_input=model_step.action_input,
                            observation=observation,
                            history=error_history,
                        )
                        if retry_hint is not None:
                            observation["retry_hint"] = retry_hint.to_dict()
                    step_record = StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                        tool_call_id=tool_call_id,
                    )
                    state.steps.append(step_record)

                    # Cache only successful calls. Don't cache the terminal
                    # tool (`answer`) — it terminates the loop anyway.
                    if (
                        cache_key is not None
                        and tool_result.ok
                        and not tool_result.is_terminal
                    ):
                        tool_cache[cache_key] = {"content": tool_result.content}

                    if tool_result.is_terminal:
                        state.answer = tool_result.answer
                        # Emit progress event before breaking out of the loop.
                        from data_agent_baseline.progress import get_progress_logger
                        logger = get_progress_logger()
                        if logger is not None:
                            logger.react_step(
                                prefix=self.stream_label_prefix,
                                step_index=step_index,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                ok=tool_result.ok,
                                cached=False,
                            )
                        break

                from data_agent_baseline.progress import get_progress_logger
                logger = get_progress_logger()
                if logger is not None:
                    logger.react_step(
                        prefix=self.stream_label_prefix,
                        step_index=step_index,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        ok=cache_hit or step_record.ok,
                        cached=cache_hit,
                    )
            except BudgetExceeded as budget_exc:
                if pending_answer is not None:
                    state.answer = pending_answer
                    state.failure_reason = (
                        f"budget_exceeded_during_self_verify: {budget_exc}; "
                        "committed most recent draft answer."
                    )
                    break
                raise
            except Exception as exc:
                last_error = str(exc)
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                # Even when the tool call raised before we got a result,
                # the model still benefits from a structured hint: the
                # most common cause is bad action_input (e.g. wrong
                # column shape on `answer`) and the helper handles that.
                failed_action = "__error__"
                failed_action_input: dict[str, object] = {}
                failed_thought = ""
                if 'model_step' in locals():
                    try:
                        failed_action = model_step.action or failed_action
                        failed_action_input = dict(model_step.action_input or {})
                        failed_thought = model_step.thought or ""
                    except Exception:  # noqa: BLE001
                        pass
                retry_hint = build_retry_hint(
                    action=failed_action,
                    action_input=failed_action_input,
                    observation=observation,
                    history=error_history,
                )
                if retry_hint is not None:
                    observation["retry_hint"] = retry_hint.to_dict()
                # When the model used a native tool_call, keep the action
                # name + id so the next-turn replay sends back a proper
                # ``tool`` message instead of orphaning the call (which
                # would make the API reject the request).
                record_action = (
                    failed_action if tool_call_id is not None else "__error__"
                )
                record_input = (
                    failed_action_input if tool_call_id is not None else {}
                )
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=failed_thought,
                        action=record_action,
                        action_input=record_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                        tool_call_id=tool_call_id,
                    )
                )

        if state.answer is None and pending_answer is not None:
            # Self-verify cycle was started but the model never confirmed.
            # Don't waste a partially-correct draft — fall back to it and
            # record that we couldn't run the full verification.
            state.answer = pending_answer
            state.failure_reason = (
                "Agent did not finish self-verification; falling back to "
                "the most recent draft answer."
            )
        elif state.answer is None and state.failure_reason is None:
            suffix = f" Last error: {last_error}" if last_error else ""
            state.failure_reason = f"Agent did not submit an answer within max_steps.{suffix}"

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            plan=state.plan,
        )
