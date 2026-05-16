from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TextIO

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Single native tool call returned by an OpenAI-compatible model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str
    # Assistant messages: any tool_calls the model emitted on its previous
    # turn. We replay them so the API can match the next tool result back
    # to the right call id.
    tool_calls: tuple[ToolCall, ...] = ()
    # Tool messages: id of the call we're answering and the tool name (the
    # API requires both when replaying a conversation that used tools).
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Structured completion result. ``text`` is the assistant text content
    (may be empty when the model only emitted tool_calls); ``tool_calls``
    is the list of native tool invocations on this turn."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(
        self,
        messages: list[ModelMessage],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        stream_label: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    def complete_with_tools(
        self,
        messages: list[ModelMessage],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        seed: int | None = None,
        stream_label: str | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Optional: native tool-calling completion. Adapters that don't
        implement this raise NotImplementedError; callers can fall back."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Streaming sink (set globally by CLI; nil-cost when not enabled)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StreamSink:
    """Minimal sink for live model output, used when ``stream_thoughts`` is on.

    Each ``begin/write/end`` pair brackets one model call so different LLM
    calls (planner / specialist / synthesizer / table_llm) don't get
    interleaved on the screen.
    """

    fp: TextIO = sys.stderr
    enabled: bool = True

    def begin(self, label: str) -> None:
        if not self.enabled:
            return
        self.fp.write(f"\n\x1b[2m» {label}\x1b[0m ")
        self.fp.flush()

    def write(self, text: str) -> None:
        if not self.enabled or not text:
            return
        self.fp.write(text)
        self.fp.flush()

    def end(self) -> None:
        if not self.enabled:
            return
        self.fp.write("\n")
        self.fp.flush()


# Process-wide singleton, set by the CLI when --stream is passed. We keep
# it module-level so adapters constructed deep inside the runner / agents
# don't need it threaded through every function signature.
_STREAM_SINK: StreamSink | None = None


def set_stream_sink(sink: StreamSink | None) -> None:
    global _STREAM_SINK
    _STREAM_SINK = sink


def get_stream_sink() -> StreamSink | None:
    return _STREAM_SINK


def _message_to_dict(message: ModelMessage) -> dict[str, Any]:
    """Serialize a ModelMessage in the shape the OpenAI Chat Completions
    API expects, including assistant ``tool_calls`` and tool-result rows."""
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            payload["name"] = message.name
    return payload


def _parse_tool_call(raw: Any) -> ToolCall:
    """Lift one OpenAI tool_call object into our ToolCall dataclass."""
    call_id = getattr(raw, "id", None) or ""
    function = getattr(raw, "function", None)
    if function is None:
        raise RuntimeError("Tool call missing function payload.")
    name = getattr(function, "name", None) or ""
    raw_arguments = getattr(function, "arguments", "") or ""
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Tool call {name!r} arguments were not valid JSON: {exc}; raw={raw_arguments!r}"
            ) from exc
    if not isinstance(arguments, dict):
        raise RuntimeError(
            f"Tool call {name!r} arguments must decode to an object, got {type(arguments).__name__}."
        )
    return ToolCall(id=call_id, name=name, arguments=arguments)


_RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether a failure looks transient enough to retry.

    Retryable: network errors, timeouts, rate limits, 5xx-class HTTP.
    Not retryable: 4xx auth / validation errors that won't pass on a retry.
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in _RETRYABLE_HTTP_STATUS
    if isinstance(exc, APIError):
        # The base APIError includes things like server-side aborts.
        status = getattr(exc, "status_code", None)
        if status is None:
            return True  # generic APIError without status; assume transient
        return status in _RETRYABLE_HTTP_STATUS
    return False


class OpenAIModelAdapter:
    """Adapter for any OpenAI-compatible chat-completions endpoint.

    Same code path serves DashScope (Qwen-Plus), vLLM, and Ollama-style
    servers. For local vLLM, point ``api_base`` to ``http://localhost:8000/v1``
    and set ``api_key`` to any non-empty placeholder.

    Transient failures (network, timeout, 429, 5xx) are retried with
    exponential backoff + jitter. Permanent failures (401, 400, etc.)
    raise immediately so they show up in ``trace.json`` and aren't masked.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        max_tokens: int | None = None,
        request_timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 16.0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.retry_backoff_max_seconds = max(retry_backoff_seconds, retry_backoff_max_seconds)

    def _client(self) -> OpenAI:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")
        return OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.request_timeout,
        )

    def _backoff_seconds(self, attempt_index: int) -> float:
        """Exponential backoff with full jitter."""
        base = self.retry_backoff_seconds * (2 ** attempt_index)
        capped = min(base, self.retry_backoff_max_seconds)
        return random.uniform(0.0, capped)

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        stream_label: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        from data_agent_baseline.budget import get_budget_controller

        label = stream_label or self.model
        budget = get_budget_controller()
        if budget is not None:
            budget.consume_llm(label)
        sink = get_stream_sink()
        # Stream only when a sink is installed; otherwise stay on the
        # blocking path so existing callers don't pay the streaming cost.
        if sink is not None and sink.enabled:
            return self._complete_streaming(
                messages=messages,
                temperature=temperature,
                seed=seed,
                sink=sink,
                stream_label=label,
                max_tokens=max_tokens,
            )
        return self._complete_blocking(
            messages=messages, temperature=temperature, seed=seed, max_tokens=max_tokens
        )

    def _build_request(
        self,
        messages: list[ModelMessage],
        *,
        temperature: float | None,
        seed: int | None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_dict(m) for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
        }
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        return kwargs

    def _complete_blocking(
        self,
        *,
        messages: list[ModelMessage],
        temperature: float | None,
        seed: int | None,
        max_tokens: int | None = None,
    ) -> str:
        client = self._client()
        request_kwargs = self._build_request(
            messages, temperature=temperature, seed=seed, max_tokens=max_tokens
        )

        last_exc: BaseException | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(**request_kwargs)
            except APIError as exc:
                last_exc = exc
                if attempt >= attempts - 1 or not _is_retryable(exc):
                    raise RuntimeError(
                        f"Model request failed (attempt {attempt + 1}/{attempts}): {exc}"
                    ) from exc
                time.sleep(self._backoff_seconds(attempt))
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= attempts - 1:
                    raise RuntimeError(
                        f"Model request failed (attempt {attempt + 1}/{attempts}): {exc}"
                    ) from exc
                time.sleep(self._backoff_seconds(attempt))
                continue

            choices = response.choices or []
            if not choices:
                raise RuntimeError("Model response missing choices.")
            content = choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("Model response missing text content.")
            return content

        raise RuntimeError(
            f"Model request failed after {attempts} attempts: {last_exc}"
        )

    def complete_with_tools(
        self,
        messages: list[ModelMessage],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        seed: int | None = None,
        stream_label: str | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        from data_agent_baseline.budget import get_budget_controller

        label = stream_label or self.model
        budget = get_budget_controller()
        if budget is not None:
            budget.consume_llm(label)

        client = self._client()
        request_kwargs = self._build_request(
            messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )

        last_exc: BaseException | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(**request_kwargs)
            except APIError as exc:
                last_exc = exc
                if attempt >= attempts - 1 or not _is_retryable(exc):
                    raise RuntimeError(
                        f"Tool-call model request failed (attempt {attempt + 1}/{attempts}): {exc}"
                    ) from exc
                time.sleep(self._backoff_seconds(attempt))
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= attempts - 1:
                    raise RuntimeError(
                        f"Tool-call model request failed (attempt {attempt + 1}/{attempts}): {exc}"
                    ) from exc
                time.sleep(self._backoff_seconds(attempt))
                continue

            choices = response.choices or []
            if not choices:
                raise RuntimeError("Model response missing choices.")
            message = choices[0].message
            text_content = message.content if isinstance(message.content, str) else ""
            raw_tool_calls = getattr(message, "tool_calls", None) or []
            parsed_tool_calls = tuple(_parse_tool_call(tc) for tc in raw_tool_calls)
            # Optional stream-side surface so the user sees that *something*
            # came back from the model — content for chatty replies,
            # otherwise a synthetic note listing the called tools.
            sink = get_stream_sink()
            if sink is not None and sink.enabled:
                sink.begin(f"{label} · {self.model}")
                try:
                    if text_content:
                        sink.write(text_content)
                    elif parsed_tool_calls:
                        names = ", ".join(call.name for call in parsed_tool_calls)
                        sink.write(f"[tool_calls: {names}]")
                finally:
                    sink.end()
            return ModelResponse(text=text_content, tool_calls=parsed_tool_calls)

        raise RuntimeError(
            f"Tool-call model request failed after {attempts} attempts: {last_exc}"
        )

    def _complete_streaming(
        self,
        *,
        messages: list[ModelMessage],
        temperature: float | None,
        seed: int | None,
        sink: StreamSink,
        stream_label: str,
        max_tokens: int | None = None,
    ) -> str:
        client = self._client()
        request_kwargs = self._build_request(
            messages, temperature=temperature, seed=seed, max_tokens=max_tokens
        )
        request_kwargs["stream"] = True

        last_exc: BaseException | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                stream = client.chat.completions.create(**request_kwargs)
            except APIError as exc:
                last_exc = exc
                if attempt >= attempts - 1 or not _is_retryable(exc):
                    raise RuntimeError(
                        f"Model request failed (attempt {attempt + 1}/{attempts}): {exc}"
                    ) from exc
                time.sleep(self._backoff_seconds(attempt))
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= attempts - 1:
                    raise RuntimeError(
                        f"Model request failed (attempt {attempt + 1}/{attempts}): {exc}"
                    ) from exc
                time.sleep(self._backoff_seconds(attempt))
                continue

            chunks: list[str] = []
            reasoning_chunks: list[str] = []
            sink.begin(f"{stream_label} · {self.model}")
            try:
                for event in stream:
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta
                    text_chunk = getattr(delta, "content", None)
                    if text_chunk:
                        chunks.append(text_chunk)
                        sink.write(text_chunk)
                    # Qwen-QwQ style: chain-of-thought arrives in
                    # `reasoning_content`. Show it dimmed so the user sees
                    # thinking too.
                    reasoning_chunk = getattr(delta, "reasoning_content", None)
                    if reasoning_chunk:
                        reasoning_chunks.append(reasoning_chunk)
                        sink.write(reasoning_chunk)
            finally:
                sink.end()
            content = "".join(chunks)
            if not content:
                # Some OpenAI-compatible endpoints stream provider-specific
                # reasoning fields but omit the final content delta. For
                # executable routes, retry once through the blocking path so
                # callers still receive the final program / JSON payload.
                if reasoning_chunks:
                    from data_agent_baseline.budget import get_budget_controller

                    budget = get_budget_controller()
                    if budget is not None:
                        budget.consume_llm(f"{stream_label} blocking-fallback")
                    return self._complete_blocking(
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=max_tokens,
                    )
                raise RuntimeError("Streaming model response missing text content.")
            return content

        raise RuntimeError(
            f"Streaming model request failed after {attempts} attempts: {last_exc}"
        )


class ScriptedModelAdapter:
    """Test adapter that replays a fixed list of responses.

    Each entry in ``responses`` is either a raw text string (returned by
    ``complete``) or a ``ModelResponse`` (returned by both ``complete``
    via its ``text`` field and ``complete_with_tools`` unchanged). This
    lets the same fixture drive both legacy text-mode and the new
    native-tool-call path."""

    def __init__(self, responses: list[str | ModelResponse]) -> None:
        self._responses: list[str | ModelResponse] = list(responses)

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        stream_label: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        del messages, temperature, seed, stream_label, max_tokens
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        item = self._responses.pop(0)
        if isinstance(item, ModelResponse):
            return item.text
        return item

    def complete_with_tools(
        self,
        messages: list[ModelMessage],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        seed: int | None = None,
        stream_label: str | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        del messages, tools, tool_choice, temperature, seed, stream_label, max_tokens
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        item = self._responses.pop(0)
        if isinstance(item, ModelResponse):
            return item
        return ModelResponse(text=item, tool_calls=())
