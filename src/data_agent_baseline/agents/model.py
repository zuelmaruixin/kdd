from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
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
class ModelMessage:
    role: str
    content: str


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
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": self.temperature if temperature is None else temperature,
        }
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens
        if seed is not None:
            kwargs["seed"] = seed
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
                    # DeepSeek-Reasoner / Qwen-QwQ style: chain-of-thought
                    # arrives in `reasoning_content`. Show it dimmed so the
                    # user sees thinking too.
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
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

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
        return self._responses.pop(0)
