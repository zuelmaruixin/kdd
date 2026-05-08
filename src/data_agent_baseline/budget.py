"""Per-task budget controller.

The controller is intentionally global within a worker process, mirroring
the existing stream/progress singletons. This lets the real call sites
(`OpenAIModelAdapter.complete` and `ToolRegistry.execute`) enforce the
budget without plumbing counters through every agent layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised when a task has exhausted its configured budget."""


@dataclass(slots=True)
class BudgetController:
    task_id: str
    max_llm_calls: int
    max_tool_calls: int
    max_seconds: float = 0.0
    # Repair-loop budgets — distinct from raw call budgets so a noisy
    # repair loop can't burn the whole task's call budget.
    max_local_repairs: int = 3
    max_reasoner_repairs: int = 1
    max_multiagent_fallbacks: int = 1
    llm_calls: int = 0
    tool_calls: int = 0
    local_repairs: int = 0
    reasoner_repairs: int = 0
    multiagent_fallbacks: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def _elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def _check_time(self) -> None:
        if self.max_seconds > 0 and self._elapsed() > self.max_seconds:
            raise BudgetExceeded(
                f"budget_exhausted: time>{self.max_seconds:.1f}s for {self.task_id}"
            )

    def consume_llm(self, label: str) -> None:
        self._check_time()
        if self.max_llm_calls >= 0 and self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded(
                f"budget_exhausted: llm_calls>{self.max_llm_calls} before {label}"
            )
        self.llm_calls += 1
        self.events.append({
            "kind": "llm",
            "label": label,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": round(self._elapsed(), 3),
        })

    def consume_tool(self, name: str) -> None:
        self._check_time()
        if self.max_tool_calls >= 0 and self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded(
                f"budget_exhausted: tool_calls>{self.max_tool_calls} before {name}"
            )
        self.tool_calls += 1
        self.events.append({
            "kind": "tool",
            "label": name,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": round(self._elapsed(), 3),
        })

    # Repair-loop counters. Callers use ``can_*`` to peek before doing
    # work, then ``consume_*`` once they actually start the round.
    def can_local_repair(self) -> bool:
        return self.local_repairs < max(self.max_local_repairs, 0)

    def consume_local_repair(self, label: str) -> None:
        if not self.can_local_repair():
            raise BudgetExceeded(
                f"budget_exhausted: local_repairs>{self.max_local_repairs} before {label}"
            )
        self.local_repairs += 1
        self.events.append({
            "kind": "local_repair",
            "label": label,
            "local_repairs": self.local_repairs,
            "elapsed_seconds": round(self._elapsed(), 3),
        })

    def can_reasoner_repair(self) -> bool:
        return self.reasoner_repairs < max(self.max_reasoner_repairs, 0)

    def consume_reasoner_repair(self, label: str) -> None:
        if not self.can_reasoner_repair():
            raise BudgetExceeded(
                f"budget_exhausted: reasoner_repairs>{self.max_reasoner_repairs} before {label}"
            )
        self.reasoner_repairs += 1
        self.events.append({
            "kind": "reasoner_repair",
            "label": label,
            "reasoner_repairs": self.reasoner_repairs,
            "elapsed_seconds": round(self._elapsed(), 3),
        })

    def can_multiagent_fallback(self) -> bool:
        return self.multiagent_fallbacks < max(self.max_multiagent_fallbacks, 0)

    def consume_multiagent_fallback(self, label: str) -> None:
        if not self.can_multiagent_fallback():
            raise BudgetExceeded(
                f"budget_exhausted: multiagent_fallbacks>{self.max_multiagent_fallbacks} before {label}"
            )
        self.multiagent_fallbacks += 1
        self.events.append({
            "kind": "multiagent_fallback",
            "label": label,
            "multiagent_fallbacks": self.multiagent_fallbacks,
            "elapsed_seconds": round(self._elapsed(), 3),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "max_llm_calls": self.max_llm_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_seconds": self.max_seconds,
            "max_local_repairs": self.max_local_repairs,
            "max_reasoner_repairs": self.max_reasoner_repairs,
            "max_multiagent_fallbacks": self.max_multiagent_fallbacks,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "local_repairs": self.local_repairs,
            "reasoner_repairs": self.reasoner_repairs,
            "multiagent_fallbacks": self.multiagent_fallbacks,
            "elapsed_seconds": round(self._elapsed(), 3),
            "events": list(self.events),
        }


_BUDGET_CONTROLLER: BudgetController | None = None


def set_budget_controller(controller: BudgetController | None) -> None:
    global _BUDGET_CONTROLLER
    _BUDGET_CONTROLLER = controller


def get_budget_controller() -> BudgetController | None:
    return _BUDGET_CONTROLLER
