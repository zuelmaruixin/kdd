"""Console event-flow logger for live demo runs.

Different from ``StreamSink`` (which emits raw model tokens): this logger
emits structured *agent events* — router decisions, plans, per-specialist
calls, tool invocations, cross-model verify outcomes — in a clean rich-
rendered format suitable for screen-sharing during a presentation.

Wiring:

* Module-level singleton, set via ``set_progress_logger`` from the CLI.
* Each agent calls ``get_progress_logger()`` and, if non-None, posts the
  relevant event. ``None`` is the default for batch / parallel runs so
  log lines don't interleave.

When ``emit_demo_events=True`` is set on the logger, every event method
additionally writes a single-line ``__DEMO_EVENT__<json>`` record to
stdout. The Streamlit demo's subprocess reader parses these to update
its live UI without waiting for the task to finish.

Render style: tree-style indented lines with semantic colors, similar to
``cargo build`` or ``pytest -v`` output. Stays readable even when piped
to a non-TTY (rich falls back to plain text).
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from rich.console import Console


DEMO_EVENT_PREFIX = "__DEMO_EVENT__"


@dataclass(slots=True)
class ProgressLogger:
    # `force_terminal=False` keeps Rich from emitting ANSI escapes when
    # stderr is piped to the demo subprocess; `soft_wrap=True` avoids
    # mid-line wrapping that would otherwise prevent each log line from
    # flushing on its own newline.
    console: Console = field(default_factory=lambda: Console(
        stderr=True, force_terminal=False, soft_wrap=True
    ))
    indent_level: int = 0
    enabled: bool = True
    lang: str = "en"
    emit_demo_events: bool = False
    _task_started_at: float = 0.0

    # ---- low-level helpers -------------------------------------------------

    def _t(self, key: str) -> str:
        if self.lang.lower().startswith("zh"):
            return {
                "question": "问题",
                "done": "完成",
                "failed": "失败",
                "router": "路由",
                "compiler": "编译器",
                "budget": "预算",
                "reasoner_repair": "推理修复",
                "planner": "规划器",
                "rationale": "依据",
                "artifact": "中间结果",
                "synthesizer": "合成器",
                "context": "上下文",
                "program": "程序",
                "executed": "执行完成",
                "record_extract": "记录抽取",
                "rules": "规则",
                "preview": "样例",
                "positive_preview": "命中样例",
                "debug_steps": "调试记录",
                "reflection": "反思器",
                "score": "分数",
            }.get(key, key)
        return {
            "question": "question",
            "done": "done",
            "failed": "failed",
            "router": "router",
            "compiler": "compiler",
            "budget": "budget",
            "reasoner_repair": "reasoner-repair",
            "planner": "planner",
            "rationale": "rationale",
            "artifact": "artifact",
            "synthesizer": "synthesizer",
            "context": "context",
            "program": "program",
            "executed": "executed",
            "record_extract": "record-extract",
            "rules": "rules",
            "preview": "preview",
            "positive_preview": "positive preview",
            "debug_steps": "debug-steps",
            "reflection": "reflection",
            "score": "score",
        }.get(key, key)

    def _prefix(self) -> str:
        if self.indent_level <= 0:
            return ""
        return "  " * self.indent_level + "[dim]│[/dim] "

    def _line(self, head: str, body: str = "") -> None:
        if not self.enabled:
            return
        prefix = self._prefix()
        text = f"{prefix}{head}"
        if body:
            text += f" [dim]·[/dim] {body}"
        self.console.print(text, soft_wrap=True, highlight=False)

    def _emit_event(self, event_type: str, **payload: Any) -> None:
        """Emit a single-line JSON record for the Streamlit demo.

        Only fires when ``emit_demo_events`` is True. The line uses a
        fixed prefix so the demo's subprocess reader can split structured
        events from regular rich-formatted log lines.
        """
        if not self.emit_demo_events:
            return
        elapsed = (
            time.perf_counter() - self._task_started_at
            if self._task_started_at
            else 0.0
        )
        record = {
            "type": event_type,
            "ts": time.time(),
            "elapsed": round(elapsed, 3),
            **payload,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            return
        sys.stdout.write(f"{DEMO_EVENT_PREFIX}{line}\n")
        sys.stdout.flush()

    @contextmanager
    def section(self, head: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._line(head)
        self.indent_level += 1
        try:
            yield
        finally:
            self.indent_level -= 1

    # ---- task lifecycle ----------------------------------------------------

    def task_start(self, *, task_id: str, difficulty: str, question: str) -> None:
        if not self.enabled:
            return
        self._task_started_at = time.perf_counter()
        rule_text = f"[bold cyan]▶ {task_id}[/bold cyan] [dim]({difficulty or 'unknown'})[/dim]"
        self.console.rule(rule_text, style="cyan", align="left")
        self._line(f"[bold]{self._t('question')}:[/bold]", question.strip())
        self._emit_event(
            "task_start",
            task_id=task_id,
            difficulty=difficulty or "",
            question=question,
        )

    def task_end(self, *, succeeded: bool, failure_reason: str | None) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._task_started_at
        if succeeded:
            self.console.print(
                f"[bold green]✓ {self._t('done')}[/bold green] [dim]· {elapsed:.1f}s[/dim]",
                soft_wrap=True, highlight=False,
            )
        else:
            self.console.print(
                f"[bold red]✗ {self._t('failed')}[/bold red] [dim]· {elapsed:.1f}s · "
                f"{failure_reason or 'unknown'}[/dim]",
                soft_wrap=True, highlight=False,
            )
        self.console.print("")
        self._emit_event(
            "task_end",
            succeeded=succeeded,
            failure_reason=failure_reason,
        )

    # ---- router events -----------------------------------------------------

    def router_decision(
        self,
        *,
        difficulty: str,
        route_name: str,
        kind: str,
        model: str,
        difficulty_source: str,
        task_type: str | None = None,
        budget_level: str | None = None,
        needs_reasoner: bool | None = None,
    ) -> None:
        prefix = ""
        if task_type:
            prefix += f"type=[cyan]{task_type}[/cyan] "
        if budget_level:
            prefix += f"budget=[cyan]{budget_level}[/cyan] "
        if needs_reasoner is not None:
            prefix += f"reasoner=[cyan]{str(needs_reasoner).lower()}[/cyan] "
        self._line(
            f"[bold magenta]{self._t('router')}[/bold magenta]",
            f"{prefix}difficulty=[cyan]{difficulty}[/cyan] ([dim]{difficulty_source}[/dim]) "
            f"→ route=[bold]{route_name}[/bold] kind=[yellow]{kind}[/yellow] "
            f"model=[green]{model}[/green]",
        )
        self._emit_event(
            "router_decision",
            difficulty=difficulty,
            route_name=route_name,
            kind=kind,
            model=model,
            difficulty_source=difficulty_source,
            task_type=task_type,
            budget_level=budget_level,
            needs_reasoner=needs_reasoner,
        )

    def router_cascade(self, *, from_route: str, to_route: str, reason: str) -> None:
        self._line(
            "[bold yellow]cascade[/bold yellow]",
            f"{from_route} → {to_route} ([dim]{reason}[/dim])",
        )
        self._emit_event(
            "router_cascade",
            from_route=from_route,
            to_route=to_route,
            reason=reason,
        )

    def task_compiled(self, *, compiled: dict[str, Any]) -> None:
        ops = ", ".join(compiled.get("operations") or [])
        tools = ", ".join(compiled.get("preferred_tools") or [])
        sources = compiled.get("data_sources") or []
        self._line(
            f"[bold blue]{self._t('compiler')}[/bold blue]",
            f"type=[cyan]{compiled.get('task_type')}[/cyan] "
            f"answer=[yellow]{compiled.get('answer_type')}[/yellow] "
            f"ops={ops or '-'} tools={tools or '-'} sources={len(sources)}",
        )
        self._emit_event(
            "task_compiled",
            task_type=compiled.get("task_type"),
            answer_type=compiled.get("answer_type"),
            operations=compiled.get("operations") or [],
            preferred_tools=compiled.get("preferred_tools") or [],
            data_source_count=len(sources),
        )

    def budget_started(self, *, max_llm_calls: int, max_tool_calls: int, max_seconds: float) -> None:
        seconds = f", {max_seconds:.0f}s" if max_seconds > 0 else ""
        self._line(
            f"[bold blue]{self._t('budget')}[/bold blue]",
            f"llm≤{max_llm_calls}, tools≤{max_tool_calls}{seconds}",
        )
        self._emit_event(
            "budget_started",
            max_llm_calls=max_llm_calls,
            max_tool_calls=max_tool_calls,
            max_seconds=max_seconds,
        )

    def reasoner_repair_start(self, *, failure_reason: str) -> None:
        self._line(
            f"[bold magenta]{self._t('reasoner_repair')}[/bold magenta]",
            failure_reason or "unknown failure",
        )
        self._emit_event("reasoner_repair_start", failure_reason=failure_reason)

    def reasoner_repair_done(self, *, succeeded: bool, failure_reason: str | None) -> None:
        status = "[green]✓[/green]" if succeeded else "[red]✗[/red]"
        self._line(
            f"{status} [bold magenta]{self._t('reasoner_repair')}[/bold magenta]",
            "repaired" if succeeded else (failure_reason or "repair failed"),
        )
        self._emit_event(
            "reasoner_repair_done",
            succeeded=succeeded,
            failure_reason=failure_reason,
        )

    # ---- planner / specialist / synthesizer (multi-agent) ------------------

    def planner_done(self, *, rationale: str, subtasks: list[dict[str, Any]]) -> None:
        with self.section(f"[bold blue]{self._t('planner')}[/bold blue]"):
            self._line(f"[dim]{self._t('rationale')}:[/dim]", rationale)
            for st in subtasks:
                deps = st.get("depends_on") or []
                dep_text = f" ← {','.join(deps)}" if deps else ""
                self._line(
                    f"[bold]{st.get('id')}[/bold]",
                    f"[yellow]{st.get('specialist')}[/yellow]: "
                    f"{st.get('instruction')}[dim]{dep_text}[/dim]",
                )
        self._emit_event("planner_done", rationale=rationale, subtasks=list(subtasks or []))

    def specialist_start(self, *, subtask_id: str, kind: str, instruction: str) -> None:
        self._line(
            f"[bold]{subtask_id}[/bold] [yellow]{kind}[/yellow] [dim]starting[/dim]",
            instruction,
        )
        self._emit_event(
            "specialist_start",
            subtask_id=subtask_id,
            kind=kind,
            instruction=instruction,
        )

    def specialist_done(
        self,
        *,
        subtask_id: str,
        kind: str,
        succeeded: bool,
        summary: str,
        artifact_columns: list[str] | None,
        artifact_rows: list[list[Any]] | None,
        step_count: int,
    ) -> None:
        status = "[green]✓[/green]" if succeeded else "[red]✗[/red]"
        body = f"{summary} [dim]({step_count} steps)[/dim]"
        self._line(f"{status} {subtask_id} [yellow]{kind}[/yellow]", body)
        if artifact_columns:
            cols = ", ".join(artifact_columns)
            row_count = len(artifact_rows or [])
            preview = ""
            if artifact_rows:
                first = artifact_rows[0]
                preview = " · " + ", ".join(str(c)[:24] for c in first[:4])
            self._line(
                f"[dim]  {self._t('artifact')}:[/dim]",
                f"[{cols}] · {row_count} row(s){preview}",
            )
        self._emit_event(
            "specialist_done",
            subtask_id=subtask_id,
            kind=kind,
            succeeded=succeeded,
            summary=summary,
            artifact_columns=list(artifact_columns or []),
            artifact_row_count=len(artifact_rows or []),
            step_count=step_count,
        )

    def synthesizer_done(self, *, columns: list[str], row_count: int) -> None:
        self._line(
            f"[bold magenta]{self._t('synthesizer')}[/bold magenta]",
            f"answer columns=[{', '.join(columns)}] rows={row_count}",
        )
        self._emit_event(
            "synthesizer_done",
            columns=list(columns or []),
            row_count=row_count,
        )

    # ---- Operator-codegen events -------------------------------------------

    def codegen_program(
        self,
        *,
        program: str,
        manifest_summary: str,
        label: str = "operator-codegen",
    ) -> None:
        with self.section(f"[bold blue]{label}[/bold blue]"):
            self._line(f"[dim]{self._t('context')}:[/dim]", manifest_summary)
            preview = program.strip()
            if len(preview) > 600:
                preview = preview[:600] + "\n…(program truncated for log)…"
            self.console.print(
                self._prefix() + f"[dim]{self._t('program')} ↓[/dim]",
                soft_wrap=True, highlight=False,
            )
            for line in preview.splitlines() or [""]:
                self.console.print(
                    f"{self._prefix()}  [dim cyan]│[/dim cyan] {line}",
                    soft_wrap=True, highlight=False,
                )
        emit_program = program.strip()
        if len(emit_program) > 4000:
            emit_program = emit_program[:4000] + "\n…(program truncated)…"
        self._emit_event(
            "codegen_program",
            label=label,
            manifest_summary=manifest_summary,
            program=emit_program,
            program_chars=len(program or ""),
        )

    def tablellm_program(self, **kwargs: Any) -> None:
        self.codegen_program(**kwargs)

    def codegen_executed(
        self,
        *,
        succeeded: bool,
        shape: tuple[int, int] | None,
        failure_reason: str | None,
    ) -> None:
        if succeeded:
            shape_text = f"shape={shape}" if shape else ""
            self._line(f"[green]✓ {self._t('executed')}[/green]", shape_text)
        else:
            self._line(f"[red]✗ {self._t('executed')}[/red]", failure_reason or "unknown")
        self._emit_event(
            "codegen_executed",
            succeeded=succeeded,
            shape=list(shape) if shape else None,
            failure_reason=failure_reason,
        )

    def tablellm_executed(self, **kwargs: Any) -> None:
        self.codegen_executed(**kwargs)

    # ---- Record-text extraction --------------------------------------------

    def structured_extract_start(self, *, files: list[str]) -> None:
        self._line(
            f"[bold blue]{self._t('record_extract')}[/bold blue]",
            ", ".join(files) or "(no files)",
        )
        self._emit_event("structured_extract_start", files=list(files or []))

    def structured_extract_done(
        self,
        *,
        synthesized_csvs: dict[str, str],
        record_counts: dict[str, int],
        cache_hits: int,
        cache_misses: int,
        notes: list[str],
        rule_notes: dict[str, list[str]] | None = None,
        previews: dict[str, list[dict[str, Any]]] | None = None,
        positive_previews: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        if synthesized_csvs:
            summary = ", ".join(
                f"{src}→{dst} ({record_counts.get(src, 0)} rows)"
                for src, dst in synthesized_csvs.items()
            )
            self._line(
                f"[green]✓ {self._t('record_extract')}[/green]",
                f"{summary} cache={cache_hits}/{cache_misses}",
            )
            for src in synthesized_csvs:
                rules = (rule_notes or {}).get(src) or []
                if rules:
                    self._line(f"[dim]  {self._t('rules')}:[/dim]", " | ".join(rules[:3])[:500])
                positives = (positive_previews or {}).get(src) or []
                preview_rows = positives or ((previews or {}).get(src) or [])
                if preview_rows:
                    compact = json.dumps(preview_rows[:3], ensure_ascii=False, default=str)
                    if len(compact) > 700:
                        compact = compact[:700] + "…"
                    label = self._t("positive_preview") if positives else self._t("preview")
                    self._line(f"[dim]  {label}:[/dim]", compact)
        else:
            note_text = "; ".join(notes) if notes else "no synthesized csvs"
            self._line(f"[red]✗ {self._t('record_extract')}[/red]", note_text)
        self._emit_event(
            "structured_extract_done",
            synthesized_csvs=dict(synthesized_csvs or {}),
            record_counts=dict(record_counts or {}),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            notes=list(notes or []),
        )

    def codegen_debug(self, *, debug: dict[str, Any]) -> None:
        compact = json.dumps(debug, ensure_ascii=False, default=str)
        if len(compact) > 900:
            compact = compact[:900] + "…"
        self._line(f"[bold blue]{self._t('debug_steps')}[/bold blue]", compact)
        self._emit_event("codegen_debug", debug=debug)

    def tablellm_debug(self, **kwargs: Any) -> None:
        self.codegen_debug(**kwargs)

    # ---- Agentic reflection ------------------------------------------------

    def reflection_start(self) -> None:
        self._line(f"[bold magenta]{self._t('reflection')}[/bold magenta]", "checking plan vs action")
        self._emit_event("reflection_start")

    def reflection_done(
        self,
        *,
        verdict: str,
        confidence: str,
        issues: list[str],
        revision_instruction: str,
    ) -> None:
        color = "green" if verdict == "accept" else "yellow"
        issue_text = "; ".join(issues[:3]) if issues else "no concrete issue"
        if revision_instruction:
            issue_text += f" | retry: {revision_instruction[:220]}"
        self._line(
            f"[{color}]● {self._t('reflection')}[/{color}]",
            f"verdict={verdict} confidence={confidence} · {issue_text}",
        )
        self._emit_event(
            "reflection_done",
            verdict=verdict,
            confidence=confidence,
            issues=list(issues or []),
            revision_instruction=revision_instruction,
        )

    # ---- ReAct step --------------------------------------------------------

    def react_step(
        self,
        *,
        prefix: str,
        step_index: int,
        action: str,
        action_input: dict[str, Any],
        ok: bool,
        cached: bool = False,
    ) -> None:
        cache_tag = " [dim](cached)[/dim]" if cached else ""
        marker = "[green]✓[/green]" if ok else "[red]✗[/red]"
        # Compact action_input for screen display.
        try:
            args_text = json.dumps(action_input, ensure_ascii=False)
        except (TypeError, ValueError):
            args_text = str(action_input)
        if len(args_text) > 160:
            args_text = args_text[:160] + "…"
        self._line(
            f"{marker} {prefix}#{step_index} [yellow]{action}[/yellow]{cache_tag}",
            args_text,
        )
        self._emit_event(
            "react_step",
            prefix=prefix,
            step_index=step_index,
            action=action,
            action_input=action_input,
            ok=ok,
            cached=cached,
        )

    # ---- cross-model verify -----------------------------------------------

    def cross_verify_start(self, *, verifier_names: list[str]) -> None:
        self._line(
            "[bold magenta]cross-verify[/bold magenta]",
            f"running verifiers: {', '.join(verifier_names)}",
        )
        self._emit_event("cross_verify_start", verifier_names=list(verifier_names or []))

    def cross_verify_done(
        self,
        *,
        outcome: str,
        column_decisions: list[dict[str, Any]],
    ) -> None:
        kept = sum(1 for d in column_decisions if d.get("kept"))
        total = len(column_decisions)
        color = {
            "full_agreement_keep_primary": "green",
            "intersection_replaces_primary": "yellow",
            "empty_intersection_keep_primary": "red",
        }.get(outcome, "white")
        self._line(
            f"[bold magenta]cross-verify[/bold magenta] [{color}]{outcome}[/{color}]",
            f"kept {kept}/{total} columns",
        )
        for d in column_decisions:
            mark = "[green]✓[/green]" if d.get("kept") else "[red]✗[/red]"
            agreed = ",".join(d.get("agreed_by") or []) or "[dim]none[/dim]"
            self._line(
                f"  {mark} col[{d.get('column_index')}] {d.get('column_name')}",
                f"agreed_by={agreed}",
            )
        self._emit_event(
            "cross_verify_done",
            outcome=outcome,
            kept=kept,
            total=total,
            column_decisions=list(column_decisions or []),
        )

    # ---- react answer guard (cheap deterministic gate) ----------------------

    def react_answer_guard(
        self,
        *,
        prefix: str,
        step_index: int,
        verdict: str,
        score: float,
        risk_codes: list[str],
        top_risk: dict[str, Any] | None = None,
    ) -> None:
        """Log a cheap-guard verdict on a React draft answer.

        ``verdict`` is one of ``"pass"`` (commit first answer directly),
        ``"escalate"`` (force a self-verify round). ``risk_codes`` is the
        list of cheap-guard signals that fired.
        """
        color = "green" if verdict == "pass" else "yellow"
        body = (
            f"score={score:.2f}"
            + (f" risks={','.join(risk_codes[:4])}" if risk_codes else " risks=none")
        )
        self._line(
            f"[bold {color}]react-guard[/bold {color}] {prefix}#{step_index} "
            f"[{color}]{verdict}[/{color}]",
            body,
        )
        self._emit_event(
            "react_answer_guard",
            prefix=prefix,
            step_index=step_index,
            verdict=verdict,
            score=score,
            risk_codes=list(risk_codes or []),
            top_risk=top_risk,
        )

    # ---- score (from local evaluator if available) -------------------------

    def score(self, *, score: float, recall: float, penalty: float) -> None:
        color = "green" if score >= 0.95 else ("yellow" if score >= 0.5 else "red")
        self._line(
            f"[bold {color}]score[/bold {color}]",
            f"= {score:.3f} [dim](recall={recall:.3f}, penalty={penalty:.3f})[/dim]",
        )
        self._emit_event(
            "score",
            score=score,
            recall=recall,
            penalty=penalty,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_PROGRESS_LOGGER: ProgressLogger | None = None


def set_progress_logger(logger: ProgressLogger | None) -> None:
    global _PROGRESS_LOGGER
    _PROGRESS_LOGGER = logger


def get_progress_logger() -> ProgressLogger | None:
    return _PROGRESS_LOGGER
