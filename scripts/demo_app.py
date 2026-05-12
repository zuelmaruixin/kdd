from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from html import escape
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.eval.column_match import score_pair


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "router.deepseek.yaml"
_RESULT_PREFIX = "DEMO_RESULT_JSON="
_EVENT_PREFIX = "__DEMO_EVENT__"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


_APP_CSS = """
<style>
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
}
.hero {
  padding: 1.2rem 1.4rem;
  border: 1px solid #d7e2ef;
  border-radius: 10px;
  background:
    linear-gradient(120deg, rgba(8, 54, 93, 0.96), rgba(22, 89, 111, 0.92)),
    linear-gradient(90deg, #08365d, #2a6f73);
  color: white;
  margin-bottom: 1rem;
}
.hero h1 {
  margin: 0 0 .2rem 0;
  font-size: 2.1rem;
  letter-spacing: 0;
}
.hero p {
  margin: 0;
  color: #d8eef7;
  font-size: .98rem;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .7rem;
  margin: .5rem 0 1rem 0;
}
.metric-card {
  padding: .78rem .88rem;
  border: 1px solid #dde7f0;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
}
.metric-card .label {
  color: #64748b;
  font-size: .75rem;
  text-transform: uppercase;
  letter-spacing: .02em;
}
.metric-card .value {
  color: #0f172a;
  font-size: 1.25rem;
  font-weight: 700;
  margin-top: .15rem;
  overflow-wrap: anywhere;
}
.agent-rail {
  display: grid;
  grid-template-columns: repeat(7, minmax(120px, 1fr));
  gap: .65rem;
  margin: .8rem 0 1rem 0;
}
.step-card {
  position: relative;
  min-height: 128px;
  padding: .75rem .78rem;
  border: 1px solid #dce7f3;
  border-radius: 9px;
  background: #ffffff;
  box-shadow: 0 1px 3px rgba(15, 23, 42, .06);
}
.step-card.ok { border-top: 5px solid #16856f; }
.step-card.warn { border-top: 5px solid #c4821a; }
.step-card.err { border-top: 5px solid #c2413a; }
.step-card.idle { border-top: 5px solid #94a3b8; }
.step-index {
  font-size: .68rem;
  font-weight: 800;
  color: #64748b;
  letter-spacing: .08em;
}
.step-title {
  margin-top: .15rem;
  color: #0f172a;
  font-size: .98rem;
  font-weight: 800;
}
.step-status {
  display: inline-block;
  margin-top: .45rem;
  padding: .14rem .38rem;
  border-radius: 999px;
  font-size: .68rem;
  font-weight: 800;
  background: #eef2f7;
  color: #334155;
}
.step-card.ok .step-status { background: #e7f7f2; color: #04705d; }
.step-card.warn .step-status { background: #fff4de; color: #9a5b00; }
.step-card.err .step-status { background: #fdeceb; color: #a92822; }
.step-detail {
  margin-top: .5rem;
  color: #475569;
  font-size: .78rem;
  line-height: 1.25;
  overflow-wrap: anywhere;
}
.panel {
  border: 1px solid #dce7f3;
  border-radius: 10px;
  padding: .9rem 1rem;
  background: #ffffff;
  margin: .6rem 0;
}
.panel-title {
  color: #0f172a;
  font-weight: 800;
  margin-bottom: .45rem;
}
.subtask {
  border-left: 4px solid #2a6f73;
  padding: .45rem .7rem;
  background: #f8fbfd;
  margin: .42rem 0;
  border-radius: 6px;
}
.subtask .sid {
  font-weight: 800;
  color: #0f4f5f;
}
.tiny {
  color: #64748b;
  font-size: .76rem;
}
.dashboard-title {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin: .4rem 0 .8rem 0;
}
.dashboard-title h2 {
  margin: 0;
  color: #0f172a;
  font-size: 1.35rem;
  letter-spacing: 0;
}
.dashboard-title .path {
  color: #64748b;
  font-size: .78rem;
  overflow-wrap: anywhere;
  text-align: right;
}
.status-strip {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: .65rem;
  margin: .6rem 0 1rem 0;
}
.status-card {
  padding: .78rem .88rem;
  border: 1px solid #dde7f0;
  border-radius: 8px;
  background: #ffffff;
}
.status-card.good { border-left: 5px solid #16856f; }
.status-card.bad { border-left: 5px solid #c2413a; }
.status-card.score { border-left: 5px solid #2f6fb3; }
.status-card.warn { border-left: 5px solid #c4821a; }
.status-card.neutral { border-left: 5px solid #64748b; }
.status-card .label {
  color: #64748b;
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .02em;
}
.status-card .value {
  color: #0f172a;
  font-size: 1.2rem;
  font-weight: 800;
  margin-top: .15rem;
}
.record-badges {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
  margin: .2rem 0 .8rem 0;
}
.record-badge {
  border: 1px solid #d8e3ee;
  border-radius: 999px;
  padding: .22rem .55rem;
  background: #fff;
  color: #334155;
  font-size: .76rem;
  font-weight: 700;
}
.record-badge.ok { background: #e7f7f2; border-color: #c5eadf; color: #04705d; }
.record-badge.fail { background: #fdeceb; border-color: #f3c7c3; color: #a92822; }
.record-badge.score { background: #eaf3ff; border-color: #c7def7; color: #1d5d9b; }
@media (max-width: 1100px) {
  .status-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .dashboard-title { display: block; }
  .dashboard-title .path { text-align: left; margin-top: .2rem; }
}
@media (max-width: 1100px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .agent-rail { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.live-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: .9rem 1.1rem;
  border: 1px solid #d7e2ef;
  border-radius: 10px;
  background: linear-gradient(120deg, rgba(8, 54, 93, .96), rgba(22, 89, 111, .92));
  color: white;
  margin-bottom: .6rem;
}
.live-hero .stage {
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: .02em;
}
.live-hero .meta {
  color: #d8eef7;
  font-size: .82rem;
}
.live-hero .pulse {
  display: inline-block;
  width: .65rem;
  height: .65rem;
  border-radius: 50%;
  margin-right: .55rem;
  background: #ffd166;
  box-shadow: 0 0 0 0 rgba(255, 209, 102, .6);
  animation: livePulse 1.2s infinite;
}
.live-hero.complete .pulse {
  background: #4ade80;
  animation: none;
}
.live-hero.failed .pulse {
  background: #f97373;
  animation: none;
}
@keyframes livePulse {
  0%   { box-shadow: 0 0 0 0 rgba(255, 209, 102, .6); }
  60%  { box-shadow: 0 0 0 12px rgba(255, 209, 102, 0); }
  100% { box-shadow: 0 0 0 0 rgba(255, 209, 102, 0); }
}
.tl {
  border: 1px solid #dce7f3;
  border-radius: 10px;
  background: #ffffff;
  padding: .65rem .65rem;
  max-height: 540px;
  overflow-y: auto;
}
.tl-row {
  display: grid;
  grid-template-columns: 78px 1fr 70px;
  gap: .6rem;
  align-items: start;
  padding: .42rem .55rem;
  border-radius: 7px;
  border-left: 3px solid #94a3b8;
  background: #f8fbfd;
  margin-bottom: .35rem;
}
.tl-row.run { background: #fff7e0; border-left-color: #c4821a; }
.tl-row.ok { background: #ecf8f3; border-left-color: #16856f; }
.tl-row.warn { background: #fff4de; border-left-color: #c4821a; }
.tl-row.err { background: #fdeceb; border-left-color: #c2413a; }
.tl-row.info { background: #eef4fb; border-left-color: #2f6fb3; }
.tl-tag {
  font-size: .68rem;
  font-weight: 800;
  color: #334155;
  letter-spacing: .04em;
  text-transform: uppercase;
  padding: .12rem .35rem;
  background: #e2e8f0;
  border-radius: 4px;
  text-align: center;
  white-space: nowrap;
}
.tl-row.run .tl-tag { background: #fde8b8; color: #7a4d00; }
.tl-row.ok .tl-tag { background: #cbe9dd; color: #04705d; }
.tl-row.warn .tl-tag { background: #fde2b3; color: #7a4d00; }
.tl-row.err .tl-tag { background: #fbcdc9; color: #a92822; }
.tl-row.info .tl-tag { background: #cfdcf0; color: #1d5d9b; }
.tl-body { color: #0f172a; font-size: .86rem; line-height: 1.35; }
.tl-body .tl-head { font-weight: 800; }
.tl-body .tl-detail { color: #475569; font-size: .8rem; margin-top: .12rem; overflow-wrap: anywhere; }
.tl-body pre {
  background: #f1f5f9;
  border-radius: 4px;
  padding: .35rem .45rem;
  margin: .25rem 0 0 0;
  font-size: .72rem;
  line-height: 1.35;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 180px;
  overflow-y: auto;
}
.tl-time {
  color: #64748b;
  font-size: .72rem;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.live-side {
  border: 1px solid #dce7f3;
  border-radius: 10px;
  padding: .65rem .8rem;
  background: #ffffff;
  margin-bottom: .55rem;
}
.live-side .ls-title {
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .04em;
  color: #64748b;
  font-weight: 800;
  margin-bottom: .35rem;
}
.live-side .ls-row {
  font-size: .82rem;
  color: #0f172a;
  padding: .18rem 0;
  border-bottom: 1px dashed #e2e8f0;
}
.live-side .ls-row:last-child { border-bottom: none; }
.live-side .ls-row b { color: #1d5d9b; }
.live-side .ls-row .muted { color: #64748b; }
.live-side.empty .ls-row { color: #94a3b8; font-style: italic; }
.step-card.run {
  border-top: 5px solid #c4821a;
  background: linear-gradient(180deg, #fffaef 0%, #ffffff 100%);
}
.step-card.run .step-status { background: #fde8b8; color: #7a4d00; }

/* ===== Run dashboard (history) card grid ===== */
.run-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: .85rem;
  margin: .6rem 0 1rem 0;
}
.run-card {
  background: #ffffff;
  border: 1px solid #dde7f0;
  border-radius: 10px;
  padding: .85rem .95rem;
  display: flex;
  flex-direction: column;
  gap: .55rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
  position: relative;
  overflow: hidden;
}
.run-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0;
  width: 6px; height: 100%;
  background: #94a3b8;
}
.run-card.ok::before    { background: #16856f; }
.run-card.fail::before  { background: #c2413a; }
.run-card.perfect::before { background: #2f6fb3; }
.run-card .rc-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: .5rem;
}
.run-card .rc-task {
  font-weight: 800;
  font-size: .92rem;
  color: #0f172a;
  letter-spacing: .01em;
}
.run-card .rc-run {
  font-size: .68rem;
  color: #64748b;
  font-variant-numeric: tabular-nums;
}
.run-card .rc-score {
  display: flex;
  align-items: baseline;
  gap: .55rem;
}
.run-card .rc-score .num {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -.02em;
  font-variant-numeric: tabular-nums;
  color: #0f172a;
}
.run-card.ok .rc-score .num    { color: #04705d; }
.run-card.fail .rc-score .num  { color: #a92822; }
.run-card.perfect .rc-score .num { color: #1d5d9b; }
.run-card .rc-score .sub {
  font-size: .76rem;
  color: #475569;
  font-weight: 600;
}
.run-card .rc-status-pill {
  display: inline-block;
  padding: .18rem .55rem;
  border-radius: 999px;
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .04em;
  background: #e2e8f0;
  color: #475569;
}
.run-card.ok .rc-status-pill   { background: #c5eadf; color: #04705d; }
.run-card.fail .rc-status-pill { background: #f3c7c3; color: #a92822; }
.run-card .rc-question {
  font-size: .82rem;
  color: #1f2937;
  line-height: 1.4;
  max-height: 4.2em;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}
.run-card .rc-badges {
  display: flex;
  flex-wrap: wrap;
  gap: .3rem;
}
.run-card .rc-badge {
  background: #f1f5f9;
  color: #334155;
  font-size: .68rem;
  font-weight: 700;
  padding: .15rem .42rem;
  border-radius: 6px;
  letter-spacing: .02em;
  text-transform: uppercase;
}
.run-card .rc-badge.route { background: #dbeafe; color: #1d4ed8; }
.run-card .rc-badge.gate-pass { background: #c5eadf; color: #04705d; }
.run-card .rc-badge.gate-fail { background: #f3c7c3; color: #a92822; }
.run-card .rc-badge.difficulty-easy { background: #d6eedb; color: #166534; }
.run-card .rc-badge.difficulty-medium { background: #fde9c5; color: #92400e; }
.run-card .rc-badge.difficulty-hard { background: #f3d2cd; color: #991b1b; }
.run-card .rc-meta {
  display: flex;
  justify-content: space-between;
  font-size: .68rem;
  color: #64748b;
  font-variant-numeric: tabular-nums;
}
.run-card .rc-failure {
  background: #fff5f4;
  border: 1px solid #f7d2cf;
  border-radius: 6px;
  padding: .35rem .55rem;
  font-size: .72rem;
  color: #a92822;
  font-family: ui-monospace, monospace;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
"""


_SUBPROCESS_RUNNER = r"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

# Force unbuffered stdout/stderr so structured events reach the parent
# Streamlit process immediately. Without this the Rich console (stderr)
# block-buffers when its destination is a pipe, which makes the live
# timeline appear to "jump to the end" — events get held in OS buffers
# until enough bytes accumulate to flush a block.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True, write_through=True)
sys.stderr.reconfigure(line_buffering=True, write_through=True)

from data_agent_baseline.config import load_app_config
from data_agent_baseline.progress import ProgressLogger, set_progress_logger
from data_agent_baseline.run.runner import create_run_output_dir, run_single_task

task_id = sys.argv[1]
config_path = Path(sys.argv[2])
run_id = sys.argv[3]

# Surface a synthetic "booting" event right away so the timeline shows
# the subprocess actually started, even before the agent emits its first
# real progress event.
print('__DEMO_EVENT__' + json.dumps({"type": "task_start", "task_id": task_id, "question": "(loading)", "difficulty": ""}, ensure_ascii=False), flush=True)

config = load_app_config(config_path)
config = replace(config, run=replace(config.run, run_id=run_id, max_workers=1))
set_progress_logger(ProgressLogger(enabled=True, lang="zh", emit_demo_events=True))

_, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)
artifact = run_single_task(
    task_id=task_id,
    config=config,
    run_output_dir=run_output_dir,
)
print("DEMO_RESULT_JSON=" + json.dumps(artifact.to_dict(), ensure_ascii=False), flush=True)
"""


def _load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _latest_trace_path() -> Path | None:
    candidates = list((PROJECT_ROOT / "artifacts" / "runs").glob("**/trace.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_run_dir() -> Path | None:
    traces = list((PROJECT_ROOT / "artifacts" / "runs").glob("*/task_*/trace.json"))
    if not traces:
        return None
    latest_trace = max(traces, key=lambda path: path.stat().st_mtime)
    return latest_trace.parent.parent


def _trace_paths_for_scope(scope: Path) -> list[Path]:
    if scope.is_file() and scope.name == "trace.json":
        return [scope]
    if (scope / "trace.json").is_file():
        return [scope / "trace.json"]
    direct = sorted(scope.glob("task_*/trace.json"))
    if direct:
        return direct
    return sorted(scope.glob("**/trace.json"))


def _run_dir_for_trace(trace_path: Path) -> Path:
    if trace_path.parent.name.startswith("task_"):
        return trace_path.parent.parent
    return trace_path.parent


def _relative_artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_score_summary(run_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = run_dir / "score_summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:  # noqa: BLE001
        return {}
    rows = payload.get("per_task") or []
    return {
        str(item.get("task_id")): item
        for item in rows
        if isinstance(item, dict) and item.get("task_id")
    }


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tail_lines(lines: list[str], *, max_lines: int = 140) -> str:
    return "".join(lines[-max_lines:]).strip() or "Starting..."


_STAGE_ORDER = ("observe", "plan", "act", "execute", "trace", "reflect", "verify")
_STAGE_LABEL = {
    "observe": "Observe",
    "plan": "Plan",
    "act": "Act",
    "execute": "Execute",
    "trace": "Observe Trace",
    "reflect": "Reflect",
    "verify": "Verify",
}
_STAGE_BY_EVENT = {
    "router_decision": "observe",
    "router_cascade": "observe",
    "task_compiled": "observe",
    "planner_done": "plan",
    "structured_extract_start": "plan",
    "structured_extract_done": "plan",
    "codegen_program": "act",
    # `react_step` is intentionally not in this map — it gets dispatched
    # to the right stage in apply_event() based on its `action` field
    # (list_context/read_* → observe, execute_* → act, first answer →
    # reflect, second answer → verify).
    "specialist_start": "act",
    "specialist_done": "act",
    "codegen_executed": "execute",
    "synthesizer_done": "execute",
    "codegen_debug": "trace",
    "reflection_start": "reflect",
    "reflection_done": "reflect",
    "reasoner_repair_start": "reflect",
    "reasoner_repair_done": "reflect",
    "cross_verify_start": "verify",
    "cross_verify_done": "verify",
    "score": "verify",
}

# ReAct harness action → stage mapping. Used when event_type == "react_step".
_REACT_OBSERVE_ACTIONS = frozenset({
    "list_context", "read_csv", "read_json", "read_doc", "inspect_sqlite_schema",
})
_REACT_ACT_ACTIONS = frozenset({"execute_python", "execute_context_sql"})


def _stage_for_react_step(action: str, prior_answer_count: int) -> str:
    if action in _REACT_OBSERVE_ACTIONS:
        return "observe"
    if action in _REACT_ACT_ACTIONS:
        return "act"
    if action == "answer":
        # First answer is the *draft* (self-verify reflection); second
        # answer is the committed verification.
        return "reflect" if prior_answer_count == 0 else "verify"
    return "act"

_EVENT_TAG = {
    "task_start": ("START", "info"),
    "task_end": ("END", "info"),
    "router_decision": ("ROUTE", "info"),
    "router_cascade": ("CASCADE", "warn"),
    "task_compiled": ("COMPILE", "info"),
    "budget_started": ("BUDGET", "info"),
    "planner_done": ("PLAN", "info"),
    "specialist_start": ("ACT", "info"),
    "specialist_done": ("ACT", "ok"),
    "synthesizer_done": ("SYNTH", "ok"),
    "codegen_program": ("CODEGEN", "info"),
    "codegen_executed": ("EXEC", "ok"),
    "codegen_debug": ("DEBUG", "info"),
    "structured_extract_start": ("EXTRACT", "info"),
    "structured_extract_done": ("EXTRACT", "ok"),
    "reflection_start": ("REFLECT", "run"),
    "reflection_done": ("REFLECT", "ok"),
    "reasoner_repair_start": ("REPAIR", "run"),
    "reasoner_repair_done": ("REPAIR", "ok"),
    "react_step": ("REACT", "ok"),
    "cross_verify_start": ("VERIFY", "run"),
    "cross_verify_done": ("VERIFY", "ok"),
    "score": ("SCORE", "ok"),
}


def _initial_stage_state() -> dict[str, dict[str, str]]:
    return {name: {"status": "idle", "label": "", "detail": ""} for name in _STAGE_ORDER}


def _clip_text(value: Any, limit: int = 140) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _format_event_row(event: dict[str, Any]) -> dict[str, str]:
    event_type = str(event.get("type") or "info")
    tag, cls = _EVENT_TAG.get(event_type, (event_type.upper(), "info"))
    elapsed = event.get("elapsed")
    ts_text = f"{float(elapsed):.1f}s" if isinstance(elapsed, (int, float)) else ""
    head, detail = _summarize_event(event)
    return {"tag": tag, "cls": cls, "head": head, "detail": detail, "time": ts_text}


def _summarize_event(event: dict[str, Any]) -> tuple[str, str]:
    et = event.get("type")
    if et == "task_start":
        return (
            f"Task {event.get('task_id') or '?'}",
            f"difficulty={event.get('difficulty') or '?'} · {_clip_text(event.get('question'), 220)}",
        )
    if et == "task_end":
        ok = event.get("succeeded")
        reason = event.get("failure_reason")
        head = "Run completed" if ok else "Run failed"
        return head, _clip_text(reason or ("answer ready" if ok else "see logs"), 220)
    if et == "router_decision":
        head = f"route → {event.get('route_name') or '?'}"
        detail = (
            f"kind={event.get('kind') or '-'} · model={event.get('model') or '-'} · "
            f"type={event.get('task_type') or '-'} · difficulty={event.get('difficulty') or '-'}"
        )
        return head, detail
    if et == "router_cascade":
        return (
            f"cascade {event.get('from_route')} → {event.get('to_route')}",
            _clip_text(event.get("reason"), 200),
        )
    if et == "task_compiled":
        ops = ", ".join(event.get("operations") or []) or "-"
        return (
            f"compiled type={event.get('task_type') or '-'}",
            f"answer={event.get('answer_type') or '-'} · ops={ops} · sources={event.get('data_source_count') or 0}",
        )
    if et == "budget_started":
        return "budget", f"llm≤{event.get('max_llm_calls')} · tools≤{event.get('max_tool_calls')} · {event.get('max_seconds') or 0}s"
    if et == "planner_done":
        subtasks = event.get("subtasks") or []
        return f"planner produced {len(subtasks)} subtask(s)", _clip_text(event.get("rationale"), 220)
    if et == "specialist_start":
        return (
            f"{event.get('subtask_id') or '?'} · {event.get('kind') or '-'}",
            _clip_text(event.get("instruction"), 200),
        )
    if et == "specialist_done":
        ok = event.get("succeeded")
        head = f"{event.get('subtask_id')} · {event.get('kind')}" + (" ok" if ok else " failed")
        return head, _clip_text(event.get("summary"), 220)
    if et == "synthesizer_done":
        return f"synthesizer wrote {event.get('row_count') or 0} row(s)", ", ".join(event.get("columns") or [])
    if et == "codegen_program":
        return (
            f"codegen {event.get('label') or 'operator'}",
            f"program ready · {event.get('program_chars') or 0} chars · {_clip_text(event.get('manifest_summary'), 200)}",
        )
    if et == "codegen_executed":
        ok = event.get("succeeded")
        shape = event.get("shape") or []
        if ok:
            shape_text = f"shape={tuple(shape)}" if shape else "shape=?"
            return "execution ok", shape_text
        return "execution failed", _clip_text(event.get("failure_reason"), 220)
    if et == "codegen_debug":
        debug = event.get("debug") or {}
        if isinstance(debug, dict):
            keys = list(debug.keys())[:6]
            return f"debug steps ({len(debug)} keys)", ", ".join(keys)
        return "debug steps", ""
    if et == "structured_extract_start":
        return "record extraction start", ", ".join(event.get("files") or [])
    if et == "structured_extract_done":
        synth = event.get("synthesized_csvs") or {}
        return f"record extraction done ({len(synth)} file(s))", ", ".join(f"{k}→{v}" for k, v in synth.items())
    if et == "reflection_start":
        return "reflection running", "checking plan vs action"
    if et == "reflection_done":
        verdict = event.get("verdict") or "-"
        issues = "; ".join(event.get("issues") or [])
        detail = f"confidence={event.get('confidence') or '-'} · {issues or 'no concrete issue'}"
        return f"reflection {verdict}", _clip_text(detail, 220)
    if et == "reasoner_repair_start":
        return "reasoner repair", _clip_text(event.get("failure_reason"), 220)
    if et == "reasoner_repair_done":
        ok = event.get("succeeded")
        return ("reasoner repair ok" if ok else "reasoner repair failed"), _clip_text(event.get("failure_reason"), 220)
    if et == "react_step":
        action = event.get("action") or "?"
        args = event.get("action_input") or {}
        try:
            args_text = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            args_text = str(args)
        cached = " · cached" if event.get("cached") else ""
        ok = "ok" if event.get("ok") else "failed"
        return (
            f"react step #{event.get('step_index')} · {action} · {ok}{cached}",
            _clip_text(args_text, 220),
        )
    if et == "cross_verify_start":
        return "cross-verify start", ", ".join(event.get("verifier_names") or [])
    if et == "cross_verify_done":
        outcome = event.get("outcome") or "-"
        return f"cross-verify {outcome}", f"kept {event.get('kept') or 0}/{event.get('total') or 0} columns"
    if et == "score":
        return (
            f"score {float(event.get('score') or 0):.3f}",
            f"recall={float(event.get('recall') or 0):.3f} · penalty={float(event.get('penalty') or 0):.3f}",
        )
    return str(et or "event"), _clip_text(json.dumps(event, ensure_ascii=False, default=str), 220)


class _LiveRenderer:
    """Renders the live agent dashboard from streamed __DEMO_EVENT__ records."""

    def __init__(self, *, task_id: str) -> None:
        self.task_id = task_id
        self.start_time = time.time()
        self.stages: dict[str, dict[str, str]] = _initial_stage_state()
        self.events: list[dict[str, Any]] = []
        self.planner: dict[str, Any] | None = None
        self.reflection: dict[str, Any] | None = None
        self.program_text: str | None = None
        self.score_event: dict[str, Any] | None = None
        self.task_meta: dict[str, Any] = {"task_id": task_id, "question": "", "difficulty": ""}
        self.complete: bool = False
        self.succeeded: bool | None = None
        self.failure_reason: str | None = None
        self.current_stage: str | None = "observe"
        self.raw_lines: list[str] = []
        # Self-verify accounting for the ReAct harness: count of `answer`
        # calls seen so the first → reflect stage and second → verify stage.
        self.answer_count: int = 0
        # Cache of the last draft answer summary (cols + row count) so the
        # reflect/verify panels can show what the model is checking.
        self.draft_summary: dict[str, Any] | None = None
        # Mark first stage as running so the rail isn't all-idle pre-event.
        self.stages["observe"]["status"] = "run"
        self.stages["observe"]["label"] = "starting"
        self.stages["observe"]["detail"] = "booting agent runtime"

        # Streamlit placeholders, all built up-front.
        self.hero_ph = st.empty()
        self.rail_ph = st.empty()

        left_col, right_col = st.columns([1.3, 1.0])
        with left_col:
            st.markdown("<div class='live-side ls-title' style='margin-bottom:.3rem;'>Event Timeline</div>", unsafe_allow_html=True)
            self.timeline_ph = st.empty()
        with right_col:
            self.planner_ph = st.empty()
            self.reflection_ph = st.empty()
            self.program_ph = st.empty()
            self.score_ph = st.empty()

        with st.expander("Raw agent log (rich output)", expanded=False):
            self.log_ph = st.empty()

        self._render_hero()
        self._render_rail()
        self._render_timeline()
        self._render_side_panels()

    # ----- event handling -----

    def apply_event(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        self.events.append(event)
        if event_type == "react_step":
            action = str(event.get("action") or "")
            prior = self.answer_count
            stage = _stage_for_react_step(action, prior)
            if action == "answer":
                self.answer_count += 1
                # Capture draft answer shape so the side panel can show it
                # while the model self-verifies.
                ai = event.get("action_input") or {}
                cols = ai.get("columns") if isinstance(ai, dict) else None
                rows = ai.get("rows") if isinstance(ai, dict) else None
                self.draft_summary = {
                    "columns": list(cols) if isinstance(cols, list) else [],
                    "row_count": len(rows) if isinstance(rows, list) else 0,
                    "is_final": prior >= 1,
                }
        else:
            stage = _STAGE_BY_EVENT.get(event_type or "")
        if stage:
            self._advance_stage_to(stage)
            self._update_stage_from_event(stage, event)
        if event_type == "task_start":
            self.task_meta.update(
                task_id=event.get("task_id") or self.task_id,
                question=event.get("question") or "",
                difficulty=event.get("difficulty") or "",
            )
        elif event_type == "planner_done":
            self.planner = {
                "rationale": event.get("rationale") or "",
                "subtasks": event.get("subtasks") or [],
            }
        elif event_type == "reflection_done":
            self.reflection = {
                "verdict": event.get("verdict") or "-",
                "confidence": event.get("confidence") or "-",
                "issues": event.get("issues") or [],
                "revision_instruction": event.get("revision_instruction") or "",
            }
        elif event_type == "codegen_program":
            self.program_text = event.get("program") or ""
        elif event_type == "score":
            self.score_event = {
                "score": event.get("score"),
                "recall": event.get("recall"),
                "penalty": event.get("penalty"),
            }
        elif event_type == "task_end":
            self.complete = True
            self.succeeded = bool(event.get("succeeded"))
            self.failure_reason = event.get("failure_reason")
            self._finalize_remaining_stages()
        self._render_all()

    def _advance_stage_to(self, target: str) -> None:
        try:
            target_idx = _STAGE_ORDER.index(target)
        except ValueError:
            return
        for idx, name in enumerate(_STAGE_ORDER):
            if idx < target_idx and self.stages[name]["status"] in {"idle", "run"}:
                self.stages[name]["status"] = "ok"
                # ReAct skips through some stages without a dedicated
                # event (e.g. "plan" / "trace"). Don't leave their cards
                # empty — annotate them so the demo reads coherently.
                if not self.stages[name].get("label"):
                    self.stages[name]["label"] = "(implicit)"
                    self.stages[name]["detail"] = "covered inline by ReAct reasoning"
            elif idx == target_idx and self.stages[name]["status"] == "idle":
                self.stages[name]["status"] = "run"
        self.current_stage = target

    def _update_stage_from_event(self, stage: str, event: dict[str, Any]) -> None:
        et = event.get("type")
        if et == "router_decision":
            self.stages[stage]["label"] = event.get("route_name") or "-"
            self.stages[stage]["detail"] = f"kind={event.get('kind') or '-'}"
        elif et == "task_compiled":
            self.stages[stage]["label"] = event.get("task_type") or "-"
            ops = ", ".join(event.get("operations") or [])
            self.stages[stage]["detail"] = f"ops={ops or '-'} · sources={event.get('data_source_count') or 0}"
            if self.stages[stage]["status"] == "run":
                self.stages[stage]["status"] = "ok"
        elif et == "planner_done":
            count = len(event.get("subtasks") or [])
            self.stages[stage]["label"] = f"{count} subtask(s)"
            self.stages[stage]["detail"] = _clip_text(event.get("rationale"), 140)
            self.stages[stage]["status"] = "ok"
        elif et == "codegen_program":
            self.stages[stage]["label"] = event.get("label") or "codegen"
            self.stages[stage]["detail"] = f"{event.get('program_chars') or 0} chars · {_clip_text(event.get('manifest_summary'), 100)}"
        elif et == "codegen_executed":
            ok = event.get("succeeded")
            self.stages[stage]["status"] = "ok" if ok else "err"
            shape = event.get("shape") or []
            self.stages[stage]["label"] = "ok" if ok else "failed"
            if ok:
                self.stages[stage]["detail"] = f"shape={tuple(shape)}" if shape else "answer ready"
            else:
                self.stages[stage]["detail"] = _clip_text(event.get("failure_reason"), 140)
        elif et == "codegen_debug":
            debug = event.get("debug") or {}
            count = len(debug) if isinstance(debug, dict) else 0
            self.stages[stage]["label"] = f"{count} key(s)"
            self.stages[stage]["detail"] = ", ".join(list(debug.keys())[:5]) if isinstance(debug, dict) else ""
            self.stages[stage]["status"] = "ok"
        elif et == "reflection_start":
            self.stages[stage]["label"] = "running"
            self.stages[stage]["detail"] = "checking plan vs action"
        elif et == "reflection_done":
            verdict = event.get("verdict") or "-"
            self.stages[stage]["label"] = verdict
            self.stages[stage]["detail"] = "; ".join(event.get("issues") or []) or "no concrete issue"
            self.stages[stage]["status"] = "ok" if verdict == "accept" else "warn"
        elif et == "react_step":
            action = str(event.get("action") or "")
            step_index = event.get("step_index")
            ai = event.get("action_input") or {}
            ok = bool(event.get("ok", True))
            cached = bool(event.get("cached"))
            tag = "cached " if cached else ""
            # Compose human-friendly detail per action.
            # Only flip status if the stage is the active one — don't
            # downgrade an already-passed stage when the model briefly
            # revisits an earlier tool (e.g. re-reads a CSV after acting).
            prior_status = self.stages[stage].get("status")
            allow_status_update = prior_status in {"idle", "run", "warn"}
            if action in _REACT_OBSERVE_ACTIONS:
                hint = ""
                if isinstance(ai, dict):
                    if ai.get("path"):
                        hint = str(ai.get("path"))
                    elif ai.get("max_depth") is not None:
                        hint = f"depth={ai.get('max_depth')}"
                self.stages[stage]["label"] = f"{tag}{action}"
                self.stages[stage]["detail"] = (
                    f"step #{step_index} · {hint}" if hint else f"step #{step_index}"
                )
                if allow_status_update:
                    self.stages[stage]["status"] = "run" if ok else "warn"
            elif action in _REACT_ACT_ACTIONS:
                hint = ""
                if isinstance(ai, dict):
                    if ai.get("code"):
                        hint = _clip_text(ai.get("code"), 100)
                    elif ai.get("sql"):
                        hint = _clip_text(ai.get("sql"), 100)
                self.stages[stage]["label"] = action
                self.stages[stage]["detail"] = f"step #{step_index} · {hint}" if hint else f"step #{step_index}"
                if allow_status_update:
                    self.stages[stage]["status"] = "run" if ok else "warn"
            elif action == "answer":
                cols = ai.get("columns") if isinstance(ai, dict) else None
                rows = ai.get("rows") if isinstance(ai, dict) else None
                col_count = len(cols) if isinstance(cols, list) else 0
                row_count = len(rows) if isinstance(rows, list) else 0
                if stage == "reflect":
                    self.stages[stage]["label"] = "draft submitted"
                    self.stages[stage]["detail"] = (
                        f"{col_count} col(s) × {row_count} row(s) · running self-verify"
                    )
                    self.stages[stage]["status"] = "run"
                    # Surface in the side reflection panel.
                    self.reflection = {
                        "verdict": "self-verify",
                        "confidence": "-",
                        "issues": [
                            f"Draft: {col_count} col(s) × {row_count} row(s)",
                            "Model is now re-checking columns + values before final commit.",
                        ],
                        "revision_instruction": (
                            "Re-running computations and re-confirming requested answer type."
                        ),
                    }
                else:  # verify
                    self.stages[stage]["label"] = "final answer"
                    self.stages[stage]["detail"] = f"{col_count} col(s) × {row_count} row(s) committed"
                    self.stages[stage]["status"] = "ok"
            else:
                self.stages[stage]["label"] = action or "react step"
                self.stages[stage]["detail"] = f"step #{step_index}"
        elif et == "score":
            self.stages[stage]["status"] = "ok"
            self.stages[stage]["label"] = f"{float(event.get('score') or 0):.3f}"
            self.stages[stage]["detail"] = f"recall={float(event.get('recall') or 0):.3f}"
        elif et == "cross_verify_done":
            outcome = event.get("outcome") or "-"
            self.stages[stage]["label"] = outcome
            self.stages[stage]["detail"] = f"kept {event.get('kept') or 0}/{event.get('total') or 0}"
            self.stages[stage]["status"] = "ok" if "agreement" in outcome else "warn"
        elif et == "router_cascade":
            self.stages[stage]["status"] = "warn"
            self.stages[stage]["detail"] = f"→ {event.get('to_route') or '?'}"

    def _finalize_remaining_stages(self) -> None:
        for name in _STAGE_ORDER:
            if self.stages[name]["status"] == "run":
                self.stages[name]["status"] = "ok" if self.succeeded else "err"
            elif self.stages[name]["status"] == "idle":
                self.stages[name]["status"] = "ok" if self.succeeded else "idle"

    def append_log_line(self, line: str) -> None:
        self.raw_lines.append(line)
        plain = _ANSI_RE.sub("", "".join(self.raw_lines[-180:])).strip() or "Booting agent runtime..."
        self.log_ph.code(plain, language="text")

    # ----- rendering -----

    def _render_all(self) -> None:
        self._render_hero()
        self._render_rail()
        self._render_timeline()
        self._render_side_panels()

    def _render_hero(self) -> None:
        if self.complete:
            if self.succeeded:
                hero_class = "complete"
                stage_label = f"Run finished · {self.task_meta.get('task_id') or self.task_id}"
                meta = f"elapsed {time.time() - self.start_time:.1f}s · answer ready"
            else:
                hero_class = "failed"
                stage_label = f"Run failed · {self.task_meta.get('task_id') or self.task_id}"
                meta = _clip_text(self.failure_reason or "see logs", 220)
        else:
            current = self.current_stage or "observe"
            hero_class = ""
            stage_label = f"Running · {_STAGE_LABEL.get(current, current)}"
            meta = (
                f"task={self.task_meta.get('task_id') or self.task_id} · "
                f"difficulty={self.task_meta.get('difficulty') or '-'} · "
                f"elapsed {time.time() - self.start_time:.1f}s · "
                f"{len(self.events)} event(s)"
            )
        html = (
            f"<div class='live-hero {hero_class}'>"
            "<div><span class='pulse'></span>"
            f"<span class='stage'>{escape(stage_label)}</span></div>"
            f"<div class='meta'>{escape(meta)}</div>"
            "</div>"
        )
        self.hero_ph.markdown(html, unsafe_allow_html=True)

    def _render_rail(self) -> None:
        cards = []
        for idx, name in enumerate(_STAGE_ORDER, start=1):
            stage = self.stages[name]
            status = stage["status"]
            cls_map = {"idle": "idle", "run": "run", "ok": "ok", "warn": "warn", "err": "err"}
            cls = cls_map.get(status, "idle")
            label = stage.get("label") or "-"
            detail = stage.get("detail") or ""
            status_text = "RUNNING" if status == "run" else status.upper()
            cards.append(
                f"<div class='step-card {cls}'>"
                f"<div class='step-index'>STEP {idx:02d}</div>"
                f"<div class='step-title'>{escape(_STAGE_LABEL[name])}</div>"
                f"<div class='step-status'>{escape(status_text)}</div>"
                f"<div class='step-detail'><b>{escape(str(label))}</b><br>{escape(str(detail))}</div>"
                "</div>"
            )
        self.rail_ph.markdown("<div class='agent-rail'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

    def _render_timeline(self) -> None:
        if not self.events:
            html = "<div class='tl'><div class='tl-row info'><div class='tl-tag'>WAIT</div>" \
                "<div class='tl-body'><div class='tl-head'>Waiting for the agent's first event</div>" \
                "<div class='tl-detail'>The runtime is booting. Structured events will appear here as they arrive.</div></div>" \
                "<div class='tl-time'>0.0s</div></div></div>"
            self.timeline_ph.markdown(html, unsafe_allow_html=True)
            return
        rows = []
        # Newest first so users always see the most recent event at the top.
        for event in reversed(self.events[-200:]):
            row = _format_event_row(event)
            extra_html = ""
            if event.get("type") == "codegen_program" and event.get("program"):
                program_preview = str(event.get("program"))
                if len(program_preview) > 1200:
                    program_preview = program_preview[:1200] + "\n…(truncated)…"
                extra_html = f"<pre>{escape(program_preview)}</pre>"
            rows.append(
                f"<div class='tl-row {row['cls']}'>"
                f"<div class='tl-tag'>{escape(row['tag'])}</div>"
                f"<div class='tl-body'>"
                f"<div class='tl-head'>{escape(row['head'])}</div>"
                f"<div class='tl-detail'>{escape(row['detail'])}</div>"
                f"{extra_html}"
                "</div>"
                f"<div class='tl-time'>{escape(row['time'])}</div>"
                "</div>"
            )
        self.timeline_ph.markdown("<div class='tl'>" + "".join(rows) + "</div>", unsafe_allow_html=True)

    def _render_side_panels(self) -> None:
        # Planner subtasks (only fires on the multi_agent route).
        if self.planner and self.planner.get("subtasks"):
            rows = [
                f"<div class='ls-row'><b>{escape(str(item.get('id') or '-'))}</b> "
                f"· <span class='muted'>{escape(str(item.get('specialist') or '-'))}</span><br>"
                f"{escape(_clip_text(item.get('instruction'), 200))}</div>"
                for item in self.planner.get("subtasks") or []
            ]
            html = (
                "<div class='live-side'>"
                "<div class='ls-title'>Planner Subtasks</div>"
                f"<div class='ls-row muted'>{escape(_clip_text(self.planner.get('rationale'), 220))}</div>"
                + "".join(rows)
                + "</div>"
            )
        else:
            # On the ReAct harness route there is no planner — show the
            # live tool-call trace instead so the panel is not empty.
            react_events = [e for e in self.events if e.get("type") == "react_step"]
            if react_events:
                rows = []
                for evt in react_events[-12:]:
                    action = str(evt.get("action") or "")
                    step_index = evt.get("step_index")
                    ai = evt.get("action_input") or {}
                    hint = ""
                    if isinstance(ai, dict):
                        if ai.get("path"):
                            hint = str(ai.get("path"))
                        elif ai.get("code"):
                            hint = _clip_text(ai.get("code"), 80)
                        elif ai.get("sql"):
                            hint = _clip_text(ai.get("sql"), 80)
                        elif ai.get("columns"):
                            cols = ai.get("columns") or []
                            rows_ai = ai.get("rows") or []
                            n_cols = len(cols) if isinstance(cols, list) else 0
                            n_rows = len(rows_ai) if isinstance(rows_ai, list) else 0
                            hint = f"{n_cols} col(s) × {n_rows} row(s)"
                    ok_glyph = "·" if evt.get("ok", True) else "✗"
                    rows.append(
                        f"<div class='ls-row'><b>#{escape(str(step_index))}</b> "
                        f"<code>{escape(action)}</code> {ok_glyph} "
                        f"<span class='muted'>{escape(hint)}</span></div>"
                    )
                html = (
                    "<div class='live-side'>"
                    "<div class='ls-title'>ReAct Tool Trace</div>"
                    + "".join(rows)
                    + "</div>"
                )
            else:
                html = (
                    "<div class='live-side empty'>"
                    "<div class='ls-title'>ReAct Tool Trace</div>"
                    "<div class='ls-row'>Agent has not invoked a tool yet.</div>"
                    "</div>"
                )
        self.planner_ph.markdown(html, unsafe_allow_html=True)

        # Reflection.
        if self.reflection:
            verdict = self.reflection.get("verdict") or "-"
            confidence = self.reflection.get("confidence") or "-"
            issues = "; ".join(self.reflection.get("issues") or []) or "no concrete issue"
            revision = self.reflection.get("revision_instruction") or ""
            extra = f"<div class='ls-row muted'>retry: {escape(_clip_text(revision, 220))}</div>" if revision else ""
            html = (
                "<div class='live-side'>"
                "<div class='ls-title'>Reflection</div>"
                f"<div class='ls-row'><b>verdict:</b> {escape(verdict)} · "
                f"<b>confidence:</b> {escape(confidence)}</div>"
                f"<div class='ls-row'>{escape(_clip_text(issues, 220))}</div>"
                + extra
                + "</div>"
            )
        else:
            html = (
                "<div class='live-side empty'>"
                "<div class='ls-title'>Reflection</div>"
                "<div class='ls-row'>Reflection critic has not run yet.</div>"
                "</div>"
            )
        self.reflection_ph.markdown(html, unsafe_allow_html=True)

        # Program preview. On agentic_operator routes this is the codegen
        # program. On the ReAct harness it's the most recent execute_python
        # or execute_context_sql payload.
        preview_title = "Generated Program (preview)"
        preview_text: str | None = self.program_text
        if not preview_text:
            for evt in reversed(self.events):
                if evt.get("type") != "react_step":
                    continue
                action = evt.get("action")
                ai = evt.get("action_input") or {}
                if action == "execute_python" and isinstance(ai, dict) and ai.get("code"):
                    preview_text = str(ai.get("code"))
                    preview_title = f"Latest execute_python (step #{evt.get('step_index')})"
                    break
                if action == "execute_context_sql" and isinstance(ai, dict) and ai.get("sql"):
                    preview_text = str(ai.get("sql"))
                    preview_title = f"Latest SQL (step #{evt.get('step_index')})"
                    break
        if preview_text:
            preview = preview_text
            if len(preview) > 1600:
                preview = preview[:1600] + "\n…(truncated)…"
            html = (
                "<div class='live-side'>"
                f"<div class='ls-title'>{escape(preview_title)}</div>"
                f"<pre style='margin:0; max-height:240px; overflow:auto; background:#f1f5f9; "
                f"padding:.5rem; border-radius:6px; font-size:.72rem;'>{escape(preview)}</pre>"
                "</div>"
            )
        else:
            html = (
                "<div class='live-side empty'>"
                f"<div class='ls-title'>{escape(preview_title)}</div>"
                "<div class='ls-row'>No code or SQL has been executed yet.</div>"
                "</div>"
            )
        self.program_ph.markdown(html, unsafe_allow_html=True)

        # Score (if available).
        if self.score_event and self.score_event.get("score") is not None:
            html = (
                "<div class='live-side'>"
                "<div class='ls-title'>Local Score</div>"
                f"<div class='ls-row'><b>score:</b> {float(self.score_event['score'] or 0):.3f}</div>"
                f"<div class='ls-row muted'>recall={float(self.score_event.get('recall') or 0):.3f} · "
                f"penalty={float(self.score_event.get('penalty') or 0):.3f}</div>"
                "</div>"
            )
            self.score_ph.markdown(html, unsafe_allow_html=True)


def _run_task_live(*, task_id: str, config_path: Path, run_id: str) -> tuple[dict[str, Any], Path]:
    command = [
        sys.executable,
        "-u",
        "-c",
        _SUBPROCESS_RUNNER,
        task_id,
        str(config_path),
        run_id,
    ]
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    renderer = _LiveRenderer(task_id=task_id)
    result_payload: dict[str, Any] | None = None

    assert process.stdout is not None
    # Use readline()-based iteration rather than `for line in pipe`. The
    # implicit iterator on a subprocess stdout pipe holds onto chunks
    # until its internal buffer fills, which is exactly the source of
    # the "timeline only updates at the very end" bug. readline() in a
    # line-buffered pipe returns each line as soon as the child flushes
    # it, so live progress events render incrementally.
    for line in iter(process.stdout.readline, ""):
        if line.startswith(_RESULT_PREFIX):
            try:
                result_payload = json.loads(line.removeprefix(_RESULT_PREFIX))
            except json.JSONDecodeError:
                result_payload = None
            continue
        if line.startswith(_EVENT_PREFIX):
            payload_str = line[len(_EVENT_PREFIX):].strip()
            if not payload_str:
                continue
            try:
                event = json.loads(payload_str)
            except json.JSONDecodeError:
                renderer.append_log_line(line)
                continue
            renderer.apply_event(event)
            # Yield to Streamlit's runtime so the placeholders we just
            # mutated actually get pushed over the WebSocket before we
            # block on the next readline(). Without this yield, several
            # events can be applied in a microsecond burst and only the
            # final DOM state shows up to the user.
            time.sleep(0)
            continue
        renderer.append_log_line(line)

    return_code = process.wait()
    renderer.append_log_line("")  # force final flush

    if return_code != 0:
        raise RuntimeError(f"subprocess exited with code {return_code}")
    if result_payload is None:
        raise RuntimeError("run finished but did not return artifact metadata")

    trace_path = Path(str(result_payload["trace_path"]))
    return result_payload, trace_path


def _answer_frame(answer: dict[str, Any] | None) -> pd.DataFrame | None:
    if not isinstance(answer, dict):
        return None
    columns = list(answer.get("columns") or [])
    rows = [list(row) for row in (answer.get("rows") or [])]
    if not columns:
        return None
    return pd.DataFrame(rows, columns=columns)


def _route_summary(trace: dict[str, Any]) -> dict[str, Any]:
    decision = trace.get("router_decision") or {}
    compiled = trace.get("compiled_task") or decision.get("compiled_task") or {}
    validation = trace.get("answer_validation") or {}
    return {
        "succeeded": trace.get("succeeded"),
        "route": decision.get("route_name"),
        "kind": decision.get("kind"),
        "model": decision.get("model"),
        "task_type": compiled.get("task_type"),
        "operations": ", ".join(compiled.get("operations") or []),
        "answer_valid": validation.get("valid"),
        "elapsed_seconds": trace.get("e2e_elapsed_seconds"),
    }


def _get_operator_block(trace: dict[str, Any]) -> dict[str, Any]:
    for key in ("agentic_operator", "operator_executor", "tablellm_direct"):
        value = trace.get(key)
        if isinstance(value, dict):
            return value
    sc = trace.get("self_consistency") or {}
    samples = sc.get("samples") or []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        for key in ("agentic_operator", "operator_executor", "tablellm_direct"):
            value = sample.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _clip(value: Any, limit: int = 140) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _iter_manifest(trace: dict[str, Any]) -> list[dict[str, Any]]:
    operator = _get_operator_block(trace)
    return [item for item in (operator.get("context_manifest") or []) if isinstance(item, dict)]


def _manifest_value(trace: dict[str, Any], key: str) -> Any:
    for item in reversed(_iter_manifest(trace)):
        if key in item:
            return item[key]
    return None


def _agentic_trace(trace: dict[str, Any]) -> dict[str, Any]:
    value = _manifest_value(trace, "agentic_operator")
    return value if isinstance(value, dict) else {}


def _semantic_plan(trace: dict[str, Any]) -> dict[str, Any]:
    value = _manifest_value(trace, "semantic_plan")
    return value if isinstance(value, dict) else {}


def _semantic_consistency(trace: dict[str, Any]) -> dict[str, Any]:
    value = _manifest_value(trace, "semantic_consistency")
    return value if isinstance(value, dict) else {}


def _cheap_guard(trace: dict[str, Any]) -> dict[str, Any]:
    value = _manifest_value(trace, "cheap_semantic_assessment")
    return value if isinstance(value, dict) else {}


def _debug_steps(trace: dict[str, Any]) -> dict[str, Any]:
    stdout = str(_get_operator_block(trace).get("exec_stdout") or "")
    for line in stdout.splitlines():
        if line.startswith("OPERATOR_CODEGEN_DEBUG="):
            payload = line.removeprefix("OPERATOR_CODEGEN_DEBUG=").strip()
        elif line.startswith("TABLELLM_DEBUG="):
            payload = line.removeprefix("TABLELLM_DEBUG=").strip()
        else:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {"raw": payload}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _answer_shape(trace: dict[str, Any]) -> str:
    answer = trace.get("answer")
    if not isinstance(answer, dict):
        return "no answer"
    cols = answer.get("columns") or []
    rows = answer.get("rows") or []
    return f"{len(rows)} row(s) x {len(cols)} column(s)"


def _operator_kind(trace: dict[str, Any]) -> str:
    for key in ("agentic_operator", "operator_executor", "tablellm_direct"):
        if isinstance(trace.get(key), dict):
            return key
    return str((trace.get("router_decision") or {}).get("kind") or "-")


def _reflection_verdict(trace: dict[str, Any]) -> str:
    rounds = (_agentic_trace(trace).get("reflection_rounds") or [])
    if rounds and isinstance(rounds[0], dict):
        decision = rounds[0].get("decision") or {}
        return str(decision.get("verdict") or "-")
    return "-"


def _score_for_trace(
    *,
    trace_path: Path,
    trace: dict[str, Any],
    app_config: Any,
    score_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_id = str(trace.get("task_id") or trace_path.parent.name)
    local_score = trace.get("local_score")
    if isinstance(local_score, dict) and local_score.get("score") is not None:
        return {
            "score": local_score.get("score"),
            "recall": local_score.get("recall"),
            "penalty": local_score.get("penalty"),
            "matched_count": local_score.get("matched_count"),
            "pred_col_count": local_score.get("pred_col_count"),
            "gold_col_count": local_score.get("gold_col_count"),
            "error": local_score.get("error"),
        }

    if task_id in score_summary:
        return score_summary[task_id]

    pred_csv = trace_path.parent / "prediction.csv"
    gold_csv = app_config.dataset.gold_root / task_id / "gold.csv"
    try:
        return score_pair(
            task_id=task_id,
            pred_csv=pred_csv,
            gold_csv=gold_csv,
            redundancy_lambda=app_config.scoring.redundancy_lambda,
            numeric_tolerance=app_config.scoring.numeric_tolerance,
            case_insensitive=app_config.scoring.case_insensitive,
            strip_whitespace=app_config.scoring.strip_whitespace,
        ).to_dict()
    except Exception as exc:  # noqa: BLE001
        return {"score": None, "recall": None, "penalty": None, "error": f"score_error:{exc}"}


def _record_for_trace(
    *,
    trace_path: Path,
    trace: dict[str, Any],
    app_config: Any,
    dataset: DABenchPublicDataset,
    score_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_id = str(trace.get("task_id") or trace_path.parent.name)
    decision = trace.get("router_decision") or {}
    compiled = trace.get("compiled_task") or decision.get("compiled_task") or {}
    validation = trace.get("answer_validation") or {}
    audit = trace.get("semantic_consistency_audit") or {}
    operator = _get_operator_block(trace)
    answer = trace.get("answer") if isinstance(trace.get("answer"), dict) else {}
    answer_columns = answer.get("columns") or []
    answer_rows = answer.get("rows") or []
    score_info = _score_for_trace(
        trace_path=trace_path,
        trace=trace,
        app_config=app_config,
        score_summary=score_summary,
    )

    try:
        task = dataset.get_task(task_id)
        difficulty = task.difficulty
        question = task.question
    except Exception:  # noqa: BLE001
        difficulty = str(decision.get("difficulty") or "-")
        question = ""

    run_dir = _run_dir_for_trace(trace_path)
    modified_at = datetime.fromtimestamp(trace_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    matched = score_info.get("matched_count")
    gold_cols = score_info.get("gold_col_count")
    matched_text = ""
    if matched is not None and gold_cols is not None:
        matched_text = f"{matched}/{gold_cols}"

    succeeded = bool(trace.get("succeeded"))
    failure = trace.get("failure_reason") or operator.get("failure_reason") or score_info.get("error") or ""
    rows = validation.get("row_count")
    if rows is None:
        rows = len(answer_rows)
    cols = validation.get("column_count")
    if cols is None:
        cols = len(answer_columns)

    return {
        "Run": run_dir.name,
        "Task": task_id,
        "Status": "OK" if succeeded else "FAIL",
        "Score": _coerce_float(score_info.get("score")),
        "Recall": _coerce_float(score_info.get("recall")),
        "Penalty": _coerce_float(score_info.get("penalty")),
        "Matched": matched_text,
        "Difficulty": difficulty,
        "Route": decision.get("route_name") or "-",
        "Agent": decision.get("kind") or _operator_kind(trace),
        "Task Type": compiled.get("task_type") or "-",
        "Gate": audit.get("final_gate") or "-",
        "Reflect": _reflection_verdict(trace),
        "Answer": f"{rows}x{cols}",
        "Elapsed": _coerce_float(trace.get("e2e_elapsed_seconds")),
        "Failure / Score Error": _clip(failure, 180),
        "Question": _clip(question, 220),
        "Trace": _relative_artifact_path(trace_path),
        "Modified": modified_at,
        "_trace_path": str(trace_path),
        "_modified_ts": trace_path.stat().st_mtime,
    }


def _build_run_records(
    *,
    scope: Path,
    app_config: Any,
    dataset: DABenchPublicDataset,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    score_summaries: dict[Path, dict[str, dict[str, Any]]] = {}
    for trace_path in _trace_paths_for_scope(scope):
        run_dir = _run_dir_for_trace(trace_path)
        if run_dir not in score_summaries:
            score_summaries[run_dir] = _load_score_summary(run_dir)
        try:
            trace = _load_trace(trace_path)
        except Exception as exc:  # noqa: BLE001
            records.append({
                "Run": run_dir.name,
                "Task": trace_path.parent.name,
                "Status": "FAIL",
                "Score": None,
                "Recall": None,
                "Penalty": None,
                "Matched": "",
                "Difficulty": "-",
                "Route": "-",
                "Agent": "-",
                "Task Type": "-",
                "Gate": "-",
                "Reflect": "-",
                "Answer": "0x0",
                "Elapsed": None,
                "Failure / Score Error": f"trace_read_error:{exc}",
                "Question": "",
                "Trace": _relative_artifact_path(trace_path),
                "Modified": datetime.fromtimestamp(trace_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "_trace_path": str(trace_path),
                "_modified_ts": trace_path.stat().st_mtime,
            })
            continue
        records.append(
            _record_for_trace(
                trace_path=trace_path,
                trace=trace,
                app_config=app_config,
                dataset=dataset,
                score_summary=score_summaries[run_dir],
            )
        )
    records.sort(key=lambda item: (item.get("_modified_ts") or 0.0), reverse=True)
    return records


def _render_dashboard_cards(records: list[dict[str, Any]], filtered_count: int) -> None:
    total = len(records)
    succeeded = sum(1 for item in records if item.get("Status") == "OK")
    failed = total - succeeded
    scores = [item["Score"] for item in records if item.get("Score") is not None]
    mean_score = sum(scores) / len(scores) if scores else None
    perfect = sum(1 for score in scores if score >= 0.999)
    values = [
        ("records", str(total), "neutral"),
        ("shown", str(filtered_count), "neutral"),
        ("success", f"{succeeded}/{total}" if total else "0/0", "good"),
        ("failed", str(failed), "bad" if failed else "good"),
        ("mean score", "-" if mean_score is None else f"{mean_score:.3f}", "score"),
        ("score=1.0", str(perfect), "warn"),
    ]
    cards = []
    for label, value, cls in values:
        cards.append(
            f"<div class='status-card {cls}'>"
            f"<div class='label'>{escape(label)}</div>"
            f"<div class='value'>{escape(value)}</div>"
            "</div>"
        )
    st.markdown("<div class='status-strip'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def _render_selected_record_badges(row: pd.Series) -> None:
    status_cls = "ok" if row.get("Status") == "OK" else "fail"
    score = row.get("Score")
    score_text = "-" if pd.isna(score) else f"{float(score):.3f}"
    badges = [
        (status_cls, str(row.get("Status") or "-")),
        ("score", f"score {score_text}"),
        ("", f"route {row.get('Route') or '-'}"),
        ("", f"gate {row.get('Gate') or '-'}"),
        ("", f"reflect {row.get('Reflect') or '-'}"),
        ("", f"answer {row.get('Answer') or '-'}"),
    ]
    html = "".join(
        f"<span class='record-badge {cls}'>{escape(text)}</span>"
        for cls, text in badges
    )
    st.markdown(f"<div class='record-badges'>{html}</div>", unsafe_allow_html=True)


def _record_card_html(record: dict[str, Any]) -> str:
    status = str(record.get("Status") or "-")
    score_raw = record.get("Score")
    score_is_num = score_raw is not None and not (isinstance(score_raw, float) and pd.isna(score_raw))
    score_val = float(score_raw) if score_is_num else None
    score_text = "-" if score_val is None else f"{score_val:.3f}"

    card_cls = "run-card"
    if status == "OK":
        card_cls += " ok"
        if score_val is not None and score_val >= 0.999:
            card_cls += " perfect"
    elif status == "FAIL":
        card_cls += " fail"

    task_id = str(record.get("Task") or "-")
    run_name = str(record.get("Run") or "-")
    question = _clip(record.get("Question") or "(no question)", 220)
    difficulty = str(record.get("Difficulty") or "-").lower()
    route = str(record.get("Route") or "-")
    agent = str(record.get("Agent") or "-")
    gate = str(record.get("Gate") or "-")
    reflect = str(record.get("Reflect") or "-")
    answer_shape = str(record.get("Answer") or "-")
    matched = str(record.get("Matched") or "")
    failure = str(record.get("Failure / Score Error") or "")
    elapsed = record.get("Elapsed")
    elapsed_text = "-" if elapsed is None or (isinstance(elapsed, float) and pd.isna(elapsed)) else f"{float(elapsed):.1f}s"
    modified = str(record.get("Modified") or "")

    recall = record.get("Recall")
    recall_is_num = recall is not None and not (isinstance(recall, float) and pd.isna(recall))
    score_sub_parts = []
    if recall_is_num:
        score_sub_parts.append(f"recall {float(recall):.2f}")
    if matched:
        score_sub_parts.append(f"matched {matched}")
    score_sub = " · ".join(score_sub_parts) or "no score"

    badges_html: list[str] = []
    if difficulty and difficulty != "-":
        diff_cls = f"difficulty-{difficulty}" if difficulty in {"easy", "medium", "hard"} else ""
        badges_html.append(
            f"<span class='rc-badge {diff_cls}'>{escape(difficulty)}</span>"
        )
    if route and route != "-":
        badges_html.append(f"<span class='rc-badge route'>{escape(route)}</span>")
    if agent and agent not in {"-", route}:
        badges_html.append(f"<span class='rc-badge'>{escape(agent)}</span>")
    if gate and gate != "-":
        gate_cls = "gate-pass" if gate in {"ok", "pass", "accept"} else (
            "gate-fail" if gate in {"failed", "fail", "error"} else ""
        )
        badges_html.append(
            f"<span class='rc-badge {gate_cls}'>gate {escape(gate)}</span>"
        )
    if reflect and reflect != "-":
        badges_html.append(f"<span class='rc-badge'>reflect {escape(reflect)}</span>")
    if answer_shape and answer_shape != "-":
        badges_html.append(f"<span class='rc-badge'>ans {escape(answer_shape)}</span>")

    failure_block = ""
    if failure and status != "OK":
        failure_block = (
            f"<div class='rc-failure' title='{escape(failure)}'>{escape(_clip(failure, 120))}</div>"
        )

    return (
        f"<div class='{card_cls}'>"
        f"<div class='rc-head'>"
        f"<span class='rc-task'>{escape(task_id)}</span>"
        f"<span class='rc-status-pill'>{escape(status)}</span>"
        f"</div>"
        f"<div class='rc-score'>"
        f"<span class='num'>{escape(score_text)}</span>"
        f"<span class='sub'>{escape(score_sub)}</span>"
        f"</div>"
        f"<div class='rc-question'>{escape(question)}</div>"
        f"<div class='rc-badges'>{''.join(badges_html)}</div>"
        f"{failure_block}"
        f"<div class='rc-meta'>"
        f"<span>{escape(run_name)} · {escape(elapsed_text)}</span>"
        f"<span>{escape(modified)}</span>"
        f"</div>"
        f"</div>"
    )


def _render_run_dashboard(
    *,
    scope: Path,
    app_config: Any,
    dataset: DABenchPublicDataset,
) -> None:
    records = _build_run_records(scope=scope, app_config=app_config, dataset=dataset)
    st.markdown(
        "<div class='dashboard-title'>"
        "<h2>Run Dashboard</h2>"
        f"<div class='path'>{escape(_relative_artifact_path(scope))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if not records:
        st.warning("No trace.json files found for this scope.")
        return

    frame = pd.DataFrame(records)
    controls = st.columns([0.85, 0.95, 1.2, 1.4])
    status_filter = controls[0].selectbox("Status", ["All", "OK", "FAIL"], key="dashboard_status")
    routes = sorted(str(item) for item in frame["Route"].dropna().unique() if str(item))
    route_filter = controls[1].selectbox("Route", ["All", *routes], key="dashboard_route")
    score_filter = controls[2].selectbox(
        "Score",
        ["All", "Has score", "Score < 1", "Score = 1"],
        key="dashboard_score",
    )
    search_text = controls[3].text_input("Search", value="", key="dashboard_search")

    filtered = frame.copy()
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]
    if route_filter != "All":
        filtered = filtered[filtered["Route"] == route_filter]
    if score_filter == "Has score":
        filtered = filtered[filtered["Score"].notna()]
    elif score_filter == "Score < 1":
        filtered = filtered[filtered["Score"].notna() & (filtered["Score"] < 0.999)]
    elif score_filter == "Score = 1":
        filtered = filtered[filtered["Score"].notna() & (filtered["Score"] >= 0.999)]
    if search_text.strip():
        needle = search_text.strip().lower()
        haystack = (
            filtered["Task"].astype(str)
            + " "
            + filtered["Question"].astype(str)
            + " "
            + filtered["Failure / Score Error"].astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(needle, regex=False)]

    _render_dashboard_cards(records, len(filtered))

    if filtered.empty:
        st.info("No records match the current filters.")
        return

    sort_col1, sort_col2 = st.columns([1.0, 0.4])
    sort_key = sort_col1.selectbox(
        "Sort by",
        ["Most recent", "Score (low → high)", "Score (high → low)", "Status (FAIL first)", "Elapsed (slow → fast)"],
        key="dashboard_sort",
    )
    page_size = sort_col2.selectbox("Per page", [12, 24, 48, 96, 240], index=1, key="dashboard_page_size")

    sorted_df = filtered.copy()
    if sort_key == "Most recent":
        sorted_df = sorted_df.sort_values("_modified_ts", ascending=False)
    elif sort_key == "Score (low → high)":
        sorted_df = sorted_df.sort_values("Score", ascending=True, na_position="first")
    elif sort_key == "Score (high → low)":
        sorted_df = sorted_df.sort_values("Score", ascending=False, na_position="last")
    elif sort_key == "Status (FAIL first)":
        sorted_df = sorted_df.assign(_fail_first=(sorted_df["Status"] != "OK").astype(int))
        sorted_df = sorted_df.sort_values(["_fail_first", "_modified_ts"], ascending=[False, False])
        sorted_df = sorted_df.drop(columns=["_fail_first"])
    elif sort_key == "Elapsed (slow → fast)":
        sorted_df = sorted_df.sort_values("Elapsed", ascending=False, na_position="last")

    visible_df = sorted_df.head(int(page_size))
    cards_html = "".join(_record_card_html(row) for row in visible_df.to_dict(orient="records"))
    if len(sorted_df) > len(visible_df):
        st.caption(f"Showing {len(visible_df)} of {len(sorted_df)} filtered records. Increase 'Per page' to see more.")
    st.markdown(f"<div class='run-grid'>{cards_html}</div>", unsafe_allow_html=True)

    inspect_options = list(visible_df["_trace_path"])
    default_index = 0
    failures = visible_df.index[visible_df["Status"] == "FAIL"].tolist()
    if failures:
        first_failure = failures[0]
        default_index = list(visible_df.index).index(first_failure)

    def _inspect_label(path: str) -> str:
        row = visible_df[visible_df["_trace_path"] == path].iloc[0]
        score = row.get("Score")
        score_text = "-" if pd.isna(score) else f"{float(score):.3f}"
        return f"{row['Task']} · {row['Run']} · {row['Status']} · score={score_text}"

    selected = st.selectbox(
        "Inspect",
        inspect_options,
        index=default_index,
        format_func=_inspect_label,
        key="dashboard_inspect",
    )
    selected_row = visible_df[visible_df["_trace_path"] == selected].iloc[0]
    _render_selected_record_badges(selected_row)
    if st.checkbox("Show selected trace", value=False, key="dashboard_show_trace"):
        selected_path = Path(str(selected))
        _render_trace(_load_trace(selected_path), selected_path)


def _status_class(status: str) -> str:
    return {
        "ok": "ok",
        "pass": "ok",
        "accept": "ok",
        "warn": "warn",
        "revise": "warn",
        "skip": "idle",
        "idle": "idle",
        "fail": "err",
        "error": "err",
    }.get(status, "idle")


def _build_agent_steps(trace: dict[str, Any]) -> list[dict[str, str]]:
    decision = trace.get("router_decision") or {}
    compiled = trace.get("compiled_task") or decision.get("compiled_task") or {}
    operator = _get_operator_block(trace)
    validation = trace.get("answer_validation") or {}
    agentic = _agentic_trace(trace)
    plan = (agentic.get("planner") or {}).get("plan") or _semantic_plan(trace)
    debug = _debug_steps(trace)
    reflection_rounds = agentic.get("reflection_rounds") or []
    reflection = {}
    if reflection_rounds and isinstance(reflection_rounds[0], dict):
        reflection = reflection_rounds[0].get("decision") or {}
    audit = trace.get("semantic_consistency_audit") or {}
    gate = str(audit.get("final_gate") or "")

    execute_succeeded = operator.get("succeeded")
    if execute_succeeded is None and not operator:
        execute_succeeded = bool(trace.get("succeeded"))
    answer_valid = validation.get("valid")

    plan_status = "ok" if plan else "skip"
    reflect_verdict = str(reflection.get("verdict") or "skip")
    verify_status = "ok"
    if gate in {"cheap_guard_only"}:
        verify_status = "warn"
    if answer_valid is False or gate in {"failed", "error"}:
        verify_status = "fail"

    return [
        {
            "title": "Observe",
            "status": "ok" if compiled else "warn",
            "label": f"{len(compiled.get('source_capabilities') or [])} source(s)",
            "detail": f"type={compiled.get('task_type') or '-'}; ops={', '.join(compiled.get('operations') or []) or '-'}",
        },
        {
            "title": "Plan",
            "status": plan_status,
            "label": "planner" if agentic.get("planner") else "semantic plan",
            "detail": _clip((plan or {}).get("rationale") or (plan or {}).get("confidence") or "fast path"),
        },
        {
            "title": "Act",
            "status": "ok" if operator.get("program") else "warn",
            "label": _operator_kind(trace),
            "detail": f"route={decision.get('route_name') or '-'}; model={decision.get('model') or '-'}",
        },
        {
            "title": "Execute",
            "status": "ok" if execute_succeeded else "fail",
            "label": "tool run",
            "detail": _clip(operator.get("failure_reason") or _answer_shape(trace)),
        },
        {
            "title": "Observe Trace",
            "status": "ok" if debug or answer_valid else "warn",
            "label": "debug + validation",
            "detail": f"debug_keys={len(debug)}; answer_valid={answer_valid}",
        },
        {
            "title": "Reflect",
            "status": reflect_verdict if reflect_verdict in {"accept", "revise"} else "skip",
            "label": f"confidence={reflection.get('confidence') or '-'}",
            "detail": _clip("; ".join(reflection.get("issues") or []) or reflection.get("revision_instruction") or "no concrete issue"),
        },
        {
            "title": "Verify",
            "status": verify_status,
            "label": gate or "validation",
            "detail": f"judge_attempts={audit.get('judge_attempts') or 0}; answer={_answer_shape(trace)}",
        },
    ]


def _render_metric_grid(trace: dict[str, Any]) -> None:
    summary = _route_summary(trace)
    local_score = trace.get("local_score") or {}
    score_value = local_score.get("score")
    result = "OK" if summary["succeeded"] else "FAIL"
    if score_value is not None:
        result = f"{result} / {score_value:.2f}"
    values = [
        ("Result", result),
        ("Route", summary["route"] or "-"),
        ("Agent Kind", summary["kind"] or _operator_kind(trace)),
        ("Elapsed", f"{summary['elapsed_seconds'] or 0}s"),
    ]
    cards = []
    for label, value in values:
        cards.append(
            "<div class='metric-card'>"
            f"<div class='label'>{escape(str(label))}</div>"
            f"<div class='value'>{escape(str(value))}</div>"
            "</div>"
        )
    st.markdown("<div class='metric-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def _render_agent_rail(trace: dict[str, Any]) -> None:
    cards = []
    for idx, step in enumerate(_build_agent_steps(trace), start=1):
        cls = _status_class(step["status"])
        cards.append(
            f"<div class='step-card {cls}'>"
            f"<div class='step-index'>STEP {idx:02d}</div>"
            f"<div class='step-title'>{escape(step['title'])}</div>"
            f"<div class='step-status'>{escape(step['status'].upper())}</div>"
            f"<div class='step-detail'><b>{escape(step['label'])}</b><br>{escape(step['detail'])}</div>"
            "</div>"
        )
    st.markdown("<div class='agent-rail'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def _render_planner_panel(trace: dict[str, Any]) -> None:
    agentic = _agentic_trace(trace)
    planner = agentic.get("planner") or {}
    plan = planner.get("plan") or {}
    if not plan:
        st.info("No agentic planner plan was recorded for this run.")
        return
    st.markdown(
        "<div class='panel'><div class='panel-title'>Planner Decomposition</div>"
        f"<div class='tiny'>{escape(plan.get('rationale') or '')}</div></div>",
        unsafe_allow_html=True,
    )
    for subtask in plan.get("subtasks") or []:
        deps = ", ".join(subtask.get("depends_on") or []) or "none"
        st.markdown(
            "<div class='subtask'>"
            f"<span class='sid'>{escape(subtask.get('id') or '-')}</span>"
            f" · {escape(subtask.get('specialist') or '-')}"
            f"<div>{escape(subtask.get('instruction') or '')}</div>"
            f"<div class='tiny'>depends_on={escape(deps)} · expected={escape(subtask.get('expected_output') or '-')}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_reflection_panel(trace: dict[str, Any]) -> None:
    agentic = _agentic_trace(trace)
    rounds = agentic.get("reflection_rounds") or []
    if not rounds:
        st.info("No reflection round was recorded.")
        return
    for item in rounds:
        decision = item.get("decision") or {}
        st.markdown(
            "<div class='panel'>"
            f"<div class='panel-title'>Reflection Round {escape(str(item.get('round', 0)))}</div>"
            f"<div><b>verdict:</b> {escape(decision.get('verdict') or '-')} · "
            f"<b>confidence:</b> {escape(decision.get('confidence') or '-')}</div>"
            f"<div class='tiny'>issues: {escape('; '.join(decision.get('issues') or []) or 'none')}</div>"
            f"<div class='tiny'>revision: {escape(decision.get('revision_instruction') or 'none')}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_verification_panel(trace: dict[str, Any]) -> None:
    audit = trace.get("semantic_consistency_audit") or {}
    guard = _cheap_guard(trace)
    judge = _semantic_consistency(trace)
    cols = st.columns(3)
    cols[0].metric("Final Gate", audit.get("final_gate") or "-")
    cols[1].metric("Judge Attempts", audit.get("judge_attempts") or 0)
    cols[2].metric("Answer Valid", str((trace.get("answer_validation") or {}).get("valid")))
    if guard:
        st.write("Cheap guard")
        st.json(guard, expanded=False)
    if judge:
        st.write("Consistency judge")
        st.json(judge, expanded=False)


def _render_trace(trace: dict[str, Any], trace_path: Path) -> None:
    summary = _route_summary(trace)
    st.markdown(
        "<div class='hero'>"
        "<h1>Data Agent Run</h1>"
        f"<p>{escape(str(trace.get('task_id') or '-'))} · "
        f"{escape(str(summary.get('task_type') or '-'))} · "
        f"{escape(str(summary.get('operations') or '-'))}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _render_metric_grid(trace)
    _render_agent_rail(trace)

    tab_agent, tab_evidence, tab_program, tab_trace = st.tabs(
        ["Agent Steps", "Evidence", "Program", "Trace"]
    )
    operator = _get_operator_block(trace)

    with tab_agent:
        left, right = st.columns([1.15, 0.85])
        with left:
            _render_planner_panel(trace)
        with right:
            _render_reflection_panel(trace)
        st.subheader("Final Answer")
        frame = _answer_frame(trace.get("answer"))
        if frame is None:
            st.warning("No answer table was produced.")
        else:
            st.dataframe(frame, width="stretch", hide_index=True)

    with tab_evidence:
        st.subheader("Verification")
        _render_verification_panel(trace)
        st.subheader("Observed Debug Steps")
        debug = _debug_steps(trace)
        if debug:
            st.json(debug, expanded=False)
        else:
            st.info("No structured debug_steps were found in stdout.")
        st.subheader("Context Manifest")
        manifest_rows = []
        for item in _iter_manifest(trace):
            if item.get("path"):
                manifest_rows.append({
                    "path": item.get("path"),
                    "kind": item.get("kind"),
                    "rows": str(item.get("row_count") or item.get("record_count") or ""),
                    "columns": ", ".join(item.get("columns") or [])[:180],
                })
        if manifest_rows:
            st.dataframe(pd.DataFrame(manifest_rows), width="stretch", hide_index=True)
        else:
            st.info("No file manifest entries were recorded.")

    with tab_program:
        st.subheader("Generated Tool Program")
        st.code(operator.get("program") or "(no program)", language="python")
        stdout = str(operator.get("exec_stdout") or "")
        stderr = str(operator.get("exec_stderr") or "")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("stdout tail")
            st.code(stdout[-8000:] or "(empty stdout)", language="text")
        with c2:
            st.caption("stderr tail")
            st.code(stderr[-8000:] or "(empty stderr)", language="text")

    with tab_trace:
        st.caption(str(trace_path))
        st.json(trace)


def main() -> None:
    st.set_page_config(page_title="DABench Demo", layout="wide")
    st.markdown(_APP_CSS, unsafe_allow_html=True)
    st.title("DABench Agent Theater")
    st.caption("A step-by-step view of how the data agent observes, plans, acts, reflects, verifies, and submits.")

    with st.sidebar:
        st.header("Agent Run")
        config_text = st.text_input("Config", value=str(DEFAULT_CONFIG))
        task_id = st.text_input("Task ID", value="")
        trace_text = st.text_input("Trace JSON (optional)", value="")
        run_dir_text = st.text_input("Run Directory", value=str(PROJECT_ROOT / "artifacts" / "runs"))
        show_task = st.checkbox("Show task context summary", value=True)
        run_clicked = st.button("Run Agent", type="primary", width="stretch")
        load_dashboard_clicked = st.button("Load Run Dashboard", width="stretch")
        latest_run_dashboard_clicked = st.button("Load Latest Run Dashboard", width="stretch")
        load_trace_clicked = st.button("Load Trace", width="stretch")
        latest_trace_clicked = st.button("Load Latest Trace", width="stretch")

    config_path = Path(config_text).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    try:
        app_config = load_app_config(config_path)
        dataset = DABenchPublicDataset(app_config.dataset.root_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load config/dataset: {exc}")
        return

    if load_dashboard_clicked or latest_run_dashboard_clicked:
        if latest_run_dashboard_clicked:
            dashboard_scope = _latest_run_dir()
            if dashboard_scope is None:
                st.error("No run directories with trace.json files found under artifacts/runs.")
                return
        else:
            dashboard_scope = Path(run_dir_text).expanduser()
            if not dashboard_scope.is_absolute():
                dashboard_scope = (PROJECT_ROOT / dashboard_scope).resolve()
        if not dashboard_scope.exists():
            st.error(f"Run scope does not exist: {dashboard_scope}")
            return
        st.session_state["view_mode"] = "dashboard"
        st.session_state["dashboard_scope"] = str(dashboard_scope)

    if load_trace_clicked or latest_trace_clicked:
        if latest_trace_clicked:
            trace_path = _latest_trace_path()
            if trace_path is None:
                st.error("No trace.json files found under artifacts/runs.")
                return
        elif not trace_text.strip():
            st.error("Please provide a trace.json path.")
            return
        else:
            trace_path = Path(trace_text).expanduser()
            if not trace_path.is_absolute():
                trace_path = (PROJECT_ROOT / trace_path).resolve()
        st.session_state["view_mode"] = "trace"
        st.session_state["trace_path"] = str(trace_path)

    if not run_clicked:
        view_mode = st.session_state.get("view_mode")
        if view_mode == "dashboard":
            dashboard_scope = Path(str(st.session_state.get("dashboard_scope") or ""))
            if not dashboard_scope.exists():
                st.error(f"Run scope does not exist: {dashboard_scope}")
                return
            _render_run_dashboard(scope=dashboard_scope, app_config=app_config, dataset=dataset)
            return
        if view_mode == "trace":
            trace_path = Path(str(st.session_state.get("trace_path") or ""))
            try:
                trace = _load_trace(trace_path)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load trace: {exc}")
                return
            _render_trace(trace, trace_path)
            return

    if show_task and task_id:
        try:
            task = dataset.get_task(task_id)
            st.markdown(
                "<div class='panel'>"
                f"<div class='panel-title'>{escape(task.task_id)} · {escape(task.difficulty)}</div>"
                f"<div>{escape(task.question)}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Task preview failed: {exc}")

    if not run_clicked:
        st.markdown(
            "<div class='hero'><h1>Ready</h1>"
            "<p>Pick a task, then watch the agent move through each decision point.</p></div>",
            unsafe_allow_html=True,
        )
        return

    if not task_id.strip():
        st.error("Please enter a task id.")
        return

    run_id = "demo-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        artifact_payload, trace_path = _run_task_live(
            task_id=task_id.strip(),
            config_path=config_path,
            run_id=run_id,
        )
        trace = _load_trace(trace_path)
        st.session_state["view_mode"] = "trace"
        st.session_state["trace_path"] = str(trace_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Run failed: {exc}")
        return

    st.success(f"Run saved to {artifact_payload.get('task_output_dir')}")
    _render_trace(trace, trace_path)


if __name__ == "__main__":
    main()
