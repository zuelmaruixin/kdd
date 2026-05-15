from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
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

# Unicode separators — declared at module level because Python 3.11
# disallows backslashes inside f-string expression parts, so we can't
# write "\u00b7" inline in f-strings. (Py3.12 relaxes this.)
_DOT = " \u00b7 "


# ============================================================================
# CSS — single-page theater layout
# ============================================================================

_APP_CSS = """
<style>
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
}

/* ====== Theater hero ====== */
.theater-hero {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 1rem;
  align-items: center;
  padding: 1rem 1.2rem;
  background: linear-gradient(120deg, #08365d 0%, #2a6f73 100%);
  color: white;
  border-radius: 10px;
  margin: 0 0 .8rem 0;
}
.theater-hero .th-task {
  font-size: .76rem; font-weight: 800;
  letter-spacing: .08em; text-transform: uppercase;
  color: #c8e6f4;
}
.theater-hero .th-question {
  font-size: 1.04rem; line-height: 1.45; margin-top: .25rem;
  color: #ffffff; font-weight: 600;
}
.theater-hero .th-meta {
  font-size: .8rem; color: #d8eef7; margin-top: .35rem;
}
.theater-hero .th-state {
  display: flex; align-items: center; gap: .6rem;
  font-size: .82rem; font-weight: 800;
  background: rgba(255,255,255,.12); border-radius: 8px;
  padding: .55rem .85rem; align-self: start;
  letter-spacing: .03em; white-space: nowrap;
}
.theater-hero .pulse {
  display: inline-block; width: .68rem; height: .68rem;
  border-radius: 50%; background: #ffd166;
  box-shadow: 0 0 0 0 rgba(255, 209, 102, .6);
  animation: livePulse 1.2s infinite;
}
.theater-hero.complete .pulse { background: #4ade80; animation: none; }
.theater-hero.failed   .pulse { background: #f97373; animation: none; }
.theater-hero.replay   .pulse { background: #94a3b8; animation: none; }
.theater-hero.idle     .pulse { background: #94a3b8; animation: none; }
@keyframes livePulse {
  0%   { box-shadow: 0 0 0 0 rgba(255, 209, 102, .6); }
  60%  { box-shadow: 0 0 0 12px rgba(255, 209, 102, 0); }
  100% { box-shadow: 0 0 0 0 rgba(255, 209, 102, 0); }
}

/* ====== Stage rail (7 pills) ====== */
.stage-rail {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: .35rem;
  margin: .2rem 0 .9rem 0;
}
.stage-pill {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: .5rem .6rem;
  display: flex; flex-direction: column; gap: .12rem;
  min-height: 78px;
}
.stage-pill .sp-title {
  font-size: .66rem; font-weight: 800;
  letter-spacing: .05em; text-transform: uppercase; color: #94a3b8;
}
.stage-pill .sp-label {
  font-size: .82rem; font-weight: 800; color: #0f172a;
  overflow-wrap: anywhere;
}
.stage-pill .sp-detail {
  font-size: .68rem; color: #64748b; line-height: 1.3;
  overflow-wrap: anywhere;
}
.stage-pill.idle .sp-label, .stage-pill.idle .sp-detail { color: #cbd5e1; }
.stage-pill.run  { background: #fff7e0; border-color: #fde8b8; }
.stage-pill.run  .sp-title { color: #7a4d00; }
.stage-pill.ok   { background: #ecf8f3; border-color: #c5eadf; }
.stage-pill.ok   .sp-title { color: #04705d; }
.stage-pill.warn { background: #fff4de; border-color: #fde8b8; }
.stage-pill.warn .sp-title { color: #7a4d00; }
.stage-pill.err  { background: #fdeceb; border-color: #f3c7c3; }
.stage-pill.err  .sp-title { color: #a92822; }

/* ====== Live stream of parsed cards ====== */
.stream-title {
  font-size: .72rem; font-weight: 800;
  letter-spacing: .06em; text-transform: uppercase;
  color: #64748b; margin: .35rem 0 .3rem 0;
}
.stream {
  display: flex; flex-direction: column; gap: .55rem;
  padding: .45rem; border-radius: 10px;
  border: 1px solid #dce7f3; background: #f7fbff;
  max-height: 720px; overflow-y: auto;
}
.stream.empty { color: #94a3b8; font-style: italic; padding: 1.5rem; text-align: center; }
.sr {
  position: relative; background: #ffffff;
  border: 1px solid #dce7f3; border-radius: 9px;
  padding: .65rem .8rem .65rem 1.05rem;
  display: flex; flex-direction: column; gap: .4rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
.sr::before {
  content: ""; position: absolute;
  top: 0; left: 0; width: 5px; height: 100%;
  background: #94a3b8; border-radius: 9px 0 0 9px;
}
.sr.ok::before   { background: #16856f; }
.sr.warn::before { background: #c4821a; }
.sr.err::before  { background: #c2413a; }
.sr.run::before  { background: #2f6fb3; }
.sr.info::before { background: #94a3b8; }
.sr-head {
  display: flex; align-items: center; gap: .55rem;
}
.sr-tag {
  display: inline-block; padding: .16rem .45rem; border-radius: 5px;
  background: #e2e8f0; color: #334155;
  font-size: .66rem; font-weight: 800;
  letter-spacing: .05em; text-transform: uppercase; flex-shrink: 0;
}
.sr.ok   .sr-tag { background: #cbe9dd; color: #04705d; }
.sr.warn .sr-tag { background: #fde8b8; color: #7a4d00; }
.sr.err  .sr-tag { background: #fbcdc9; color: #a92822; }
.sr.run  .sr-tag { background: #cfdcf0; color: #1d5d9b; }
.sr-title {
  flex: 1; font-weight: 800; color: #0f172a; font-size: .92rem;
  overflow-wrap: anywhere; line-height: 1.3;
}
.sr-meta {
  font-size: .7rem; color: #64748b; flex-shrink: 0;
  font-variant-numeric: tabular-nums;
}
.sr-thought {
  font-style: italic; color: #475569; font-size: .82rem;
  border-left: 3px solid #cbd5e1; padding: .12rem .6rem;
  line-height: 1.4; overflow-wrap: anywhere;
}
.sr-subtitle {
  color: #1f2937; font-size: .84rem; line-height: 1.4;
  overflow-wrap: anywhere;
}
.sr-code {
  background: #0f172a; color: #e2e8f0; border-radius: 6px;
  padding: .55rem .7rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .74rem; line-height: 1.45;
  max-height: 280px; overflow: auto;
  white-space: pre; word-break: normal; margin: 0;
}
.sr-code.sql { background: #1e293b; }
.sr-obs {
  background: #f1f5f9; border-radius: 6px; padding: .45rem .6rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .74rem; line-height: 1.4; color: #1f2937;
  max-height: 220px; overflow: auto;
  white-space: pre-wrap; word-break: break-word; margin: 0;
}
.sr-badges { display: flex; flex-wrap: wrap; gap: .28rem; }
.sr-badge {
  font-size: .66rem; font-weight: 700;
  padding: .14rem .42rem; border-radius: 5px;
  background: #f1f5f9; color: #334155; letter-spacing: .02em;
}
.sr-badge.good { background: #cbe9dd; color: #04705d; }
.sr-badge.bad  { background: #fbcdc9; color: #a92822; }
.sr-badge.info { background: #dbeafe; color: #1d4ed8; }

/* ====== Side stats / file list / answer box ====== */
.side-box {
  border: 1px solid #dce7f3; border-radius: 10px;
  padding: .65rem .85rem; background: #ffffff;
  margin-bottom: .55rem;
}
.side-box .sb-title {
  font-size: .68rem; font-weight: 800; letter-spacing: .06em;
  text-transform: uppercase; color: #64748b;
  margin-bottom: .45rem;
}
.side-box .sb-row {
  display: flex; justify-content: space-between; gap: .55rem;
  padding: .22rem 0; border-bottom: 1px dashed #e2e8f0;
  font-size: .82rem;
}
.side-box .sb-row:last-child { border-bottom: none; }
.side-box .sb-row .label { color: #64748b; }
.side-box .sb-row .value {
  color: #0f172a; font-weight: 800;
  font-variant-numeric: tabular-nums;
}
.side-box.empty { color: #94a3b8; font-style: italic; }
.side-box .sb-answer-table {
  width: 100%; border-collapse: collapse; font-size: .76rem;
}
.side-box .sb-answer-table th, .side-box .sb-answer-table td {
  border-bottom: 1px solid #e2e8f0; padding: .25rem .35rem;
  text-align: left; vertical-align: top;
}
.side-box .sb-answer-table th { color: #475569; font-weight: 800; }
.side-box .sb-answer-table td { color: #0f172a; }
.side-box .sb-file {
  font-family: ui-monospace, monospace; font-size: .74rem;
  color: #0f172a; padding: .2rem 0; word-break: break-all;
}
.side-box .sb-file .muted { color: #64748b; }

/* ====== Top hero (idle screen) ====== */
.hero {
  padding: 1.1rem 1.4rem;
  border: 1px solid #d7e2ef;
  border-radius: 10px;
  background: linear-gradient(120deg, rgba(8, 54, 93, .96), rgba(22, 89, 111, .92));
  color: white; margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 .2rem 0; font-size: 2rem; }
.hero p { margin: 0; color: #d8eef7; font-size: .96rem; }

/* ====== Run dashboard summary cards ====== */
.dashboard-title {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 1rem; margin: .4rem 0 .8rem 0;
}
.dashboard-title h2 { margin: 0; color: #0f172a; font-size: 1.35rem; }
.dashboard-title .path { color: #64748b; font-size: .78rem; overflow-wrap: anywhere; text-align: right; }
.status-strip {
  display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: .65rem; margin: .6rem 0 1rem 0;
}
.status-card {
  padding: .78rem .88rem; border: 1px solid #dde7f0; border-radius: 8px;
  background: #ffffff;
}
.status-card.good    { border-left: 5px solid #16856f; }
.status-card.bad     { border-left: 5px solid #c2413a; }
.status-card.score   { border-left: 5px solid #2f6fb3; }
.status-card.warn    { border-left: 5px solid #c4821a; }
.status-card.neutral { border-left: 5px solid #64748b; }
.status-card .label {
  color: #64748b; font-size: .72rem;
  text-transform: uppercase; letter-spacing: .03em;
}
.status-card .value {
  color: #0f172a; font-size: 1.2rem; font-weight: 800; margin-top: .15rem;
}

/* ====== Run dashboard (history) card grid ====== */
.run-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: .85rem;
  margin: .6rem 0 1rem 0;
}
.run-card-link {
  text-decoration: none; color: inherit; display: block;
  border-radius: 10px;
}
.run-card-link:hover .run-card {
  transform: translateY(-2px);
  box-shadow: 0 6px 14px rgba(15, 23, 42, .10);
  border-color: #1d5d9b;
}
.run-card {
  background: #ffffff;
  border: 1px solid #dde7f0;
  border-radius: 10px;
  padding: .85rem .95rem;
  display: flex; flex-direction: column; gap: .55rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
  position: relative; overflow: hidden;
  cursor: pointer;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.run-card::before {
  content: ""; position: absolute;
  top: 0; left: 0; width: 6px; height: 100%;
  background: #94a3b8;
}
.run-card.ok::before      { background: #16856f; }
.run-card.fail::before    { background: #c2413a; }
.run-card.perfect::before { background: #2f6fb3; }
.run-card .rc-head {
  display: flex; align-items: center; justify-content: space-between; gap: .5rem;
}
.run-card .rc-task {
  font-weight: 800; font-size: .92rem; color: #0f172a; letter-spacing: .01em;
}
.run-card .rc-score { display: flex; align-items: baseline; gap: .55rem; }
.run-card .rc-score .num {
  font-size: 2rem; font-weight: 800; letter-spacing: -.02em;
  font-variant-numeric: tabular-nums; color: #0f172a;
}
.run-card.ok      .rc-score .num { color: #04705d; }
.run-card.fail    .rc-score .num { color: #a92822; }
.run-card.perfect .rc-score .num { color: #1d5d9b; }
.run-card .rc-score .sub { font-size: .76rem; color: #475569; font-weight: 600; }
.run-card .rc-status-pill {
  display: inline-block; padding: .18rem .55rem; border-radius: 999px;
  font-size: .68rem; font-weight: 800; letter-spacing: .04em;
  background: #e2e8f0; color: #475569;
}
.run-card.ok   .rc-status-pill { background: #c5eadf; color: #04705d; }
.run-card.fail .rc-status-pill { background: #f3c7c3; color: #a92822; }
.run-card .rc-question {
  font-size: .82rem; color: #1f2937; line-height: 1.4;
  max-height: 4.2em; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
}
.run-card .rc-badges { display: flex; flex-wrap: wrap; gap: .3rem; }
.run-card .rc-badge {
  background: #f1f5f9; color: #334155;
  font-size: .68rem; font-weight: 700;
  padding: .15rem .42rem; border-radius: 6px;
  letter-spacing: .02em; text-transform: uppercase;
}
.run-card .rc-badge.route          { background: #dbeafe; color: #1d4ed8; }
.run-card .rc-badge.gate-pass      { background: #c5eadf; color: #04705d; }
.run-card .rc-badge.gate-fail      { background: #f3c7c3; color: #a92822; }
.run-card .rc-badge.difficulty-easy   { background: #d6eedb; color: #166534; }
.run-card .rc-badge.difficulty-medium { background: #fde9c5; color: #92400e; }
.run-card .rc-badge.difficulty-hard   { background: #f3d2cd; color: #991b1b; }
.run-card .rc-meta {
  display: flex; justify-content: space-between;
  font-size: .68rem; color: #64748b; font-variant-numeric: tabular-nums;
}
.run-card .rc-failure {
  background: #fff5f4; border: 1px solid #f7d2cf; border-radius: 6px;
  padding: .35rem .55rem; font-size: .72rem; color: #a92822;
  font-family: ui-monospace, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.app-topbar {
  background: linear-gradient(120deg, #08365d 0%, #2a6f73 100%);
  color: #fff;
  padding: .8rem 1.1rem;
  border-radius: 10px;
  margin: 0 0 .55rem 0;
  display: flex; flex-direction: column; gap: .1rem;
  box-shadow: 0 2px 8px rgba(0,0,0,.10);
}
.app-topbar .tb-title {
  font-size: 1.05rem; font-weight: 800; letter-spacing: .02em;
}
.app-topbar .tb-sub {
  font-size: .76rem; color: #cfe6ee;
}

/* Top tabs: tighter, less busy. */
.stTabs [data-baseweb="tab-list"] {
  gap: .25rem;
  border-bottom: 1px solid #dde7f0;
}
.stTabs [data-baseweb="tab"] {
  padding: .45rem .95rem;
  font-weight: 600;
  font-size: .85rem;
}

.batch-progress {
  position: sticky; top: 0; z-index: 50;
  background: linear-gradient(120deg, #08365d 0%, #2a6f73 100%);
  color: #fff; padding: .55rem .85rem; border-radius: 8px;
  font-size: .82rem; margin: .6rem 0; box-shadow: 0 2px 8px rgba(0,0,0,.12);
}
.batch-progress code {
  background: rgba(255,255,255,.18); padding: .05rem .35rem;
  border-radius: 4px; color: #fff; font-size: .78rem;
}
.batch-task-header {
  margin: 1.1rem 0 .35rem 0;
  padding: .35rem .6rem;
  background: #eef3f8; border-left: 3px solid #08365d;
  border-radius: 4px;
  font-size: .82rem; color: #1a3a5c; font-weight: 600;
}
.batch-task-header code {
  background: #fff; padding: .05rem .3rem; border-radius: 3px;
  font-size: .8rem; color: #08365d;
}

@media (max-width: 1100px) {
  .stage-rail   { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .status-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .dashboard-title { display: block; }
  .dashboard-title .path { text-align: left; margin-top: .2rem; }
  .theater-hero { grid-template-columns: 1fr; }
}
</style>
"""


# ============================================================================
# Subprocess runner (unchanged)
# ============================================================================

_SUBPROCESS_RUNNER = r"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True, write_through=True)
sys.stderr.reconfigure(line_buffering=True, write_through=True)

from data_agent_baseline.config import load_app_config
from data_agent_baseline.progress import ProgressLogger, set_progress_logger
from data_agent_baseline.run.runner import create_run_output_dir, run_single_task

task_id = sys.argv[1]
config_path = Path(sys.argv[2])
run_id = sys.argv[3]
dataset_root_override = sys.argv[4] if len(sys.argv) > 4 else ""
batch_run_dir_override = os.environ.get("DEMO_BATCH_RUN_DIR", "")

print('__DEMO_EVENT__' + json.dumps({"type": "task_start", "task_id": task_id, "question": "(loading)", "difficulty": ""}, ensure_ascii=False), flush=True)

config = load_app_config(config_path)
if dataset_root_override:
    config = replace(config, dataset=replace(config.dataset, root_path=Path(dataset_root_override)))
config = replace(config, run=replace(config.run, run_id=run_id, max_workers=1))
set_progress_logger(ProgressLogger(enabled=True, lang="zh", emit_demo_events=True))

if batch_run_dir_override:
    run_output_dir = Path(batch_run_dir_override)
    run_output_dir.mkdir(parents=True, exist_ok=True)
else:
    _, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)
artifact = run_single_task(
    task_id=task_id,
    config=config,
    run_output_dir=run_output_dir,
)
print("DEMO_RESULT_JSON=" + json.dumps(artifact.to_dict(), ensure_ascii=False), flush=True)
"""


# ============================================================================
# Trace / artifact lookup helpers
# ============================================================================


def _load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _latest_trace_path() -> Path | None:
    candidates = list((PROJECT_ROOT / "artifacts" / "runs").glob("**/trace.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_run_dir() -> Path | None:
    traces = list((PROJECT_ROOT / "artifacts" / "runs").glob("*/task_*/trace.json"))
    if not traces:
        return None
    latest_trace = max(traces, key=lambda p: p.stat().st_mtime)
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


def _clip(value: Any, limit: int = 140) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"


# ============================================================================
# Stage rail (7 phases of the agent loop)
# ============================================================================

# Stage rail — mirrors the ReAct harness trajectory.
# Each stage activates from a concrete event/action, so the rail walks
# along the agent's actual path instead of skipping ahead with "(implicit)"
# placeholders.
_STAGE_ORDER = ("route", "discover", "inspect", "compute", "draft", "verify", "score")
_STAGE_LABEL = {
    "route":    "Route",
    "discover": "Discover",
    "inspect":  "Inspect",
    "compute":  "Compute",
    "draft":    "Draft",
    "verify":   "Verify",
    "score":    "Score",
}
_STAGE_BY_EVENT = {
    "task_start":              "route",
    "router_decision":         "route",
    "router_cascade":          "route",
    "task_compiled":           "route",
    "budget_started":          "route",
    # Multi-agent fallback events (agentic_operator route) — folded
    # into the same rail so ablation runs still light up.
    "planner_done":            "compute",
    "structured_extract_start":"inspect",
    "structured_extract_done": "inspect",
    "codegen_program":         "compute",
    "specialist_start":        "compute",
    "specialist_done":         "compute",
    "codegen_executed":        "compute",
    "synthesizer_done":        "compute",
    "codegen_debug":           "compute",
    "reflection_start":        "verify",
    "reflection_done":         "verify",
    "reasoner_repair_start":   "verify",
    "reasoner_repair_done":    "verify",
    "cross_verify_start":      "verify",
    "cross_verify_done":       "verify",
    "score":                   "score",
}

_REACT_DISCOVER_ACTIONS = frozenset({"list_context"})
_REACT_INSPECT_ACTIONS = frozenset({
    "read_csv", "read_json", "read_doc", "inspect_sqlite_schema",
})
_REACT_COMPUTE_ACTIONS = frozenset({"execute_python", "execute_context_sql"})

# Aliases for legacy call-sites that still bucket actions as
# "observe vs act" for stats accounting.
_REACT_OBSERVE_ACTIONS = _REACT_DISCOVER_ACTIONS | _REACT_INSPECT_ACTIONS
_REACT_ACT_ACTIONS = _REACT_COMPUTE_ACTIONS


def _stage_for_react_step(action: str, prior_answer_count: int) -> str:
    if action in _REACT_DISCOVER_ACTIONS:
        return "discover" if prior_answer_count == 0 else "verify"
    if action in _REACT_INSPECT_ACTIONS:
        return "inspect" if prior_answer_count == 0 else "verify"
    if action in _REACT_COMPUTE_ACTIONS:
        return "compute" if prior_answer_count == 0 else "verify"
    if action == "answer":
        return "draft" if prior_answer_count == 0 else "verify"
    return "compute"


def _initial_stage_state() -> dict[str, dict[str, str]]:
    return {name: {"status": "idle", "label": "", "detail": ""} for name in _STAGE_ORDER}


# ============================================================================
# ParsedStep — the single shape used by both live stream + replay
# ============================================================================

_KIND_TAG = {
    "task":    "TASK",
    "route":   "ROUTE",
    "compile": "COMPILE",
    "budget":  "BUDGET",
    "plan":    "PLAN",
    "tool":    "TOOL",
    "code":    "CODE",
    "sql":     "SQL",
    "obs":     "OBS",
    "draft":   "DRAFT",
    "answer":  "ANSWER",
    "reflect": "REFLECT",
    "verify":  "VERIFY",
    "score":   "SCORE",
    "extract": "EXTRACT",
    "info":    "INFO",
    "warn":    "WARN",
    "error":   "ERROR",
}


@dataclass
class ParsedStep:
    """Unified card model — live stream and replay both render from this."""
    kind: str
    title: str
    subtitle: str = ""
    status: str = "info"   # ok / warn / err / run / info
    thought: str = ""
    code: str = ""
    code_lang: str = "python"
    observation: str = ""
    badges: list[tuple[str, str]] = field(default_factory=list)
    elapsed: float | None = None
    step_index: int | None = None
    stage: str = ""

    @property
    def tag(self) -> str:
        return _KIND_TAG.get(self.kind, self.kind.upper())


# ============================================================================
# Event → ParsedStep parsing
# ============================================================================


def _format_observation_summary(obs: Any) -> str:
    """Compact summary of a tool observation for the obs box."""
    if obs is None:
        return ""
    if isinstance(obs, str):
        return _clip(obs, 600)
    if not isinstance(obs, dict):
        try:
            return _clip(json.dumps(obs, ensure_ascii=False, default=str), 600)
        except (TypeError, ValueError):
            return _clip(str(obs), 600)
    if "error" in obs and obs.get("error"):
        return "error: " + _clip(str(obs["error"]), 500)
    if "stdout" in obs and obs.get("stdout"):
        return _clip(str(obs["stdout"]), 600)
    if "files" in obs and isinstance(obs.get("files"), list):
        files = obs["files"]
        head = ", ".join(str(f) for f in files[:8])
        return f"{len(files)} entry: {head}"
    if "tables" in obs and isinstance(obs.get("tables"), list):
        tables = obs["tables"]
        names = []
        for t in tables[:6]:
            if isinstance(t, dict):
                names.append(str(t.get("name") or t.get("table") or "?"))
            else:
                names.append(str(t))
        return f"{len(tables)} table(s): {', '.join(names)}"
    if isinstance(obs.get("columns"), list) and isinstance(obs.get("rows"), list):
        cols = obs["columns"]
        rows = obs["rows"]
        sample = ""
        if rows:
            first = rows[0]
            if isinstance(first, list):
                sample = " \u00b7 sample: " + ", ".join(str(c)[:24] for c in first[:6])
        return f"{len(cols)} column(s) \u00d7 {len(rows)} row(s){sample}"
    if isinstance(obs.get("schema"), (dict, list)):
        return _clip(json.dumps(obs["schema"], ensure_ascii=False, default=str), 600)
    if obs.get("text"):
        return _clip(str(obs["text"]), 600)
    try:
        return _clip(json.dumps(obs, ensure_ascii=False, default=str), 600)
    except (TypeError, ValueError):
        return _clip(str(obs), 600)


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_react_action(
    *,
    action: str,
    action_input: Any,
    step_index: Any,
    ok: bool,
    cached: bool,
    elapsed: float | None,
    prior_answer_count: int,
    thought: str = "",
    observation: Any = None,
) -> ParsedStep:
    """Turn one ReAct step into a ParsedStep."""
    ai = action_input if isinstance(action_input, dict) else {}
    obs_text = _format_observation_summary(observation) if observation is not None else ""

    if action == "execute_python":
        code = str(ai.get("code") or "")
        return ParsedStep(
            kind="code", title="execute_python",
            step_index=_to_int(step_index),
            code=code, code_lang="python",
            status="ok" if ok else "err",
            elapsed=elapsed, thought=thought, observation=obs_text,
            badges=[("info", "cached")] if cached else [],
            stage="verify" if prior_answer_count > 0 else "compute",
        )
    if action == "execute_context_sql":
        sql = str(ai.get("sql") or "")
        return ParsedStep(
            kind="sql", title="execute_context_sql",
            step_index=_to_int(step_index),
            code=sql, code_lang="sql",
            status="ok" if ok else "err",
            elapsed=elapsed, thought=thought, observation=obs_text,
            badges=[("info", "cached")] if cached else [],
            stage="verify" if prior_answer_count > 0 else "compute",
        )
    if action in _REACT_OBSERVE_ACTIONS:
        hint = ""
        if ai.get("path"):
            hint = str(ai["path"])
        elif ai.get("max_depth") is not None:
            hint = f"depth={ai['max_depth']}"
        ps_stage = "discover" if action in _REACT_DISCOVER_ACTIONS else "inspect"
        if prior_answer_count > 0:
            ps_stage = "verify"
        return ParsedStep(
            kind="tool", title=action, subtitle=hint,
            step_index=_to_int(step_index),
            status="ok" if ok else "warn",
            elapsed=elapsed, thought=thought, observation=obs_text,
            badges=[("info", "cached")] if cached else [],
            stage=ps_stage,
        )
    if action == "answer":
        cols = ai.get("columns") if isinstance(ai, dict) else None
        rows = ai.get("rows") if isinstance(ai, dict) else None
        n_cols = len(cols) if isinstance(cols, list) else 0
        n_rows = len(rows) if isinstance(rows, list) else 0
        is_draft = prior_answer_count == 0
        return ParsedStep(
            kind="draft" if is_draft else "answer",
            title="draft answer \u00b7 self-verify" if is_draft else "final answer committed",
            subtitle=f"{n_cols} column(s) \u00d7 {n_rows} row(s)",
            step_index=_to_int(step_index),
            status="run" if is_draft else "ok",
            elapsed=elapsed, thought=thought,
            badges=[("info", str(c)[:24]) for c in (cols or [])[:6]],
            stage="draft" if is_draft else "verify",
        )
    # Unknown action
    try:
        args_text = json.dumps(ai, ensure_ascii=False)
    except (TypeError, ValueError):
        args_text = str(ai)
    return ParsedStep(
        kind="info", title=action or "react step",
        subtitle=_clip(args_text, 220),
        step_index=_to_int(step_index),
        status="ok" if ok else "warn",
        elapsed=elapsed, thought=thought, observation=obs_text,
        stage="compute",
    )


def _parse_event(event: dict[str, Any], *, prior_answer_count: int) -> ParsedStep | None:
    """Convert one streamed __DEMO_EVENT__ into a ParsedStep (or None to skip)."""
    et = event.get("type")
    elapsed = event.get("elapsed")
    try:
        elapsed_f = float(elapsed) if elapsed is not None else None
    except (TypeError, ValueError):
        elapsed_f = None

    if et == "task_start":
        return ParsedStep(
            kind="task",
            title=f"Task {event.get('task_id') or '?'} \u00b7 starting",
            subtitle=_clip(event.get("question") or "(loading)", 500),
            status="run", elapsed=elapsed_f,
            badges=[("info", f"difficulty {event.get('difficulty') or '-'}")],
            stage="route",
        )
    if et == "task_end":
        ok = bool(event.get("succeeded"))
        return ParsedStep(
            kind="task",
            title="Run completed" if ok else "Run failed",
            subtitle=_clip(event.get("failure_reason") or ("Answer ready" if ok else "see logs"), 300),
            status="ok" if ok else "err", elapsed=elapsed_f,
            stage="verify" if ok else "compute",
        )
    if et == "router_decision":
        return ParsedStep(
            kind="route",
            title=f"route \u2192 {event.get('route_name') or '?'}",
            subtitle=f"kind={event.get('kind') or '-'} \u00b7 model={event.get('model') or '-'} \u00b7 task_type={event.get('task_type') or '-'}",
            status="info", elapsed=elapsed_f,
            badges=[("info", f"difficulty {event.get('difficulty') or '-'}")],
            stage="route",
        )
    if et == "router_cascade":
        return ParsedStep(
            kind="route", title=f"cascade \u2192 {event.get('to_route') or '?'}",
            subtitle=_clip(event.get("reason") or "", 220),
            status="warn", elapsed=elapsed_f,
            badges=[("bad", f"from {event.get('from_route') or '-'}")],
            stage="route",
        )
    if et == "task_compiled":
        ops = event.get("operations") or []
        return ParsedStep(
            kind="compile",
            title=f"task type \u2192 {event.get('task_type') or '-'}",
            subtitle=f"answer={event.get('answer_type') or '-'} \u00b7 sources={event.get('data_source_count') or 0}",
            status="info", elapsed=elapsed_f,
            badges=[("info", op) for op in ops[:6]],
            stage="route",
        )
    if et == "budget_started":
        return ParsedStep(
            kind="budget", title="Budget",
            subtitle=f"llm \u2264 {event.get('max_llm_calls')} \u00b7 tools \u2264 {event.get('max_tool_calls')} \u00b7 {int(float(event.get('max_seconds') or 0))}s wall",
            status="info", elapsed=elapsed_f, stage="route",
        )
    if et == "planner_done":
        subtasks = event.get("subtasks") or []
        badges = []
        for s in subtasks[:6]:
            if isinstance(s, dict):
                badges.append(("info", f"{s.get('id') or '?'}:{s.get('specialist') or '?'}"))
        return ParsedStep(
            kind="plan", title=f"planner \u00b7 {len(subtasks)} subtask(s)",
            subtitle=_clip(event.get("rationale") or "", 240),
            status="ok", elapsed=elapsed_f,
            badges=badges, stage="compute",
        )
    if et == "specialist_start":
        return ParsedStep(
            kind="info",
            title=f"specialist \u00b7 {event.get('subtask_id') or '?'} \u00b7 {event.get('kind') or '-'}",
            subtitle=_clip(event.get("instruction") or "", 240),
            status="run", elapsed=elapsed_f, stage="compute",
        )
    if et == "specialist_done":
        ok = bool(event.get("succeeded"))
        return ParsedStep(
            kind="info",
            title=f"specialist done \u00b7 {event.get('subtask_id') or '?'}",
            subtitle=_clip(event.get("summary") or "", 240),
            status="ok" if ok else "err", elapsed=elapsed_f, stage="compute",
        )
    if et == "synthesizer_done":
        cols = event.get("columns") or []
        return ParsedStep(
            kind="answer",
            title=f"synthesizer \u00b7 {event.get('row_count') or 0} row(s)",
            subtitle=f"columns: {', '.join(cols)}",
            status="ok", elapsed=elapsed_f, stage="compute",
        )
    if et == "codegen_program":
        return ParsedStep(
            kind="code",
            title=f"codegen \u00b7 {event.get('label') or 'operator'}",
            subtitle=_clip(event.get("manifest_summary") or "", 220),
            code=str(event.get("program") or ""), code_lang="python",
            status="info", elapsed=elapsed_f, stage="compute",
        )
    if et == "codegen_executed":
        ok = bool(event.get("succeeded"))
        shape = event.get("shape") or []
        sub = f"shape={tuple(shape)}" if ok and shape else (_clip(event.get("failure_reason") or "", 240) if not ok else "answer ready")
        return ParsedStep(
            kind="obs",
            title="execution ok" if ok else "execution failed",
            subtitle=sub,
            status="ok" if ok else "err",
            elapsed=elapsed_f, stage="compute",
        )
    if et == "codegen_debug":
        debug = event.get("debug") or {}
        keys = list(debug.keys())[:6] if isinstance(debug, dict) else []
        try:
            preview = json.dumps(debug, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            preview = str(debug)
        return ParsedStep(
            kind="obs", title="codegen debug",
            subtitle=", ".join(keys),
            observation=_clip(preview, 800),
            status="info", elapsed=elapsed_f, stage="compute",
        )
    if et == "structured_extract_start":
        files = event.get("files") or []
        return ParsedStep(
            kind="extract", title="record extract \u00b7 start",
            subtitle=_clip(", ".join(str(f) for f in files), 240),
            status="run", elapsed=elapsed_f, stage="inspect",
        )
    if et == "structured_extract_done":
        synth = event.get("synthesized_csvs") or {}
        return ParsedStep(
            kind="extract", title=f"record extract \u00b7 {len(synth)} file(s)",
            subtitle=_clip(", ".join(f"{k}\u2192{v}" for k, v in synth.items()), 240),
            status="ok" if synth else "warn", elapsed=elapsed_f, stage="inspect",
        )
    if et == "reflection_start":
        return ParsedStep(
            kind="reflect", title="reflection \u00b7 checking plan vs action",
            status="run", elapsed=elapsed_f, stage="verify",
        )
    if et == "reflection_done":
        verdict = event.get("verdict") or "-"
        issues = "; ".join(event.get("issues") or []) or "no concrete issue"
        revision = event.get("revision_instruction") or ""
        sub = issues
        if revision:
            sub += " \u00b7 retry: " + _clip(revision, 160)
        return ParsedStep(
            kind="reflect", title=f"reflection \u00b7 {verdict}",
            subtitle=_clip(sub, 320),
            status="ok" if verdict == "accept" else "warn",
            elapsed=elapsed_f,
            badges=[("info", f"confidence {event.get('confidence') or '-'}")],
            stage="verify",
        )
    if et == "reasoner_repair_start":
        return ParsedStep(
            kind="reflect", title="reasoner repair",
            subtitle=_clip(event.get("failure_reason") or "", 240),
            status="run", elapsed=elapsed_f, stage="verify",
        )
    if et == "reasoner_repair_done":
        ok = bool(event.get("succeeded"))
        return ParsedStep(
            kind="reflect", title="reasoner repair \u00b7 ok" if ok else "reasoner repair \u00b7 failed",
            subtitle=_clip(event.get("failure_reason") or "", 240),
            status="ok" if ok else "err", elapsed=elapsed_f, stage="verify",
        )
    if et == "cross_verify_start":
        names = event.get("verifier_names") or []
        return ParsedStep(
            kind="verify", title="cross-verify start",
            subtitle=", ".join(names),
            status="run", elapsed=elapsed_f, stage="verify",
        )
    if et == "cross_verify_done":
        outcome = event.get("outcome") or "-"
        return ParsedStep(
            kind="verify", title=f"cross-verify \u00b7 {outcome}",
            subtitle=f"kept {event.get('kept') or 0}/{event.get('total') or 0} columns",
            status="ok" if "agreement" in outcome else "warn",
            elapsed=elapsed_f, stage="verify",
        )
    if et == "score":
        try:
            score = float(event.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        recall = _coerce_float(event.get("recall")) or 0.0
        penalty = _coerce_float(event.get("penalty")) or 0.0
        return ParsedStep(
            kind="score", title=f"local score = {score:.3f}",
            subtitle=f"recall={recall:.3f} \u00b7 penalty={penalty:.3f}",
            status="ok" if score >= 0.95 else ("warn" if score >= 0.5 else "err"),
            elapsed=elapsed_f, stage="score",
        )
    if et == "react_step":
        return _parse_react_action(
            action=str(event.get("action") or ""),
            action_input=event.get("action_input") or {},
            step_index=event.get("step_index"),
            ok=bool(event.get("ok", True)),
            cached=bool(event.get("cached")),
            elapsed=elapsed_f,
            prior_answer_count=prior_answer_count,
        )
    return None


# ============================================================================
# Render a ParsedStep into HTML card
# ============================================================================


def _render_step_html(step: ParsedStep) -> str:
    title_text = step.title
    if step.step_index is not None:
        title_text = f"#{step.step_index} \u00b7 {title_text}"

    meta_bits: list[str] = []
    if step.elapsed is not None:
        try:
            meta_bits.append(f"{float(step.elapsed):.1f}s")
        except (TypeError, ValueError):
            pass

    head = (
        "<div class='sr-head'>"
        f"<span class='sr-tag'>{escape(step.tag)}</span>"
        f"<span class='sr-title'>{escape(title_text)}</span>"
        + (f"<span class='sr-meta'>{escape(_DOT.join(meta_bits))}</span>" if meta_bits else "")
        + "</div>"
    )

    body_bits: list[str] = []
    if step.thought:
        body_bits.append(f"<div class='sr-thought'>{escape(step.thought)}</div>")
    if step.subtitle:
        body_bits.append(f"<div class='sr-subtitle'>{escape(step.subtitle)}</div>")
    if step.code:
        code_preview = step.code if len(step.code) <= 2400 else step.code[:2400] + "\n\u2026(truncated)\u2026"
        cls = "sr-code" + (" sql" if step.code_lang == "sql" else "")
        body_bits.append(f"<pre class='{cls}'>{escape(code_preview)}</pre>")
    if step.observation:
        body_bits.append(f"<pre class='sr-obs'>{escape(step.observation)}</pre>")
    if step.badges:
        chips = "".join(
            f"<span class='sr-badge {escape(str(cls))}'>{escape(str(text))}</span>"
            for cls, text in step.badges
        )
        body_bits.append(f"<div class='sr-badges'>{chips}</div>")

    return f"<div class='sr {escape(step.status)}'>{head}{''.join(body_bits)}</div>"


# ============================================================================
# _StreamView — the unified live/replay view (hero + rail + stream + sidebar)
# ============================================================================


class _StreamView:
    """Streamlit view: hero + stage rail + parsed-step stream + side stats.

    Used identically for live runs (events arrive one-by-one) and replay
    (all ParsedSteps are injected at once). This is the single UI language
    for both modes.
    """

    def __init__(self, *, task_id: str, mode: str = "live") -> None:
        self.task_id = task_id
        self.mode = mode  # "live" | "replay"
        self.start_time = time.time()
        self.stages: dict[str, dict[str, str]] = _initial_stage_state()
        self.stages["route"]["status"] = "run"
        self.stages["route"]["label"] = "starting"
        self.stages["route"]["detail"] = "booting agent runtime"
        self.steps: list[ParsedStep] = []
        self.task_meta: dict[str, Any] = {
            "task_id": task_id, "question": "", "difficulty": "",
        }
        self.complete: bool = False
        self.succeeded: bool | None = None
        self.failure_reason: str | None = None
        self.answer_count: int = 0
        self.event_count: int = 0
        self.tool_call_count: int = 0
        self.code_call_count: int = 0
        self.current_stage: str | None = "route"
        self.score: dict[str, Any] | None = None
        self.latest_answer: dict[str, Any] | None = None
        self.context_files: list[str] = []
        self.raw_lines: list[str] = []

        # Layout: hero + rail at top; left column stream, right column stats.
        self.hero_ph = st.empty()
        self.rail_ph = st.empty()
        left_col, right_col = st.columns([1.7, 1.0])
        with left_col:
            st.markdown("<div class='stream-title'>Live Parsing Stream \u00b7 newest first</div>", unsafe_allow_html=True)
            self.stream_ph = st.empty()
        with right_col:
            self.stats_ph = st.empty()
            self.files_ph = st.empty()
            self.answer_ph = st.empty()
            self.score_ph = st.empty()
        with st.expander("Raw agent log (rich console output)", expanded=False):
            self.log_ph = st.empty()

        self._render_all()

    # --- public ingestion (live mode) ---

    def apply_event(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        self.event_count += 1
        et = event.get("type")

        # Sticky bookkeeping
        if et == "task_start":
            self.task_meta.update(
                task_id=event.get("task_id") or self.task_id,
                question=event.get("question") or "",
                difficulty=event.get("difficulty") or "",
            )
        elif et == "task_end":
            self.complete = True
            self.succeeded = bool(event.get("succeeded"))
            self.failure_reason = event.get("failure_reason")
        elif et == "score":
            self.score = {
                "score": _coerce_float(event.get("score")),
                "recall": _coerce_float(event.get("recall")),
                "penalty": _coerce_float(event.get("penalty")),
            }
        elif et == "react_step":
            action = str(event.get("action") or "")
            ai = event.get("action_input") if isinstance(event.get("action_input"), dict) else {}
            if action in _REACT_ACT_ACTIONS:
                self.code_call_count += 1
            elif action in _REACT_OBSERVE_ACTIONS:
                self.tool_call_count += 1
            if action == "answer" and isinstance(ai, dict):
                cols = ai.get("columns") or []
                rows = ai.get("rows") or []
                self.latest_answer = {
                    "columns": list(cols),
                    "rows": [list(r) for r in rows][:8],
                    "row_count": len(rows),
                    "is_draft": (self.answer_count == 0),
                }

        # Stage rail update
        if et == "react_step":
            stage = _stage_for_react_step(str(event.get("action") or ""), self.answer_count)
        else:
            stage = _STAGE_BY_EVENT.get(et or "")
        if stage:
            self._advance_stage_to(stage)
            self._update_stage_from_event(stage, event)

        # Parse event into a card
        step = _parse_event(event, prior_answer_count=self.answer_count)
        if step is not None:
            self.steps.append(step)

        # Increment answer counter AFTER parsing
        if et == "react_step" and str(event.get("action") or "") == "answer":
            self.answer_count += 1

        if et == "task_end":
            self._finalize_remaining_stages()

        self._render_all()

    def append_log_line(self, line: str) -> None:
        self.raw_lines.append(line)
        plain = _ANSI_RE.sub("", "".join(self.raw_lines[-180:])).strip() or "Booting agent runtime..."
        self.log_ph.code(plain, language="text")

    # --- stage rail logic ---

    def _advance_stage_to(self, target: str) -> None:
        """Walk the rail to ``target`` without faking activity.

        Stages strictly before ``target`` only roll up to ``ok`` if they
        were genuinely ``run`` (active); ``idle`` stages stay ``idle`` so
        the user can see which legs of the route were actually walked.
        """
        try:
            target_idx = _STAGE_ORDER.index(target)
        except ValueError:
            return
        for idx, name in enumerate(_STAGE_ORDER):
            if idx < target_idx and self.stages[name]["status"] == "run":
                self.stages[name]["status"] = "ok"
            elif idx == target_idx and self.stages[name]["status"] == "idle":
                self.stages[name]["status"] = "run"
        self.current_stage = target

    def _update_stage_from_event(self, stage: str, event: dict[str, Any]) -> None:
        et = event.get("type")
        slot = self.stages[stage]
        prior_status = slot.get("status")

        if et == "router_decision":
            slot["label"] = event.get("route_name") or "-"
            slot["detail"] = f"kind={event.get('kind') or '-'} \u00b7 model={event.get('model') or '-'}"
        elif et == "router_cascade":
            slot["status"] = "warn"
            slot["detail"] = f"\u2192 {event.get('to_route') or '?'}"
        elif et == "task_compiled":
            slot["label"] = event.get("task_type") or "-"
            ops = ", ".join(event.get("operations") or [])
            slot["detail"] = f"ops={ops or '-'} \u00b7 sources={event.get('data_source_count') or 0}"
            if prior_status == "run":
                slot["status"] = "ok"
        elif et == "planner_done":
            count = len(event.get("subtasks") or [])
            slot["label"] = f"{count} subtask(s)"
            slot["detail"] = _clip(event.get("rationale") or "", 140)
            slot["status"] = "ok"
        elif et == "codegen_program":
            slot["label"] = event.get("label") or "codegen"
            slot["detail"] = f"{event.get('program_chars') or 0} chars"
        elif et == "codegen_executed":
            ok = bool(event.get("succeeded"))
            slot["status"] = "ok" if ok else "err"
            shape = event.get("shape") or []
            slot["label"] = "ok" if ok else "failed"
            slot["detail"] = (f"shape={tuple(shape)}" if shape else "answer ready") if ok else _clip(event.get("failure_reason") or "", 140)
        elif et == "codegen_debug":
            debug = event.get("debug") or {}
            count = len(debug) if isinstance(debug, dict) else 0
            slot["label"] = f"{count} key(s)"
            slot["detail"] = ", ".join(list(debug.keys())[:5]) if isinstance(debug, dict) else ""
            slot["status"] = "ok"
        elif et == "reflection_start":
            slot["label"] = "running"
            slot["detail"] = "checking plan vs action"
        elif et == "reflection_done":
            verdict = event.get("verdict") or "-"
            slot["label"] = verdict
            slot["detail"] = "; ".join(event.get("issues") or []) or "no concrete issue"
            slot["status"] = "ok" if verdict == "accept" else "warn"
        elif et == "react_step":
            action = str(event.get("action") or "")
            step_index = event.get("step_index")
            ai = event.get("action_input") if isinstance(event.get("action_input"), dict) else {}
            ok = bool(event.get("ok", True))
            cached = bool(event.get("cached"))
            tag = "cached " if cached else ""
            allow_update = prior_status in {"idle", "run", "warn"}
            if action in _REACT_OBSERVE_ACTIONS:
                hint = str(ai.get("path") or "") or (f"depth={ai.get('max_depth')}" if ai.get("max_depth") is not None else "")
                slot["label"] = f"{tag}{action}"
                slot["detail"] = f"step #{step_index} \u00b7 {hint}" if hint else f"step #{step_index}"
                if allow_update:
                    slot["status"] = "run" if ok else "warn"
            elif action in _REACT_ACT_ACTIONS:
                hint = _clip(str(ai.get("code") or ai.get("sql") or ""), 100)
                slot["label"] = action
                slot["detail"] = f"step #{step_index} \u00b7 {hint}" if hint else f"step #{step_index}"
                if allow_update:
                    slot["status"] = "run" if ok else "warn"
            elif action == "answer":
                cols = ai.get("columns") if isinstance(ai, dict) else None
                rows = ai.get("rows") if isinstance(ai, dict) else None
                c = len(cols) if isinstance(cols, list) else 0
                r = len(rows) if isinstance(rows, list) else 0
                if stage == "draft":
                    slot["label"] = "draft submitted"
                    slot["detail"] = f"{c} col(s) \u00d7 {r} row(s) \u00b7 self-verify"
                    slot["status"] = "run"
                else:
                    slot["label"] = "final answer"
                    slot["detail"] = f"{c} col(s) \u00d7 {r} row(s) committed"
                    slot["status"] = "ok"
        elif et == "score":
            slot["status"] = "ok"
            slot["label"] = f"{float(event.get('score') or 0):.3f}"
            slot["detail"] = f"recall={float(event.get('recall') or 0):.3f}"
        elif et == "cross_verify_done":
            outcome = event.get("outcome") or "-"
            slot["label"] = outcome
            slot["detail"] = f"kept {event.get('kept') or 0}/{event.get('total') or 0}"
            slot["status"] = "ok" if "agreement" in outcome else "warn"

    def _finalize_remaining_stages(self) -> None:
        for name in _STAGE_ORDER:
            slot = self.stages[name]
            if slot["status"] == "run":
                slot["status"] = "ok" if self.succeeded else "err"
            # idle stages stay idle — honesty about what wasn't walked.

    # --- rendering ---

    def _render_all(self) -> None:
        self._render_hero()
        self._render_rail()
        self._render_stream()
        self._render_stats()
        self._render_files()
        self._render_answer()
        self._render_score()

    def _render_hero(self) -> None:
        if self.complete:
            cls = "complete" if self.succeeded else "failed"
            stage_text = f"Run finished \u00b7 {self.task_meta.get('task_id') or self.task_id}"
            meta = (
                _clip(self.failure_reason or "Answer ready", 280)
                if not self.succeeded else
                f"elapsed {time.time() - self.start_time:.1f}s \u00b7 {self.event_count} event(s)"
            )
        elif self.mode == "replay":
            cls = "replay"
            stage_text = f"Replay \u00b7 {self.task_meta.get('task_id') or self.task_id}"
            meta = f"{len(self.steps)} step card(s)"
        else:
            cls = "idle" if not self.steps else ""
            stage = self.current_stage or "observe"
            stage_text = f"Running \u00b7 {_STAGE_LABEL.get(stage, stage)}"
            meta = (
                f"elapsed {time.time() - self.start_time:.1f}s \u00b7 "
                f"{self.event_count} event(s) \u00b7 "
                f"{self.tool_call_count} tool \u00b7 {self.code_call_count} exec"
            )
        question = _clip(self.task_meta.get("question") or "", 320)
        difficulty = self.task_meta.get("difficulty") or "-"
        task_id = self.task_meta.get("task_id") or self.task_id
        html = (
            f"<div class='theater-hero {cls}'>"
            "<div>"
            f"<div class='th-task'>Task \u00b7 {escape(str(task_id))} \u00b7 difficulty {escape(str(difficulty))}</div>"
            f"<div class='th-question'>{escape(question or '(no question)')}</div>"
            f"<div class='th-meta'>{escape(meta)}</div>"
            "</div>"
            "<div class='th-state'>"
            "<span class='pulse'></span>"
            f"<span>{escape(stage_text)}</span>"
            "</div>"
            "</div>"
        )
        self.hero_ph.markdown(html, unsafe_allow_html=True)

    def _render_rail(self) -> None:
        pills: list[str] = []
        for idx, name in enumerate(_STAGE_ORDER, start=1):
            slot = self.stages[name]
            status = slot["status"]
            cls = {"idle": "idle", "run": "run", "ok": "ok", "warn": "warn", "err": "err"}.get(status, "idle")
            label = slot.get("label") or "\u2014"
            detail = slot.get("detail") or ""
            title = f"{idx:02d} \u00b7 {_STAGE_LABEL[name]}"
            pills.append(
                f"<div class='stage-pill {cls}'>"
                f"<div class='sp-title'>{escape(title)}</div>"
                f"<div class='sp-label'>{escape(str(label))}</div>"
                f"<div class='sp-detail'>{escape(str(detail))}</div>"
                "</div>"
            )
        self.rail_ph.markdown("<div class='stage-rail'>" + "".join(pills) + "</div>", unsafe_allow_html=True)

    def _render_stream(self) -> None:
        if not self.steps:
            self.stream_ph.markdown(
                "<div class='stream empty'>Waiting for the agent's first event\u2026</div>",
                unsafe_allow_html=True,
            )
            return
        cards = [_render_step_html(step) for step in reversed(self.steps[-220:])]
        self.stream_ph.markdown(
            "<div class='stream'>" + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )

    def _render_stats(self) -> None:
        elapsed_s = time.time() - self.start_time if self.mode == "live" else None
        rows: list[tuple[str, str]] = [
            ("status", "complete" if self.complete else ("replay" if self.mode == "replay" else "running")),
            ("stage", _STAGE_LABEL.get(self.current_stage or "observe", "-")),
            ("events", str(self.event_count)),
            ("tool calls", str(self.tool_call_count)),
            ("code/sql runs", str(self.code_call_count)),
            ("answer drafts", str(self.answer_count)),
        ]
        if elapsed_s is not None:
            rows.append(("elapsed", f"{elapsed_s:.1f}s"))
        body = "".join(
            f"<div class='sb-row'><span class='label'>{escape(k)}</span><span class='value'>{escape(v)}</span></div>"
            for k, v in rows
        )
        self.stats_ph.markdown(
            "<div class='side-box'><div class='sb-title'>Run Stats</div>" + body + "</div>",
            unsafe_allow_html=True,
        )

    def _render_files(self) -> None:
        seen: list[str] = []
        seen_set: set[str] = set()
        for step in self.steps:
            if step.kind == "tool" and step.subtitle:
                path = step.subtitle
                if path and path not in seen_set:
                    seen.append(path)
                    seen_set.add(path)
        if not seen:
            self.files_ph.markdown(
                "<div class='side-box empty'><div class='sb-title'>Files Touched</div>"
                "<div class='sb-row'>No files inspected yet.</div></div>",
                unsafe_allow_html=True,
            )
            return
        rows = "".join(f"<div class='sb-file'>{escape(p)}</div>" for p in seen[:24])
        self.files_ph.markdown(
            "<div class='side-box'><div class='sb-title'>Files Touched</div>" + rows + "</div>",
            unsafe_allow_html=True,
        )

    def _render_answer(self) -> None:
        if not self.latest_answer:
            self.answer_ph.markdown("", unsafe_allow_html=True)
            return
        cols = self.latest_answer.get("columns") or []
        rows = self.latest_answer.get("rows") or []
        row_count = self.latest_answer.get("row_count") or len(rows)
        is_draft = bool(self.latest_answer.get("is_draft"))
        title = "Draft Answer (self-verify)" if is_draft else "Final Answer"
        head_html = "".join(f"<th>{escape(str(c))}</th>" for c in cols[:8])
        body_html = ""
        for r in rows[:5]:
            if not isinstance(r, (list, tuple)):
                continue
            body_html += "<tr>" + "".join(f"<td>{escape(str(v))[:80]}</td>" for v in r[:8]) + "</tr>"
        more = f"<div class='sb-row'><span class='label'>showing</span><span class='value'>{min(5, len(rows))}/{row_count}</span></div>" if rows else ""
        self.answer_ph.markdown(
            "<div class='side-box'>"
            f"<div class='sb-title'>{escape(title)}</div>"
            + more
            + f"<table class='sb-answer-table'><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>"
            + "</div>",
            unsafe_allow_html=True,
        )

    def _render_score(self) -> None:
        if not self.score or self.score.get("score") is None:
            self.score_ph.markdown("", unsafe_allow_html=True)
            return
        score = self.score.get("score") or 0.0
        recall = self.score.get("recall") or 0.0
        penalty = self.score.get("penalty") or 0.0
        rows = [
            ("score", f"{score:.3f}"),
            ("recall", f"{recall:.3f}"),
            ("penalty", f"{penalty:.3f}"),
        ]
        body = "".join(
            f"<div class='sb-row'><span class='label'>{escape(k)}</span><span class='value'>{escape(v)}</span></div>"
            for k, v in rows
        )
        self.score_ph.markdown(
            "<div class='side-box'><div class='sb-title'>Local Score</div>" + body + "</div>",
            unsafe_allow_html=True,
        )


# ============================================================================
# Live run: subprocess pipe → _StreamView
# ============================================================================


def _run_task_live(
    *,
    task_id: str,
    config_path: Path,
    run_id: str,
    dataset_root: Path | None = None,
    batch_run_dir: Path | None = None,
    view: "_StreamView | None" = None,
) -> tuple[dict[str, Any], Path]:
    command = [
        sys.executable, "-u", "-c", _SUBPROCESS_RUNNER,
        task_id, str(config_path), run_id,
        str(dataset_root) if dataset_root else "",
    ]
    env = None
    if batch_run_dir is not None:
        import os as _os
        env = {**_os.environ, "DEMO_BATCH_RUN_DIR": str(batch_run_dir)}
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    if view is None:
        view = _StreamView(task_id=task_id, mode="live")
    result_payload: dict[str, Any] | None = None

    assert process.stdout is not None
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
                view.append_log_line(line)
                continue
            view.apply_event(event)
            time.sleep(0)
            continue
        view.append_log_line(line)

    return_code = process.wait()
    view.append_log_line("")

    if return_code != 0:
        raise RuntimeError(f"subprocess exited with code {return_code}")
    if result_payload is None:
        raise RuntimeError("run finished but did not return artifact metadata")

    trace_path = Path(str(result_payload["trace_path"]))
    return result_payload, trace_path


def _run_batch(
    *,
    task_ids: list[str],
    config_path: Path,
    dataset_root: Path | None,
) -> dict[str, Any]:
    """Run a list of tasks sequentially under one shared batch run directory.

    Each task gets its own _StreamView container so the theater pans down the
    page as the batch progresses, while a sticky progress banner shows
    cumulative i/N · OK/FAIL counts above the theater.
    """
    total = len(task_ids)
    base_run_id = "batch-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_run_dir = PROJECT_ROOT / "artifacts" / "runs" / base_run_id
    batch_run_dir.mkdir(parents=True, exist_ok=True)

    progress_ph = st.empty()
    summary_ph = st.empty()

    ok_count = 0
    fail_count = 0
    outcomes: list[dict[str, Any]] = []
    last_trace: Path | None = None

    def _render_progress(current: str, done_idx: int, *, finished: bool = False) -> None:
        label = "Batch complete" if finished else "Batch progress"
        cur = "-" if not current else current
        progress_ph.markdown(
            "<div class='batch-progress'>"
            f"<b>{escape(label)}</b> {done_idx}/{total} · "
            f"current <code>{escape(cur)}</code> · "
            f"OK <b>{ok_count}</b> · FAIL <b>{fail_count}</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    _render_progress(task_ids[0] if task_ids else "", 0)

    for i, tid in enumerate(task_ids, start=1):
        _render_progress(tid, i - 1)
        st.markdown(
            f"<div class='batch-task-header'>Task {i} / {total} · "
            f"<code>{escape(tid)}</code></div>",
            unsafe_allow_html=True,
        )
        slot = st.container()
        with slot:
            view = _StreamView(task_id=tid, mode="live")
        try:
            payload, trace_path = _run_task_live(
                task_id=tid,
                config_path=config_path,
                run_id=f"{base_run_id}-{tid}",
                dataset_root=dataset_root,
                batch_run_dir=batch_run_dir,
                view=view,
            )
            succeeded = bool(payload.get("succeeded"))
            outcomes.append({
                "task_id": tid,
                "ok": succeeded,
                "trace_path": str(trace_path),
            })
            last_trace = trace_path
            if succeeded:
                ok_count += 1
            else:
                fail_count += 1
        except Exception as exc:  # noqa: BLE001
            outcomes.append({"task_id": tid, "ok": False, "error": str(exc)})
            fail_count += 1
            slot.error(f"Task {tid} failed: {exc}")
        _render_progress(tid, i)

    _render_progress(task_ids[-1] if task_ids else "", total, finished=True)
    summary_ph.success(
        f"Batch saved → {batch_run_dir} · OK {ok_count} / FAIL {fail_count}"
    )

    if last_trace is not None:
        st.session_state["dashboard_scope"] = str(batch_run_dir)

    return {
        "batch_run_dir": str(batch_run_dir),
        "ok": ok_count,
        "fail": fail_count,
        "outcomes": outcomes,
    }


# ============================================================================
# Replay: trace.json → ParsedStep list → _StreamView
# ============================================================================


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


def _trace_to_parsed_steps(trace: dict[str, Any]) -> list[ParsedStep]:
    """Convert a full trace.json into a list of ParsedSteps for replay."""
    steps: list[ParsedStep] = []

    task_id = trace.get("task_id") or "-"
    succeeded = bool(trace.get("succeeded"))
    elapsed_total = _coerce_float(trace.get("e2e_elapsed_seconds"))

    decision = trace.get("router_decision") or {}
    compiled = trace.get("compiled_task") or decision.get("compiled_task") or {}

    # 1. Task start
    steps.append(ParsedStep(
        kind="task",
        title=f"Task {task_id} \u00b7 {trace.get('agent_mode') or 'agent'}",
        subtitle="(question not stored in trace.json \u2014 see history card)",
        status="run", stage="route",
        badges=[("info", f"agent {trace.get('agent_mode') or '-'}")],
    ))

    # 2. Router decision
    if decision:
        steps.append(ParsedStep(
            kind="route",
            title=f"route \u2192 {decision.get('route_name') or '?'}",
            subtitle=f"kind={decision.get('kind') or '-'} \u00b7 model={decision.get('model') or '-'}",
            status="info", stage="route",
            badges=[("info", f"difficulty {decision.get('difficulty') or '-'}")],
        ))

    # 3. Compiled task
    if compiled:
        ops = compiled.get("operations") or []
        steps.append(ParsedStep(
            kind="compile",
            title=f"task type \u2192 {compiled.get('task_type') or '-'}",
            subtitle=f"answer={compiled.get('answer_type') or '-'} \u00b7 {len(compiled.get('source_capabilities') or [])} source(s)",
            badges=[("info", op) for op in ops[:6]],
            status="info", stage="route",
        ))

    # 4. Budget snapshot
    budget = trace.get("budget") or {}
    if budget:
        steps.append(ParsedStep(
            kind="budget", title="Budget consumed",
            subtitle=(
                f"llm={budget.get('llm_calls') or 0} \u00b7 tools={budget.get('tool_calls') or 0} \u00b7 "
                f"repairs={budget.get('local_repairs') or 0} \u00b7 reasoner={budget.get('reasoner_repairs') or 0}"
            ),
            status="info", stage="route",
        ))

    # 5. Planner subtasks
    agentic = _agentic_trace(trace)
    planner = (agentic.get("planner") or {}).get("plan") if isinstance(agentic.get("planner"), dict) else None
    if planner:
        subtasks = planner.get("subtasks") or []
        steps.append(ParsedStep(
            kind="plan", title=f"planner \u00b7 {len(subtasks)} subtask(s)",
            subtitle=_clip(planner.get("rationale") or "", 240),
            status="ok", stage="compute",
            badges=[("info", f"{s.get('id')}:{s.get('specialist')}") for s in subtasks[:6] if isinstance(s, dict)],
        ))

    # 6. ReAct step-by-step replay
    react_steps = trace.get("steps") or []
    answer_count = 0
    for raw in react_steps:
        if not isinstance(raw, dict):
            continue
        action = str(raw.get("action") or "")
        ps = _parse_react_action(
            action=action,
            action_input=raw.get("action_input") or {},
            step_index=raw.get("step_index"),
            ok=bool(raw.get("ok", True)),
            cached=False,
            elapsed=None,
            prior_answer_count=answer_count,
            thought=_clip(raw.get("thought") or "", 400),
            observation=raw.get("observation"),
        )
        steps.append(ps)
        if action == "answer":
            answer_count += 1

    # 7. Codegen operator (non-React routes)
    operator = _get_operator_block(trace)
    if operator.get("program") and not react_steps:
        steps.append(ParsedStep(
            kind="code",
            title="codegen operator \u00b7 program",
            subtitle=_clip(", ".join(item.get("path") or "" for item in _iter_manifest(trace) if isinstance(item, dict))[:300], 240),
            code=str(operator.get("program") or ""), code_lang="python",
            status="info", stage="compute",
        ))
        if operator.get("succeeded") is not None:
            ok = bool(operator.get("succeeded"))
            steps.append(ParsedStep(
                kind="obs",
                title="execution ok" if ok else "execution failed",
                subtitle=_clip(operator.get("failure_reason") or "", 240),
                status="ok" if ok else "err", stage="compute",
            ))
        stdout_tail = str(operator.get("exec_stdout") or "")[-2400:]
        if stdout_tail:
            steps.append(ParsedStep(
                kind="obs", title="stdout tail",
                observation=stdout_tail, status="info", stage="compute",
            ))

    # 8. Reflection rounds
    for rd in agentic.get("reflection_rounds") or []:
        if not isinstance(rd, dict):
            continue
        decision_obj = rd.get("decision") or {}
        verdict = decision_obj.get("verdict") or "-"
        issues = "; ".join(decision_obj.get("issues") or []) or "no concrete issue"
        revision = decision_obj.get("revision_instruction") or ""
        sub = issues + (f" \u00b7 retry: {_clip(revision, 200)}" if revision else "")
        steps.append(ParsedStep(
            kind="reflect",
            title=f"reflection round {rd.get('round', 0)} \u00b7 {verdict}",
            subtitle=_clip(sub, 320),
            status="ok" if verdict == "accept" else "warn",
            badges=[("info", f"confidence {decision_obj.get('confidence') or '-'}")],
            stage="verify",
        ))

    # 9. Cross-verify / semantic consistency
    audit = trace.get("semantic_consistency_audit") or {}
    if audit:
        gate = audit.get("final_gate") or "-"
        steps.append(ParsedStep(
            kind="verify", title=f"semantic gate \u00b7 {gate}",
            subtitle=f"judge_attempts={audit.get('judge_attempts') or 0} \u00b7 plan_ran={audit.get('plan_ran')}",
            status="ok" if gate in {"plan+judge", "escalated+judge", "ok", "pass"} else "warn",
            stage="verify",
        ))
    cmv = trace.get("cross_model_verify") or {}
    if cmv.get("applied"):
        outcome = cmv.get("outcome") or "-"
        verifiers = cmv.get("verifiers") or []
        steps.append(ParsedStep(
            kind="verify", title=f"cross-model verify \u00b7 {outcome}",
            subtitle=f"verifiers: {', '.join(verifiers)}",
            status="ok" if "agreement" in str(outcome) else "warn",
            stage="verify",
        ))

    # 10. Local score
    local_score = trace.get("local_score") or {}
    if local_score.get("score") is not None:
        score = float(local_score["score"])
        recall = _coerce_float(local_score.get("recall")) or 0.0
        penalty = _coerce_float(local_score.get("penalty")) or 0.0
        steps.append(ParsedStep(
            kind="score", title=f"local score = {score:.3f}",
            subtitle=f"recall={recall:.3f} \u00b7 penalty={penalty:.3f}",
            status="ok" if score >= 0.95 else ("warn" if score >= 0.5 else "err"),
            stage="score",
        ))

    # 11. Final answer card
    answer = trace.get("answer")
    if isinstance(answer, dict) and (answer.get("columns") or answer.get("rows")):
        cols = answer.get("columns") or []
        rows = answer.get("rows") or []
        steps.append(ParsedStep(
            kind="answer",
            title=f"final answer \u00b7 {len(cols)} col(s) \u00d7 {len(rows)} row(s)",
            subtitle=", ".join(str(c) for c in cols[:8]),
            badges=[("info", str(c)[:24]) for c in cols[:6]],
            status="ok" if succeeded else "warn", stage="verify",
        ))

    # 12. Task end
    steps.append(ParsedStep(
        kind="task",
        title="Run completed" if succeeded else "Run failed",
        subtitle=_clip(trace.get("failure_reason") or "", 300) if not succeeded else (f"elapsed {elapsed_total:.1f}s" if elapsed_total else "answer ready"),
        status="ok" if succeeded else "err",
        stage="verify" if succeeded else "compute",
    ))
    return steps


def _render_replay_view(trace: dict[str, Any], trace_path: Path, *, question: str = "") -> None:
    """Render a saved trace.json using the same _StreamView used for live runs."""
    task_id = str(trace.get("task_id") or trace_path.parent.name)
    view = _StreamView(task_id=task_id, mode="replay")
    view.task_meta["question"] = question or ""
    decision = trace.get("router_decision") or {}
    view.task_meta["difficulty"] = (
        (trace.get("compiled_task") or decision.get("compiled_task") or {}).get("difficulty") or "-"
    )

    parsed_steps = _trace_to_parsed_steps(trace)
    view.steps = parsed_steps

    # Sticky counters
    view.tool_call_count = sum(1 for s in parsed_steps if s.kind == "tool")
    view.code_call_count = sum(1 for s in parsed_steps if s.kind in {"code", "sql"})
    view.answer_count = sum(1 for s in parsed_steps if s.kind in {"draft", "answer"})
    view.event_count = len(parsed_steps)
    local = trace.get("local_score") or {}
    if local.get("score") is not None:
        view.score = {
            "score": _coerce_float(local.get("score")),
            "recall": _coerce_float(local.get("recall")),
            "penalty": _coerce_float(local.get("penalty")),
        }
    answer = trace.get("answer")
    if isinstance(answer, dict) and (answer.get("columns") or answer.get("rows")):
        view.latest_answer = {
            "columns": list(answer.get("columns") or []),
            "rows": [list(r) for r in (answer.get("rows") or [])[:8]],
            "row_count": len(answer.get("rows") or []),
            "is_draft": False,
        }
    view.complete = True
    view.succeeded = bool(trace.get("succeeded"))
    view.failure_reason = trace.get("failure_reason")
    # Mark stages from replayed steps
    for s in parsed_steps:
        if not s.stage:
            continue
        slot = view.stages.get(s.stage)
        if not slot:
            continue
        if s.status in {"ok", "warn", "err"}:
            slot["status"] = s.status
        elif slot["status"] == "idle":
            slot["status"] = "ok"
        if s.title:
            slot["label"] = _clip(s.title, 40)
        if s.subtitle and not slot.get("detail"):
            slot["detail"] = _clip(s.subtitle, 80)
    view.current_stage = "verify" if view.succeeded else "execute"

    view._render_all()
    with st.expander("Raw trace.json", expanded=False):
        st.caption(str(trace_path))
        st.json(trace)


# ============================================================================
# History grid — _render_run_dashboard (bottom section, always visible)
# ============================================================================


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
        trace_path=trace_path, trace=trace,
        app_config=app_config, score_summary=score_summary,
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
                "Score": None, "Recall": None, "Penalty": None,
                "Matched": "", "Difficulty": "-", "Route": "-",
                "Agent": "-", "Task Type": "-", "Gate": "-",
                "Reflect": "-", "Answer": "0x0", "Elapsed": None,
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
                trace_path=trace_path, trace=trace,
                app_config=app_config, dataset=dataset,
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
    score_sub = " \u00b7 ".join(score_sub_parts) or "no score"

    badges_html: list[str] = []
    if difficulty and difficulty != "-":
        diff_cls = f"difficulty-{difficulty}" if difficulty in {"easy", "medium", "hard"} else ""
        badges_html.append(f"<span class='rc-badge {diff_cls}'>{escape(difficulty)}</span>")
    if route and route != "-":
        badges_html.append(f"<span class='rc-badge route'>{escape(route)}</span>")
    if agent and agent not in {"-", route}:
        badges_html.append(f"<span class='rc-badge'>{escape(agent)}</span>")
    if gate and gate != "-":
        gate_cls = "gate-pass" if gate in {"ok", "pass", "accept"} else (
            "gate-fail" if gate in {"failed", "fail", "error"} else ""
        )
        badges_html.append(f"<span class='rc-badge {gate_cls}'>gate {escape(gate)}</span>")
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
        f"<span>{escape(run_name)} \u00b7 {escape(elapsed_text)}</span>"
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
    """Render the history grid with filters, sort, and replay button."""
    records = _build_run_records(scope=scope, app_config=app_config, dataset=dataset)
    st.markdown(
        "<div class='dashboard-title'>"
        "<h2>Run History</h2>"
        f"<div class='path'>{escape(_relative_artifact_path(scope))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if not records:
        st.info("No trace.json files found for this scope yet \u2014 run a task first.")
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
            filtered["Task"].astype(str) + " " +
            filtered["Question"].astype(str) + " " +
            filtered["Failure / Score Error"].astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(needle, regex=False)]

    _render_dashboard_cards(records, len(filtered))

    if filtered.empty:
        st.info("No records match the current filters.")
        return

    sort_col1, sort_col2 = st.columns([1.0, 0.4])
    sort_key = sort_col1.selectbox(
        "Sort by",
        ["Most recent", "Score (low \u2192 high)", "Score (high \u2192 low)", "Status (FAIL first)", "Elapsed (slow \u2192 fast)"],
        key="dashboard_sort",
    )
    page_size = sort_col2.selectbox("Per page", [12, 24, 48, 96, 240], index=1, key="dashboard_page_size")

    sorted_df = filtered.copy()
    if sort_key == "Most recent":
        sorted_df = sorted_df.sort_values("_modified_ts", ascending=False)
    elif sort_key == "Score (low \u2192 high)":
        sorted_df = sorted_df.sort_values("Score", ascending=True, na_position="first")
    elif sort_key == "Score (high \u2192 low)":
        sorted_df = sorted_df.sort_values("Score", ascending=False, na_position="last")
    elif sort_key == "Status (FAIL first)":
        sorted_df = sorted_df.assign(_fail_first=(sorted_df["Status"] != "OK").astype(int))
        sorted_df = sorted_df.sort_values(["_fail_first", "_modified_ts"], ascending=[False, False])
        sorted_df = sorted_df.drop(columns=["_fail_first"])
    elif sort_key == "Elapsed (slow \u2192 fast)":
        sorted_df = sorted_df.sort_values("Elapsed", ascending=False, na_position="last")

    visible_df = sorted_df.head(int(page_size))
    if len(sorted_df) > len(visible_df):
        st.caption(f"Showing {len(visible_df)} of {len(sorted_df)} filtered records. Increase 'Per page' to see more.")

    # Each card is wrapped in an anchor with `?replay=<trace_path>` so the
    # whole card is clickable. main() consumes the query param at the top
    # of the run and routes it into st.session_state["replay_trace_path"].
    import urllib.parse as _ulib
    parts: list[str] = []
    for row in visible_df.to_dict(orient="records"):
        trace_path = str(row.get("_trace_path") or "")
        card_html = _record_card_html(row)
        if trace_path:
            href = "?replay=" + _ulib.quote(trace_path, safe="")
            parts.append(
                f"<a class='run-card-link' href='{href}' target='_self'>{card_html}</a>"
            )
        else:
            parts.append(card_html)
    st.markdown(f"<div class='run-grid'>{''.join(parts)}</div>", unsafe_allow_html=True)



# ============================================================================
# Main — single-page flow: stream at top, history at bottom, no tabs
# ============================================================================


def main() -> None:
    st.set_page_config(
        page_title="DABench Demo \u00b7 Theater",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_APP_CSS, unsafe_allow_html=True)

    # ---- Clickable history cards: consume ?replay= from the URL into
    # session state. Anchor links inside the dashboard grid set this
    # query param so the whole card behaves like a button.
    qp = st.query_params
    qp_replay = qp.get("replay")
    if qp_replay:
        st.session_state["replay_trace_path"] = str(qp_replay)
        try:
            del st.query_params["replay"]
        except Exception:  # noqa: BLE001
            pass

    # ---- Top control strip (replaces left sidebar) ----
    st.markdown(
        "<div class='app-topbar'>"
        "<div class='tb-title'>Data Agent Theater</div>"
        "<div class='tb-sub'>One-click runs \u00b7 live trace stream \u00b7 replayable history</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    settings_tab, single_tab, batch_tab = st.tabs([
        "\u2699 Settings", "\u25b6 Single Task", "\u23f5 Batch (Run All)",
    ])

    with settings_tab:
        cfg_col, root_col = st.columns([1.0, 1.0])
        config_text = cfg_col.text_input(
            "Config",
            value=str(DEFAULT_CONFIG),
            key="topbar_config",
        )
        # Resolve config first so the Tasks-folder input can default to the
        # inherited dataset root.
        config_path_preview = Path(config_text).expanduser()
        if not config_path_preview.is_absolute():
            config_path_preview = (PROJECT_ROOT / config_path_preview).resolve()
        default_tasks_root = ""
        try:
            _preview_config = load_app_config(config_path_preview)
            default_tasks_root = str(_preview_config.dataset.root_path)
        except Exception:  # noqa: BLE001
            pass
        tasks_root_text = root_col.text_input(
            "Tasks folder (overrides config dataset root)",
            value=default_tasks_root,
            key="sidebar_tasks_root",
            help="A folder that contains task_* subdirectories.",
        )
        run_dir_text = st.text_input(
            "History run directory",
            value=str(PROJECT_ROOT / "artifacts" / "runs"),
            key="sidebar_run_dir",
        )
        if st.button("Reset View"):
            for key in ("replay_trace_path", "dashboard_scope"):
                st.session_state.pop(key, None)
            st.rerun()

    with single_tab:
        single_id_col, single_btn_col = st.columns([3.0, 1.0])
        task_id = single_id_col.text_input(
            "Task ID", value="", key="sidebar_task_id"
        )
        single_btn_col.markdown("<div style='height: 1.7rem'></div>", unsafe_allow_html=True)
        run_clicked = single_btn_col.button(
            "Run Agent", type="primary", use_container_width=True,
            key="topbar_run_single",
        )

    with batch_tab:
        diff_col, limit_col, batch_btn_col = st.columns([2.0, 1.0, 1.0])
        difficulty_options = ["easy", "medium", "hard"]
        difficulty_pick = diff_col.multiselect(
            "Difficulty filter",
            difficulty_options,
            default=[],
            help="Empty = include all difficulties.",
            key="sidebar_batch_difficulty",
        )
        batch_limit = limit_col.number_input(
            "Limit (0 = no cap)",
            min_value=0, max_value=10000, value=0, step=1,
            key="sidebar_batch_limit",
        )
        batch_btn_col.markdown("<div style='height: 1.7rem'></div>", unsafe_allow_html=True)
        run_all_clicked = batch_btn_col.button(
            "Run All Tasks", type="primary", use_container_width=True,
            key="sidebar_run_all",
        )

    config_path = Path(config_text).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    try:
        app_config = load_app_config(config_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load config: {exc}")
        return

    # Resolve effective dataset root: sidebar override wins if non-empty.
    effective_root = app_config.dataset.root_path
    dataset_root_override: Path | None = None
    if tasks_root_text.strip():
        candidate = Path(tasks_root_text.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate != app_config.dataset.root_path:
            dataset_root_override = candidate
            effective_root = candidate

    dataset = DABenchPublicDataset(effective_root)
    if not dataset.exists:
        st.error(
            f"Tasks folder does not exist or has no task_* dirs: {effective_root}"
        )
        return

    # ---- Determine history scope (always shown at bottom) ----
    scope_path: Path | None = None
    scope_raw = st.session_state.get("dashboard_scope")
    if scope_raw:
        scope_path = Path(str(scope_raw))
    else:
        scope_input = Path(run_dir_text).expanduser()
        if not scope_input.is_absolute():
            scope_input = (PROJECT_ROOT / scope_input).resolve()
        if scope_input.exists():
            scope_path = scope_input
        else:
            scope_path = _latest_run_dir()

    # ==================================================================
    # TOP SECTION: Live run OR Replay OR Idle hero
    # ==================================================================

    if run_all_clicked:
        # --- Batch: Run All ---
        all_ids = dataset.list_task_ids()
        if difficulty_pick:
            picked = set(difficulty_pick)
            tasks = dataset.iter_tasks(difficulties=list(picked))
            selected_ids = [t.task_id for t in tasks]
        else:
            selected_ids = list(all_ids)
        if batch_limit and batch_limit > 0:
            selected_ids = selected_ids[: int(batch_limit)]
        if not selected_ids:
            st.warning(
                "No tasks selected. Check the Tasks folder path and the "
                "Difficulty filter."
            )
            return
        st.session_state.pop("replay_trace_path", None)
        try:
            _run_batch(
                task_ids=selected_ids,
                config_path=config_path,
                dataset_root=dataset_root_override,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Batch failed: {exc}")
            return
        # _run_batch sets dashboard_scope itself; fall through to history.

    elif run_clicked:
        # --- Live run ---
        if not task_id.strip():
            st.error("Please enter a task id.")
            return
        run_id = "demo-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            artifact_payload, trace_path = _run_task_live(
                task_id=task_id.strip(),
                config_path=config_path,
                run_id=run_id,
                dataset_root=dataset_root_override,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Run failed: {exc}")
            return
        st.success(f"Run saved \u2192 {artifact_payload.get('task_output_dir')}")
        st.session_state["replay_trace_path"] = str(trace_path)
        st.session_state["dashboard_scope"] = str(_run_dir_for_trace(trace_path))
        # Fall through to render history below

    elif st.session_state.get("replay_trace_path"):
        # --- Replay (from history card click or previous run) ---
        trace_path = Path(str(st.session_state["replay_trace_path"]))
        try:
            trace = _load_trace(trace_path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load trace: {exc}")
            st.session_state.pop("replay_trace_path", None)
        else:
            question_text = ""
            try:
                question_text = dataset.get_task(
                    str(trace.get("task_id") or trace_path.parent.name)
                ).question
            except Exception:  # noqa: BLE001
                pass
            _render_replay_view(trace, trace_path, question=question_text)
            # Update scope to the run that owns this trace
            if not scope_path or not scope_path.exists():
                scope_path = _run_dir_for_trace(trace_path)
                st.session_state["dashboard_scope"] = str(scope_path)
    else:
        # --- Idle state ---
        st.markdown(
            "<div class='hero'><h1>Data Agent Theater</h1>"
            "<p>Real-time parsing of the agent's thought / action / observation stream. "
            "Pick a task and click <b>Run Agent</b>, or click a history card below to replay it.</p></div>",
            unsafe_allow_html=True,
        )

    # ==================================================================
    # BOTTOM SECTION: History grid (always visible)
    # ==================================================================

    st.divider()

    if scope_path and scope_path.exists():
        _render_run_dashboard(scope=scope_path, app_config=app_config, dataset=dataset)
    else:
        fallback = _latest_run_dir()
        if fallback:
            _render_run_dashboard(scope=fallback, app_config=app_config, dataset=dataset)
        else:
            st.info("No past runs found yet. Use the sidebar to start a task.")


if __name__ == "__main__":
    main()
