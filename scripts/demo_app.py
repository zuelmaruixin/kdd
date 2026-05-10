from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "router.deepseek.yaml"
_RESULT_PREFIX = "DEMO_RESULT_JSON="


_SUBPROCESS_RUNNER = r"""
import json
import sys
from dataclasses import replace
from pathlib import Path

from data_agent_baseline.config import load_app_config
from data_agent_baseline.progress import ProgressLogger, set_progress_logger
from data_agent_baseline.run.runner import create_run_output_dir, run_single_task

task_id = sys.argv[1]
config_path = Path(sys.argv[2])
run_id = sys.argv[3]

config = load_app_config(config_path)
config = replace(config, run=replace(config.run, run_id=run_id, max_workers=1))
set_progress_logger(ProgressLogger(enabled=True, lang="zh"))

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


def _tail_lines(lines: list[str], *, max_lines: int = 140) -> str:
    return "".join(lines[-max_lines:]).strip() or "Starting..."


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

    lines: list[str] = []
    result_payload: dict[str, Any] | None = None

    status = st.status(f"Running {task_id}", state="running", expanded=True)
    log_box = status.empty()
    log_box.code("Starting...", language="text")

    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith(_RESULT_PREFIX):
            result_payload = json.loads(line.removeprefix(_RESULT_PREFIX))
            continue
        lines.append(line)
        log_box.code(_tail_lines(lines), language="text")

    return_code = process.wait()
    log_box.code(_tail_lines(lines), language="text")

    if return_code != 0:
        status.update(label=f"Run failed: {task_id}", state="error", expanded=True)
        raise RuntimeError(f"subprocess exited with code {return_code}")
    if result_payload is None:
        status.update(label=f"Run failed: {task_id}", state="error", expanded=True)
        raise RuntimeError("run finished but did not return artifact metadata")

    status.update(label=f"Run finished: {task_id}", state="complete", expanded=False)
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
    compiled = trace.get("compiled_task") or {}
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
    for key in ("operator_executor", "tablellm_direct"):
        value = trace.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _stage_status(trace: dict[str, Any]) -> list[tuple[str, str, str]]:
    compiled = trace.get("compiled_task") or {}
    operator = _get_operator_block(trace)
    validation = trace.get("answer_validation") or {}
    manifest = operator.get("context_manifest") or []

    semantic_plan = None
    cheap_guard = None
    semantic_consistency = None
    semantic_status = None
    for item in manifest:
        if not isinstance(item, dict):
            continue
        if item.get("cheap_semantic_assessment"):
            cheap_guard = item["cheap_semantic_assessment"]
        if item.get("semantic_plan"):
            semantic_plan = item["semantic_plan"]
        semantic_value = item.get("semantic_consistency")
        if isinstance(semantic_value, dict):
            semantic_consistency = semantic_value
        elif isinstance(semantic_value, str):
            semantic_status = semantic_value

    rows = [
        ("Schema inspect", "done", f"{len(compiled.get('source_capabilities') or [])} source(s)"),
        ("Route", "done", (trace.get("router_decision") or {}).get("route_name") or "-"),
        (
            "Codegen",
            "done" if operator.get("program") else "skipped",
            operator.get("failure_reason") or "",
        ),
        (
            "Execute",
            "done" if operator.get("succeeded") else "failed",
            operator.get("failure_reason") or "",
        ),
        ("Answer validation", "done" if validation.get("valid") else "failed", ""),
    ]

    if cheap_guard:
        label = "escalated" if cheap_guard.get("should_escalate") else "passed"
        rows.append(("Cheap semantic guard", label, f"risk={cheap_guard.get('score')}"))
    else:
        rows.append(("Cheap semantic guard", "passed/skipped", "no risk event recorded"))

    if semantic_plan:
        details = f"confidence={semantic_plan.get('confidence')}"
        if semantic_plan.get("_force_semantic_consistency"):
            details += " | forced by cheap guard"
        if semantic_plan.get("plan_failure"):
            failure = semantic_plan["plan_failure"]
            details += (
                f" | fallback because {failure.get('stage')}: "
                f"{failure.get('exception_type') or failure.get('reason') or ''}"
            )
        rows.append(("Semantic plan", "used", details))
    elif semantic_status == "fallback_plan_from_cheap_guard":
        rows.append(("Semantic plan", "fallback", "built from cheap guard risks"))
    elif semantic_status == "escalation_failed_no_plan":
        rows.append(("Semantic plan", "failed", "cheap guard escalated, but no plan was produced"))
    else:
        rows.append(("Semantic plan", "skipped", "fast path"))

    if semantic_consistency:
        rows.append((
            "Consistency judge",
            semantic_consistency.get("final_verdict") or "used",
            semantic_consistency.get("failure_reason") or "",
        ))
    elif semantic_status == "lazy_escalation":
        rows.append((
            "Consistency judge",
            "pending/missing",
            "semantic plan was created but no verdict found",
        ))
    elif semantic_status == "escalation_failed_no_plan":
        rows.append(("Consistency judge", "skipped", "blocked because semantic plan failed"))
    else:
        rows.append(("Consistency judge", "skipped", "not needed"))

    return rows


def _render_stage_timeline(trace: dict[str, Any]) -> None:
    st.subheader("Reasoning Timeline")
    for name, status, detail in _stage_status(trace):
        state = "complete"
        if status in {"failed", "fail"}:
            state = "error"
        with st.status(f"{name}: {status}", state=state, expanded=False):
            st.write(detail or "ok")


def _render_trace(trace: dict[str, Any], trace_path: Path) -> None:
    summary = _route_summary(trace)
    cols = st.columns(4)
    # Result 显示逻辑：既要跑通，也要得分（如果有标准答案）
    local_score = trace.get("local_score") or {}
    score_value = local_score.get("score")

    if not summary["succeeded"]:
        result_label = "FAIL"
    elif score_value is not None and score_value < 0.1:
        result_label = f"FAIL (score={score_value:.2f})"
    elif score_value is not None:
        result_label = f"OK (score={score_value:.2f})"
    else:
        result_label = "OK"  # 没有标准答案，不能判对错
    if "FAIL" in result_label:
        st.error(f"✗ {result_label}")
    elif score_value is not None and score_value >= 0.99:
        st.success(f"✓ {result_label}")
    cols[0].metric("Result", result_label)
    cols[1].metric("Route", str(summary["route"] or "-"))
    cols[2].metric("Task Type", str(summary["task_type"] or "-"))
    cols[3].metric("Elapsed", f"{summary['elapsed_seconds'] or 0}s")

    st.caption(
        f"kind={summary['kind'] or '-'} | model={summary['model'] or '-'} | "
        f"ops={summary['operations'] or '-'} | answer_valid={summary['answer_valid']}"
    )

    _render_stage_timeline(trace)
     # --- Verification Gate badge ---
    audit = trace.get("semantic_consistency_audit") or {}
    gate = audit.get("final_gate")
    if gate:
        badge = {
            "plan+judge":       (":green[plan + judge]",
                                 "analyst produced a plan and judge ran — strong path"),
            "escalated+judge":  (":blue[escalated + judge]",
                                 "plan was recovered via escalation (analyst-exception retry, "
                                 "cheap-guard lazy escalation, or fallback plan) — recovered path"),
            "cheap_guard_only": (":orange[cheap guard only]",
                                 "cheap guard flagged risk but judge did not run — investigate"),
            "bypassed":         (":gray[bypassed]",
                                 "no plan, no judge — task deemed low-risk"),
        }.get(gate, (f":gray[{gate}]", ""))
        st.markdown(f"**Verification Gate:** {badge[0]}")
        if badge[1]:
            st.caption(badge[1])
        # small extras if the cache hit or judge attempted repair
        extras = []
        if audit.get("plan_cache_hit"):
            extras.append("plan cache hit")
        if audit.get("plan_source") == "llm":
            extras.append("plan from LLM")
        if audit.get("judge_attempts"):
            extras.append(f"judge attempts: {audit['judge_attempts']}")
        if audit.get("judge_repaired_code"):
            extras.append("judge repaired code")
        if extras:
            st.caption(" · ".join(extras))
    st.subheader("Answer")
    frame = _answer_frame(trace.get("answer"))
    if frame is None:
        st.warning("No answer table was produced.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)

    operator = _get_operator_block(trace)
    with st.expander("Generated Program", expanded=False):
        st.code(operator.get("program") or "(no program)", language="python")

    with st.expander("Debug / Execution Output", expanded=False):
        stdout = operator.get("exec_stdout") or ""
        stderr = operator.get("exec_stderr") or ""
        st.code(stdout[-6000:] or "(empty stdout)", language="text")
        if stderr:
            st.code(stderr[-6000:], language="text")

    with st.expander("Full trace.json", expanded=False):
        st.caption(str(trace_path))
        st.json(trace)


def main() -> None:
    st.set_page_config(page_title="DABench Demo", layout="wide")
    st.title("DABench Reasoning Demo")
    st.caption(
        "Enter a task id, run the existing agent pipeline, "
        "and show the reasoning trace cleanly."
    )

    with st.sidebar:
        st.header("Run")
        config_text = st.text_input("Config", value=str(DEFAULT_CONFIG))
        task_id = st.text_input("Task ID", value="")
        show_task = st.checkbox("Show task context summary", value=True)
        run_clicked = st.button("Run Task", type="primary", use_container_width=True)

    config_path = Path(config_text).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    try:
        app_config = load_app_config(config_path)
        dataset = DABenchPublicDataset(app_config.dataset.root_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load config/dataset: {exc}")
        return

    if show_task and task_id:
        try:
            task = dataset.get_task(task_id)
            st.info(f"{task.task_id} | {task.difficulty}\n\n{task.question}")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Task preview failed: {exc}")

    if not run_clicked:
        st.write("Ready.")
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
    except Exception as exc:  # noqa: BLE001
        st.error(f"Run failed: {exc}")
        return

    st.success(f"Run saved to {artifact_payload.get('task_output_dir')}")
    _render_trace(trace, trace_path)


if __name__ == "__main__":
    main()
