"""Three-arm ablation of the semantic-consistency pipeline.

Runs the same picked tasks under three configurations and prints a comparison
table. No extra dependencies -- uses the project's own Python API directly.

  A. codegen_only          : semantic layer fully off (guard + judge + repair).
  B. guard_judge_no_repair : guard runs, judge runs once, 0 repair rounds.
  C. full                  : guard + judge + up to 3 repair rounds (default).

Example:

    uv run python scripts/ablation.py \\
        --config configs/router.deepseek.yaml \\
        --easy 8 --medium 5 --hard 3

Outputs a CSV plus a rich-printed summary. All arms share the same picked task
list so the comparison is 1:1.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any

from rich.console import Console
from rich.table import Table

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig, load_app_config
from data_agent_baseline.eval.column_match import score_run
from data_agent_baseline.run.runner import create_run_output_dir, run_single_task

console = Console()


# --------------------------------------------------------------------------- #
# Ablation arm definitions
# --------------------------------------------------------------------------- #

ARMS: list[dict[str, Any]] = [
    {
        "name": "A_codegen_only",
        "label": "A. codegen only",
        "semantic_enabled": False,
        "max_repairs": 0,
    },
    {
        "name": "B_guard_judge_no_repair",
        "label": "B. guard+judge, 0 repair",
        "semantic_enabled": True,
        "max_repairs": 0,
    },
    {
        "name": "C_full",
        "label": "C. full (guard+judge+repair)",
        "semantic_enabled": True,
        "max_repairs": 3,
    },
]


# --------------------------------------------------------------------------- #
# Config override
# --------------------------------------------------------------------------- #


def apply_arm(cfg: AppConfig, *, semantic_enabled: bool, max_repairs: int) -> AppConfig:
    """Return a new AppConfig with the arm's overrides applied to every route."""
    new_routes = {
        name: replace(
            route,
            semantic_consistency_enabled=semantic_enabled,
            semantic_consistency_max_repairs=max_repairs,
        )
        for name, route in cfg.agent.router.routes.items()
    }
    return replace(
        cfg,
        agent=replace(
            cfg.agent,
            router=replace(cfg.agent.router, routes=new_routes),
        ),
    )


# --------------------------------------------------------------------------- #
# Task selection
# --------------------------------------------------------------------------- #


def pick_tasks(
    dataset: DABenchPublicDataset,
    want: dict[str, int],
) -> list[tuple[str, str]]:
    """Pick the first N tasks for each difficulty bucket.

    Tolerates capitalization differences (``Easy`` vs ``easy``).
    Returns a list of ``(task_id, difficulty_label)``.
    """
    all_tasks = dataset.iter_tasks()
    # group by lowercased difficulty
    buckets: dict[str, list[Any]] = {}
    for task in all_tasks:
        buckets.setdefault(task.difficulty.strip().lower(), []).append(task)

    picked: list[tuple[str, str]] = []
    for requested, count in want.items():
        key = requested.strip().lower()
        bucket = buckets.get(key, [])
        if len(bucket) < count:
            console.print(
                f"[yellow]warning:[/yellow] only {len(bucket)} tasks available "
                f"for difficulty={requested!r}, requested {count}"
            )
        picked.extend((t.task_id, t.difficulty) for t in bucket[:count])
    return picked


# --------------------------------------------------------------------------- #
# Trace metric extraction
# --------------------------------------------------------------------------- #


def extract_trace_metrics(trace_path: Path) -> dict[str, Any]:
    """Pull LLM / tool call counts + route cascade info out of trace.json."""
    if not trace_path.exists():
        return {"llm_calls": 0, "tool_calls": 0, "route": None, "cascade": 0}

    data = json.loads(trace_path.read_text())

    # Budget counters (features branch format)
    budget = data.get("budget") or {}
    llm_calls = int(budget.get("llm_calls_used", budget.get("llm_calls", 0)) or 0)
    tool_calls = int(budget.get("tool_calls_used", budget.get("tool_calls", 0)) or 0)

    # Router decision info
    decision = data.get("router_decision") or {}
    route = decision.get("route_name")
    cascade_attempts = len(decision.get("cascade_attempts") or [])

    # Did the cheap guard fire?
    escalated = False
    operator = data.get("operator_executor") or data.get("tablellm_direct") or {}
    for item in operator.get("context_manifest") or []:
        if isinstance(item, dict) and item.get("cheap_semantic_assessment"):
            if item["cheap_semantic_assessment"].get("should_escalate"):
                escalated = True
                break

    return {
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "route": route,
        "cascade": cascade_attempts,
        "guard_escalated": escalated,
    }


# --------------------------------------------------------------------------- #
# Per-arm run
# --------------------------------------------------------------------------- #


def run_arm(
    arm: dict[str, Any],
    base_cfg: AppConfig,
    picked: list[tuple[str, str]],
    stamp: str,
) -> list[dict[str, Any]]:
    """Run every picked task under this arm and return per-task metric rows."""
    arm_cfg = apply_arm(
        base_cfg,
        semantic_enabled=arm["semantic_enabled"],
        max_repairs=arm["max_repairs"],
    )
    run_id = f"ablation-{stamp}-{arm['name']}"
    arm_cfg = replace(arm_cfg, run=replace(arm_cfg.run, run_id=run_id))

    _, run_output_dir = create_run_output_dir(arm_cfg.run.output_dir, run_id=run_id)
    console.print(f"\n[bold cyan]=== Arm: {arm['label']} ===[/bold cyan]")
    console.print(f"Run dir: {run_output_dir}")

    rows: list[dict[str, Any]] = []
    for idx, (task_id, difficulty) in enumerate(picked, 1):
        console.print(f"  [{idx}/{len(picked)}] {task_id} ({difficulty}) ...", end="")
        t0 = time.perf_counter()
        try:
            artifact = run_single_task(
                task_id=task_id,
                config=arm_cfg,
                run_output_dir=run_output_dir,
            )
            elapsed = time.perf_counter() - t0
            row = {
                "arm": arm["name"],
                "task_id": task_id,
                "difficulty": difficulty,
                "succeeded": bool(artifact.succeeded),
                "elapsed_s": round(elapsed, 2),
                "failure_reason": artifact.failure_reason or "",
            }
            console.print(
                f" [green]ok[/green]" if artifact.succeeded else f" [red]fail[/red]",
                f"({elapsed:.1f}s)",
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            row = {
                "arm": arm["name"],
                "task_id": task_id,
                "difficulty": difficulty,
                "succeeded": False,
                "elapsed_s": round(elapsed, 2),
                "failure_reason": f"exception: {exc}",
            }
            console.print(f" [red]exception[/red] ({elapsed:.1f}s): {exc}")

        # Pull trace metrics
        trace_path = run_output_dir / task_id / "trace.json"
        row.update(extract_trace_metrics(trace_path))
        rows.append(row)

    # Score the whole run against gold and merge scores in
    try:
        summary = score_run(
            run_dir=run_output_dir,
            gold_root=base_cfg.dataset.gold_root,
            redundancy_lambda=base_cfg.scoring.redundancy_lambda,
            numeric_tolerance=base_cfg.scoring.numeric_tolerance,
            case_insensitive=base_cfg.scoring.case_insensitive,
            strip_whitespace=base_cfg.scoring.strip_whitespace,
        )
        by_task = {s.task_id: s for s in summary.per_task}
        for row in rows:
            s = by_task.get(row["task_id"])
            row["score"] = round(s.score, 4) if s else 0.0
            row["recall"] = round(s.recall, 4) if s else 0.0
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]scoring failed for arm {arm['name']}: {exc}[/yellow]")
        for row in rows:
            row.setdefault("score", 0.0)
            row.setdefault("recall", 0.0)

    return rows


# --------------------------------------------------------------------------- #
# Aggregation and printing
# --------------------------------------------------------------------------- #


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate rows into per-arm and per-arm-per-difficulty stats."""
    out: dict[str, dict[str, float]] = {}
    arms = sorted({r["arm"] for r in rows})
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        out[arm] = _aggregate_block(arm_rows)
    return out


def _aggregate_block(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"n": 0, "mean_score": 0, "mean_llm_calls": 0, "success_rate": 0, "median_elapsed": 0}
    scores = [r.get("score", 0.0) for r in rows]
    llms = [r.get("llm_calls", 0) for r in rows]
    elapsed = [r.get("elapsed_s", 0.0) for r in rows]
    succ = sum(1 for r in rows if r.get("succeeded"))
    return {
        "n": len(rows),
        "mean_score": round(mean(scores), 4),
        "mean_llm_calls": round(mean(llms), 2),
        "success_rate": round(succ / len(rows), 3),
        "median_elapsed": round(median(elapsed), 2),
    }


def print_comparison(rows: list[dict[str, Any]]) -> None:
    # Overall
    table = Table(title="Ablation Summary (overall)")
    table.add_column("Arm")
    table.add_column("n", justify="right")
    table.add_column("mean score", justify="right")
    table.add_column("success rate", justify="right")
    table.add_column("mean LLM calls", justify="right")
    table.add_column("median elapsed (s)", justify="right")

    summary = summarize(rows)
    for arm_spec in ARMS:
        name = arm_spec["name"]
        stats = summary.get(name, {})
        table.add_row(
            arm_spec["label"],
            str(stats.get("n", 0)),
            f"{stats.get('mean_score', 0):.4f}",
            f"{stats.get('success_rate', 0):.2f}",
            f"{stats.get('mean_llm_calls', 0):.2f}",
            f"{stats.get('median_elapsed', 0):.1f}",
        )
    console.print(table)

    # By difficulty
    diff_table = Table(title="Ablation Summary (by difficulty)")
    diff_table.add_column("Arm")
    diff_table.add_column("Difficulty")
    diff_table.add_column("n", justify="right")
    diff_table.add_column("mean score", justify="right")
    diff_table.add_column("mean LLM calls", justify="right")

    difficulties = sorted({r["difficulty"] for r in rows})
    for arm_spec in ARMS:
        for diff in difficulties:
            block = [
                r for r in rows
                if r["arm"] == arm_spec["name"] and r["difficulty"] == diff
            ]
            stats = _aggregate_block(block)
            diff_table.add_row(
                arm_spec["label"],
                diff,
                str(stats.get("n", 0)),
                f"{stats.get('mean_score', 0):.4f}",
                f"{stats.get('mean_llm_calls', 0):.2f}",
            )
    console.print(diff_table)


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "arm", "task_id", "difficulty", "succeeded",
        "score", "recall", "llm_calls", "tool_calls",
        "guard_escalated", "cascade", "route",
        "elapsed_s", "failure_reason",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    console.print(f"\nPer-task CSV: [bold]{out_path}[/bold]")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="YAML config path")
    parser.add_argument("--easy", type=int, default=8, help="# of Easy tasks (default 8)")
    parser.add_argument("--medium", type=int, default=5, help="# of Medium tasks (default 5)")
    parser.add_argument("--hard", type=int, default=3, help="# of Hard tasks (default 3)")
    parser.add_argument("--out", type=Path, default=Path("ablation_summary.csv"))
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[a["name"] for a in ARMS],
        help="Only run these arms (default: all three).",
    )
    args = parser.parse_args()

    base_cfg = load_app_config(args.config)
    dataset = DABenchPublicDataset(base_cfg.dataset.root_path)
    if not dataset.exists:
        console.print(f"[red]Dataset not found: {base_cfg.dataset.root_path}[/red]")
        raise SystemExit(1)

    picked = pick_tasks(dataset, {
        "Easy": args.easy,
        "Medium": args.medium,
        "Hard": args.hard,
    })
    console.print(f"Picked {len(picked)} tasks: "
                  f"{[f'{tid}({d})' for tid, d in picked]}")

    arms_to_run = ARMS if not args.arms else [a for a in ARMS if a["name"] in args.arms]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    all_rows: list[dict[str, Any]] = []
    for arm in arms_to_run:
        all_rows.extend(run_arm(arm, base_cfg, picked, stamp))

    print_comparison(all_rows)
    write_csv(all_rows, args.out)


if __name__ == "__main__":
    main()
