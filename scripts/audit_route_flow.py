#!/usr/bin/env python3
"""Audit router flow without running agents or calling model APIs.

This script only reads task metadata/context files and the selected config.
It runs the deterministic compiler plus router route selection, then prints
where each task would flow.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_agent_baseline.agents.router import _route_for_compiled_task  # noqa: E402
from data_agent_baseline.agents.task_compiler import compile_task  # noqa: E402
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset  # noqa: E402
from data_agent_baseline.config import load_app_config  # noqa: E402


FIELDS = [
    "task_id",
    "difficulty",
    "task_type",
    "answer_type",
    "budget",
    "llm_budget",
    "tool_budget",
    "route",
    "route_reason",
    "kind",
    "model",
    "primary_tool",
    "auxiliary_tools",
    "modalities",
    "operations",
    "source_kinds",
    "sources",
    "flags",
]


def _split_task_ids(raw_items: list[str]) -> set[str]:
    task_ids: set[str] = set()
    for item in raw_items:
        for piece in item.split(","):
            task_id = piece.strip()
            if task_id:
                task_ids.add(task_id)
    return task_ids


def _cell(value: object) -> str:
    return str(value).replace("\t", " ").replace("\n", " ")


def _iter_rows(
    *,
    config_path: Path,
    task_ids: set[str],
    difficulty: str | None,
) -> list[dict[str, str]]:
    config = load_app_config(config_path)
    dataset = DABenchPublicDataset(config.dataset.root_path)
    rows: list[dict[str, str]] = []

    selected = sorted(task_ids) if task_ids else None
    tasks = dataset.iter_tasks(task_ids=selected, difficulty=difficulty)
    for task in tasks:
        compiled = compile_task(task)
        route_name, route_reason, _fallback_used = _route_for_compiled_task(
            config.agent.router,
            compiled,
        )
        route = config.agent.router.routes[route_name]
        rows.append({
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "task_type": compiled.task_type,
            "answer_type": compiled.answer_type,
            "budget": compiled.budget_level,
            "llm_budget": str(compiled.max_llm_calls),
            "tool_budget": str(compiled.max_tool_calls),
            "route": route_name,
            "route_reason": route_reason,
            "kind": route.kind,
            "model": route.model,
            "primary_tool": compiled.primary_tool,
            "auxiliary_tools": ",".join(compiled.auxiliary_tools),
            "modalities": ",".join(compiled.modalities),
            "operations": ",".join(compiled.operations),
            "source_kinds": ";".join(
                f"{cap.kind}:{cap.role}" for cap in compiled.source_capabilities
            ),
            "sources": ";".join(
                f"{cap.path}:{cap.kind}" for cap in compiled.source_capabilities
            ),
            "flags": ",".join(compiled.ambiguity_flags),
        })
    return rows


def _write_tsv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _print_tsv(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)


def _print_summary(rows: list[dict[str, str]]) -> None:
    task_type = Counter(row["task_type"] for row in rows)
    route = Counter(f'{row["route"]}/{row["kind"]}/{row["model"]}' for row in rows)
    difficulty_to_type = Counter(
        f'{row["difficulty"]} -> {row["task_type"]}' for row in rows
    )
    primary_tool = Counter(row["primary_tool"] for row in rows)

    print(f"tasks: {len(rows)}")
    _print_counter("task_type", task_type)
    _print_counter("route", route)
    _print_counter("primary_tool", primary_tool)
    _print_counter("difficulty -> task_type", difficulty_to_type)


def _print_counter(title: str, counter: Counter[str]) -> None:
    print(f"\n{title}:")
    for key, count in counter.most_common():
        print(f"  {count:>3}  {key}")


def _print_table(rows: list[dict[str, str]], columns: Iterable[str]) -> None:
    selected = list(columns)
    widths = {
        column: max(
            len(column),
            *(len(_cell(row.get(column, ""))) for row in rows),
        )
        for column in selected
    }
    print("  ".join(column.ljust(widths[column]) for column in selected))
    print("  ".join("-" * widths[column] for column in selected))
    for row in rows:
        print("  ".join(_cell(row.get(column, "")).ljust(widths[column]) for column in selected))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show local task compiler/router flow without model API calls.",
    )
    parser.add_argument(
        "--config",
        default="configs/router.deepseek.yaml",
        type=Path,
        help="Config file to audit. Default: configs/router.deepseek.yaml",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task id to include, e.g. --task task_418. Can repeat or use commas.",
    )
    parser.add_argument(
        "--difficulty",
        help="Only include one difficulty label, e.g. easy, medium, hard, extreme.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write full TSV to this path.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "tsv", "summary"),
        default="table",
        help="Print format. Default: table.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Shortcut for --format summary.",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="In table mode, include source and flag columns.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    rows = _iter_rows(
        config_path=config_path,
        task_ids=_split_task_ids(args.task),
        difficulty=args.difficulty,
    )

    if args.summary:
        args.format = "summary"

    if args.output:
        output = args.output
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        _write_tsv(rows, output)
        print(f"wrote: {output}")

    if args.format == "summary":
        _print_summary(rows)
    elif args.format == "tsv":
        _print_tsv(rows)
    else:
        columns = [
            "task_id",
            "difficulty",
            "task_type",
            "route",
            "kind",
            "model",
            "primary_tool",
        ]
        if args.wide:
            columns.extend(["modalities", "operations", "source_kinds", "flags"])
        _print_table(rows, columns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
