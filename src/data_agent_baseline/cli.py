from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from dataclasses import replace

from data_agent_baseline.agents.model import StreamSink, set_stream_sink
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig, RouteConfig, load_app_config
from data_agent_baseline.eval.column_match import score_run
from data_agent_baseline.progress import ProgressLogger, set_progress_logger
from data_agent_baseline.run.runner import TaskRunArtifacts, create_run_output_dir, run_benchmark, run_single_task
from data_agent_baseline.tools.filesystem import list_context_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    table = Table(title="DABench Baseline Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row("dataset_root", str(app_config.dataset.root_path), _status_value(app_config.dataset.root_path))
    table.add_row("config_path", str(config_path), _status_value(config_path))

    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(
                f"{difficulty}={count}" for difficulty, count in sorted(counts.items())
            )
            console.print(f"Public task counts: {rendered_counts}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show task metadata and available context files."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


def _override_route(route: RouteConfig, *, model: str | None, api_base: str | None, api_key: str | None) -> RouteConfig:
    if model is None and api_base is None and api_key is None:
        return route
    return replace(
        route,
        model=model if model is not None else route.model,
        api_base=api_base if api_base is not None else route.api_base,
        api_key=api_key if api_key is not None else route.api_key,
    )


def _apply_overrides(
    app_config: AppConfig,
    *,
    mode: str | None,
    num_samples: int | None,
    model: str | None = None,
    route_overrides: dict[str, dict[str, str | None]] | None = None,
) -> AppConfig:
    """Apply CLI overrides to a freshly-loaded :class:`AppConfig`.

    ``route_overrides`` is keyed by route name (``"easy" / "medium" / "hard" / "extreme"``)
    and each value is a dict with optional ``model``, ``api_base``, ``api_key``.
    """
    new_agent = app_config.agent

    if mode is not None:
        normalized_mode = mode.strip().lower()
        valid_modes = {"react", "multi_agent", "router"}
        if normalized_mode not in valid_modes:
            raise typer.BadParameter(
                f"agent mode must be one of {sorted(valid_modes)}.",
                param_hint="--mode",
            )
        new_agent = replace(new_agent, mode=normalized_mode)

    if num_samples is not None:
        new_agent = replace(
            new_agent,
            self_consistency=replace(new_agent.self_consistency, num_samples=num_samples),
        )

    if model is not None:
        new_agent = replace(new_agent, model=model)

    if route_overrides:
        new_routes = dict(new_agent.router.routes)
        for route_name, fields in route_overrides.items():
            if route_name not in new_routes:
                # Gracefully ignore unknown route names so a typo doesn't kill the run,
                # but warn so it shows up in the console.
                console.print(
                    f"[yellow]warning:[/yellow] --{route_name}-* override given but no "
                    f"such route in {sorted(new_routes.keys())}; ignored."
                )
                continue
            new_routes[route_name] = _override_route(
                new_routes[route_name],
                model=fields.get("model"),
                api_base=fields.get("api_base"),
                api_key=fields.get("api_key"),
            )
        new_agent = replace(new_agent, router=replace(new_agent.router, routes=new_routes))

    return replace(app_config, agent=new_agent)


def _setup_streaming(stream: bool) -> None:
    if stream:
        set_stream_sink(StreamSink(enabled=True))
    else:
        set_stream_sink(None)


def _setup_progress_logger(*, enabled: bool, lang: str = "en") -> None:
    if enabled:
        set_progress_logger(ProgressLogger(enabled=True, lang=lang))
    else:
        set_progress_logger(None)


def _collect_route_overrides(
    *,
    easy_model: str | None,
    medium_model: str | None,
    hard_model: str | None,
    extreme_model: str | None,
    easy_api_base: str | None = None,
    medium_api_base: str | None = None,
    hard_api_base: str | None = None,
    extreme_api_base: str | None = None,
    easy_api_key: str | None = None,
    medium_api_key: str | None = None,
    hard_api_key: str | None = None,
    extreme_api_key: str | None = None,
) -> dict[str, dict[str, str | None]]:
    raw: dict[str, dict[str, str | None]] = {
        "easy": {"model": easy_model, "api_base": easy_api_base, "api_key": easy_api_key},
        "medium": {"model": medium_model, "api_base": medium_api_base, "api_key": medium_api_key},
        "hard": {"model": hard_model, "api_base": hard_api_base, "api_key": hard_api_key},
        "extreme": {"model": extreme_model, "api_base": extreme_api_base, "api_key": extreme_api_key},
    }
    # Drop entries where every field is None — keeps trace clean.
    return {
        route: fields
        for route, fields in raw.items()
        if any(value is not None for value in fields.values())
    }


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    mode: str | None = typer.Option(None, help="Override agent.mode: 'react', 'multi_agent', or 'router'."),
    num_samples: int | None = typer.Option(None, min=1, help="Override agent.self_consistency.num_samples."),
    model: str | None = typer.Option(None, help="Override the top-level agent.model (used when mode is react/multi_agent)."),
    easy_model: str | None = typer.Option(None, help="Override agent.router.routes.easy.model."),
    medium_model: str | None = typer.Option(None, help="Override agent.router.routes.medium.model."),
    hard_model: str | None = typer.Option(None, help="Override agent.router.routes.hard.model."),
    extreme_model: str | None = typer.Option(None, help="Override agent.router.routes.extreme.model."),
    easy_api_base: str | None = typer.Option(None),
    medium_api_base: str | None = typer.Option(None),
    hard_api_base: str | None = typer.Option(None),
    extreme_api_base: str | None = typer.Option(None),
    easy_api_key: str | None = typer.Option(None),
    medium_api_key: str | None = typer.Option(None),
    hard_api_key: str | None = typer.Option(None),
    extreme_api_key: str | None = typer.Option(None),
    stream: bool = typer.Option(False, "--stream", help="Stream raw model tokens to stderr while running."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress the structured event-flow log (router decision, plan, specialist findings, ...)."),
    lang: str = typer.Option("en", "--lang", help="Structured event-log language: en or zh. Raw data is not translated."),
) -> None:
    """Run the agent on one task. Omit --mode to use the config's agent.mode."""
    _setup_streaming(stream)
    if lang not in {"en", "zh"}:
        raise typer.BadParameter("lang must be 'en' or 'zh'.", param_hint="--lang")
    _setup_progress_logger(enabled=not quiet, lang=lang)
    route_overrides = _collect_route_overrides(
        easy_model=easy_model, medium_model=medium_model, hard_model=hard_model, extreme_model=extreme_model,
        easy_api_base=easy_api_base, medium_api_base=medium_api_base, hard_api_base=hard_api_base, extreme_api_base=extreme_api_base,
        easy_api_key=easy_api_key, medium_api_key=medium_api_key, hard_api_key=hard_api_key, extreme_api_key=extreme_api_key,
    )
    app_config = _apply_overrides(
        load_app_config(config),
        mode=mode,
        num_samples=num_samples,
        model=model,
        route_overrides=route_overrides,
    )
    try:
        _, run_output_dir = create_run_output_dir(app_config.run.output_dir, run_id=app_config.run.run_id)
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    mode: str | None = typer.Option(None, help="Override agent.mode: 'react', 'multi_agent', or 'router'."),
    num_samples: int | None = typer.Option(None, min=1, help="Override agent.self_consistency.num_samples."),
    model: str | None = typer.Option(None, help="Override the top-level agent.model."),
    easy_model: str | None = typer.Option(None),
    medium_model: str | None = typer.Option(None),
    hard_model: str | None = typer.Option(None),
    extreme_model: str | None = typer.Option(None),
    easy_api_base: str | None = typer.Option(None),
    medium_api_base: str | None = typer.Option(None),
    hard_api_base: str | None = typer.Option(None),
    extreme_api_base: str | None = typer.Option(None),
    easy_api_key: str | None = typer.Option(None),
    medium_api_key: str | None = typer.Option(None),
    hard_api_key: str | None = typer.Option(None),
    extreme_api_key: str | None = typer.Option(None),
    stream: bool = typer.Option(False, "--stream", help="Stream raw model tokens. Forces max_workers=1."),
    show_events: bool = typer.Option(False, "--show-events", help="Force-enable structured event-flow log (off by default in batch mode because workers interleave)."),
    lang: str = typer.Option("en", "--lang", help="Structured event-log language: en or zh. Raw data is not translated."),
) -> None:
    """Run the agent on multiple tasks from the config selection."""
    if lang not in {"en", "zh"}:
        raise typer.BadParameter("lang must be 'en' or 'zh'.", param_hint="--lang")
    route_overrides = _collect_route_overrides(
        easy_model=easy_model, medium_model=medium_model, hard_model=hard_model, extreme_model=extreme_model,
        easy_api_base=easy_api_base, medium_api_base=medium_api_base, hard_api_base=hard_api_base, extreme_api_base=extreme_api_base,
        easy_api_key=easy_api_key, medium_api_key=medium_api_key, hard_api_key=hard_api_key, extreme_api_key=extreme_api_key,
    )
    app_config = _apply_overrides(
        load_app_config(config),
        mode=mode,
        num_samples=num_samples,
        model=model,
        route_overrides=route_overrides,
    )
    if stream:
        # Streaming with parallel workers would interleave bytes from
        # different tasks on the screen. Force single-worker so the user
        # gets one coherent thought stream at a time.
        if app_config.run.max_workers != 1:
            console.print(
                "[yellow]--stream forces max_workers=1[/yellow] so thought "
                "streams don't interleave."
            )
            app_config = replace(
                app_config,
                run=replace(app_config.run, max_workers=1),
            )
    _setup_streaming(stream)
    # In batch mode the event log only renders cleanly when workers are
    # serialized — otherwise lines from concurrent tasks interleave on
    # the screen. Enable when the user opts in or when stream is on
    # (which already pins workers to 1).
    _setup_progress_logger(enabled=show_events or stream, lang=lang)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task_total = len(dataset.iter_tasks())
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


@app.command("score-run")
def score_run_command(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="artifacts/runs/<run_id> directory."),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    write_summary: bool = typer.Option(True, help="Write score_summary.json into the run directory."),
) -> None:
    """Score predictions in <run_dir> against gold CSVs using the official rubric."""
    app_config = load_app_config(config)
    summary = score_run(
        run_dir=run_dir,
        gold_root=app_config.dataset.gold_root,
        redundancy_lambda=app_config.scoring.redundancy_lambda,
        numeric_tolerance=app_config.scoring.numeric_tolerance,
        case_insensitive=app_config.scoring.case_insensitive,
        strip_whitespace=app_config.scoring.strip_whitespace,
    )

    table = Table(title=f"Scores for {run_dir.name}")
    table.add_column("Task")
    table.add_column("Matched")
    table.add_column("Pred")
    table.add_column("Gold")
    table.add_column("Recall")
    table.add_column("Penalty")
    table.add_column("Score")
    table.add_column("Note")
    for entry in summary.per_task:
        table.add_row(
            entry.task_id,
            str(entry.matched_count),
            str(entry.pred_col_count),
            str(entry.gold_col_count),
            f"{entry.recall:.3f}",
            f"{entry.penalty:.3f}",
            f"{entry.score:.3f}",
            entry.error or "",
        )
    console.print(table)
    console.print(
        f"Tasks with predictions: {summary.scored_task_count}/{summary.task_count} | "
        f"total score: {summary.total_score:.4f}/{summary.task_count} | "
        f"mean score: {summary.mean_score:.4f} | mean recall: {summary.mean_recall:.4f}"
    )

    if write_summary:
        summary_path = run_dir / "score_summary.json"
        import json as _json

        summary_path.write_text(_json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n")
        console.print(f"Score summary: {summary_path}")


def main() -> None:
    app()
