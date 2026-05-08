"""Unified repair coordinator for the operator execution pipeline.

Manages all *structural* repair strategies in priority order:

1. Local repair  — deterministic AST patches, up to max_local_repairs.
2. Schema-guided LLM retry — one shot using schema_diagnostics as ground truth.

Semantic repair (plan-vs-code consistency) is NOT here.  That lives in
SemanticConsistencyPipeline because it runs only on successfully executed
programs and needs to reference the semantic plan.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.execution_context import ExecutionContext
from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.agents.tablellm_direct import (
    CodegenRunResult,
    _read_result_csv,
    _wrap_for_execution,
    extract_python_program,
    validate_python_syntax,
)
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.budget import BudgetExceeded, get_budget_controller
from data_agent_baseline.tools.python_exec import execute_python_code


_SCHEMA_RETRY_SYSTEM = (
    "You are a tool-first data operator. Repair the failed Python program "
    "using the deterministic schema scan and the execution error. Output "
    "exactly one fenced ```python``` block assigning the final result to "
    "`answer`. Do not include prose. Use only real paths, tables, columns, "
    "and join keys listed in the diagnostics; never invent schema names. "
    "Treat knowledge.md as semantic/rule hypotheses only, never as final "
    "proof that a column exists. Before using dataframe columns, write code "
    "that records actual loaded columns in debug_steps['schema_inspection'] "
    "and ground schema_mapping on those real columns. "
    "For JSON object wrappers use payload['records']; for pandas merge "
    "suffixes use explicit names such as suffixes=('_exam', '_patient') "
    "and choose the correct suffixed field explicitly; for zero-row "
    "filters inspect dtype/value normalization before changing routes. "
    "Do semantic schema linking from question concepts to real fields; "
    "do not follow fixed keyword shortcuts. If a knowledge.md rule or "
    "runtime evidence changes a prior semantic assumption, record it in "
    "debug_steps['plan_override'] with concept, field/source, old_value, "
    "new_value, and evidence."
)

_SCHEMA_RETRY_MARKERS = frozenset({
    "empty_program",
    "syntax_error",
    "static_error",
    "exec_error",
    "no_result_table",
    "request_error",
    "missing_answer",
    "no_such_column",
    "pandas_keyerror",
    "KeyError",
    "IndexError",
    "FileNotFoundError",
    "JSONDecodeError",
    "json_decode_error",
    "index_error",
    "bad_join_key",
    "no_such_table",
    "no_such_file",
    "structured_doc_synth",
})


def exec_program(
    *,
    task: PublicTask,
    program: str,
    raw_response: str,
    manifest: list[dict[str, Any]],
    python_timeout: int = 30,
    label: str = "repair",
) -> CodegenRunResult:
    """Execute a program and return a CodegenRunResult.

    Shared by RepairCoordinator and SemanticConsistencyPipeline so that
    execution logic lives in exactly one place.
    """
    syntax_error = validate_python_syntax(program)
    if syntax_error is not None:
        return CodegenRunResult(
            answer=None,
            succeeded=False,
            failure_reason=f"{label}_syntax_error:{syntax_error}",
            raw_response=raw_response,
            program=program,
            manifest=list(manifest or []),
        )

    with tempfile.TemporaryDirectory() as result_dir:
        result_path = Path(result_dir) / f"{label}_answer.csv"
        wrapped = _wrap_for_execution(program, result_path=result_path)
        budget = get_budget_controller()
        if budget is not None:
            budget.consume_tool(f"execute_python:{label}")
        exec_result = execute_python_code(
            context_root=task.context_dir,
            code=wrapped,
            timeout_seconds=python_timeout,
        )
        stdout = str(exec_result.get("output", ""))
        stderr = str(exec_result.get("stderr", ""))
        if not exec_result.get("success"):
            return CodegenRunResult(
                answer=None,
                succeeded=False,
                failure_reason=f"{label}_exec_error:{exec_result.get('error', 'unknown')}",
                raw_response=raw_response,
                program=program,
                manifest=list(manifest or []),
                exec_stdout=stdout,
                exec_stderr=stderr,
            )
        answer = _read_result_csv(result_path)

    if answer is None or not answer.columns:
        return CodegenRunResult(
            answer=None,
            succeeded=False,
            failure_reason=f"{label}_no_result_table",
            raw_response=raw_response,
            program=program,
            manifest=list(manifest or []),
            exec_stdout=stdout,
            exec_stderr=stderr,
        )
    return CodegenRunResult(
        answer=answer,
        succeeded=True,
        failure_reason=None,
        raw_response=raw_response,
        program=program,
        manifest=list(manifest or []),
        exec_stdout=stdout,
        exec_stderr=stderr,
    )


class RepairCoordinator:
    """Coordinates structural repair strategies for a failed codegen result.

    Strategy order (deterministic before LLM):
      1. Local repair  — fast AST patches, ≤ max_local_repairs iterations
      2. Schema-guided LLM retry — one LLM call, uses schema_diagnostics

    The caller must handle BudgetExceeded propagation.
    """

    def __init__(
        self,
        model: OpenAIModelAdapter,
        context: ExecutionContext,
        *,
        python_timeout: int = 30,
        max_local_repairs: int = 3,
        sample_temperature: float | None = None,
        sample_seed: int | None = None,
    ) -> None:
        self.model = model
        self.context = context
        self.python_timeout = python_timeout
        self.max_local_repairs = max_local_repairs
        self.sample_temperature = sample_temperature
        self.sample_seed = sample_seed

    def run(self, task: PublicTask, result: CodegenRunResult) -> CodegenRunResult:
        """Local repair then schema-retry in one call (convenience for simple cases)."""
        result, repair_log = self.local_repair_loop(task, result)
        if repair_log:
            result.manifest = list(result.manifest or []) + [{"local_repair_log": repair_log}]
        if result.succeeded:
            return result
        return self.schema_retry(task, result)

    # ------------------------------------------------------------------
    # Public stage methods — OperatorExecutor uses these directly when it
    # needs to interleave synthesis between local repair and schema-retry.
    # ------------------------------------------------------------------

    def local_repair_loop(
        self,
        task: PublicTask,
        result: CodegenRunResult,
    ) -> tuple[CodegenRunResult, list[dict[str, Any]]]:
        from data_agent_baseline.agents.local_repair import (
            issues_from_exec_error,
            repair_answer_table,
            try_program_repair,
        )
        from data_agent_baseline.agents.static_checker import (
            StaticIssue,
            check_program,
            has_blocking,
        )
        from data_agent_baseline.benchmark.schema import AnswerTable
        from data_agent_baseline.eval.answer_validator import validate_answer_table
        from data_agent_baseline.progress import get_progress_logger

        compiled = self.context.compiled_task
        repair_log: list[dict[str, Any]] = []
        logger = get_progress_logger()

        for attempt in range(self.max_local_repairs):
            budget = get_budget_controller()
            if budget is not None and not budget.can_local_repair():
                repair_log.append({
                    "attempt": attempt,
                    "stage": "budget",
                    "outcome": {"action": "skipped", "notes": ["max_local_repairs reached"]},
                })
                break

            if result.succeeded and result.answer is not None:
                # --- zero-row check ---
                if not result.answer.rows and result.program:
                    issues = [StaticIssue(
                        code="zero_rows",
                        severity="error",
                        message="Program executed successfully but produced zero answer rows.",
                        repair_hint=(
                            "inspect filters for dtype/value mismatch before cascading; "
                            "normalize booleans/dates where possible"
                        ),
                    )]
                    if logger:
                        logger.codegen_debug(debug={
                            "repair_attempt": attempt,
                            "static_or_runtime_issues": [it.to_dict() for it in issues],
                        })
                    outcome = try_program_repair(
                        program=result.program,
                        issues=issues,
                        compiled=compiled,
                    )
                    if outcome is not None:
                        if budget is not None:
                            budget.consume_local_repair(outcome.action)
                        new_result = exec_program(
                            task=task,
                            program=outcome.patched_program,
                            raw_response=result.raw_response,
                            manifest=result.manifest,
                            python_timeout=self.python_timeout,
                            label="local_repair",
                        )
                        repair_log.append({
                            "attempt": attempt,
                            "stage": "zero_row_answer",
                            "issues": [it.to_dict() for it in issues],
                            "outcome": outcome.to_dict(),
                            "exec_succeeded": bool(new_result.succeeded),
                            "post_failure_reason": new_result.failure_reason,
                        })
                        result = new_result
                        continue

                # --- answer table validation ---
                validation = validate_answer_table(result.answer)
                if validation.valid:
                    break
                outcome = repair_answer_table(
                    answer=result.answer.to_dict(), validation=validation
                )
                if outcome is None or outcome.patched_answer is None:
                    break
                result.answer = AnswerTable(
                    columns=list(outcome.patched_answer.get("columns") or []),
                    rows=[list(r) for r in (outcome.patched_answer.get("rows") or [])],
                )
                repair_log.append({
                    "attempt": attempt,
                    "stage": "answer_validator",
                    "outcome": outcome.to_dict(),
                })
                continue

            # --- failure path: static check + exec error ---
            program = result.program or ""
            issues = check_program(program, capabilities=compiled.source_capabilities)
            issues.extend(issues_from_exec_error(result.exec_stderr))
            issues.extend(issues_from_exec_error(str(result.failure_reason or "")))

            if not issues or not has_blocking(issues):
                break

            if logger:
                logger.codegen_debug(debug={
                    "repair_attempt": attempt,
                    "static_or_runtime_issues": [it.to_dict() for it in issues],
                })

            outcome = try_program_repair(
                program=program, issues=issues, compiled=compiled
            )
            if outcome is None:
                break

            if budget is not None:
                budget.consume_local_repair(outcome.action)

            new_result = exec_program(
                task=task,
                program=outcome.patched_program,
                raw_response=result.raw_response,
                manifest=result.manifest,
                python_timeout=self.python_timeout,
                label="local_repair",
            )
            repair_log.append({
                "attempt": attempt,
                "stage": "static_or_exec_error",
                "issues": [it.to_dict() for it in issues],
                "outcome": outcome.to_dict(),
                "exec_succeeded": bool(new_result.succeeded),
                "post_failure_reason": new_result.failure_reason,
            })
            result = new_result

        return result, repair_log

    # ------------------------------------------------------------------
    # Private: schema-guided LLM retry
    # ------------------------------------------------------------------

    def schema_retry(
        self,
        task: PublicTask,
        failed: CodegenRunResult,
    ) -> CodegenRunResult:
        failure_str = str(failed.failure_reason or "")
        combined_failure = "\n".join([
            failure_str,
            str(failed.exec_stderr or "")[-3000:],
            str(failed.exec_stdout or "")[-1500:],
        ])
        combined_lower = combined_failure.lower()
        if not any(m.lower() in combined_lower for m in _SCHEMA_RETRY_MARKERS):
            return failed

        ctx = self.context
        prior_static_issues: list[dict[str, Any]] = []
        for item in failed.manifest or []:
            if isinstance(item, dict) and isinstance(item.get("static_issues"), list):
                prior_static_issues.extend(item["static_issues"])
            if isinstance(item, dict) and isinstance(item.get("local_repair_log"), list):
                for entry in item["local_repair_log"]:
                    if isinstance(entry, dict) and isinstance(entry.get("issues"), list):
                        prior_static_issues.extend(entry["issues"])

        packet = {
            "question": task.question,
            "compiled_task": ctx.compiled_task.to_dict(),
            "failure_reason": failed.failure_reason,
            "static_issues": prior_static_issues[-12:],
            "previous_program": failed.program,
            "exec_stdout": (failed.exec_stdout or "")[-3000:],
            "exec_stderr": (failed.exec_stderr or "")[-3000:],
            "schema_scan": ctx.schema_scan,
            "schema_diagnostics": ctx.schema_diagnostics,
            "runtime_policy": {
                "hard_errors_must_repair": [
                    "KeyError",
                    "IndexError",
                    "FileNotFoundError",
                    "JSONDecodeError",
                ],
                "schema_first_contract": (
                    "Every loaded dataframe/result table must have its real "
                    "columns recorded in debug_steps['schema_inspection']; "
                    "debug_steps['schema_mapping'] must choose fields from "
                    "that real inspection, not from knowledge.md wording alone."
                ),
                "knowledge_policy": (
                    "knowledge.md supplies rules and hypotheses; it does not "
                    "prove that a dataframe column/table/key exists."
                ),
            },
        }

        try:
            raw_response = self.model.complete(
                [
                    ModelMessage(role="system", content=_SCHEMA_RETRY_SYSTEM),
                    ModelMessage(
                        role="user",
                        content=(
                            "Repair locally. Use the schema_scan as ground truth. "
                            "If a previous column/key is absent, map it to a real field "
                            "from schema_diagnostics or rewrite the operation. "
                            "Start from actual schema inspection in code: after every "
                            "load/query, write debug_steps['schema_inspection'][var] = "
                            "list(df.columns), then choose fields from that inspection. "
                            "Knowledge rules may guide semantics but are never final "
                            "schema. If a rule correction changes a prior semantic "
                            "assumption, write debug_steps['plan_override'] so the "
                            "consistency judge compares against the corrected plan. "
                            "The working directory is the task context directory.\n\n"
                            f"{json.dumps(packet, ensure_ascii=False, indent=2)}"
                        ),
                    ),
                ],
                temperature=self.sample_temperature,
                seed=self.sample_seed,
                stream_label="operator schema-retry",
                max_tokens=max(int(getattr(self.model, "max_tokens", 0) or 0), 2048),
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            failed.failure_reason = f"{failure_str} | schema_retry_runtime_error:{exc}"
            return failed

        program = extract_python_program(raw_response)
        if not program:
            failed.failure_reason = f"{failure_str} | schema_retry_empty_program"
            return failed

        budget = get_budget_controller()
        if budget is not None:
            budget.consume_tool("execute_python:schema_retry")

        return exec_program(
            task=task,
            program=program,
            raw_response=raw_response,
            manifest=[{"schema_diagnostics": ctx.schema_diagnostics}],
            python_timeout=self.python_timeout,
            label="schema_retry",
        )
