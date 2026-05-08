"""Local repair module for failed operator executions.

This module is intentionally not a general "solve the task again" path.
It receives a narrow failure packet and asks a repair model for a bounded
artifact: a replacement Python program that assigns `answer`.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.agents.tablellm_direct import (
    _read_result_csv,
    _wrap_for_execution,
    extract_python_program,
)
from data_agent_baseline.agents.task_compiler import CompiledTask
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.budget import get_budget_controller
from data_agent_baseline.tools.python_exec import execute_python_code


_REPAIR_SYSTEM = """You are a local repair module for a table/data operator.
You do not solve the original problem from scratch.
You only repair the previous Python program using the failure evidence.

Output rules:
- Output exactly one fenced ```python``` block.
- The code must assign the final result to a variable named `answer`.
- `answer` must be a pandas DataFrame, Series, scalar, list, or dict.
- Do not include prose, analysis, markdown outside the code block, or network access.
"""


def _failed_block(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("operator_executor", "tablellm_direct"):
        block = payload.get(key)
        if isinstance(block, dict):
            return block
    return payload


def _compact_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in manifest[:12]:
        compact.append({
            "path": item.get("path"),
            "kind": item.get("kind"),
            "columns": item.get("columns"),
            "row_count": item.get("row_count"),
        })
    return compact


def build_repair_prompt(
    *,
    task: PublicTask,
    compiled_task: CompiledTask,
    failed_payload: dict[str, Any],
) -> str:
    block = _failed_block(failed_payload)
    packet = {
        "task_id": task.task_id,
        "question": task.question,
        "compiled_task": compiled_task.to_dict(),
        "failure_reason": failed_payload.get("failure_reason") or block.get("failure_reason"),
        "answer_validation": failed_payload.get("answer_validation"),
        "context_manifest": _compact_manifest(list(block.get("context_manifest") or [])),
        "previous_program": block.get("program", ""),
        "exec_stdout": str(block.get("exec_stdout", ""))[-3000:],
        "exec_stderr": str(block.get("exec_stderr", ""))[-3000:],
    }
    return (
        "Repair the failed operator program. Use only the provided context paths; "
        "the working directory is the task context directory.\n\n"
        f"Failure packet JSON:\n{json.dumps(packet, ensure_ascii=False, indent=2)}"
    )


def run_reasoner_repair(
    *,
    task: PublicTask,
    model: OpenAIModelAdapter,
    compiled_task: CompiledTask,
    failed_payload: dict[str, Any],
    python_timeout: int,
) -> dict[str, Any]:
    block = _failed_block(failed_payload)
    if not block.get("program"):
        return {
            "attempted": False,
            "succeeded": False,
            "failure_reason": "repair_skipped:no_previous_program",
        }

    raw_response = model.complete(
        [
            ModelMessage(role="system", content=_REPAIR_SYSTEM),
            ModelMessage(
                role="user",
                content=build_repair_prompt(
                    task=task,
                    compiled_task=compiled_task,
                    failed_payload=failed_payload,
                ),
            ),
        ],
        temperature=0.0,
        stream_label="reasoner-repair",
    )
    program = extract_python_program(raw_response)
    if not program:
        return {
            "attempted": True,
            "succeeded": False,
            "failure_reason": "repair_empty_program",
            "raw_response": raw_response,
        }

    with tempfile.TemporaryDirectory() as result_dir:
        result_path = Path(result_dir) / "repaired_answer.csv"
        wrapped = _wrap_for_execution(program, result_path=result_path)
        budget = get_budget_controller()
        if budget is not None:
            budget.consume_tool("execute_python:repair")
        exec_result = execute_python_code(
            context_root=task.context_dir,
            code=wrapped,
            timeout_seconds=python_timeout,
        )
        stdout = str(exec_result.get("output", ""))
        stderr = str(exec_result.get("stderr", ""))
        if not exec_result.get("success"):
            return {
                "attempted": True,
                "succeeded": False,
                "failure_reason": f"repair_exec_error:{exec_result.get('error', 'unknown')}",
                "raw_response": raw_response,
                "program": program,
                "exec_stdout": stdout,
                "exec_stderr": stderr,
            }
        answer = _read_result_csv(result_path)

    if answer is None or not answer.columns:
        return {
            "attempted": True,
            "succeeded": False,
            "failure_reason": "repair_no_result_table",
            "raw_response": raw_response,
            "program": program,
            "exec_stdout": stdout,
            "exec_stderr": stderr,
        }

    return {
        "attempted": True,
        "succeeded": True,
        "failure_reason": None,
        "answer": answer.to_dict(),
        "raw_response": raw_response,
        "program": program,
        "exec_stdout": stdout,
        "exec_stderr": stderr,
    }
