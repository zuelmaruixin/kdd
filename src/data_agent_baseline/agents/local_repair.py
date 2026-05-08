"""Deterministic local repairs for failed operator runs.

Philosophy (matches the project's "能局部修的不整题重跑" principle):
the LLM is expensive, slow, and stochastic — for the failure modes we
can recognize from a static check, an exec error, or a validator
verdict, we apply a fixed code transformation and retry **without**
calling the LLM. Only when none of the deterministic actions apply
do we fall through to ``reasoner_repair``.

Each repair function:
- takes the previous program (string) + structured failure evidence,
- returns a new program string + a description of what was patched, or
- returns ``None`` if the failure is outside its remit.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.agents.static_checker import StaticIssue
from data_agent_baseline.agents.task_compiler import CompiledTask
from data_agent_baseline.eval.answer_validator import ValidationResult


@dataclass(slots=True)
class LocalRepairOutcome:
    """Result of applying one local repair."""

    succeeded: bool
    patched_program: str = ""
    patched_answer: dict[str, Any] | None = None
    action: str = ""                              # which fixer ran
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "action": self.action,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Static-check based repairs (program patches)
# ---------------------------------------------------------------------------


_FENCE_MARKER_LINE_RE = re.compile(r"^\s*```(?:python|py)?\s*$", re.IGNORECASE)


def _strip_markdown_fence_lines(program: str) -> str:
    lines = program.strip().splitlines()
    while lines and _FENCE_MARKER_LINE_RE.match(lines[0]):
        lines.pop(0)
    while lines and _FENCE_MARKER_LINE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def repair_python_syntax(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Fast deterministic repair for markdown fence leakage.

    This handles the common malformed output where the model includes an
    opening ```python line but no closing fence, causing Python execution
    to fail before any task logic runs.
    """
    if not any(it.code == "python_syntax" for it in issues):
        return None
    patched = _strip_markdown_fence_lines(program)
    if patched == program:
        return None
    try:
        ast.parse(patched)
    except SyntaxError:
        return None
    return LocalRepairOutcome(
        succeeded=True,
        patched_program=patched,
        action="strip_markdown_code_fence",
        notes=["removed markdown code-fence marker lines before execution"],
    )


def _comment_out_lines(program: str, *, line_numbers: set[int]) -> str:
    """Comment out the specified 1-indexed source lines."""
    if not line_numbers:
        return program
    out_lines: list[str] = []
    for idx, line in enumerate(program.splitlines(), start=1):
        if idx in line_numbers:
            out_lines.append(f"# AUTO-PATCH: removed by local_repair (line {idx}): {line}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def repair_no_such_table(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Drop SQL strings that reference non-existent tables.

    For a ``mixed_context`` task, the canonical fix is: stop trying to
    JOIN tables that live in different sources, and instead orchestrate
    a doc-extract → sql query → json/pandas lookup pipeline. We can't
    rewrite the orchestration deterministically, but we *can* prune
    the offending SQL strings and signal the executor to ask the
    reasoner for an orchestrated rewrite.
    """
    target = [it for it in issues if it.code == "no_such_table"]
    if not target:
        return None

    # Collect every line containing a no_such_table SQL literal.
    bad_lines: set[int] = set()
    notes: list[str] = []
    for issue in target:
        line = issue.location.get("line")
        if isinstance(line, int):
            bad_lines.add(line)
        notes.append(
            f"removed SQL referencing missing table {issue.location.get('table')!r}"
        )
    if not bad_lines:
        return None

    patched = _comment_out_lines(program, line_numbers=bad_lines)
    return LocalRepairOutcome(
        succeeded=True,
        patched_program=patched,
        action="drop_sql_lines_with_unknown_table",
        notes=notes,
    )


def repair_no_such_file(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Substitute the offending literal with the closest known path.

    "Closest" = same suffix + smallest Levenshtein distance. If we can't
    find any candidate of the same suffix, we just fall through (no
    repair possible).
    """
    target = [it for it in issues if it.code == "no_such_file"]
    if not target:
        return None

    known_paths = [cap.path for cap in compiled.source_capabilities]
    if not known_paths:
        return None

    patched = program
    notes: list[str] = []
    for issue in target:
        bad = issue.location.get("literal")
        if not isinstance(bad, str):
            continue
        suffix = bad.rsplit(".", 1)[-1].lower() if "." in bad else ""
        candidates = [
            p for p in known_paths if (not suffix or p.lower().endswith("." + suffix))
        ] or known_paths
        best = min(
            candidates,
            key=lambda candidate: _levenshtein(candidate, bad),
        )
        if best == bad:
            continue
        # Replace exact occurrences of the bad literal in the program.
        patched_new = patched.replace(repr(bad), repr(best)).replace(f"'{bad}'", f"'{best}'").replace(f'"{bad}"', f'"{best}"')
        if patched_new != patched:
            notes.append(f"path {bad!r} → {best!r}")
            patched = patched_new

    if not notes:
        return None
    return LocalRepairOutcome(
        succeeded=True,
        patched_program=patched,
        action="substitute_known_path",
        notes=notes,
    )


def repair_json_records_read_with_pandas(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Rewrite ``df = pd.read_json('x.json')`` for object-wrapped records."""
    target = [it for it in issues if it.code == "json_records_read_with_pandas"]
    if not target:
        return None

    lines = program.splitlines()
    changed = False
    notes: list[str] = []
    for issue in sorted(target, key=lambda it: int(it.location.get("line") or 0), reverse=True):
        line_no = issue.location.get("line")
        path = issue.location.get("literal")
        record_key = issue.location.get("record_key") or "records"
        if not isinstance(line_no, int) or not isinstance(path, str):
            continue
        idx = line_no - 1
        if idx < 0 or idx >= len(lines):
            continue
        line = lines[idx]
        match = re.match(
            r"^(?P<indent>\s*)(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:pd|pandas)\.read_json\((?P<quote>['\"])(?P<path>[^'\"]+)(?P=quote)\)\s*$",
            line,
        )
        if not match or match.group("path") != path:
            continue
        indent = match.group("indent")
        var = match.group("var")
        payload_var = f"_{var}_json_payload"
        lines[idx:idx + 1] = [
            f"{indent}with open({path!r}, 'r') as _json_f:",
            f"{indent}    {payload_var} = __import__('json').load(_json_f)",
            (
                f"{indent}{var} = pd.DataFrame("
                f"{payload_var}.get({str(record_key)!r}, {payload_var} if isinstance({payload_var}, list) else []))"
            ),
        ]
        changed = True
        notes.append(f"{var} read from {path!r} via json.load(...)[{record_key!r}]")

    if not changed:
        return None
    patched = "\n".join(lines)
    try:
        ast.parse(patched)
    except SyntaxError:
        return None
    return LocalRepairOutcome(
        succeeded=True,
        patched_program=patched,
        action="rewrite_json_records_loader",
        notes=notes,
    )


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Exec-error based repairs
# ---------------------------------------------------------------------------


_NO_SUCH_TABLE_RE = re.compile(r"no such table:\s*(\S+)", re.IGNORECASE)
_NO_SUCH_COLUMN_RE = re.compile(r"no such column:\s*(\S+)", re.IGNORECASE)
_KEYERROR_RE = re.compile(r"KeyError:\s*['\"]?([^'\"\n]+)['\"]?")
_NOT_IN_INDEX_RE = re.compile(r"\[([^\]]+)\]\s+not in index", re.IGNORECASE)
_NONE_OF_INDEX_RE = re.compile(r"None of \[Index\(\[([^\]]+)\]", re.IGNORECASE)
_FILE_NOT_FOUND_RE = re.compile(r"FileNotFoundError.+?'([^']+)'", re.DOTALL)
_INDEX_ERROR_RE = re.compile(
    r"(IndexError|single positional indexer is out-of-bounds|index \d+ is out of bounds)",
    re.IGNORECASE,
)
_JSON_DECODE_RE = re.compile(
    r"(JSONDecodeError|json\.decoder\.JSONDecodeError|Expecting value|Extra data)",
    re.IGNORECASE,
)
_MISSING_ANSWER_RE = re.compile(r"did not define an `answer` variable", re.IGNORECASE)
_MERGE_DTYPE_RE = re.compile(
    r"merge on .* columns for key ['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


def issues_from_exec_error(error_text: str) -> list[StaticIssue]:
    """Translate runtime error text into ``StaticIssue``-shaped codes
    so the same dispatch logic handles them.
    """
    issues: list[StaticIssue] = []
    if not error_text:
        return issues

    if _MISSING_ANSWER_RE.search(error_text):
        issues.append(StaticIssue(
            code="missing_answer_assignment",
            severity="error",
            message="Operator program did not define an `answer` variable.",
            repair_hint="append `answer = <final_dataframe_or_scalar>`",
        ))

    if match := _MERGE_DTYPE_RE.search(error_text):
        issues.append(StaticIssue(
            code="merge_dtype_mismatch",
            severity="error",
            message=match.group(0),
            location={"key": match.group(1)},
            repair_hint="cast both merge-key columns to string before merging",
        ))

    if match := _NO_SUCH_TABLE_RE.search(error_text):
        issues.append(StaticIssue(
            code="no_such_table",
            severity="error",
            message=match.group(0),
            location={"table": match.group(1)},
            repair_hint="drop the SQL JOIN to that table and orchestrate the lookup separately",
        ))
    if match := _NO_SUCH_COLUMN_RE.search(error_text):
        issues.append(StaticIssue(
            code="no_such_column",
            severity="error",
            message=match.group(0),
            location={"column": match.group(1)},
            repair_hint="fuzzy-match the column name against the source schema",
        ))
    if match := _KEYERROR_RE.search(error_text):
        issues.append(StaticIssue(
            code="pandas_keyerror",
            severity="error",
            message=match.group(0),
            location={"key": match.group(1)},
            repair_hint="fuzzy-match the column name against the dataframe columns",
        ))
    for pattern in (_NOT_IN_INDEX_RE, _NONE_OF_INDEX_RE):
        if match := pattern.search(error_text):
            raw_items = match.group(1)
            names = re.findall(r"['\"]([^'\"]+)['\"]", raw_items)
            for name in names or [raw_items.strip().strip("'\"")]:
                if not name:
                    continue
                issues.append(StaticIssue(
                    code="pandas_keyerror",
                    severity="error",
                    message=f"{name!r} not in dataframe columns/index",
                    location={"key": name},
                    repair_hint=(
                        "inspect dataframe columns after merge/filter; handle pandas "
                        "suffixes such as _x/_y or choose a real source field"
                    ),
                ))
    if match := _FILE_NOT_FOUND_RE.search(error_text):
        issues.append(StaticIssue(
            code="no_such_file",
            severity="error",
            message=match.group(0),
            location={"literal": match.group(1)},
            repair_hint="substitute with a known context path",
        ))
    if match := _INDEX_ERROR_RE.search(error_text):
        issues.append(StaticIssue(
            code="index_error",
            severity="error",
            message=match.group(0),
            repair_hint=(
                "inspect intermediate row counts before indexing; guard empty "
                "filters and repair the schema/filter mapping"
            ),
        ))
    if match := _JSON_DECODE_RE.search(error_text):
        issues.append(StaticIssue(
            code="json_decode_error",
            severity="error",
            message=match.group(0),
            repair_hint=(
                "inspect the actual file format and loader; use json.load only "
                "for valid JSON and pandas/read_csv or text parsing otherwise"
            ),
        ))
    return issues


def repair_pandas_keyerror_or_no_such_column(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Fuzzy-match a missing column/key to a real one and rewrite the literal."""
    target = [
        it for it in issues
        if it.code in {"no_such_column", "pandas_keyerror"}
    ]
    if not target:
        return None

    # Build a flat list of all known column-like names. Static checker
    # issues may also carry dataframe-specific available_columns; prefer
    # those so we don't replace a Patient column with an Examination one.
    known_columns: list[str] = []
    for cap in compiled.source_capabilities:
        known_columns.extend(cap.columns)
        known_columns.extend(cap.json_record_fields)
        for table in cap.tables:
            for col in table.get("columns") or []:
                if isinstance(col, dict) and "name" in col:
                    known_columns.append(col["name"])
    known_columns = list(dict.fromkeys(known_columns))  # de-dup, preserve order
    if not known_columns:
        return None

    patched = program
    notes: list[str] = []
    for issue in target:
        bad = issue.location.get("column") or issue.location.get("key")
        if not isinstance(bad, str) or not bad:
            continue
        bad_clean = bad.strip().strip("'\"")
        scoped_columns = issue.location.get("available_columns")
        candidates = (
            [str(c) for c in scoped_columns if str(c)]
            if isinstance(scoped_columns, list)
            else known_columns
        )
        if not candidates:
            continue
        scoped_matches = issue.location.get("closest_matches")
        if isinstance(scoped_matches, list):
            scoped_matches = [str(c) for c in scoped_matches if str(c)]
            suffix_matches = _suffix_column_matches(bad_clean, candidates)
            if suffix_matches:
                scoped_matches = suffix_matches
            if len(scoped_matches) != 1:
                # Ambiguous schema mapping (e.g. link_to_event vs
                # link_to_member/link_to_budget) should be rewritten with
                # schema diagnostics, not guessed locally.
                continue
            candidates = scoped_matches

        best = min(candidates, key=lambda c: _levenshtein(c.lower(), bad_clean.lower()))
        confidence = difflib.SequenceMatcher(
            None,
            re.sub(r"[^a-z0-9]+", "", bad_clean.lower()),
            re.sub(r"[^a-z0-9]+", "", best.lower()),
        ).ratio()
        if confidence < 0.68:
            continue
        if best.lower() == bad_clean.lower():
            continue
        # Rewrite both quoted occurrences and SQL bareword occurrences.
        patched_new = patched
        for needle in (f"'{bad_clean}'", f'"{bad_clean}"'):
            patched_new = patched_new.replace(needle, f'"{best}"')
        # SQL barewords (rare but possible with `SELECT bad FROM ...`).
        patched_new = re.sub(
            rf"\b{re.escape(bad_clean)}\b",
            best,
            patched_new,
        )
        if patched_new != patched:
            notes.append(f"column {bad_clean!r} → {best!r}")
            patched = patched_new
    if not notes:
        return None
    return LocalRepairOutcome(
        succeeded=True,
        patched_program=patched,
        action="fuzzy_match_column_name",
        notes=notes,
    )


def _suffix_column_matches(bare_column: str, candidates: list[str]) -> list[str]:
    """Pick a deterministic suffixed version for a bare post-merge column.

    Generated pandas code often merges two sources that both contain
    Diagnosis, then projects bare "Diagnosis". Static checker sees only
    Diagnosis_x/Diagnosis_y or explicit suffixes. Prefer patient/right-side
    suffixes for patient-level descriptors, otherwise only repair when
    there is one unambiguous suffixed candidate.
    """
    suffixed = [
        c for c in candidates
        if c.startswith(f"{bare_column}_")
    ]
    if not suffixed:
        return []
    if len(suffixed) == 1:
        return suffixed
    lower_map = {c.lower(): c for c in suffixed}
    for suffix in ("_patient", "_right", "_y", "_exam", "_left", "_x"):
        key = f"{bare_column}{suffix}".lower()
        if key in lower_map:
            return [lower_map[key]]
    return []


def repair_bad_join_key(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Patch a merge key only when the checker found an unambiguous key pair."""
    target = [it for it in issues if it.code == "bad_join_key"]
    if not target:
        return None

    patched = program
    notes: list[str] = []
    for issue in target:
        bad = issue.location.get("key")
        if not isinstance(bad, str) or not bad:
            continue
        candidates = issue.location.get("candidate_join_keys")
        if not isinstance(candidates, list) or len(candidates) != 1:
            # Multiple possible joins need schema-aware rewrite, not blind
            # string replacement.
            continue
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            continue
        left_on = candidate.get("left_on")
        right_on = candidate.get("right_on")
        if not isinstance(left_on, str) or not isinstance(right_on, str):
            continue
        if left_on != right_on:
            # on='x' cannot represent asymmetric names safely; schema-retry
            # should rewrite to left_on/right_on.
            continue
        patched_new = patched
        for needle in (f"'{bad}'", f'"{bad}"'):
            patched_new = patched_new.replace(needle, f'"{left_on}"')
        if patched_new != patched:
            notes.append(f"join key {bad!r} → {left_on!r}")
            patched = patched_new

    if not notes:
        return None
    return LocalRepairOutcome(
        succeeded=True,
        patched_program=patched,
        action="substitute_unambiguous_join_key",
        notes=notes,
    )


def repair_merge_dtype_mismatch(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Insert ``astype(str)`` casts before pandas merge key dtype failures."""
    if not any(it.code == "merge_dtype_mismatch" for it in issues):
        return None
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return None

    patches: dict[int, list[str]] = {}
    notes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue

        left_var = right_var = left_on = right_on = None
        line_no = getattr(node, "lineno", None)
        if node.func.attr == "merge":
            # pd.merge(left, right, ...) OR left.merge(right, ...)
            if isinstance(node.func.value, ast.Name) and node.func.value.id in {"pd", "pandas"}:
                if len(node.args) >= 2:
                    left_var = node.args[0].id if isinstance(node.args[0], ast.Name) else None
                    right_var = node.args[1].id if isinstance(node.args[1], ast.Name) else None
            elif isinstance(node.func.value, ast.Name) and node.args:
                left_var = node.func.value.id
                right_var = node.args[0].id if isinstance(node.args[0], ast.Name) else None

            on = None
            for kw in node.keywords:
                if kw.arg == "left_on" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    left_on = kw.value.value
                elif kw.arg == "right_on" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    right_on = kw.value.value
                elif kw.arg == "on" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    on = kw.value.value
            if on:
                left_on = right_on = on

        if not (line_no and left_var and right_var and left_on and right_on):
            continue

        def _cast_expr(var: str, col: str) -> str:
            base = f"{var}[{col!r}].astype(str)"
            if "id" in col.lower():
                return base + r".str.replace(r'\.0$', '', regex=True).str.strip()"
            return base

        patches.setdefault(line_no, []).extend([
            f"{left_var}[{left_on!r}] = {_cast_expr(left_var, left_on)}",
            f"{right_var}[{right_on!r}] = {_cast_expr(right_var, right_on)}",
        ])
        if "id" in f"{left_on} {right_on}".lower():
            notes.append(
                f"cast and normalized ID merge keys {left_var}.{left_on} and {right_var}.{right_on}"
            )
        else:
            notes.append(f"cast merge keys {left_var}.{left_on} and {right_var}.{right_on} to str")

    if not patches:
        return None
    out: list[str] = []
    for idx, line in enumerate(program.splitlines(), start=1):
        if idx in patches:
            indent = re.match(r"^\s*", line).group(0)
            out.extend(indent + patch for patch in patches[idx])
        out.append(line)
    patched = "\n".join(out)
    try:
        ast.parse(patched)
    except SyntaxError:
        return None
    return LocalRepairOutcome(
        succeeded=True,
        patched_program=patched,
        action="cast_merge_keys_to_string",
        notes=notes,
    )


def repair_zero_row_common_filters(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Patch common filters that produce empty answers due to dtype drift."""
    if not any(it.code == "zero_rows" for it in issues):
        return None

    patched = program
    notes: list[str] = []

    try:
        tree = ast.parse(program)
    except SyntaxError:
        tree = None

    # Common cross-source ID bug:
    #   float-like IDs from one source -> "163109.0"
    #   integer/string IDs from another -> "163109"
    # After a string-cast merge this yields zero rows. Normalize the
    # generated assignment in place before rerunning.
    id_cast_re = re.compile(
        r"("
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\[\s*['\"][^'\"]*ID[^'\"]*['\"]\s*\]"
        r"\s*=\s*"
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\[\s*['\"][^'\"]*ID[^'\"]*['\"]\s*\]"
        r"\s*\.astype\s*\(\s*(?:str|['\"]str['\"])\s*\)"
        r")"
    )

    def _patch_id_cast(match: re.Match[str]) -> str:
        expr = match.group(1)
        if ".str.replace" in expr:
            return expr
        return expr + r".str.replace(r'\.0$', '', regex=True).str.strip()"

    patched_new = id_cast_re.sub(_patch_id_cast, patched)
    if patched_new != patched:
        patched = patched_new
        notes.append("normalized string-cast ID keys by stripping trailing .0 before merge")

    # If the program never wrote explicit ID normalization, add it directly
    # before pandas merge calls on ID-like keys. This catches the common
    # zero-row shape where both sources have the right IDs but one side was
    # rendered as a float-looking string.
    if tree is not None:
        patches: dict[int, list[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue

            left_var = right_var = left_on = right_on = None
            line_no = getattr(node, "lineno", None)
            if node.func.attr == "merge":
                if isinstance(node.func.value, ast.Name) and node.func.value.id in {"pd", "pandas"}:
                    if len(node.args) >= 2:
                        left_var = node.args[0].id if isinstance(node.args[0], ast.Name) else None
                        right_var = node.args[1].id if isinstance(node.args[1], ast.Name) else None
                elif isinstance(node.func.value, ast.Name) and node.args:
                    left_var = node.func.value.id
                    right_var = node.args[0].id if isinstance(node.args[0], ast.Name) else None

                on = None
                for kw in node.keywords:
                    if kw.arg == "left_on" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        left_on = kw.value.value
                    elif kw.arg == "right_on" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        right_on = kw.value.value
                    elif kw.arg == "on" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        on = kw.value.value
                if on:
                    left_on = right_on = on

            if not (line_no and left_var and right_var and left_on and right_on):
                continue
            if "id" not in f"{left_on} {right_on}".lower():
                continue
            patches.setdefault(line_no, []).extend([
                f"{left_var}[{left_on!r}] = {left_var}[{left_on!r}].astype(str).str.replace(r'\\.0$', '', regex=True).str.strip()",
                f"{right_var}[{right_on!r}] = {right_var}[{right_on!r}].astype(str).str.replace(r'\\.0$', '', regex=True).str.strip()",
            ])

        if patches:
            out: list[str] = []
            for idx, line in enumerate(patched.splitlines(), start=1):
                if idx in patches:
                    indent = re.match(r"^\s*", line).group(0)
                    out.extend(indent + patch for patch in patches[idx])
                out.append(line)
            patched_new = "\n".join(out)
            if patched_new != patched:
                patched = patched_new
                notes.append("inserted ID key normalization immediately before merge")

    patched_new = re.sub(
        r"(\b[A-Za-z_][A-Za-z0-9_]*\s*\[\s*['\"]approved['\"]\s*\])\s*==\s*['\"]true['\"]",
        r"\1.astype(str).str.lower().eq('true')",
        patched,
        flags=re.IGNORECASE,
    )
    if patched_new != patched:
        patched = patched_new
        notes.append("normalized approved == 'true' to string/bool-safe comparison")

    if not notes:
        return None
    try:
        ast.parse(patched)
    except SyntaxError:
        return None
    return LocalRepairOutcome(
        succeeded=True,
        patched_program=patched,
        action="zero_row_filter_normalize",
        notes=notes,
    )


def repair_missing_answer_assignment(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Append ``answer = <last useful variable>`` for protocol-only failures."""
    if not any(it.code == "missing_answer_assignment" for it in issues):
        return None
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return None

    candidates: list[str] = []
    for node in tree.body:
        target_name: str | None = None
        if isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
        if not target_name:
            continue
        if target_name.startswith("_") or target_name in {
            "debug_steps", "conn", "cursor", "engine", "reader",
        }:
            continue
        candidates.append(target_name)

    if not candidates:
        return None

    preferred_order = (
        "answer_df", "final_answer", "final_df", "result_df", "result",
        "output", "out", "ratio", "count", "total", "value", "df",
    )
    chosen = None
    seen = set(candidates)
    for name in preferred_order:
        if name in seen:
            chosen = name
            break
    if chosen is None:
        chosen = candidates[-1]

    patched = (
        program.rstrip()
        + "\n\n# AUTO-PATCH: execution protocol requires final variable named `answer`.\n"
        + f"answer = {chosen}\n"
    )
    try:
        ast.parse(patched)
    except SyntaxError:
        return None
    return LocalRepairOutcome(
        succeeded=True,
        patched_program=patched,
        action="append_missing_answer_assignment",
        notes=[f"set answer = {chosen}"],
    )


# ---------------------------------------------------------------------------
# Validator-result based repairs (operate on the answer dict, not program)
# ---------------------------------------------------------------------------


def repair_answer_table(
    *,
    answer: dict[str, Any] | None,
    validation: ValidationResult,
) -> LocalRepairOutcome | None:
    """Patch an answer dict for the validation issues we can fix without re-running.

    Handles:
    - ragged_row → pad / truncate every row to header width
    - empty_column → drop the all-empty columns
    """
    if not isinstance(answer, dict) or not validation.errors:
        return None

    columns = list(answer.get("columns") or [])
    rows = [list(r) for r in (answer.get("rows") or [])]

    notes: list[str] = []
    changed = False

    # 1) ragged rows → pad / truncate.
    if any(it.code == "ragged_row" for it in validation.errors) and columns:
        width = len(columns)
        new_rows: list[list[Any]] = []
        for row in rows:
            if len(row) < width:
                new_rows.append(list(row) + [None] * (width - len(row)))
            elif len(row) > width:
                new_rows.append(list(row[:width]))
            else:
                new_rows.append(list(row))
        if new_rows != rows:
            rows = new_rows
            notes.append("padded/truncated ragged rows to header width")
            changed = True

    # 2) drop entirely-empty columns.
    empty_indices: list[int] = []
    for it in validation.errors:
        if it.code == "empty_column":
            idx = it.location.get("column_index")
            if isinstance(idx, int) and 0 <= idx < len(columns):
                empty_indices.append(idx)
    if empty_indices:
        keep = [i for i in range(len(columns)) if i not in set(empty_indices)]
        if keep and keep != list(range(len(columns))):
            columns = [columns[i] for i in keep]
            rows = [[r[i] for i in keep if i < len(r)] for r in rows]
            notes.append(f"dropped empty columns at indices {empty_indices}")
            changed = True

    if not changed:
        return None
    return LocalRepairOutcome(
        succeeded=True,
        patched_answer={"columns": columns, "rows": rows},
        action="answer_table_normalize",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def try_program_repair(
    *,
    program: str,
    issues: list[StaticIssue],
    compiled: CompiledTask,
) -> LocalRepairOutcome | None:
    """Apply program-level fixers in order of safety. First one that
    returns ``succeeded=True`` wins."""
    for fixer in (
        repair_python_syntax,
        repair_missing_answer_assignment,
        repair_json_records_read_with_pandas,
        repair_no_such_table,
        repair_no_such_file,
        repair_pandas_keyerror_or_no_such_column,
        repair_bad_join_key,
        repair_merge_dtype_mismatch,
        repair_zero_row_common_filters,
    ):
        outcome = fixer(program=program, issues=issues, compiled=compiled)
        if outcome is not None and outcome.succeeded:
            return outcome
    return None
