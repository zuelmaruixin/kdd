"""Static check on a generated Python program *before* execution.

Catches the most common cause of wasted LLM rounds: codegen wrote SQL /
file paths / pandas columns / join keys that don't actually exist in the
task's context. We detect these by walking the AST + scraping SQL
strings, then comparing against the deterministic
``CompiledTask.source_capabilities`` ground truth.

The output is a list of structured issues. The local repair kit
(``agents/local_repair.py``) maps each issue code to a deterministic
patch action — no LLM round required for the obvious cases.

This is intentionally narrow. We only catch issues where we can bind a
program variable to a concrete source capability and therefore know the
referenced field/table/path is impossible. The repair layer can then
patch obvious protocol mistakes locally or ask for a schema-aware rewrite
with high-signal diagnostics.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.agents.task_compiler import CompiledTask, SourceCapability


@dataclass(frozen=True, slots=True)
class StaticIssue:
    """One thing the static checker found wrong with a generated program."""

    code: str            # no_such_file | no_such_table | no_such_column | bad_join_key | python_syntax
    severity: str        # "error" | "warning"
    message: str
    location: dict[str, Any] = field(default_factory=dict)
    repair_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "location": dict(self.location),
            "repair_hint": self.repair_hint,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FROM_OR_JOIN_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([\"`\[]?)([A-Za-z_][A-Za-z0-9_\.]*)\1",
    re.IGNORECASE,
)
_SELECT_RE = re.compile(r"\bSELECT\b", re.IGNORECASE)


def _capability_index(capabilities: list[SourceCapability]) -> dict[str, SourceCapability]:
    """Map relative path → SourceCapability for fast lookup."""
    index: dict[str, SourceCapability] = {}
    for cap in capabilities:
        index[cap.path] = cap
        index[cap.path.lower()] = cap
    return index


def _known_paths(capabilities: list[SourceCapability]) -> set[str]:
    return {cap.path for cap in capabilities}


def _known_table_names(capabilities: list[SourceCapability]) -> dict[str, str]:
    """Return ``{table_name_lower: source_path}`` across all sqlite caps."""
    out: dict[str, str] = {}
    for cap in capabilities:
        if cap.kind not in {"db", "sqlite", "sqlite3"}:
            continue
        for table in cap.tables:
            name = str(table.get("table") or "").strip()
            if name:
                out[name.lower()] = cap.path
    return out


def _columns_for_capability(cap: SourceCapability) -> list[str]:
    """Return all column-like names that are real for one source."""
    out: list[str] = []
    out.extend(cap.columns)
    out.extend(cap.json_record_fields)
    out.extend(cap.structured_record_fields)
    for table in cap.tables:
        for col in table.get("columns") or []:
            if isinstance(col, dict) and col.get("name"):
                out.append(str(col["name"]))
            elif isinstance(col, str):
                out.append(col)
    return list(dict.fromkeys(str(c) for c in out if str(c)))


def _json_records_key(cap: SourceCapability) -> str | None:
    """Return the top-level list key for object-wrapped JSON records."""
    keys = {str(k).lower(): str(k) for k in cap.json_top_keys}
    if cap.kind in {"json", "jsonl"} and cap.json_record_fields:
        if "records" in keys:
            return keys["records"]
        # Some tasks use a single object wrapper around the real list.
        for key in cap.json_top_keys:
            if str(key).lower() not in {"table", "schema", "metadata"}:
                return str(key)
    return None


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _closest_names(name: str, candidates: list[str], *, limit: int = 5) -> list[str]:
    if not candidates:
        return []
    lowered = {_normalize_name(c): c for c in candidates}
    close = difflib.get_close_matches(
        _normalize_name(name),
        list(lowered),
        n=limit,
        cutoff=0.45,
    )
    return [lowered[item] for item in close]


def _join_candidates(left_cols: list[str], right_cols: list[str]) -> list[dict[str, Any]]:
    """Small deterministic join-key hint set, based on real column names."""
    candidates: list[dict[str, Any]] = []
    left_norm = {_normalize_name(c): c for c in left_cols}
    right_norm = {_normalize_name(c): c for c in right_cols}
    for norm in sorted(set(left_norm) & set(right_norm)):
        candidates.append({
            "left_on": left_norm[norm],
            "right_on": right_norm[norm],
            "reason": "same_normalized_name",
        })
    if candidates:
        return candidates[:8]

    left_id = [c for c in left_cols if _normalize_name(c).endswith("id")]
    right_id = [c for c in right_cols if _normalize_name(c).endswith("id")]
    for left in left_id:
        left_base = _normalize_name(left).removesuffix("id")
        for right in right_id:
            right_base = _normalize_name(right).removesuffix("id")
            if left_base and right_base and (left_base in right_base or right_base in left_base):
                candidates.append({
                    "left_on": left,
                    "right_on": right,
                    "reason": "related_id_name",
                })
    for left in left_cols:
        left_norm = _normalize_name(left)
        if not left_norm.startswith("linkto"):
            continue
        target = left_norm.removeprefix("linkto")
        for right in right_cols:
            if _normalize_name(right) == f"{target}id":
                candidates.append({
                    "left_on": left,
                    "right_on": right,
                    "reason": "link_to_foreign_key",
                })
    for right in right_cols:
        right_norm = _normalize_name(right)
        if not right_norm.startswith("linkto"):
            continue
        target = right_norm.removeprefix("linkto")
        for left in left_cols:
            if _normalize_name(left) == f"{target}id":
                candidates.append({
                    "left_on": left,
                    "right_on": right,
                    "reason": "link_to_foreign_key",
                })
    return candidates[:8]


def _target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    return None


def _df_arg_name_and_columns(
    arg: ast.AST,
    df_sources: dict[str, dict[str, Any]],
) -> tuple[str | None, list[str] | None]:
    """Return dataframe variable name + selected columns for merge args."""
    if isinstance(arg, ast.Name) and arg.id in df_sources:
        return arg.id, list(df_sources[arg.id].get("columns") or [])
    if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
        name = arg.value.id
        if name in df_sources:
            selected = _strings_from_node(arg.slice)
            return name, selected or list(df_sources[name].get("columns") or [])
    return None, None


def _df_expr_name_and_columns(
    expr: ast.AST,
    df_sources: dict[str, dict[str, Any]],
) -> tuple[str | None, list[str] | None]:
    """Infer the source dataframe + columns for common dataframe expressions.

    This is deliberately conservative, but it covers the generated-code
    shapes that matter for schema safety:
    - filtered = df[df["flag"] == 1]
    - selected = df[["a", "b"]]
    - copied = df[df["flag"]].copy()
    - selected = df.loc[mask, ["a", "b"]]

    Without this pass, downstream merge inference loses columns after a
    filter assignment and misses pandas suffix bugs such as Diagnosis_x /
    Diagnosis_y.
    """
    if isinstance(expr, ast.Name) and expr.id in df_sources:
        return expr.id, list(df_sources[expr.id].get("columns") or [])

    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        if expr.func.attr in {"copy", "drop_duplicates", "reset_index"}:
            return _df_expr_name_and_columns(expr.func.value, df_sources)

    if isinstance(expr, ast.Subscript):
        # df[mask] or df[["a", "b"]]
        if isinstance(expr.value, ast.Name) and expr.value.id in df_sources:
            base = expr.value.id
            selected = _strings_from_node(expr.slice)
            if selected and isinstance(expr.slice, (ast.List, ast.Tuple, ast.Set)):
                return base, selected
            return base, list(df_sources[base].get("columns") or [])

        # df.loc[mask, ["a", "b"]] / df.iloc[...]
        if isinstance(expr.value, ast.Attribute) and expr.value.attr in {"loc", "iloc"}:
            owner = expr.value.value
            if isinstance(owner, ast.Name) and owner.id in df_sources:
                base = owner.id
                selected: list[str] = []
                if isinstance(expr.slice, ast.Tuple) and len(expr.slice.elts) >= 2:
                    selected = _strings_from_node(expr.slice.elts[1])
                return base, selected or list(df_sources[base].get("columns") or [])

    return None, None


def _literal_path_from_call(call: ast.Call) -> str | None:
    for value, _node in _const_strings_from_call(call):
        if value.startswith("http://") or value.startswith("https://"):
            continue
        if "/" in value or "." in value:
            return value
    return None


def _json_payload_name_from_dataframe_call(call: ast.Call) -> str | None:
    """Find payload var in pd.DataFrame(payload['records'] / payload.get(...))."""
    if _call_full_name(call) not in {"pd.DataFrame", "pandas.DataFrame"} or not call.args:
        return None
    arg = call.args[0]
    if isinstance(arg, ast.Name):
        return arg.id
    if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
        return arg.value.id
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
        owner = arg.func.value
        if isinstance(owner, ast.Name):
            return owner.id
    return None


def _table_columns_from_sql(sql: str, cap: SourceCapability) -> list[str]:
    """Best-effort columns for simple read_sql results."""
    refs = _FROM_OR_JOIN_RE.findall(sql)
    if not refs:
        return []
    table_names = {ref.split(".")[-1].strip().lower() for _quote, ref in refs}
    cols: list[str] = []
    for table in cap.tables:
        if str(table.get("table") or "").lower() not in table_names:
            continue
        for col in table.get("columns") or []:
            if isinstance(col, dict) and col.get("name"):
                cols.append(str(col["name"]))
            elif isinstance(col, str):
                cols.append(col)
    return list(dict.fromkeys(cols))


def _infer_dataframe_sources(
    tree: ast.AST,
    *,
    capabilities: list[SourceCapability],
    cap_by_path: dict[str, SourceCapability],
) -> dict[str, dict[str, Any]]:
    """Map dataframe variable names to concrete source columns.

    We intentionally support only common generated-code shapes:
    ``df = pd.read_csv("...")``, ``df = pd.read_json("...")``,
    ``conn = sqlite3.connect("..."); df = pd.read_sql_query(sql, conn)``.
    Unknown transformations are left unchecked.
    """
    conn_to_cap: dict[str, SourceCapability] = {}
    file_handle_to_cap: dict[str, SourceCapability] = {}
    json_payload_to_cap: dict[str, SourceCapability] = {}
    df_to_source: dict[str, dict[str, Any]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if not isinstance(item.context_expr, ast.Call):
                    continue
                if _call_full_name(item.context_expr) != "open":
                    continue
                path = _literal_path_from_call(item.context_expr)
                cap = cap_by_path.get(path or "") or cap_by_path.get((path or "").lower())
                if cap is not None and isinstance(item.optional_vars, ast.Name):
                    file_handle_to_cap[item.optional_vars.id] = cap
                    handle_name = item.optional_vars.id
                    for body_node in node.body:
                        if not isinstance(body_node, ast.Assign) or not body_node.targets:
                            continue
                        target = _target_name(body_node.targets[0])
                        if not target or not isinstance(body_node.value, ast.Call):
                            continue
                        call = body_node.value
                        if (
                            isinstance(call.func, ast.Attribute)
                            and call.func.attr == "load"
                            and call.args
                            and isinstance(call.args[0], ast.Name)
                            and call.args[0].id == handle_name
                        ):
                            json_payload_to_cap[target] = cap

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        target = _target_name(node.targets[0])
        if not target or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "load"
            and call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id in file_handle_to_cap
            and target not in json_payload_to_cap
        ):
            json_payload_to_cap[target] = file_handle_to_cap[call.args[0].id]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        target = _target_name(node.targets[0])
        if not target or not isinstance(node.value, ast.Call):
            continue

        call = node.value
        full = _call_full_name(call)

        if full in {"pd.DataFrame", "pandas.DataFrame"}:
            payload_name = _json_payload_name_from_dataframe_call(call)
            cap = json_payload_to_cap.get(payload_name or "")
            if cap is not None:
                df_to_source[target] = {
                    "path": cap.path,
                    "kind": cap.kind,
                    "columns": _columns_for_capability(cap),
                }
            continue

        if full in _SQLITE_CONNECTORS:
            path = _literal_path_from_call(call)
            cap = cap_by_path.get(path or "") or cap_by_path.get((path or "").lower())
            if cap is not None:
                conn_to_cap[target] = cap
            continue

        if full in {
            "pd.read_csv", "pandas.read_csv",
            "pd.read_json", "pandas.read_json",
            "pd.read_table", "pandas.read_table",
            "pd.read_excel", "pandas.read_excel",
            "pd.read_parquet", "pandas.read_parquet",
        }:
            path = _literal_path_from_call(call)
            cap = cap_by_path.get(path or "") or cap_by_path.get((path or "").lower())
            if cap is not None:
                if full in {"pd.read_json", "pandas.read_json"} and _json_records_key(cap):
                    # pd.read_json({"records": [...]}) produces a nested
                    # object column instead of the record fields. Treat it as
                    # unknown here; check_program emits a repairable issue.
                    df_to_source[target] = {
                        "path": cap.path,
                        "kind": cap.kind,
                        "columns": list(cap.json_top_keys),
                    }
                    continue
                df_to_source[target] = {
                    "path": cap.path,
                    "kind": cap.kind,
                    "columns": _columns_for_capability(cap),
                }
            continue

        if full.split(".")[-1] in {"read_sql", "read_sql_query"} and call.args:
            sql = call.args[0].value if isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str) else ""
            conn_name = None
            if len(call.args) >= 2 and isinstance(call.args[1], ast.Name):
                conn_name = call.args[1].id
            for kw in call.keywords:
                if kw.arg in {"con", "conn"} and isinstance(kw.value, ast.Name):
                    conn_name = kw.value.id
            cap = conn_to_cap.get(conn_name or "")
            if cap is not None:
                df_to_source[target] = {
                    "path": cap.path,
                    "kind": cap.kind,
                    "columns": _table_columns_from_sql(sql, cap) or _columns_for_capability(cap),
                }

    # Second pass: infer common dataframe transformations once base
    # sources are known. This catches pandas merge suffixes such as
    # Diagnosis_x / Diagnosis_y before execution.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not node.targets:
                continue
            target = _target_name(node.targets[0])
            if not target or target in df_to_source:
                continue

            base_name, expr_cols = _df_expr_name_and_columns(node.value, df_to_source)
            if base_name in df_to_source and expr_cols is not None:
                df_to_source[target] = {
                    "path": df_to_source[base_name].get("path"),
                    "kind": df_to_source[base_name].get("kind", "dataframe"),
                    "columns": list(expr_cols),
                }
                changed = True
                continue

            if not isinstance(node.value, ast.Call):
                continue
            call = node.value
            full = _call_full_name(call)
            left_name = right_name = None
            left_cols_override = right_cols_override = None
            if full in {"pd.merge", "pandas.merge"} and len(call.args) >= 2:
                left_name, left_cols_override = _df_arg_name_and_columns(call.args[0], df_to_source)
                right_name, right_cols_override = _df_arg_name_and_columns(call.args[1], df_to_source)
            elif isinstance(call.func, ast.Attribute) and call.func.attr == "merge" and call.args:
                left_name, left_cols_override = _df_arg_name_and_columns(call.func.value, df_to_source)
                right_name, right_cols_override = _df_arg_name_and_columns(call.args[0], df_to_source)
            if left_name not in df_to_source or right_name not in df_to_source:
                continue
            on = left_on = right_on = None
            for kw in call.keywords:
                if kw.arg == "on":
                    on = _first_string_from_node(kw.value)
                elif kw.arg == "left_on":
                    left_on = _first_string_from_node(kw.value)
                elif kw.arg == "right_on":
                    right_on = _first_string_from_node(kw.value)
            left_cols = list(left_cols_override or df_to_source[left_name].get("columns") or [])
            right_cols = list(right_cols_override or df_to_source[right_name].get("columns") or [])
            df_to_source[target] = {
                "path": f"merge({df_to_source[left_name].get('path')},{df_to_source[right_name].get('path')})",
                "kind": "dataframe",
                "columns": _merge_output_columns(
                    left_cols,
                    right_cols,
                    on=on,
                    left_on=left_on,
                    right_on=right_on,
                    suffixes=_suffixes_from_merge_call(call),
                ),
            }
            changed = True

    return df_to_source


def _strings_from_node(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for item in node.elts:
            out.extend(_strings_from_node(item))
        return out
    return []


def _first_string_from_node(node: ast.AST | None) -> str | None:
    strings = _strings_from_node(node)
    return strings[0] if strings else None


def _suffixes_from_merge_call(call: ast.Call) -> tuple[str, str]:
    for kw in call.keywords:
        if kw.arg != "suffixes" or not isinstance(kw.value, (ast.List, ast.Tuple)):
            continue
        values = _strings_from_node(kw.value)
        if len(values) >= 2:
            return values[0], values[1]
    return "_x", "_y"


def _merge_output_columns(
    left_cols: list[str],
    right_cols: list[str],
    *,
    on: str | None,
    left_on: str | None,
    right_on: str | None,
    suffixes: tuple[str, str],
) -> list[str]:
    left_key = left_on or on
    right_key = right_on or on
    duplicate_non_keys = (
        set(left_cols)
        & set(right_cols)
        - {c for c in (left_key, right_key) if c}
    )
    out: list[str] = []
    for col in left_cols:
        out.append(col + suffixes[0] if col in duplicate_non_keys else col)
    for col in right_cols:
        if right_key and left_key and col == right_key and right_key == left_key:
            continue
        out.append(col + suffixes[1] if col in duplicate_non_keys else col)
    return list(dict.fromkeys(out))


def _subscript_columns(node: ast.Subscript) -> list[str]:
    return _strings_from_node(node.slice)


def _base_dataframe_name(node: ast.AST) -> str | None:
    cur = node
    while isinstance(cur, ast.Subscript):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def _has_answer_assignment(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "answer" for t in node.targets):
                return True
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "answer":
                return True
    return False


# ---------------------------------------------------------------------------
# AST scanners
# ---------------------------------------------------------------------------


def _const_strings_from_call(call: ast.Call) -> list[tuple[str, ast.AST]]:
    """Collect string-literal positional / keyword args of a Call."""
    out: list[tuple[str, ast.AST]] = []
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append((arg.value, arg))
    for kw in call.keywords:
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            out.append((kw.value.value, kw.value))
    return out


def _walk_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _call_full_name(call: ast.Call) -> str:
    """Best-effort dotted name for a call target, e.g. ``pd.read_csv``."""
    func = call.func
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_FILE_OPENERS = {
    "pd.read_csv", "pandas.read_csv",
    "pd.read_json", "pandas.read_json",
    "pd.read_excel", "pandas.read_excel",
    "pd.read_table", "pandas.read_table",
    "pd.read_parquet", "pandas.read_parquet",
    "open",
    "json.load", "json.loads",  # json.load takes a file handle but path may be a literal sometimes
    "Path", "pathlib.Path",
}

_SQLITE_CONNECTORS = {"sqlite3.connect"}


def check_program(
    program: str,
    *,
    capabilities: list[SourceCapability],
) -> list[StaticIssue]:
    """Static-check a Python program against the task's source capabilities."""
    issues: list[StaticIssue] = []

    if not program.strip():
        issues.append(StaticIssue(
            code="empty_program",
            severity="error",
            message="Generated program is empty.",
            repair_hint="regenerate the codegen output",
        ))
        return issues

    try:
        tree = ast.parse(program)
    except SyntaxError as exc:
        issues.append(StaticIssue(
            code="python_syntax",
            severity="error",
            message=f"Python syntax error: {exc.msg} (line {exc.lineno}).",
            location={"line": exc.lineno},
            repair_hint="regenerate the codegen output",
        ))
        return issues

    known_paths = _known_paths(capabilities)
    known_tables = _known_table_names(capabilities)
    cap_by_path = _capability_index(capabilities)
    df_sources = _infer_dataframe_sources(
        tree,
        capabilities=capabilities,
        cap_by_path=cap_by_path,
    )

    if not _has_answer_assignment(tree):
        issues.append(StaticIssue(
            code="missing_answer_assignment",
            severity="error",
            message="Program computes values but never assigns the final result to `answer`.",
            repair_hint=(
                "append `answer = <final_dataframe_or_scalar>`; the execution harness "
                "requires this exact variable name"
            ),
        ))

    def add_no_such_column(
        *,
        df_name: str,
        column: str,
        node: ast.AST,
        access: str,
    ) -> None:
        source = df_sources.get(df_name)
        if not source:
            return
        available = list(source.get("columns") or [])
        if not available:
            return
        normalized = {_normalize_name(c): c for c in available}
        if _normalize_name(column) in normalized:
            return
        issues.append(StaticIssue(
            code="no_such_column",
            severity="error",
            message=(
                f"DataFrame `{df_name}` from {source.get('path')} has no "
                f"column/key `{column}`."
            ),
            location={
                "line": getattr(node, "lineno", None),
                "dataframe": df_name,
                "source_path": source.get("path"),
                "column": column,
                "access": access,
                "available_columns": available,
                "closest_matches": _closest_names(column, available),
            },
            repair_hint=(
                "map the natural-language concept to one of available_columns; "
                "do not invent a column name"
            ),
        ))

    for node in ast.walk(tree):
        # 4) pandas column access: df["col"] / df[["a", "b"]].
        if isinstance(node, ast.Subscript):
            df_name = _base_dataframe_name(node)
            if df_name in df_sources:
                for column in _subscript_columns(node):
                    add_no_such_column(
                        df_name=df_name,
                        column=column,
                        node=node,
                        access="subscript",
                    )

        # 5) pandas methods that take column names as string literals.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            owner = node.func.value
            df_name = owner.id if isinstance(owner, ast.Name) else None
            if df_name in df_sources and method in {
                "groupby", "sort_values", "drop_duplicates", "set_index",
                "reset_index", "value_counts",
            }:
                candidates: list[str] = []
                if node.args:
                    candidates.extend(_strings_from_node(node.args[0]))
                for kw in node.keywords:
                    if kw.arg in {"by", "subset", "keys"}:
                        candidates.extend(_strings_from_node(kw.value))
                for column in candidates:
                    add_no_such_column(
                        df_name=df_name or "",
                        column=column,
                        node=node,
                        access=f"method:{method}",
                    )

            # 6) dataframe merge join keys.
            if method == "merge" and isinstance(owner, ast.Name) and node.args:
                left_name = owner.id
                right_name = node.args[0].id if isinstance(node.args[0], ast.Name) else None
                if left_name in df_sources and right_name in df_sources:
                    left_cols = list(df_sources[left_name].get("columns") or [])
                    right_cols = list(df_sources[right_name].get("columns") or [])
                    for kw in node.keywords:
                        if kw.arg == "on":
                            for key in _strings_from_node(kw.value):
                                missing: list[str] = []
                                if _normalize_name(key) not in {_normalize_name(c) for c in left_cols}:
                                    missing.append("left")
                                if _normalize_name(key) not in {_normalize_name(c) for c in right_cols}:
                                    missing.append("right")
                                if missing:
                                    issues.append(StaticIssue(
                                        code="bad_join_key",
                                        severity="error",
                                        message=(
                                            f"merge key `{key}` is not present on "
                                            f"{', '.join(missing)} dataframe(s)."
                                        ),
                                        location={
                                            "line": getattr(node, "lineno", None),
                                            "left_dataframe": left_name,
                                            "right_dataframe": right_name,
                                            "left_source_path": df_sources[left_name].get("path"),
                                            "right_source_path": df_sources[right_name].get("path"),
                                            "key": key,
                                            "left_columns": left_cols,
                                            "right_columns": right_cols,
                                            "candidate_join_keys": _join_candidates(left_cols, right_cols),
                                        },
                                        repair_hint=(
                                            "use one of candidate_join_keys or inspect sample values; "
                                            "do not invent relationship fields"
                                        ),
                                    ))
                        elif kw.arg in {"left_on", "right_on"}:
                            side = "left" if kw.arg == "left_on" else "right"
                            columns = left_cols if side == "left" else right_cols
                            for key in _strings_from_node(kw.value):
                                if _normalize_name(key) not in {_normalize_name(c) for c in columns}:
                                    issues.append(StaticIssue(
                                        code="bad_join_key",
                                        severity="error",
                                        message=f"merge {kw.arg} `{key}` is not present on the {side} dataframe.",
                                        location={
                                            "line": getattr(node, "lineno", None),
                                            "side": side,
                                            "left_dataframe": left_name,
                                            "right_dataframe": right_name,
                                            "left_source_path": df_sources[left_name].get("path"),
                                            "right_source_path": df_sources[right_name].get("path"),
                                            "key": key,
                                            "left_columns": left_cols,
                                            "right_columns": right_cols,
                                            "candidate_join_keys": _join_candidates(left_cols, right_cols),
                                        },
                                        repair_hint="replace the join key with a real column from this side",
                                    ))

        # 7) pd.merge(left, right, on=...).
        if isinstance(node, ast.Call) and _call_full_name(node) in {"pd.merge", "pandas.merge"}:
            if len(node.args) < 2:
                continue
            left_name = node.args[0].id if isinstance(node.args[0], ast.Name) else None
            right_name = node.args[1].id if isinstance(node.args[1], ast.Name) else None
            if left_name not in df_sources or right_name not in df_sources:
                continue
            left_cols = list(df_sources[left_name].get("columns") or [])
            right_cols = list(df_sources[right_name].get("columns") or [])
            for kw in node.keywords:
                if kw.arg != "on":
                    continue
                for key in _strings_from_node(kw.value):
                    missing = []
                    if _normalize_name(key) not in {_normalize_name(c) for c in left_cols}:
                        missing.append("left")
                    if _normalize_name(key) not in {_normalize_name(c) for c in right_cols}:
                        missing.append("right")
                    if missing:
                        issues.append(StaticIssue(
                            code="bad_join_key",
                            severity="error",
                            message=(
                                f"pd.merge key `{key}` is not present on "
                                f"{', '.join(missing)} dataframe(s)."
                            ),
                            location={
                                "line": getattr(node, "lineno", None),
                                "left_dataframe": left_name,
                                "right_dataframe": right_name,
                                "left_source_path": df_sources[left_name].get("path"),
                                "right_source_path": df_sources[right_name].get("path"),
                                "key": key,
                                "left_columns": left_cols,
                                "right_columns": right_cols,
                                "candidate_join_keys": _join_candidates(left_cols, right_cols),
                            },
                            repair_hint="use one of candidate_join_keys or inspect sample values",
                        ))

    for call in _walk_calls(tree):
        full = _call_full_name(call)

        if full in {"pd.read_json", "pandas.read_json"}:
            path = _literal_path_from_call(call)
            cap = cap_by_path.get(path or "") or cap_by_path.get((path or "").lower())
            record_key = _json_records_key(cap) if cap is not None else None
            if cap is not None and record_key:
                issues.append(StaticIssue(
                    code="json_records_read_with_pandas",
                    severity="error",
                    message=(
                        f"`pd.read_json` is unsafe for '{cap.path}' because it is "
                        f"an object-wrapped JSON table with top-level key "
                        f"`{record_key}`."
                    ),
                    location={
                        "line": getattr(call, "lineno", None),
                        "literal": cap.path,
                        "record_key": record_key,
                        "record_fields": list(cap.json_record_fields),
                    },
                    repair_hint=(
                        "load with json.load and build pd.DataFrame(payload['"
                        + record_key
                        + "'])"
                    ),
                ))

        # 1) File-opener calls — first string arg is the path.
        if full in _FILE_OPENERS:
            for value, node in _const_strings_from_call(call):
                # Skip obvious non-paths (URLs, in-memory).
                if value.startswith("http://") or value.startswith("https://"):
                    continue
                if "/" not in value and "." not in value:
                    continue
                if value not in known_paths:
                    issues.append(StaticIssue(
                        code="no_such_file",
                        severity="error",
                        message=(
                            f"`{full}` references file '{value}' but the task "
                            f"context has no such path."
                        ),
                        location={
                            "line": getattr(node, "lineno", None),
                            "literal": value,
                        },
                        repair_hint=(
                            "fix the path to one of the known files; available paths: "
                            + ", ".join(sorted(known_paths))
                        ),
                    ))

        # 2) sqlite3.connect — path arg + URI form.
        if full in _SQLITE_CONNECTORS:
            for value, node in _const_strings_from_call(call):
                # Strip URI scheme + read-only suffix.
                target = value
                if target.startswith("file:"):
                    target = target.split("?", 1)[0][len("file:"):]
                if target not in known_paths:
                    issues.append(StaticIssue(
                        code="no_such_file",
                        severity="error",
                        message=(
                            f"`sqlite3.connect` opens '{value}' but the task "
                            f"context has no such file."
                        ),
                        location={
                            "line": getattr(node, "lineno", None),
                            "literal": value,
                        },
                        repair_hint=(
                            "use one of the known sqlite files; available paths: "
                            + ", ".join(sorted(p for p in known_paths if p.endswith(('.db', '.sqlite', '.sqlite3'))))
                        ),
                    ))

        # 3) SQL strings — heuristic: anything passed as first arg to a method
        #    named `execute` / `read_sql*` and contains SELECT.
        method = full.split(".")[-1] if full else ""
        if method in {"execute", "executemany", "read_sql", "read_sql_query", "read_sql_table"}:
            for value, node in _const_strings_from_call(call):
                if not _SELECT_RE.search(value):
                    continue
                table_refs = _FROM_OR_JOIN_RE.findall(value)
                for _quote, ref in table_refs:
                    bare = ref.split(".")[-1].strip().lower()
                    if not bare:
                        continue
                    if bare in known_tables:
                        continue
                    # Tolerate aliases (FROM results r) — bare match already
                    # used the actual table name; if it doesn't match, it's
                    # a real reference to a missing table.
                    issues.append(StaticIssue(
                        code="no_such_table",
                        severity="error",
                        message=(
                            f"SQL references table `{ref}` but the only "
                            f"sqlite tables in this task are "
                            f"{sorted(known_tables.keys()) or '(none)'}."
                        ),
                        location={
                            "line": getattr(node, "lineno", None),
                            "table": ref,
                            "sql": value[:200],
                        },
                        repair_hint=(
                            "remove the JOIN to that table and orchestrate the "
                            "lookup separately (e.g. doc-extract or json lookup)"
                            if any(c.kind in {"record_text", "structured_table", "json", "jsonl", "md", "txt", "docx"} for c in capabilities)
                            else "drop or rename the table reference"
                        ),
                    ))

    return issues

def has_blocking(issues: list[StaticIssue]) -> bool:
    return any(item.severity == "error" for item in issues)
