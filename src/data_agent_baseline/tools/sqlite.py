from __future__ import annotations

import sqlite3
from pathlib import Path


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def inspect_sqlite_schema(
    path: Path,
    *,
    sample_rows: int = 3,
) -> dict[str, object]:
    """Inspect tables in a sqlite DB.

    Returns, per table: the CREATE TABLE statement, column metadata
    (name + sqlite type + nullable + primary-key index), the row count,
    and up to ``sample_rows`` sample rows. The sample is what the model
    actually needs to write a correct execute_context_sql query without
    a round-trip per column name.
    """
    sample_rows = max(0, int(sample_rows))
    tables: list[dict[str, object]] = []
    with _connect_read_only(path) as conn:
        rows = conn.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        for name, create_sql in rows:
            quoted = f'"{name.replace(chr(34), chr(34) * 2)}"'

            columns: list[dict[str, object]] = []
            try:
                column_rows = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
            except sqlite3.DatabaseError:
                column_rows = []
            for cid, col_name, col_type, notnull, _dflt, pk in column_rows:
                columns.append(
                    {
                        "cid": cid,
                        "name": col_name,
                        "type": col_type,
                        "notnull": bool(notnull),
                        "pk": int(pk),
                    }
                )

            row_count: int | None = None
            sample: list[list[object]] = []
            sample_columns: list[str] = []
            try:
                row_count = conn.execute(
                    f"SELECT COUNT(*) FROM {quoted}"
                ).fetchone()[0]
            except sqlite3.DatabaseError:
                pass

            if sample_rows > 0:
                try:
                    cursor = conn.execute(
                        f"SELECT * FROM {quoted} LIMIT ?",
                        (sample_rows,),
                    )
                    sample_columns = [item[0] for item in cursor.description or []]
                    sample = [list(row) for row in cursor.fetchall()]
                except sqlite3.DatabaseError:
                    sample = []

            tables.append(
                {
                    "name": name,
                    "create_sql": create_sql,
                    "columns": columns,
                    "row_count": row_count,
                    "sample_columns": sample_columns,
                    "sample_rows": sample,
                }
            )
    return {
        "path": str(path),
        "tables": tables,
    }


def execute_read_only_sql(path: Path, sql: str, *, limit: int = 200) -> dict[str, object]:
    normalized_sql = sql.lstrip().lower()
    if not normalized_sql.startswith(("select", "with", "pragma")):
        raise ValueError("Only read-only SQL statements are allowed.")

    with _connect_read_only(path) as conn:
        cursor = conn.execute(sql)
        column_names = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    return {
        "path": str(path),
        "columns": column_names,
        "rows": [list(row) for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
    }
