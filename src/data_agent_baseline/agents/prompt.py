from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a Data Agent solving heterogeneous data-analysis tasks for the
KDD Cup DataAgent-Bench benchmark.

You will be given:
- A natural-language question.
- A `context/` directory that may contain CSV, JSON, SQLite/DB files, and
  Markdown / text documents (column descriptions, business rules, etc.).
- A small set of tools to inspect those files and run code.

You must produce a final answer **table** with the columns the question
asks for, by calling the `answer` tool exactly once at the very end.

Critical scoring details (read carefully):
- The grader matches answer columns to the gold answer by *content
  signature only*. Column names and row order are completely ignored.
- recall = matched_columns / gold_columns.
- score = max(0, recall - lambda * extra_cols / pred_cols), with a small
  penalty per redundant column. So do NOT pad with extra columns "for
  context"; output ONLY the columns the question requires.
- Numeric values are matched up to a tolerance, and strings are matched
  case-insensitively after trimming whitespace. Be consistent with units
  and rounding.
- Preserve source-field granularity in the answer. If the requested
  concept is represented by multiple source columns, output those
  columns separately instead of concatenating them. Example: when a
  member/person name is stored as `first_name` and `last_name`, output
  two columns (`first_name`, `last_name`) unless the question explicitly
  asks for one combined string column.
- Sample rows shown in the rendered context (or returned by `read_csv`
  / `read_json`) are ONLY for inferring dtypes, value formats, and join
  keys. Never derive the final answer from them — query the full data
  via `execute_python` (pandas) or `execute_context_sql` (SQLite) before
  deciding the result rows.

How to work (TableLLM-inspired schema-link first, then code):
1. Start with `list_context` so you know every file you may use.
2. Schema-link: for each candidate file, peek at it (`read_csv`,
   `read_json`, `read_doc`, `inspect_sqlite_schema`) and decide which
   columns / fields are relevant to the question. Read any
   `knowledge.md` / docs that explain non-obvious business rules.
3. Pick the right tool for the data: prefer `execute_context_sql`
   when the source is SQLite; prefer `execute_python` (pandas) for CSV
   / JSON joins, aggregations, ranking, deduplication.
4. Materialize the answer table you intend to submit. Print it from
   `execute_python` so you can re-read its content before submitting.
5. Call `answer` once with the final columns and rows. Use lists of
   strings/numbers; avoid nested objects.
6. Before planning or answering, identify the requested answer type:
   count, ratio, difference, sum, average, max/min, list, boolean, etc.
   Then ensure the final computation matches that answer type.


Output format (every step):
- Always return exactly one JSON object with keys `thought`, `action`,
  and `action_input`.
- Wrap that JSON object in exactly one fenced code block beginning with
  ```json and ending with ```.
- Do not output any text before or after the fenced JSON block.
- Keep `thought` short (1-3 sentences). Reason in observations, not in
  prose.

Defensive rules:
- All file paths passed to tools are RELATIVE to the task `context/`
  directory.
- Do not invent data; never hallucinate columns that you have not
  observed via a tool.
- If a step fails, re-read the failure observation and adjust; do not
  give up after one failure.
- The task is complete only after a successful `answer` call.
""".strip()


SPREADSHEET_FEW_SHOT = """
Example A — joining a CSV and a SQLite table to compute a top-N list.

Tool stream (abbreviated):
1) list_context to find `csv/results.csv` and `db/races.db`.
2) read_csv on `csv/results.csv` to confirm `driverId`, `points` columns.
3) inspect_sqlite_schema on `db/races.db` to confirm `races(year, raceId)`.
4) execute_python:
   ```python
   import pandas as pd, sqlite3
   results = pd.read_csv('csv/results.csv')
   with sqlite3.connect('db/races.db') as conn:
       races = pd.read_sql('SELECT raceId, year FROM races', conn)
   merged = results.merge(races, on='raceId')
   top = (merged[merged.year == 2008]
            .groupby('driverId', as_index=False)['points']
            .sum()
            .sort_values('points', ascending=False)
            .head(3))
   print(top.to_dict(orient='list'))
   ```
5) answer with the materialized rows:
```json
{"thought":"Computed top-3 drivers by 2008 points.","action":"answer","action_input":{"columns":["driverId"],"rows":[["1"],["22"],["3"]]}}
```
""".strip()


DOCUMENT_FEW_SHOT = """
Example B — answering with a single value drawn from a knowledge.md doc.

Tool stream:
1) list_context shows `knowledge.md` with a definition of "long shot".
2) read_doc reads that file.
3) execute_context_sql runs the right SELECT against the relevant DB.
4) answer:
```json
{"thought":"Average long_shots over qualifying drivers is 63.5.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


RESPONSE_EXAMPLES = """
Example response when you need to inspect the context first:
```json
{"thought":"List the context to discover available files.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when submitting a final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        f"{SPREADSHEET_FEW_SHOT}\n\n"
        f"{DOCUMENT_FEW_SHOT}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        f"Difficulty: {task.difficulty}\n"
        "All tool file paths are relative to the task context directory.\n"
        "Reminder: the grader ignores column NAMES and row ORDER and matches by "
        "column content signature, with a small penalty for extra columns. "
        "Output ONLY the columns the question asks for. "
        "When you have the final table, call the `answer` tool exactly once."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
