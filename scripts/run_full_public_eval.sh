#!/usr/bin/env bash
# Run the full public benchmark, score it, and summarize routing/cascade flow.
#
# This script DOES call configured model APIs through `dabench run-benchmark`.
# Use `scripts/audit_route_flow.py` when you only want local route inspection.

set -Eeo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/router.deepseek.yaml}"
LIMIT="${LIMIT:-}"
RUN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --show-events|--stream)
      RUN_ARGS+=("$1")
      shift
      ;;
    *)
      RUN_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p artifacts/runs artifacts/audits artifacts/logs

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="artifacts/logs/full_eval_${STAMP}.log"
AUDIT="artifacts/audits/route_flow_${STAMP}.tsv"

echo "project: $ROOT" | tee "$LOG"
echo "config : $CONFIG" | tee -a "$LOG"
if [[ -n "$LIMIT" ]]; then
  echo "limit  : $LIMIT" | tee -a "$LOG"
else
  echo "limit  : full public set" | tee -a "$LOG"
fi
echo "log    : $LOG" | tee -a "$LOG"
echo

echo "== status ==" | tee -a "$LOG"
uv run dabench status --config "$CONFIG" 2>&1 | tee -a "$LOG"
echo

echo "== local route audit, no API calls ==" | tee -a "$LOG"
uv run python scripts/audit_route_flow.py \
  --config "$CONFIG" \
  --output "$AUDIT" \
  --format summary 2>&1 | tee -a "$LOG"
echo "route audit TSV: $AUDIT" | tee -a "$LOG"
echo

echo "== run benchmark, API calls start here ==" | tee -a "$LOG"

RUN_DIR="artifacts/runs/${STAMP}_easy_medium"
mkdir -p "$RUN_DIR"

TASKS=$(uv run python - <<'PY'
import json
from pathlib import Path

root = Path("data/public/input")

tasks = []
for task_dir in sorted(root.glob("task_*")):
    meta_path = task_dir / "task.json"
    if not meta_path.exists():
        continue

    meta = json.loads(meta_path.read_text())
    difficulty = meta.get("difficulty")

    if difficulty in {"easy", "medium"}:
        tasks.append(task_dir.name)

for t in tasks:
    print(t)
PY
)

echo "selected tasks:" | tee -a "$LOG"
printf '%s\n' "$TASKS" | tee -a "$LOG"

for task_id in $TASKS; do
  echo
  echo "== run $task_id ==" | tee -a "$LOG"

  CMD=(uv run dabench run-task "$task_id" --config "$CONFIG" --mode router)

  if (( ${#RUN_ARGS[@]} > 0 )); then
    CMD+=("${RUN_ARGS[@]}")
  fi

  echo "+ ${CMD[*]}" | tee -a "$LOG"
  "${CMD[@]}" 2>&1 | tee -a "$LOG"
done

RUN_ID="$(basename "$RUN_DIR")"
echo
echo "run_id : $RUN_ID" | tee -a "$LOG"
echo "run_dir: $RUN_DIR" | tee -a "$LOG"
echo

echo "== score-run ==" | tee -a "$LOG"
uv run dabench score-run "$RUN_DIR" --config "$CONFIG" 2>&1 | tee -a "$LOG"
echo

if [[ -f eval.py ]]; then
  echo "== eval.py ==" | tee -a "$LOG"
  uv run python -c "from eval import evaluate_batch; evaluate_batch('$RUN_ID')" 2>&1 | tee -a "$LOG"
  echo
fi

echo "== trace flow summary ==" | tee -a "$LOG"
uv run python - "$RUN_DIR" <<'PY' 2>&1 | tee -a "$LOG"
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

run_dir = Path(sys.argv[1])
traces = sorted(run_dir.glob("task_*/trace.json"))

initial_routes = Counter()
cascade_paths = Counter()
failures = Counter()
succeeded = 0

for trace_path in traces:
    try:
        payload = json.loads(trace_path.read_text())
    except Exception as exc:  # noqa: BLE001
        failures[f"trace_read_error:{exc}"] += 1
        continue

    decision = payload.get("router_decision") or {}
    initial_routes[
        f"{decision.get('route_name')} / {decision.get('kind')} / {decision.get('model')}"
    ] += 1

    attempts = decision.get("cascade_attempts") or []
    if attempts:
        path = " -> ".join(
            f"{item.get('route_name')}/{item.get('kind')}/{item.get('model')}"
            for item in attempts
        )
        cascade_paths[path] += 1

    if payload.get("succeeded"):
        succeeded += 1
    else:
        failures[str(payload.get("failure_reason") or "unknown")] += 1

print(f"tasks with trace: {len(traces)}")
print(f"succeeded       : {succeeded}")
print(f"failed          : {len(traces) - succeeded}")

print("\ninitial routes:")
for key, count in initial_routes.most_common():
    print(f"  {count:>3}  {key}")

print("\ncascade paths:")
if cascade_paths:
    for key, count in cascade_paths.most_common():
        print(f"  {count:>3}  {key}")
else:
    print("  none")

print("\nfailure reasons:")
if failures:
    for key, count in failures.most_common(15):
        print(f"  {count:>3}  {key[:220]}")
else:
    print("  none")
PY

echo
echo "done." | tee -a "$LOG"
echo "run_dir       : $RUN_DIR" | tee -a "$LOG"
echo "score summary : $RUN_DIR/score_summary.json" | tee -a "$LOG"
echo "route audit   : $AUDIT" | tee -a "$LOG"
echo "log           : $LOG" | tee -a "$LOG"
