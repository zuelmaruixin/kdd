<div align="center">

# DataAgent-Bench Starter Kit

English | [中文](README.zh.md)

[![Official Website](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo Dataset](https://img.shields.io/badge/Demo%20Dataset-Download%20Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1c6u5WlFw4KV7CBRyXh5BvFYbKqxhBSbL/view)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

> Official starter kit for the KDD Cup 2026 DataAgent-Bench challenge. The repository reads tasks from `data/public/input/` and writes predictions for downstream evaluation.

## Overview

| Item | Value |
| --- | --- |
| Dataset input | `data/public/input/` |
| Public demo ground truth | `data/public/output/task_<id>/gold.csv` |
| Hidden test data | `input/` only, no `output/` |
| Entry command | `uv run dabench <command> --config PATH` |
| Default run output | `artifacts/runs/` |

## Quick Start

1. Install `uv` by following the official guide:
   - https://docs.astral.sh/uv/getting-started/installation/
2. On macOS and Linux, the standalone installer is:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install project dependencies:

   ```bash
   uv sync
   ```

4. Confirm the dataset root is visible:

   ```bash
   uv run dabench status --config configs/react_baseline.example.yaml
   ```

5. Run the baseline:

   ```bash
   uv run dabench run-benchmark --config configs/react_baseline.example.yaml
   ```

## Dataset

The public demo dataset lives under `data/public/input/`. Each task directory follows this structure:

```text
data/public/input/task_<id>/
├── task.json
└── context/
```

The corresponding public demo answers live separately under `data/public/output/task_<id>/gold.csv`.
Hidden test sets only include `input/`, so there is no `output/` directory there.

`task.json` contains:

- `task_id`
- `difficulty`
- `question`

The `context/` directory may contain one or more of:

- CSV files
- JSON files
- SQLite / DB files
- Text documents

## Configuration

An example config file lives at `configs/react_baseline.example.yaml`.

```yaml
dataset:
  root_path: data/public/input

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 4
  task_timeout_seconds: 600
```

Config fields:

| Field | Meaning |
| --- | --- |
| `dataset.root_path` | Root directory of the public demo `input/` dataset. Relative paths are resolved from the project root. |
| `agent.model` | Model name. |
| `agent.api_base` | OpenAI-compatible API base URL. |
| `agent.api_key` | API key, read directly from the config file. |
| `agent.max_steps` | Maximum ReAct steps per task. |
| `agent.temperature` | Sampling temperature. |
| `run.output_dir` | Output directory for run artifacts. |
| `run.run_id` | Optional run directory name. Defaults to a UTC timestamp if omitted. Must be a single directory name; existing run directories are rejected. |
| `run.max_workers` | Parallel worker count for `run-benchmark`. |
| `run.task_timeout_seconds` | Maximum wall-clock time per task. Set to `0` or a negative value to disable the task-level timeout. |

## CLI

```bash
uv run dabench <command> --config PATH [options]
```

| Command | Purpose | Example |
| --- | --- | --- |
| `status` | Show project paths, config path, dataset root, and public task counts. | `uv run dabench status --config configs/react_baseline.example.yaml` |
| `inspect-task` | Show task metadata and list accessible files under `context/`. | `uv run dabench inspect-task task_1 --config configs/react_baseline.local.yaml` |
| `run-task` | Run the baseline on one task and write outputs. | `uv run dabench run-task task_1 --config configs/react_baseline.local.yaml` |
| `run-benchmark` | Run the baseline across the public dataset. | `uv run dabench run-benchmark --config configs/react_baseline.local.yaml` |

`run-benchmark` also supports `--limit N` to cap the number of tasks.

## Tools

The baseline exposes these tools to the model:

| Tool | Purpose | Inputs |
| --- | --- | --- |
| `list_context` | List files and directories under `context/`. | `max_depth` |
| `read_csv` | Read a CSV preview. | `path`, `max_rows` |
| `read_json` | Read a JSON preview. | `path`, `max_chars` |
| `read_doc` | Read a text document preview. | `path`, `max_chars` |
| `inspect_sqlite_schema` | Inspect tables in a SQLite / DB file. | `path` |
| `execute_context_sql` | Execute read-only SQL against a SQLite / DB file in `context/`. | `path`, `sql`, `limit` |
| `execute_python` | Execute arbitrary Python code inside the task `context/` directory. | `code` |
| `answer` | Submit the final answer table and terminate the task. | `columns`, `rows` |

All file paths passed to tools must be relative to the task `context/` directory.

## Outputs

Each successful task run may produce:

- `trace.json`
- `prediction.csv`

Per-task outputs are written to:

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

Benchmark runs also write:

```text
artifacts/runs/<run_id>/summary.json
```

## Contact

- Open issues: https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues
- Official website: https://dataagent.top
- Discord: https://discord.com/invite/7eFwJQN3Fx
- WeChat official account: `数据智能与分析实验室 DIAL`

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://dataagent.top">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://dataagent.top&bgcolor=ffffff&color=111827&margin=8"
            alt="Official website QR code"
            width="144"
          />
        </a>
        <br />
        Official Website
      </td>
      <td align="center">
        <a href="https://discord.com/invite/7eFwJQN3Fx">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://discord.com/invite/7eFwJQN3Fx&bgcolor=ffffff&color=111827&margin=8"
            alt="Discord QR code"
            width="144"
          />
        </a>
        <br />
        Discord
      </td>
      <td align="center">
        <img
          src="https://dataagent.top/HKUSTGZ_DIAL.jpg"
          alt="WeChat official account QR code"
          width="144"
        />
        <br />
        WeChat Official Account
      </td>
    </tr>
  </table>
</div>

## 快速开始（按 LLM 后端选 config）

| 你想用什么后端 | 用哪个 config | 备注 |
| --- | --- | --- |
| **DashScope (Qwen) API**          | `configs/router.dashscope.yaml`        | Easy = qwen-coder-32B, Medium/Hard = qwen-plus, Extreme = qwen-max |
| **DeepSeek API**                  | `configs/router.deepseek.yaml`         | 全路径用 deepseek-chat，Extreme 加 4-sample SC |
| **本地 Ollama (16GB Mac)**        | `configs/router.lite.yaml`             | 单模型替身，先把架构跑通 |
| **本地 vLLM (NVIDIA GPU)**        | `configs/router.example.yaml` 改两个 URL | TableLLM-13b + Qwen3-8B 全尺寸，冲分用 |

最简流程（以 DashScope 为例）：

```bash
# 1. 装依赖
uv sync

# 2. 把 key 填进 config
sed -i '' 's/REPLACE_WITH_YOUR_DASHSCOPE_KEY/sk-你的key/g' configs/router.dashscope.yaml

# 3. 跑通一个任务
uv run dabench run-task task_19 --config configs/router.dashscope.yaml

# 4. 跑 5 个 + 评分
uv run dabench run-benchmark --config configs/router.dashscope.yaml --limit 5
RUN=$(ls -t artifacts/runs | head -1)
uv run dabench score-run artifacts/runs/$RUN --config configs/router.dashscope.yaml
```

更详细的硬件对应方案见 `RUNNING.md`。

## Difficulty-aware router (recommended top-level mode)

Heavy multi-agent reasoning is overkill on simple single-table questions
and a single ReAct loop is too weak on multi-source 128K-context tasks,
so the recommended setup is `agent.mode: router`. The router reads
`task.difficulty` from `task.json` and dispatches to the right path:

```
Easy     -> TableLLMDirectAgent  (one-shot pandas/SQL via the open-source TableLLM-13b)
Medium   -> ReActAgent            (single-agent ReAct, no planning overhead)
Hard     -> MultiAgentOrchestrator (planner -> specialists -> synthesizer)
Extreme  -> MultiAgentOrchestrator + 4-sample column-vote self-consistency
```

Every route can point at its own OpenAI-compatible endpoint, so you can
mix providers (e.g. TableLLM via DeepInfra / HF Inference Endpoints,
Qwen via DashScope or your own vLLM box). See
`configs/router.example.yaml` for the full layout. The router writes a
`router_decision` block into every `trace.json` so you can audit which
path each task took.

You don't need to fine-tune to use this — `tablellm_direct` calls the
already-open `RUCKBReasoning/TableLLM-13b` model directly. Fine-tuning
Qwen3-8B is still useful for the medium/hard routes when you want to run
fully locally; see `colab/`.

## Multi-agent pipeline

This fork adds a planner / specialist / synthesizer pipeline on top of
the original ReAct baseline. Switch it on by setting `agent.mode:
multi_agent` in the config or `--mode multi_agent` on the CLI.

```
PlannerAgent          (LLM call) -> Plan(rationale, subtasks DAG)
   │
   ▼
Specialist DAG         (one of: schema / sql / python / document / generic)
   │   layered topological execution (independent layers run in parallel)
   ▼
SynthesizerAgent      (LLM call) -> final AnswerTable via the `answer` tool
   │   if synthesis fails: one round of iterative re-planning
   ▼
trace.json with `plan`, `findings`, `synthesizer_steps`
```

The pipeline matches the three reasoning topologies the competition
asks for: sequential chain (linear `depends_on`), branching parallel +
merge (independent specialists in the same layer), and iterative loop
refinement (replan on synthesizer failure).

## Cross-way self-consistency

Inspired by the TableLLM cross-way validation idea, you can sample the
agent N times at temperature `T` and column-vote at the official content
signature level. Set:

```yaml
agent:
  self_consistency:
    num_samples: 4
    sample_temperature: 0.7
    aggregator: column_vote   # or first_success
    min_votes: 2
```

This works for both `react` and `multi_agent` modes — every sample runs
the entire chosen pipeline.

## Local scoring

`data/public/output/<task_id>/gold.csv` holds the public reference
answers. Score a finished run with:

```bash
uv run dabench score-run artifacts/runs/<run_id> --config configs/react_baseline.yaml
```

The scorer implements the official rule:
`score = max(0, recall - lambda * extra_cols / pred_cols)` with
column-content signature matching that ignores column names and row
order.

## Fine-tuning Qwen3-8B (Colab)

End-to-end loop:

1. Run `scripts/build_sft_dataset.py` locally to produce a JSONL of
   verified ReAct rollouts (rollouts whose final answer column-matches
   the gold).
2. Open `colab/finetune_qwen3_8b.ipynb` on Colab. It does 4-bit
   LoRA fine-tuning with Unsloth, then merges the adapter for vLLM.
3. Serve the merged model locally with vLLM and point
   `configs/react_local_vllm.example.yaml` at `http://localhost:8000/v1`.
4. Run with multi-agent + self-consistency:
   ```bash
   uv run dabench run-benchmark --config configs/react_local_vllm.example.yaml
   uv run dabench score-run artifacts/runs/<run_id> --config configs/react_local_vllm.example.yaml
   ```

See `colab/README.md` for the step-by-step.

## Main Modules

| Module | Responsibility |
| --- | --- |
| `src/data_agent_baseline/benchmark/dataset.py` | Public dataset loader |
| `src/data_agent_baseline/tools/filesystem.py` | `list_context`, `read_csv`, `read_json`, `read_doc` |
| `src/data_agent_baseline/tools/python_exec.py` | `execute_python` |
| `src/data_agent_baseline/tools/sqlite.py` | `inspect_sqlite_schema`, `execute_context_sql` |
| `src/data_agent_baseline/tools/registry.py` | Tool registration, terminal `answer`, specialist `report` |
| `src/data_agent_baseline/agents/prompt.py` | System / task / observation prompts (TableLLM-inspired) |
| `src/data_agent_baseline/agents/react.py` | ReAct runtime with JSON action protocol |
| `src/data_agent_baseline/agents/planning.py` | `Plan` / `Subtask` / `Finding` data contracts + DAG layers |
| `src/data_agent_baseline/agents/planner.py` | Planner agent (question -> DAG plan) |
| `src/data_agent_baseline/agents/specialist.py` | SQL / Python / Document / Schema specialists |
| `src/data_agent_baseline/agents/synthesizer.py` | Findings -> final AnswerTable via `answer` |
| `src/data_agent_baseline/agents/orchestrator.py` | Planner -> Specialist DAG -> Synthesizer + refinement |
| `src/data_agent_baseline/agents/tablellm_direct.py` | One-shot code-solution agent backed by TableLLM |
| `src/data_agent_baseline/agents/router.py` | Difficulty-aware dispatcher across the three paths |
| `src/data_agent_baseline/eval/column_match.py` | Official column-signature scorer |
| `src/data_agent_baseline/run/runner.py` | Single-task / benchmark / self-consistency dispatch |
| `src/data_agent_baseline/run/self_consistency.py` | Column-vote self-consistency utilities |
| `scripts/build_sft_dataset.py` | TableLLM-style SFT data builder |
| `colab/finetune_qwen3_8b.ipynb` | Qwen3-8B LoRA training on Colab |
