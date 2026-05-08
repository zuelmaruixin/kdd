# DataAgent-Bench Course Project

这是一个面向 KDD Cup 2026 DataAgent-Bench 的课程项目工程。我们基于官方
starter kit 做了完整的 agent pipeline 改造：从简单表格题的快速程序执行，到
多源/中难题的 schema grounding、repair、semantic guard 和可视化展示。

本仓库不再只是原始 ReAct baseline，而是一个可运行、可评分、可展示推理过程的
数据问答系统。

## 项目目标

DataAgent-Bench 的任务输入是一个 `task.json` 和若干上下文文件，可能包含
CSV、JSON、SQLite、Markdown/text 等数据源。系统需要读取问题，调用模型和工具，
最终输出一个 `prediction.csv`。

我们的目标是：

- 简单题尽量快：优先使用一次性代码生成和本地执行，不做多余 LLM judge。
- 中难题尽量稳：加入 schema grounding、静态检查、执行修复、semantic guard。
- 结果可审计：每个任务都写出 `trace.json`，记录 route、程序、执行结果和验证信息。
- 课堂可展示：提供 Streamlit 页面，输入 task id 后实时显示执行日志和最终推理摘要。

## 当前效果

按目前本地实验观察：

| 难度 | 大致表现 |
| --- | --- |
| 简单题 | 约 70-80 分 |
| 中等到困难题 | 约 60 分左右 |

这些分数不是官方最终 hidden test 结果，只是我们在公开任务上的阶段性实验表现。
从课程项目角度看，系统已经具备完整工程闭环：能跑单题、批量跑、评分、保存 trace、
解释过程，并且有针对错误模式的修复和 guard。

## 我们做了什么

### 1. Tool-first Operator Executor

核心路径从“模型一步步 ReAct 调工具”改成更直接的程序执行：

```text
schema/context inspect
-> codegen LLM 生成 Python / SQL / pandas 程序
-> static check
-> execute
-> answer validation
-> cheap semantic guard
-> 必要时 semantic plan + judge + repair
```

相关文件：

- `src/data_agent_baseline/agents/operator_executor.py`
- `src/data_agent_baseline/agents/tablellm_direct.py`
- `src/data_agent_baseline/agents/static_checker.py`
- `src/data_agent_baseline/agents/repair_coordinator.py`

### 2. Schema Grounding

为了减少字段猜错，我们加入了 deterministic schema grounding：

- 从问题中抽取关键 concept。
- 用真实 schema、列名、低基数字段样本、dtype 做匹配。
- 在 prompt 中给 codegen 明确的候选字段。
- 在后处理阶段检查实际使用字段是否覆盖了高风险 concept。

相关文件：

- `src/data_agent_baseline/agents/schema_grounding.py`
- `src/data_agent_baseline/agents/semantic_guard.py`

### 3. Cheap Semantic Guard

简单题不会默认跑昂贵的一致性检测，但也不会只看 `answer valid` 就放行。

fast path 通过条件大致是：

```text
执行成功
答案结构有效
schema grounding 高置信
没有 ambiguous mapping
没有空结果 / zero intermediate count
没有 suspicious fallback
trace 中 used_columns 覆盖了风险字段
```

如果 cheap guard 发现风险，会升级到 semantic analyst / judge / repair。典型会拦：

- 问题问 `severe degree of thrombosis`，代码却只用了 `disease == "thrombosis"`。
- filter 后中间结果为 0。
- schema retry 或 local repair 才得到答案。
- join/filter trace 缺失。
- 字段 grounding 低置信或歧义。

### 4. Semantic Consistency and Repair

对于风险较高的任务，系统会启动 semantic plan 和 judge：

```text
semantic analyst
-> rule/schema resolution
-> compare plan vs program/debug_steps/answer
-> semantic repair if mismatch
```

这里的 `difficulty` 只作为预算提示，不直接决定是否跑 semantic judge。真正决定因素是：

- risk score
- grounding confidence
- rule alignment
- execution / validation failure
- verifiability

相关文件：

- `src/data_agent_baseline/agents/semantic_consistency.py`
- `src/data_agent_baseline/agents/local_repair.py`

### 5. Router and Mixed-source Handling

Router 不再简单按 difficulty 分配路线，而是结合 task type、context size、
source shape、verifiability 和 operation complexity。多源任务会尽量走
tool-first mixed path，再根据失败类型 fallback。

相关文件：

- `src/data_agent_baseline/agents/router.py`
- `src/data_agent_baseline/agents/task_compiler.py`
- `src/data_agent_baseline/agents/structured_doc_executor.py`

### 6. Streamlit Demo UI

为了课堂展示，我们加了一个 Streamlit 页面：

- 输入 task id。
- 选择 config。
- 点击运行。
- 实时显示执行日志。
- 跑完后展示 route、timeline、answer、generated program、debug output 和完整 trace。

相关文件：

- `scripts/demo_app.py`

## 快速运行

### 1. 安装依赖

```bash
uv sync
```

如果当前镜像下载 Streamlit 失败，可以使用官方 PyPI：

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uv sync
```

### 2. 检查数据集

```bash
uv run dabench status --config configs/router.deepseek.yaml
```

数据目录默认是：

```text
data/public/input/
data/public/output/
```

### 3. 跑单个任务

```bash
uv run dabench run-task task_415 --config configs/router.deepseek.yaml
```

输出会写到：

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

### 4. 跑一批任务

```bash
uv run dabench run-benchmark --config configs/router.deepseek.yaml --limit 20
```

### 5. 本地评分

```bash
uv run dabench score-run artifacts/runs/<run_id> --config configs/router.deepseek.yaml
```

评分使用官方列内容匹配逻辑：

```text
score = max(0, recall - lambda * extra_cols / pred_cols)
```

## 课堂展示页面

启动 Streamlit demo：

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uv run streamlit run scripts/demo_app.py
```

页面中输入：

```text
Config: configs/router.deepseek.yaml
Task ID: task_415
```

点击 `Run Task` 后会实时显示：

- router / compiler / codegen 日志
- static check / execute / repair 信息
- cheap semantic guard 是否通过或升级
- semantic plan / judge 是否触发
- 最终 answer table
- generated Python program
- `trace.json`

## 常用配置

| Config | 用途 |
| --- | --- |
| `configs/router.deepseek.yaml` | 当前主要实验配置，OpenAI-compatible API |
| `configs/router.dashscope.yaml` | DashScope / Qwen API 配置 |
| `configs/router.example.yaml` | 路由配置模板 |

注意：配置文件里包含 API endpoint 和 key 字段，提交或展示前请确认是否需要脱敏。

## 关键产物

| 路径 | 说明 |
| --- | --- |
| `src/data_agent_baseline/agents/operator_executor.py` | 主 tool-first 执行器 |
| `src/data_agent_baseline/agents/tablellm_direct.py` | 代码生成 prompt 和执行封装 |
| `src/data_agent_baseline/agents/schema_grounding.py` | 问题 concept 到真实 schema 的匹配 |
| `src/data_agent_baseline/agents/semantic_guard.py` | 非 LLM 的 cheap semantic risk 检查 |
| `src/data_agent_baseline/agents/semantic_consistency.py` | semantic analyst / judge / repair |
| `src/data_agent_baseline/agents/repair_coordinator.py` | 本地修复和 schema retry |
| `src/data_agent_baseline/agents/router.py` | 路由和 fallback |
| `src/data_agent_baseline/agents/task_compiler.py` | 任务画像、预算、execution profile |
| `src/data_agent_baseline/eval/column_match.py` | 本地评分 |
| `scripts/demo_app.py` | 展示用 Streamlit 页面 |

## 工程完整性

目前系统已经具备课程项目所需的完整闭环：

- 可配置模型后端。
- 可运行单题和批量任务。
- 可生成标准 `prediction.csv`。
- 可用公开答案本地评分。
- 可保存可审计 `trace.json`。
- 有 fast path、guard path、heavy path 的分层。
- 有静态检查、执行修复、schema retry、semantic repair。
- 有展示页面解释运行过程。

后续如果继续冲分，可以优先做：

- 收集失败 case，按错误类型补规则。
- 改进 mixed-context / long-doc extraction。
- 对高风险 task 开更多 self-consistency 或 cross-model verification。
- 加一个更细粒度的实时事件 logger，替代现在 Streamlit 中的控制台日志流。

## 结论

对于课程提交，这已经不是一个简单 starter kit，而是一个有明确工程设计、
可运行 pipeline、可视化展示和实验结果的完整项目。当前分数还有提升空间，
但工作量和系统完整度是够提交的。
