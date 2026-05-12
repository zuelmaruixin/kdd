# DataAgent-Bench Course Project

这是一个面向 KDD Cup 2026 DataAgent-Bench 的课程项目工程。我们基于官方
starter kit 做了 agent 本体改造：从原始 ReAct 的自由工具调用，升级为
`Plan -> Act -> Reflect -> Revise` 的数据智能体控制环。系统会先规划任务和工具，
再执行 Python / SQL / RAG 动作，随后由 reflection critic 检查本轮行动是否真正
满足题意，必要时带反馈重跑。

Schema grounding、semantic guard、answer validation 不再是主贡献，而是服务于
agent 控制环的记忆、证据和自检模块。

> 给新队友交接或答辩准备时，建议直接看
> [README.zh.md](README.zh.md) 里的“给小白队友看的系统设计详解”。那里按每一步
> 解释了设计动机、输入输出、失败处理和对应代码位置。

## 项目目标

DataAgent-Bench 的任务输入是一个 `task.json` 和若干上下文文件，可能包含
CSV、JSON、SQLite、Markdown/text 等数据源。系统需要读取问题，调用模型和工具，
最终输出一个 `prediction.csv`。

我们的目标是优化 agent 行为本身：

- 更会规划：`PlannerAgent` 先把问题拆成 schema / SQL / Python / document 等子任务。
- 更会行动：`AgenticOperatorExecutor` 把 codegen、SQL/Python/RAG 执行和 repair 当作工具动作。
- 更会反思：reflection critic 检查 plan、program、debug trace、answer shape 是否一致。
- 更会修正：如果 critic 发现具体问题，agent 会把反馈注入下一轮工具行动并重跑一次。
- 结果可审计：每个任务都写出 `trace.json`，记录 plan、action、reflection、repair 和最终答案。
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

### 1. Agentic Operator Executor

提交版核心路径从“准入/输出优化”改成“agent 控制环优化”：

```text
PlannerAgent 生成任务级 plan
-> OperatorExecutor 执行 Python / SQL / RAG 工具动作
-> 程序输出 answer + debug_steps 作为观察结果
-> Reflection critic 自检本轮行动
-> 如有具体问题，带 revision_instruction 重跑一次
-> semantic judge / answer validator 作为最终保险
```

相关文件：

- `src/data_agent_baseline/agents/agentic_operator.py`
- `src/data_agent_baseline/agents/planner.py`
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
source shape、verifiability 和 operation complexity。多源任务会优先走
agentic mixed path，再根据失败类型 fallback。

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

## 项目结构

当前仓库已经从 starter kit 拆成了可运行、可评分、可展示的完整工程：

```text
.
├── configs/                         # router / model / scoring 配置
├── data/public/                     # public input/output 数据集
├── scripts/
│   ├── demo_app.py                  # Streamlit 课堂展示页面
│   ├── audit_route_flow.py          # 路由审计脚本
│   └── run_full_public_eval.sh      # 全量 public eval 辅助脚本
├── src/data_agent_baseline/
│   ├── agents/                      # agent controller、router、operator、grounding、repair、guard
│   ├── benchmark/                   # 数据集读取和 AnswerTable schema
│   ├── eval/                        # 本地 column-match 评分和答案校验
│   ├── run/                         # 单题/批量运行和 self-consistency
│   ├── tools/                       # 文件、SQLite、Python 执行工具
│   ├── cli.py                       # dabench 命令行入口
│   └── config.py                    # YAML 配置加载
├── tests/                           # 本地测试，含 semantic guard case
├── artifacts/                       # runs、logs、audit、cache 产物
├── README.md
├── README.zh.md
└── RUNNING.md
```

`agents/` 是现在最核心的目录：

| 文件 | 作用 |
| --- | --- |
| `router.py` | 编译 task profile，选择 route，并在失败时 cascade fallback。 |
| `task_compiler.py` | 扫描真实上下文，生成 source capabilities、operations、预算和基础置信度。 |
| `agentic_operator.py` | 提交版主 agent loop：plan、act、reflect、revise。 |
| `operator_executor.py` | 工具行动执行器：codegen、repair、guard、judge。 |
| `tablellm_direct.py` | 生成并执行 Python/pandas/SQL 程序。 |
| `schema_grounding.py` | 将问题 concept 绑定到真实字段候选。 |
| `semantic_guard.py` | 非 LLM cheap semantic risk gate。 |
| `semantic_consistency.py` | semantic analyst / consistency judge / semantic repair。 |
| `repair_coordinator.py`、`local_repair.py` | 本地修复、schema retry 和执行错误修复。 |
| `structured_doc_executor.py` | 长文档/病历式文本抽取为结构化 CSV 后再计算。 |

## 置信度与风险分

项目里有几类 `confidence` / `score`，它们不是同一个东西，也不是官方分数。

### 1. Task Compiler 基础置信度

`task_compiler.py` 在路由前给任务画像打三个确定性分数，并写入
`trace.json -> router_decision.compiled_task`：

- `task_type_confidence`：根据数据模态和语义规则判断任务类型。纯表格/文档通常是
  `0.9`，语义规则表格约 `0.86`，mixed context 约 `0.8`，未知或纯推理会更低。
- `operation_confidence`：根据问题关键词判断需要 filter、join、aggregate、compute
  等操作。没有明显操作时是 `0.55`，弱关键词命中是 `0.7`，清晰命中通常是 `0.9`。
- `source_confidence`：从 `1.0` 开始扣分；unsupported file 扣 `0.25`，文件很多扣
  `0.1`，大文档/大表扣 `0.1`，record text 扣 `0.05`，最低保留到 `0.4`。

这些分数主要用于预算、route 和风险判断，不直接表示答案正确率。

### 2. Schema Grounding 字段匹配分

`schema_grounding.py` 会从问题里抽取 concept，再和真实字段做匹配：

- 字段名完全匹配、词匹配接近 `1.0`。
- 子串匹配约 `0.85`。
- 其他情况用 Levenshtein 相似度得到 `0..1` 分。
- 如果 concept 出现在低基数字段样本值中，分数至少提升到 `0.9`。
- 低基数字段且已有一定匹配时会加小 bonus，最多到 `1.0`。

在 `semantic_guard.py` 里，风险词字段低于 `0.68` 会触发
`grounding_low_confidence`；第一、第二候选都不低且差距小于 `0.08` 会触发
`grounding_ambiguous`。

### 3. Cheap Semantic Guard 风险分

`semantic_guard.py` 的 `score` 是“风险分”，不是“置信度”。它只在 fast path
已有结构化答案、且还没有先验 semantic plan 时运行。计算方式是：

```text
risk_score = min(1.0, sum(unique_risk_weights))
should_escalate = any(error risk) or risk_score >= 0.5
```

典型权重包括：

| 风险 | 权重 |
| --- | --- |
| 执行失败 / 无答案 | `1.0` |
| invalid answer | `1.0` |
| 空答案 | `0.75` |
| schema retry 后才成功 | `0.8` |
| local repair 后才成功 | `0.35` 或 `0.65` |
| 缺少 schema inspection trace | `0.8` |
| 缺少 used columns trace | `0.7` |
| filter / join trace 缺失 | `0.65` |
| used column 不在真实 schema 中 | `0.8` |
| 中间过滤结果为 0 | `0.8` |
| 高风险 concept 字段未覆盖 | `0.75` |
| 公式/派生字段使用和问题不匹配 | `0.75` |

所以这里是“越高越危险”。一旦发现 error 级风险，或者累计分达到 `0.5`，系统就升级到
semantic analyst / judge / repair。

### 4. Semantic Plan / Judge confidence

`semantic_consistency.py` 里还有 LLM 输出的 `confidence`：

- Semantic Analyst 输出 plan confidence。prompt 明确要求：如果字段映射还有未解决歧义，
  confidence 必须 `<= 0.65`。
- 当 plan confidence `<= 0.65` 或 `requires_rule_resolution=true` 时，系统会尝试从
  `knowledge.md` 等规则文档中解析规则。
- Consistency Judge 只有在 `verdict == "pass"` 且 `confidence >= 0.55` 时才放行；
  否则进入 semantic repair 或最终 fail。

复盘时主要看这些 trace 字段：

```text
router_decision.compiled_task.task_type_confidence
router_decision.compiled_task.operation_confidence
router_decision.compiled_task.source_confidence
manifest[].cheap_semantic_assessment.score
manifest[].cheap_semantic_assessment.grounding
manifest[].semantic_plan.confidence
manifest[].semantic_consistency.judge_history[].confidence
```

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
uv run dabench status --config configs/agentic_router.example.yaml
```

数据目录默认是：

```text
data/public/input/
data/public/output/
```

### 3. 跑单个任务

```bash
uv run dabench run-task task_415 --config configs/agentic_router.example.yaml
```

输出会写到：

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

### 4. 跑一批任务

```bash
uv run dabench run-benchmark --config configs/agentic_router.example.yaml --limit 20
```

### 5. 本地评分

```bash
uv run dabench score-run artifacts/runs/<run_id> --config configs/agentic_router.example.yaml
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
Config: configs/agentic_router.example.yaml
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
| `configs/agentic_router.example.yaml` | 提交版 agent-first 路由模板，已脱敏 |
| `configs/router.deepseek.yaml` | 本地实验配置，含个人 OpenAI-compatible endpoint/key，默认被 git 忽略 |
| `configs/router.dashscope.yaml` | DashScope / Qwen API 配置 |
| `configs/router.example.yaml` | 路由配置模板 |

注意：配置文件里包含 API endpoint 和 key 字段，提交或展示前请确认是否需要脱敏。

## 关键产物

| 路径 | 说明 |
| --- | --- |
| `src/data_agent_baseline/agents/agentic_operator.py` | 提交版主 agent：plan、act、reflect、revise |
| `src/data_agent_baseline/agents/operator_executor.py` | 工具行动执行器 |
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

对于课程提交，这已经不是一个简单 starter kit，而是一个有明确 agent 控制环、
可运行系统、可视化展示和实验结果的完整项目。当前分数还有提升空间，
但工作量和系统完整度是够提交的。
