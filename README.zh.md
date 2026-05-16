# DataAgent-Bench 课程项目

这是一个面向 KDD Cup 2026 DataAgent-Bench 的课程项目工程。我们基于官方
starter kit 做了 agent 本体改造：从原始 ReAct 的自由工具调用，升级为
`Plan -> Act -> Reflect -> Revise` 的数据智能体控制环。系统会先规划任务和工具，
再执行 Python / SQL / RAG 动作，随后由 reflection critic 检查本轮行动是否真正
满足题意，必要时带反馈重跑。

Schema grounding、semantic guard、answer validation 不再是主贡献，而是服务于
agent 控制环的记忆、证据和自检模块。

## 项目介绍

DataAgent-Bench 的任务输入是一个 `task.json` 和若干上下文文件，可能包含
CSV、JSON、SQLite、Markdown/text 等数据源。系统需要读取问题，调用模型和工具，
最终输出一个 `prediction.csv`。

我们的设计目标是优化 agent 行为本身：

- 更会规划：`PlannerAgent` 先把问题拆成 schema / SQL / Python / document 等子任务。
- 更会行动：`AgenticOperatorExecutor` 把 codegen、SQL/Python/RAG 执行和 repair 当作工具动作。
- 更会反思：reflection critic 检查 plan、program、debug trace、answer shape 是否一致。
- 更会修正：如果 critic 发现具体问题，agent 会把反馈注入下一轮工具行动并重跑一次。
- 结果可审计：每个任务都写出 `trace.json`，记录 plan、action、reflection、repair 和最终答案。
- 课堂可展示：提供 Streamlit 页面，输入 task id 后实时显示执行日志和最终推理摘要。

按目前本地实验观察，简单题约 `70-80` 分，中等到困难题约 `60` 分左右。这不是官方
hidden test 结果，只是公开任务上的阶段性实验表现。作为课程项目，系统已经形成完整
闭环：能跑单题、批量跑、评分、保存 trace、解释过程，并且有针对错误模式的修复和
guard。

## Agent Loop

提交版核心路径从“准入/输出优化”改成“agent 控制环优化”：

```text
PlannerAgent 生成任务级 plan
-> OperatorExecutor 执行 Python / SQL / RAG 工具动作
-> 程序输出 answer + debug_steps 作为观察结果
-> Reflection critic 自检本轮行动
-> 如有具体问题，带 revision_instruction 重跑一次
-> semantic judge / answer validator 作为最终保险
```

主要能力：

- `Agentic Operator Executor`：plan、act、reflect、revise 的主 agent loop。
- `Schema Grounding`：用真实 schema、列名、低基数字段样本和 dtype 给字段候选。
- `Cheap Semantic Guard`：非 LLM 风险检查，低风险快放行，高风险升级。
- `Semantic Consistency and Repair`：语义计划、程序对齐检查和语义修复。
- `Router and Mixed-source Handling`：结合 task type、context size、source shape、
  verifiability 和 operation complexity 选择路线。
- `Streamlit Demo UI`：课堂展示 route、timeline、answer、generated program、
  debug output 和完整 trace。

## 给队友看的系统设计详解

### 0. 一句话理解

这个项目不是“把问题直接丢给大模型，让它凭感觉回答”。我们的思路是：

```text
先确定数据长什么样
-> 再让模型写一段可执行、可检查的程序
-> 程序跑出表格答案
-> 用多层规则检查它有没有明显错
-> 如果风险高，再请模型做语义审题和修复
```

这样做的好处是：模型可以发挥理解问题和写代码的能力，但最终答案不是一段自由文本，
而是由真实文件、真实字段和真实程序执行得到的 `prediction.csv`。每一步还能写入
`trace.json`，方便我们知道它为什么这么答。

### 1. 读取任务：系统先拿到题目和上下文

每个 benchmark task 大概长这样：

```text
data/public/input/task_<id>/
├── task.json
└── context/
```

`task.json` 里有题目 id、difficulty 和 question；`context/` 里可能有 CSV、JSON、
SQLite、Markdown、txt 等文件。入口在：

- `src/data_agent_baseline/benchmark/dataset.py`
- `src/data_agent_baseline/run/runner.py`
- `src/data_agent_baseline/cli.py`

这一层只负责“把任务读出来”，还不开始解题。它会把任务对象传给 router 或指定 agent。
最后 runner 会把每题的运行结果写成：

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

### 2. Task Compiler：先给任务做体检

真正开始前，`task_compiler.py` 会扫描 `context/`，给任务做一个确定性的 profile。
这里尽量不用 LLM，因为这一步只是在回答“文件有哪些、列有哪些、数据规模多大”。

它会生成这些信息：

| 信息 | 含义 | 为什么需要 |
| --- | --- | --- |
| `source_capabilities` | 每个文件的类型、列名、样例值、行数、SQLite 表结构等 | 后面 codegen 和 static checker 都要用真实 schema |
| `modalities` | 任务涉及 table、document、image 等哪类数据 | router 需要知道走表格路径还是文档路径 |
| `operations` | 问题可能需要 filter、join、aggregate、compute 等操作 | codegen prompt 和 guard 都需要知道预期操作 |
| `task_type` | `table_computation`、`mixed_context`、`table_with_semantic_rule` 等 | 决定是否需要更重的 semantic path |
| `foreign_key_candidates` | 多表之间可能的 join key | 避免模型乱猜 join 字段 |
| `ambiguity_flags` | `ambiguous_schema`、`large_context`、`record_text_context` 等风险标记 | 决定预算和是否升级 |
| `budget_level` | small / medium / large / xlarge | 控制最大 LLM 调用和工具调用次数 |

一个关键设计是：我们把“真实 schema”整理成 `SourceCapability`，后面所有模块都基于
这份结构化结果工作。这样 prompt、静态检查、repair、guard 看到的是同一套事实，
减少“代码以为有这个字段，但检查器不知道”的漂移。

### 3. Router：按任务形状选路线，而不是只看 difficulty

早期 baseline 容易按 `difficulty` 直接分配路线，但这个标签不一定精确。现在 router
先看 Task Compiler 的结果，再决定走哪条执行路径：

| 路径 | 适合任务 | 说明 |
| --- | --- | --- |
| `agentic_operator` | 大多数表格、混合、多源任务 | 主路径：规划、工具行动、反思、必要时重跑 |
| `operator_executor` | 程序化工具行动 | 被 agentic operator 调用，负责 codegen / execute / repair |
| `tablellm_direct` | 简单表格题或 legacy direct path | 更轻量的直接代码生成 |
| `react` | 需要一步步工具调用的 fallback | 保留 starter kit 风格 |
| `multi_agent` | 更复杂的规划/专家协作路径 | 保留可扩展空间 |

Router 的判断依据包括：

- `task_type`：纯表格、文档、混合源、带规则表格等。
- `source_shape`：单表、多表、多文件、多模态。
- `operation_complexity`：直接 lookup、单 filter、join、aggregate、multi-step。
- `verifiability`：答案能不能靠程序强验证。
- `ambiguity_flags`：是否有大上下文、record text、规则文档、unsupported file。

如果首选 route 失败、答案结构不合法或验证不通过，router 会尝试 cascade fallback。
这些信息会写进：

```text
trace.json -> router_decision
```

### 4. ExecutionContext：把 schema 诊断缓存起来

进入 `operator_executor.py` 后，系统会先创建 `ExecutionContext`。可以把它理解成
“本题的共享资料夹”：里面有 compiled task、schema diagnostics、真实字段、可疑字段、
执行器后面反复需要的上下文。

这样设计是为了避免每一层重复扫描文件，也避免不同模块看到的 schema 不一致。后面的
semantic analyst、codegen、static checker、repair、judge 都围绕这份 context 工作。

相关文件：

- `src/data_agent_baseline/agents/execution_context.py`
- `src/data_agent_baseline/agents/operator_executor.py`

### 5. Schema Grounding：把题目词语绑定到真实字段

LLM 最容易犯的错误之一是“理解了题目，但猜了一个不存在或相近但错误的列名”。比如题目问：

```text
Which patients have severe degree of thrombosis?
```

表里可能同时有：

```text
disease
degree_of_thrombosis
```

如果模型只看到 `thrombosis`，可能会错误写成：

```python
df[df["disease"] == "thrombosis"]
```

但真正应该用的是：

```python
df[df["degree_of_thrombosis"] == "severe"]
```

`schema_grounding.py` 做的事就是：从问题中抽取 concept，然后和真实字段匹配。匹配依据包括：

- 字段名相似度。
- concept 是否出现在低基数字段的样例值中。
- 字段 dtype。
- 字段 cardinality，低基数字段更可能是分类 filter 字段。

输出会进入 codegen prompt，告诉模型：

```text
'severe' -> patients.csv::degree_of_thrombosis
'thrombosis' -> patients.csv::degree_of_thrombosis
```

注意：grounding 只是“候选提示”，不是最终证明。最终还要看代码是否真的用了这些字段，
这就是后面 cheap semantic guard 要检查的内容。

### 6. Semantic Analyst：先审题，但只在需要时启动

`SemanticConsistencyPipeline.plan()` 是“审题官”。它不写代码，只输出一份语义执行计划，
包括：

- 自然语言概念应该映射到哪些真实字段。
- 是否需要 join。
- filter 条件是什么。
- aggregation 或 computation 是什么。
- 输出列应该是什么。
- 有哪些不确定点。
- plan confidence 是多少。

为了省钱和提速，低复杂度表格题会先跳过这一步，直接走 fast path。代码里的规则是：

- `table_computation` / `table_with_semantic_rule`
- 操作复杂度是 `direct_lookup` 或 `single_filter`
- 没有 semantic rule required
- 且不是被 cheap guard 强制升级

这种情况下先不请审题官，等程序跑完后 cheap guard 再决定是否放心放行。如果 cheap guard
发现风险，会重新调用：

```python
plan(task, force=True)
```

这样就形成了一个省成本的分层策略：简单题快跑，风险题再升级。

相关文件：

- `src/data_agent_baseline/agents/semantic_consistency.py`
- `src/data_agent_baseline/agents/operator_executor.py`

### 7. Codegen：让模型写程序，而不是直接写答案

`tablellm_direct.py` / `CodegenDirectAgent` 会把下面这些信息拼进 prompt：

- 原始 question。
- 真实 source capabilities。
- schema grounding block。
- foreign-key candidates。
- semantic plan，若已启用。
- 输出格式要求。
- debug_steps 要求。

模型需要生成一段 Python 程序，程序在本地读取 `context/` 文件，执行 pandas / SQL / JSON
处理，然后产出 `AnswerTable`。我们要求程序记录 `debug_steps`，例如：

```text
schema_inspection       实际读到哪些列
used_columns            最终用了哪些字段
filter_conditions       过滤条件
join_keys               join key
intermediate_counts     每一步剩多少行
knowledge_rules_used    使用了哪些规则或公式
plan_override           如果覆盖了 semantic plan，理由是什么
```

执行 harness 会把这些 debug 信息打印成：

```text
OPERATOR_CODEGEN_DEBUG=<json>
```

后面的 guard 和 judge 都会读取它。也就是说，`debug_steps` 是系统可审计性的核心，不只是日志。

### 8. Static Checker：程序运行前先查明显错误

模型生成程序后，不是马上运行。`static_checker.py` 会先做静态检查，主要查这些问题：

- 代码引用了不存在的文件。
- SQL 里用了不存在的表。
- pandas / SQL 用了不存在的列。
- join key 明显不在两边表里。
- Python 语法错误。

如果发现明确错误，会生成 `StaticIssue`，交给 repair 层。这样能避免浪费一次真实执行，
也能把错误信息变成结构化 repair hint。

相关文件：

- `src/data_agent_baseline/agents/static_checker.py`
- `src/data_agent_baseline/agents/local_repair.py`

### 9. Execute + Answer Validation：程序跑完后检查答案表格

程序通过静态检查后才会真正执行。执行成功不代表答案就能提交，还要通过
`answer_validator.py` 的结构检查：

| 检查项 | 为什么重要 |
| --- | --- |
| 是否有列 | 没有列就无法评分 |
| 是否有行 | 空结果通常是 filter/join 出错 |
| 是否有整列空值 | 说明字段映射或 projection 可能错了 |
| 每行宽度是否一致 | CSV 输出必须是规整表格 |
| 是否混合类型 | warning，不一定失败，但值得记录 |

如果不通过，router 或 semantic judge 会把它当作失败状态，触发 fallback 或 repair。

相关文件：

- `src/data_agent_baseline/eval/answer_validator.py`
- `src/data_agent_baseline/agents/router.py`

### 10. RepairCoordinator：先本地修，再让 LLM 重写

如果 codegen 失败，系统不会马上放弃。`repair_coordinator.py` 会按顺序尝试：

1. `local_repair_loop`：对明显问题做 deterministic patch。例如列名近似、JSON records key、
   输出格式错误、空结果 probe 等。
2. `schema_retry`：如果本地修不了，就把 schema diagnostics、static issues、错误日志给 LLM，
   要求它重写一版 schema-aware 程序。
3. `post_schema_retry_local_repair_loop`：LLM 重写后再跑一轮本地修复。

如果是 record text / structured prose 任务，还会尝试 `structured_doc_executor.py`：

```text
长文本或病历式段落
-> 抽取成结构化 records
-> 合成临时 CSV
-> 重新交给 codegen 做表格计算
```

这个设计是为了避免让 pandas 代码直接在一大段自然语言里乱找字段。先把文本结构化，再做计算，
稳定性会更高。

### 11. Cheap Semantic Guard：低成本判断“能不能放心快放行”

当程序已经成功、答案结构也合法，而且还没有 semantic plan 时，cheap guard 会判断能不能直接返回。
它是非 LLM 的，速度很快。它不会判断“答案一定对”，只判断“风险是否低到可以不跑昂贵 judge”。

它主要看：

- 程序是否成功，答案是否为空。
- 是否经过 local repair / schema retry 才成功。
- `debug_steps` 是否包含 schema inspection、used columns、filters、joins。
- used columns 是否真实存在。
- filter 或 join 后是否出现 0 行。
- 风险词 concept 的 grounded 字段是否真的被程序使用。
- 是否用了问题没有要求的公式或派生指标。

如果有 error 级风险，或者累计风险分 `>= 0.5`，就升级到 semantic analyst / judge / repair。
这层拦截最典型的错误就是“表格形状合法，但语义字段用错了”。

相关文件：

- `src/data_agent_baseline/agents/semantic_guard.py`
- `tests/test_semantic_guard.py`

### 12. Consistency Judge：语义审查和修复

当 semantic plan 存在时，`judge_and_repair()` 会进入“执行官”阶段。它会比较四件事：

- 题目原意。
- Semantic Analyst 的 plan。
- 实际生成的 program。
- `debug_steps` 和 answer preview。

Judge 的输出大概包括：

```text
verdict: pass / fail / low_confidence
confidence: 0.0-1.0
failure_types: 错误类型
mismatches: plan 和程序不一致的点
repair_hint: 应该怎么修
```

只有满足下面条件才真正放行：

```text
verdict == "pass" and confidence >= 0.55
```

否则会进入 semantic repair。repair 会尽量只改必要的程序片段，然后重新执行、重新 judge。
最多修复次数由配置控制，默认最多 `3` 次。这样做的目标是：不要因为一次 codegen 偶然用错字段，
就让整个任务直接失败。

### 13. 写出结果：prediction、trace、summary

如果最终得到合法答案，runner 会写：

- `prediction.csv`：提交/评分用，只包含最终表格。
- `trace.json`：复盘用，包含 route、compiled_task、程序、debug、validation、guard、judge 等。
- `summary.json`：批量运行时的每题状态摘要。
- `score_summary.json`：本地评分后生成。

课堂展示页面 `scripts/demo_app.py` 其实也是读这些信息，只是把命令行日志、answer、program、
trace 用更直观的 Streamlit 页面展示出来。

## 新队友怎么读一个失败 case

如果某个任务结果不好，建议按这个顺序看 `trace.json`：

1. 看 `router_decision.compiled_task.task_type`：任务有没有被分错类型。
2. 看 `source_capabilities`：真实文件和列是否扫描完整。
3. 看 `schema_grounding` 或 `cheap_semantic_assessment.grounding`：问题词有没有正确映射字段。
4. 看 `program`：模型到底生成了什么代码。
5. 看 `OPERATOR_CODEGEN_DEBUG` 解析出的 `debug_steps`：用了哪些列、filter 后剩几行、join key 是什么。
6. 看 `answer_validation`：答案是不是空表、空列或 ragged rows。
7. 看 `cheap_semantic_assessment.risks`：fast path 为什么升级或为什么没有升级。
8. 看 `semantic_consistency.judge_history`：judge 认为哪里不一致，repair 有没有修掉。

这套顺序比直接盯着 `prediction.csv` 有用得多，因为它能定位错误发生在哪个阶段。

## 项目结构

```text
.
├── configs/                         # router / model / scoring 配置
├── data/public/                     # public input/output 数据集
├── scripts/
│   ├── demo_app.py                  # Streamlit 课堂展示页面
│   ├── audit_route_flow.py          # 路由审计脚本
│   └── run_full_public_eval.sh      # 全量 public eval 辅助脚本
├── src/data_agent_baseline/
│   ├── agents/                      # router、operator、grounding、repair、guard
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

## 置信度与风险分怎么评估

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

典型权重：

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

安装依赖：

```bash
uv sync
```

如果当前镜像下载 Streamlit 失败，可以使用官方 PyPI：

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uv sync
```

检查数据集：

```bash
uv run dabench status --config configs/agentic_router.example.yaml
```

跑单个任务：

```bash
uv run dabench run-task task_415 --config configs/agentic_router.example.yaml
```

跑一批任务：

```bash
uv run dabench run-benchmark --config configs/agentic_router.example.yaml --limit 20
```

本地评分：

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
| `configs/router.qwen.yaml` | Qwen / DashScope OpenAI-compatible 全路径配置，主推 react_harness 主路径 |
| `configs/router.dashscope.yaml` | DashScope / Qwen API 配置（按 difficulty 路由的版本） |
| `configs/router.example.yaml` | 路由配置模板 |

注意：配置文件里包含 API endpoint 和 key 字段，提交或展示前请确认是否需要脱敏。

## 输出产物

单任务输出：

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

批量运行还会生成：

```text
artifacts/runs/<run_id>/summary.json
```

评分后会生成：

```text
artifacts/runs/<run_id>/score_summary.json
```

## 结论

对于课程提交，这已经不是一个简单 starter kit，而是一个有明确 agent 控制环、
可运行系统、可视化展示和实验结果的完整项目。当前分数还有提升空间，
但工作量和系统完整度已经可以支撑展示和答辩。
