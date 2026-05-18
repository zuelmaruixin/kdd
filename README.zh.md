# DataAgent-Bench 技术报告

---

## 总体架构

![ChatGPT Image 2026年5月18日 10_18_19](/Users/maruixin/Downloads/ChatGPT Image 2026年5月18日 10_18_19.png)

### 3.1 上下文进入模型的方式

系统不会将 `context/` 下的文件整体拼接至 prompt。不同阶段以不同粒度的
上下文表示进入模型：

| 阶段                | 代码位置                                           | 送入模型的内容                                               | 目的                                                         |
| ------------------- | -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| React 预规划        | `react_planner.py:_context_summary()`              | 由 `list_context_tree()` 得到的文件路径、文件类型与大小（最多前 25 项） | 在不读取文件内容的前提下确定工具调用顺序                     |
| React 主循环        | `prompt.py:build_task_prompt()` 与工具 observation | 初始仅包含问题与系统约束；文件内容必须通过工具逐步读取       | 防止模型在未检视数据时直接作答，并保留可复盘的工具轨迹       |
| Multi-agent planner | `planner.py:_build_context_overview()`             | `render_compact_schema()` 生成的文件清单、schema 与行数（不含数据行） | 为 planner 提供拆分子任务所需的结构信息                      |
| Operator/codegen    | `tablellm_direct.py:render_context_for_codegen()`  | `context_render.py` 生成的表格样本、JSON 摘要、文档相关章节或 RAG top-K 片段 | 让 codegen 能够生成与真实 schema 匹配的程序，同时控制输入长度 |

预规划阶段不读取数据内容；codegen 阶段仅接收按预算筛选后的切片；真正的
计算必须由模型通过 `execute_python` 或 `execute_context_sql` 在任务上下
文中完成。这种分层使得 prompt 规模可控，最终答案仍基于完整数据执行。

---

## 4. 任务画像与路由机制

### 4.1 任务画像（TaskCompiler）

`agents/task_compiler.py` 实现了无 LLM 的任务编译层，输出一个共享的事实
快照 `CompiledTask`，主要字段如下：

| 字段                     | 含义                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `task_type`              | 任务类型，取值集合为 `table_computation` / `table_with_semantic_rule` / `record_text_with_semantic_rule` / `document_qa` / `mixed_context` / `image_understanding` / `pure_reasoning` |
| `answer_type`            | 答案形态：`scalar` / `boolean` / `table`                     |
| `source_capabilities`    | 每个上下文文件的 schema、行数、样本值、低基数字段、SQLite 表结构等。CSV、JSON、SQLite、record_text 各有不同扫描深度 |
| `operations`             | 候选操作集合，如 `retrieve` / `extract` / `filter` / `join` / `groupby` / `aggregate` / `sort` / `topk` / `compare` / `compute` |
| `foreign_key_candidates` | 基于样本值 Jaccard 重叠推断的候选 join 关系                  |
| `ambiguity_flags`        | 风险标记，如 `large_context` / `record_text_context` / `record_extraction_required` / `semantic_rule_context` / `large_document_context` / `large_table_context` / `needs_vision` / `unsupported_file_type` 等 |
| `execution_profile`      | 由 `context_size` / `source_shape` / `verifiability` / `operation_complexity` / `recommended_strategy` / `semantic_rule_required` 等组成的执行画像 |

任务画像统一了 prompt、静态检查器与修复器之间对数据的认知，避免不同模块
基于不同字段集工作而引入隐性漂移。

对于含 record_text 文件的任务，`record_text_classifier.py` 还会进行一次
轻量分类调用，将问题判定为 `aggregate`（适合先抽取为 CSV 再计算）或
`read`（阅读理解，应保留原始叙述）。该裁决决定后续是否启用 structured-
doc synthesis 路径。

### 4.2 失败回退（cascade）

`router.py` 通过 `_failure_type()` 将失败原因归类，并由
`_next_repair_route()` 选择后继路径。当前配置中的 cascade 顺序为：

```yaml
cascade_on_failure: true
cascade_order: [react_harness, tool_first_mixed, extreme, fallback_multi_agent]
cascade_max_extra_attempts: 1
```

单题至多额外尝试一次回退；后继路径全部为 agentic_operator + RAG 或
multi_agent。`_failure_type` 与对应后继策略如下：

| `failure_type`                                               | 触发条件                                            | 后继策略                                                     |
| ------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------ |
| `unsupported_file_type`                                      | 任务画像携带 `unsupported_file_type` flag           | 优先 `fallback_multi_agent`，否则任意 `multi_agent` / `react` route |
| `budget_exhausted`                                           | 失败原因含 `budget_exhausted`                       | 同上                                                         |
| `retrieval_empty` / `doc_context_miss`                       | RAG 无命中或文档定位失败                            | 优先启用 RAG 的 operator / tablellm / react route            |
| `semantic_consistency_failed`                                | judge 判定失败或 filter 语义错误                    | 优先 `tool_first_mixed`，否则任意 operator 类 route          |
| `missing_answer` / `syntax_error` / `static_error` / `exec_error` | 程序未生成 answer、语法错误、静态检查失败、执行异常 | 优先 operator 类 route；对小规模、可程序化任务避免越级到 `extreme` |
| `zero_rows`                                                  | 答案有列但行数为 0                                  | 不再换路（local repair 已处理）                              |

每次回退的尝试与结果均写入 `router_decision.cascade_attempts`，便于事后
定位问题阶段。

### 4.4 Cross-model verification

`router.py` 提供 cross-model verification 的完整实现：当
`agent.cross_model_verify.enabled=true` 时，路由器在主路径产出答案后，
并行使用一组 verifier endpoint（沿用同一 route 配置但替换模型）独立求解，
然后对**列签名**取交集——主答案的每一列签名必须被至少 `min_agreement-1`
个 verifier 也产出，方可保留；若交集为空，则回退到主答案并在 trace 中
记录 `empty_intersection_keep_primary`。该机制针对"主模型多输出冗余列"
这一评分风险，但在当前配置中保持关闭，以避免在固定的单题预算内叠加额外
模型调用。

### 4.5 Reasoner Repair

`_try_reasoner_repair` 在以下条件全部满足时触发：

1. `agent.reasoner_repair.enabled=true`；
2. 当前 payload 处于失败态（`succeeded=False` 或答案列表为空）；
3. 当前 route 为 program 类（`agentic_operator` / `operator_executor` /
   `tablellm_direct`），且失败 payload 中包含一段已生成的程序；
4. `compiled_task.needs_reasoner=True`，或任务为 `document_qa` 且画像携
   带 `large_document_context` flag；
5. `BudgetController.can_reasoner_repair()` 仍有额度。

由于 React Harness 不属于 program 类 route，Reasoner Repair 仅在 cascade
进入 `tool_first_mixed` 或 `extreme` 之后才会启动。其执行流程为：基于失
败程序、stderr/stdout 与 capabilities 摘要让独立的 reasoner 模型重写程
序，立即在任务上下文中执行；若执行成功且产出有效表格，则替换原 payload，
否则保留原始失败 payload 并将修复信息附加为 `payload["reasoner_repair"]`。

---

## 5. 主路径：React Harness

它基于 ReAct 的"思考—行动—观察"循环，
在工具协议、上下文管理、知识引导、错误归类、self-verification 与状态可
观测六个维度上加入了系统性约束。下文按这六个维度分别说明。

### 5.1 工具协议层

工具描述由 `tools/registry.py` 中的 `ToolSpec` **同源**维护，对外提供两
种视图：面向 prompt 的人类可读说明（`describe_for_prompt`）与面向
OpenAI-compatible tools API 的 JSON Schema（`describe_for_tool_api`）。
同源避免了"prompt 声称的参数"与"API 校验的参数"漂移。

模型经原生 function-call 协议调用工具；
当服务端返回纯文本而非 tool calls 时，系统保留文本 JSON 解析回退，并附
带三层确定性容错（按顺序尝试）：

1. `_strip_json_fence`：去除 ```json``` 与裸 ``` 围栏；
2. `_escape_control_chars_inside_json_strings`：把字符串内裸出的换行、
   回车、制表符转义为 `\n`、`\r`、`\t`；
3. `_append_missing_json_closers`：扫描栈式括号，追加被截断的 `}` / `]`。

此外，当模型在原生模式一次性返回多个 `tool_call` 时，系统**只采纳第一
个**，且不将其余 tool_call 写入 step；这一选择是必要的，否则下一轮重建
messages 时未被回应的 `tool_call_id` 会让 OpenAI-compatible 服务端以
`400` 拒绝整次请求。

### 5.2 上下文管理：工具结果缓存与切片 observation

为避免模型在同一任务内重复读取同一文件，React 主循环对只读工具维护一
个 per-task 内存缓存。缓存键由 `(action, sorted_keys(action_input))`
决定，使语义相同但参数顺序不同的两次调用仍能命中。
可缓存工具集合：

```
list_context, read_csv, read_json, read_doc, head_doc,
grep_doc, inspect_sqlite_schema, execute_context_sql, consult_knowledge
```

`execute_python` 不缓存以保留副作用语义；`answer` 作为终止动作亦不缓
存。**缓存命中路径不会走入终止逻辑**（react.py:327-328）：即使先前调用
是 terminal-eligible 的工具，缓存命中也只生成一个普通 observation，避
免误触发 `state.answer = result.answer` 的提交分支。

工具层本身亦采用切片式 observation：`read_csv` 支持 `columns_only=true`
仅返回表头与行数、`offset` 分页；`read_doc` / `read_json` 按 `max_chars`

+ `offset` 分页（单次最多 6 KB）；`head_doc` 仅返回前若干行；`grep_doc`
  返回命中行及其上下文。该约束保证了 observation 长度可控，与 system
  prompt 中"样本行禁止据答"的策略协同。

### 5.3 预规划（pre-loop planner）

`react_planner.py` 在循环开始前进行一次轻量 LLM 调用，输出一段子任务
sketch（建议读取的文件、可能用到的工具、大致计算顺序）。其设计要点：

- **输入受限**：planner 仅看到 `_context_summary()` 生成的截断文件清单
  （路径与字节数，最多 25 项），不读取文件内容；
- **prompt 强约束**：planner 系统提示明确规定，若上下文中存在
  `knowledge.md / *rule* / *glossary*`，第二步必须读取该文档（其内容
  覆盖先验知识）；最后一步必须为 `answer`；
- **plan 持续可见**：plan 被注入为一条 user 消息前缀，主循环每轮重构
  messages 时都会重新放入这条消息（react.py:254-260），避免 plan 随轮
  数推移被遗忘；
- **优雅降级**：planner 调用失败、JSON 解析失败、字段缺失等任一情况下，
  整个循环继续运行，仅在 `state.plan` 中记录失败原因；
- **`skip_for_easy`** 默认开启：当任务标签为 easy 时跳过 planner，省去
  一次模型调用。

multi-agent fallback 中的 `PlannerAgent` 与 React 预规划不同：它使用
`render_compact_schema()` 提供文件清单、schema 与行数（不含数据行），
具体数据仍由 specialist 通过工具调用获取。

### 5.4 System Prompt 设计（schema-link first，then code）

`prompt.py` 中的 `REACT_SYSTEM_PROMPT` 把若干评分对齐与领域约束写入系
统提示，主要包括：

- **Knowledge file 不可议价**：`knowledge.md` 等文档为单一事实源；当其
  定义与模型先验冲突时，**始终以文档为准**；并要求**语义化**应用规则
  ——把规则解析为条件/操作后映射到真实列名，而不是照抄变量；
- **样本行使用约束**：`read_csv` 等预览返回的样本行**仅可**用于推断
  dtype、字段格式与 join key，**禁止**据此输出最终答案；最终答案必须
  通过 `execute_python` 或 `execute_context_sql` 在完整数据上计算；
- **粒度保留**：当问题概念由源中多个字段表达时（如 `first_name` /
  `last_name`），输出应保留分列，避免合并；与评分函数的列签名匹配规则
  对齐；
- **数值与字符串归一化提醒**：与本地评分器一致（数值 $10^{-2}$ 容差、
  字符串 strip + lower）；
- **Few-shot 示例**：附 spreadsheet（CSV + SQLite join + top-N）与
  document（基于 knowledge 的标量答案）两类典型样例；
- **Native tools API 模式下的输出规则**：当走原生 function-call 时，系
  统提示移除"必须返回 fenced JSON"的指令，仅要求每轮调用一个 tool 并
  附简短 rationale，避免模型被旧规则拉回到文本 JSON 模式而丢失原生
  `tool_call_id`。

### 5.5 错误归类与 retry hints

`react_retry_hints.py` 将工具调用错误转换为结构化 `RetryHint`，由 React
主循环写入 observation 的 `retry_hint` 字段。其结构与机制为：

- **按工具分组的错误模式集合**：Python 错误 10 类（覆盖
  `FileNotFoundError` / `ModuleNotFoundError` / `KeyError` /
  `AttributeError` / 数值转换 / `TypeError` / 语法错误 / 30 秒超时 /
  CSV 解析 / 编码错误）、SQL 错误 4 类（unknown table / unknown column
  / 语法错误 / 写操作被拒）、Path 错误 1 类、`answer` 错误 3 类；
- **错误签名归一化**（`_signature`）：去除 `line N`、绝对路径、引号内
  内容，使 `no such column: foo` 与 `no such column: bar` 归并为同一签
  名，从而能统计"同类错误"；
- **`ErrorHistory` 重复升级**：同签名出现 ≥ 2 次时，提示前置一句
  "Stop retrying this approach — switch tools or read the data first"，
  把"重复修复"行为切换为"换策略"；
- **per-action 分组分派**：`execute_python` 走 Python 模式集，
  `execute_context_sql` 走 SQL 模式集，`read_*` 与
  `inspect_sqlite_schema` 走路径模式集，`answer` 走答案结构模式集；
- **Tool 异常时仍跑 hint**：即便底层 tool 抛异常未返回 result（典型情
  形为 `answer` 行宽不一致触发结构异常），主循环仍然构造一次 hint 写
  入 observation（react.py:660-681），让模型下一轮看到具体修正建议；
- **未匹配时的兜底建议**：未命中任何模式时仍返回一句"Re-read the
  error message; verify with `inspect_sqlite_schema` / `read_csv` /
  `list_context` first" 的通用提示。

此外，当模型走 native tool calling 时，工具调用失败仍会保留 action 名
与 `tool_call_id` 写入 step（react.py:682-691），下一轮重建 messages
时不会出现 orphan tool message——避免 OpenAI-compatible 服务端因
`tool_call_id` 不匹配而 400。

### 5.6 React Cheap Answer Guard

`react_answer_guard.py` 是主路径中的低成本风险检查模块。它不调用 LLM，
而是基于已有工具轨迹与候选答案做确定性评估。原则是：能由静态错误、运行时错
误或答案结构错误唯一确定的修复，不再请求 LLM 重写；只有当本地无法安全
判断时才升级到 schema-guided retry 或更上层的修复路径。

guard 与主循环的 `verification_rounds` 协同。当前配置固定
`verification_rounds=1` 与 `use_answer_guard=true`：所有任务都强制进行
一轮 self-verification，guard 在第一次 `answer` 调用时同步运行，并将
其风险代码与最高风险写入 verify observation。

部分guard 检查的风险代码：

| 风险代码                             | 触发条件                                                     | severity / weight |
| ------------------------------------ | ------------------------------------------------------------ | ----------------- |
| `execution_failed_or_missing_answer` | answer 调用未产生 AnswerTable                                | error / 1.0       |
| `empty_answer_rows`                  | 候选答案有列但行数为 0                                       | error / 0.75      |
| `invalid_answer`                     | `validate_answer_table` 检查失败                             | error / 1.0       |
| `answer_without_computation`         | 提交前未成功调用 `execute_python` 或 `execute_context_sql`   | error / 0.85      |
| `knowledge_doc_not_read`             | 上下文存在 `knowledge / rule(s) / definition(s) / glossary / schema_notes` 命名的 md/markdown/txt 文档但未被 `read_doc` 读取 | error / 0.8       |
| `shape_mismatch_scalar_question`     | 问题语气暗示标量答案但行数 > 1                               | warning / 0.35    |
| `shape_mismatch_list_question`       | 问题语气暗示列表但答案是 1×1 标量                            | warning / 0.35    |

总分由各项权重相加并截断至 1.0。该机制将"重答时应重点检查什么"以可审
计的局部证据形式传递给模型，避免在缺乏外部信号时的无差别
self-correction（参见 Huang et al., ICLR 2024；CRITIC, ICLR 2024）。

下表为常见的错误及修复：

| 顺序 | 修复器                                     | 主要 issue                           | 修复策略                                                     |
| ---: | ------------------------------------------ | ------------------------------------ | ------------------------------------------------------------ |
|    1 | `repair_python_syntax`                     | `python_syntax`                      | 去除 Markdown fence 残留并以 `ast.parse` 验证                |
|    2 | `repair_missing_answer_assignment`         | `missing_answer_assignment`          | 在顶层变量中按优先级选择结果变量并追加 `answer = <var>`      |
|    3 | `repair_json_records_read_with_pandas`     | `json_records_read_with_pandas`      | 将 `pd.read_json("x.json")` 改为 `json.load` 后构造 DataFrame |
|    4 | `repair_no_such_table`                     | `no_such_table`                      | 注释掉引用不存在表的 SQL 行                                  |
|    5 | `repair_no_such_file`                      | `no_such_file`                       | 在已知 context 路径中按后缀与编辑距离替换为最近的真实文件    |
|    6 | `repair_pandas_keyerror_or_no_such_column` | `no_such_column` / `pandas_keyerror` | 基于 `available_columns` 与 `closest_matches` 做保守列名替换 |
|    7 | `repair_bad_join_key`                      | `bad_join_key`                       | 仅当静态检查给出唯一 join key 候选且左右键同名时替换         |
|    8 | `repair_merge_dtype_mismatch`              | `merge_dtype_mismatch`               | 在 merge 前插入左右 key 的 `astype(str)`；ID 字段额外去除 `.0` 后缀 |
|    9 | `repair_zero_row_common_filters`           | `zero_rows`                          | 规范化 ID 字段字符串、布尔字符串比较等常见零行过滤原因       |

此外，`repair_answer_table()` 在答案层处理结构问题：`ragged_row` 将行宽
统一至 header 宽度，`empty_column` 删除完全为空的列。每次成功修复通过
`BudgetController.consume_local_repair` 计数，达到 `max_local_repairs`
后立即停止循环。Local repair 的关键不在于"尽量多修"，而在于"仅在证据
充分时修"：列名替换需达到相似度阈值，多候选时不做猜测；join key 修复
要求唯一候选且左右字段名一致。修复记录写入 `local_repair_log` 或
`post_schema_retry_local_repair_log`。

### 5.7 Self-verification 闭环

当 `verification_rounds=1` 时，主循环在模型第一次调用 `answer` 时**不
立即终止**：原本的 terminal observation 被改写为一条非终止 observation，
其 `content` 携带：

- `status: draft_submitted`；
- `verification_round` 与 `remaining_rounds`（用于模型自检自己处于哪一
  轮）；
- `draft_columns`、`draft_row_count`（让模型直接看到当前候选答案的表
  形）；
- `guard_score`、`guard_risk_codes`、`guard_top_risk`（来自 §5.6）；
- `instructions` 字段：由 `build_verify_instructions` 生成，把每个具体
  风险代码翻译为对应的行动指令（例如 `knowledge_doc_not_read` 翻译为
  "Per the project contract, read the knowledge doc and re-answer"），
  最高风险以 `Most important: ...` 突出显示。

模型在第二次 `answer` 调用时才真正提交。该设计将"是否重检"与"重检什
么"分离：前者由配置决定（强制 1 轮），后者由 guard 提供具体证据。

为应对 self-verification 阶段的非典型结束情况，主循环还提供两类 draft
兜底：

- **预算耗尽兜底**（react.py:381-390）：self-verify 阶段抛 `BudgetExceeded`
  时，若已存在 `pending_answer`，立即将其写入 `state.answer` 并以
  `budget_exceeded_during_self_verify: …` 作为 failure 原因，避免有效
  结果被预算超时整体丢弃；
- **未确认兜底**（react.py:717-725）：循环结束时模型未发出第二次 `answer`，
  系统以最近 draft 写入 `state.answer` 并在 trace 中标记
  "Agent did not finish self-verification"。

此外，当模型 API 自身抛异常时，主循环单独记录一条
`__model_error__` step、把异常字符串作为 `failure_reason` 并优雅终止
循环（react.py:392-419），避免单次模型调用故障导致整轮挂起。

### 5.8 预算控制与可观测性

`BudgetController`（`budget.py`）维护六个维度的预算：LLM 调用、工具调
用、运行时间，以及 local repair、reasoner repair、multi-agent
fallback 三类修复轮次。在进入修复或回退路径前，调用方通过 `can_*` 方
法预检；触发上限时以明确的 `budget_exhausted:*` 原因终止。当前配置中
仅 `max_seconds = 600` 为硬上限，其余维度均为不限或大额放行。

主路径同时为外部观察者输出结构化事件：

- 每一步携带 `stream_label` 前缀（如 `react step N`、`react.planner`、
  `react.budget`），使 `ProgressLogger` 能够在多阶段日志中区分阶段；
- guard 单独发 `react_answer_guard` 事件，包含 verdict、score、risk
  codes 与 top risk；
- `failure_reason` 区分多种结局：`budget_exceeded_during_self_verify`、
  "did not finish self-verification"、"did not submit an answer within
  max\_steps" 等，以便 trace 复盘准确归因。

### 5.9 小结

主路径上的优化遵循"低成本约束优先、模型调用克制"的原则：协议层用同源
schema + 三层 JSON 容错保证形式正确；上下文层用切片 observation 与缓
存抑制冗余；预规划用受限输入与可选 skip 抑制成本；system prompt 把领
域约束（knowledge 不可议价、样本行禁用、粒度保留）固化进首条消息；错
误处理通过签名归一化与重复升级把"反思"外化为确定性规则；self-verify
在强制一轮的前提下用 guard 提供具体证据；最后由六维预算与多态
`failure_reason` 保证执行边界与可复盘性。这些优化共同使 React Harness
在公开评测中作为统一首发路径的稳定性得到保证。

---

## 6. 工具层、知识工具与长文档检索

工具层提供对任务上下文的受控访问。`tools/registry.py` 默认注册 10 个工
具；当配置启用 helper 模型且当前 route 为 React 时，第 11 个工具
`consult_knowledge` 会被额外注册。

| 工具                        | 作用                                                         | 终止 |
| --------------------------- | ------------------------------------------------------------ | ---- |
| `answer`                    | 提交最终答案表（columns + rows）；唯一的终止动作             | ✓    |
| `list_context`              | 列出 context 下的可用文件树                                  |      |
| `read_csv`                  | 分页读取 CSV 预览，支持 `columns_only` 模式                  |      |
| `read_json`                 | 读取 JSON 预览，按 `max_chars`/`offset` 分页                 |      |
| `read_doc`                  | 读取 Markdown/TXT 文档片段（单次最多 6 KB）                  |      |
| `head_doc`                  | 读取文档前若干行                                             |      |
| `grep_doc`                  | 在文档中执行正则/子串检索，返回命中行及上下文                |      |
| `inspect_sqlite_schema`     | 列出 SQLite 表结构与少量样本行                               |      |
| `execute_context_sql`       | 在 SQLite 上执行只读 SQL（带 limit）                         |      |
| `execute_python`            | 在 context 目录中执行 Python 代码（独立子进程，30 秒硬超时） |      |
| `consult_knowledge`（可选） | 调用 helper 模型解释 knowledge / rule 文档                   |      |

### 6.1 受控执行环境

`execute_python` 在独立 `multiprocessing.Process` 中执行模型生成的代码
（见 `tools/python_exec.py`），工作目录固定为当前任务的 `context/`。
stdout 与 stderr 通过 `multiprocessing.Queue` 由父进程读取。若代码运行
超过 `EXECUTE_PYTHON_TIMEOUT_SECONDS = 30` 秒，子进程被
`terminate()`/`kill()` 并返回超时错误。该机制并非完整安全沙箱，但能够
隔离长时间运行、防止执行命名空间污染主进程，并在异常退出时保留可读的错
误信息。`execute_context_sql` 仅允许只读语句；文件读取工具均限制单次返
回内容规模，以控制 observation 长度。

文件读取工具采取切片式 observation：`read_csv` 默认仅返回有限行数，并支
持 `columns_only=true` 仅返回表头与行数；`read_doc` 与 `read_json` 通过
`max_chars` 与 `offset` 分页，单次字符数有上限；`head_doc` 仅返回前若
干行；`grep_doc` 返回命中行及其上下文。该设计与系统提示中的约束一致：
样本行可用于判断字段、类型、格式与 join key，最终结果必须通过 Python
或 SQL 在完整数据上计算。

### 6.2 `consult_knowledge` 规则文档工具

部分任务包含 `knowledge.md`、`rules.md`、`glossary.txt` 等规则文档。
`consult_knowledge`（`tools/knowledge.py`）将"规则解释"与"数据计算"解
耦：仅当 `agent.helper_model.enabled=true` 且当前 route 为 React 时，
`router.py:_run_one_route` 会安装一个带剩余调用次数的 `HelperRuntime`，
并将该工具注册至工具集。

执行流程为：

1. 解析显式 `files` 列表；若未指定，则用正则
   `(knowledge|rule(s)|definition(s)|glossary|schema_notes)\.(md|markdown|txt)`
   自动扫描；
2. 对每条路径执行上下文边界检查；
3. 将文档按字符预算合并为带文件名的片段，连同问题一起发送给 helper 模型；
4. 系统提示约束 helper 仅可使用所给文档作答，无法在文档中找到答案时必
   须返回 `not_specified`；
5. 将 helper 返回的简短解释作为 observation 喂回 React 主循环，由主模型
   决定如何将规则转化为筛选条件、派生字段或最终答案。

helper 调用受 `HelperRuntime.calls_remaining` 与 `(question, file_names)`
缓存键约束，避免对同一文档的重复提问。

### 6.3 长文档检索

对于较长的 Markdown、TXT、DOCX 与 JSON 上下文，`document_retriever.py`
提供面向 Operator 路径的 RAG 流水线：

```text
load document → chunk by heading / JSON path → BM25 sparse retrieval
        ↘  optional dense retrieval (embedding model)  ↗
                    Reciprocal Rank Fusion (rrf_k)
                                ↓
        optional query expansion with paraphrases + HyDE
                                ↓
                    optional cross-encoder rerank
                                ↓
                          top-K rendered context
```

文档切分保留标题路径，JSON 切分保留对象路径，因此返回片段同时携带正文
与位置信息。BM25 适合实体名、ID、字段名等精确匹配；启用稠密检索后，系
统使用 Reciprocal Rank Fusion 融合稀疏与稠密结果；启用 query expansion
后，系统额外生成若干改写查询与一段假设性回答（HyDE）以提高同义表述的
召回率；启用 reranker 后，系统先取较大候选集合（`first_stage_top_n`，
默认 30）再由 cross-encoder 联合打分得到 top-K（默认 6）。

该流水线遵循渐进增强原则：当 embedding、query expansion、reranker 任一
组件不可用时，链路退化至 BM25 检索而非阻断任务。当前配置在
`tool_first_mixed` 与 `extreme` 两条 cascade route 中启用 BM25 + DashScope
`text-embedding-v4` 稠密检索（RRF 融合）+ query expansion（含 HyDE），未
配置 reranker；React 主路径不进入该流水线，文档检索由模型通过 `read_doc`
/ `grep_doc` / `head_doc` 自行完成。

---

## 7. 程序化路径（Operator / Codegen / Multi-agent）

部分任务以一段完整的 pandas/SQL 程序求解更为直接。系统在 React Harness
之外保留了三类程序化路径，用于 cascade 兜底与消融实验：`OperatorExecutor`
（`agents/operator_executor.py`）、其上层包装 `AgenticOperatorExecutor`
（`agents/agentic_operator.py`）以及 multi-agent 编排器
（`agents/orchestrator.py`）。`OperatorExecutor` 是程序化路径的核心，其
执行流程包含六个阶段：

```
Phase 0  record_text query_type 分类（仅当上下文含 record_text）
Phase 1  SemanticConsistencyPipeline.plan()              — 审题官（无代码）
Phase 2  初始 codegen（CodegenDirectAgent）
         或 structured-doc 预抽取（record_text + aggregate）
Phase 3a RepairCoordinator.local_repair_loop()           — 确定性 local repair
Phase 3b structured-doc 合成兜底（仅 record_text 失败时）
Phase 3c schema-guided LLM retry + 二次 local repair
Phase 3d 恢复 semantic plan（应对 analyst 异常或 cheap-guard 升级）
Phase 4  SemanticConsistencyPipeline.judge_and_repair()  — 执行官
```

### 7.1 程序生成（CodegenDirectAgent）

`tablellm_direct.py` 构造 codegen prompt，要求模型生成 Python 程序并将
最终结果赋值给变量 `answer`。执行外壳 `_EXEC_HARNESS` 将 `answer` 归一
化为 DataFrame、写出 CSV，并打印 `OPERATOR_CODEGEN_RESULT_OK`、shape 与
`OPERATOR_CODEGEN_DEBUG=<json>`。若模型返回 Series、dict、list 或标量，
外壳会尝试转换为二维表格。

prompt 中的 `Available context` 由 `render_context_for_codegen()` 生成，
按任务形态选择三种渲染方式：当 route 提供 RAG 参数时调用
`render_with_rag()`；当提供问题文本但未启用 RAG 时调用 `render_focused()`；
否则调用 `render_with_samples()`。`context_render.py` 对不同文件类型采
取不同切片策略（CSV：列名/行数/head-spread-tail；SQLite：表结构/列类型
/行数/样本；JSON：top-level keys/记录数/样本；Markdown/TXT：按标题切分
后选择相关 section；超过预算的文件仅保留 stub）。渲染结果同时写入
`context_manifest`，记录每个文件的实际可见切片，以便事后归因。

prompt 进一步要求模型维护 `debug_steps` 字段，记录读取到的字段、使用过
的字段、过滤条件、join keys、中间行数、引用过的规则文档以及与语义计划
不一致时的 override 原因。该字段不参与评分，但供 cheap semantic guard
与 semantic judge 消费。

### 7.2 Schema grounding

`schema_grounding.py` 综合字段名相似度（Levenshtein）、低基数字段样本、
字段类型与 cardinality，向模型提示自然语言概念到字段的候选映射。该模块
仅给出候选，不强制做最终绑定，从而避免在候选不唯一时将不确定判断固化为
系统性错误。

### 7.3 静态检查（StaticChecker）

`static_checker.py` 在程序运行前进行 AST 层检查，通过追踪 DataFrame 变
量与数据源之间的来源关系输出结构化 `StaticIssue`。当前覆盖的 issue code
包括 `python_syntax`、`missing_answer_assignment`、`no_such_file` /
`no_such_table` / `no_such_column`、`pandas_keyerror`、`bad_join_key`、
`merge_dtype_mismatch` 与 `json_records_read_with_pandas`，以及由 stderr
归并而来的 `exec_error` 信号。静态检查的目标是在执行前发现确定性错误，
并将错误转换为可由 local repair 机械修复的形式。

### 7.5 Schema-guided LLM Retry

当 local repair 未能恢复但仍存在结构化证据时，
`RepairCoordinator.schema_retry()` 将 `static_checker` 的全部 issue、
真实 schema、`semantic_plan`（如有）以及上一次失败的 stderr/stdout 组装
为 `schema_diagnostics`，由 LLM 重新生成程序。重写完成后再触发一次
`local_repair_loop`，构成"LLM 重写 → 确定性微调"的串联。

### 7.6 Cheap Semantic Guard 与语义一致性

Operator 路径中的 cheap guard 位于 `semantic_guard.py:assess_cheap_semantic_risk`，
由 `operator_executor.py:_cheap_semantic_assessment` 在需要时调用。其
作用是判断"看似成功"的程序结果是否足够低风险以直接通过，从而决定是否
调用更昂贵的 semantic judge。输入包括 `CodegenRunResult` 的成功状态、
答案与执行输出，`debug_steps` 中记录的 schema inspection、used columns、
filters、join keys、intermediate counts、knowledge rules used，以及
`CompiledTask` 的真实 schema、任务类型、操作集合与执行画像；同时参考
local repair 与 schema retry 的历史以及 schema grounding 的候选映射。

升级条件与 React 一致：任意 error 级风险，或累计权重达到阈值。若 cheap
guard 判定需要升级，`OperatorExecutor` 先尝试以 `sc_pipeline.plan(task,
force=True)` 强制生成 semantic plan；若 analyst 仍未产出 plan，则构造
低置信度的 `_fallback_semantic_plan`，将 cheap guard 发现的风险写入
`uncertainties` 与 `consistency_checks`，并设置 `_force_semantic_consistency=True`，
确保 judge 不会被静默跳过。

随后 `semantic_consistency.py:judge_and_repair` 进入：先做
`_preflight_judge_failure` 的确定性前置检查（`execution_failed` /
`zero_rows` / `invalid_answer` / `runtime_exception` /
`schema_inspection_missing` 等）；调用 `judge_consistency` 得到 verdict
与 confidence；若 verdict 为 pass 且 confidence ≥ 0.55，则接受当前结果；
否则 `run_semantic_repair` 重写程序、跑静态检查、重新执行，再回到 judge，
最多 `max_repairs` 轮。

### 7.7 Record-text 处理

对于结构化程度较弱的自然语言记录，`structured_doc_executor.py` 提供一条
基于 LLM 的抽取路径：将原始 record 文本按字符预算切片，由模型针对每个
切片输出符合统一 schema 的 JSON 记录。schema 字段由 `_guess_schema` 基
于文件结构提示推断，切片由 `_select_relevant_chunks` 按问题语义与已知
schema 字段进行筛选，以控制 LLM 调用成本。抽取得到的记录写入合成 CSV，
并以 `SourceCapability` 形式追加到 `compiled_task.source_capabilities`，
随后再次调用 `CodegenDirectAgent` 让 codegen 将其作为普通表格使用。

入口由 `OperatorExecutor` 控制：当任务画像为
`record_text_with_semantic_rule` 或携带 `record_extraction_required`
flag、且 record-text 分类器未将问题判定为 `read` 时，先抽取再 codegen
（Phase 2）；当初始 codegen 失败、且分类器未判定为 `read` 时，再抽取一
次作为兜底（Phase 3b）。这种"文本理解 → 表格计算"的分阶段处理便于将
错误归因于抽取或计算之一。

### 7.8 Multi-agent fallback

`agents/orchestrator.py` 中的 multi-agent 编排由 planner、specialist 与
synthesizer 构成。`PlannerAgent` 输出 JSON plan：
`{rationale, subtasks: [{id, specialist, instruction, depends_on,
expected_output}]}`，可指定五类 specialist（schema / sql / python /
document / generic）。`SpecialistAgent` 拥有受限工具子集与专属系统提示，
终止动作为 `report`（产出 Finding，可附小规模证据表），不可调用
`answer`；同一 specialist 内 `read_doc` 调用次数被限制为 2。
`SynthesizerAgent` 拥有完整工具集，是唯一可调用 `answer` 的 agent，最
多 6 步。子任务按 DAG 拓扑分层执行，同层多节点可由 `ThreadPoolExecutor`
并行（受 `max_specialist_workers` 约束）。当 `enable_iterative_refinement
=true` 时，若 synthesizer 未能产出答案，会以 `_RefinementTask` 将失败上
下文回灌至 planner 再跑一轮。multi-agent fallback 同样受
`BudgetController.can_multiagent_fallback` 约束，仅在 `cascade_attempts`
满足条件时被启用。

---

## 8. 输出验证与评分对齐

`answer_validator` 在写出 `prediction.csv` 前对答案进行结构检查，覆
盖：缺失答案、零列、零行、ragged rows（行宽与 header 不一致）、整列为
空（fully-null column）以及 mixed-type column 警告。每一次路由 pass 之
后均通过 `_stamp_validation` 将检查结果写回 payload；已被判定为非法的
答案会被强制将 `succeeded` 降为 `False`，从而触发 cascade。验证模块本身
不判断语义正确性，仅保证输出至少可被评分。

`column_match` 实现本地列匹配评分。由于评分函数对列名不敏感但对多余
列有惩罚，prompt、answer guard、operator codegen 与最终评分均围绕"只输
出问题要求的列"建立约束；self-consistency 与 cross-model verification
也以列签名为单位，而非整张表的字符串表示。

### 8.1 Self-consistency 列签名投票

当 `self_consistency.num_samples > 1` 时，系统按官方评分的列匹配逻辑做
self-consistency 聚合（`router.py:_vote_self_consistency` 与
`run/self_consistency.py`）。每个成功样本被转置为列；每列计算
`column_signature`，归一化规则与本地评分器一致。聚合时，每个样本对同一
列签名最多投一票；以样本列数的众数作为目标列数；满足 `min_votes` 的签
名按票数排序，保留至目标列数。最终表格不是任意拼接，而是优先选择一个
真实样本：该样本最大化覆盖获胜签名，并尽量少包含额外列。若该样本完全
覆盖获胜签名，则将其投影到获胜列集合；否则保留其原列集合。聚合策略支
持 `column_vote`（默认）与 `first_success`。

主路径 React Harness 在当前配置中使用 `num_samples=1`，仅 cascade 中的
`fallback_multi_agent` 启用 `num_samples=3` 并触发列签名投票。

### 8.2 语义一致性审计

Operator 路径的语义一致性输出不仅给出 pass/fail，还在
`router.py:_build_semantic_consistency_audit` 中扫描 `result.manifest`
汇总 plan、judge、repair 的运行情况。审计字段包括 `plan_ran`、`judge_ran`、
`judge_attempts`、`judge_final_verdict`、`judge_repaired_code`、
`plan_source`、`plan_cache_hit`，以及若干升级标志
（`analyst_exception_retry_succeeded`、`analyst_exception_fallback_used`、
`lazy_escalation_plan_built`、`fallback_plan_from_cheap_guard`）。

`final_gate` 是审计中最直接的字段，取值如下：

- `plan+judge`：semantic plan 与 judge 均按主路径运行；
- `escalated+judge`：初始 plan 失败或被 cheap guard 升级后，判定仍进入
  judge；
- `cheap_guard_only`：cheap guard 已发现风险但后续 plan 未能建立；
- `bypassed`：判定为低风险，未进入语义一致性验证。

该审计仅在 Operator 类 route 上有意义；React 路径以
`react_answer_guard` 事件作为对应的低成本审计。

---

## 9. Public 评测结果与分析

public split 的本次评测结果如下：

| 指标        |        数值 |
| ----------- | ----------: |
| Records     |          49 |
| Shown       |          49 |
| Success     |     47 / 49 |
| Failed      |           2 |
| Total score | 35.600 / 49 |
| Mean score  |       0.727 |
| Mean recall |       0.745 |
| Score = 1.0 |          34 |

本地评分文件位于 `artifacts/runs/batch-20260516-204243/score_summary.json`。
49 个任务均生成 `trace.json`，47 个任务生成 `prediction.csv`。两条未生
成预测的任务为：

- `task_344`：单题运行超过 600 秒；
- `task_396`：语义修复后仍未通过静态检查。

部分得分的任务（如 `task_38` 0.60、`task_249` 0.25、`task_379` 0.75）
显示系统在这些任务中召回了部分正确列，但同时输出了额外列或遗漏了部分
标准答案列。结合评分函数，这类失分应优先从最终列投影与字段语义绑定两
个环节排查。从执行轨迹看，绝大多数任务由 React Harness 直接完成；进入
cascade 的任务比例较低，与系统的设计定位一致——主路径承担稳定性与协议
约束，cascade 路径处理程序化与语义风险较高的失败样例。



---

## 11. 局限性与后续工作

当前系统在 public split 上取得 0.727 的平均分，仍存在若干局限。

首先，最终列投影不够保守。评分函数对额外列设置惩罚，部分任务召回了正
确列但输出了冗余列，导致得分低于 1.0。后续工作应在最终 projection 阶段
更清晰地区分"中间调试字段"与"最终答案字段"。

其次，字段语义绑定仍是主要风险来源。即使真实字段已被扫描，模型仍可能
将问题概念映射到近似但错误的字段。后续可加强 schema grounding 与
answer guard 的联动，在答案提交前显式检查关键概念是否落到被使用字段
之上。

第三，语义修复后的静态错误仍可能出现。当前修复链路在部分任务中能恢复
失败程序，但也可能在重写后引入新的列名或语法问题。后续可对 semantic
repair 输出强制再执行静态检查，并设置短路策略以避免长时间停留在无效修
复循环。

第四，单题超时控制仍有优化空间。`task_344` 的失败说明 600 秒上限仍可
能被复杂路径耗尽。后续应在路由层更早识别低收益路径，并在重复同类错误
时提前终止。

第五，对于视觉任务与强领域推理任务，本系统主要依赖模型自身能力与
cascade 路径，尚未实现专门的视觉执行器或领域知识模块。

---

