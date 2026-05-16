# DataAgent-Bench 技术设计报告

> 本报告记录本课程项目针对 KDD Cup 2026 DataAgent-Bench 在 starter kit 之上
> 所做的系统性改造：解决的问题、采取的方案、对应的代码改动，以及由内部
> 评估观察到的效果。基础概念（pandas / SQLite / OpenAI-compatible 接口）
> 不再展开，重点说明在该 benchmark 下做出每项设计选择的具体依据。

---

## 1 起点与改造路线

### 1.1 starter kit 的可用基线与暴露出的问题

官方 starter kit 提供了一份最小可行的 React-style agent：模型在 assistant
消息中输出 ```json fenced block，由外部正则解析；工具集仅含
`list_context` / `read_csv` / `execute_python` / `answer`；预算控制为简
单计数；无任何后置校验。在本课程项目的内部评估中，该基线暴露出以下问题：

1. **JSON 解析失败率显著**。对超过 6 KB 的复杂 observation、含嵌套字符串
   或不匹配括号的输出，正则解析失败导致约 10% 的步骤被浪费。
2. **答案"形状对但内容错"无法识别**。模型时常输出列数正确、但语义错配
   的 AnswerTable（典型为 `disease='thrombosis'` 与
   `degree_of_thrombosis='severe'` 混淆），评分阶段直接记零。
3. **重复工具调用浪费预算**。模型在长上下文中遗失"已检视过该文件"的事
   实，反复 `read_csv` 同一 CSV、反复 `list_context`，每次重复均产生数千
   token 开销。
4. **`knowledge.md` 规则被忽略或误用**。模型默认按通用先验作答，而 KDD
   任务中 `knowledge.md` 经常重新定义术语（例如"long shot"、"qualifying
   driver"），偏离规则即记零。
5. **明确程序化任务上 React 路径过长**。对一道清晰的 join + aggregate
   题，React 需 6-8 步铺垫才进入计算，相比直接生成完整程序明显低效。
6. **预算耗尽即视为失败**。模型常在最后一步已得到正确答案但因 self-verify
   再开销而触发 `BudgetExceeded`，整个任务被记为 failure，浪费已完成的
   全部工作。

### 1.2 改造路线总览

围绕上述问题，本工作对 starter kit 做了如下系统性改造，括号内为对应的
后文章节：

- 将路由决策与任务画像从模型决定改为确定性规则决定（§2）。
- 将 React 工具协议从"文本 JSON"切换为"OpenAI Native Tool Calling"，
  并增加 `tool_call_id` 重放、文本模式回退、prompt 一致性约束等配套（§3.2）。
- 在 React 循环中加入"只读工具结果缓存"机制，并维持透明 trace 契约（§3.3）。
- 在 React `answer` 提交点增设确定性的 Cheap Answer Guard，按证据触发
  self-verify 升级（§3.4）。
- 将工具异常翻译为结构化 retry hint，并对重复错误签名切换提示模式（§3.5）。
- 将"解读 knowledge.md"独立为 `consult_knowledge` helper LLM 工具（§3.6）。
- 在 `BudgetExceeded` 路径中保留 pending answer（§3.7）。
- 增加 Operator 备选路径：Schema Grounding + Semantic Analyst + Codegen
  + Cheap Semantic Guard + 三档 Repair（§4）。
- 修复 Router 中 `-1`（无上限）预算被错误覆盖为 compiler 建议值的 bug（§5.2）。
- 增加 Self-Consistency 列级签名投票、Reasoner Repair、Answer Validator
  三层后置兜底（§6）。
- 将 `trace.json` 设计为结构化契约而非追加式日志，约定下游模块的读写字
  段（§7）。

整体架构如下：

```
  ┌─────────────────────────────────────────────────────────────┐
  │  CLI (cli.py) → Runner (run/runner.py)                      │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
              TaskCompiler  (agents/task_compiler.py)       ← 本工作新增, 无 LLM
                                 ▼
              Router        (agents/router.py)              ← 本工作改造为规则路由
                                 │
                  ┌──────────────┴───────────────┐
                  ▼                              ▼
        React Harness (主)                Operator 路径 (本工作新增)
        agents/react.py                   agents/agentic_operator.py
        + react_planner                   + operator_executor
        + react_answer_guard              + tablellm_direct
        + react_retry_hints               + schema_grounding
        + tool-result cache               + static_checker
        + consult_knowledge helper        + semantic_guard
                                          + semantic_consistency
                                          + repair_coordinator
                  └──────────────┬───────────────┘
                                 ▼
              工具运行时 (tools/, budget.py, model.py)
                                 ▼
              后置兜底 (本工作新增)
              ├─ Self-Consistency  (run/self_consistency.py)
              ├─ Reasoner Repair   (agents/reasoner_repair.py)
              └─ Answer Validator  (eval/answer_validator.py)
                                 ▼
                AnswerTable / prediction.csv / trace.json
```

---

## 2 任务编译与路由：把决策从模型手中拿走

starter kit 没有显式的路由层——所有任务走同一条 React 循环。本工作判断
KDD 任务形态差异显著（document_qa、record_text、纯表格、混合上下文等），
单一路径难以同时高效。但本工作没有沿用"让模型自己选择路径"的方案，理由
是路径失败时的归因将与模型行为耦合，无法区分"路径不合适"与"模型当时表
现不稳"。基于此，本工作把"任务画像"和"路由决策"两步均改为确定性规则。

### 2.1 TaskCompiler：决策前的事实快照

本工作新增 `agents/task_compiler.py`，在任务进入路由前生成一份
`CompiledTask`，全部由文件扫描和规则启发式产出，**不发起任何 LLM 调用**。
关键字段如下：

| 字段 | 含义 | 后续依赖者 |
| --- | --- | --- |
| `task_type` | `table_computation` / `table_with_semantic_rule` / `record_text_with_semantic_rule` / `document_qa` / `mixed_context` / `image_understanding` / `pure_reasoning` | Router、prompt 构造 |
| `answer_type` | `boolean` / `scalar` / `table` | Cheap guard、Answer Validator |
| `source_capabilities` | 各文件的真实 schema、行数、低基数列样本、SQLite 表结构 | 所有下游模块共享 |
| `operations` | filter / join / aggregate / compute 等候选操作集合 | codegen 与 guard |
| `ambiguity_flags` | `ambiguous_schema` / `large_context` / `record_text_context` / `unsupported_file` 等 | 风险预算 |
| `budget_level` | `small` / `medium` / `large` / `xlarge` | BudgetController |

本工作要求该层稳定性优先于精度——精度损失可在下游通过 cheap guard 补
救，但若不同模块对 schema 的认知不一致，将产生难以归因的漂移型故障（例
如 codegen 引用了 static checker 不认识的字段）。`source_capabilities`
被强约定为下游所有 LLM 模块共享的同一份参照系。

### 2.2 Router：基于任务形状的规则路由

`agents/router.py:_route_for_compiled_task()` 接收 `CompiledTask`，按
下列优先级匹配 YAML 中的 `route` 表：

```
task_type   →   source_shape   →   operation_complexity   →   ambiguity_flags
```

YAML 中由用户命名 route（例如 `react_harness`），实际执行器由
`RouteConfig.kind ∈ {react, agentic_operator, operator_executor,
tablellm_direct, multi_agent}` 决定。本工作进一步增加 **cascade
fallback**：YAML 配 `cascade_order`，首选路径产出非法 answer 或被 Answer
Validator 拒收时，依次尝试后备路径，所有尝试现场写入
`trace.json.router_decision`。这一设计的代价是 YAML 配置稍重，收益是失
败案例的复盘路径完全确定，可与代码 diff 对齐归因。

---

## 3 React Harness：主控制环的逐项改造

本工作对 starter kit 的 React 循环做了较深的改造，最终形态由
`agents/react.py:ReActAgent` 及其周边模块构成。术语 "harness" 取其工程
语义：一个承载工具调用、生命周期管理、错误恢复与可审计性的统一框架。
单步骨架如下：

```
        ┌────────────────────────────────────────────────────────┐
        │  pre-loop: React Planner (单次 LLM 调用, advisory)      │
        └─────────────────────────────┬──────────────────────────┘
                                      ▼
                    for step in range(max_steps):
                                      │
                                      ▼
              model.complete_with_tools()
              (native tools API + JSON Schema 校验)
                                      ▼
                       ToolRegistry.execute()
                                      │
              ┌───────────────────────┼────────────────────────┐
              ▼                       ▼                        ▼
        cache hit               普通工具返回             terminal = answer
              │                       │                        │
              └───────────┬───────────┘                        ▼
                          ▼                          Cheap Answer Guard
                  build observation                          │
                  (含 retry_hint 若失败)                    risk?
                          │                       ┌─────┴─────┐
                          ▼                       否           是
                  append StepRecord               ▼            ▼
                  (含 tool_call_id)            commit     注入 verify
                                                          observation
```

下文逐项说明本工作做了哪些改造，以及为什么。

### 3.1 增加预规划层以减少冷启动开销

**观察**：在 medium/hard 任务上，React 模型前 3 步常重复执行
`list_context` 与 `read_csv`，并未形成可执行的解题思路；尤其在长上下文
任务中，模型频繁回到"我现在还不知道有哪些文件"的状态。

**改造**：本工作新增 `agents/react_planner.py`，在 React 循环开始前发起
一次独立 LLM 调用，要求模型输出不超过 6 项的有序子任务列表（含 `goal` /
`suggested_tools` / `reads`），作为后续每一轮 user message 的固定前缀。

**关键决策**：

1. **建议性而非约束性**。本工作最初尝试让 plan 强制约束后续工具调用顺
   序，实测发现循环中途发现新事实（例如字段名实际不同）时反馈通道丧失，
   反而比无 plan 更脆弱。最终改为仅作 prompt 前缀，模型自行权衡是否遵循。
2. **软降级**。Plan 调用失败、解析失败或返回零子任务时，循环按无 plan
   照常运行，仅在 `state.plan.error` 字段记录原因。Harness 在没有 plan
   的情形下可独立工作，避免新增组件成为新的失败源。
3. **成本守门**。本工作设 `PlannerConfig.skip_for_easy=True`，对
   `difficulty==easy` 的任务直接跳过预规划，避免简单任务因启动开销而
   损害平均得分——简单任务的瓶颈不在规划。

### 3.2 切换至 Native Tool Calling 并处理由此引入的协议约束

**观察**：starter kit 让模型在 assistant 消息文本中输出 ```json fenced
block，外部用正则解析。在 6 KB 以上的复杂 observation、嵌套字符串、转
义符、偶发不匹配括号的输出上不稳定，内部评估约 10% 的步骤因解析失败而
被浪费。

**改造**：本工作改用 OpenAI Chat Completions 的 **native tools API**。每
个工具携带严格 JSON Schema（`tools/registry.py:ToolSpec.parameters`），
含 `type` / `required` / `additionalProperties:false` / `minimum` /
`maximum` / `default`。服务端在 LLM 返回前完成参数结构校验，agent 内部
原本针对参数合法性的防御性代码因此可大量删除。

但切换至 native API 引入了三个非显性约束，本工作需逐一处理：

**约束一：`tool_call_id` 必须配对**。OpenAI 与 DashScope 均强制要求
assistant 消息中的每个 `tool_calls` 在下一条 `tool` 消息中有完全匹配的
`tool_call_id`，否则请求被服务端拒绝。这意味着若 `tools.execute()` 内部
抛出异常，简单地把 step 标记为 `__error__` 会使下一轮重放时该 tool_call
缺少回复，整次会话失效。本工作在 `react.py` 的两个 outer-except 分支中
保留原始 `action` 与 `action_input`：

```
if tool_call_id is not None:
    record_action = failed_action
    record_input  = failed_action_input
```

StepRecord 由此携带真实 action 与 tool_call_id 写入状态，下一轮
`_build_messages()` 重放出合法的 `assistant(tool_calls=…)` +
`tool(content=error msg)` 配对，会话得以继续。

**约束二：system prompt 与 tools API 必须风格一致**。本工作在 DashScope
上的 Qwen 模型上实测发现：即使已启用 native tools API，若 system prompt
仍含有"请始终返回 ```json fenced block"等文本模式输出指令，模型会顺从
prompt 输出文本 JSON 而绕过 tools API。内部 smoke test 显示：prompt 修
改前 native 命中率为 0/3，修改后回升至 3/4。本工作在
`agents/prompt.py:build_system_prompt()` 增加 `use_native_tool_calls=True`
分支，移除 fenced-JSON 指令并切换至 `TOOL_API_OUTPUT_RULES` 变体。

**约束三：必须保留文本模式回退**。即使 native 启用，模型仍偶有返回纯文
本（无 `tool_calls`）的情况。本工作保留 `parse_model_step()` 的文本解析
路径，并增设三段恢复（JSON 转义修复 → 控制字符替换 → 缺失闭合括号补
足）。OpenAI 与 DashScope 均接受同一 transcript 中部分轮次 native、部
分轮次文本，该回退构成稳定安全网而非临时补丁。

### 3.3 加入只读工具结果缓存

**观察**：内部 trace 显示模型在 long-context 任务中反复 `read_csv` 同一
CSV、反复 `list_context`，根因是 6 KB observation 上限截断后的不同片段
被模型误判为新信息。每次重复消耗数千 token，并未带来新事实。

**改造**：本工作在 `agents/react.py:_CACHEABLE_TOOLS` 圈定一组只读、纯
函数性质的工具（`list_context` / `read_csv` / `read_json` / `read_doc` /
`head_doc` / `grep_doc` / `inspect_sqlite_schema` /
`execute_context_sql` / `consult_knowledge`），返回结果以下式作键缓存于
循环内：

```
cache_key = action + "::" + json.dumps(action_input, sort_keys=True)
```

`execute_python` 显式排除——模型可能依赖打印副作用区分中间状态；
`answer` 为终止器，缓存其结果会触发误终止。

**关键约束：透明缓存契约**。缓存命中时本工作仍构造一条
`observation={"cached": True, "content": cached["content"]}` 的
StepRecord 写入 `state.steps`，并通过 progress logger 报告 `cached=True`
事件。原因是 harness 的所有下游模块（Cheap Answer Guard、Retry Hints、
Reasoner Repair）均从 `state.steps` 反向重建 agent 行为轨迹；若缓存命
中即吞步，guard 将错误地判定"模型未曾检视相关字段"。性能优化的实现细节
不应破坏 trace 的语义完整性，本工作将此作为 harness 与下游评估模块的核
心契约固化在代码注释中。

### 3.4 Cheap Answer Guard：以证据触发 self-verify 升级

**观察**：starter kit 无任何答案校验，"形状对但内容错"类失败完全沉默。
直觉上的修复是"无条件二次确认"，但本工作的早期实验显示该方案有两个问
题：(a) 对绝大多数本可一次通过的任务是纯粹成本浪费；(b) 模型在重检时
有非零概率将已正确答案"修正"为错误答案，引入回归型故障。

**改造**：本工作设 `ReActAgentConfig.verification_rounds=0` 为默认，并
新增 `agents/react_answer_guard.py:assess_react_answer_risk()` ——一个
**纯确定性**的风险评估器，在第一次 `answer` 调用进入时无条件运行。该函
数扫描 trace 中的具体证据：

- trace 中是否出现 `execute_python` 或 `execute_context_sql`：若否，模
  型可能未经计算给出答案；
- `knowledge.md` 存在但 trace 中无相应读取记录：模型可能错过关键规则；
- `answer.columns` 数量与 `CompiledTask.answer_type` 是否兼容（`scalar`
  和 `boolean` 应为单列）；
- 问题中的关键概念词是否曾出现在任何工具调用的输入中。

`guard_score` 为上述信号的加权和，超过 0.5 阈值或命中任意 error 级风险
时 `should_escalate=True`。此时 harness 将 `required_answers` 临时提升
至 2，拦截当前 answer，注入一份携带 `guard_risk_codes` 与
`guard_top_risk` 的 verify observation。模型据此有针对性地复查，而非接
收通用的"请重新检查"提示。

**效果**：内部评估观察到超过 80% 的 answer 经此路径一次通过且零额外开
销；触发升级的 answer 中超过 60% 在二次提交时修正了字段映射或公式。

### 3.5 Retry Hints：将工具异常外化为结构化提示

**观察**：React 循环中模型自省能力不可靠——常见在重复错误模式下耗尽预
算，例如反复尝试同一个不存在的列名。

**改造**：本工作新增 `agents/react_retry_hints.py`，在两个层面发挥作用：

1. **结构化建议生成**。将异常类型与上下文翻译为针对性提示。例如：
   `answer` 工具行宽不匹配时，hint 精确指出"声明 3 列，第 2 行包含 2
   个值"；`execute_python` 报 `KeyError: 'foo'` 时，hint 附上实际可用
   列名摘要。
2. **错误签名去重与模式切换**。`ErrorHistory` 记录每种错误签名的出现次
   数；同一签名第二次触发时，hint 的措辞从"修复该错误"切换为"采用替
   代策略"。例如多次 `KeyError` 时建议先调用
   `read_csv(columns_only=true)` 而非反复猜测列名。

本工作将"识别重复失败模式"这一能力以确定性规则外化，不依赖模型自身具
备反思能力——这是 starter kit 的一项隐含但有问题的假设。

### 3.6 把 knowledge.md 解读独立为 helper LLM 工具

**观察**：KDD 任务中 `knowledge.md` 经常以自然语言定义高度领域化的术语
（例如 "qualifying driver"、"long shot"）。若每轮 prompt 都携带 knowledge
全文，长任务的上下文窗口被持续占用；若不携带，模型默认按通用先验作答
而偏离规则。

**改造**：本工作把"解读规则"独立为一项工具调用。`tools/knowledge.py:HelperRuntime`
持有独立的 `ModelAdapter`（可配置为不同模型与温度），由
`tools/registry.py` 条件注册为 `consult_knowledge` 工具。主循环中的模
型可发起：

```
consult_knowledge(question="...", files=["knowledge.md"])
```

Helper 在后端读取指定文件、执行一次独立 LLM 推理、返回简短事实摘要。

**关键决策**：

- 主循环仅在需要时咨询，不在每轮 prompt 中重复携带 knowledge 全文。
- 咨询结果以工具 observation 形式进入 trace，与其它工具结果同等可缓
  存、可审计。
- 主模型与 helper 可为不同模型——例如本工作的典型配置是 helper 使用更
  强模型解析规则，主循环使用更便宜的模型执行工具。
- 调用计数纳入主 `BudgetController`，多 LLM 不会突破总预算。

Helper LLM 与 Planner 均经 `model.complete()` 文本通道，不参与 native
tool calling。本工作认为 native tools API 适用于"对真实数据采取行动"
的调用，不适用于"对规则进行解读"或"对计划进行规划"的元 LLM 调用。

### 3.7 预算敏感的错误恢复

**观察**：starter kit 在 `BudgetExceeded` 抛出时直接将任务标记为
`failure_reason="budget exceeded"`。但内部 trace 显示模型可能已在前一
步提交了 draft answer，仅因 self-verify 中途预算耗尽而未最终提交——此
时把整个任务记为失败浪费了已完成的全部工作。

**改造**：本工作在 `react.py` 的两处 `BudgetExceeded` 捕获块中加入：

```
except BudgetExceeded as budget_exc:
    if pending_answer is not None:
        state.answer = pending_answer
        state.failure_reason = "budget_exceeded_during_self_verify; committed draft."
        break
    raise
```

改动很小，但能挽救一类典型回归——"正确路径已识别但预算在校验阶段耗尽"。

---

## 4 Operator 路径：本工作新增的程序合成备选

### 4.1 引入 Operator 路径的动机

**观察**：本工作发现 React Harness 对"明确程序化（清晰 join + aggregate）"
任务存在显著开销——模型需若干步完成 `list_context`、`read_csv`、
`inspect_schema` 之后才能进入实际计算，而正确路径其实是一段约 30 行的
pandas 程序。

**改造**：本工作新增 Operator 备选路径（`agents/agentic_operator.py:AgenticOperatorExecutor`），
直接生成完整 Python 程序，执行并校验。Router 在 `CompiledTask` 显示任
务具有清晰程序化语义（明确的 join / aggregate / 输出列）时优先派发该路
径。该路径包含五个核心子模块，下文逐个说明它们解决的问题。

### 4.2 Schema Grounding：避免概念词与列名的错绑

**观察**：本工作在错误样本中发现一类高频失败——模型理解了任务，但绑定
了不存在或近似但错误的列名。典型例子：题目问 "patients with severe
degree of thrombosis"，真实表同时含 `disease` 与 `degree_of_thrombosis`，
模型易将 `thrombosis` 绑定为 `disease='thrombosis'`，而非
`degree_of_thrombosis='severe'`。

**改造**：本工作新增 `agents/schema_grounding.py`，从问题中抽取概念词，
按"字段名相似度 + 低基数样本值匹配 + dtype + cardinality"组合打分，输
出概念到候选字段的映射清单，注入 codegen prompt。

**关键决策**：Grounding 仅产出**候选提示**，不做终判。本工作早期尝试强
制绑定（直接把概念词替换为最高分字段），结果将概率事件升级为系统性故
障——一旦绑定错误，整个程序无法自愈。最终改为保留候选、由 codegen 选
择、由 guard 验证，三阶段分离。

### 4.3 Cheap Semantic Guard：与 React 路径的设计对仗

**观察**：Operator 路径的失败模式与 React 路径不同——程序可能执行成功
但语义错误，例如使用了不存在的字段、忽略了 knowledge.md 中的过滤条件、
或修复路径过深表明早期决策已偏离。

**改造**：本工作新增 `agents/semantic_consistency.py:SemanticConsistencyPipeline.plan()`
担任"语义分析"角色——仅产出 semantic plan（字段映射、filter、aggregation、
输出列、不确定点、plan confidence），不生成代码。配合
`agents/semantic_guard.py:assess_semantic_risk()`（**Cheap Semantic Guard**）
在程序执行完毕后做风险评估。该 guard 与 Cheap Answer Guard 共享设计思
路——确定性、轻量、基于具体证据触发升级。其检测信号包括：

- 程序未产出非空答案；
- 是否经 `local_repair` 或 `schema_retry` 修复后才成功（修复路径越
  深，风险权重越高）；
- `debug_steps` 中缺失 schema_inspection / used_columns / filter / join
  痕迹；
- 程序使用的字段不在真实 schema 中；
- 关键风险词所对应的 grounded 字段未被程序实际使用；
- 使用了问题未要求的派生指标或公式。

**按需启动**：对低复杂度表格题（`table_computation` + `direct_lookup`
或 `single_filter`，且无 semantic rule 要求），本工作让 plan 阶段默认跳
过，直接进入 fast path；仅在 guard 累计风险分超过 0.5 或命中任一 error
级风险时，触发 `plan(force=True)` 补做语义审题，并进入 consistency
judge 与 semantic repair 流程。

**双 Cheap Guard 的设计对仗**：本工作中存在两个 Cheap Guard——Cheap
**Answer** Guard 守在 React 路径的 `answer` 提交点（§3.4），Cheap
**Semantic** Guard 守在 Operator 路径的程序执行点。两者均为非 LLM 的确
定性预过滤器，定位为"低风险任务零成本放行、高风险任务升级到昂贵的 LLM
校验"。这一双层结构在保证语义校验覆盖率的同时控制了整体的 LLM 调用开销。

### 4.4 Codegen 与 `debug_steps`：把可审计性写入代码

**观察**：仅以"程序执行成功"为通过条件无法识别"程序运行但语义错误"
类故障——这正是 §4.3 的 Cheap Semantic Guard 需要的输入。

**改造**：本工作在 `agents/tablellm_direct.py` 的 codegen prompt 中要求
模型在生成程序中显式记录以下字段：

```
schema_inspection      实际读取到的列
used_columns           最终使用的字段
filter_conditions      过滤条件
join_keys              join 键
intermediate_counts    各阶段剩余行数
knowledge_rules_used   使用的规则或公式
plan_override          覆盖 semantic plan 时的理由
```

本工作把 `debug_steps` 定位为契约字段而非日志——后续 Cheap Semantic
Guard 与 Consistency Judge 均依据该字段判断"程序行为是否与其声称一
致"。这一契约把语义校验从"读懂代码 AST"降级为"读取若干结构化字段"，
guard 复杂度因此可控。

### 4.5 Static Checker 与 Repair Coordinator：三档梯度修复

**观察**：让 LLM 直接重写失败的程序成本高且不稳定——LLM 重写虽能力较
强，但有非零概率在修复一项错误的同时引入新错误。

**改造**：本工作新增两个组件协同工作：

`agents/static_checker.py` 在程序实际 `exec()` 前进行静态扫描，覆盖问题
类别：`no_such_file` / `no_such_table` / `no_such_column` /
`bad_join_key` / `python_syntax`。每个问题产出 `StaticIssue` 结构体交予
修复层，避免浪费真实执行并将错误转化为结构化提示。

`agents/repair_coordinator.py` 将修复组织为三档梯度：

```
1. local_repair_loop                    确定性 AST / 正则补丁
                                        （列名近似、JSON records key 缺失、
                                         输出格式修正等）
2. schema_retry                         附带 schema diagnostics、static
                                        issues、错误日志的 LLM 重写
3. post_schema_retry_local_repair_loop  对 LLM 重写结果再执行本地修复
```

**关键决策**：本工作先采用低成本、确定性、不引入新错误的本地修复；本地
修复失效后才升级至 LLM 重写；末端再次执行本地修复以回收 LLM 重写引入
的增量错误。该梯度结构与 §3.4 的 Cheap Answer Guard 升级机制是同一思路
的不同实例——确定性组件优先，LLM 调用作为升级路径。

### 4.6 Structured Doc Executor：把文本理解与表格计算解耦

**观察**：`record_text_with_semantic_rule` 任务的上下文为长自然语言段落
（病历、新闻、推文），但目标答案为表格。让 pandas 代码直接在非结构化文
本中检索字段稳定性很差。

**改造**：本工作新增 `agents/structured_doc_executor.py`，先由 LLM 把段
落抽取为结构化 records，合成临时 CSV，再交回标准 codegen 路径执行表格
计算。本工作将"文本理解"与"表格计算"两类能力解耦，使每个阶段可独立
验证。

---

## 5 工具运行时：协议一致性与若干 bug 修复

### 5.1 ToolRegistry 双视图：保证 prompt 与 API 一致

**改造**：本工作在 `tools/registry.py:ToolRegistry` 中为每个工具维护两
份描述，均派生自同一 `ToolSpec`：

- `describe_for_prompt()`：人类可读的示例 `input_schema`，拼装入 system
  prompt 用于 in-context learning；
- `describe_for_tool_api()`：严格 JSON Schema，拼装入 `tools=[{type:
  "function", function:{name, description, parameters}}, ...]` 提交至
  native API，由服务端执行参数校验。

**为什么共享同一 `ToolSpec`**：避免"prompt 声称工具叫 `read_csv` 但 API
注册的是 `readCsv`"类的隐性漂移。本工作早期分别维护两份描述时确曾出现
过此类不一致，导致一段时间内 native tools API 几乎完全失效。最终强约定
两份描述派生自同一来源，任何工具变更同时反映到两侧。

### 5.2 修复 Router 中的预算覆盖 bug

**观察**：用户在 YAML 中显式设置 `max_llm_calls: -1`（无上限）后仍被
截断为约 22 次调用，导致 hard 任务大量假性失败。

**根因**：Router 的 `_budget_limit()` 旧实现把 -1 替换为 compiler 建议
值，破坏了 `BudgetController` 的"负值表示无上限"约定。

**修复**：本工作改为 `-1 → -1` 透传，遵循 `budget.py:BudgetController`
的原始约定（`if self.max_*_calls >= 0 and self.*_calls >= self.max_*_calls:
raise`）。预算耗尽抛出 `BudgetExceeded` 后，React Harness 按 §3.7 的策
略保留 pending answer 后退出。

### 5.3 Observation 大小约束与执行隔离

`tools/filesystem.py` 中所有文件系工具统一遵守 `_MAX_OBSERVATION_CHARS=
6000` 的硬上限，超出部分截断并标记 `truncated=true`；`execute_context_sql`
接受 `limit` 参数（默认 200 行）；`execute_python` 运行于独立
`multiprocessing.Process`（`tools/python_exec.py`），超时固定 30 秒，工
作目录强制为 `task.context_dir`，子进程被 terminate 时仍可收集已产生的
stdout/stderr。

**已识别但未修复的项**：`execute_python` 当前未截断 stdout/stderr。模型
若不慎打印宽表可产生数 MB 的 tool message，在 native tool calling 下尤
易触发服务端单消息大小限制。计划增加 stdout 16 KB / stderr 4 KB 上限，
与文件系工具的 cap 对齐。

---

## 6 多样性与最终兜底：本工作新增的三层后置

### 6.1 Self-Consistency：列级签名投票

**观察**：朴素的"对整张 AnswerTable 多数投票"在列顺序不一致时会将等价
答案误判为不同，损害投票质量。

**改造**：本工作在 `run/self_consistency.py` 实现列级签名投票，把投票
粒度对齐到评分粒度：

```
score  = max(0, recall - lambda * extra_cols / pred_cols)
recall = matched_gold_cols / total_gold_cols
```

KDD 评分按列内容签名匹配，忽略列名与行顺序。因此投票亦在列粒度执行：
(1) 对每个样本的每一列计算 `column_signature`（数值四舍五入、字符串归
一化后构造多重集签名）；(2) 跨样本统计每个列签名的出现频次；(3) 选出
"覆盖最多多数列签名、冗余列数最少"的样本作为最终答案。

**关键决策**：把投票函数与评分函数的等价类对齐——这是该模块设计成立的
前提。聚合器另支持 `first_success` 模式（取首个产出合法 `AnswerTable`
的样本），节省预算。两种聚合器共用同一 `BudgetController`，多样性不会
突破总预算。启用条件为 `agent.self_consistency.num_samples > 1`。

### 6.2 Reasoner Repair：跨路径的最后一手

**观察**：工具循环失败后重新跑工具循环往往复现同一失败模式——模型在相
同 trace 上倾向产出相同决策。

**改造**：本工作新增 `agents/reasoner_repair.py`，在主路径返回失败、答
案为空、或最终 `AnswerTable` 被 Answer Validator 拒收时触发。该模块接
收完整 trace、当前 answer 与错误现象，要求**配置中的更强模型**直接重写
答案，**不再执行工具循环**。

**关键决策**：工具循环执行"逐步收集事实"，Reasoner Repair 执行"综合
所有事实做出判断"——本工作把两者视为能力互补的两个推理模式。前置工具
循环失败时改换"综合推理"路径可获取互补能力。

### 6.3 Answer Validator：提交前的硬性结构检查

`eval/answer_validator.py` 在最终提交前做结构合法性检查，覆盖不变式包
括 `empty_columns` / `empty_rows` / `empty_column`（整列为空）/
`ragged_rows`（行宽不一致）/ `mixed_types`（warning 级）/
`invalid_answer_type`。验证失败的 `AnswerTable` 不写入 `prediction.csv`，
由 Router 标记为需进入 fallback 路径。

本工作把 Answer Validator 定位为纯结构层的硬性检查——上游所有语义校验
不可能完整覆盖结构错误，此模块作为最后一道防线保证落地 CSV 至少格式合
法。

---

## 7 可审计性：把 trace.json 设计为结构化契约

starter kit 的 trace 是追加式日志——记录所有步骤，但下游模块（评估、
guard、judge）需各自解析。本工作将 `trace.json` 重新设计为**结构化契约**
——各 agent 子模块负责写入约定字段，下游模块严格按字段读取。稳定字段
如下：

| 字段 | 内容 |
| --- | --- |
| `task_id` / `succeeded` / `agent_mode` | 基本标识 |
| `compiled_task` | TaskCompiler 完整输出（含 source_capabilities） |
| `router_decision` | 首选 route、cascade 尝试序列、各 route 失败原因 |
| `budget` | LLM / 工具调用计数及上限 |
| `manifest[]` | 各 route pass 的内部细节：steps（含 raw_response、observation、tool_call_id、retry_hint）、program、debug_steps、guard、judge |
| `self_consistency` | 启用时所有样本的 answer 与投票结果 |
| `reasoner_repair` | 触发时的输入 trace 与重写后 answer |
| `answer_validation` | 最终结构校验结果 |
| `local_score` | 评分后的本地复盘分数 |

本工作据此固化了失败 case 的标准复盘顺序：

```
router_decision.compiled_task.task_type     任务被分至错误类型？
  ↓
source_capabilities                          真实文件与列扫描完整？
  ↓
schema_grounding / cheap_guard.grounding     问题词到字段映射正确？
  ↓
manifest[].steps                             harness 真实执行了哪些工具？
  ↓
program / OPERATOR_CODEGEN_DEBUG.debug_steps codegen 生成了什么代码？
  ↓
cheap_semantic_assessment.risks              guard 升级或未升级的原因？
  ↓
semantic_consistency.judge_history           judge 识别的不一致点？
```

该顺序较直接审视 `prediction.csv` 更能将错误定位至具体阶段。

---

## 8 评分函数与实验结果

### 8.1 评分约定

`eval/column_match.py:score_table()` 实现 DataAgent-Bench 官方语义的列匹
配评分：

```
score   = max(0, recall - lambda * extra_pred_cols / pred_col_count)
recall  = matched_gold_cols / total_gold_cols
lambda  = 0.5  (默认)
```

列匹配按值的多重集签名比对，忽略列名与行顺序，数值小数容差，字符串大
小写与空格归一化。该评分函数对预测结果存在强约束：**严禁多输出无关列**。
本系统的所有 prompt 均明确要求"仅输出问题所需列"，Cheap Answer Guard
的列数检查、Operator 路径的 plan-judge 检查均承担该约束的执行。

### 8.2 性能数据

> 以下表格留待提交前根据最终批量评估结果填入。

| 评测集 | 任务数 | Easy 平均分 | Medium 平均分 | Hard 平均分 | 总平均分 |
| --- | --- | --- | --- | --- | --- |
| Public dev |   |   |   |   |   |
| Hidden test |   |   |   |   |   |

### 8.3 主要观察

> 留待提交前根据最终批量运行结果补充。建议覆盖如下维度：
>
> - 各 `task_type` 的得分分布，及 Cheap Answer Guard 升级对得分的影响；
> - `ambiguity_flags` 与失败率的相关性；
> - Self-Consistency 在不同 `num_samples` 下的边际收益拐点；
> - Reasoner Repair 触发率与挽救率；
> - Native Tool Calling 命中率与文本模式回退触发率。
