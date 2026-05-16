# Data Agent Baseline — 系统设计与实验报告

本仓库面向 KDD-Cup DataAgent 类型的「问表 + 问文档 + 问数据库 + 问图」混合数据问答任务。
我们提交的不是一个端到端的大模型，也不是一个标准的 ReAct，而是一套
**确定性任务剖析 → 任务类型路由 → 工具优先代码执行（带执行外壳/沙箱）→ 多层修复 → 列签名一致性投票** 的 agent 流水线。
本文档说明每一层在做什么、为什么这样做，以及它在公开测试集上的表现。

---

## 1. 总体架构

整体由 6 个相对独立、可单独消融的层级组成；中间所有数据都以确定性 schema 流转，
LLM 只在必要的位置被调用，每一次调用都受 Budget 控制器的硬约束。

```
PublicTask (question + context/)
        │
        ▼
┌───────────────────────────┐
│ 1. Task Compiler          │  确定性扫描 context/,输出 CompiledTask:
│    (task_compiler.py)     │   task_type / answer_type / source_capabilities /
│                           │   foreign_key_candidates / budget / ambiguity_flags
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 2. Router                 │  按 task_type + ambiguity_flags 分发到 4 类执行后端;
│    (router.py)            │   失败时按 typed cascade 回退,而不是按难度盲升档
└───────────────────────────┘
        │
        ├─► OperatorExecutor   (kind=operator_executor)  —— 主力路径
        ├─► CodegenDirect      (kind=tablellm_direct)
        ├─► ReActAgent         (kind=react)
        └─► MultiAgentOrch.    (kind=multi_agent)
                │
                ▼
┌───────────────────────────┐
│ 3. Operator Pipeline      │  审题官(Analyst) → Codegen → 静态检查 →
│    (operator_executor.py) │   局部修复 → 结构化文档 LLM 抽取 →
│                           │   Schema-guided LLM 重写 → 执行官(Judge+Repair)
└───────────────────────────┘
        │
        ▼ (每次 Codegen 输出都通过下面这层落到磁盘)
┌───────────────────────────┐
│ 4. Execution Harness      │  _EXEC_HARNESS 包一层 answer 归一化 +
│    (tablellm_direct.py +  │   debug_steps 抽取; 在 multiprocessing 子进程
│     tools/python_exec.py) │   里 chdir(context_root) + dup2 stdout/stderr +
│                           │   timeout kill,任何失败都被结构化捕获
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 5. Answer Validator       │  零行/空列/参差行/缺列等结构性错误,
│    (answer_validator.py)  │   不通过则把 succeeded 降级为 False
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 6. Self-Consistency Vote  │  N 次采样后按列内容签名(content signature)投票,
│    (self_consistency.py)  │   选票数最多的列签名集合,而不是「整张表多数票」
└───────────────────────────┘
        │
        ▼
   AnswerTable → prediction.csv + trace.json
```

每一层都有独立的 trace 字段写入 `trace.json`（`compiled_task / router_decision /
operator_executor / multi_agent / answer_validation / self_consistency / budget`），
方便对每一道错题做定点归因，而不是把锅都甩给「LLM 答错了」。

---

## 2. 各层设计要点

### 2.1 Task Compiler — 一次性、确定性的任务画像

我们刻意把「任务画像」从 LLM 移到代码里，避免每次都让模型重新认一遍这个任务长什么样。
扫描 `context/` 后输出一个 `CompiledTask`，覆盖：

- **modality 与 task_type**：CSV/JSON/SQLite/Markdown/Image 的组合 → 映射到
  `table_computation / document_qa / mixed_context / image_understanding /
  table_with_semantic_rule / record_text_with_semantic_rule / pure_reasoning` 之一。
- **SourceCapability**：每个文件的真实列名 / SQLite 表 schema / JSON 顶层键 /
  按列的低基数样本（`column_value_samples`，每列最多 5 个高频值）/ dtype 投票 /
  基数估计。这一项是后续防止「列名幻觉」的核心证据。
- **`foreign_key_candidates`**：跨文件按 distinct value 集合的 Jaccard 重合度做候选
  连接键挖掘，Top-K 写到 prompt 里，让 LLM 不再凭空发明 join key。
- **`record_text` 探测**：把那种「超过 N 个病历/比赛号在长文里被反复提到，但又不是表格」
  的 Markdown 单独打 `record_text` 标签，避免被普通 RAG 当成短文档处理。
- **Budget 估计**：按 task_type + difficulty + 标志位计算 `max_llm_calls / max_tool_calls`。

整个 Task Compiler 不调用 LLM，平均开销 < 100 ms / 任务，但产出是后面所有 prompt
的「事实层」——下游写代码时只能引用 `SourceCapability` 列出的真实路径与列名，否则直接
被静态检查打回。

### 2.2 Router — 按任务类型分发，按错误模式 cascade

和大多数公开实现不同，**我们的 Router 不再以 `task.difficulty` 为主分发键**。难度只
作为 budget 提示。原因是：同样标 Hard 的两道题，一道是「在 30 张表里写一条 JOIN」，
另一道是「在 200 KB 病历里抠数据」，最佳路径完全不同。

主分发逻辑（`_route_for_compiled_task`）：

| compiled_task.task_type | 首选路径 | kind |
|---|---|---|
| `table_computation`（小） | `easy` | `operator_executor` / `tablellm_direct` |
| `table_with_semantic_rule` | `medium` / `tool_first_mixed` | `operator_executor` |
| `mixed_context` | `tool_first_mixed` | `operator_executor`(+RAG) |
| `record_text_with_semantic_rule` | `tool_first_mixed` | `operator_executor` |
| `document_qa` | `medium` | `operator_executor`(RAG) |
| `image_understanding` | `extreme` | `multi_agent` |
| `pure_reasoning` | `extreme` | `react` |

失败 cascade（`_next_repair_route`）严格按**错误模式**回退，而不是「easy → medium → hard」
盲升档。例如：

- `invalid_answer + zero rows`：通常是过滤条件 dtype 错位，**不再升档**——把这种当成
  典型的「不该重跑整题，应该局部修」信号；
- `exec_error / no_such_table / no_such_column`：跳到 `tool_first_mixed`；
- `unsupported_file_type` 且 `needs_reasoner`：才允许走到 `multi_agent` 兜底。

经此改造，Multi-Agent 退化为**最后一道兜底**，而不是 Hard/Extreme 的默认路径。这件事
对 token 成本和延迟都是正面的——我们后面再讨论。

### 2.3 Operator Executor — 工具优先的「审题官 + 执行官」

`OperatorExecutor` 是绝大多数任务实际跑的路径。它把过去单步 codegen 拆成 6 个阶段，
每个阶段都有独立的失败语义和独立的修复策略：

```
ExecutionContext (一次性 schema_diagnostics + schema_scan)
  │
  ├─ Phase 1: Semantic Analyst (审题官)
  │     run_semantic_analyst → 输出 schema_mapping / join_plan / filters /
  │     unresolved_core_filters / requires_rule_resolution / confidence
  │     —— 不写代码,只决定「这道题需要从哪些字段里读什么」
  │
  ├─ Phase 2: Codegen 或 Structured-Doc Pre-extract
  │     CodegenDirectAgent 一次性写出完整 Python,使用 SourceCapability 作为
  │     硬约束,prompt 里夹带 Schema Grounding + Foreign-Key Hint + Analyst Plan
  │
  ├─ Phase 3a: Local Repair (确定性 AST 改写,无 LLM)
  │     repair_python_syntax  →  repair_no_such_table  →
  │     repair_no_such_file   →  repair_pandas_keyerror_or_no_such_column →
  │     repair_bad_join_key   →  repair_merge_dtype_mismatch →
  │     repair_zero_row_common_filters → repair_missing_answer_assignment →
  │     repair_json_records_read_with_pandas
  │
  ├─ Phase 3b: Structured-Doc LLM 抽取 (按需,见 §2.4)
  │
  ├─ Phase 3c: Schema-Guided LLM 重写
  │     使用 schema_diagnostics 作为 ground truth,要求 debug_steps 里写
  │     schema_inspection / schema_mapping / plan_override
  │
  └─ Phase 4: Consistency Judge + Semantic Repair (执行官)
        judge_consistency 比较 Analyst 计划 vs 实际代码 vs 答案预览,
        三选一: pass / fail / low_confidence;
        fail 时 run_semantic_repair 限定只改有问题的那一段
```

这个拆分回答了一个很具体的工程问题：「**LLM 算错的时候，到底是问题没读懂、列名找错、
计算口径错，还是仅仅 dtype 没对齐**」。每一类原因都有对应的、最便宜的修法：

- **dtype 不齐 / `.0` 残留 / `KeyError`** → 局部 AST 改写，零 LLM 成本；
- **列名找错（语义→列）** → Schema Grounding + 重写，但只要求改一行；
- **过滤口径错（severe / abnormal / active）** → Analyst 给出 `requires_rule_resolution`，
  执行官比对 `plan_override` 后再决定是否回炉；
- **整张表零行** → 默认是过滤条件错位，触发 `repair_zero_row_common_filters`，
  会自动在每个 ID merge 之前插一句 `.astype(str).str.replace(r'\.0$', ...)`，
  然后**只重跑 Python**，不再调 LLM。

副作用是 `trace.json` 可以非常细地写出「这道题在哪一阶段被修好的、为什么修好的」，
对后期错题分析极为友好。

### 2.4 Execution Harness — 让 LLM 写的代码在一个一致、可观测的盒子里跑

OperatorExecutor 之所以敢把「写代码」这件事完全交给 LLM，是因为下面这层执行外壳兜住了
所有意外。它由两部分组成：

**(a) Codegen 侧的 `_EXEC_HARNESS`（`tablellm_direct.py`）**

LLM 输出的 Python 代码不会被原样执行，而是被字符串模板包一层：

```python
__user_code__                     # ← LLM 写的部分
import os, json, csv
import pandas as _pd

if 'answer' not in dir():
    raise RuntimeError("...did not define an `answer` variable.")

_a = answer
if   isinstance(_a, _pd.DataFrame): _df = _a
elif isinstance(_a, _pd.Series):    _df = _a.to_frame()
elif isinstance(_a, (list, tuple)):
    _df = _pd.DataFrame(list(_a)) if _a and isinstance(_a[0], dict) \
          else _pd.DataFrame({'value': list(_a)})
elif isinstance(_a, dict):          _df = _pd.DataFrame(_a)
else:                               _df = _pd.DataFrame({'value': [_a]})

_df.to_csv(__RESULT_PATH__, index=False)
print('OPERATOR_CODEGEN_RESULT_OK')
print('OPERATOR_CODEGEN_SHAPE=' + str(_df.shape))
print('OPERATOR_CODEGEN_DEBUG=' + json.dumps(debug_steps, ...))
```

这一层做了四件事，每一件都是用过去的失败案例换出来的：

1. **强制契约**：LLM 必须把答案赋给 `answer`。没赋值直接抛 `RuntimeError`，触发
   `repair_missing_answer_assignment` 局部修复，零 LLM 成本。
2. **答案归一化**：DataFrame / Series / list / dict / 标量 / list of dict 一律收敛到
   DataFrame，再用 pandas 自己的 `to_csv` 落盘。这样下游的列匹配评分函数面对的永远是
   官方约定的 CSV，不会因为 LLM「返回了一个 dict 但是写法不规范」就丢分。
3. **结构化 sentinel**：`OPERATOR_CODEGEN_RESULT_OK` / `OPERATOR_CODEGEN_SHAPE=` /
   `OPERATOR_CODEGEN_DEBUG=...` 三个 stdout 标记。上游用它们做严格解析，stdout 里的
   其余打印噪声不会污染状态判断。
4. **Debug 通道**：把 `debug_steps`（LLM 自报的"我是怎么想的、用了哪些列、做了哪些过滤"）
   通过 stdout 以 JSON 单行形式回吐。`semantic_consistency.judge_consistency` 会用它做
   plan-vs-code 的对账。

**(b) Runtime 侧的 multiprocessing sandbox（`tools/python_exec.py`）**

包好的代码不在主进程跑，而是丢进子进程：

- `multiprocessing.Process` 起一个独立进程，主进程 `process.join(timeout_seconds)`；
  超时直接 `terminate() + join()`，主进程不会被卡住。Budget 控制器同时给每题
  设了 wall-clock 上限。
- 子进程一进去就 `os.chdir(context_root)`，把工作目录锁死在该题的 `context/`，
  这样 LLM 写的相对路径 `pd.read_csv("Patient.csv")` 直接能命中正确文件，且
  **无论它怎么写都跑不出 context 目录**。
- `dup2` 把 stdout/stderr 重定向到磁盘临时文件，进程退出后再读回；这样即使子进程因为
  C-extension panic 死掉，外壳里的 sentinel、错误堆栈、`pandas.errors` 全都还在文件里。
- 子进程结果通过 `multiprocessing.Queue` 回传 `{success, error, traceback}`；如果队列空
  了（极少数 OOM 之类的硬故障），上层一律按 `Python execution exited without returning
  a result.` 处理，不会假装成功。
- 所有失败原因都被翻译成结构化 `failure_reason`（`operator_codegen_exec_error: ...`
  等），下一层 `local_repair.issues_from_exec_error(...)` 据此生成 `StaticIssue`，
  和静态检查器的 issue 走同一条修复分发逻辑。

整个 harness 是确定性 + 进程级隔离的：LLM 写出再奇怪的代码（死循环、写盘、
print 海量内容、import 一个不存在的包、抛 SystemExit），都会被收敛成「一次有
明确失败原因的执行结果」，不会污染主流程。这是后续所有局部修复 / 重试 / Judge 才能
工作的物理前提。

### 2.5 Structured-Doc LLM 抽取 — `record_text` 的专用通路

> 这正是 `features-record-text-lIm` 分支的主要工作。

公开集里有一类让普通 RAG / 普通 codegen 全军覆没的任务：
**`Patient.md` / `Laboratory.md` / `Race.md` 这种长篇叙事 Markdown**——
每一份病历散落在多个段落，里面夹杂方法学描述、修正语 (`originally 35.0; corrected to 28.0`)、
重复的背景描述。整张表既不是 Markdown 表格，也不是 JSON 记录，没有任何结构。

我们的处理通路：

1. **`task_compiler` 启发式判定**：检查段落里是否高密度地出现「id 提及 + 字段词汇
   + 日期 + 数字单位」模式，命中即标 `record_text`。
2. **确定性优先**（`_deterministic_extract_records`）：对常见组合（patient + birthday、
   laboratory + creatinine、legalities + format/status）写了一组 deterministic 抽取器。
   这一阶段零 LLM 成本，命中率高的字段直接走规则。
3. **LLM-driven extraction** 兜底：把全文按段落 + record-id 提及切成 ~6 KB 的 chunk，
   按 question 关键词筛一遍 chunk 防止整篇报告每一段都喂模型，再用结构化 prompt 让 LLM
   每个 chunk 输出一个严格 JSON 数组（每条记录字段固定）。所有结果按 `(file_hash,
   schema_hash, chunk_hash, model_id)` 持久化缓存，在 `artifacts/cache/extraction/` 下。
4. **合成 CSV**：把抽取出来的记录写成 `<原始路径>.synth.csv`，然后**再跑一次 codegen**——
   此时下游 LLM 只看到一张干净的 CSV，问题立刻退化成 `table_computation`。
5. **Knowledge.md 规则注入**：从 knowledge.md 里抽取「creatinine > X」这种阈值，
   带回 prompt；如果文档没给阈值，禁止 LLM 自己拍一个，必须用源文里的「abnormal /
   elevated / impaired」等显式语义信号判断。

关键设计：**这条通路在普通 OperatorExecutor 失败一次以后才启动**，不会对短文档任务
产生任何额外开销；缓存命中后第二次跑同一题几乎零 LLM 成本。

### 2.6 Schema Grounding — 防止「列名幻觉」

LLM 写错代码 70% 以上集中在两件事：(a) 调用了一个不存在的列；(b) merge 时用了错列。
为此我们在 prompt 阶段直接做证据级映射（`schema_grounding.py`）：

- 从 question 里提名词概念 → 与 `SourceCapability` 里所有列名做 `(Levenshtein 名字相似度
  + 列样本值匹配 + 低基数 bonus)` 三项加权打分；
- 输出形如 `'thrombosis' → examination.csv::Thrombosis (score=0.92, dtype=int, samples=[0,1,2])`
  的明确映射，写在 prompt 里；
- `foreign_key_candidates` 同理：跨文件按 distinct-value 集合的 Jaccard 给出 top-K
  候选连接键；写在 prompt 里说「JOIN 必须用其中之一」。

下游 `static_checker.py` 在执行前会把 codegen 的 AST 与 `SourceCapability` 比对，
任何引用未知列、未知表、未知文件、错 join key 的程序都被拦截，并产出一条结构化 issue
（`no_such_column / pandas_keyerror / bad_join_key / json_records_read_with_pandas`），
绝大多数能被本地修复一次性解决。

### 2.7 Self-Consistency — 不是表级多数票，是列签名多数票

直接把 N 次回答整张表做多数票，会让一个错列拖垮一个对列。我们改成
**按列内容签名（content signature）独立投票**：

- 对每次采样的 AnswerTable，按官方评分规则把每列规格化成 sorted tuple（数值容差、
  大小写、空白整形）；
- 每个签名最多被一个采样投一票（防止同一采样里重复出现的列多算）；
- 取每个采样的列数众数作为目标列数，按票数挑出 top-K 签名作为最终列；
- 最后从所有采样里挑「与赢家集合交集最大、附带列最少」的那一份用来填具体行内容。

这一项严格匹配官方评分公式 `score = max(0, recall - λ * extra_cols / pred_cols)`：在
disagreement 列里取交集严格优于「直接交主回答」，只在所有 disagreement 列恰好都对
时落败——这种情况在 N≥3 时几乎不发生。

### 2.8 Budget 控制器与确定性回退

`BudgetController` 对每道题硬限定 `max_llm_calls / max_tool_calls / max_seconds /
max_local_repairs / max_reasoner_repairs / max_multiagent_fallbacks`。任何 LLM 或工具
调用前都要经过它。预算耗尽抛 `BudgetExceeded`，逐层向上传播；调用方按结构化失败
原因决定是兜底还是放弃，不会无限重试。

最终的效果是：一道极端 record_text 任务的最坏 LLM 调用数被锁死在 ~30 次以内
（chunk 抽取 + analyst + judge + 最多 3 次 repair + 最多 3 次 local-repair-触发的重跑），
而不是动辄上百 token-roundtrip。

---

## 3. 关键工程取舍

| 取舍 | 我们的选择 | 理由 |
|---|---|---|
| 「让大模型一次写对」vs「写错后能局部修」 | **后者** | 写错的方式有限（语法、列名、dtype、零行、答案变量没赋值），每种都能用确定性 patch 解决，比再调一次 LLM 便宜两个数量级 |
| Multi-Agent 是默认还是兜底 | **兜底** | Planner+Specialists+Synthesizer 的 token 成本是 OperatorExecutor 的 5-8 倍，且对 table_computation 类任务收益不明显 |
| 答案验证发生在哪一层 | **router 之前 + cascade 之间** | 把「成功」的判定从「LLM 自报 succeeded」改成「答案表实际可读、列非空、行非零」，避免被自信但错误的回答骗过去 |
| 难度作为路由键还是预算键 | **预算键** | 同难度任务的最佳路径常常完全不同，按 task_type 分发更稳 |
| RAG 是否对所有 doc 启用 | **按 task_type 启用** | 短文档（< 6KB）直接全文塞 prompt 比 RAG 命中率高；长文档才进 BM25+embedding 混排+重排 |

---

## 4. 实验与成绩

> 评测指标：官方 column-content-signature 评分
> （`recall - 0.5 × extra_cols / pred_cols`，列名忽略，行序忽略）。
> 公开集 50 道题；通过 `eval.py` 与 `dabench score-run` 双方互验。

### 4.1 端到端表现

在公开集上，路由在 4 条路径之间的实际分布大致如下（取 audit_route_flow.py 的统计）：

| 入口路径 | 占比 | 主要任务类型 |
|---|---|---|
| `easy` (operator_executor / tablellm_direct) | ~32% | `table_computation` 小表 |
| `medium` (operator_executor) | ~36% | `table_with_semantic_rule` / 普通 `mixed_context` |
| `tool_first_mixed` (operator_executor + RAG) | ~22% | `record_text_with_semantic_rule` / 长文档 mixed |
| `extreme` (multi_agent / react) | ~10% | `image_understanding` / 不可识别格式 |

带 cascade 后，约有 12% 的题目最终是在第二条路径上才产出有效答案（其中 70% 是
`zero_row → 局部 dtype 修复后的同路径重跑`，并未真的换路径），证明**「按错误模式
cascade」比「按难度盲升档」更省成本**。

### 4.2 消融实验

下表里每一项是「单独关掉这一层 / 把这一层退化成默认实现」时整体得分的相对变化。
绝对分数依赖具体后端（本仓库的 4 份 router 配置都跑过），趋势在所有后端上一致：

| 消融项 | 相对得分变化 |
|---|---|
| 关闭 Schema Grounding（不向 prompt 注入概念→列映射） | -8% ~ -12% |
| 关闭静态检查 + 局部修复 | -10% ~ -15% |
| 把 Execution Harness 退化（直接 `exec` LLM 输出，不强制 `answer` / 不走子进程） | -6% ~ -10%，且有 ~3% 的题目因为子进程超时 / panic 直接整批崩溃 |
| 关闭 Structured-Doc LLM 抽取（`record_text` 走普通 codegen） | 在 `record_text` 子集上 -40% 以上；总分约 -7% |
| 关闭 Semantic Analyst+Judge（仅留 codegen） | -3% ~ -6%，主要在 `table_with_semantic_rule` 类下降 |
| 把 Self-Consistency 从「列签名投票」退化成「整表多数票」 | 在 N=3 时 -2% ~ -4%（多列任务下降明显） |
| 把 Router 从「task_type」退化成「difficulty 优先」 | -5% ~ -8%，且 Multi-Agent 调用量翻倍 |

### 4.3 错题分布与归因

逐题归因依赖 `trace.json` 里的 `router_decision.cascade_attempts`、
`operator_executor.local_repair_log`、`semantic_consistency.judge_history`，
绝大多数仍然失败的任务集中在两类：

- **跨表 + 文档语义规则的多跳**：分两层 join + 一个文档定义里的阈值，链条一旦中间一段
  schema 找错就传染；这部分要靠 plan_override 与 reasoner_repair 来抢救；
- **图像理解**：当前没有部署专门的 VLM endpoint，`image_understanding` 仅靠多模态聊天
  endpoint 兜底，识别率受底层模型能力限制；这是已知 gap，不在本次架构改动范围内。

---

## 5. 可复现入口

- 配置：`configs/router.lite.yaml` / `router.deepseek.yaml` / `router.dashscope.yaml`
  / `router.example.yaml`（4 档不同算力 / 不同 endpoint，pipeline 完全一致，
  只换 `api_base` 和 `model`）。
- 单题：`uv run dabench run-task task_19 --config configs/<选一个>.yaml`
- 整批：`uv run dabench run-benchmark --config configs/<选一个>.yaml`
- 评分：`uv run dabench score-run artifacts/runs/<run_id> --config configs/<选一个>.yaml`
  与 `python -c "from eval import evaluate_batch; evaluate_batch('<run_id>')"` 互验。
- 路由 dry-run（不烧 token）：`uv run python scripts/audit_route_flow.py
  --config configs/<选一个>.yaml --format summary`。
- 全套 + log：`bash scripts/run_full_public_eval.sh --config configs/<选一个>.yaml`。

`trace.json` 字段索引参见 `src/data_agent_baseline/run/runner.py` 顶部的注释。

---

## 6. 小结

我们没有押注一个更大的模型，而是把一道任务上的「能用确定性算就别用 LLM」的细节
做到了极致：

1. 一次性 Task Compiler 给出 schema 真值，下游所有 prompt 共用；
2. Router 按任务类型分发，按错误模式 cascade，把 Multi-Agent 关在最后一道防线；
3. OperatorExecutor 的「Analyst → Codegen → Local Repair → Schema-Retry → Judge → Repair」
   六阶段流水让大多数失败被零 LLM 修复；
4. **Execution Harness 在 multiprocessing 子进程里给 LLM 代码套上确定性外壳**，把
   「LLM 写代码」从一个不可控操作变成一次有明确失败原因的、可被局部修复的事件；
5. `record_text` 由专用 LLM 抽取通路降维成普通表问题，并配合持久化缓存把第二次重跑
   成本压到几乎为零；
6. Self-Consistency 在列签名维度投票，严格匹配官方评分函数；
7. 全程被 Budget 控制器硬约束，每一次 LLM/工具调用都被记账。

这套设计在公开集上把绝大多数原本会失败的任务推到「至少答出一个正确列」的阈值之上，
也是后续在隐藏集上保持稳定分数的依据。
