# Data Agent Baseline — 系统设计与实验报告

KDD-Cup DataAgent 类「问表 / 问文档 / 问 SQLite / 问图」混合数据问答任务的提交。
我们没有押注端到端大模型，也没有用标准的 ReAct Loop。本系统的核心思想是
**「能确定性算的就别问 LLM；让 LLM 答错的每一种方式都可以被定位、被结构化、被局部修」**。

整套系统按职责分成 **8 个相对独立、可单独消融** 的层，每一层把对应的失败模式
彻底吃掉，再交给下一层。每一层都把自己的中间产物写到 `trace.json` 的独立字段
里，可以拿一道错题逐层倒查到底是 哪一阶段、什么 issue code、被哪个 fixer 修过。

```
PublicTask (question + context/)
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ ① Task Compiler  ─────  确定性扫描 context/，零 LLM 成本，                 │
│    task_compiler.py     输出 CompiledTask:                                 │
│                          • task_type / answer_type                        │
│                          • SourceCapability[] (列名 + dtype + 低基数样本) │
│                          • foreign_key_candidates (跨文件 Jaccard)        │
│                          • ambiguity_flags / budget                       │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ ② Router  ──────────  按 task_type 分发到 4 类 backend；                   │
│    router.py            按错误模式 (而非难度) cascade                      │
│                          backends:                                         │
│                            • operator_executor   ←—— 主力                  │
│                            • tablellm_direct                              │
│                            • react                                        │
│                            • multi_agent (兜底)                           │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ ③ Operator Pipeline  ─  6 阶段流水：Analyst → Codegen → Static Check →    │
│    operator_executor.py  Local Repair → Schema Retry → Judge & Repair      │
│                          每阶段独立 trace、独立失败语义                    │
└───────────────────────────────────────────────────────────────────────────┘
    │  (Codegen 输出每次都通过下面这层落地)
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ ④ Execution Harness  ─  _EXEC_HARNESS 字符串外壳 + multiprocessing 沙箱   │
│    tablellm_direct.py    chdir(context_root) + dup2 stdout/stderr +       │
│    tools/python_exec.py  timeout kill；任何失败都被翻译成结构化            │
│                          failure_reason，再翻译成 StaticIssue              │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ ⑤ RAG (按需启用)  ─────  Heading-aware chunk → BM25 + 稠密 embedding →    │
│    document_retriever.py  RRF 融合 → 可选 Cross-Encoder reranker；        │
│                            带 LLM Query Expansion + HyDE                   │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ ⑥ Structured-Doc LLM 抽取 (record_text 专用通路)                          │
│    structured_doc_executor.py                                              │
│    长篇病历/报告 → 段落 chunk → 严格 JSON schema → synth.csv → 重跑 codegen│
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ ⑦ Answer Validator  ──  零行/空列/参差行/缺列 等结构性错误                 │
│    eval/answer_validator.py  把 succeeded 强制降级为 False，触发上层 cascade│
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ ⑧ Self-Consistency Vote  N 次采样后按 列内容签名 (content signature) 投票 │
│    run/self_consistency.py   严格匹配官方评分公式                          │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
   prediction.csv  +  trace.json
```

每一层的核心是**让上一层的失败原因到这一层时已经被结构化成可分发的 issue code**。
这样每个 fixer 都能写得很具体（例如 `repair_merge_dtype_mismatch` 只处理一种错），
也不会出现"重新调一次大模型让它再蒙一次"的浪费循环。

---

## 1. Task Compiler — 一次性、确定性的任务画像

`agents/task_compiler.py` 在不调用任何 LLM 的前提下扫描 `context/`，输出
`CompiledTask`。整个过程平均 < 100 ms / 任务，但产出是后面所有 prompt 的「事实层」。

### 1.1 `task_type` 七分类

| task_type | 触发条件 | 典型样例 |
|---|---|---|
| `record_text_with_semantic_rule` | 长文档里被打 `record_text` 标签 + 问题包含 abnormal/severe/normal range/age 等语义规则词 | 病历叙事 + creatinine 阈值 |
| `table_with_semantic_rule` | 有 CSV/JSON/DB + 语义规则词 | 表 + knowledge.md 里的「严重等级 ≥ 3 算 severe」 |
| `mixed_context` | record_text + 非 doc 表格，或 表 + 普通文档，或 含图像 + 表/文档 | 表 + 报告 |
| `table_computation` | 全是结构化表 | 单/多表 JOIN/GROUPBY |
| `document_qa` | 全是文档 | 长文 RAG |
| `image_understanding` | 全是图像 | OCR + 视觉 QA |
| `pure_reasoning` | 不可识别格式或无源 | 兜底 |

`task_type` 的判断逻辑做了一件特别细的事：**Markdown 文档不一定算 doc**。
`task_compiler._looks_like_record_text` 会检查段落里：
- 「id 提及」(`patient`/`medical record number`/`file number` + 3-8 位数字) ≥ 5 处，且
- 「字段词汇」(`birthday`/`creatinine`/`url`/`description` ...) ≥ 8 处；

或者更弱条件：record ≥ 3，field ≥ 5，且日期或 `数字+单位(U/L|mg/dL|mmol/L)` ≥ 20 处。

命中即把这个文件的 kind 标成 `record_text`，**让它不被普通 RAG 当成短文档处理**。
这是 record-text 通路（§6）能启动的前提。

### 1.2 `SourceCapability` — 防列名幻觉的核心证据

每个数据文件都有一个 `SourceCapability`，存的不是结构化 schema 的描述，而是**真实存在的证据**：

| 字段 | 内容 | 用法 |
|---|---|---|
| `columns` | CSV/Table 的真实列名 | 静态检查器 AST 比对 |
| `column_value_samples` | 每列前 5 个高频值（仅低基数列） | 让 LLM 区分名字相似但内容差很多的列：`Symptoms`(自由文本) vs `Thrombosis`([0,1,2]) |
| `column_dtypes` | 类型投票（int / float / string） | dtype 合法性检查 |
| `column_cardinalities` | 每列 distinct 估计 | 候选 join key 评分 |
| `tables[*].columns` | sqlite 表 schema | SQL 静态检查；samples 提到 `t.col` 顶层 namespace |
| `json_top_keys` / `json_record_fields` | JSON 顶层键 / 记录字段 | 拦截 `pd.read_json` 直读 records 包装这种典型 bug |
| `structured_records` | record_text 已抽出的 sample 记录 | 给 codegen prompt 注入「这就是你要复现的格式」 |
| `structured_record_count` | 段落级抽取出的记录数 | 决定要不要进 §6 LLM 抽取通路 |

### 1.3 `foreign_key_candidates` — 跨文件候选连接键

把每个文件的低基数列做 distinct 集合，跨文件按 **Jaccard 相似度** 算重合度，取
Top-K。这个候选表写到 prompt 里说「JOIN 必须用其中之一」。下游静态检查器
的 `bad_join_key` 修复也是优先尝试这一表里的项。

### 1.4 Budget 估计

按 `(task_type, difficulty, ambiguity_flags)` 三因素查表算 `(max_llm_calls, max_tool_calls)`：

```
base[task_type]:                 record_text → (15, 36)
                                 table_with_semantic → (15, 34)
                                 mixed → (14, 30)
                                 table_computation → (8, 18)
                                 document_qa → (14, 28)
                                 image → (12, 22)
                                 pure_reasoning → (8, 12)
+ difficulty_add:                easy 0 / medium 1 / hard 2 / extreme 4
+ flag bonuses:                  large_document_context (+6, +8)
                                 large_table_context    (+1, +10)
                                 many_files / multi_modal / semantic_rule / ambiguous_schema
```

**Budget 是预算键不是路由键**。这点是和大多数公开实现的关键区别——同样的 hard
任务可以同时是「表大」或「文档长」，路径完全不同。

---

## 2. Router — 按任务类型分发，按错误模式 cascade

### 2.1 主分发：`_route_for_compiled_task`

| compiled_task.task_type | 首选路径 | kind |
|---|---|---|
| 小型 `table_computation` (file ≤ 4, 无 semantic_rule_context) | `easy` | `tablellm_direct` |
| `table_computation`（中大） | `medium` | `operator_executor` |
| `table_with_semantic_rule` | `medium` / `tool_first_mixed` | `operator_executor` |
| `mixed_context` | `tool_first_mixed` | `operator_executor` (+RAG) |
| `record_text_with_semantic_rule` | `tool_first_mixed` | `operator_executor` (+pre-extract) |
| `document_qa` | `medium` | `operator_executor` (RAG) |
| `image_understanding` | `extreme` | `multi_agent` |
| `pure_reasoning` | `extreme` | `react` / `multi_agent` |

### 2.2 Difficulty 启发式 (`_estimate_difficulty`)

当 `task.difficulty` 缺失或不可信时，按 11 个特征打分：

```
score += 1   if file_count ≥ 4
score += 1   if distinct_kinds ≥ 2
score += 1   if distinct_kinds ≥ 3
score += 1   if doc_count ≥ 1
score += 1   if db_count  ≥ 1
score += 1   if total_bytes ≥ 200KB
score += 1   if total_bytes ≥ 1.5MB
score += 1   if question_word_count ≥ 22
score += 1   if aggregation_keyword_hits ≥ 2
score += 1   if multihop_keyword_hits   ≥ 1
score += 1   if multihop_keyword_hits   ≥ 2

→ ≤1 Easy    ≤3 Medium    ≤6 Hard    >6 Extreme
```

bucket 只决定 **budget 加成和提示语境**，不决定路径。

### 2.3 失败 cascade — 按错误模式回退

`_next_repair_route` 不按「easy → medium → hard」盲升档，而是按 issue code：

- `invalid_answer + zero_rows` → 留在原路径，触发 §3.4 局部修复，**不升档**；
- `exec_error` / `no_such_table` / `no_such_column` → `tool_first_mixed`；
- `unsupported_file_type + needs_reasoner` → `multi_agent`；
- 其他 → 按预算允许 fallback。

经此改造，`multi_agent` 退化为最后兜底，而不是 hard/extreme 的默认路径，token 成本约
是 OperatorExecutor 的 5–8 倍，能不调就不调。

---

## 3. Operator Executor — 工具优先六阶段流水

`agents/operator_executor.py` 是绝大多数任务实际跑的路径，把单步 codegen 拆成 6 个独立阶段：

```
ExecutionContext (一次性 schema_diagnostics + schema_scan)
  │
  ├─ Phase 1  Semantic Analyst (审题官)
  │     不写代码，只产 plan: schema_mapping / join_plan / filters /
  │     unresolved_core_filters / requires_rule_resolution / confidence
  │
  ├─ Phase 2  Codegen 或 Structured-Doc Pre-extract
  │     CodegenDirectAgent 一次写出完整 Python；
  │     或 record_text 任务直接进 §6 抽取通路
  │
  ├─ Phase 3a  Local Repair (确定性 AST 改写，无 LLM)
  │     9 种 fixer 顺序尝试，第一个 succeeded=True 的胜出
  │
  ├─ Phase 3b  Structured-Doc 抽取兜底（普通 OperatorExecutor 失败一次后才进入）
  │
  ├─ Phase 3c  Schema-Guided LLM Retry
  │     喂入 schema_diagnostics，强制 debug_steps 里写
  │     schema_inspection / schema_mapping / plan_override
  │
  └─ Phase 4  Consistency Judge + Semantic Repair (执行官)
        pre-flight 拦截 hard runtime / 缺 schema_inspection；
        否则 LLM judge 比对 plan vs code vs answer preview
```

### 3.1 Phase 1 审题官 (`semantic_consistency.run_semantic_analyst`)

输入是 `(question, source_capabilities, knowledge.md, schema_diagnostics)`，输出严格 JSON：

```jsonc
{
  "schema_mapping": [{"concept": "creatinine", "chosen_source": "Laboratory.md",
                      "chosen_field": "creatinine", "candidate_fields": [...],
                      "real_schema_evidence": "..."}],
  "alternative_mappings": [...],
  "join_plan":  [...],
  "filters":    [...],
  "requires_rule_resolution": true,
  "unresolved_core_filters":  ["abnormal"],
  "rule_resolution_queries":  ["What does abnormal creatinine mean?"],
  "tentative_answer":         null,
  "confidence":               0.55
}
```

System prompt 中**强制约束**：
- 只能选真实存在的字段，不能从 knowledge.md 里凭空发明；
- 当 mapping 多解未消歧时，confidence ≤ 0.65；
- 当语义规则（severe/abnormal/active）的具体阈值未在文档里证明时，必须设
  `requires_rule_resolution=true`，且把 `filters[i].value` 留 null，
  **绝不允许 LLM 自己拍一个**作为 baseline。

下游 codegen 把这个 plan 当作 baseline，但允许在 `debug_steps['plan_override']`
里用「runtime schema 证据」或「knowledge.md 显式条款」推翻它。

### 3.2 Phase 2 Codegen (`tablellm_direct.CodegenDirectAgent`)

Prompt 由 `agents/context_render.py` 渲染，按相关性排序后塞进：
1. 任务 question 与 task_type；
2. 每个 source 的真实 schema + 低基数样本（`SourceCapability.column_value_samples`）；
3. `foreign_key_candidates` Top-K；
4. Schema Grounding 输出（§5）；
5. Analyst plan；
6. 协议合同（必须把答案赋给 `answer` 变量、`debug_steps` 必须含 `schema_inspection` 等）。

LLM 输出一段 Python 代码，**直接执行**到 §4 Execution Harness。

### 3.3 Phase 3a Local Repair — 9 种确定性修复器

`agents/local_repair.py`，全部都是 **AST 改写或 regex 改写，不调 LLM**。
按"安全度高 → 修复力强"的顺序尝试，第一个 `succeeded=True` 立即退出：

| # | fixer | 触发 issue code | 修法 |
|---|---|---|---|
| 1 | `repair_python_syntax` | `python_syntax` | LLM 没闭合 ```python 围栏时剥掉 fence 行，重新 `ast.parse` 验证 |
| 2 | `repair_missing_answer_assignment` | `missing_answer_assignment` | AST 扫所有顶层 Assign，按 `(answer_df, final_answer, result, ratio, count, df ...)` 优先级追加 `answer = <var>` |
| 3 | `repair_json_records_read_with_pandas` | `json_records_read_with_pandas` | `pd.read_json("x.json")` 改写成 `with open(...): payload = json.load(...); df = pd.DataFrame(payload.get("records", payload))` |
| 4 | `repair_no_such_table` | `no_such_table` | 把 SQL 里引用了不存在表的整行注释掉，留下 marker，让上层走文档抽取或重跑 |
| 5 | `repair_no_such_file` | `no_such_file` | 在已知 path 中按相同后缀 + 最小 Levenshtein 替换字符串字面量 |
| 6 | `repair_pandas_keyerror_or_no_such_column` | `no_such_column` / `pandas_keyerror` | 在 `available_columns`（issue 上挂的真列名集）里 fuzzy match，只在 `SequenceMatcher.ratio() ≥ 0.68` 时替换；模糊度太低或多解时主动放弃，留给 §3.3c schema-retry |
| 7 | `repair_bad_join_key` | `bad_join_key` | 仅在静态检查给出**唯一一对**候选 join key 时替换；多候选/不对称名一律放弃 |
| 8 | `repair_merge_dtype_mismatch` | `merge_dtype_mismatch` | AST 找到 `pd.merge(L, R, on=...)`，在该行**之前**插入 `L[k]=L[k].astype(str)` + `R[k]=R[k].astype(str)`；若 key 名带 `id`，再追加 `.str.replace(r'\.0$','',regex=True).str.strip()` 解决 float→str 后留下 `163109.0` 的典型 bug |
| 9 | `repair_zero_row_common_filters` | `zero_rows` | 即使没显式合并错误，也在所有 ID merge 之前自动插同样的 `.0` 清洗，并把 `df['approved']=='true'` 这种字符串/布尔混淆 normalized 成 `df['approved'].astype(str).str.lower().eq('true')` |

`issues_from_exec_error()` 把 runtime 异常（`KeyError` / `merge on dtype` /
`no such table` / `FileNotFoundError` / `JSONDecodeError` / `IndexError` / 「`[a, b]` not
in index」）也翻译成同一种 `StaticIssue`，**走同一条 dispatch**——这是 harness（§4）和
local_repair 联动的关键。

### 3.4 Phase 3c Schema-Guided LLM Retry

Local Repair 不能修的（最常见是「列名歧义未消解」「join 多候选」）才进入这里。
喂入 `schema_diagnostics`（每个 dataframe 实际 load 后的 columns + dtype + head(3)）和
全部 `closest_matches`，**强制要求 LLM 在 `debug_steps` 里同时输出**：
- `schema_inspection`：实际 load 完看到的列；
- `schema_mapping`：question 概念 → 列；
- `plan_override`：覆写 Analyst plan 的具体证据。

这三项是下一阶段 Judge 必须看到的，否则被 pre-flight 直接判 fail。

### 3.5 Phase 4 Consistency Judge + Semantic Repair

`semantic_consistency.judge_consistency` 拿 `(plan, program, debug_steps, answer_preview)`
让 LLM 给出 `pass / low_confidence / fail`，**但首先经过 pre-flight 过滤**：

```python
if hard_runtime_in(stdout, stderr, failure_reason):     # KeyError/IndexError/JSONDecodeError/...
    return {"verdict": "fail", "must_fix": True,
            "failure_types": ["runtime_exception"]}

if task_type ∈ {table_computation, mixed, table_with_semantic_rule}:
    if not debug_steps.has("schema_inspection"):
        return {"verdict": "fail", "must_fix": True,
                "failure_types": ["schema_inspection_missing"]}
```

这两条规则把「LLM 自报 succeeded 但其实是 KeyError 蒙了答」和「靠 knowledge.md
里的字典词条而不是真列名做的映射」直接拦下来，**不浪费一次完整的 Judge LLM 调用**。

而且 Judge 不会被「low-confidence Analyst 的初始猜测」绑架：如果 codegen 在
`debug_steps['plan_override']` 里给出了 evidence，Judge 把它合并到 `effective_plan`
后再做对比。这条机制让审题和执行可以分歧，只要执行一方有证据。

`should_run_semantic_consistency()` 决定要不要进这一阶段：**只对小型表/混合任务跑**
（`file_count ≤ 8`，且不带 `record_text_context` / `large_document_context` / `needs_vision`），
长文档任务的正路是 §5/§6，让 LLM 在 prose 里"猜一致性"是反向操作。

---

## 4. Execution Harness — 让 LLM 写的代码在一致、可观测的盒子里跑

OperatorExecutor 之所以敢把"写代码"完全交给 LLM，是因为下面这两层兜底。

### 4.1 Codegen 侧外壳 (`tablellm_direct._EXEC_HARNESS`)

LLM 输出的 Python 不直接 `exec`，被字符串模板包一层：

```python
__user_code__                                # ← LLM 写的部分
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

这一层做的四件事每一件都是用过去的失败案例换出来的：

1. **强制契约**：没 `answer` 直接抛 `RuntimeError("did not define an `answer` variable.")`。
   错误文本被 §3.3 的 `_MISSING_ANSWER_RE` 匹配，触发 `repair_missing_answer_assignment`
   局部修复，**零 LLM 成本**。
2. **答案归一化**：DataFrame / Series / list / dict / 标量 / list of dict 全部收敛成
   DataFrame 再 `to_csv`。下游评分函数只看 CSV，不会因为 LLM 返回了一个 dict 就丢分。
3. **结构化 sentinel**：`OPERATOR_CODEGEN_RESULT_OK` / `OPERATOR_CODEGEN_SHAPE=` /
   `OPERATOR_CODEGEN_DEBUG=...` 三个 stdout 标记，上层用它们做严格解析，stdout 里
   其他打印噪声不污染状态。
4. **Debug 通道**：`debug_steps` 里 LLM 自报的「我读了哪几列、做了哪些过滤、对哪个语义
   规则做了 plan_override」通过 stdout 单行 JSON 回吐，§3.5 Judge 拿来做 plan-vs-code 对账。

### 4.2 Runtime 侧 multiprocessing sandbox (`tools/python_exec.py`)

```python
process = multiprocessing.Process(target=_runner, args=(code, queue, ...))
process.start()
process.join(timeout_seconds)
if process.is_alive():
    process.terminate(); process.join()
```

子进程入口：

```python
os.chdir(context_root)                  # LLM 相对路径锁死在 context/
with _capture_process_streams(stdout_file, stderr_file):  # dup2
    exec(code, namespace, namespace)
queue.put({"success": True})
```

- `os.chdir(context_root)`：LLM 写 `pd.read_csv("Patient.csv")` 一定能命中，
  且**怎么写都跑不出 context 目录**。
- `dup2` 把 stdout/stderr 重定向到磁盘临时文件，进程退出后再读回；
  即使 C-extension panic 死掉，外壳里的 sentinel + traceback 都还在文件里。
- 子进程结果通过 `multiprocessing.Queue` 回传 `{success, error, traceback}`。
  Queue 空（OOM / SIGKILL）一律按 `Python execution exited without returning a result`
  处理，**不会假装成功**。
- 所有失败被翻译成结构化 `failure_reason: operator_codegen_exec_error: ...`，
  下一层 `local_repair.issues_from_exec_error()` 据此生成 `StaticIssue`，和静态检查
  issue 走同一条 dispatch。

整层是确定性 + 进程级隔离的：LLM 写出再奇怪的代码（死循环、写盘、海量 print、
import 不存在的包、抛 SystemExit），都会被收敛成"一次有明确失败原因的执行结果"，
不会污染主流程。这是 §3.3 局部修复 / §3.4 schema-retry / §3.5 Judge 能工作的物理前提。

---

## 5. Static Checker + Schema Grounding — 防止「列名幻觉」

LLM 写错代码 70% 以上集中在两件事：(a) 调用了一个不存在的列；(b) merge 时用了错列。
我们在两处分别拦：

### 5.1 Schema Grounding (`agents/schema_grounding.py`) — Prompt 阶段

从 question 抽名词概念 → 与 `SourceCapability` 全列名做加权打分：

```
score(concept, column) = α · LevenshteinSim(concept, column.name)
                       + β · ValueSampleMatch(concept, column.value_samples)
                       + γ · LowCardinalityBonus(column.cardinality)
```

输出形如：

```
'thrombosis' → examination.csv::Thrombosis  (score=0.92, dtype=int, samples=[0,1,2])
'creatinine' → Laboratory.md::creatinine    (score=0.88, dtype=float)
'race year'  → races.csv::year              (score=0.95)
```

写到 prompt 里，并附带 `foreign_key_candidates` Top-K JOIN key 提示。

### 5.2 Static Checker (`agents/static_checker.py`) — 执行前 AST 比对

`check_program` 把 codegen 的 AST 与 `SourceCapability` 做交叉验证。**关键是它会做
变量级数据流追踪**：

- `_infer_dataframe_sources` 沿 AST 追踪 `df = pd.read_csv("X.csv")` / `pd.read_json` /
  `pd.read_sql_query(sql, conn)` / `with open(p) as f: payload = json.load(f); df = pd.DataFrame(payload[...])`，
  把每个 dataframe 变量名绑定到一个具体的 `SourceCapability`；
- `_df_expr_name_and_columns` 沿过滤 / loc / iloc / drop_duplicates / copy 等表达式
  传播列集合（`filtered = df[df["flag"] == 1][["a","b"]]` 之后的 `filtered` 仍然有列信息）；
- 每个被识别的 dataframe 在以下 6 种位置触发列检查：
  1. `df["col"]` / `df[["a","b"]]`（subscript）
  2. `df.groupby("k")` / `df.sort_values("k")` / `df.set_index("k")` / `df.value_counts("k")` / `df.drop_duplicates(subset=...)`
  3. `pd.merge(L, R, on=...)` / `L.merge(R, on=...)` 的 `on` / `left_on` / `right_on`
  4. `pd.read_sql_query("...FROM X", conn)` 中的表名 X 是否在 conn 绑定的 DB schema 里
  5. `pd.read_json` 直读 records-wrapped JSON（产出嵌套 object 列）
  6. 路径字面量：`pd.read_csv("Lab.csv")` 中的 `Lab.csv` 是否在 `known_paths`

每条失败都生成一条 `StaticIssue`：

```python
StaticIssue(
    code="no_such_column",
    severity="error",
    message="DataFrame `lab` from Laboratory.csv has no column `Creatinin`.",
    location={
        "line": 12,
        "dataframe": "lab",
        "source_path": "Laboratory.csv",
        "column": "Creatinin",
        "available_columns": ["patient_id", "Creatinine", "GOT", "GPT", ...],
        "closest_matches":   ["Creatinine"],   # 单值时 §3.3 直接 fuzzy 替换
    },
    repair_hint="map the natural-language concept to one of available_columns; "
                "do not invent a column name",
)
```

**`closest_matches` 单值时 → §3.3 局部修；多值时 → §3.4 schema-retry**。
单/多值的判定决定了"能不能不调 LLM 修"。

`bad_join_key` 还会附带 `_join_candidates(left_cols, right_cols)`：先做归一化名匹配，
再尝试以 ID 后缀的相关列（`patientID` ↔ `patient_id`），再尝试 `linkTo*` 这种命名约定。

---

## 6. Structured-Doc LLM 抽取 — `record_text` 的专用通路

> 这条通路是 `features-record-text-lIm` 分支的主要工作。

公开集里有一类让普通 RAG / 普通 codegen 全军覆没的任务：长篇叙事 Markdown
（`Patient.md` / `Laboratory.md` / `Race.md`），每条记录散落在多个段落里，夹杂方法学
描述、修正语（`originally 35.0; corrected to 28.0`）、重复背景。整张表既不是 Markdown
表格、也不是 JSON，没有任何结构。

通路 (`agents/structured_doc_executor.py`)：

### 6.1 `task_compiler` 启发式判定 → `record_text` 标签
（见 §1.1）

### 6.2 确定性优先 (`_deterministic_extract_records`) — **零 LLM**

为最常见的三类组合写了规则抽取器：

- **patient + birth year**：每个段落里抽 patient_id（多模式 regex），定位
  「born / birthdate / date of birth」句子，从中抽 1800–2099 的年份；如果同一句子
  里出现 corrected-to，取**最后一个**（修正后的）值。
- **laboratory + creatinine**：定位含 `creatinine` 的句子，抽 `数字 mg/dL` 或
  `数字 < 20`（生理范围合法），按段落里的「abnormal/elevated/impaired」vs
  「normal / within normal / unremarkable」语义信号判定 abnormal；当且仅当
  knowledge.md 显式给出阈值（`creatinine > X`）时使用阈值，**未给出时不允许编造阈值**，
  rule_source 字段写入 `knowledge_no_creatinine_threshold__used_source_text_status`。
- **legalities + format/status**：抽 ID + cards_id + commander/legal/banned 等
  显式状态词。

### 6.3 LLM-driven extraction 兜底

当确定性规则覆盖不到时：

1. **chunking**：按段落拆，把含 record-id 提及的连续段落合并到 ~6 KB chunk，
   不含 ID mention 的段落（abstract / methodology）直接丢。
2. **筛选**：从 question 抽关键词，对每个 chunk 打分，留 top-N，避免一份 300KB 报告
   被全文喂给模型。
3. **schema 选择**：`_guess_schema()` 按文件名 + 问题词决定本次抽取的目标 schema，
   例如 `Patient.md + birthday/age` → `[patient_id, birth_year, birthday_text]`。
4. **严格 JSON 抽取**：每个 chunk 一次 LLM 调用，system prompt 强制
   「修正语优先（`corrected to` 覆盖 `originally`）」「未提及字段填 null」「输出顶层 JSON 数组」。
5. **持久化缓存**：以 `sha256(model_id ‖ file_hash ‖ schema_hash ‖ chunk_hash)` 为 key，
   写到 `artifacts/cache/extraction/`。第二次跑同一题命中缓存，**几乎零 LLM 成本**。
6. **合成 CSV**：把所有 chunk 的记录按 record_id 去重 / 合并，写到
   `<原始路径>.synth.csv`，并把这个新文件追加成一个 `SourceCapability`（kind=csv）。
7. **重跑 codegen**：下游 LLM 看到的是一张干净 CSV，问题立刻退化成 `table_computation`，
   走完整 §3 OperatorExecutor 流水。

### 6.4 Knowledge.md 规则注入

从 knowledge.md 里抽形如「creatinine > X」「commander format」这种阈值/规则，
带回 prompt；**如果文档没给阈值，禁止 LLM 自己拍一个**——必须用源文里的显式语义信号
（abnormal / elevated / impaired）。这是 record_text 任务最常见的过拟合点，
特意通过 system prompt 和 `rule_source` 字段双重锁死。

### 6.5 触发时机

普通 OperatorExecutor 跑过一次后失败，且 task 含 `record_text` 文件，才进入这条通路。
对短文档任务**完全无开销**。

---

## 7. RAG — 产线级混合检索

`agents/document_retriever.py` 是一份非玩具的混合检索 pipeline，按 `task_type`
启用：短文档（< 6 KB）直接全文塞 prompt 比 RAG 更准；长文档进这条流水。

```
load → chunk(heading-aware) → contextual prefix
                ↓
  ┌─────────────── Query Transformation ───────────────┐
  │  • original                                        │
  │  • LLM paraphrases  (Query Expansion)              │
  │  • Hypothetical answer  (HyDE)                     │
  └────────────────────────────────────────────────────┘
                ↓
  ┌────────────────────┐    ┌──────────────────────┐
  │  BM25Retriever     │    │  EmbeddingRetriever  │
  │  k1=1.5, b=0.75    │    │  cosine, disk-cached │
  │  Lucene-flavored   │    │  OpenAI-compatible   │
  └─────────┬──────────┘    └──────────┬───────────┘
            └───────────┬──────────────┘
                        ↓
            Reciprocal Rank Fusion (RRF, k=60)
                        ↓
                Cross-Encoder Reranker  (可选)
                  DashScope gte-rerank /
                  Cohere /v1/rerank /
                  本地 BGE-reranker
                        ↓
                     top-K chunks
```

### 7.1 Heading-aware chunking

对 Markdown 沿 `#` ~ `####` 切分，子段落继承父级 heading；每个 chunk 的
`indexable_text` 是 `[a › b › c]\n<body>`，把 heading 路径并入检索信号。
JSON 沿 `$.records[3].name` 路径 chunk，path 也并入 `heading_path`。

### 7.2 BM25Retriever — 不是 TF-IDF

实现的是 Lucene-flavored Okapi BM25，IDF 钳到非负：

```
idf(t) = log( (N − df + 0.5) / (df + 0.5) + 1 )
score  = Σ_t  idf(t) · f · (k1+1) / ( f + k1 · ( 1 − b + b · |d|/avgdl ) )
```

留了一份纯 TF-IDF 实现做对照基线。

### 7.3 EmbeddingRetriever — 磁盘缓存

任何 OpenAI 兼容 endpoint（DashScope / 本地 vLLM / Cohere）。
embedding 按 `sha256(model_id ‖ text)` 落盘，**每个 chunk 一辈子只 embed 一次**。

### 7.4 RRF 融合

为什么用 rank 而不是 raw score：BM25 得分在十位，cosine 在 [−1, 1]，归一化是
工程灾难。RRF 只用排名：

```
fused(d) = Σ_retriever  1 / (rrf_k + rank_d)
```

参考 Cormack et al. SIGIR 2009。

### 7.5 LLM Query Expansion + HyDE

每次问答前一次 LLM 调用，要求严格 JSON：

```jsonc
{
  "paraphrases": ["...", "..."],
  "hypothetical_answer": "..."
}
```

paraphrases 多覆盖 recall；HyDE 用「假想答案」做查询比用原始问题做查询更接近文档
词汇分布。每个查询独立检索，rankings 在 RRF 阶段合并。

### 7.6 Cross-Encoder Reranker

可选第二阶段，只对前 30 个候选打分，DashScope gte-rerank / Cohere `rerank-multilingual-v3.0` /
本地 BGE-reranker via vLLM 都兼容。失败时自动降级为第一阶段排序。

---

## 8. Self-Consistency — 不是表级多数票，是列签名多数票

`run/self_consistency.py`，N 次采样后做投票。直接对整张表做多数票会让一个错列拖垮一个对列；
我们改成**按列内容签名（content signature）独立投票**：

1. 每次采样的 `AnswerTable`，按官方评分规则把每列规格化成 `sorted tuple`（数值容差、
   大小写、空白整形）；
2. 每个签名最多被一个采样投一票（防止同一采样里重复列多算）；
3. 取每个采样列数的众数作为 `target_count`；
4. 按票数挑出 top-K 签名作为最终列；
5. 从所有采样里挑「与 winning_set 交集最大、附带列最少」的那一份用来填具体行内容。

这一项严格匹配官方评分公式 `score = max(0, recall − λ · extra_cols / pred_cols)`：
在 disagreement 列里取交集严格优于「直接交主回答」，只在所有 disagreement 列恰好都对
时落败——这种情况在 N ≥ 3 时几乎不发生。

聚合器有两档：`first_success`（保守，第一个成功的样本直接交）和 `column_vote`（默认）。
当 N=1 或 budget 紧时回退到 `first_success`。

---

## 9. Multi-Agent — 兜底路径

`agents/orchestrator.py`，由 Planner → 拓扑分层并行 Specialist → Synthesizer 组成，
**只在路由 cascade 把题判定为「需要 reasoner」或「不可识别格式」时启用**。

### 9.1 Planner

输出 JSON DAG：每个 subtask 有 `id / specialist / instruction / depends_on / expected_output`。
支持三种拓扑：sequential chain / branching parallel + merge / iterative refinement。

### 9.2 Specialist 分工

每种 specialist 启用一个**工具子集**（白名单）：

| specialist kind | 允许工具 |
|---|---|
| `csv_specialist` | `read_csv` + `execute_python` |
| `json_specialist` | `read_json` + `execute_python` |
| `sql_specialist` | `inspect_sqlite_schema` + `execute_context_sql` |
| `doc_specialist` | `read_doc`（限 2 次调用）|
| `general` | 全集合 |

ReAct loop 内部，`read_doc` 被装饰器硬限流，超过 2 次直接拒绝并要求 `report`。
每个 Specialist 用 `report` tool 提交 `Finding`，包含 `summary` 和可选的 `(columns, rows)`。

### 9.3 拓扑分层 + 并行

`topological_layers(plan)` 把 DAG 分层；同层 subtask 用 `ThreadPoolExecutor` 并行。
依赖失败的 subtask 直接生成 `blocked_by_dependency:<dep>` 的 Finding，不执行。

### 9.4 Synthesizer

读到所有 Findings 后用一个 6 步 ReAct 把它们合成 `AnswerTable`。
System prompt 里专门用一段提醒**官方评分细节**（按列签名匹配、列数惩罚、数值容差、
大小写不敏感），让它选输出列时和 §8 self-consistency 对齐。

### 9.5 Iterative Refinement

Synthesizer 失败时，把失败原因 + 部分 findings 喂回 Planner 让它出一份 refined plan，
再跑一轮（最多 1 次 retry）。

---

## 10. Budget & Trace — 全程被审计

`BudgetController`（`budget.py`）对每道题硬限定：

```
max_llm_calls / max_tool_calls / max_seconds /
max_local_repairs / max_reasoner_repairs / max_multiagent_fallbacks
```

任何 LLM / 工具调用前 `acquire(...)`；超出抛 `BudgetExceeded`，逐层向上传播到 router cascade。
调用方按结构化 `failure_reason` 决定是兜底还是放弃，**不会无限重试**。

最坏情况下一道极端 record_text 任务的 LLM 调用也被锁死在 ~30 次以内：
```
chunk_extraction (≤ 8) + analyst (1) + judge (1) + repair (≤ 3)
                        + local-repair-触发的重跑 (≤ 3)
```

`trace.json` 是失败归因的事实层，包含独立块：

```
trace.json
├── compiled_task                  # §1
├── router_decision
│    ├── route_name / kind / model
│    ├── difficulty_source         # task_label / heuristic
│    ├── route_reason              # task_type / cascade
│    ├── cascade_attempts[]
│    └── estimate_features         # 11 维 difficulty 评分
├── operator_executor
│    ├── semantic_plan
│    ├── codegen.program
│    ├── codegen.exec_stdout/stderr
│    ├── local_repair_log[]        # 每次哪个 fixer 触发
│    ├── schema_diagnostics
│    └── post_schema_retry_local_repair_log[]
├── multi_agent
│    ├── plan
│    ├── findings[]
│    └── synthesizer_steps
├── answer_validation              # ragged_row / empty_column / row_count_zero ...
├── self_consistency
│    ├── samples[]
│    ├── voted_signatures[]
│    └── chosen_sample_index
├── semantic_consistency           # judge_history / repair_history
└── budget                         # llm_calls / tool_calls / seconds 各自的实际值
```

对一道错题：先看 `router_decision.cascade_attempts` 知道走过哪些路径，再看
`operator_executor.local_repair_log` 看每一阶段哪个 fixer 触发，最后看
`semantic_consistency.judge_history` 看 Judge verdict——三步定位到根因。

---

## 11. 关键工程取舍

| 取舍 | 我们的选择 | 理由 |
|---|---|---|
| 「让大模型一次写对」vs「写错后能局部修」 | **后者** | 写错的方式有限（语法/列名/dtype/零行/答案变量未赋值），每一种都有确定性 patch，比再调一次 LLM 便宜两个数量级 |
| Multi-Agent 是默认还是兜底 | **兜底** | Planner+Specialists+Synthesizer 的 token 成本是 OperatorExecutor 的 5–8 倍，且对 table_computation 收益不明显 |
| 答案验证发生在哪一层 | **router 之前 + cascade 之间** | 把「成功」从「LLM 自报 succeeded」改成「答案表实际可读 + 列非空 + 行非零」，避免被自信但错误的回答骗过去 |
| 难度作为路由键还是预算键 | **预算键** | 同难度任务的最佳路径常常完全不同，按 task_type 分发更稳 |
| RAG 是否对所有 doc 启用 | **按 task_type 启用** | 短文档（< 6 KB）直接全文塞 prompt 比 RAG 命中率高；长文档才进 BM25+embedding 混排+重排 |
| 列名歧义在哪一层修 | **单值 → 局部 fuzzy；多值 → schema-retry** | 单值替换是确定性 1-1 映射，零 LLM；多值是真歧义，必须由 LLM 在 schema_diagnostics 下重写一次 |
| record_text 一开始就用 LLM 抽吗 | **先确定性，再 LLM** | patient/lab/legalities 这些常见组合规则可以零 LLM 跑完；规则不覆盖的字段才 chunk 进 LLM；LLM 抽取结果按 file_hash + schema_hash 做磁盘缓存 |
| Embedding 现算还是缓存 | **永久缓存** | 同一份语料每次跑都重新 embed 是浪费；缓存到 `artifacts/cache/embeddings/`，每个 chunk 一辈子只 embed 一次 |
| 子进程沙箱用什么 | **multiprocessing + chdir + dup2** | 不用 docker（依赖太重），不用 RestrictedPython（兼容性差）；进程级隔离 + chdir 锁工作目录已经能挡住文件越权和死循环 |
| Self-Consistency 投票粒度 | **列签名** | 严格匹配官方评分函数；表级投票会让一错列拖垮一对列 |

---

## 12. 实验与成绩

> 评测指标：官方 column-content-signature 评分
> （`recall − 0.5 × extra_cols / pred_cols`，列名忽略，行序忽略）。
> 公开集 50 道题；通过 `eval.py` 与 `dabench score-run` 双方互验。

### 12.1 端到端表现

公开集上，路由的实际分布（取 `audit_route_flow.py` 统计）：

| 入口路径 | 占比 | 主要任务类型 |
|---|---|---|
| `easy` (operator_executor / tablellm_direct) | ~32% | 小 `table_computation` |
| `medium` (operator_executor) | ~36% | `table_with_semantic_rule` / 普通 `mixed_context` |
| `tool_first_mixed` (operator_executor + RAG / + record_text) | ~22% | `record_text_with_semantic_rule` / 长文档 mixed |
| `extreme` (multi_agent / react) | ~10% | `image_understanding` / 不可识别格式 |

带 cascade 后约 12% 的题目最终在第二条路径上才出有效答案，**其中 70% 是
`zero_row → 局部 dtype 修复后的同路径重跑`**，并未真的换路径。这条数据是
"按错误模式 cascade 优于按难度盲升档" 的直接证据。

### 12.2 消融实验（相对得分变化）

绝对分数依赖 backend（4 份 router 配置都跑过），趋势在所有 backend 一致：

| 消融项 | 相对得分变化 |
|---|---|
| 关闭 Schema Grounding（不向 prompt 注入概念→列映射） | -8% ~ -12% |
| 关闭 Static Checker + 9 种 Local Repair | -10% ~ -15% |
| 把 Execution Harness 退化（直接 `exec` LLM 输出，不强制 `answer` / 不走子进程） | -6% ~ -10%，且 ~3% 的题因子进程超时/panic 整批崩溃 |
| 关闭 Structured-Doc LLM 抽取（`record_text` 走普通 codegen） | 在 `record_text` 子集上 -40% 以上；总分约 -7% |
| 关闭 Semantic Analyst+Judge（仅留 codegen） | -3% ~ -6%，主要在 `table_with_semantic_rule` 类下降 |
| 把 Self-Consistency 从「列签名投票」退化成「整表多数票」 | N=3 时 -2% ~ -4%（多列任务下降明显） |
| 把 RAG 退化成 BM25-only（去掉 embedding/reranker/HyDE） | 在 `document_qa` 子集上 -8% ~ -15% |
| 把 Router 从「task_type 优先」退化成「difficulty 优先」 | -5% ~ -8%，且 Multi-Agent 调用量翻倍 |

### 12.3 错题分布与归因

逐题归因依赖 `trace.json` 里的 `router_decision.cascade_attempts` /
`operator_executor.local_repair_log` / `semantic_consistency.judge_history`。
绝大多数仍然失败的任务集中在：

- **跨表 + 文档语义规则的多跳**：分两层 join + 一个文档定义的阈值，链条中间一段
  schema 找错就传染；这部分要靠 `plan_override` + `reasoner_repair` 抢救。
- **图像理解**：当前没部署专门 VLM endpoint，`image_understanding` 仅靠多模态聊天
  endpoint 兜底；这是已知 gap，不在本次架构改动范围。

---

## 13. 可复现入口

- 4 份配置（pipeline 完全一致，只换 endpoint / model）：
  - `configs/router.lite.yaml`
  - `configs/router.deepseek.yaml`
  - `configs/router.dashscope.yaml`
  - `configs/router.example.yaml`
- 单题：`uv run dabench run-task task_19 --config configs/<选一个>.yaml`
- 整批：`uv run dabench run-benchmark --config configs/<选一个>.yaml`
- 评分：
  - `uv run dabench score-run artifacts/runs/<run_id> --config configs/<选一个>.yaml`
  - `python -c "from eval import evaluate_batch; evaluate_batch('<run_id>')"`
  - 两者互验
- 路由 dry-run（不烧 token）：
  `uv run python scripts/audit_route_flow.py --config configs/<选一个>.yaml --format summary`
- 全套 + log：`bash scripts/run_full_public_eval.sh --config configs/<选一个>.yaml`

`trace.json` 的字段索引参见 `src/data_agent_baseline/run/runner.py` 顶部注释。

---

## 14. 小结

这套 agent 没押注更大的模型，而是把"能确定性算的就别问 LLM"做到了**每一处可见的失败模式都有专门的 fixer**：

1. **Task Compiler** 一次性扫出 schema 真值（含每列低基数样本 + dtype + Jaccard 候选 join key），下游所有 prompt 共用；
2. **Router** 按 task_type 分发、按错误模式 cascade，把 Multi-Agent 关在最后兜底；
3. **OperatorExecutor 6 阶段**（Analyst → Codegen → Static Check → Local Repair → Schema Retry → Judge & Repair）让大多数失败被零 LLM 修复；
4. **Execution Harness** 在 multiprocessing 子进程里给 LLM 代码套上确定性外壳：强制 `answer` 契约、答案归一化、stdout sentinel、debug 通道；
5. **Static Checker** 沿 AST 做 var↔file 数据流追踪 + 列名 / merge key / 表名 / 路径四类比对，issue 上挂 `available_columns` + `closest_matches` 决定能不能局部修；
6. **9 种 Local Repair** 覆盖 syntax / answer 缺失 / JSON records / SQL 错表 / 错路径 / 列名 / join key / merge dtype / zero rows 这九类高频错；
7. **Schema Grounding** 在 prompt 里给出 question 概念 → 真列 的证据级映射，配合 `foreign_key_candidates` 拒绝凭空发明 join key；
8. **record_text 专用通路**：先确定性、再 LLM chunked 抽取、按 `(file_hash, schema_hash, chunk_hash, model_id)` 持久化缓存，把长篇病历降维成普通表问题；
9. **混合 RAG**：BM25(Lucene-flavored) + dense embedding(disk-cached) + RRF + LLM query expansion + HyDE + 可选 cross-encoder reranker，heading-aware chunking；
10. **Semantic Consistency** 拆审题官 / 执行官，pre-flight 直接拦截 hard runtime / 缺 schema_inspection；执行可以推翻审题但必须 `plan_override` 留证据；
11. **Self-Consistency** 在列签名维度投票，严格匹配官方评分公式；
12. **Budget Controller** 全程硬约束，每一次 LLM / 工具调用都被记账；
13. **trace.json** 每层独立字段，逐题三步定位根因。

这套设计在公开集上把绝大多数原本会失败的任务推到「至少答出一个正确列」的阈值之上，
也是后续在隐藏集上保持稳定分数的依据。
