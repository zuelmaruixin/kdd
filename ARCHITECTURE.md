# Data Agent — 系统架构（KDD Cup 2026 · DataAgent-Bench）

> 一份按"答辩 PPT 一页一节"组织的完整架构说明。
> 每一节都已经按照可拆成一张幻灯片的粒度写好。

---

## 0. 一句话总览

> **能用程序检查的不问模型；能局部修的不整题重跑；能 tool-first 的不进 multi-agent；reasoner 只做局部修复；multi-agent 只做最后兜底。**

整套系统就是这五条原则在工程上的展开。

---

## 1. 顶层数据流

```
            ┌──────────────────────────────────────────────────────────────┐
            │  question + heterogeneous context                            │
            │  (CSV · JSON · SQLite · Markdown · DOCX · 长篇散文报告 · 图)  │
            └───────────────────────────────┬──────────────────────────────┘
                                            ▼
                         ┌──────────────────────────────────┐
                         │  Deterministic Task Compiler     │   零 LLM 调用
                         │  • 扫描每个文件的 schema/记录    │
                         │  • 识别 task_type / answer_shape │
                         │  • 输出 source_capabilities      │
                         │  • 估预算 (max_llm_calls 等)     │
                         └─────────────────┬────────────────┘
                                           ▼
                         ┌──────────────────────────────────┐
                         │            Router                │
                         │  按 task_type 选 route            │
                         │  (难度只影响预算，不影响主路由)  │
                         └──────┬─────┬─────┬─────┬─────────┘
                                │     │     │     │
              table_computation │     │     │     │ pure_reasoning / fallback
              mixed_context     │     │     │     │
              table_with_rule   │     │     │     │
              document_qa       ▼     ▼     ▼     ▼
                       ┌─────────────┐  ┌─────────┐  ┌─────────────────┐
                       │ Operator    │  │ ReAct   │  │ Multi-Agent     │
                       │ Executor    │  │ Loop    │  │ Orchestrator    │
                       │ (tool-first)│  │         │  │ (planner+spec.+ │
                       │             │  │         │  │  synthesizer)   │
                       └──────┬──────┘  └────┬────┘  └────────┬────────┘
                              │              │                 │
                              ▼              │                 ▼
                ┌───────────────────────────┐│   ┌─────────────────────────┐
                │ ① codegen (LLM)            ││   │ Cross-Model Verify      │
                │ ② static_check (AST + SQL) ││   │ (列签名交集，仅此路径)  │
                │ ③ sandbox exec             ││   └─────────────────────────┘
                │ ④ answer_validator         ││
                │ ⑤ local_repair (确定性 ≤N)  ││            │
                │ ⑥ structured_doc_synth     ││            │
                │ ⑦ schema_retry (LLM)       ││            │
                │ ⑧ reasoner_repair (LLM)    ││            │
                │ —— 仍失败 → cascade ──────┐│↓            │
                └────────────────────────────┘└─────┬───────┘
                                                    ▼
                                ┌────────────────────────────┐
                                │ Column-Signature Scorer    │
                                │ score = max(0, recall      │
                                │  − λ·extra/pred)           │
                                └────────────────────────────┘
```

横切（每一层都可以读到）：
- **Budget Controller**：单进程内限 LLM/tool/local-repair/reasoner-repair 各自次数 + 总耗时
- **Progress Logger**：rich 渲染的事件流（演示用）
- **Stream Sink**：raw token 流（debug 用）
- **Embedding cache + Extraction cache**：磁盘缓存，跨任务复用
- **trace.json**：固定 schema，每个阶段在自己的顶层 key 下落字段

---

## 2. 设计原则与赛题映射

| 赛题要求 | 系统对应组件 |
| --- | --- |
| **自主拆解与规划** | TaskCompiler（数据/任务画像）+ Router（路径选择）+ Multi-Agent Planner（DAG 子任务）|
| **工具自主调用** | OperatorExecutor 的 `pandas / sqlite / json / regex / RAG` 工具栈；codegen prompt 里硬编码 source-tool 边界 |
| **异构数据推理** | source_capabilities 元数据驱动；不同 kind 走不同执行路径；StructuredDocExecutor 处理散文报告 |
| **结果合成输出** | answer_validator + local_repair + scorer 闭环；统一为 AnswerTable(columns, rows) |

---

## 3. Deterministic Task Compiler

唯一**完全无 LLM**的入口阶段。`agents/task_compiler.py`。

### 3.1 输出结构 `CompiledTask`

```
task_type            : "table_computation" | "mixed_context" |
                       "table_with_semantic_rule" | "document_qa" |
                       "image_understanding" | "pure_reasoning"
answer_type          : "scalar" | "boolean" | "table"
modalities           : ["table", "document", "image"]
operations           : ["retrieve", "extract", "filter", "join",
                        "groupby", "aggregate", "sort", "topk",
                        "compare", "compute", "format"]
primary_tool         : "pandas" | "sql" | "rag" | "vision" | "react"
auxiliary_tools      : […]
source_capabilities  : list[SourceCapability]   ←── 单一事实源
budget_level         : "small" | "medium" | "large" | "xlarge"
max_llm_calls        : int
max_tool_calls       : int
ambiguity_flags      : ["large_document_context", "many_files",
                        "structured_table_context", ...]
```

### 3.2 SourceCapability — 每个文件的"能力卡片"

| 字段 | 说明 |
| --- | --- |
| `path` / `kind` / `tool` | 路径、`csv`/`db`/`json`/`structured_table`/`doc` 等、可用的工具名 |
| `tables[].{table, columns, row_count}` | sqlite：实际存在的表 + 列 + 行数 |
| `columns` / `row_count` | csv：表头 + 行数 |
| `json_top_keys` / `json_record_fields` / `json_record_count` | json：顶级 keys + records 字段 |
| `structured_records` (≤3) / `structured_record_fields` / `structured_record_count` | structured_table：直接抽出的样本 records + 字段集 |
| `structured_record_splitter` / `structured_id_pattern` | structured_table：分段方式 + id 提取正则 |

**为什么这层重要**：codegen prompt、static_checker、local_repair、structured_doc_executor 全部读这一份。改一处，全链路一致；任何模型生成的代码都会被对照同一份元数据校验。

---

## 4. Router

### 4.1 路由策略

`agents/router.py`。**主路由依据是 `task_type`，不是 `difficulty`**（隐藏测试集多半不会暴露 difficulty）。

```yaml
task_type_routing:
  table_computation:        easy            # 单源表格 → operator_executor
  table_with_semantic_rule: tool_first_mixed
  mixed_context:            tool_first_mixed
  document_qa:              extreme         # 长文档 → operator_executor + RAG
  image_understanding:      medium          # → multi_agent fallback
  pure_reasoning:           medium          # → react
```

每条 route 携带：模型 / endpoint / max_steps / RAG 配置 / SC 配置。

### 4.2 失败级联（cascade）

```yaml
cascade_on_failure: true
cascade_order: [easy, tool_first_mixed, medium, hard, extreme,
                fallback_multi_agent]
cascade_max_extra_attempts: 1
```

route 失败时升档；每次只升一档以免空转。

### 4.3 启发式难度估算（仅当 difficulty 缺失时）

10 个手写特征：文件数 / 文件类型种类 / 总字节 / 题目长度 / 聚合关键词数 / 多跳关键词数 / 是否有 doc / 是否有 db / …  
得分映射 → Easy/Medium/Hard/Extreme，**只用来给 budget 加成**，不改变主路由。

---

## 5. Operator Executor — tool-first 主力

`agents/operator_executor.py`。所有 `kind=operator_executor` 的 route 都用它。

### 5.1 主流水线

```
        ┌──────────────────────────────────────────────────────┐
        │  ① codegen (one-shot)                                 │
        │     prompt 含 source_capabilities (硬约束)             │
        │     + mixed_context 多步范式模板                       │
        │     + structured_prose 抽取代码模板                    │
        │     + sample rows = 仅 dtype hint, 不可作为答案依据    │
        └─────────────────────┬────────────────────────────────┘
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │  ② static_check (零 LLM)                              │
        │     • AST 扫 pd.read_csv / sqlite3.connect 路径       │
        │     • 正则提 SQL FROM/JOIN 表名                       │
        │     • 比对 source_capabilities → 标 no_such_file /   │
        │       no_such_table / python_syntax                   │
        └─────────────────────┬────────────────────────────────┘
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │  ③ sandbox exec (子进程, 30s 超时)                    │
        │     落 CSV → AnswerTable                              │
        └─────────────────────┬────────────────────────────────┘
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │  ④ answer_validator                                   │
        │     列数=0 / 行数=0 / 空列 / ragged_row / mixed_types │
        └─────────────────────┬────────────────────────────────┘
                              │ 通过 → 返回
                              │ 失败 ↓
        ┌──────────────────────────────────────────────────────┐
        │  ⑤ local_repair (确定性 patch, ≤max_local_repairs)    │
        │     no_such_table   → 删 SQL JOIN 行                  │
        │     no_such_file    → 模糊匹配已知路径                │
        │     no_such_column  → 模糊匹配 column                 │
        │     ragged_row      → pad/truncate 行宽               │
        │     empty_column    → 投影删除该列                    │
        └─────────────────────┬────────────────────────────────┘
                              │ 修不了 ↓
        ┌──────────────────────────────────────────────────────┐
        │  ⑥ StructuredDocExecutor 抽取 (仅 structured_table)   │
        │     LLM 按 schema 抽 records → 落 .synthesized/*.csv  │
        │     注入 source_capabilities → 重跑 codegen           │
        │     磁盘缓存 (file_hash + schema_hash + endpoint)     │
        └─────────────────────┬────────────────────────────────┘
                              │ 仍失败 ↓
        ┌──────────────────────────────────────────────────────┐
        │  ⑦ schema_retry (LLM, 1 次)                           │
        │     喂全部 schema_scan + 失败上下文 → 重写程序        │
        └─────────────────────┬────────────────────────────────┘
                              │ 仍失败 ↓
        ┌──────────────────────────────────────────────────────┐
        │  ⑧ reasoner_repair (LLM, budget 限上限)                │
        │     用更强模型 + 严格"只输出 ```python``` "约束        │
        └──────────────────────────────────────────────────────┘
                              │ 仍失败 ↓
                          cascade 升档
```

### 5.2 静态检查器（亮点）

**`agents/static_checker.py`** 是把 LLM 失误降一档的关键防线：

```python
# 模型生成的代码片段
"SELECT r.constructorId FROM results r JOIN races ra ..."

# Source Capability 显示 db/results.db 只有一张 results 表
# Static check 直接抛 no_such_table，不进 exec，
# 转交 local_repair 把那行 SQL 注释掉。
```

省掉一次最贵的"拿到 sqlite OperationalError 才发现表不存在"的循环。

### 5.3 Local Repair 动作集（确定性）

| 触发码 | 动作 |
| --- | --- |
| `no_such_table` (静态 / OperationalError) | 把含错误表名的 SQL 行注释掉，留给后续重生成走多步范式 |
| `no_such_file` / `FileNotFoundError` | Levenshtein 找最接近的已知路径，字符串替换 |
| `no_such_column` / `pandas KeyError` | 在 capabilities 的所有列名中模糊匹配，重写 quote/SQL/bareword |
| `ragged_row` (validator) | pad/truncate 到 header 宽度 |
| `empty_column` (validator) | 投影删除该列 |

整个 local_repair 模块**完全无 LLM 调用**。

### 5.4 StructuredDocExecutor（散文报告抽取）

赛题里有任务给的是**长篇研究报告**（task_418 那种），records 散在 prose 中。RAG 不适配（要全量），regex 不可靠（句式多变）。

```
┌── doc/Patient.md (85 KB)  ──┐         ┌── doc/Laboratory.md (286 KB) ──┐
│  按段落 + id 提及切大块      │         │  同上                            │
│  → ~15 chunks @ 6KB          │         │  → ~50 chunks @ 6KB             │
└──────────┬───────────────────┘         └──────────┬─────────────────────┘
           ▼                                         ▼
   每块独立 LLM 调用 + 严格 JSON schema (字段集来自 source_capabilities)
   缓存 key = sha256(file_content + schema + endpoint + chunk)
           ▼
   合并 / 按 patient_id 去重 (后到的覆盖, 自动应用 "corrected to" 修订)
           ▼
   .synthesized/Patient.synth.csv  +  Laboratory.synth.csv
           ▼
   注入 compiled_task.source_capabilities → 重跑 codegen
   模型现在只需写一次 pandas filter/count
```

成本：单道 hard 任务首跑 60-70 次 ~1500-token 抽取调用，全部缓存；二跑零增量。

---

## 6. RAG（长文档检索）

`agents/document_retriever.py`。仅在 route 显式启用时挂上。

```
                [.md / .txt / .docx / json-path 全部支持]
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Loader + Chunker   │   header-aware 切段
                     │                      │   + sliding window 重叠
                     └──────────┬───────────┘
                                ▼
        ┌────────────────────────────────────────────────────┐
        │  Query Transformer                                 │
        │   • original                                       │
        │   • LLM paraphrases (Query Expansion)              │
        │   • Hypothetical Answer (HyDE)                     │
        └──────┬──────────────────────────┬──────────────────┘
               ▼                          ▼
       ┌─────────────────┐        ┌─────────────────────┐
       │ BM25Retriever   │        │ EmbeddingRetriever  │
       │ (Okapi BM25,    │        │ (OpenAI 兼容        │
       │  k1=1.5/b=0.75) │        │  /v1/embeddings)    │
       └────────┬────────┘        └─────────┬───────────┘
                └──────── RRF (rrf_k=60) ───┘
                          │
                          ▼
              ┌─────────────────────────────┐
              │ Cross-Encoder Reranker      │  optional
              │ (gte-rerank / Cohere /      │
              │  BGE-reranker)              │
              └──────────────┬──────────────┘
                             ▼
                       Top-K chunks (text only)
                             ▼
                       拼回 codegen prompt
```

embedding cache 落到 `artifacts/cache/embeddings/<hash>.json`，跨任务复用。

---

## 7. Multi-Agent Fallback

仅当所有 tool-first 路径 + reasoner_repair 全部失败时进入。`agents/orchestrator.py`。

```
   PlannerAgent          (LLM)  →  Plan(rationale, subtasks DAG)
        │
        ▼
   Specialist DAG               schema · sql · python · document · generic
        │   topo 分层 + ThreadPoolExecutor 并发
        ▼
   SynthesizerAgent      (LLM)  →  唯一允许调 `answer` 的角色
        │
        ▼
   迭代 refinement        synthesizer 失败时让 planner 再 plan 一次
        │
        ▼
   AnswerTable
```

每个 Specialist 拿到的是**受限工具子集**：sql 专家只能查 db，python 专家只能跑 pandas，document 专家只能读 md/json。降低工具误用概率。

---

## 8. Cross-Model Verify（列签名交集）

**只挂在 multi_agent fallback 路径上**（tool-first 已经有 static + local + reasoner repair，不付 2× LLM）。

```
primary route 出 answer_A
                                  ┐
                                  │
verifier route 并发跑 → answer_B  │ ──→  按列内容签名比对
                                  │
                                  ┘
       ┌──────────────────────────┴───────────────────────────┐
       ▼                          ▼                           ▼
  全列签名一致           列签名部分一致            完全不一致 (交集为空)
   ✓ 直接交 primary       → 取交集为最终 answer        ✓ 退回 primary
                          (去掉所有不一致列，
                           penalty 归 0)
```

数学保证：在官方公式 `score = max(0, recall − λ·extra/pred)` 下，只要不一致列里**真错的多于真对的**，取交集严格优于单模型。

---

## 9. 评分（与官方对齐）

`eval/column_match.py` 完全实现官方公式：

```
score = max(0, recall − λ · extra_pred_cols / max(pred_cols, 1))
recall = matched_cols / gold_cols
```

匹配规则要点：
- **列名忽略** / **行序忽略**
- 列内容签名 = 排序后归一化多重集
- 数值容差 `numeric_tolerance` (默认 1e-2)
- 字符串 strip + 大小写不敏感
- 二分最大匹配（DFS 增广路径）

**`run_single_task` 已经把这个评分内嵌**：当 gold 存在时直接打到屏幕日志，不必 score-run 第二次。

---

## 10. 横切机制

### 10.1 Budget Controller (`budget.py`)

| 计数器 | 上限默认 | 谁会消耗 |
| --- | --- | --- |
| `llm_calls` | TaskCompiler 估算 | 每次 OpenAI client.complete |
| `tool_calls` | TaskCompiler 估算 | 每次 sandbox exec / SQL 执行 |
| `local_repairs` | 3 | local_repair 每应用一次 |
| `reasoner_repairs` | 1 | reasoner_repair 每触发一次 |
| `multiagent_fallbacks` | 1 | multi_agent 每触发一次 |
| `max_seconds` | 0 (off) | 任意阶段超时 |

任一计数到顶 → 抛 `BudgetExceeded`，cascade 接管。**避免 repair 循环空转。**

### 10.2 ProgressLogger (`progress.py`)

事件流到 stderr，rich 渲染。事件包括：
`task_start / task_compiled / budget_started / router_decision / router_cascade / planner_done / specialist_start / specialist_done / synthesizer_done / tablellm_program / tablellm_executed / react_step / reasoner_repair_start/done / cross_verify_start/done / score / task_end`

CLI `--quiet` 关，`run-benchmark --show-events` 强开。

### 10.3 trace.json 固定 schema

```
trace.json
├─ task_id / succeeded / answer / failure_reason
├─ agent_mode = react | multi_agent | router
├─ compiled_task               { task_type, source_capabilities, ops, ... }
├─ router_decision             { route_name, kind, model, cascade_attempts }
├─ budget                      { llm_calls, tool_calls, local_repairs, ... }
├─ tablellm_direct / operator_executor
│    { program, exec_stdout/stderr, context_manifest, local_repair_log,
│      structured_doc_synthesis }
├─ multi_agent                 { plan, findings[], synthesizer_steps[] }
├─ reasoner_repair             { attempted, succeeded, program }
├─ cross_model_verify          { verifiers[], outcome, intersection }
├─ self_consistency            { num_samples, samples, decision }
├─ answer_validation           { valid, errors[], warnings[] }
└─ local_score                 { recall, penalty, score }
```

每加一个 stage 都必须开新顶层 key，不能内嵌。trace 解析器永远稳定。

---

## 11. 故障传播链（按从轻到重）

```
codegen 出错             ──► static_check 拦截     ──► local_repair (确定性)
              │                       │                              │
              │                       │ 标 no_such_table              │ 修不动
              │                       │       ↓                       │
              │                       │  drop SQL JOIN 那行            │
              │                       │       ↓                       │
              │                       │  rerun exec                   │
              │                       │                               ▼
              │                                            structured_doc_synth
              │                                                       │
              │                                          仅当含 structured_table
              │                                                       │
              │                                                       ▼
              ▼                                              schema_retry (1×)
       exec runtime error                                             │
              │                                                       ▼
              ▼                                            reasoner_repair (1×)
       answer_validator (空表/ragged/empty col)                       │
              │                                                       ▼
              ▼                                          cascade 升档下一条 route
       同走 local_repair / structured_doc / ...                       │
                                                                      ▼
                                                          multi_agent fallback
                                                                      │
                                                                      ▼
                                                       cross_model_verify
                                                                      │
                                                                      ▼
                                                                AnswerTable / 失败
```

每一步都有 trace 落字段、budget 计数、progress 日志事件。**所有失败模式都可观测，所有 LLM 调用都被预算约束。**

---

## 12. 三个真实例子（PPT 可秀的 worked examples）

### Easy · `task_19`
> *"List the full name of the Student_Club members that grew up in Illinois state."*

```
Profiler: kind={csv, json, doc(knowledge.md)}, task_type=mixed_context,
          tool=pandas + json + rag, budget=medium (≤14 LLM, ≤30 tool)
Router : task_type → tool_first_mixed → operator_executor (qwen3.5-35b-a3b)
Codegen: pd.read_csv('csv/member.csv') + json.load('json/zip_code.json')
         + merge by zip → filter state='Illinois' → project [first_name, last_name]
Static : ✓
Exec   : ✓
Validate: ✓ 3 rows × 2 cols
Score  : recall=1.0, penalty=0, → 1.000
```

### Hard · `task_415`
> *"What is the constructor reference name of the champion in the 2009 Singapore Grand Prix? Please give its website."*

```
Profiler: kind={db, json, structured_table doc/races.md},
          task_type=mixed_context, tool=pandas+sql+rag
Router : tool_first_mixed
Codegen: 
  raceId = re.search(r'Singapore.*Race ID:\s*(\d+)', races_md)  # 14
  cid    = sqlite3 SELECT constructorId FROM results
              WHERE raceId=14 AND positionOrder=1               # 1
  row    = pd.DataFrame(constructors_json['records'])
              .query('constructorId == @cid')
              [['constructorRef', 'url']]
Static : ✓ (静态检查放过单表 results 的 SQL，无 JOIN 到 races)
Exec   : ✓
Validate: ✓ 1 row × 2 cols ([mclaren, http://en.wikipedia.org/wiki/McLaren])
Score  : 1.000
```

### Extreme · `task_418`（散文报告）
> *"Among the patients whose creatinine level is abnormal, how many of them aren't 70 yet?"*

```
Profiler: kind={structured_table doc/Patient.md, structured_table
                doc/Laboratory.md, doc(knowledge.md)},
          task_type=table_with_semantic_rule, tool=pandas+rag
Router : tool_first_mixed → operator_executor
① codegen: 模型见到 structured_prose_report + sample records + 抽取模板，
           尝试一发；可能成功，也可能吐 empty / syntax error
② static_check: 通过
③ exec: 失败 / 空答案
④ local_repair: 散文这种情况确定性 patch 不适用
⑥ StructuredDocExecutor 触发:
     Patient.md (85KB)    → 15 chunks → LLM 抽取 → 30+ patient records
     Laboratory.md (286KB) → 50 chunks → LLM 抽取 → 600+ lab records
     去重合并 → .synthesized/doc__Patient.synth.csv (患者 + 生日)
                  doc__Laboratory.synth.csv (患者 + 化验值)
     注入 source_capabilities
② 重跑 codegen:
     pd.merge(patients, labs, on='patient_id')
     filter creatinine > threshold (knowledge.md)
     filter (today - birthday).years < 70
     count
Validate: ✓ 1 row × 1 col
Score  : 1.000
```

成本：首跑 65 次抽取调用（约几块钱），缓存命中后 0 增量。

---

## 13. 关键代码定位（按答辩时被追问的概率排序）

| 题目 | 文件 / 函数 |
| --- | --- |
| 难度怎么判断 | `agents/router.py` `_pick_route` + `_estimate_difficulty` |
| 怎么拆解任务 | `agents/task_compiler.py` `compile_task` (deterministic) + `agents/planner.py` `plan` (multi_agent only) |
| 工具怎么调 | `tools/registry.py` + `agents/operator_executor.py` (codegen→exec) |
| 异构数据怎么统一 | `agents/task_compiler.py` `SourceCapability` |
| 长文档怎么检索 | `agents/document_retriever.py` (BM25 + dense + RRF + rerank) |
| 散文报告怎么处理 | `agents/structured_doc_executor.py` |
| 答案怎么校验 | `eval/answer_validator.py` |
| 错怎么修 | `agents/local_repair.py` + `agents/static_checker.py` |
| 评分公式怎么实现 | `eval/column_match.py` (官方公式 1:1 复刻) |
| 预算怎么管 | `budget.py` |
| 调用过程可视化 | `progress.py` (事件流) + `agents/model.py` `StreamSink` (raw token) |

---

## 14. 答辩时 4 个核心 talking points

**1. "我们不堆模型，先用程序判定能不能确定地修。"**
静态检查 + 模糊匹配 + 列签名校验全部零 LLM；只有真正绕不过的语义问题才进 reasoner。这一条把 LLM 调用从 N 次循环砍到 1-2 次。

**2. "Source capabilities 是单一事实源。"**
profiler 一次扫描，prompt / 静态检查 / 局部修复 / 抽取器 全部从这一份元数据读，不会出现"prompt 说一套，校验做另一套"的漂移。

**3. "散文报告也能 tool-first。"**
对 task_418 这类长篇 prose-report，我们做了一个 LLM-driven 结构化抽取阶段，先把散文转成内存 CSV，主 codegen 退化成普通 pandas 一句 filter。比直接喂全文给主模型省 token 又稳。

**4. "评分公式即损失函数。"**
列签名匹配 + recall − λ·extra/pred 是官方评分；我们本地 1:1 实现，self-consistency 投票口径、cross-model intersection 取交集、validator 删空列，全部围绕这条公式做工程优化。

---

## 15. 一份可直接搬进 PPT 的"功能矩阵表"

| 维度 | 实现 | 决定性优势 |
| --- | --- | --- |
| 任务画像 | TaskCompiler (deterministic) | 零 LLM、可审计 |
| 路由 | task_type 路由 + 启发式难度估算 + 失败级联 | 测试集无 difficulty 也能跑 |
| 主执行 | OperatorExecutor (codegen + static + exec + validate + repair) | tool-first, 模型可错可修 |
| 散文/长文档 | RAG 流水线 (BM25 + dense + RRF + rerank) + StructuredDocExecutor | 双策略覆盖 doc QA 和散文报告 |
| 修错 | 静态检查 + 确定性 LocalRepair → 受限 ReasonerRepair | 大部分错不调模型修 |
| 兜底 | Multi-Agent (planner + specialist DAG + synthesizer) | 复杂题最后一击 |
| 答案二次校验 | answer_validator + cross_model_verify (列签名交集) | 自动避免 penalty |
| 预算 | BudgetController (5 个独立计数器) | 防 repair 循环空转 |
| 可观测 | trace.json 固定 schema + ProgressLogger 事件流 + StreamSink raw token | 答辩可现场 demo |
| 缓存 | embedding cache + extraction cache | 二跑零增量 LLM 成本 |
| 评分 | column-signature scorer (官方公式 1:1) | 本地评分 = 线上评分 |

---

PPT 时按 0 → 1 → 5 → 11 → 12 → 14 → 15 这条线讲，6 - 8 张片就能完整覆盖。
