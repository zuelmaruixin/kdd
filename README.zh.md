# DataAgent-Bench 技术报告

## 摘要

本报告说明本项目针对 KDD Cup 2026 DataAgent-Bench public split 构建的
data-agent 系统。DataAgent-Bench 的任务形式为：给定一个自然语言问题与
若干上下文文件，系统需要输出一张结构化表格。上下文文件可能包含 CSV、
SQLite、JSON、Markdown/TXT 规则文档以及结构化程度较弱的自然语言记录。
与一般开放域问答不同，该任务的答案必须能够由给定数据推导，模型仅凭先验
直接回答通常无法获得稳定分数。

本系统采用“模型负责规划与生成操作，确定性模块负责约束与验证”的设计思
路。系统首先对任务上下文进行确定性扫描，得到文件类型、字段、样本值、任
务类型和预算建议；随后由路由器选择执行路径。默认配置以
`react_harness` 作为主路径，在主路径无法产生合法答案时，再根据错误类型
进入 agentic/operator 或 multi-agent 兜底路径。最终答案在写出前经过结构
校验，并按照列内容签名匹配的评分方式进行本地评估。

public split 的本次记录为：49 条记录均展示并纳入评分，47 条生成有效预测，
2 条失败；总分为 35.600 / 49，平均分为 0.727，其中 34 个任务得分为 1.0。
这些结果表明，当前系统能够处理多数公开任务，但在部分列投影、语义修复和
超时场景上仍存在明确改进空间。

---

## 1. 任务与评分简述

每个任务由 `task.json` 和 `context/` 目录组成。`task.json` 给出问题与难
度标签，`context/` 提供可使用的数据文件。系统最终只需提交一张二维表格
`prediction.csv`。

评分按列内容签名匹配：列名和行顺序不参与匹配，单元格值经过数值容差、大
小写和空白归一化后形成无序多重集签名。若预测列与标准答案列签名相同，则
该列视为命中。公式为：

```text
recall  = matched_cols / gold_cols
penalty = lambda * extra_pred_cols / pred_cols
score   = max(0, recall - penalty)
```

其中 `extra_pred_cols` 是未匹配到标准答案的多余列。由此可见，本任务并不
鼓励“多给一些字段”。系统必须同时做到召回正确列和控制冗余列。

---

## 2. 系统设计目标

本项目围绕公开任务中观察到的失败模式进行系统设计。主要失败模式包括：

1. **模型未充分读取数据即作答**。部分任务的自然语言问题看似简单，但实际
   答案依赖完整表格或规则文档。仅根据文件预览或模型先验作答会造成列内容
   错误。
2. **字段语义映射错误**。任务常涉及自然语言概念与真实字段之间的映射。
   例如问题中的概念可能对应某个字段值，而非字段名本身；若模型把概念词直
   接当作列名或过滤值，会产生“形状正确但内容错误”的答案。
3. **工具协议不稳定**。若工具调用依赖模型输出自由文本 JSON，复杂
   observation、转义字符或括号不匹配都可能导致解析失败。
4. **错误缺乏可恢复信息**。Python、SQL 或答案格式错误如果只返回原始异常，
   模型容易重复同一错误，而不是切换到更合适的检视方式。
5. **程序生成路径存在静态错误**。在 codegen 路径中，模型可能引用不存在
   的文件、表名或列名；这些问题如果等到运行时才暴露，会消耗更多时间和模
   型调用。
6. **预算与超时控制不足**。复杂任务可能在验证、修复或多路径尝试中耗尽时
   间，需要明确的任务级边界与失败记录。

为应对这些问题，本系统采用以下设计原则：

- **先扫描，再推理**：在任何模型求解前，先用确定性程序扫描上下文文件，
  形成统一的任务画像。
- **工具行为显式化**：模型需要通过工具读取数据、执行 SQL 或运行 Python，
  中间动作被记录到 `trace.json` 中。
- **本地检查优先**：结构检查、静态检查、风险判断和部分修复尽量由确定性
  规则完成，只有在必要时才升级为额外模型调用。
- **路径失败可复盘**：路由选择、失败原因、兜底路径和最终答案均写入结构
  化 trace，便于定位错误阶段。

---

## 3. 总体架构

系统的整体流程如下：

```text
task.json + context/
        |
        v
TaskCompiler
  - 扫描文件类型、字段、样本值、行数、规则文档
  - 推断 task_type、answer_type、operations、ambiguity_flags
        |
        v
Router
  - 依据任务类型与执行画像选择 route
  - 主路径为 react_harness
  - 失败时根据 failure_type 选择兜底路径
        |
        v
React Harness
  - native tool calling
  - planner
  - retry hints
  - answer guard
        |
        +-------------------------------+
        |                               |
        v                               v
Agentic / Operator fallback        Multi-agent fallback
  - codegen                         - planner
  - static checker                  - specialists
  - local repair                    - synthesizer
  - semantic judge
        |
        v
Answer Validator
        |
        v
prediction.csv + trace.json
        |
        v
Column-match scorer
```

系统入口位于 `src/data_agent_baseline/cli.py`，批量运行逻辑位于
`src/data_agent_baseline/run/runner.py`。每个任务会生成一个独立输出目录，
其中包括 `prediction.csv` 和 `trace.json`。`prediction.csv` 用于评分；
`trace.json` 用于记录任务画像、路由决策、工具步骤、预算消耗、答案校验
结果和兜底路径。

### 3.1 上下文进入模型的方式

本系统没有把 `context/` 下的文件整体拼接进 prompt。不同阶段接收的是不同
粒度的上下文表示：

| 阶段 | 代码位置 | 送入模型的内容 | 目的 |
| --- | --- | --- | --- |
| React 预规划 | `react_planner.py:_context_summary()` | 由 `list_context_tree()` 得到的文件路径、文件类型和大小，最多保留前 25 个文件条目 | 让模型先形成工具调用顺序，而不是提前读取数据内容 |
| React 主循环 | `prompt.py:build_task_prompt()` 与工具 observation | 初始只包含问题、难度和规则；文件内容必须通过工具逐步读取 | 避免模型在未检视数据时直接作答，并保留可复盘的工具轨迹 |
| Multi-agent planner | `planner.py:_build_context_overview()` | `render_compact_schema()` 生成的文件清单、schema 和行数，不包含数据行 | 给 planner 足够的结构信息用于拆分任务 |
| Operator/codegen | `tablellm_direct.py:render_context_for_codegen()` | `context_render.py` 生成的表格样本、JSON 摘要、文档相关章节或 RAG top-K 片段 | 让 codegen 能写出类型和字段匹配的程序，同时控制输入长度 |

因此，报告中所说的“预规划”和“上下文渲染”不是把全量文件交给模型。预规划
阶段只做轻量文件发现；程序生成阶段也只接收按预算筛选后的切片。真正需要
完整数据时，系统要求模型通过 `execute_python` 或 `execute_context_sql`
在任务目录中读取原始文件并完成计算。这个设计使 prompt 规模可控，同时让
最终答案仍基于完整数据执行，而不是基于预览样本推断。

---

## 4. 任务画像与路由机制

### 4.1 确定性任务画像

`src/data_agent_baseline/agents/task_compiler.py` 实现了无 LLM 的任务编译
层。该层不负责解题，而是生成下游模块共享的事实快照。主要输出字段包括：

| 字段 | 含义 |
| --- | --- |
| `task_type` | 任务类型，如 `table_computation`、`document_qa`、`mixed_context`、`record_text_with_semantic_rule` |
| `answer_type` | 答案形态，如 scalar、boolean 或 table |
| `source_capabilities` | 每个上下文文件的 schema、行数、样本值、SQLite 表结构等 |
| `operations` | 根据问题推断出的候选操作，如 filter、join、aggregate、retrieve、compute |
| `foreign_key_candidates` | 基于样本值重叠推断出的候选 join 关系 |
| `ambiguity_flags` | 风险标记，如 large context、record text、semantic rule、unsupported file |
| `execution_profile` | 对上下文规模、数据形态、可验证性和推荐策略的概括 |

该层的关键作用是统一“系统对数据的认知”。例如，prompt、静态检查器和程序
修复器都应基于同一份 `source_capabilities` 工作。如果模型 prompt 中看到
的字段和静态检查器使用的字段不一致，就会出现难以归因的系统漂移。因此，
任务画像被设计为所有下游模块共享的参照系。

### 4.2 路由原则

`src/data_agent_baseline/agents/router.py` 不直接让模型选择执行路径。路由
优先依据 `task_type_routing` 和 `execution_profile`，而不是单纯依据
`difficulty`。难度标签主要用于预算提示；当任务标签缺失时，router 会使用
文件数量、模态数量、上下文大小、问题长度、聚合词和多跳词等特征估计任务
难度，并把估计特征写入 trace。

默认示例配置 `configs/agentic_router.example.yaml` 中，所有 task type 首
选 `react_harness`。这是因为主路径具备较稳定的工具调用、答案校验和错误
恢复机制。agentic/operator 路径被保留为兜底和消融实验路径，而不是在默认
配置中替代主路径。

### 4.3 按错误类型兜底

系统没有采用简单的 “easy -> medium -> hard -> extreme” 线性升档，而是按
失败类型选择后继路径。原因是不同错误需要不同处理方式：列名错误需要更强
的 schema 约束，文档检索失败需要更充分的检索路径，预算耗尽则可能需要更
综合的 fallback。

当前 router 会识别若干 `failure_type`，包括：

- `budget_exhausted`；
- `unsupported_file_type`；
- `retrieval_empty` 或 `doc_context_miss`；
- `semantic_consistency_failed`；
- `missing_answer`、`syntax_error`、`static_error`、`exec_error`；
- `zero_rows`。

每次尝试及其结果都会进入 `router_decision.cascade_attempts`。这样在复盘
失败任务时，可以清楚看到初始路径、失败原因、是否触发兜底，以及最终采用
的路径。

---

## 5. 主路径：React Harness

主路径由 `src/data_agent_baseline/agents/react.py` 实现。它沿用 ReAct 的
“模型思考、选择工具、观察结果、继续行动”基本模式，但围绕工具协议、错误
提示和答案提交做了约束。

### 5.1 Native tool calling

工具描述由 `src/data_agent_baseline/tools/registry.py` 中的 `ToolSpec`
统一维护。每个工具同时有两种视图：

- 面向 prompt 的人类可读说明；
- 面向 OpenAI-compatible tools API 的 JSON Schema。

这样可以减少 prompt 声称的工具参数与实际 API schema 不一致的问题。对支
持 tools API 的模型，服务端可在返回前对工具参数结构做基础校验。若某些
服务端偶尔返回普通文本而非 tool calls，系统仍保留文本 JSON 解析路径作
为回退。

### 5.2 预规划

`react_planner.py` 在部分任务开始前生成简短的子任务计划，包括建议读取的
文件、可能需要的工具和大致计算顺序。该计划仅作为提示，而不是硬性执行约
束。这样做的原因是，任务执行过程中可能发现真实字段名、数据类型或文件结
构与问题表述不同；若计划过度刚性，反而会阻止模型根据新证据调整路径。

需要强调的是，React 预规划并不读取文件内容。`make_plan()` 传给模型的是
问题、难度，以及 `_context_summary()` 生成的截断文件清单：每个条目只有
相对路径和文件大小。它的作用是决定“先列文件、再读规则文档、再查表格/数
据库、最后提交答案”这类操作顺序，而不是从文件内容中直接抽取答案。

multi-agent 的 planner 与 React 预规划也不同。它使用
`render_compact_schema()` 构造 context overview，包含文件清单、schema 和
行数，但不包含数据行。也就是说，预规划阶段使用的是结构摘要，不是全量数
据；后续 specialist 或工具调用仍需要继续读取具体切片或执行计算。

### 5.3 Retry hints

当工具调用失败时，`react_retry_hints.py` 会把错误转换为结构化提示。例
如：

- `KeyError` 会提示先确认真实列名；
- SQL `no such table` 会提示检查 SQLite schema；
- 文件路径错误会提示重新枚举 context 下的真实路径；
- `answer` 行宽不一致会提示统一行宽。

同类错误重复出现时，提示会从“修复当前错误”转向“更换策略”。这相当于把
一部分错误反思能力外化为确定性规则，而不是完全依赖模型自我反省。

### 5.4 React Cheap Guard

`react_answer_guard.py` 是主路径中的第一个 Cheap Guard。它只在模型第一次
调用 `answer` 时运行，不调用额外 LLM，而是根据已经发生的工具轨迹和候选
答案做确定性风险评估。其输出复用 `semantic_guard.py` 中的
`SemanticRisk` / `SemanticRiskAssessment` 数据结构，因此 React 路径与
Operator 路径的风险记录在 trace 中具有一致形态。

该 guard 的设计动机是避免两类极端：一方面，完全不检查会让“形状正确但内
容错误”的答案静默通过；另一方面，无条件 self-verify 会增加成本，也可能
让模型把已正确答案改错。因此系统采用“有证据才升级”的策略。

当前 React Cheap Guard 主要检查五组信号：

| 风险代码 | 触发条件 | 处理含义 |
| --- | --- | --- |
| `execution_failed_or_missing_answer` | `answer` 工具没有产生候选表格 | 直接视为高风险 |
| `empty_answer_rows` / `invalid_answer` | 候选答案为空或未通过结构校验 | 需要复核或进入后续失败处理 |
| `answer_without_computation` | 在提交前没有成功调用 `execute_python` 或 `execute_context_sql` | 说明模型可能只看了样本预览或凭先验回答 |
| `knowledge_doc_not_read` | context 中存在 knowledge / rule / glossary 文档，但没有成功读取 | 说明模型可能忽略了任务内定义的领域规则 |
| `shape_mismatch_*` | 问题暗示 scalar/list，但答案行列形态明显不一致 | 作为 warning 累计风险分 |

风险分由各项风险权重相加并截断到 1.0；若存在 error 级风险，或累计分数达
到阈值，`should_escalate=True`。此时 React Harness 不立即提交第一次
answer，而是把 `guard_risk_codes` 和 `guard_top_risk` 写入 observation，
要求模型基于具体风险再提交一次。若未触发风险，则第一次答案直接提交。

这个设计把“是否需要再想一轮”从模型自我判断中拿出来，交给可审计的本地证
据。它并不判断答案语义一定正确，但能拦截几类代价很低、影响很大的错误：
未计算即作答、未读规则文档、空答案和答案形状明显不匹配。

### 5.5 预算敏感恢复

系统使用 `BudgetController` 记录 LLM 调用、工具调用、修复次数和任务级时
间。当 self-verify 阶段触发预算异常且已有 draft answer 时，React Harness
会保留该 draft，并在 trace 中记录预算耗尽原因。这一策略针对的是“答案已
经形成，但验证阶段消耗超限”的场景，可以避免把已有有效结果直接丢弃。

预算控制不是单一的总时长限制，而是分层记录不同类型的资源消耗。普通模型
调用和工具调用分别计数；local repair、reasoner repair 和 multi-agent
fallback 也有独立上限。这样做的原因是，不同阶段的收益和风险不同：一次
数据检视通常是必要成本，而多轮修复如果持续失败，就容易吞掉整道题的预算。
因此系统在进入修复或兜底前先检查对应预算，触发上限时写入明确的
`budget_exhausted:*` 原因，而不是让任务在无效路径中继续运行。

---

## 6. 工具层、知识工具与长文档检索

工具层提供对任务上下文的受控访问。核心工具包括：

- `list_context`：列出 context 下可用文件；
- `read_csv`、`read_json`、`read_doc`、`head_doc`、`grep_doc`：读取或检
  索上下文文件；
- `inspect_sqlite_schema`、`execute_context_sql`：检查并查询 SQLite；
- `execute_python`：在任务 context 目录中运行 Python；
- `answer`：提交最终表格；
- `consult_knowledge`：在 helper model 启用时解释规则文档。

### 6.1 受控执行环境

`execute_python` 使用独立 `multiprocessing.Process` 执行代码，工作目录
固定为当前任务的 `context/`。stdout 与 stderr 被重定向到临时文件，再由
父进程读取。若代码超过 30 秒未结束，子进程会被终止并返回超时错误。该机
制不是完整安全沙箱，但对本项目而言具有三个实际作用：隔离死循环，避免执
行命名空间污染主进程，并在异常退出时保留可读的错误信息。

SQLite 查询工具只允许只读语句，避免模型执行写操作破坏上下文数据。文件
读取工具会限制单次返回内容规模，以减少超长 observation 对后续模型调用
造成的干扰。

React 主循环中的文件读取也采用切片式 observation，而不是一次返回完整文
件。`read_csv` 默认只返回有限行数，并支持 `columns_only=true` 只取表头
和行数；`offset` 可用于分页查看后续行。`read_doc` 与 `read_json` 通过
`max_chars` 和 `offset` 返回片段，单次字符数有上限；`head_doc` 只返回文
档开头若干行；`grep_doc` 返回关键词匹配行及其周围上下文。这个工具设计
与 system prompt 中的约束一致：样本行只能用于判断字段、类型、格式和
join key，最终结果必须通过 Python 或 SQL 在完整数据上计算。

### 6.2 `consult_knowledge` 规则文档工具

部分任务包含 `knowledge.md`、`rules.md`、`glossary.txt` 或类似命名的规
则文档。主模型如果只读取表格，容易忽略这些文档中定义的阈值、公式或字段
语义。为此系统提供 `consult_knowledge` 工具，在配置启用时由 React 路径
调用。

该工具的职责不是直接生成答案，而是回答关于规则文档的窄问题。其执行逻辑
包括：

1. 在 context 内解析显式文件列表；若未指定文件，则自动寻找 knowledge、
   rule、definition、glossary、schema notes 等命名模式的 Markdown/TXT
   文档。
2. 对路径做 context 边界检查，避免工具读取任务目录之外的文件。
3. 将规则文档按字符预算合并为带文件名的片段，连同模型提出的具体问题一
   起发送给 helper model。
4. 在系统提示中约束 helper model：只能使用给定文档，不得发明阈值、公式
   或 cut-off；若文档未说明，应显式返回 `not_specified`。
5. 将简短解释作为普通 observation 返回给 React 主循环，由主模型再决定
   如何把规则转化为筛选条件、派生字段或最终答案。

这个设计把“读规则”和“执行数据计算”分开。规则解释由更聚焦的 helper 调
用完成，数据操作仍由主路径通过 Python、SQL 或文件读取工具完成。这样既
降低规则文档被忽略的概率，也避免 helper model 直接越权替系统生成最终表
格。

### 6.3 长文档与 RAG 检索

对于较长 Markdown、TXT、DOCX 或 JSON 上下文，系统不能把全文无差别塞入
prompt。`document_retriever.py` 提供面向 Operator 路径的检索组件，其流
程为：

```text
load document
        |
        v
chunk by heading / JSON path
        |
        v
BM25 sparse retrieval
        |
        +-------------------------+
        | optional dense retrieval |
        +-------------------------+
        |
        v
rank fusion
        |
        v
optional query expansion / hypothetical answer
        |
        v
optional cross-encoder rerank
        |
        v
top-K rendered context
```

文档切分保留标题路径，JSON 切分保留对象路径，因此返回片段不仅有正文，也
有来源位置。默认稀疏检索采用 BM25，适合实体名、ID、字段名等精确匹配；
当配置了向量模型时，系统可加入 dense retrieval，并用 Reciprocal Rank
Fusion 按排名融合稀疏和稠密结果。若启用 query expansion，系统会生成若
干改写查询和一个假设性答案段落，以提高长文档中同义表述的召回率。若启用
reranker，则先取较大的候选集合，再由 cross-encoder 对 query 与 chunk
联合打分，最后保留 top-K。

该检索链路采用渐进增强原则：外部向量模型、query expansion 或 reranker
不可用时，系统仍退化为 BM25 检索，而不是阻断任务。这样可以让长文档任务
获得更稳定的证据片段，同时保持默认路径的可运行性。

---

## 7. Operator 兜底路径

在部分任务中，一段完整的 pandas / SQL 程序比多轮 ReAct 更直接。为此，
系统保留 agentic/operator 兜底路径。当前默认配置并不把它作为所有任务的
首选路径，而是在主路径失败或消融实验中使用。

### 7.1 程序生成

`tablellm_direct.py` 会构造 codegen prompt，要求模型生成 Python 程序并把
最终结果赋给变量 `answer`。执行外壳负责把 `answer` 归一化为 DataFrame，
再写为 CSV。若模型返回的是 Series、dict、list 或标量，外壳也会尝试转换
为二维表格。

Codegen prompt 中的 `Available context` 由
`render_context_for_codegen()` 生成，不是原始文件拼接。该函数按任务形态
选择三种渲染方式：

- 若 route 提供 RAG 参数，则调用 `render_with_rag()`：表格文件保留 schema
  和少量代表性样本，文档和 JSON 进入检索流程，只把 top-K 片段送入 prompt；
- 若提供问题文本但未启用 RAG，则调用 `render_focused()`：表格保留 schema
  和样本行，文档按问题关键词选择相关章节；
- 否则调用 `render_with_samples()`：主要面向表格任务，提供 schema 与少
  量分散样本行。

`context_render.py` 对不同文件类型有不同切片策略。CSV 会展示列名、行数
和 head/spread/tail 样本；SQLite 会展示表结构、列类型、行数和少量样本；
JSON 会展示 top-level keys、record 数量和有限 record 示例；Markdown/TXT
会按标题切分并选择相关 section；超出预算的文件只保留一行 stub。渲染结
果同时写入 `context_manifest`，记录每个文件的路径、类型、大小、行数、章
节数、检索结果和是否截断，便于解释模型当时实际看到了哪些切片。

因此，Operator 路径的设计不是让模型直接“看完所有文件后写答案”，而是让
模型基于结构摘要和代表性切片写出可执行程序；程序运行时再从 `context/`
读取完整文件。这个分离可以减少 prompt 中的无关内容，也能避免模型把样本
行误当成完整数据。

同时，prompt 要求模型维护 `debug_steps`，记录：

- 实际读取到的字段；
- 使用过的字段；
- 过滤条件；
- join keys；
- 中间行数；
- 使用的规则文档内容；
- 与语义计划不一致时的 override 原因。

这些信息不是面向评分的答案，而是面向系统审计的中间证据。语义检查模块可
以根据 `debug_steps` 判断程序是否真的执行了它声称的字段映射和过滤逻辑。

### 7.2 Schema grounding

`schema_grounding.py` 负责在自然语言概念和真实字段之间建立候选关系。它
综合字段名相似度、低基数字段样本、字段类型和 cardinality 等信号，向模型
提示可能相关的列。该模块只给出候选，不直接替模型做最终绑定。这样可以避
免在候选不唯一时，把一次不确定判断固化为系统性错误。

### 7.3 静态检查

`static_checker.py` 在程序运行前进行 AST 层面的检查。它会追踪 DataFrame
变量与数据源之间的关系，并检查以下问题：

- 打开的文件路径是否存在于任务 context；
- SQLite 查询中的表名是否存在；
- DataFrame 下标访问、groupby、sort、drop_duplicates 等使用的列是否存
  在；
- merge 的 `on`、`left_on`、`right_on` 是否是对应 DataFrame 的真实列；
- object-wrapped JSON 是否被 `pd.read_json` 误读为嵌套对象列；
- 程序是否缺少最终 `answer` 赋值。

静态检查的目标是在执行前发现确定性错误，并把错误转换为结构化
`StaticIssue`，供后续修复器使用。

### 7.4 Local Repair

`local_repair.py` 是 Operator 路径中最重要的确定性修复层。它的基本原则
是：能由静态错误、运行时错误或答案结构错误唯一确定的修复，不再请求模型
重写；只有当本地修复无法安全判断时，才进入 schema-guided LLM retry 或其
他兜底路径。

外层调度位于 `repair_coordinator.py:local_repair_loop()`。该循环在三种情
况下触发：

1. 程序执行成功但答案为零行时，构造 `zero_rows` issue，尝试修复常见的
   dtype 或过滤条件漂移；
2. 程序执行成功但答案未通过 `answer_validator` 时，调用
   `repair_answer_table()` 修复可局部处理的表格结构问题；
3. 程序执行失败时，合并 `static_checker.check_program()` 和
   `issues_from_exec_error()` 的结果，再交给 `try_program_repair()`。

`try_program_repair()` 中 fixer 的顺序是固定的，按“越安全越靠前”的原则排
列。当前顺序如下：

| 顺序 | 修复器 | 主要触发问题 | 修复策略 |
| ---: | --- | --- | --- |
| 1 | `repair_python_syntax` | `python_syntax` | 去除泄漏到程序中的 Markdown fence，并用 `ast.parse` 验证 |
| 2 | `repair_missing_answer_assignment` | `missing_answer_assignment` | 从顶层变量中按优先级选择结果变量，追加 `answer = <var>` |
| 3 | `repair_json_records_read_with_pandas` | `json_records_read_with_pandas` | 将 `pd.read_json("x.json")` 改为 `json.load` 后读取 object-wrapped records |
| 4 | `repair_no_such_table` | `no_such_table` | 注释掉引用不存在表的 SQL 行，并把问题交给后续重写路径 |
| 5 | `repair_no_such_file` | `no_such_file` | 在已知 context 路径中按后缀和编辑距离替换最接近的文件名 |
| 6 | `repair_pandas_keyerror_or_no_such_column` | `no_such_column` / `pandas_keyerror` | 使用静态检查器提供的 `available_columns` 和 `closest_matches` 做保守列名替换 |
| 7 | `repair_bad_join_key` | `bad_join_key` | 仅当静态检查给出唯一 join key 候选且左右键同名时替换 |
| 8 | `repair_merge_dtype_mismatch` | `merge_dtype_mismatch` | 在 merge 前插入左右 key 的 `astype(str)`；ID 字段额外去除 `.0` 后缀 |
| 9 | `repair_zero_row_common_filters` | `zero_rows` | 规范化 ID 字段字符串、布尔字符串比较等常见零行过滤原因 |

此外，`repair_answer_table()` 不改程序，而是直接修复答案 dict。它只处理
两类结构问题：`ragged_row` 会将行补齐或截断到 header 宽度；
`empty_column` 会删除整列为空的列。这类修复发生在答案层，独立于程序层
fixer。

Local Repair 的关键不是“尽量多修”，而是“只在证据充分时修”。例如列名替
换要求相似度达到阈值；若静态检查器给出多个 `closest_matches`，本地修复
不会猜测。join key 修复也只接受唯一候选且左右字段名一致的情况。对于
`link_to_*` 这类语义相近但方向不同的字段，修复器会主动放弃，避免把概率
错误固化为确定性错误。

该层的输出会写入 `local_repair_log` 或
`post_schema_retry_local_repair_log`，包括修复轮次、issue、采取的 action、
重新执行是否成功以及后续失败原因。后面的 Cheap Semantic Guard 会把“是否
经历过 local repair / post-schema repair”也作为风险信号，因为深层修复路
径本身说明初始程序的语义可靠性较低。

### 7.5 Cheap Semantic Guard 与语义一致性

Operator 路径中的 Cheap Guard 位于 `semantic_guard.py`，由
`operator_executor.py` 在需要时调用。它与 React Cheap Guard 的定位一致：
不直接替代 LLM judge，而是在不调用 LLM 的情况下判断一个“看似成功”的程序
结果是否足够低风险，可以直接通过，或是否应升级到 Semantic Analyst /
Judge 流程。

Cheap Semantic Guard 的输入包括：

- `CodegenRunResult` 的成功状态、答案和执行输出；
- `debug_steps` 中记录的 schema inspection、used columns、filters、
  join keys、intermediate counts、knowledge rules used；
- `CompiledTask` 中的真实 schema、任务类型、操作集合和执行画像；
- Local Repair 与 schema retry 的历史；
- `schema_grounding.py` 给出的概念到字段候选。

它主要检查以下风险：

| 风险类型 | 典型代码 | 含义 |
| --- | --- | --- |
| 执行或结构失败 | `execution_failed_or_missing_answer`、`invalid_answer` | 程序没有可靠地产生表格 |
| 修复路径可疑 | `suspicious_fallback:*` | 答案依赖 schema retry 或较深 local repair |
| trace 缺失 | `trace_missing:schema_inspection`、`trace_missing:used_columns`、`trace_missing:filter_conditions`、`trace_missing:join_keys` | 程序没有留下足够证据证明其读取、过滤或连接正确 |
| trace 与 schema 冲突 | `trace_unknown_used_columns` | `debug_steps` 声称使用的字段不在真实 schema 中 |
| 中间结果异常 | `suspicious_trace:zero_intermediate_count` | 中间行数出现 0，提示过滤或 join 可能错误 |
| 概念无法可靠落地 | `grounding_unmapped`、`grounding_low_confidence`、`grounding_ambiguous` | 风险概念在 schema 中没有明确候选或候选竞争 |
| 字段覆盖不一致 | `field_coverage_mismatch` | 问题中的风险概念有候选字段，但程序 trace 没有使用这些字段 |
| 公式使用不匹配 | `knowledge_formula_without_question_trigger` | 程序使用了规则公式，但问题并未明确要求该派生指标 |

若 Cheap Semantic Guard 判断需要升级，`operator_executor.py` 会先尝试强制
生成 semantic plan；如果语义分析器仍未产出 plan，则构造一个低置信度的
fallback plan，把 cheap guard 发现的风险写入 `uncertainties` 和
`consistency_checks`，确保后续 judge 不会被静默跳过。随后
`semantic_consistency.py` 执行 judge 与必要的 semantic repair。

因此，Operator 路径实际有两层 Cheap Guard：React 主路径的
`react_answer_guard.py` 守住答案提交边界，Operator 路径的
`semantic_guard.py` 守住“程序执行成功但语义可能不对”的边界。二者都遵循
同一原则：低风险样例快速通过，高风险样例带着明确证据进入更重的验证流程。

### 7.6 Record-text 处理

对于结构化程度较弱的自然语言记录，`structured_doc_executor.py` 先尝试抽
取结构化 records，再交给标准表格计算流程。该设计把文本理解和表格计算分
开：前者负责从段落中恢复字段，后者负责筛选、聚合和输出。分阶段处理可以
提高可复盘性，也便于定位错误来自抽取还是计算。

### 7.7 Reasoner Repair

`reasoner_repair.py` 是比 Local Repair 更重的一层修复，但它仍不是“重新解
题”路径。它只接收失败程序、失败原因、结构校验结果、上下文摘要、stdout
和 stderr，要求 repair model 输出一个新的 Python 程序，并且必须把最终结
果赋给 `answer`。

Router 对 Reasoner Repair 设置了较严格的触发条件：

- 只有 agentic/operator/tablellm 这类程序路径失败时才考虑；
- 失败 payload 中必须存在上一轮程序；
- 任务画像需要显示 `needs_reasoner=True`，或任务是带有大文档上下文的
  `document_qa`；
- 对应的 reasoner repair 预算尚未耗尽。

修复程序生成后会立即在任务 context 中执行，并重新读取输出表格。若执行失
败、没有产生结果表，或预算不足，系统不会把该轮修复视为成功。成功时，修
复结果会作为新的 payload 返回，同时保留原始失败 payload，方便比较修复前
后的程序和答案。

这一设计与 Local Repair 形成分层关系：Local Repair 处理证据充分、可机
械改写的问题；Reasoner Repair 处理需要模型综合失败证据进行局部重写的程
序问题。二者都避免从空白 prompt 重新生成完整方案，以降低修复阶段引入新
错误的概率。

### 7.8 Multi-agent 兜底

系统还保留 multi-agent fallback。其结构为 planner、specialists 和
synthesizer：planner 将问题拆成若干可执行子目标，specialists 分别进行
数据读取、检索或计算，synthesizer 再把中间结果合并为最终表格。默认配置
下它不是主路径，因为 public 任务中多数样例可以由更稳定的 React Harness
完成；multi-agent 更适合作为预算允许时的复杂失败兜底。

multi-agent fallback 也受 `BudgetController` 约束。Router 只在相应失败
类型和预算条件满足时进入该路径，并把尝试记录写入
`cascade_attempts`。因此它在报告中的定位不是“更强的万能模型”，而是一个
可审计的后备执行组织方式。

---

## 8. 输出验证与评分对齐

`answer_validator.py` 在写出 `prediction.csv` 前执行结构检查，覆盖：

- 缺失答案；
- 零列；
- 零行；
- ragged rows；
- 整列为空；
- mixed-type column warning。

验证失败的答案会被标记为失败，使 router 有机会进入兜底路径。该模块不判
断语义正确性，只保证输出至少是可评分的二维表格。

`column_match.py` 实现本地列匹配评分。由于官方评价对列名不敏感，但对多
余列有惩罚，系统在 prompt、answer guard、operator codegen 和最终评分中
都围绕“只输出问题要求的列”建立约束。self-consistency 与 optional
cross-model verification 也以列签名为单位，而非以整张表的字符串表示为
单位。

### 8.1 Self-consistency 列签名投票

当配置多样本运行时，系统不会简单选择最长答案或最后一个成功答案，而是按
官方评分的列匹配逻辑做 self-consistency 聚合。每个成功样本先被转置为若
干列，再对每列计算内容签名。签名计算继承本地评分器的归一化规则，包括数
值容差、大小写归一化和空白归一化。

聚合时，每个样本对同一列签名最多投一票；系统统计所有成功样本的签名票数，
并以样本列数的众数作为目标列数。满足 `min_votes` 的签名按票数排序，保
留到目标列数为止。最终表格不是重新拼接任意列，而是优先选择一个真实样本：
该样本需要最大覆盖获胜签名，并尽量少包含额外列。若该样本包含所有获胜签
名，系统再把它投影到获胜列集合。

这种设计与评分函数保持一致：投票对象是“列内容”而不是“自然语言列名”。
它可以减少单次采样中偶然多出调试列或遗漏某列造成的波动，也避免把来自不
同样本的行对齐关系随意混合。

### 8.2 Cross-model verification

cross-model verification 用于在成本允许时检查主模型答案是否得到其他模型
支持。Router 会复制主 route 的执行设置，只替换 verifier 的模型或端点，
从而让 verifier 在相同工具、相同路由类型和相同约束下独立求解。

验证阶段同样以列签名为单位。主答案中的每一列会与 verifier 结果的列签名
集合比较；只有获得至少 `min_agreement` 个模型支持的主答案列才被保留。
如果交集为空，系统保留主答案并在 trace 中记录
`empty_intersection_keep_primary`，避免因 verifier 全部失败而把已有答案
删空。如果交集非空且小于主答案列集合，则用交集投影替换主答案，并重新执
行结构校验。

该机制主要针对“主模型输出了正确列之外的额外列”这一评分风险。由于官方
评价惩罚多余列，模型间一致的列更可能是稳定答案；但系统也保留空交集回退，
避免验证器不稳定时造成过度删除。

### 8.3 语义一致性审计

Operator 路径中的 semantic consistency 不只输出 pass/fail，还会在
`semantic_consistency_audit` 中记录 plan、judge 和 repair 是否真正运行。
审计字段包括 `plan_ran`、`judge_ran`、`judge_attempts`、
`judge_final_verdict`、`judge_repaired_code` 和 `final_gate`。

`final_gate` 是复盘时最直接的字段：`plan+judge` 表示语义计划和 judge 均
正常运行；`escalated+judge` 表示初始计划失败或被 Cheap Guard 升级后，
系统仍进入了 judge；`cheap_guard_only` 表示 Cheap Guard 发现风险但后续
语义计划未能建立；`bypassed` 表示系统认为风险较低，没有进入重验证。这
些字段使我们能区分“未发现风险所以跳过”和“应该验证但验证链路失败”两种
完全不同的情况。

---

## 9. Public 评测结果与分析

本次 public 记录如下：

| 指标 | 数值 |
| --- | ---: |
| Records | 49 |
| Shown | 49 |
| Success | 47 / 49 |
| Failed | 2 |
| Total score | 35.600 / 49 |
| Mean score | 0.727 |
| Scored | 49 / 49 |
| Score = 1.0 | 34 |

本地评分文件为：

```text
artifacts/runs/batch-20260516-204243/score_summary.json
```

其中 `total_score=35.6`，`mean_score=0.726531`，`mean_recall=0.744898`。
49 个任务都生成了 `trace.json`，其中 47 个任务生成了 `prediction.csv`。
两条未生成预测的任务分别对应：

- `task_344`：任务级运行超过 600 秒；
- `task_396`：语义修复后仍未通过静态检查。

满分任务数量为 34。其余任务中，一部分为 0 分，一部分获得部分分。例如
`task_38` 得分 0.60，`task_249` 得分 0.25，`task_379` 得分 0.75。这类
部分分说明系统在部分任务中找到了部分正确列，但输出了额外列或遗漏了标准
答案列。结合评分公式，这类错误应优先从“最终列投影”和“字段语义绑定”两
个环节排查。

从运行轨迹看，多数任务由 `react_harness` 完成，少量任务进入 agentic 路
径。默认配置下 operator/multi-agent 并非主要得分来源，而是用于失败后的
可控补救。该结果与系统设计定位一致：主路径先保证协议稳定和答案结构，兜
底路径再处理程序化或语义风险较高的失败样例。

---

## 10. 主要优化点

### 10.1 工具协议稳定化

系统将工具协议从自由文本动作描述转为 OpenAI-compatible native tool
calling，并由统一 `ToolSpec` 维护工具名称、说明和参数 schema。这样可以
减少由格式错误造成的无效步骤。对于服务端未返回工具调用的情况，系统仍保
留文本解析回退，从而提高端点兼容性。

### 10.2 确定性风险检查

系统在多个边界设置了非 LLM 检查：

- React answer guard 检查答案提交前的证据充分性；
- Answer validator 检查最终表格结构；
- Static checker 检查 codegen 程序中不存在的路径、表名和列名；
- Semantic guard 检查程序结果与任务语义之间的显著不一致。

这些检查的共同特点是低成本、可解释、可写入 trace。它们并不保证语义完全
正确，但能减少明显错误静默进入最终提交。

### 10.3 分层修复

当程序或答案出现错误时，系统优先使用本地确定性修复。只有当候选不明确或
本地规则无法处理时，才进入 schema-guided LLM retry、semantic repair 或
更重的 fallback。这样可以避免每个错误都触发完整重写，也降低“修复一个错
误又引入另一个错误”的概率。

### 10.4 可复盘 trace

每个任务的 `trace.json` 不只是日志，而是结构化复盘材料。推荐排查顺序为：

```text
compiled_task.task_type / execution_profile
        |
        v
router_decision.route_name / cascade_attempts
        |
        v
steps / tool observations / retry hints
        |
        v
answer_validation
        |
        v
operator program / static issues / local repair log
        |
        v
semantic_consistency_audit
        |
        v
score_summary / per-task score
```

这一结构使失败案例可以定位到任务画像、路由选择、工具执行、程序生成、修
复或最终列投影中的具体阶段。

---

## 11. 局限性与后续工作

当前系统在 public split 上取得了 0.727 的平均分，但仍有若干局限。

首先，列投影仍不够保守。评分函数对额外列有明确惩罚，部分任务虽然召回了
正确列，但同时输出了冗余列，导致得分低于 1.0。后续应加强最终 projection
阶段，使系统更明确地区分“中间调试字段”和“最终答案字段”。

其次，字段语义绑定仍是主要风险来源。即使真实字段已被扫描，模型仍可能把
问题概念映射到近似但错误的字段。后续可加强 schema grounding 与 answer
guard 的联动，在答案提交前检查关键概念是否真正落到被使用字段上。

第三，语义修复后的静态错误仍可能发生。当前修复流程在部分任务中能够恢复
失败程序，但也可能在重写后引入新的列名或语法问题。后续可对 semantic
repair 输出强制再执行静态检查和短路策略，避免长时间停留在无效修复循环。

第四，超时控制仍有优化空间。`task_344` 的失败说明单任务 600 秒上限仍可
被复杂路径耗尽。后续应在路由层更早识别低收益路径，并在多次错误同质化时
提前停止。

第五，对于视觉任务或强领域推理任务，目前系统主要依赖模型本身能力和兜底
路径，尚未实现专门的视觉执行器或领域知识模块。

这些限制说明当前系统仍是一个以工程稳健性为重点的 data-agent baseline，
而不是对所有任务类型都完全优化的最终系统。

---

## 12. 结论

本项目构建了一个以 React Harness 为主、以 agentic/operator 路径为兜底的
DataAgent-Bench 系统。系统的主要特点是：先用确定性任务画像统一上下文认
知，再让模型通过工具和程序完成数据操作，最后用结构检查、静态检查和语义
检查控制明显错误。public split 上的结果为 35.600 / 49，平均分 0.727，
说明该方法能够覆盖多数公开任务，但在最终列投影、字段语义绑定、复杂修复
和超时控制方面仍有改进空间。
