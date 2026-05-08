# 怎么跑（按硬件分档）

> 16GB Mac → 走 **路径 A**（lite 配置 + Ollama 小模型）。
> 24GB+ NVIDIA GPU 工作站 / Colab A100 → 走 **路径 B**（全尺寸 + vLLM）。
> 如果你只想验证 pipeline 不在乎成本 → 走 **路径 C**（纯云 endpoint）。

三条路径用的代码完全一样，只是 `agent.api_base` / `agent.model` 不同。

---

## 0. 共通准备（任何路径都先做）

```bash
cd /Users/maruixin/Downloads/kdd

# 1. 装项目依赖（一次性）
curl -LsSf https://astral.sh/uv/install.sh | sh    # 如果你还没有 uv
uv sync

# 2. 看看公共数据 / config 是不是齐
uv run dabench status --config configs/react_baseline.example.yaml
```

`status` 应该打出 `dataset_root: present` 和 `Public tasks: 50` 左右，否则数据没解压好。

---

## 1. 路径 A：16GB Mac 本机（推荐你用这个开发）

思路：用 **Ollama** 在本机起一个 OpenAI 兼容 server，挂一个 4-bit 量化的小模型当所有路由的统一后端，先把架构跑通再说。

> 16GB 内存能扛得住的尺寸大概是 4B-Q4 或 7B-Q4。准确率会比 8B/13B 全精度差，但**架构、router、specialist DAG、self-consistency、评分这些能力都能完整演示**——这就是 lite 配置的目的。

### 1.1 装 Ollama 并拉一个小模型

```bash
# 安装（一次性）：https://ollama.com/download/mac
brew install ollama         # 或者从官网下 .pkg

# 启动后台服务（开机后默认就在 :11434 端口）
ollama serve &

# 拉一个轻量替身。两个选项：
#   a) Qwen3-4B Q4，约 2.6GB 显存/内存，最稳
ollama pull qwen3:4b

#   b) Qwen2.5-Coder-7B Q4，代码任务更强但更吃内存
ollama pull qwen2.5-coder:7b
```

测试 Ollama 跑没跑起来：

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"hi"}]}'
```

返回 JSON 带 `choices[0].message.content` 就 OK。

### 1.2 用 lite 配置跑

仓库里我已经放好了一份 **`configs/router.lite.yaml`**，所有路由共用同一个 Ollama endpoint，路由架构（router → 三条路径 → SC → 评分）原样保留。

```bash
# 单任务跑通（最快验证）
uv run dabench run-task task_19 --config configs/router.lite.yaml

# 看 trace
cat artifacts/runs/<run_id>/task_19/trace.json | head -80

# 跑 5 个任务做小冒烟
uv run dabench run-benchmark --config configs/router.lite.yaml --limit 5

# 拿本地分（gold 已经在 data/public/output/）
uv run dabench score-run artifacts/runs/<run_id> --config configs/router.lite.yaml
```

### 1.3 Mac 上建议的运行节拍

- `run.max_workers: 2`（lite 里默认 2，Ollama 单实例本质上是串行的，给到 2 已经够）
- `agent.self_consistency.num_samples: 1`（先关，跑通再开多采样）
- `agent.router.routes.extreme.self_consistency.num_samples: 2`（Extreme 想感受多采样的话只在这一档开）
- `task_timeout_seconds: 1500`（小模型慢，给宽点）

如果 4B 模型挡不住有些 hard 任务，先把 `agent.router.difficulty_routing.Hard` / `Extreme` 都降到 `medium`（即 ReAct 单 agent，不开 multi-agent）跑一轮，把链路验证通了再回头加难度。

---

## 2. 路径 B：有 GPU 的机器 / Colab A100（出榜单分用）

只有这条路径能跑全尺寸 Qwen3-8B + TableLLM-13b，比赛冲分用。

### 2.1 起两个 vLLM server

```bash
pip install vllm

# Server 1: TableLLM-13b 在 :8001（Easy 路径）
python -m vllm.entrypoints.openai.api_server \
  --model RUCKBReasoning/TableLLM-13b \
  --served-model-name tablellm-13b \
  --host 0.0.0.0 --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.45 &

# Server 2: Qwen3-8B 在 :8000（Medium / Hard / Extreme 路径）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --served-model-name qwen3-8b-data-agent \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 &
```

> 单卡 24GB（3090/4090）能装下两个 4-bit 量化版；A100 40/80GB 可以全精度。
> 想用 LoRA 微调过的 Qwen3-8B：把 `--model` 换成你 merge 后的目录即可。

### 2.2 跑（用 router.example.yaml 模板，改两个 URL）

```bash
cp configs/router.example.yaml configs/router.yaml
# 编辑 configs/router.yaml：
#   easy.api_base    -> http://localhost:8001/v1
#   easy.model       -> tablellm-13b
#   medium/hard/extreme.api_base -> http://localhost:8000/v1
#   medium/hard/extreme.model    -> qwen3-8b-data-agent
#   把所有 api_key 改成 EMPTY 或任意非空字符串

uv run dabench run-benchmark --config configs/router.yaml --limit 50
uv run dabench score-run    artifacts/runs/<run_id> --config configs/router.yaml
```

### 2.3 Colab 上跑

如果你只有 Colab A100，可以把这台机器当成"远端推理后端"：用 cloudflared / ngrok 把 Colab 上的 vLLM 端口暴露到公网，本地 router config 的 `api_base` 改成那个公网 URL；推理在 Colab，本地只做 orchestration。这样 16GB Mac 也能驱动比赛级别的模型。

---

## 3. 路径 C：纯云 endpoint（最快上手，按调用量花钱）

任何 OpenAI 兼容的服务都能直接挂上来。两个角色：

- **Easy 路径需要的 TableLLM-13b**：
  - 自己用 [HuggingFace Inference Endpoints](https://huggingface.co/inference-endpoints) 起一个 dedicated endpoint（约 $0.6-1.5/小时），endpoint 自带 OpenAI-compatible URL。
  - 或者退一步——把 Easy 路径的模型改成 SiliconFlow / OpenRouter 上的 `Qwen2.5-Coder-7B-Instruct`，code-solution 能力相近。配置上只需要把 `easy.kind` 保持 `tablellm_direct`，model 字段填上替身模型即可。
- **Medium / Hard / Extreme 路径的 Qwen3-8B**：
  - SiliconFlow（国内）有 Qwen 系列，按 token 收费且有免费额度。
  - OpenRouter 全球聚合，多家 provider 同款 Qwen 价格透明。
  - DeepInfra / Together AI 也都有。

把这些 endpoint 的 base URL + key 填进 `configs/router.yaml` 就能跑。

```bash
uv run dabench run-benchmark --config configs/router.yaml --limit 50
uv run dabench score-run    artifacts/runs/<run_id> --config configs/router.yaml
```

> 16GB Mac + 路径 C 是另一个可选起点：本地零模型负担，全部走云推理。优点是发挥稳定，缺点是要花钱。可以和路径 A 互相切换——开发期用 lite 验证逻辑，提交前临阵切换到大模型 endpoint 出分。

---

## 4. 调试小抄

任何路径下都通用：

```bash
# 看一个任务的工具调用链 / planner 输出 / specialist findings
uv run dabench inspect-task task_415 --config configs/router.lite.yaml
cat artifacts/runs/<run_id>/task_415/trace.json | jq '.multi_agent.plan'
cat artifacts/runs/<run_id>/task_415/trace.json | jq '.multi_agent.findings[].summary'

# 临时把所有任务都丢给 ReAct（绕过 router 排错）
uv run dabench run-task task_415 \
  --config configs/router.lite.yaml \
  --mode react

# 多采样一次性开起来
uv run dabench run-task task_415 \
  --config configs/router.lite.yaml \
  --num-samples 3
```

`trace.json` 里关键字段：
- `router_decision` — 这个任务被路由到了哪条路径、用了哪个 endpoint
- `tablellm_direct.program` — Easy 路径下 TableLLM 写出来的 Python 代码
- `multi_agent.plan` — Hard/Extreme 路径下 planner 给出的 DAG
- `multi_agent.findings[].summary` — 每个 specialist 的报告
- `synthesizer_steps` — 合成器的最后一击
- `self_consistency.voted_signatures` — 多采样时每个列签名收到的票数

---

## 5. 常见问题

**Q: 路径 A 上 TableLLM 直出代码经常跑不出 answer？**
A: 4B 模型写 pandas 的稳定性确实不如 13B。lite 配置里默认让 Easy 路径也指向 Qwen 替身（不是真的 TableLLM-13b），但 TableLLMDirectAgent 的 prompt 还是 code-solution 风格，所以中等水平模型也能干。如果某些 Easy 任务跑不通，可以临时把它们路由到 medium：编辑 `configs/router.lite.yaml` 的 `difficulty_routing` 把 `Easy: easy` 改成 `Easy: medium`。

**Q: synthesizer 一直没产出 answer，进了 refinement loop 还是失败？**
A: 多半是 specialist 的 finding 没把答案表小化好。看 `findings[].artifact_columns / artifact_rows`：如果 sql 专家返回了 200 行明细，synthesizer 会被淹没。把 plan 里相应 subtask 的 `expected_output` 写得更具体能改善（例如 "返回单值"）。

**Q: 提交答辩里要写部署，怎么不暴露说我用的是某商业 API？**
A: `ARCHITECTURE.md` 里所有 LLM 都已经表述成"本地 vLLM endpoint 上的 Qwen3-8B"和"开源 TableLLM-13b"，这两个组合在路径 B 下是真实可复现的。开发期你用什么后端是工程选择，不需要写在架构文档里。
