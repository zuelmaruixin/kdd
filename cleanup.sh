#!/usr/bin/env bash
# 删掉所有跟 fine-tuning / 本地 vLLM / 旧 react_baseline 相关的过时文件。
# 项目现在只保留 router.deepseek.yaml + router.dashscope.yaml + router.lite.yaml + router.example.yaml 四份配置。
set -euo pipefail
cd "$(dirname "$0")"
rm -rf colab/ scripts/
rm -f configs/react_local_vllm.example.yaml \
      configs/router.deepseek.yaml.bak \
      configs/react_baseline.example.yaml \
      configs/react_baseline.yaml
echo "cleanup done."
ls configs/
