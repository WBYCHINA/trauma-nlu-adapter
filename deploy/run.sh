#!/usr/bin/env bash
# 运行急救伤情理解推理
# 用法：
#   bash run.sh                          # 交互式
#   bash run.sh --text "被刀割伤直流血"   # 单句测试
set -e
cd "$(dirname "$0")"
python3 pi_inference.py "$@"
