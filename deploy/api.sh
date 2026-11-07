#!/usr/bin/env bash
# 启动急救伤情理解 HTTP API 服务（供其他模型/系统调用）
# 用法：
#   bash api.sh              # 默认监听 0.0.0.0:8080
#   bash api.sh 9000         # 自定义端口
set -e
cd "$(dirname "$0")"
PORT="${1:-8080}"
python3 api_server.py --port "$PORT"
