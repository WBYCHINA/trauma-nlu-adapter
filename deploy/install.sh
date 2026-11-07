#!/usr/bin/env bash
# Orange Pi 5 Pro (aarch64 Linux) 一键安装脚本
# 用法：bash install.sh
set -e

echo "[1/3] 安装系统依赖（编译 llama-cpp-python 需要 cmake + g++）"
sudo apt-get update
sudo apt-get install -y python3 python3-pip build-essential cmake

echo "[2/3] 升级 pip"
python3 -m pip install --upgrade pip

echo "[3/3] 安装 Python 依赖（CPU 版 llama-cpp-python，编译约需 5-10 分钟）"
# 明确关掉 CUDA/BLAS，纯 CPU 构建，避免因缺 CUDA 库而编译失败
CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_BLAS=OFF" \
  python3 -m pip install -r requirements.txt

echo ""
echo "✅ 安装完成。下一步：bash run.sh --text \"被刀割伤直流血\""
