#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地边缘计算对话脚本 - 增强记忆版
使用 Qwen2.5-1.5B-Instruct 进行本地对话，支持多轮对话记忆
"""

import os
import sys

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "models", "cache")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "models--Qwen--Qwen2.5-1.5B-Instruct")

def find_local_model_path():
    snapshots_dir = os.path.join(MODEL_CACHE_PATH, "snapshots")
    if os.path.exists(snapshots_dir):
        for entry in os.listdir(snapshots_dir):
            entry_path = os.path.join(snapshots_dir, entry)
            if os.path.isdir(entry_path) and not entry.startswith("."):
                config_path = os.path.join(entry_path, "config.json")
                if os.path.exists(config_path):
                    return entry_path
    return None

LOCAL_MODEL_PATH = find_local_model_path()

if LOCAL_MODEL_PATH:
    print(f"[INFO] 发现本地模型: {LOCAL_MODEL_PATH}")
else:
    print("[ERROR] 未找到本地模型，请先下载！")
    print(f"[INFO] 预期路径: {MODEL_CACHE_PATH}/snapshots/")
    sys.exit(1)

def load_model():
    """加载模型"""
    print(f"[INFO] 正在加载模型...")
    print(f"[INFO] 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=CACHE_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=CACHE_DIR,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        model = torch.compile(model, mode="max-autotune")
        print("[INFO] 已启用 torch.compile 加速")
    
    print("[INFO] 模型加载完成！")
    print("[INFO] 输入 'quit' 或 'exit' 退出对话")
    print("[INFO] 输入 'history' 查看当前对话历史\n")
    
    return model, tokenizer

def chat(model, tokenizer, user_input: str, history: list) -> str:
    """生成回复（带记忆）"""
    system_prompt = """你是一个专业的急救指导助手，帮助没有专业急救知识的人处理外伤紧急情况。请记住之前的对话内容，在回答时参考之前的信息。请用简洁清晰的语言提供指导。"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in history:
        messages.append(msg)
    
    messages.append({"role": "user", "content": user_input})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()

def show_history(history: list):
    """显示对话历史"""
    if not history:
        print("当前没有对话历史\n")
        return
    
    print("\n对话历史（最近5轮）:")
    print("-" * 60)
    
    display_history = history[-10:]
    
    for i in range(0, len(display_history), 2):
        if i + 1 < len(display_history):
            round_num = (len(history) - len(display_history) + i) // 2 + 1
            print(f"第 {round_num} 轮:")
            print(f"  你: {display_history[i]['content']}")
            print(f"  AI: {display_history[i+1]['content']}")
            print()
    
    print("-" * 60)
    print(f"总对话轮数: {len(history) // 2} 轮\n")

def main():
    """主对话循环"""
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保已安装依赖: pip install torch transformers")
        sys.exit(1)
    
    history = []
    
    print("=" * 50)
    print("  智能外伤急救指导 AI - 本地对话")
    print("=" * 50)
    print()
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("\n再见！")
                break
            
            if user_input.lower() == "history":
                show_history(history)
                continue
            
            if not user_input:
                continue
            
            print("\nAI: ", end="", flush=True)
            
            response = chat(model, tokenizer, user_input, history)
            print(response)
            print()
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            if len(history) > 40:
                history = history[-40:]
            
            print(f"当前记忆 {len(history) // 2} 轮对话\n")
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")

if __name__ == "__main__":
    main()