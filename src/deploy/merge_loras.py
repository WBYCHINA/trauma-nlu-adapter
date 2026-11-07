# -*- coding: utf-8 -*-
"""
把各任务的 LoRA 适配器合并进同一个底座，输出独立完整模型（供后续转 GGUF）。

合并对象（推理实际用到的 4 个）：
  step1_type_lora                 -> models/merged/step1_type
  step2_open_group_lora           -> models/merged/step2_group
  step2_open_mechanical_detail_lora -> models/merged/step2_detail
  step3_location_lora             -> models/merged/step3_location

nlu_clean_lora 不合并：清洗逻辑已改为纯 base + 正则兜底，不参与推理。

用法：
  D:\\python\\python.exe src\\deploy\\merge_loras.py
"""

import os
import sys
import shutil

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HOME"] = "D:\\Trae CN\\Trae Project\\First AID Agent\\models\\cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\Trae CN\\Trae Project\\First AID Agent\\models\\cache"

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cache",
    "models--Qwen--Qwen2.5-1.5B-Instruct",
    "snapshots", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
MERGED_DIR = os.path.join(BASE_DIR, "models", "merged")

# 适配器名 -> 输出目录名
TASKS = {
    "step1_type_lora": "step1_type",
    "step2_open_group_lora": "step2_group",
    "step2_open_mechanical_detail_lora": "step2_detail",
    "step3_location_lora": "step3_location",
}


def merge_one(lora_name, out_name):
    lora_path = os.path.join(BASE_DIR, "models", lora_name)
    out_path = os.path.join(MERGED_DIR, out_name)
    if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        print(f"[SKIP] {lora_name}: 找不到 adapter_config.json")
        return

    print(f"[MERGE] {lora_name} -> {out_name}")
    # CPU float16 合并，1.5B 约 3GB 内存，够用
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, lora_path)
    model = model.merge_and_unload()  # 把 LoRA 焊进底座

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    model.save_pretrained(out_path, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    tok.save_pretrained(out_path)
    print(f"[DONE] 已保存 {out_path}")


def main():
    os.makedirs(MERGED_DIR, exist_ok=True)
    for lora_name, out_name in TASKS.items():
        try:
            merge_one(lora_name, out_name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] {lora_name} 合并失败: {e}")


if __name__ == "__main__":
    main()