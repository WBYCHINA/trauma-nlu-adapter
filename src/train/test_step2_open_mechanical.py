#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 step2 开放式机械性损伤细分 LoRA：11 个细分类型识别
支持交互式测试和批量测试
"""

import os
import sys
import json
import argparse
from collections import defaultdict

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
MODEL_PATH = os.path.join(BASE_DIR, "models", "cache", "models--Qwen--Qwen2.5-1.5B-Instruct", "snapshots", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
LORA_PATH = os.path.join(BASE_DIR, "models", "step2_open_mechanical_detail_lora")
DATA_DIR = os.path.join(BASE_DIR, "data", "step1_nlu", "subtype", "detail", "open", "mechanical")

SUBTYPES = [
    "切割伤", "刺伤", "挫裂伤", "撕裂伤", "撕脱伤",
    "开放性骨折", "爆炸碎片伤", "火器伤", "挤压破裂", "抓咬伤", "擦伤",
]

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model = PeftModel.from_pretrained(model, LORA_PATH)
    return model, tokenizer

def extract_subtype(text):
    first_line = text.strip().split('\n')[0].strip()
    for s in SUBTYPES:
        if s in first_line:
            return s
    for s in SUBTYPES:
        if s in text:
            return s
    # 简短别称匹配
    aliases = {
        "切割": "切割伤", "刺伤": "刺伤", "挫裂": "挫裂伤",
        "撕裂": "撕裂伤", "撕脱": "撕脱伤", "开放骨折": "开放性骨折",
        "骨折": "开放性骨折", "爆炸碎片": "爆炸碎片伤",
        "火器": "火器伤", "挤压": "挤压破裂",
        "抓咬": "抓咬伤", "狗咬": "抓咬伤", "猫抓": "抓咬伤",
        "擦伤": "擦伤",
    }
    for alias, full in aliases.items():
        if alias in first_line:
            return full
    return None

def predict(model, tokenizer, text):
    prompt = f'<|im_start|>user\n判断开放机械损伤类型（切割/刺伤/挫裂/撕裂/撕脱/开放骨折/爆炸碎片/火器/挤压/抓咬/擦伤）：{text}<|im_end|>\n<|im_start|>assistant\n'

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return extract_subtype(result.strip()) or f"未知({result.strip()})"

def batch_test(model, tokenizer, test_file):
    test_path = os.path.join(DATA_DIR, test_file)
    with open(test_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"加载 {len(cases)} 条测试用例...\n")

    correct = 0
    total = len(cases)
    errors = []
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    sub_errors = defaultdict(list)

    for i, case in enumerate(cases):
        text = case["input"]
        expected = case["output"]
        result = predict(model, tokenizer, text)

        is_correct = result == expected
        cat_stats[expected]["total"] += 1
        if is_correct:
            correct += 1
            cat_stats[expected]["correct"] += 1
        else:
            errors.append((i + 1, text, expected, result))
            sub_errors[expected].append((text, result))

        status = "✓" if is_correct else "✗"
        print(f"[{i+1:3d}/{total}] {status} \"{text}\" → {result} (期望: {expected})")

    acc = correct / total * 100
    print(f"\n{'='*60}")
    print(f"总准确率: {correct}/{total} = {acc:.1f}%")
    print(f"{'='*60}")

    print("\n📊 按类型统计:")
    for s, s2 in cat_stats.items():
        a = s2["correct"] / s2["total"] * 100
        print(f"  {s:8s}: {s2['correct']:3d}/{s2['total']:3d} = {a:.0f}%")

    if errors:
        print(f"\n❌ 错误详情 ({len(errors)} 条):")
        for idx, text, expected, result in errors[:30]:
            print(f"  #{idx} \"{text}\"")
            print(f"     期望: {expected}  实际: {result}")
        if len(errors) > 30:
            print(f"  ... 还有 {len(errors)-30} 条")

    print(f"\n💡 按期望类型分组的错误:")
    for expected, errs in sub_errors.items():
        print(f"  期望「{expected}」错了 {len(errs)} 次:")
        for text, result in errs[:10]:
            print(f"    \"{text}\" → {result}")

    return acc

def interactive(model, tokenizer):
    print("Step2 开放式机械性损伤细分模型已加载（输入 q 退出）")
    print(f"可选类型: {'、'.join(SUBTYPES)}\n")
    while True:
        try:
            text = input("> ").strip()
            if text.lower() == 'q':
                break
            if not text:
                continue

            result = predict(model, tokenizer, text)
            print(f"  → {result}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="Step2 开放式机械性损伤细分测试")
    parser.add_argument("--batch", type=str, default=None,
                       help="批量测试JSON文件名（相对于data/step1_nlu/subtype/detail/open/mechanical/）")
    parser.add_argument("--interactive", action="store_true", default=True)
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.batch:
        batch_test(model, tokenizer, args.batch)
    else:
        interactive(model, tokenizer)

if __name__ == "__main__":
    main()
