#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 step2 组别分类 LoRA：7 个组别识别
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
LORA_PATH = os.path.join(BASE_DIR, "models", "step2_open_group_lora")

GROUPS = [
    "机械性损伤", "热力损伤", "电能损伤", "化学腐蚀",
    "辐射损伤", "生物毒素-感染", "病理自发性破溃",
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

def extract_group(text):
    first_line = text.strip().split('\n')[0].strip()
    for g in GROUPS:
        if g in first_line:
            return g
    for g in GROUPS:
        if g in text:
            return g
    return None

def predict(model, tokenizer, text):
    prompt = f'<|im_start|>user\n判断受伤组别（机械/热力/电能/化学/辐射/生物/病理）：{text}<|im_end|>\n<|im_start|>assistant\n'

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return extract_group(result.strip()) or f"未知({result.strip()})"

def batch_test(model, tokenizer, test_file):
    test_path = os.path.join(BASE_DIR, "data", "step1_nlu", "subtype", "group", test_file)
    with open(test_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"加载 {len(cases)} 条测试用例...\n")

    correct = 0
    total = len(cases)
    errors = []
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    group_errors = defaultdict(list)

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
            group_errors[expected].append((text, result))

        status = "✓" if is_correct else "✗"
        print(f"[{i+1:3d}/{total}] {status} \"{text}\" → {result} (期望: {expected})")

    acc = correct / total * 100
    print(f"\n{'='*60}")
    print(f"总准确率: {correct}/{total} = {acc:.1f}%")
    print(f"{'='*60}")

    print("\n📊 按组别统计:")
    for g, s in cat_stats.items():
        a = s["correct"] / s["total"] * 100
        print(f"  {g:12s}: {s['correct']:3d}/{s['total']:3d} = {a:.0f}%")

    if errors:
        print(f"\n❌ 错误详情 ({len(errors)} 条):")
        for idx, text, expected, result in errors[:30]:
            print(f"  #{idx} \"{text}\"")
            print(f"     期望: {expected}  实际: {result}")
        if len(errors) > 30:
            print(f"  ... 还有 {len(errors)-30} 条")

    print(f"\n💡 按期望组别分组的错误:")
    for expected, errs in group_errors.items():
        print(f"  期望「{expected}」错了 {len(errs)} 次:")
        for text, result in errs[:10]:
            print(f"    \"{text}\" → {result}")

    return acc

def interactive(model, tokenizer):
    print("Step2 组别分类模型已加载（输入 q 退出）")
    print(f"可选组别: {', '.join(GROUPS)}\n")
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
    parser = argparse.ArgumentParser(description="Step2 组别分类测试")
    parser.add_argument("--batch", type=str, default=None,
                       help="批量测试JSON文件名（相对于data/step1_nlu/subtype/group/）")
    parser.add_argument("--interactive", action="store_true", default=True)
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.batch:
        batch_test(model, tokenizer, args.batch)
    else:
        interactive(model, tokenizer)

if __name__ == "__main__":
    main()
