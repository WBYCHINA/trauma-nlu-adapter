#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Step2-Open 具体类型分类器（按组别）
用法：python step2_open_detail_test.py --group mechanical
加载 models/step2_open_{group}_lora/
"""

import os
import sys
import argparse

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

GROUP_EN_MAP = {
    "mechanical": "机械性损伤",
    "thermal": "热力损伤",
    "electrical": "电能损伤",
    "chemical": "化学腐蚀",
    "radiation": "辐射损伤",
    "bioinfection": "生物毒素-感染",
    "pathological": "病理自发性破溃",
}

GROUP_TO_TYPES = {
    "mechanical": ["擦伤", "切割伤", "刺伤", "撕裂伤", "撕脱伤", "挫裂伤", "火器伤", "抓咬伤", "开放性骨折", "爆炸碎片伤", "挤压破裂"],
    "thermal": ["火焰烧伤", "烫伤破溃", "冻伤破溃"],
    "electrical": ["电烧伤", "电弧烧伤"],
    "chemical": ["强酸腐蚀", "强碱腐蚀", "化学腐蚀"],
    "radiation": ["放射性溃疡", "紫外灼伤破溃"],
    "bioinfection": ["海洋生物伤", "毒虫咬伤", "脓肿破溃", "感染性坏疽"],
    "pathological": ["压疮破溃", "缺血性坏疽", "糖尿病足溃疡", "静脉性溃疡", "肿瘤破溃", "黏膜破损"],
}

GROUP_DEFINITION = {
    "mechanical": "外力机械作用导致的皮肤破损",
    "thermal": "高温或低温导致的皮肤破溃",
    "electrical": "电流通过或电弧放电导致的皮肤坏死破溃",
    "chemical": "酸碱或其他化学物质的侵蚀反应导致的皮肤破溃",
    "radiation": "放射线或强紫外线的能量作用导致的皮肤溃疡或剥脱",
    "bioinfection": "动物叮咬、毒虫毒素、感染性细菌等生物因素导致的皮肤坏死、溃疡、坏疽",
    "pathological": "疾病、缺血、压力、肿瘤等身体内部病变自发导致的皮肤/黏膜破溃",
}

def load_model(group_en):
    lora_path = os.path.join(BASE_DIR, "models", f"step2_open_{group_en}_lora")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model = PeftModel.from_pretrained(model, lora_path)

    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    return model, tokenizer

def predict(model, tokenizer, text, group_en):
    cn = GROUP_EN_MAP[group_en]
    types_str = "、".join(GROUP_TO_TYPES[group_en])
    defn = GROUP_DEFINITION[group_en]
    prompt = f'<|im_start|>user\n开放式受伤分类 — 组别「{cn}」（{defn}），仅从以下类型选择：{types_str}\n\n伤情：{text}<|im_end|>\n<|im_start|>assistant\n'

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    result = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    for t in GROUP_TO_TYPES[group_en]:
        if t in result:
            return t
    return f"未知（原始: {result}）"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True,
                        choices=list(GROUP_EN_MAP.keys()),
                        help="组别英文名")
    args = parser.parse_args()

    group_en = args.group
    group_cn = GROUP_EN_MAP[group_en]
    print(f"加载 step2_open_{group_en}_lora — 组别「{group_cn}」")
    print(f"可选类型：{', '.join(GROUP_TO_TYPES[group_en])}\n")

    model, tokenizer = load_model(group_en)

    while True:
        try:
            text = input("> ").strip()
            if text.lower() == 'q':
                break
            if not text:
                continue
            print(f"  → {predict(model, tokenizer, text, group_en)}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    main()
