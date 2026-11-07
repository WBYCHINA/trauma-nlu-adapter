#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Step2-Open 分组器：识别开放式损伤所属组别
加载 models/step2_open_group_lora/
"""

import os
import sys

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
    "机械性损伤",
    "热力损伤",
    "电能损伤",
    "化学腐蚀",
    "辐射损伤",
    "生物毒素-感染",
    "病理自发性破溃",
]

GROUP_DEFINITION = {
    "机械性损伤": "外力机械作用导致的皮肤破损（碰撞、切割、挤压、撕扯、打击、骨折端刺破等）",
    "热力损伤": "高温或低温导致的皮肤破溃（火焰、烫伤、冻伤坏死等）",
    "电能损伤": "电流通过或电弧放电导致的皮肤坏死破溃（电烧伤、电弧烧伤）",
    "化学腐蚀": "酸碱或其他化学物质的侵蚀反应导致的皮肤破溃",
    "辐射损伤": "放射线或强紫外线的能量作用导致的皮肤溃疡或剥脱",
    "生物毒素-感染": "动物叮咬、毒虫毒素、感染性细菌等生物因素导致的皮肤坏死、溃疡、坏疽",
    "病理自发性破溃": "疾病、缺血、压力、肿瘤等身体内部病变自发导致的皮肤/黏膜破溃",
}

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model = PeftModel.from_pretrained(model, LORA_PATH)

    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    return model, tokenizer

def predict(model, tokenizer, text):
    groups_str = "、".join(GROUPS)
    defs = "\n".join(f"- {g}：{GROUP_DEFINITION[g]}" for g in GROUPS)
    prompt = f'<|im_start|>user\n判断开放式受伤所属组别（仅从以下选择：{groups_str}）\n{defs}\n\n伤情：{text}<|im_end|>\n<|im_start|>assistant\n'

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    result = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    for g in GROUPS:
        if g in result:
            return g
    return f"未知（原始: {result}）"

def main():
    model, tokenizer = load_model()
    print(f"模型加载完成：step2_open_group — 7 组分类器")
    print(f"输入伤情描述，返回组别。q 退出。\n")

    while True:
        try:
            text = input("> ").strip()
            if text.lower() == 'q':
                break
            if not text:
                continue
            print(f"  → {predict(model, tokenizer, text)}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    main()
