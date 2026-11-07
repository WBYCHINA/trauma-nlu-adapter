#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 Step2 细分类型分类器（按 (type, group) 组合分训）

原理：Step1 给出 open/closed → Step2a 给出 group → 本脚本在 (type, group) 上下文下细分
用法：
  开放式机械性损伤: python step2_detail_train.py --type open --group mechanical --dataset mech_1.json
  闭合式机械性损伤: python step2_detail_train.py --type closed --group mechanical --dataset mech_closed_1.json
输出 LoRA: models/step2_{type}_{group}_detail_lora/
"""

import os
import sys
import json
import argparse

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HOME"] = "D:\\Trae CN\\Trae Project\\First AID Agent\\models\\cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\Trae CN\\Trae Project\\First AID Agent\\models\\cache"

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = os.path.join(BASE_DIR, "models", "cache")

TYPE_EN_MAP = {"open": "开放式", "closed": "闭合式"}
GROUP_EN_MAP = {
    "mechanical": "机械性损伤",
    "thermal": "热力损伤",
    "electrical": "电能损伤",
    "chemical": "化学腐蚀",
    "radiation": "辐射损伤",
    "bioinfection": "生物毒素-感染",
    "pathological": "病理自发性破溃",
}

# (type, group) → 细分类型列表
# 设计原则：同一个 group 在 open/closed 下的类型完全不同
TYPE_GROUP_TO_TYPES = {
    # === 开放式 ===
    ("open", "mechanical"): [
        "切割伤", "刺伤", "挫裂伤", "撕裂伤", "撕脱伤",
        "开放性骨折", "爆炸碎片伤", "火器伤", "挤压破裂", "抓咬伤", "擦伤",
    ],
    ("open", "thermal"): ["火焰烧伤破溃", "烫伤破溃", "冻伤破溃"],
    ("open", "electrical"): ["电烧伤破溃", "电弧烧伤破溃"],
    ("open", "chemical"): ["强酸腐蚀", "强碱腐蚀", "化学腐蚀溃烂"],
    ("open", "radiation"): ["放射性溃疡", "紫外灼伤破溃"],
    ("open", "bioinfection"): ["海洋生物伤", "毒虫咬伤", "脓肿破溃", "感染性坏疽"],
    ("open", "pathological"): ["压疮破溃", "缺血性坏疽", "糖尿病足溃疡", "静脉性溃疡", "肿瘤破溃", "黏膜破损"],

    # === 闭合式 ===
    ("closed", "mechanical"): [
        "闭合性骨折", "关节脱位", "扭伤", "拉伤", "挫伤",
        "脑震荡", "内脏损伤", "内出血",
    ],
    ("closed", "thermal"): ["一度烧伤（发红）", "轻度烫伤", "冻伤红斑期"],
    ("closed", "electrical"): ["电击伤（无出口伤口）"],
    ("closed", "chemical"): ["轻度化学灼伤（发红无破溃）"],
    ("closed", "radiation"): ["放射性皮炎（发红）", "轻度紫外晒伤"],
    ("closed", "bioinfection"): ["蜂窝织炎（未破溃）", "虫咬皮炎（无破皮）"],
    ("closed", "pathological"): ["压疮一期", "糖尿病足早期", "深静脉血栓"],
}

ALL_TYPES = list(TYPE_EN_MAP.keys())
ALL_GROUPS = list(GROUP_EN_MAP.keys())


def format_instruction(input_text, output_text, type_en, group_en):
    """type_en/group_en 保留为接口参数，标签集已内嵌进精简 prompt"""
    return (
        f'<|im_start|>user\n'
        f'判断开放机械损伤类型（切割/刺伤/挫裂/撕裂/撕脱/开放骨折/爆炸碎片/火器/挤压/抓咬/擦伤）：{input_text}'
        f'<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>'
    )


def load_and_prepare_dataset(dataset_path, type_en, group_en):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        formatted_data.append({
            "text": format_instruction(item["input"], item["output"], type_en, group_en)
        })

    dataset_dir = os.path.dirname(dataset_path)
    temp_path = os.path.join(dataset_dir, "_formatted.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False)

    dataset = load_dataset("json", data_files=temp_path)
    return dataset["train"]


def train(type_en, group_en, num_epochs=8, batch_size=16,
          dataset_file="train.json", force=False, max_length=128):
    if type_en not in TYPE_EN_MAP:
        print(f"错误: --type 必须是 {ALL_TYPES}")
        return
    if group_en not in GROUP_EN_MAP:
        print(f"错误: --group 必须是 {ALL_GROUPS}")
        return
    if (type_en, group_en) not in TYPE_GROUP_TO_TYPES:
        print(f"错误: ({type_en}, {group_en}) 组合未定义细分类型")
        return

    output_dir = os.path.join(
        BASE_DIR, "models", f"step2_{type_en}_{group_en}_detail_lora"
    )
    if os.path.isabs(dataset_file) or os.sep in dataset_file or "/" in dataset_file:
        dataset_path = dataset_file
    else:
        dataset_path = os.path.join(
            BASE_DIR, "data", "step1_nlu", "subtype",
            "detail", type_en, group_en, dataset_file
        )

    type_cn = TYPE_EN_MAP[type_en]
    group_cn = GROUP_EN_MAP[group_en]
    types = TYPE_GROUP_TO_TYPES[(type_en, group_en)]

    print(f"[Training] ({type_cn} + {group_cn}) → 细分类型")
    print(f"[Training] Model: {MODEL_NAME}")
    print(f"[Training] Dataset: {dataset_path}")
    print(f"[Training] Subtypes ({len(types)}): {types}")
    print(f"[Training] Output: {output_dir}")

    if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        if not force:
            print(f"\n⚠  已有训练结果存在: {output_dir}")
            print(f"   如确定覆盖，加 --force 参数")
            return
        else:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"[Training] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Training] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    model.print_trainable_parameters()

    dataset = load_and_prepare_dataset(dataset_path, type_en, group_en)
    print(f"[Training] Dataset loaded: {len(dataset)} samples")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=5,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        report_to="none",
        save_total_limit=1,
        save_steps=400,
        dataloader_num_workers=2
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print(f"[Training] Starting: epochs={num_epochs}, batch={batch_size}, max_len={max_length}")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Training] Done! LoRA saved → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=ALL_TYPES,
                        help="Step1 输出的类型: open(开放式) / closed(闭合式)")
    parser.add_argument("--group", type=str, required=True, choices=ALL_GROUPS,
                        help="Step2a 输出的组别: mechanical/thermal/electrical/chemical/radiation/bioinfection/pathological")
    parser.add_argument("--dataset", type=str, default="train.json",
                        help="JSON 文件名（默认去 data/.../detail/{type}/{group}/ 下找），也可直接传完整路径")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="强制覆盖已有训练结果")
    args = parser.parse_args()

    train(type_en=args.type, group_en=args.group,
          num_epochs=args.epochs, batch_size=args.batch_size,
          dataset_file=args.dataset, force=args.force,
          max_length=args.max_length)
