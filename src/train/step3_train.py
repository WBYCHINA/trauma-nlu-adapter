#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 Step3 部位识别（PLAN.md 定义）：识别伤情所在身体部位（8选1）
输出 LoRA: models/step3_location_lora/
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
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "step3_location_lora")
CACHE_DIR = os.path.join(BASE_DIR, "models", "cache")

LOCATIONS = [
    "头部", "颈部", "躯干", "手臂", "手部", "腿部", "足部", "多部位",
]

def format_instruction(input_text, output_text):
    return f'<|im_start|>user\n判断受伤部位（头/颈/躯干/手臂/手/腿/足/多发）：{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>'

def load_and_prepare_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        formatted_data.append({"text": format_instruction(item["input"], item["output"])})

    dataset_dir = os.path.dirname(dataset_path)
    temp_path = os.path.join(dataset_dir, "_formatted.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False)

    dataset = load_dataset("json", data_files=temp_path)
    return dataset["train"]

def train(num_epochs=8, batch_size=16, dataset_file="train.json", force=False, max_length=96):
    dataset_path = os.path.join(BASE_DIR, "data", "step1_nlu", "location", dataset_file)
    print(f"[Training step3_location] Loading model: {MODEL_NAME}")
    print(f"[Training step3_location] Dataset: {dataset_path}")
    print(f"[Training step3_location] Locations: {LOCATIONS}")

    if os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
        if not force:
            print(f"\n[Training step3_location] 已有训练结果存在: {OUTPUT_DIR}")
            print(f"   如确定覆盖，加 --force 参数")
            return
        else:
            import shutil
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"[Training step3_location] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Training step3_location] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    model.print_trainable_parameters()

    dataset = load_and_prepare_dataset(dataset_path)
    print(f"[Training step3_location] Dataset loaded: {len(dataset)} samples")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=5,
        save_strategy="no",
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print(f"[Training step3_location] Starting training with {num_epochs} epochs...")
    print(f"[Training step3_location] max_length={max_length}, batch_size={batch_size}, grad_accum=1")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[Training step3_location] Done! LoRA saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--dataset", type=str, default="train.json")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有训练结果")
    args = parser.parse_args()

    train(num_epochs=args.epochs, batch_size=args.batch_size, dataset_file=args.dataset, force=args.force, max_length=args.max_length)
