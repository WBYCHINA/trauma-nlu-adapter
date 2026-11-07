#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练步骤1：识别受伤大类型（开放式/闭合式/其他）
支持多类型输出，用逗号分隔
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
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "step1_type_lora")
CACHE_DIR = os.path.join(BASE_DIR, "models", "cache")

# 不再添加多类型样本，模型只学单类型判断
# 多类型由推理脚本拆分输入实现
MULTI_TYPE_SAMPLES = []

def format_instruction(input_text, output_text):
    return f'<|im_start|>user\n识别受伤类型：{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>'

def load_and_prepare_dataset(dataset_paths):
    all_data = []
    for dataset_path in dataset_paths:
        with open(dataset_path, "r", encoding="utf-8") as f:
            all_data.extend(json.load(f))

    # 添加多类型样本
    all_data = all_data + MULTI_TYPE_SAMPLES

    formatted_data = []
    for item in all_data:
        formatted_data.append({"text": format_instruction(item["input"], item["output"])})

    dataset_dir = os.path.dirname(dataset_paths[0])
    temp_path = os.path.join(dataset_dir, "formatted_train.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False)

    dataset = load_dataset("json", data_files=temp_path)
    return dataset["train"]

def train(num_epochs=3, batch_size=8, dataset_file="type_train_1.json", force=False, max_length=128, resume=False, lr=None):
    # 支持逗号分隔的多个数据集文件（继续训练时旧数据+新数据混合，防止灾难性遗忘）
    dataset_paths = [os.path.join(BASE_DIR, "data", "step1_nlu", "type", f.strip())
                     for f in dataset_file.split(",") if f.strip()]
    print(f"[Training Step 1] Loading model: {MODEL_NAME}")
    print(f"[Training Step 1] Datasets: {[os.path.basename(p) for p in dataset_paths]}")
    print(f"[Training Step 1] Multi-type samples added: {len(MULTI_TYPE_SAMPLES)}")

    for p in dataset_paths:
        if not os.path.exists(p):
            print(f"⚠ 数据集不存在: {p}")
            return

    if resume:
        # 继续训练：在现有 LoRA 适配器基础上接着学，不删除已有模型
        if not os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
            print(f"⚠ --resume 需要已有模型，但未找到: {OUTPUT_DIR}")
            return
        print(f"[Training Step 1] Resume 模式：从现有适配器继续训练（保留已学能力）")
    elif os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
        if not force:
            print(f"\n⚠  已有训练结果存在: {OUTPUT_DIR}")
            print(f"   如确定覆盖，加 --force 参数；如需在原模型上继续训练，加 --resume 参数")
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
    
    if resume:
        # 加载现有适配器并设为可训练，LoRA 结构（r/模块）自动沿用原配置
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, OUTPUT_DIR, is_trainable=True)
    else:
        # 从零训练：恢复 Step1 原始容量配置（r=16, 7 个线性模块）
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"[Training Step 1] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Training Step 1] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    model.print_trainable_parameters()
    
    dataset = load_and_prepare_dataset(dataset_paths)
    print(f"[Training Step 1] Dataset loaded: {len(dataset)} samples")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 学习率：继续训练用低学习率，避免冲掉已学能力
    effective_lr = lr if lr is not None else (5e-5 if resume else 2e-4)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=5,
        save_steps=1000,
        learning_rate=effective_lr,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        report_to="none",
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    print(f"[Training Step 1] Starting training with {num_epochs} epochs...")
    print(f"[Training Step 1] max_length={max_length}, batch_size={batch_size}, grad_accum=1")
    trainer.train()
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"[Training Step 1] Training completed! LoRA adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="type_train_1.json",
                        help="数据集文件，多个用英文逗号分隔（继续训练时建议旧数据+新数据混合）")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有训练结果（从零训练）")
    parser.add_argument("--resume", action="store_true", help="在现有模型基础上继续训练（不覆盖旧能力）")
    parser.add_argument("--lr", type=float, default=None, help="学习率；缺省时继续训练 5e-5，从零训练 2e-4")
    args = parser.parse_args()

    train(num_epochs=args.epochs, batch_size=args.batch_size, dataset_file=args.dataset,
          force=args.force, max_length=args.max_length, resume=args.resume, lr=args.lr)
