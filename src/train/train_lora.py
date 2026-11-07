#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用LoRA训练脚本：步骤1-3通用
用法: python train_lora.py --task step1_type --dataset type_train_5.json --epochs 8
"""

import os
import sys
import json
import argparse
import shutil

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HOME"] = "D:\\Trae CN\\Trae Project\\First AID Agent\\models\\cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\Trae CN\\Trae Project\\First AID Agent\\models\\cache"

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = os.path.join(BASE_DIR, "models", "cache")

TASK_CONFIG = {
    "step1_type": {
        "output_dir": "models/step1_type_lora",
        "data_dir": "data/step1_nlu/type",
        "prompt_prefix": "判断受伤类型",
        "rule": "规则：\n- 开放式：皮肤破损、有伤口、外出血、断肢、破皮的烧烫伤\n- 闭合式：无伤口的内部损伤、骨折、脱位、扭伤、内出血、挫伤\n- 其他：中毒、辐射、电击、溺水、中暑、冻伤(不破皮)、一度烧烫伤(无破皮)\n\n只输出一个类型词",
    },
    "step2_open": {
        "output_dir": "models/step2_open_subtype_lora",
        "data_dir": "data/step1_nlu/subtype",
        "prompt_prefix": "识别开放式受伤小类型",
        "rule": "",
    },
    "step2_closed": {
        "output_dir": "models/step2_closed_subtype_lora",
        "data_dir": "data/step1_nlu/subtype",
        "prompt_prefix": "识别闭合式受伤小类型",
        "rule": "",
    },
    "step3_location": {
        "output_dir": "models/step3_location_lora",
        "data_dir": "data/step1_nlu/location",
        "prompt_prefix": "识别受伤部位",
        "rule": "",
    },
}

def format_instruction(task, input_text, output_text):
    cfg = TASK_CONFIG[task]
    rule = cfg.get("rule", "")
    if rule:
        return f'<|im_start|>user\n{cfg["prompt_prefix"]}：{input_text}\n{rule}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>'
    return f'<|im_start|>user\n{cfg["prompt_prefix"]}：{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>'

def load_dataset(task, dataset_file):
    cfg = TASK_CONFIG[task]
    dataset_path = os.path.join(BASE_DIR, cfg["data_dir"], dataset_file)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 任务特定补充样本
    extra_samples = get_extra_samples(task)
    all_data = data + extra_samples
    
    formatted = [{"text": format_instruction(task, item["input"], item["output"])} for item in all_data]
    
    temp_path = os.path.join(os.path.dirname(dataset_path), "_temp_train.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False)
    
    dataset = Dataset.from_json(temp_path)
    
    try:
        os.remove(temp_path)
    except:
        pass
    
    print(f"  原始数据: {len(data)}条, 补充样本: {len(extra_samples)}条, 总计: {len(all_data)}条")
    return dataset, dataset_path

def get_extra_samples(task):
    """任务特定补充样本"""
    if task == "step1_type":
        return [
            # 断肢/扯断类 → 开放式（有伤口）
            {"input": "腿扯断了", "output": "开放式"},
            {"input": "胳膊被扯掉了", "output": "开放式"},
            {"input": "手指断成两半了", "output": "开放式"},
            {"input": "摔成两截了", "output": "开放式"},
            {"input": "身体被撕成两半", "output": "开放式"},
            {"input": "手被绞断了", "output": "开放式"},
            {"input": "脚被切断了", "output": "开放式"},
            {"input": "耳朵撕下来了", "output": "开放式"},
            {"input": "舌头咬断了", "output": "开放式"},
            {"input": "肠子流出来了", "output": "开放式"},
            # 明确骨折类 → 闭合式（无伤口）
            {"input": "腿骨折了没破皮", "output": "闭合式"},
            {"input": "胳膊断了皮没破", "output": "闭合式"},
            {"input": "肋骨断了皮肤完好", "output": "闭合式"},
            {"input": "脊椎骨折没出血", "output": "闭合式"},
            {"input": "鼻骨骨折皮肤没破", "output": "闭合式"},
        ]
    return []

def cleanup_checkpoints(output_dir):
    """删除所有checkpoint目录，只保留最终adapter"""
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            shutil.rmtree(item_path, ignore_errors=True)
            print(f"  已删除: {item}")

def train(task, dataset_file, epochs=8, batch_size=8, force=False, max_length=128):
    if task not in TASK_CONFIG:
        print(f"错误: 未知任务 {task}")
        print(f"可用任务: {list(TASK_CONFIG.keys())}")
        return
    
    cfg = TASK_CONFIG[task]
    output_dir = os.path.join(BASE_DIR, cfg["output_dir"])
    
    print(f"[训练] 任务: {task}")
    print(f"[训练] 模型: {MODEL_NAME}")
    print(f"[训练] 输出: {output_dir}")
    
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        if not force:
            print(f"\n⚠  已有训练结果存在: {output_dir}")
            print(f"   如果确定要重新训练覆盖它, 加 --force 参数")
            print(f"   否则, 先备份或换个输出目录名")
            return
        else:
            print(f"[训练] --force 已设置, 清空旧输出...")
            shutil.rmtree(output_dir, ignore_errors=True)
    elif os.path.exists(output_dir):
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
    model.print_trainable_parameters()
    
    dataset, dataset_path = load_dataset(task, dataset_file)
    print(f"[训练] 数据: {dataset_path} ({len(dataset)}条)")
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    tokenized = dataset.map(tokenize_fn, batched=True)
    
    steps_per_epoch = len(tokenized) // batch_size
    save_steps = max(steps_per_epoch, 100)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=save_steps,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        report_to="none",
        save_total_limit=1,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    print(f"[训练] 开始: {epochs} epochs, batch={batch_size}, max_length={max_length}, save每{save_steps}步")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("[训练] 清理checkpoint...")
    cleanup_checkpoints(output_dir)
    
    total_size = sum(
        os.path.getsize(os.path.join(dp, f)) 
        for dp, dn, fn in os.walk(output_dir) 
        for f in fn
    )
    print(f"[训练] 完成! 适配器大小: {total_size/1024/1024:.1f}MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA训练脚本")
    parser.add_argument("--task", type=str, required=True,
                       choices=list(TASK_CONFIG.keys()),
                       help="训练任务类型")
    parser.add_argument("--dataset", type=str, required=True,
                       help="训练数据JSON文件名")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--force", action="store_true",
                       help="强制覆盖已存在的训练结果")
    args = parser.parse_args()
    
    train(args.task, args.dataset, args.epochs, args.batch_size, args.force, args.max_length)
