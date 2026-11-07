#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练步骤2：识别受伤小类型（出血、骨折、烧伤等）
输出格式: {"type_digital": "具体类型"}
"""

import json
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "data/step1_nlu/subtype/train.json"
OUTPUT_DIR = "models/step2_subtype_lora"

def format_instruction(input_text, output_text):
    return f'<|im_start|>user\n识别受伤小类型：{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>'

def load_and_prepare_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        formatted_data.append({"text": format_instruction(item["input"], item["output"])})
    
    with open("data/step1_nlu/subtype/formatted_train.json", "w", encoding="utf-8") as f:
        json.dump(formatted_data, f)
    
    dataset = load_dataset("json", data_files="data/step1_nlu/subtype/formatted_train.json")
    return dataset["train"]

def train(num_epochs=5, batch_size=8, max_length=128):
    print(f"[Training Step 2] Loading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
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
    
    dataset = load_and_prepare_dataset()
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_steps=100,
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
    
    print("[Training Step 2] Starting training...")
    print(f"[Training Step 2] max_length={max_length}, batch_size={batch_size}, grad_accum=1")
    trainer.train()
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"[Training Step 2] Training completed! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    
    train(num_epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length)