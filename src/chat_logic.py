#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能外伤急救指导 AI - 逻辑引擎版
流程：自然语言理解 → 识别大类型 → 识别小类型 → 识别部位 → 匹配急救方案
"""

import os
import sys
import json

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "models", "cache")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "models--Qwen--Qwen2.5-1.5B-Instruct")
KB_DIR = os.path.join(BASE_DIR, "data", "step2_guide")

OPEN_KEYWORDS = ["流血", "割伤", "划伤", "刺伤", "咬伤", "烫伤", "烧伤", "撕裂", "破皮", "伤口", "流血不止", "出血"]
CLOSED_KEYWORDS = ["骨折", "扭伤", "崴脚", "脱臼", "挫伤", "撞伤", "淤青", "肿胀", "骨头断", "错位", "拉伤"]
BODY_PARTS = {"头部": ["头", "脑袋", "额头", "后脑勺", "太阳穴", "头皮"],
              "面部": ["脸", "眼睛", "鼻子", "嘴巴", "耳朵", "下巴", "脸颊"],
              "颈部": ["脖子", "颈"],
              "胸部": ["胸", "胸口", "心脏", "肋骨"],
              "腹部": ["肚子", "腹部", "胃", "腰腹"],
              "上臂": ["上臂", "大臂"],
              "前臂": ["前臂", "小臂"],
              "手腕": ["手腕"],
              "手掌": ["手掌", "掌心"],
              "手指": ["手指", "手指头"],
              "大腿": ["大腿"],
              "小腿": ["小腿"],
              "膝盖": ["膝盖", "髌骨"],
              "脚踝": ["脚踝"],
              "脚掌": ["脚掌", "脚心"],
              "脚趾": ["脚趾"],
              "背部": ["背", "后背", "脊柱"],
              "腰部": ["腰", "腰椎"],
              "臀部": ["屁股", "臀部"],
              "肩部": ["肩膀", "肩"],
              "肘部": ["手肘", "肘部"]}

def find_local_model_path():
    snapshots_dir = os.path.join(MODEL_CACHE_PATH, "snapshots")
    if os.path.exists(snapshots_dir):
        for entry in os.listdir(snapshots_dir):
            entry_path = os.path.join(snapshots_dir, entry)
            if os.path.isdir(entry_path) and not entry.startswith("."):
                config_path = os.path.join(entry_path, "config.json")
                if os.path.exists(config_path):
                    return entry_path
    return None

LOCAL_MODEL_PATH = find_local_model_path()

class InjuryClassifier:
    """伤情分类器 - 规则匹配快速版"""
    
    @staticmethod
    def classify_type(text):
        text_lower = text.lower()
        for kw in OPEN_KEYWORDS:
            if kw in text_lower:
                return "开放式"
        for kw in CLOSED_KEYWORDS:
            if kw in text_lower:
                return "闭合式"
        return None
    
    @staticmethod
    def classify_subtype(text, injury_type):
        text_lower = text.lower()
        if injury_type == "开放式":
            if "割伤" in text_lower or "划伤" in text_lower:
                return "割伤"
            if "刺伤" in text_lower:
                return "刺伤"
            if "咬伤" in text_lower:
                return "咬伤"
            if "烫伤" in text_lower or "烧伤" in text_lower:
                return "烧伤"
            if "流血" in text_lower or "出血" in text_lower:
                return "出血"
            return "其他开放式"
        else:
            if "骨折" in text_lower or "骨头断" in text_lower:
                return "骨折"
            if "扭伤" in text_lower or "崴" in text_lower:
                return "扭伤"
            if "脱臼" in text_lower or "错位" in text_lower:
                return "脱位"
            if "挫伤" in text_lower or "淤青" in text_lower:
                return "挫伤"
            if "拉伤" in text_lower:
                return "拉伤"
            return "其他闭合式"
    
    @staticmethod
    def classify_location(text):
        text_lower = text.lower()
        for part, keywords in BODY_PARTS.items():
            for kw in keywords:
                if kw in text_lower:
                    return part
        
        if "手臂" in text_lower or "胳膊" in text_lower:
            return "手臂"
        if "腿" in text_lower:
            return "腿部"
        if "脚" in text_lower:
            return "脚部"
        if "手" in text_lower:
            return "手部"
        
        return "未知"

class KnowledgeBase:
    """知识库检索"""
    
    @staticmethod
    def get_guide(injury_type, subtype, location):
        if injury_type == "开放式":
            dir_name = "open"
        else:
            dir_name = "closed"
        
        filename = subtype + ".json"
        filepath = os.path.join(KB_DIR, dir_name, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        
        fallback_file = os.path.join(KB_DIR, dir_name, "default.json")
        if os.path.exists(fallback_file):
            try:
                with open(fallback_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        
        return None

def load_model():
    if not LOCAL_MODEL_PATH:
        return None, None
    
    print("[INFO] 正在加载模型...")
    print(f"[INFO] 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=CACHE_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=CACHE_DIR,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            model = torch.compile(model, mode="max-autotune")
            print("[INFO] 已启用 torch.compile 加速")
        
        print("[INFO] ✅ 模型加载完成！")
    except Exception as e:
        print(f"[WARN] 模型加载失败: {e}")
        return tokenizer, None
    
    return tokenizer, model

def llm_analyze(tokenizer, model, text, history):
    """LLM分析（当规则匹配不确定时使用）"""
    if model is None:
        return None
    
    system_prompt = """请分析以下受伤描述，提取信息：
1. injury_type: 开放式或闭合式
2. subtype: 具体受伤类型（如割伤、骨折等）
3. location: 受伤部位（如头部、手臂等）

只返回JSON格式，不要其他内容。"""
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append(msg)
    messages.append({"role": "user", "content": text})
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    try:
        return json.loads(response)
    except:
        return None

def generate_response(classifier, kb, tokenizer, model, user_input, history):
    """生成回复（完整逻辑流程）"""
    
    injury_type = classifier.classify_type(user_input)
    if injury_type is None and model:
        llm_result = llm_analyze(tokenizer, model, user_input, history)
        if llm_result:
            injury_type = llm_result.get("injury_type")
    
    if injury_type is None:
        injury_type = "未知"
    
    subtype = classifier.classify_subtype(user_input, injury_type)
    location = classifier.classify_location(user_input)
    
    guide = kb.get_guide(injury_type, subtype, location)
    
    if guide:
        steps = guide.get("steps", [])
        notes = guide.get("notes", [])
        
        response = f"【识别结果】\n"
        response += f"- 受伤类型: {injury_type}\n"
        response += f"- 具体类型: {subtype}\n"
        response += f"- 受伤部位: {location}\n\n"
        response += "【急救指导】\n"
        for i, step in enumerate(steps, 1):
            response += f"{i}. {step}\n"
        if notes:
            response += "\n【注意事项】\n"
            for note in notes:
                response += f"- {note}\n"
    else:
        response = f"【识别结果】\n"
        response += f"- 受伤类型: {injury_type}\n"
        response += f"- 具体类型: {subtype}\n"
        response += f"- 受伤部位: {location}\n\n"
        response += "未找到对应的急救方案，请提供更多信息。"
    
    return response, {"type": injury_type, "subtype": subtype, "location": location}

def main():
    classifier = InjuryClassifier()
    kb = KnowledgeBase()
    tokenizer, model = load_model()
    
    history = []
    
    print("=" * 60)
    print("  智能外伤急救指导 AI - 逻辑引擎版")
    print("=" * 60)
    print("流程: 自然语言理解 → 识别大类型 → 识别小类型 → 识别部位 → 匹配急救方案")
    print("输入 'quit' 退出，输入 'history' 查看历史\n")
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("\n再见！")
                break
            
            if user_input.lower() == "history":
                if not history:
                    print("当前没有对话历史\n")
                    continue
                print("\n对话历史:")
                print("-" * 60)
                for i in range(0, len(history), 2):
                    if i + 1 < len(history):
                        print(f"你: {history[i]['content']}")
                        print(f"AI: {history[i+1]['content'][:100]}...")
                        print()
                print("-" * 60 + "\n")
                continue
            
            if not user_input:
                continue
            
            print("\nAI: ")
            response, intent = generate_response(classifier, kb, tokenizer, model, user_input, history)
            print(response)
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            if len(history) > 20:
                history = history[-20:]
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")

if __name__ == "__main__":
    main()