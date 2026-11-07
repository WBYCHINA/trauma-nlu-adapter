#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试步骤1 LoRA模型：识别受伤大类型
模型只输出单类型，脚本负责拆分多伤情描述并合并结果
"""

import os
import sys
import re
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
LORA_PATH = os.path.join(BASE_DIR, "models", "step1_type_lora")

ALLOWED_TYPES = ["开放式", "闭合式"]

RULE = """核心原则：以皮肤/粘膜是否完整为准
- 开放式：皮肤或粘膜有破损、有伤口、有外出血、有溃烂、有焦痂、有水泡破溃
- 闭合式：皮肤和粘膜完整无损（包括骨折、脱臼、扭伤、内出血、中毒、中暑、溺水、电击、不破皮的烧烫伤、冻伤等）

只输出一个类型词"""

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

def extract_single_type(text):
    first_line = text.strip().split('\n')[0].strip()
    for t in ALLOWED_TYPES:
        if t in first_line:
            return t
    return None

def predict_single(model, tokenizer, text):
    prompt = f'<|im_start|>user\n判断受伤类型：{text}\n{RULE}<|im_end|>\n<|im_start|>assistant\n'

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
    generated_ids = outputs[0][input_len:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return extract_single_type(result.strip())

NEGATE_OPEN = ['没破皮', '没破', '没流血', '没出血', '内出血', '皮下出血', '淤血', '内出血']

OPEN_KEYWORDS = [
    '流血', '出血', '渗血', '鲜血', '外出血', '破了', '破皮', '破口', '裂开',
    '割', '划', '刺', '扎', '戳', '切', '撕', '扯', '绞',
    '伤口', '口子', '皮肉', '皮开', '断肢', '截', '离断', '撕脱',
    '戳穿', '刺穿', '刺破', '咬',
]

CLOSED_KEYWORDS = [
    '骨折', '骨头断', '骨头裂', '骨头戳', '断了', '折了', '裂了',
    '脱臼', '脱位', '掉出来', '脱出来',
    '扭伤', '崴', '闪了', '拉伤', '拧了',
    '砸', '撞', '摔', '跌', '磕', '碰', '挫',
    '肿', '青', '淤', '包', '鼓', '内出血', '皮下出血', '淤血',
    '脑震荡', '动不了', '伸不直', '弯不了', '抬不起', '站不起', '眩晕', '头晕',
]

NON_MECHANICAL_KEYWORDS = [
    '中毒', '吃药', '服药', '吞', '煤气', '一氧化碳',
    '中暑', '热射', '热痉挛',
    '溺水', '淹', '窒息', '喘不上', '喘不过气',
    '电击', '电了', '电着', '电伤',
    '冻伤', '冻', '失温',
    '辐射', '放射',
    '起疹子', '过敏',
    '心梗', '中风', '脑梗', '胸口疼', '胸痛',
    '烫得发红', '烫红', '没起泡', '发红没起泡',
    '化学', '腐蚀', '酸碱',
]

def detect_types(text):
    found = {"open": [], "closed": []}

    for kw in CLOSED_KEYWORDS:
        if kw in text:
            found["closed"].append(kw)

    has_negate_open = any(neg in text for neg in NEGATE_OPEN)
    for kw in OPEN_KEYWORDS:
        if kw in text:
            if has_negate_open and kw in ('出血', '流血', '破了', '破皮'):
                continue
            found["open"].append(kw)

    return found

def has_non_mechanical_injury(text):
    """检测是否有非机械损伤关键词（需结合皮肤状态判断类型）"""
    return any(kw in text for kw in NON_MECHANICAL_KEYWORDS)

def extract_phrase_without(text, keywords_to_remove):
    result = text
    for kw in keywords_to_remove:
        result = result.replace(kw, ' ')
    result = re.sub(r'[，,、和加伴有伴还有同时此外另外然后接着并且还又及或跟]', ' ', result)
    result = re.sub(r'^(被|把|让|给|将)', '', result).strip()
    result = re.sub(r'\s+', ' ', result).strip()

    if len(result) <= 2 and len(text) >= 3:
        if text.startswith('被'):
            result = '被' + result + '了'
        else:
            result = result + '了'

    return result

def split_input(text):
    conjunctions = r'[，,。；;、和加伴有伴还有同时此外另外然后接着]'
    parts = re.split(conjunctions, text)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 1]

    if not parts:
        return [text]

    final = []
    for p in parts:
        found = detect_types(p)
        has_open = len(found["open"]) > 0
        has_closed = len(found["closed"]) > 0

        active_types = []
        if has_open: active_types.append("open")
        if has_closed: active_types.append("closed")

        if len(active_types) <= 1:
            final.append(p)
        else:
            sub_phrases = []

            if "closed" in active_types and "open" in active_types:
                closed_only = extract_phrase_without(p, found["open"])
                if closed_only and closed_only != p and len(closed_only) >= 1:
                    sub_phrases.append(closed_only)

            if not sub_phrases:
                final.append(p)
            else:
                final.extend(sub_phrases)
                final.append(p)

    return final if final else [text]

AMBIGUITY_PATTERNS = [
    {
        "triggers": ["骨折", "骨头断", "骨头裂", "骨头折", "骨头摔", "骨头戳", "断了", "折了", "裂了"],
        "skin_positive": ["破皮", "破了", "流血", "出血", "渗血", "外出血", "伤口", "皮肉", "裂开", "戳穿", "刺穿", "刺破", "撕脱"],
        "skin_negative": ["没破皮", "没破", "没流血", "没出血", "内出血", "皮下出血", "淤血"],
        "question": "请问皮肤有没有被骨头戳破？（有/没有）",
        "followup_question": "破皮是骨头戳出来的还是擦伤蹭破的？（骨头戳的/擦伤蹭的）",
        "bone_pierce_keywords": ["骨头戳", "骨头露", "骨头顶", "骨头穿", "刺出", "戳出", "露骨头", "骨头刺", "骨头穿出"],
    },
    {
        "triggers": ["断肢", "扯断", "切断", "砍断", "绞断", "撕断", "离断"],
        "skin_positive": [],
        "skin_negative": ["没破皮", "没破"],
        "question": "断肢处是否有伤口或出血？（通常断肢是开放式）",
    },
    {
        "triggers": ["烧烫伤", "烧伤", "烫伤", "烫了", "烧了"],
        "skin_positive": ["破皮", "破了", "水泡", "起水泡", "脱皮", "溃烂", "焦痂", "发黑"],
        "skin_negative": ["没破皮", "没破", "发红", "没起泡", "烫红"],
        "question": "烧烫伤处皮肤是否有破损、水泡或发黑？（有/没有）",
    },
    {
        "triggers": ["化学", "腐蚀", "酸碱", "酸溅", "碱溅"],
        "skin_positive": ["溃烂", "腐蚀", "破皮", "水泡", "发黑", "焦痂", "脱皮"],
        "skin_negative": ["没破皮", "没破", "发红", "没起泡"],
        "question": "化学品接触处皮肤是否有溃烂、水泡或发黑？（有=开放式/没有=闭合式）",
    },
    {
        "triggers": ["冻伤", "冻", "失温"],
        "skin_positive": ["破皮", "溃烂", "发黑", "焦痂", "破溃"],
        "skin_negative": ["没破皮", "没破", "发红", "苍白", "发紫", "僵硬"],
        "question": "冻伤部位皮肤是否有破皮、溃烂或发黑？（有/没有）",
    },
    {
        "triggers": ["电击", "电了", "电着", "电伤", "触电"],
        "skin_positive": ["破皮", "伤口", "焦痂", "发黑", "溃烂", "出血"],
        "skin_negative": ["没破皮", "没破", "没伤口", "完好"],
        "question": "电击处皮肤是否有伤口或焦痂？（有/没有）",
    },
]

ALL_SKIN_POSITIVE = [
    "破皮", "破了", "破口", "流血", "出血", "渗血", "外出血", "鲜血",
    "伤口", "口子", "皮肉", "皮开", "裂开", "开裂", "溃烂", "破溃",
    "水泡破", "水泡破了", "脱皮", "脱套", "焦痂", "发黑", "焦黑",
    "戳穿", "刺穿", "刺破", "撕脱", "撕断", "扯断", "切断", "砍断",
    "咬", "断肢", "截", "离断", "腐蚀", "皮肤粘膜有破损",
    "骨头戳破", "骨头戳出来", "骨头露出来", "骨头顶破", "骨头穿破",
    "露骨头", "骨头穿出", "骨头刺破",
]

ALL_SKIN_NEGATIVE = [
    "没破皮", "没破", "没流血", "没出血", "没伤口", "没起泡",
    "内出血", "皮下出血", "淤血", "瘀青",
    "皮肤完整", "完好无损", "完好", "皮肤没破", "皮肤没伤",
    "皮肤发红", "烫红", "发红没起泡", "发红",
    "苍白", "发紫", "僵硬",
    "皮肤粘膜完整无损",
]

def has_skin_info(text):
    """是否提及了皮肤/粘膜状态（无论破损还是完好）"""
    return any(p in text for p in ALL_SKIN_POSITIVE) or any(n in text for n in ALL_SKIN_NEGATIVE)

def check_ambiguity(text):
    """返回需要补充的问题列表，空列表表示信息充足"""
    questions = []
    specific_triggered = False
    for pat in AMBIGUITY_PATTERNS:
        has_trigger = any(t in text for t in pat["triggers"])
        if not has_trigger:
            continue
        specific_triggered = True
        has_skin_pos = any(p in text for p in pat["skin_positive"])
        has_skin_neg = any(n in text for n in pat["skin_negative"])
        if not has_skin_pos and not has_skin_neg:
            questions.append({"q": pat["question"], "pat": pat})
        elif has_skin_pos and "followup_question" in pat:
            has_bone_pierce = any(k in text for k in pat.get("bone_pierce_keywords", []))
            if not has_bone_pierce:
                questions.append({"q": pat["followup_question"], "pat": pat, "is_followup": True})

    if not specific_triggered and not has_skin_info(text):
        questions.append({
            "q": "请问受伤部位皮肤或粘膜有没有破损？（有=开放式/没有=闭合式）",
            "pat": None,
            "is_generic": True,
        })

    return questions

def resolve_ambiguity(text, answers):
    """根据用户对歧义问题的回答，补全原文本的皮肤状态"""
    resolved = text
    for item in answers:
        q = item["q"]
        ans = item["ans"]
        pat = item["pat"]
        is_followup = item.get("is_followup", False)
        is_generic = item.get("is_generic", False)

        if is_generic:
            if ans in ("有", "是", "对", "有破损", "有破皮", "破了", "流血", "流血了", "出血", "烂了", "烂"):
                resolved += " 皮肤粘膜有破损"
            elif ans in ("没有", "无", "没", "没破", "没破皮", "完好", "皮肤完整", "没有破损", "没流血", "没出血"):
                resolved += " 皮肤粘膜完整无损"
            continue

        if is_followup and "followup_question" in pat:
            if ans in ("骨头戳的", "骨头戳出来的", "骨头顶的", "骨头穿的", "是", "骨头戳"):
                resolved += " 骨头戳破了皮肤"
            elif ans in ("擦伤蹭的", "擦伤", "蹭的", "蹭破的", "表皮擦伤"):
                resolved += " 皮肤擦伤蹭破，骨头没戳出来"
            continue

        skin_break_q = any(kw in q for kw in ["破皮", "流血", "破损", "溃烂", "焦痂", "发黑", "腐蚀"])
        if skin_break_q:
            if ans in ("有", "是", "对", "有破皮", "有流血", "破了", "流血了", "有破损", "有溃烂", "有焦痂", "有发黑", "有水泡"):
                resolved += " 皮肤粘膜有破损"
            elif ans in ("没有", "无", "没", "没破皮", "没破", "没流血", "完好", "皮肤完整"):
                resolved += " 皮肤粘膜完整无损"
        elif "断肢" in q:
            if ans in ("没有", "无", "没"):
                resolved += " 皮肤没破"
    return resolved

def post_process_type(text, model_output):
    """
    兜底修正：
    1. 模型输出了ALLOWED_TYPES外的词（如旧训练的「其他」）→ 按皮肤状态规则映射
    2. 非机械损伤场景下，按皮肤/粘膜完整性决定类型
    """
    valid = ["开放式", "闭合式"]
    if model_output in valid:
        return model_output

    has_skin_break = any(k in text for k in [
        "皮肤破", "破皮", "破了", "流血", "出血", "溃烂", "焦痂", "发黑",
        "水泡破", "脱皮", "伤口", "断肢", "骨头戳", "露骨头", "皮肤粘膜有破损"
    ])
    has_skin_intact = any(k in text for k in [
        "皮肤完整", "完好无损", "没破皮", "没破", "没流血", "没出血",
        "皮肤没破", "皮肤没伤", "皮肤粘膜完整"
    ])

    if has_skin_break:
        return "开放式"
    if has_skin_intact:
        return "闭合式"

    if has_non_mechanical_injury(text):
        return "闭合式"

    return "未知"

BONE_PIERCE_CONFIRMED = ["骨头戳破", "骨头戳出来", "骨头露出来", "骨头顶破", "骨头穿破", "露骨头", "骨头穿出", "骨头刺破"]
OPEN_FROM_BONE = ["骨头戳", "骨头露", "骨头顶", "骨头穿", "刺出", "戳出", "露骨头", "骨头刺", "骨头穿出"]

def resolve_mutual_exclusion(text, results):
    """
    1. 骨头戳破型骨折：只保留开放式
    2. 任意骨折 + 确认皮肤破损（皮肤粘膜有破损）：视为开放式，去掉闭合式
    """
    has_bone_pierce = any(k in text for k in BONE_PIERCE_CONFIRMED + OPEN_FROM_BONE)
    has_fracture = any(k in text for k in ["骨折", "骨头断", "骨头裂", "骨头折", "断了", "折了", "裂了"])
    has_skin_break = any(k in text for k in ["皮肤粘膜有破损", "破皮", "破了", "流血", "出血", "溃烂", "焦痂", "发黑"])

    if has_bone_pierce and has_fracture:
        return [r for r in results if r != "闭合式"]
    if has_fracture and has_skin_break:
        return [r for r in results if r != "闭合式"]
    return results

def predict(model, tokenizer, text):
    sub_inputs = split_input(text)
    results = []

    for sub in sub_inputs:
        t = predict_single(model, tokenizer, sub)
        t = post_process_type(sub, t)
        if t and t not in results:
            results.append(t)

    results = resolve_mutual_exclusion(text, results)
    return ",".join(results) if results else "未知"

def normalize(s):
    return set(t.strip() for t in s.replace("，", ",").split(","))

def batch_test(model, tokenizer, test_file):
    test_path = os.path.join(BASE_DIR, "data", "step1_nlu", "type", test_file)
    with open(test_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"加载 {len(cases)} 条测试用例...\n")

    correct = 0
    total = len(cases)
    errors = []
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    type_errors = defaultdict(list)

    for i, case in enumerate(cases):
        cat = case.get("category", "未分类")
        text = case["input"]
        expected = case["expected"]
        result = predict(model, tokenizer, text)

        is_correct = normalize(result) == normalize(expected)
        cat_stats[cat]["total"] += 1
        if is_correct:
            correct += 1
            cat_stats[cat]["correct"] += 1
        else:
            errors.append((i + 1, cat, text, expected, result))
            type_errors[expected].append((text, result))

        status = "✓" if is_correct else "✗"
        print(f"[{i+1:3d}/{total}] {status} [{cat}] \"{text}\" → {result} (期望: {expected})")

    acc = correct / total * 100
    print(f"\n{'='*60}")
    print(f"总准确率: {correct}/{total} = {acc:.1f}%")
    print(f"{'='*60}")

    print("\n📊 按类别统计:")
    for cat, s in cat_stats.items():
        a = s["correct"] / s["total"] * 100
        print(f"  {cat:12s}: {s['correct']:2d}/{s['total']:2d} = {a:.0f}%")

    if errors:
        print(f"\n❌ 错误详情 ({len(errors)} 条):")
        for idx, cat, text, expected, result in errors:
            print(f"  #{idx} [{cat}] \"{text}\"")
            print(f"     期望: {expected}  实际: {result}")

    print(f"\n💡 按期望类型分组的错误:")
    for expected, errs in type_errors.items():
        print(f"  期望「{expected}」错了 {len(errs)} 次:")
        for text, result in errs:
            print(f"    \"{text}\" → {result}")

    return acc

def interactive(model, tokenizer):
    print("模型加载完成，开始识别（输入 q 退出）")
    while True:
        try:
            text = input("> ").strip()
            if text.lower() == 'q':
                break
            if not text:
                continue

            questions = check_ambiguity(text)
            if questions:
                print()
                answers = []
                for item in questions:
                    ans = input(f"  {item['q']} ").strip()
                    item["ans"] = ans
                    answers.append(item)
                text = resolve_ambiguity(text, answers)
                print(f"  [补全后] {text}")

            result = predict(model, tokenizer, text)
            print(f"  → {result}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")

def main():
    model, tokenizer = load_model()
    interactive(model, tokenizer)

if __name__ == "__main__":
    main()
