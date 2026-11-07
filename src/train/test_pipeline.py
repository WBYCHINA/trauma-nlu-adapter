#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1 + Step2a 串联推理
- base model (LoRA 轻载) 做语义解析：去噪 + 拆分多伤情
- step1 LoRA 做类型分类：logprob margin 判断不确定性 → 驱动追问
- step2 LoRA 做组别分类
"""

import os, sys, json, argparse, re

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
MODEL_PATH = os.path.join(BASE_DIR, "models", "cache",
    "models--Qwen--Qwen2.5-1.5B-Instruct",
    "snapshots", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
STEP1_LORA = os.path.join(BASE_DIR, "models", "step1_type_lora")
STEP2_LORA = os.path.join(BASE_DIR, "models", "step2_open_group_lora")

ALLOWED_TYPES = ["开放式", "闭合式"]
GROUPS = ["机械性损伤", "热力损伤", "电能损伤", "化学腐蚀",
          "辐射损伤", "生物毒素-感染", "病理自发性破溃"]

STEP1_RULE = """核心原则：以皮肤/粘膜是否完整为准
- 开放式：皮肤或粘膜有破损、有伤口、有外出血、有溃烂、有焦痂、有水泡破溃
- 闭合式：皮肤和粘膜完整无损（包括骨折、脱臼、扭伤、内出血、中毒、中暑、溺水、电击、不破皮的烧烫伤、冻伤等）

只输出一个类型词"""

TYPE_AMBIGUITY_THRESHOLD = 0.5

CLEAN_PROMPT = "删除脏话、感叹词、求助语、时间地点，只保留伤病内容，用逗号分隔，只输出清理结果：{text}"

INJURY_KEYWORDS = [
    '骨折', '骨头断', '骨头裂', '断了', '折了', '裂了',
    '脱臼', '脱位',
    '扭伤', '拉伤', '挫伤', '砸伤', '撞伤', '砍伤', '打伤',
    '割伤', '刺伤', '切伤', '划伤', '扎伤', '戳伤', '砍',
    '撕裂', '撕脱', '挤压', '碾压',
    '烧伤', '烫伤', '冻伤', '晒伤',
    '电击', '触电', '电伤',
    '腐蚀', '中毒',
    '辐射', '放射', '爆炸', '炸伤',
    '咬伤', '抓伤', '蛰伤',
    '出血', '流血', '破皮', '破了', '伤口',
    '内出血', '淤血', '脑震荡',
    '中暑', '溺水', '窒息',
    '糖尿病足', '溃疡', '压疮', '坏疽',
    '动脉', '静脉', '血管破裂', '劈裂',
    '摔', '坠', '高空',
]


def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 两个独立实例，互不污染
    nlu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        local_files_only=True, dtype=dtype
    ).to(device)

    lora_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        local_files_only=True, dtype=dtype
    ).to(device)
    lora_model = PeftModel.from_pretrained(lora_model, STEP1_LORA, adapter_name="step1")
    lora_model.load_adapter(STEP2_LORA, adapter_name="step2")

    print(f"✓ NLU 模型（纯 base，无 adapter）")
    print(f"✓ Step1 adapter → {STEP1_LORA}")
    print(f"✓ Step2 adapter → {STEP2_LORA}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    return nlu_model, lora_model, tokenizer


def _generate(model, tokenizer, prompt, max_new_tokens=16):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# ============ Part 1: NLU 清理 + 拆分 ============

def _extract_keywords(text):
    return [kw for kw in INJURY_KEYWORDS if kw in text]


def _nlu_parse(nlu_model, tokenizer, text):
    """让 base model 清理后输出，支持逗号或换行分隔"""
    prompt = f'<|im_start|>user\n{CLEAN_PROMPT.format(text=text)}<|im_end|>\n<|im_start|>assistant\n'
    raw = _generate(nlu_model, tokenizer, prompt, max_new_tokens=96)

    print(f"  [NLU 原始输出: {raw.strip()[:120]}]")

    if not raw.strip():
        return None

    # 先按换行拆，每行再按逗号拆
    items = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('```'):
            continue
        # 去掉 "1." "1、" 等编号前缀
        line = re.sub(r'^[\d一二三四五六七八九十]+[\.\、\)\]]\s*', '', line)
        # 行内按逗号分
        for part in re.split(r'[，,]', line):
            part = part.strip().strip('。.')
            if part and len(part) >= 2:
                items.append(part)

    # 去重（保持顺序）
    seen = set()
    unique = []
    for i in items:
        if i not in seen:
            seen.add(i)
            unique.append(i)

    print(f"  [NLU 解析: {unique}]")
    return unique if unique else None


def _fallback_split(text):
    """NLU 失败时：连词/标点 + 伤情关键词边界"""
    # Step 1: 按连词和标点拆
    parts = re.split(r'[，,；;、和与及或跟以及同时此外另外还有伴有伴随而且并且然后接着还又也再加]', text)
    parts = [p.strip().strip('。').strip() for p in parts]
    parts = [p for p in parts if p and len(p) >= 2]

    # Step 2: 如果只出一段，尝试用伤情关键词做边界
    if len(parts) == 1:
        p = parts[0]
        # 找到所有伤情关键词在文本中的位置
        kw_positions = []
        for kw in INJURY_KEYWORDS:
            start = 0
            while True:
                idx = p.find(kw, start)
                if idx == -1:
                    break
                kw_positions.append((idx, idx + len(kw), kw))
                start = idx + len(kw)
        # 按起始位置排序
        kw_positions.sort(key=lambda x: x[0])

        # 如果有多个关键词，在相邻关键词之间切分
        if len(kw_positions) >= 2:
            split_points = [0]
            for i in range(1, len(kw_positions)):
                # 在上一个关键词结束和当前关键词开始之间找一个切分点
                prev_end = kw_positions[i - 1][1]
                curr_start = kw_positions[i][0]
                if curr_start > prev_end:
                    # 中间有内容，在最后一个助词/语气词之后切
                    middle = p[prev_end:curr_start]
                    # 找到助词的位置
                    helper_match = re.search(r'[了的着过啊呀]', middle)
                    if helper_match:
                        split_points.append(prev_end + helper_match.end())
                    else:
                        split_points.append(prev_end)
            split_points.append(len(p))
            parts = [p[split_points[i]:split_points[i + 1]].strip()
                     for i in range(len(split_points) - 1)]
            parts = [p for p in parts if p and len(p) >= 2]

    if not parts:
        return [text]
    return parts


def parse_injuries(nlu_model, tokenizer, text):
    items = _nlu_parse(nlu_model, tokenizer, text)

    # 安全网: base model 可能把伤情也删了（如"我去你大爷砍死我了"→"我大爷"）
    # 如果清理后没有伤情关键词，但原文有，说明删过了，回退原文
    if items and not any(_extract_keywords(i) for i in items):
        if _extract_keywords(text):
            print(f"  [NLU 删丢了伤情，回退原文]")
            items = None

    if not items:
        print(f"  [NLU 解析失败，用原文]")
        items = [text]

    print(f"  [原始: {text}]")
    print(f"  [伤情: {items}]")
    return [{"injury": s} for s in items]


# ============ Part 2: 类型分类 + 不确定性判断 ============

def _type_forward_loss(model, tokenizer, text, target_type):
    """
    跑一次 forward，计算 prompt + target_type 的 loss
    loss 越低 = 模型越倾向于这个类型
    """
    prompt = f'<|im_start|>user\n判断受伤类型：{text}\n{STEP1_RULE}<|im_end|>\n<|im_start|>assistant\n'
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target_type, return_tensors="pt").input_ids.to(model.device)

    full_ids = torch.cat([prompt_ids, target_ids], dim=1)
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(full_ids, labels=labels)
    return outputs.loss.item()


def predict_type_with_confidence(model, tokenizer, text):
    """
    对"开放式"和"闭合式"分别算 loss，取差值
    返回 (predicted_type, margin, is_ambiguous)
      margin 小 → 模型犹豫 → is_ambiguous=True → 需要追问
    """
    model.set_adapter("step1")

    losses = {}
    for t in ALLOWED_TYPES:
        losses[t] = _type_forward_loss(model, tokenizer, text, t)

    best = min(ALLOWED_TYPES, key=lambda t: losses[t])
    margin = abs(losses["开放式"] - losses["闭合式"])
    is_ambiguous = margin < TYPE_AMBIGUITY_THRESHOLD

    return best, margin, is_ambiguous, losses


def predict_type(model, tokenizer, text):
    t, _, _, _ = predict_type_with_confidence(model, tokenizer, text)
    return t


# ============ Part 3: 追问 + 补全 ============

def _append_skin_status(injury, answer):
    a = answer.strip().lower()
    if a in ("有", "是", "对", "破了", "流血", "流血了", "出血", "烂了",
             "骨头戳", "骨头戳出来", "骨头露", "骨头露出来"):
        return injury + " 皮肤粘膜有破损"
    if a in ("没有", "无", "没", "没破", "没破皮", "完好", "皮肤完整", "没流血", "没出血"):
        return injury + " 皮肤粘膜完整无损"
    return None


def _ask_skin_question(injury):
    if "骨折" in injury or "断" in injury:
        return f"[{injury}] 皮肤有没有破？骨头有没有戳出来？（有/没有）"
    if any(k in injury for k in ["烧", "烫"]):
        return f"[{injury}] 皮肤有没有破损、水泡或发黑？（有/没有）"
    if any(k in injury for k in ["冻伤", "冻"]):
        return f"[{injury}] 皮肤有没有破皮、溃烂或发黑？（有/没有）"
    if any(k in injury for k in ["化学", "腐蚀", "酸碱"]):
        return f"[{injury}] 皮肤有没有溃烂、水泡或发黑？（有=开放式/没有=闭合式）"
    if any(k in injury for k in ["电击", "电伤", "触电"]):
        return f"[{injury}] 皮肤有没有伤口或焦痂？（有/没有）"
    return f"[{injury}] 皮肤有没有破损？（有=开放式/没有=闭合式）"


def _ask_skin_once(injury, q):
    for _ in range(5):
        ans = input(f"  {q} ").strip()
        injury_new = _append_skin_status(injury, ans)
        if injury_new is not None:
            return injury_new, True
        if ans == "跳过":
            break
        print(f"  ⚠ 回答无效，请输入 '有' 或 '没有'。跳过追问可输入 '跳过'")
    return injury, False


# ============ Part 4: 组别分类 ============

def predict_group(model, tokenizer, text):
    model.set_adapter("step2")
    prompt = f'<|im_start|>user\n判断受伤组别（机械/热力/电能/化学/辐射/生物/病理）：{text}<|im_end|>\n<|im_start|>assistant\n'
    raw = _generate(model, tokenizer, prompt, max_new_tokens=8)
    first_line = raw.split('\n')[0].strip()
    for g in GROUPS:
        if g in first_line:
            return g
    for g in GROUPS:
        if g in raw:
            return g
    return "未知"


# ============ Part 5: 主流水线 ============

def predict_pipeline(nlu_model, lora_model, tokenizer, raw_text):
    parsed = parse_injuries(nlu_model, tokenizer, raw_text)
    injuries = []

    for i, item in enumerate(parsed):
        injury = item["injury"]

        pred_type, margin, ambiguous, losses = predict_type_with_confidence(
            lora_model, tokenizer, injury)

        # 追问触发: 模型不确定 OR 高风险伤情（皮肤状态直接决定类型）
        MUST_ASK_KEYWORDS = ['骨折', '骨头断', '骨头裂', '脱臼', '扭伤',
                             '烧', '烫', '冻伤', '冻',
                             '化学', '腐蚀', '酸碱',
                             '电击', '电伤', '触电']
        must_ask = any(k in injury for k in MUST_ASK_KEYWORDS)

        if ambiguous or must_ask:
            q = _ask_skin_question(injury)
            injury, _ = _ask_skin_once(injury, q)
            pred_type, margin, _, losses = predict_type_with_confidence(
                lora_model, tokenizer, injury)

        if pred_type == "开放式":
            g = predict_group(lora_model, tokenizer, injury)
        elif pred_type == "闭合式":
            g = "(跳过-闭合式)"
        else:
            g = "未知"

        injuries.append({
            "idx": i + 1,
            "injury": item["injury"],
            "resolved": injury,
            "type": pred_type,
            "group": g,
            "margin": round(margin, 3),
            "losses": {t: round(v, 3) for t, v in losses.items()},
        })

    return {"original": raw_text, "injuries": injuries}


def interactive(nlu_model, lora_model, tokenizer):
    print("\n" + "=" * 60)
    print("  急救 NLU 流水线 - 交互式测试")
    print("  输入 q 退出")
    print("=" * 60)
    while True:
        try:
            raw_text = input("\n> ").strip()
            if raw_text.lower() == 'q':
                break
            if not raw_text:
                continue

            result = predict_pipeline(nlu_model, lora_model, tokenizer, raw_text)

            print(f"  原始: {result['original']}")
            for inv in result["injuries"]:
                if inv["resolved"] != inv["injury"]:
                    print(f"    伤情{inv['idx']}: {inv['injury']}  →  {inv['resolved']}")
                else:
                    print(f"    伤情{inv['idx']}: {inv['injury']}")
                print(f"           type={inv['type']}  group={inv['group']}  "
                      f"margin={inv['margin']}  losses={inv['losses']}")
            print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"错误: {e}")


def main():
    nlu_model, lora_model, tokenizer = load_pipeline()
    interactive(nlu_model, lora_model, tokenizer)


if __name__ == "__main__":
    main()
