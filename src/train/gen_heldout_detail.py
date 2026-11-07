#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Step2b 细分类型留出集（当前仅 open/mechanical 11 类）

原则：
- 用纯 base 模型（不挂任何 LoRA）采样生成，风格与训练数据生产方式一致
- 标签泄漏过滤：input 不得出现 11 个类名
- 与训练集 + 270 条留出集做相似度去重（bigram Jaccard）
- 产物字段对齐 eval_heldout.py：input/type/should_ask/group/detail

用法：
  python gen_heldout_detail.py --per_class 15
"""

import os
import sys
import json
import re
import random
import argparse

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HOME"] = "D:\\Trae CN\\Trae Project\\First AID Agent\\models\\cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\Trae CN\\Trae Project\\First AID Agent\\models\\cache"

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = os.path.join(BASE_DIR, "models", "cache")

# 每类的判定要点（提炼自数据生产提示词，防止生成串类）
CLASS_RULES = {
    "切割伤": "锐器（刀、玻璃、金属薄片）划开皮肤，线形伤口、边缘整齐、出血明显。例：被菜刀切了、玻璃划了道口子、刀片割破",
    "刺伤": "尖锐物（钉子、针、刀尖、竹签）扎入，伤口小而深。例：踩到钉子扎进脚里、螺丝刀戳进手、针扎进肉里",
    "挫裂伤": "钝物撞击导致皮肤裂开，伤口边缘不规则、可伴瘀青。例：头撞桌角裂了道口子、锤子砸破皮、脸磕地上磕开口子",
    "撕裂伤": "外力拉扯导致皮肤组织撕开，伤口不规则，局部撕开（面积不大）。例：手被机器卷进去撕开了、被钩子里的铁丝撕开一道",
    "撕脱伤": "大面积皮肤被碾/绞脱套、撕掉一大片皮、创面大。例：车轮碾过去皮都脱了、机器绞掉一大片皮、皮带卷进去脱了一层皮",
    "开放性骨折": "必须同时有骨折信号（断了/折了/骨头）和皮肤破损信号（戳破/顶破/骨头外露/破皮流血）。例：腿骨折骨头戳出来了、手臂断了骨头露在外面",
    "爆炸碎片伤": "必须有爆炸场景（爆竹/烟花/炸弹/爆炸/爆裂）且碎片造成伤口。例：烟花碎片扎进胳膊、爆竹炸了满身小口子",
    "火器伤": "枪弹/弹丸造成，有入口或贯通。例：被枪打中胳膊、气枪子弹打进手里、弹片打穿大腿",
    "挤压破裂": "重物挤压/碾压导致组织破碎、皮肤裂开。例：脚被混凝土块砸扁皮都裂开、液压机压破手、卡车碾过腿碎了",
    "抓咬伤": "动物或人抓咬造成破皮，齿痕/抓痕/咬洞。例：狗咬破手、猫抓了脸几道血印、老鼠咬了一口见血",
    "擦伤": "粗糙面摩擦造成表皮浅层破损、渗血、创面浅。例：膝盖蹭破皮、胳膊在地上蹭了一片、砂纸磨破手",
}

CLASSES = list(CLASS_RULES.keys())

# 皮肤破损信号（生成句必须含其一，开放性前提）
BREAK_SIGNALS = [
    "破", "流血", "出血", "血", "伤口", "口子", "戳", "扎进", "扎穿", "扎了",
    "露", "撕开", "撕了", "撕掉", "脱了", "脱皮", "蹭", "磨破", "割", "划了道",
    "打中", "打穿", "打进", "炸", "咬", "抓了", "裂开", "碎了", "贯穿", "洞",
]

NOISE_PREFIX = re.compile(r"^(伤情|描述|患者说|患者|输入|文本)[：:]\s*")


def bigrams(s):
    s = re.sub(r"[\s，。！？、…~!?,.;：:\"'（）()卧槽我靠妈的妈的啊呀呢吧了的我了个去]", "", s)
    return set(s[i:i+2] for i in range(len(s) - 1))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_json_array(text):
    """从模型输出中容错抽取 JSON 数组"""
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        # 逐条捞 {"input": ..., "output": ...}
        out = []
        for mm in re.finditer(r"\{[^{}]*\}", text, re.S):
            try:
                out.append(json.loads(mm.group(0)))
            except Exception:
                pass
        return out


def build_prompt(label, n=10):
    rules = "\n".join(f"- {k}：{v}" for k, v in CLASS_RULES.items())
    return (
        "你在为一个急救伤情分类系统造测试句。请只生成「{lab}」这一类的口语化伤情描述 {n} 条，"
        "输出 JSON 数组，每条形如 {{\"input\": \"一句话\", \"output\": \"{lab}\"}}，不要输出别的。\n\n"
        "【{lab}判定标准】{rule}\n\n"
        "【11个类别全集，防止串类】\n{rules}\n\n"
        "【硬性要求】\n"
        "1. 必须是普通人（可能惊慌、带脏话感叹词）的报案/呼救口吻，5~36个字，第一或第三人称均可；\n"
        "2. 句中必须体现皮肤破损/流血/伤口等开放性信号，但不要出现\"{lab}\"或任何类别名；\n"
        "3. 覆盖手、臂、腿、脚、头脸等部位与家庭、工地、车祸、户外、厨房、工厂、农场等场景；\n"
        "4. 10 条彼此说法要明显不同，不要套路化句式。"
    ).format(lab=label, n=n, rule=CLASS_RULES[label], rules=rules)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=6, help="每类最多采样轮数")
    ap.add_argument("--out", default=os.path.join(
        BASE_DIR, "data", "heldout", "heldout_detail_mechanical.json"))
    args = ap.parse_args()

    random.seed(20260913)

    # ---- 去重底库：训练集 + 旧留出集 ----
    train_path = os.path.join(
        BASE_DIR, "data", "step1_nlu", "subtype", "detail", "open", "mechanical",
        "subtype_detail_open_merged_3500.json")
    held270_path = os.path.join(BASE_DIR, "data", "heldout", "heldout.json")
    base_texts = []
    for p in (train_path, held270_path):
        for x in json.load(open(p, encoding="utf-8")):
            t = x.get("input") or ""
            if t:
                base_texts.append(t)
    base_bi = [bigrams(t) for t in base_texts]
    print(f"去重底库：{len(base_bi)} 条")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR,
        local_files_only=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    def gen_one_round(label):
        msg = [{"role": "user", "content": build_prompt(label)}]
        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=900, do_sample=True,
                temperature=0.95, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_json_array(text)

    final = []
    stats = {}
    for label in CLASSES:
        kept, seen_bi, reject = [], set(), {"leak": 0, "nosignal": 0, "dup": 0, "label": 0, "len": 0}
        for r in range(args.rounds):
            if len(kept) >= args.per_class:
                break
            for item in gen_one_round(label):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("input", "")).strip()
                out = str(item.get("output", "")).strip()
                text = NOISE_PREFIX.sub("", text)
                if out != label:
                    reject["label"] += 1
                    continue
                if not (5 <= len(text) <= 40):
                    reject["len"] += 1
                    continue
                if any(c in text for c in CLASSES):
                    reject["leak"] += 1
                    continue
                if not any(sig in text for sig in BREAK_SIGNALS):
                    reject["nosignal"] += 1
                    continue
                bi = bigrams(text)
                if any(jaccard(bi, b) > 0.6 for b in base_bi) or \
                   any(jaccard(bi, b) > 0.7 for b in seen_bi):
                    reject["dup"] += 1
                    continue
                seen_bi.add(frozenset(bi))
                kept.append(text)
                if len(kept) >= args.per_class:
                    break
        stats[label] = (len(kept), reject)
        for t in kept:
            final.append({
                "input": t, "type": "开放式", "should_ask": False,
                "group": "机械性损伤", "detail": label,
            })
        print(f"{label}: 保留 {len(kept)}/{args.per_class}  剔除 {reject}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(final, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n共 {len(final)} 条 -> {args.out}")
    short = [k for k, (n, _) in stats.items() if n < args.per_class]
    if short:
        print(f"⚠ 数量不足的类：{short}，可加大 --rounds 重跑")


if __name__ == "__main__":
    main()
