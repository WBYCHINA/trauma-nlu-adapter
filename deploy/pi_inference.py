#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
急救伤情理解 —— Orange Pi 5 Pro GGUF CPU 推理版

与 GPU 版 src/train/pipeline.py 同一套逻辑（论文 4.3 三级分类 + 双通道追问），
仅把模型后端从 transformers + LoRA 换成 llama.cpp 的 Q4_K_M GGUF 模型。

流水线（与 pipeline.py 逐级对齐）：
  报案原文
    → NLU 清洗（纯 base GGUF，只删噪声，防幻觉校验）→ 多伤情拆分
    → 逐条追问补全（方案 A：原文未交代皮肤状态就问）
    → Step1 类型（开放式/闭合式，N=2）+ 医学互斥修正
    → Step2a 组别（受伤原因七选一，N=7，开放/闭合皆判）
    → Step2b 细分（机械组十一选一，N=11，仅开放式+机械性损伤）
    → Step3 部位（八选一，N=8）
    → 输出结构化 JSON

模型加载：5 个 GGUF 按需顺序加载（同一时刻只驻留一个，节省 8GB 内存）。
  base_clean       : 纯底座，仅做 NLU 清洗拆分
  step1_type       : Step1 类型
  step2_group      : Step2a 组别（受伤原因）
  step2_detail     : Step2b 细分（开放机械）
  step3_location   : Step3 部位

用法：
  交互式：python3 pi_inference.py
  单句  ：python3 pi_inference.py --text "卧槽骨折了烫伤"
  程序化：from pi_inference import build_classifier; classify(text, ask_fn)

依赖（Orange Pi 上）：
  pip install numpy llama-cpp-python
"""

import os
import sys
import json
import re
import math
import argparse

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ============ 模型文件（与 models/gguf/ 下的 Q4_K_M 产物对应）============
_MODEL_FILES = {
    "clean":   "base_clean-q4_k_m.gguf",
    "step1":   "step1_type-q4_k_m.gguf",
    "step2a":  "step2_group-q4_k_m.gguf",
    "step2b":  "step2_detail-q4_k_m.gguf",
    "step3":   "step3_location-q4_k_m.gguf",
}

# ============ 分类标签（与 pipeline.py / step3_train.py 完全一致）============
ALLOWED_TYPES = ["开放式", "闭合式"]
GROUPS = ["机械性损伤", "热力损伤", "电能损伤", "化学腐蚀",
          "辐射损伤", "生物毒素-感染", "病理自发性破溃"]
MECH_SUBTYPES = ["切割伤", "刺伤", "挫裂伤", "撕裂伤", "撕脱伤",
                 "开放性骨折", "爆炸碎片伤", "火器伤", "挤压破裂", "抓咬伤", "擦伤"]
LOCATIONS = ["头部", "颈部", "躯干", "手臂", "手部", "腿部", "足部", "多部位"]

STEP1_RULE = """核心原则：以皮肤/粘膜是否完整为准
- 开放式：皮肤或粘膜有破损、有伤口、有外出血、有溃烂、有焦痂、有水泡破溃
- 闭合式：皮肤和粘膜完整无损（包括骨折、脱臼、扭伤、内出血、中毒、中暑、溺水、电击、不破皮的烧烫伤、冻伤等）

只输出一个类型词"""

CLEAN_PROMPT = "删除脏话、感叹词、求助语、时间地点，保留伤情和受伤经过，用逗号分隔，只输出清理结果：{text}"
GROUP_PROMPT = "判断受伤组别（机械/热力/电能/化学/辐射/生物/病理）：{text}"
DETAIL_PROMPT = "判断开放机械损伤类型（切割/刺伤/挫裂/撕裂/撕脱/开放骨折/爆炸碎片/火器/挤压/抓咬/擦伤）：{text}"
LOCATION_PROMPT = "判断受伤部位（头/颈/躯干/手臂/手/腿/足/多发）：{text}"

# ===== 双通道不确定性阈值（论文 4.3，与 pipeline.py 一致）=====
ENTROPY_RATIO_GATE = 0.90
PROB_MARGIN_THRESHOLD = 0.25
SEVERE_PROB_MARGIN = 0.46
HIGH_SEVERITY_SUBTYPES = ["开放性骨折", "挤压破裂", "撕脱伤", "火器伤"]

# ===== 皮肤/粘膜状态正则（追问规则依据）=====
SKIN_INTACT_RE = re.compile(
    r'没破|未破|没伤|没有伤口|没出血|没流血|无破损|完好|完整|'
    r'皮肤没事|没裂开|没伤到皮|皮好好的|皮肤好好的')
SKIN_BROKEN_RE = re.compile(
    r'破皮|破了|皮肤[^，。]{0,3}破|伤口|一道口子|血止不住|'
    r'(?<!内)(?<!脑)(?<!皮下)出血|流血|血直流|直流血|'
    r'裂开|豁开|溃烂|烂了|烂掉|烂穿|烂开|流脓|化脓|焦痂|'
    r'焦了|皮焦|烧焦|烧掉|烧出.{0,2}洞|烧了个洞|烫脱|烫掉|'
    r'水泡[^，。]{0,2}破|皮肉翻开|撕开|'
    r'皮都?掉|皮粘.{0,6}(掉|撕)|掉了一块|烧穿|穿了?个洞|皮开肉绽|掀了皮|'
    r'骨头[^，。]{0,2}(戳|露)|外露|贯通')

# ===== 脏话/感叹词/求助语（只删噪声不动伤情）=====
PROFANITY_RE = re.compile(
    r'卧槽|握草|沃草|我操你|操你[妈的]*|他妈的|特么的|特么|妈的|尼玛|草泥马|'
    r'去你[大爷妈的]*|该死|见鬼|'
    r'救命[啊呀哈]?|帮帮我[吧呗]?|来人[啊呀]?|'
    r'疼死[我了]*|痛死[我了]*|急死[我了]*|吓死[我了]*|'
    r'妈呀|妈耶|天哪|我的天[哪呀]?|哎呀|哎哟|我靠|'
    r'[哈]{2,}|[嘿]{2,}|[呵]{2,}|[哦嗯呀哦欸噢诶]'
)

INJURY_KEYWORDS = [
    '骨折', '骨头断', '骨头裂', '断了', '折了', '裂了',
    '脱臼', '脱位',
    '扭伤', '拉伤', '挫伤', '砸伤', '撞伤', '砍伤', '打伤',
    '割伤', '割', '刺伤', '切伤', '划伤', '扎伤', '戳伤', '砍',
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
    '摔', '坠', '高空', '滑倒', '跌倒', '磕', '肿',
]

_PUNCT_RE = re.compile(r'[，,。.！!？?、\s]')
_PARTICLE_RE = re.compile(r'[了的着过啊呀哦嗯吧呢]')


# ============ 模型池：按需顺序加载（一次只驻留一个）============

class GGUFModels:
    """懒加载 + 顺序驻留。get(name) 关闭上一个模型再加载新模型，控制内存。"""

    def __init__(self, model_dir, n_ctx=512, n_batch=512):
        self.model_dir = model_dir
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self._current_name = None
        self._current_llm = None

    def _path(self, name):
        return os.path.join(self.model_dir, _MODEL_FILES[name])

    def get(self, name):
        if self._current_name == name and self._current_llm is not None:
            return self._current_llm
        self.release()
        from llama_cpp import Llama
        llm = Llama(
            model_path=self._path(name),
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_gpu_layers=0,      # 纯 CPU 推理
            logits_all=True,     # 保留每 token logits，用于候选评分
            verbose=False,
        )
        self._current_name = name
        self._current_llm = llm
        return llm

    def release(self):
        if self._current_llm is not None:
            try:
                self._current_llm.close()
            except Exception:
                pass
            self._current_llm = None
            self._current_name = None


# ============ 候选分布（论文 4.3 核心，与 pipeline.py 的 _candidate_distribution 对齐）============

def _log_softmax(x):
    x = x - np.max(x)
    return x - np.log(np.sum(np.exp(x)))


def _candidate_logprob(llm, prompt_ids, cand_ids):
    """对 prompt + candidate 做前向，累加 candidate 各 token 的 log 概率。

    HF 版 _candidate_loss 用的是 candidate 区间的平均交叉熵 loss；
    这里先求 Σ log p(token)，再由调用方除以 token 数得到平均 CE。

    llama-cpp-python 下 eval() 不返回 logits，需用 logits_all=True 后
    从 llm.scores[位置] 读取：scores[base+i-1] 即预测第 i 个候选 token 的 logits。
    """
    llm.reset()
    llm.eval(prompt_ids)
    base = len(prompt_ids)
    total = 0.0
    n = len(cand_ids)
    for i in range(n):
        pos = base + i - 1                       # 预测 candidate[i] 的 logits 位置
        logits = np.asarray(llm.scores[pos], dtype=np.float64)
        total += _log_softmax(logits)[cand_ids[i]]
        if i < n - 1:
            llm.eval([cand_ids[i]])
    return total


def candidate_distribution(llm, prompt, candidates, special=True):
    prompt_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=special)
    cand_ids = {
        c: llm.tokenize(c.encode("utf-8"), add_bos=False, special=False)
        for c in candidates
    }
    losses = {}
    for c, ids in cand_ids.items():
        lp = _candidate_logprob(llm, prompt_ids, ids)
        losses[c] = -lp / len(ids)          # 平均 CE == HF 的 labels 口径
    names = list(losses.keys())
    scores = np.array([-losses[c] for c in names], dtype=np.float64)
    probs_v = np.exp(scores - np.max(scores))
    probs_v = probs_v / probs_v.sum()
    probs = dict(zip(names, probs_v.tolist()))
    ranked = sorted(names, key=lambda c: probs[c], reverse=True)
    entropy = -sum(p * math.log2(p) for p in probs_v if p > 0.0)
    h_max = math.log2(len(names)) if len(names) > 1 else 1.0
    margin = probs[ranked[0]] - (probs[ranked[1]] if len(names) > 1 else 0.0)
    return {
        "best": ranked[0],
        "second": ranked[1] if len(names) > 1 else None,
        "probs": probs,
        "losses": losses,
        "entropy": entropy,
        "h_max": h_max,
        "h_ratio": entropy / h_max,
        "margin": margin,
    }


# ============ 通用小工具（与 pipeline.py 一致）============

def _generate(llm, prompt, max_new_tokens=16):
    res = llm.create_completion(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0,
        stop=["<|im_end|>"],
        echo=False,
    )
    return res["choices"][0]["text"].strip()


def _extract_keywords(text):
    return [kw for kw in INJURY_KEYWORDS if kw in text]


def has_explicit_skin_status(text):
    return bool(SKIN_INTACT_RE.search(text) or SKIN_BROKEN_RE.search(text))


def needs_skin_question(text):
    """方案 A：原文没交代皮肤状态就追问（与 pipeline.py 共用同一规则）。"""
    return not has_explicit_skin_status(text)


def _append_skin_status(injury, answer):
    a = str(answer).strip().lower()
    if a in ("有", "是", "对", "破了", "流血", "流血了", "出血", "烂了",
             "骨头戳", "骨头戳出来", "骨头露", "骨头露出来"):
        return injury + " 皮肤粘膜有破损"
    if a in ("没有", "无", "没", "没破", "没破皮", "完好", "皮肤完整", "没流血", "没出血"):
        return injury + " 皮肤粘膜完整无损"
    return None


def _ask_skin_question(injury):
    if "骨折" in injury or "断" in injury:
        return f"皮肤有没有破？骨头有没有戳出来？（有/没有）"
    if any(k in injury for k in ["烧", "烫"]):
        return f"皮肤有没有破损、水泡或发黑？（有/没有）"
    if "冻" in injury:
        return f"皮肤有没有破皮、溃烂或发黑？（有/没有）"
    if any(k in injury for k in ["化学", "腐蚀", "酸碱"]):
        return f"皮肤有没有溃烂、水泡或发黑？（有/没有）"
    if any(k in injury for k in ["电击", "电伤", "触电"]):
        return f"皮肤有没有伤口或焦痂？（有/没有）"
    return f"皮肤有没有破损？（有=开放式/没有=闭合式）"


def _fallback_split(text):
    parts = re.split(r'[，,；;、和与及或跟以及同时此外另外还有伴有伴随而且并且然后接着还又也再加]', text)
    parts = [p.strip().strip('。').strip() for p in parts]
    parts = [p for p in parts if p and len(p) >= 2]

    if len(parts) == 1:
        p = parts[0]
        kw_positions = []
        for kw in INJURY_KEYWORDS:
            start = 0
            while True:
                idx = p.find(kw, start)
                if idx == -1:
                    break
                kw_positions.append((idx, idx + len(kw)))
                start = idx + len(kw)
        kw_positions.sort()
        if len(kw_positions) >= 2:
            split_points = [0]
            for i in range(1, len(kw_positions)):
                prev_end, curr_start = kw_positions[i - 1][1], kw_positions[i][0]
                if curr_start > prev_end:
                    middle = p[prev_end:curr_start]
                    m = re.search(r'[了的着过啊呀]', middle)
                    split_points.append(prev_end + (m.end() if m else 0))
            split_points.append(len(p))
            parts = [p[split_points[i]:split_points[i + 1]].strip()
                     for i in range(len(split_points) - 1)]
            parts = [p for p in parts if p and len(p) >= 2]

    return parts if parts else [text]


def _grounded_filter(items, text):
    """防幻觉：清洗只允许删噪声，不允许生成。"""
    norm_text = _PUNCT_RE.sub('', text)
    kept = []
    for it in items:
        norm_it = _PUNCT_RE.sub('', it)
        if norm_it and norm_it in norm_text:
            kept.append(it)
            continue
        loose = _PARTICLE_RE.sub('', norm_it)
        if loose and loose in norm_text:
            kept.append(it)
    return kept


# ============ 阶段 1：NLU 清洗 + 拆分（纯 base GGUF）============

def _nlu_parse(models, text):
    llm = models.get("clean")
    prompt = f'<|im_start|>user\n{CLEAN_PROMPT.format(text=text)}<|im_end|>\n<|im_start|>assistant\n'
    raw = _generate(llm, prompt, max_new_tokens=96)
    if not raw.strip():
        return None

    items = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('```'):
            continue
        line = re.sub(r'^[\d一二三四五六七八九十]+[\.\、\)\]]\s*', '', line)
        for part in re.split(r'[，,]', line):
            part = part.strip().strip('。.')
            if part and len(part) >= 2:
                items.append(part)

    seen, unique = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique if unique else None


def parse_injuries(models, text):
    """清洗拆分（纯 base）。与 pipeline.py 一致：先防幻觉、再词表校验、最后正则兜底。"""
    items = _nlu_parse(models, text)

    if items:
        items = _grounded_filter(items, text)
        kept = [i for i in items if _extract_keywords(i)]
        if kept:
            items = kept
        elif _extract_keywords(text):
            items = None

    if not items:
        items = _fallback_split(text)
        kept = [p for p in items if _extract_keywords(p)]
        items = kept if kept else [text]

    return [{"injury": s} for s in items]


# ============ 阶段 2~6：Step1 / 追问 / Step2a / Step2b / Step3 ============

def predict_type(models, text):
    llm = models.get("step1")
    prompt = f'<|im_start|>user\n判断受伤类型：{text}\n{STEP1_RULE}<|im_end|>\n<|im_start|>assistant\n'
    dist = candidate_distribution(llm, prompt, ALLOWED_TYPES)
    global_confused = dist["h_ratio"] >= ENTROPY_RATIO_GATE
    local_ambiguous = dist["margin"] < PROB_MARGIN_THRESHOLD
    return dist, global_confused, local_ambiguous


def predict_group(models, text):
    llm = models.get("step2a")
    prompt = f'<|im_start|>user\n{GROUP_PROMPT.format(text=text)}<|im_end|>\n<|im_start|>assistant\n'
    return candidate_distribution(llm, prompt, GROUPS)


def predict_detail(models, text):
    llm = models.get("step2b")
    prompt = f'<|im_start|>user\n{DETAIL_PROMPT.format(text=text)}<|im_end|>\n<|im_start|>assistant\n'
    dist = candidate_distribution(llm, prompt, MECH_SUBTYPES)
    severe_second = dist["second"] in HIGH_SEVERITY_SUBTYPES
    need_ask = (dist["h_ratio"] >= ENTROPY_RATIO_GATE
                or dist["margin"] < PROB_MARGIN_THRESHOLD
                or (severe_second and dist["margin"] < SEVERE_PROB_MARGIN))
    return dist, need_ask


def predict_location(models, text):
    llm = models.get("step3")
    prompt = f'<|im_start|>user\n{LOCATION_PROMPT.format(text=text)}<|im_end|>\n<|im_start|>assistant\n'
    return candidate_distribution(llm, prompt, LOCATIONS)


def resolve_mutual_exclusion(injury, pred_type):
    if "皮肤粘膜有破损" in injury and pred_type == "闭合式":
        return "开放式"
    if "皮肤粘膜完整无损" in injury and pred_type == "开放式":
        return "闭合式"
    return pred_type


# ============ 主流水线 ============

def classify(models, raw_text, ask_fn=None):
    """对一段报案原文做完整分类。ask_fn(问题字符串)->回答字符串；为 None 则跳过追问。"""
    parsed = parse_injuries(models, raw_text)
    # 单一伤情时直接用原文（清洗模型会误删受伤经过）
    if len(parsed) == 1:
        parsed = [{"injury": raw_text.strip()}]
    if not any(_extract_keywords(it["injury"]) for it in parsed):
        print("  ⚠ 未检测到明确伤情词，分类结果可能不可靠")

    injuries = []
    for i, item in enumerate(parsed):
        injury = item["injury"]

        # 正则剥脏话
        stripped = PROFANITY_RE.sub('', injury).strip('，。,、.！!？?~ ')
        if stripped and stripped != injury:
            injury = stripped
            item["injury"] = stripped

        dist, global_confused, local_ambiguous = predict_type(models, injury)

        # 通道一：全局混乱 → 开放式提问先降熵（有 ask_fn 才执行）
        open_followup_done = False
        if global_confused and ask_fn is not None:
            q = f"[{injury}] 信息不太够，能用一句话说明受伤经过吗？具体是怎么弄伤的？（不知道输'跳过'）"
            ans = ask_fn(q)
            if ans and str(ans).strip() != "跳过":
                injury = f"{injury} {str(ans).strip()}"
                open_followup_done = True
                dist, global_confused, local_ambiguous = predict_type(models, injury)

        # 通道二：皮肤状态缺失 → 定向追问（方案 A）
        if needs_skin_question(injury) and ask_fn is not None:
            q = _ask_skin_question(injury)
            ans = ask_fn(q)
            injury_new = _append_skin_status(injury, ans)
            if injury_new is not None:
                injury = injury_new
                dist, _, _ = predict_type(models, injury)

        pred_type = resolve_mutual_exclusion(injury, dist["best"])

        # Step2a 组别（开放/闭合皆判）
        gdist = predict_group(models, injury)
        group = gdist["best"]
        if (not open_followup_done and ask_fn is not None
                and (gdist["h_ratio"] >= ENTROPY_RATIO_GATE
                     or gdist["margin"] < PROB_MARGIN_THRESHOLD)):
            q = f"[{injury}] 信息不太够，能用一句话说明受伤经过吗？具体是怎么弄伤的？（不知道输'跳过'）"
            ans = ask_fn(q)
            if ans and str(ans).strip() != "跳过":
                injury = f"{injury} {str(ans).strip()}"
                open_followup_done = True
                gdist = predict_group(models, injury)
                group = gdist["best"]

        # Step2b 细分：仅开放式 + 机械性损伤
        detail, detail_margin, detail_entropy = None, None, None
        if pred_type == "开放式" and group == "机械性损伤":
            ddist, need_ask = predict_detail(models, injury)
            if need_ask and not open_followup_done and ask_fn is not None:
                q = (f"这个伤具体怎么造成的？有没有剧烈疼痛、变形或动不了？（一句话说明，跳过输'跳过'）")
                ans = ask_fn(q)
                if ans and str(ans).strip() != "跳过":
                    injury = f"{injury} {str(ans).strip()}"
                    open_followup_done = True
                    ddist, _ = predict_detail(models, injury)
            detail, detail_margin, detail_entropy = (
                ddist["best"], ddist["margin"], ddist["entropy"])
        elif pred_type == "开放式":
            detail = "(跳过-非机械组)"
        else:
            detail = "(跳过-闭合式)"

        # Step3 部位（八选一）
        ldist = predict_location(models, injury)
        location = ldist["best"]

        injuries.append({
            "idx": i + 1,
            "injury": item["injury"],
            "resolved": injury,
            "type": pred_type,
            "group": group,
            "detail": detail,
            "location": location,
            "margin": round(dist["margin"], 3),
            "entropy": round(dist["entropy"], 3),
            "group_entropy": round(gdist["entropy"], 3),
            "detail_margin": round(detail_margin, 3) if detail_margin is not None else None,
            "detail_entropy": round(detail_entropy, 3) if detail_entropy is not None else None,
            "location_margin": round(ldist["margin"], 3),
        })

    return {"original": raw_text, "injuries": injuries}


def build_classifier(model_dir):
    """返回 (classify_callable, models)。model_dir 为存放 5 个 GGUF 的目录。"""
    models = GGUFModels(model_dir)

    def _classify(text, ask_fn=None):
        return classify(models, text, ask_fn=ask_fn)

    return _classify, models


def print_result(result):
    print(f"\n  原始: {result['original']}")
    for inv in result["injuries"]:
        if inv["resolved"] != inv["injury"]:
            print(f"    伤情{inv['idx']}: {inv['injury']}  →  {inv['resolved']}")
        else:
            print(f"    伤情{inv['idx']}: {inv['injury']}")
        print(f"          type={inv['type']}  group={inv['group']}  detail={inv['detail']}  location={inv['location']}")
        print(f"          margin={inv['margin']}  H={inv['entropy']}  loc_margin={inv['location_margin']}")
    print(f"\n  JSON: {json.dumps(result, ensure_ascii=False)}\n")


def main():
    ap = argparse.ArgumentParser(description="急救伤情理解 GGUF CPU 推理")
    ap.add_argument("--text", type=str, default=None, help="单句测试，不进交互")
    ap.add_argument("--model-dir", type=str, default=None,
                    help="GGUF 目录（默认 ../.. /models/gguf）")
    args = ap.parse_args()

    if args.model_dir:
        model_dir = args.model_dir
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        # 部署包：脚本同级有 models/ 目录
        if os.path.isdir(os.path.join(here, "models")):
            model_dir = os.path.join(here, "models")
        # 项目内调试：../../models/gguf
        else:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(here)), "models", "gguf")

    classify_fn, models = build_classifier(model_dir)

    if args.text:
        print_result(classify_fn(args.text, ask_fn=None))
        models.release()
        return

    print("\n" + "=" * 60)
    print("  急救伤情理解（GGUF CPU：清洗→追问→类型→组别→细分→部位）")
    print("  输入 q 退出")
    print("=" * 60)
    while True:
        try:
            raw_text = input("\n> ").strip()
            if raw_text.lower() == 'q':
                break
            if not raw_text:
                continue
            result = classify_fn(raw_text, ask_fn=input)
            print_result(result)
        except KeyboardInterrupt:
            break
        except Exception:
            import traceback
            traceback.print_exc()

    models.release()


if __name__ == "__main__":
    main()