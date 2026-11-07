#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整急救 NLU 流水线（打包版）
  报案原文
    → NLU 清洗（纯 base Qwen，只许删噪声，防幻觉校验）
    → 多伤情拆分（NLU 解析 → 连词正则兜底 → 逐段伤情词校验）
    → 逐条追问补全（双通道不确定性，论文 4.3：
        整体信息熵 H/Hmax ≥ 0.90 → 开放式提问先降熵；
        top-2 概率差 margin < 0.25 或命中高风险词 → 定向追问）
    → Step1 类型分类（开放式/闭合式，N=2）+ 医学互斥修正
    → Step2a 组别分类（七选一，N=7，Hmax≈2.807 bit，开放/闭合皆判）
    → Step2b 组内细分（机械组十一选一，N=11，仅开放式+机械性损伤）
    → 输出结构化 JSON

用法：
  交互式：D:\\python\\python.exe src\\train\\pipeline.py
  单句  ：D:\\python\\python.exe src\\train\\pipeline.py --text "卧槽骨折了烫伤"
"""

import os, sys, json, argparse, re, math

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"


def _app_dir():
    """返回模型根目录：PyInstaller 打包后取 exe 同级目录，否则取项目根目录"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_BASE = _app_dir()
os.environ["HF_HOME"] = os.path.join(_BASE, "models", "cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_BASE, "models", "cache")

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = _BASE
MODEL_PATH = os.path.join(BASE_DIR, "models", "cache",
    "models--Qwen--Qwen2.5-1.5B-Instruct",
    "snapshots", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
STEP1_LORA = os.path.join(BASE_DIR, "models", "step1_type_lora")
STEP2_LORA = os.path.join(BASE_DIR, "models", "step2_open_group_lora")
STEP2B_LORA = os.path.join(BASE_DIR, "models", "step2_open_mechanical_detail_lora")

ALLOWED_TYPES = ["开放式", "闭合式"]
GROUPS = ["机械性损伤", "热力损伤", "电能损伤", "化学腐蚀",
          "辐射损伤", "生物毒素-感染", "病理自发性破溃"]
MECH_SUBTYPES = ["切割伤", "刺伤", "挫裂伤", "撕裂伤", "撕脱伤",
                 "开放性骨折", "爆炸碎片伤", "火器伤", "挤压破裂", "抓咬伤", "擦伤"]

STEP1_RULE = """核心原则：以皮肤/粘膜是否完整为准
- 开放式：皮肤或粘膜有破损、有伤口、有外出血、有溃烂、有焦痂、有水泡破溃
- 闭合式：皮肤和粘膜完整无损（包括骨折、脱臼、扭伤、内出血、中毒、中暑、溺水、电击、不破皮的烧烫伤、冻伤等）

只输出一个类型词"""

# ===== 双通道不确定性判定（论文 4.3）=====
# 通道一（整体信息熵）：H/Hmax ≥ 0.90 → 全局混乱（概率摊平）→ 开放式提问先降熵
ENTROPY_RATIO_GATE = 0.90
# 通道二（局部 Margin）：top-2 概率差 < 0.25 → 在两个候选间犹豫 → 定向追问
# （0.25 概率差 ≈ 旧版 loss 差 0.5：p1-p2 = tanh(Δloss/2)）
PROB_MARGIN_THRESHOLD = 0.25
# 次优候选是重伤类型时放宽到 0.46（≈ 旧版 loss 差 1.0），宁可多问不漏重伤
SEVERE_PROB_MARGIN = 0.46
HIGH_SEVERITY_SUBTYPES = ["开放性骨折", "挤压破裂", "撕脱伤", "火器伤"]
CLEAN_PROMPT = "删除脏话、感叹词、求助语、时间地点，保留伤情和受伤经过，用逗号分隔，只输出清理结果：{text}"
GROUP_PROMPT = "判断受伤组别（机械/热力/电能/化学/辐射/生物/病理）：{text}"
DETAIL_PROMPT = "判断开放机械损伤类型（切割/刺伤/挫裂/撕裂/撕脱/开放骨折/爆炸碎片/火器/挤压/抓咬/擦伤）：{text}"

# 皮肤/粘膜状态是否已在原文明确交代（追问规则依据：缺失就必须问）
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


def has_explicit_skin_status(text):
    """原文是否已明确说明皮肤/粘膜破没破"""
    return bool(SKIN_INTACT_RE.search(text) or SKIN_BROKEN_RE.search(text))


def needs_skin_question(text):
    """方案 A：原文没交代皮肤状态就追问，不依赖模型置信度。

    模型对非机械伤常自信地猜成闭合式（margin 很大），阈值通道够不到，
    而追问后的纠正率接近 100%，因此以“信息缺失”而非“模型犹豫”作为触发条件。
    """
    return not has_explicit_skin_status(text)

# 脏话/感叹词/求助语正则：词组级匹配，避免误伤"操场"等正常词；只删噪声不动伤情内容
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


# ============ 模型加载 ============

def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 实例一：清洗模型，纯 base，禁止挂任何 adapter（防幻觉）
    nlu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        local_files_only=True, dtype=dtype
    ).to(device)
    print(f"✓ NLU 清洗（纯 base，无 adapter）")

    # 实例二：分类模型，挂三个分类适配器
    lora_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        local_files_only=True, dtype=dtype
    ).to(device)
    lora_model = PeftModel.from_pretrained(lora_model, STEP1_LORA, adapter_name="step1")
    lora_model.load_adapter(STEP2_LORA, adapter_name="step2")
    if os.path.exists(os.path.join(STEP2B_LORA, "adapter_config.json")):
        lora_model.load_adapter(STEP2B_LORA, adapter_name="step2b")
        print(f"✓ Step2b adapter → {STEP2B_LORA}")
    else:
        print(f"⚠ Step2b adapter 不存在，细分将跳过")

    print(f"✓ Step1 adapter → {STEP1_LORA}")
    print(f"✓ Step2a adapter → {STEP2_LORA}")
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


# ============ 阶段 1：NLU 清洗 + 拆分 ============

def _extract_keywords(text):
    return [kw for kw in INJURY_KEYWORDS if kw in text]


def _nlu_parse(nlu_model, tokenizer, text):
    prompt = f'<|im_start|>user\n{CLEAN_PROMPT.format(text=text)}<|im_end|>\n<|im_start|>assistant\n'
    raw = _generate(nlu_model, tokenizer, prompt, max_new_tokens=96)

    print(f"  [NLU 原始输出: {raw.strip()[:120]}]")
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

    print(f"  [NLU 解析: {unique}]")
    return unique if unique else None


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


_PUNCT_RE = re.compile(r'[，,。.！!？?、\s]')
_PARTICLE_RE = re.compile(r'[了的着过啊呀哦嗯吧呢]')


def _grounded_filter(items, text):
    """防幻觉：清洗只允许删噪声，不允许生成。
    片段不在原文中（连去掉助词后也不在）就是模型编的，丢弃。"""
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
        else:
            print(f"  [丢弃幻觉片段: {it}]")
    return kept


def parse_injuries(nlu_model, tokenizer, text):
    items = _nlu_parse(nlu_model, tokenizer, text)

    if items:
        # 第一层：防幻觉，编造的片段直接丢
        items = _grounded_filter(items, text)
        # 第二层：逐段校验，没有伤情词的片段（救命/疼死了）丢弃
        kept = [i for i in items if _extract_keywords(i)]
        if kept:
            items = kept
        elif _extract_keywords(text):
            print(f"  [清洗结果全部无效，回退原文]")
            items = None
        # 原文也没伤情词时，保留清洗结果原样

    if not items:
        print(f"  [NLU 无效，连词正则兜底]")
        items = _fallback_split(text)
        kept = [p for p in items if _extract_keywords(p)]
        items = kept if kept else [text]

    print(f"  [原始: {text}]")
    print(f"  [伤情: {items}]")
    return [{"injury": s} for s in items]


# ============ 阶段 2：Step1 类型分类 + 不确定性 ============

def _candidate_loss(model, tokenizer, prompt, target):
    """对 prompt+target 跑一次 forward，返回 target 区间的交叉熵"""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target, return_tensors="pt").input_ids.to(model.device)

    full_ids = torch.cat([prompt_ids, target_ids], dim=1)
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(full_ids, labels=labels)
    return outputs.loss.item()


def _candidate_distribution(model, tokenizer, prompt, candidates):
    """论文 4.3 双通道不确定性的统一计算（Step1/2a/2b 共用）。

    1) 对 N 个候选标签各算一次完整序列交叉熵 loss_i；
    2) 用 softmax(-loss_i) 把候选得分还原成概率分布 P={p_i}；
    3) 整体信息熵 H(P) = -Σ p_i·log2(p_i)（bit），最大熵 Hmax = log2(N)
       （如 N=7 时 Hmax≈2.807）；
    4) 局部 Margin = p_(1) - p_(2)，取概率最高的两项做差。

    H 逼近 Hmax → 概率摊平、全局混乱；Margin 小 → 只在最高两项间犹豫。
    """
    losses = {c: _candidate_loss(model, tokenizer, prompt, c) for c in candidates}
    names = list(losses.keys())
    scores = torch.tensor([-losses[c] for c in names], dtype=torch.float64)
    probs_v = torch.softmax(scores, dim=0).tolist()
    probs = dict(zip(names, probs_v))
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


def predict_type_with_confidence(model, tokenizer, text):
    """Step1（N=2）双通道判定。
    返回 (候选分布, 全局混乱, 局部犹豫)：
      全局混乱 → 先开放式提问降熵；局部犹豫 → 定向皮肤追问。"""
    model.set_adapter("step1")
    prompt = f'<|im_start|>user\n判断受伤类型：{text}\n{STEP1_RULE}<|im_end|>\n<|im_start|>assistant\n'
    dist = _candidate_distribution(model, tokenizer, prompt, ALLOWED_TYPES)
    global_confused = dist["h_ratio"] >= ENTROPY_RATIO_GATE
    local_ambiguous = dist["margin"] < PROB_MARGIN_THRESHOLD
    return dist, global_confused, local_ambiguous


# ============ 阶段 3：追问 + 补全 ============

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
    if "冻" in injury:
        return f"[{injury}] 皮肤有没有破皮、溃烂或发黑？（有/没有）"
    if any(k in injury for k in ["化学", "腐蚀", "酸碱"]):
        return f"[{injury}] 皮肤有没有溃烂、水泡或发黑？（有/没有）"
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
        print(f"  ⚠ 请输入 '有' 或 '没有'，跳过输入 '跳过'")
    return injury, False


def _ask_open_question(injury):
    """通道一（整体信息熵逼近最大熵）的开放式提问：不给选项，让报案人
    自由描述受伤经过，回答原样拼回后重新计算分布——先把整体熵降下来，
    再交给 Margin 通道判断最高两项是否仍然接近。每条伤情至多问一次。"""
    q = (f"[{injury}] 信息不太够，能用一句话说明受伤经过吗？"
         f"具体是怎么弄伤的？（不知道输'跳过'）")
    ans = input(f"  {q} ").strip()
    if ans and ans != "跳过":
        return f"{injury} {ans}", True
    return injury, False


# ============ 阶段 4：医学互斥修正 ============

def resolve_mutual_exclusion(injury, pred_type):
    """用户已明确回答皮肤状态时，回答是硬事实，模型必须服从"""
    if "皮肤粘膜有破损" in injury and pred_type == "闭合式":
        return "开放式"
    if "皮肤粘膜完整无损" in injury and pred_type == "开放式":
        return "闭合式"
    return pred_type


# ============ 阶段 5：Step2a 组别 + Step2b 细分 ============

def predict_group_with_confidence(model, tokenizer, text):
    """Step2a 七候选分布（N=7，Hmax≈2.807 bit），与 Step1 共用同一套
    双通道不确定性计算（论文 4.3）。"""
    model.set_adapter("step2")
    prompt = f'<|im_start|>user\n{GROUP_PROMPT.format(text=text)}<|im_end|>\n<|im_start|>assistant\n'
    return _candidate_distribution(model, tokenizer, prompt, GROUPS)


def predict_detail_with_confidence(model, tokenizer, text):
    """Step2b 十一候选分布（N=11）+ 双通道追问判定（论文 4.3）。
    整体信息熵接近最大熵（全局混乱）或 top-2 概率差小（局部犹豫）都要问；
    次优候选是重伤类型（开放性骨折等）时放宽 Margin 阈值，宁可多问不漏重伤。"""
    model.set_adapter("step2b")
    prompt = f'<|im_start|>user\n{DETAIL_PROMPT.format(text=text)}<|im_end|>\n<|im_start|>assistant\n'
    dist = _candidate_distribution(model, tokenizer, prompt, MECH_SUBTYPES)
    severe_second = dist["second"] in HIGH_SEVERITY_SUBTYPES
    need_ask = (dist["h_ratio"] >= ENTROPY_RATIO_GATE
                or dist["margin"] < PROB_MARGIN_THRESHOLD
                or (severe_second and dist["margin"] < SEVERE_PROB_MARGIN))
    return dist, need_ask


# ============ 主流水线 ============

def predict_pipeline(nlu_model, lora_model, tokenizer, raw_text):
    parsed = parse_injuries(nlu_model, tokenizer, raw_text)
    # 单一伤情时直接用原文：清洗模型会误删受伤经过（如"从高处"），
    # 且分类模型本就是在带噪声的原生报案文本上训练的，原文更贴近训练分布。
    # 清洗的价值只在多伤情拆分。
    if len(parsed) == 1:
        parsed = [{"injury": raw_text.strip()}]
    # 软警告：整段输入没有任何伤情词时分类结果大概率是噪声
    if not any(_extract_keywords(it["injury"]) for it in parsed):
        print("  ⚠ 未检测到明确伤情词，分类结果可能不可靠")
    injuries = []

    for i, item in enumerate(parsed):
        injury = item["injury"]

        # 正则剥脏话：NLU 清洗的确定性补充，防止噪声干扰组别分类
        stripped = PROFANITY_RE.sub('', injury).strip('，。,、.！!？?~ ')
        if stripped and stripped != injury:
            print(f"  [剥噪: {injury} → {stripped}]")
            injury = stripped
            item["injury"] = stripped

        dist, global_confused, local_ambiguous = predict_type_with_confidence(
            lora_model, tokenizer, injury)

        # 通道一：整体信息熵逼近最大熵（全局混乱）→ 开放式提问，先把熵降下来
        open_followup_done = False
        if global_confused:
            print(f"  [H={dist['entropy']:.3f}/{dist['h_max']:.3f} 全局混乱 → 开放式提问降熵]")
            injury, got = _ask_open_question(injury)
            open_followup_done = got
            if got:
                dist, global_confused, local_ambiguous = predict_type_with_confidence(
                    lora_model, tokenizer, injury)

        # 通道二：原文未交代皮肤状态 → 定向皮肤追问（方案 A）
        if needs_skin_question(injury):
            print(f"  [皮肤状态缺失 → 定向追问]")
            q = _ask_skin_question(injury)
            injury, asked = _ask_skin_once(injury, q)
            if asked:
                dist, _, _ = predict_type_with_confidence(
                    lora_model, tokenizer, injury)

        # 医学互斥：用户确认的皮肤状态是硬事实
        pred_type = resolve_mutual_exclusion(injury, dist["best"])
        margin, losses = dist["margin"], dist["losses"]
        type_entropy = dist["entropy"]

        detail, detail_gap, detail_entropy = None, None, None
        group, group_entropy = None, None

        # Step2a 组别（受伤原因）：开放/闭合都判
        gdist = predict_group_with_confidence(lora_model, tokenizer, injury)
        group, group_entropy = gdist["best"], gdist["entropy"]
        # 组别的全局混乱/局部犹豫：复用同一个开放式受伤经过提问（每条伤情至多一次）
        if (not open_followup_done and
                (gdist["h_ratio"] >= ENTROPY_RATIO_GATE
                 or gdist["margin"] < PROB_MARGIN_THRESHOLD)):
            print(f"  [组别 H={gdist['entropy']:.3f}/{gdist['h_max']:.3f}，"
                  f"margin={gdist['margin']:.3f} → 开放式提问]")
            injury, got = _ask_open_question(injury)
            open_followup_done = got
            if got:
                gdist = predict_group_with_confidence(lora_model, tokenizer, injury)
                group, group_entropy = gdist["best"], gdist["entropy"]

        # Step2b 细分：仅"开放式 + 机械性损伤"（细分模型为开放机械专用）
        if pred_type == "开放式" and group == "机械性损伤":
            if os.path.exists(os.path.join(STEP2B_LORA, "adapter_config.json")):
                ddist, need_ask = predict_detail_with_confidence(
                    lora_model, tokenizer, injury)
                if need_ask and not open_followup_done:
                    ans = input(f"  [{injury}] 这个伤具体怎么造成的？有没有剧烈疼痛、变形或动不了？"
                                f"（一句话说明，跳过输入'跳过'）").strip()
                    if ans and ans != "跳过":
                        injury = f"{injury} {ans}"
                        open_followup_done = True
                        ddist, _ = predict_detail_with_confidence(
                            lora_model, tokenizer, injury)
                detail, detail_gap, detail_entropy = (
                    ddist["best"], ddist["margin"], ddist["entropy"])
            else:
                detail = "(step2b未训练)"
        elif pred_type == "开放式":
            detail = "(跳过-非机械组)"
        else:
            detail = "(跳过-闭合式)"

        injuries.append({
            "idx": i + 1,
            "injury": item["injury"],
            "resolved": injury,
            "type": pred_type,
            "group": group,
            "detail": detail,
            "margin": round(margin, 3),
            "entropy": round(type_entropy, 3),
            "group_entropy": round(group_entropy, 3) if group_entropy is not None else None,
            "detail_margin": round(detail_gap, 3) if detail_gap is not None else None,
            "detail_entropy": round(detail_entropy, 3) if detail_entropy is not None else None,
            "losses": {t: round(v, 3) for t, v in losses.items()},
        })

    return {"original": raw_text, "injuries": injuries}


def print_result(result):
    print(f"\n  原始: {result['original']}")
    for inv in result["injuries"]:
        if inv["resolved"] != inv["injury"]:
            print(f"    伤情{inv['idx']}: {inv['injury']}  →  {inv['resolved']}")
        else:
            print(f"    伤情{inv['idx']}: {inv['injury']}")
        print(f"           type={inv['type']}  group={inv['group']}  detail={inv['detail']}  "
              f"margin={inv['margin']}  H={inv['entropy']}  losses={inv['losses']}")
    print(f"\n  JSON: {json.dumps(result, ensure_ascii=False)}\n")


def main():
    parser = argparse.ArgumentParser(description="急救 NLU 完整流水线")
    parser.add_argument("--text", type=str, default=None, help="单句测试，不进交互")
    args = parser.parse_args()

    nlu_model, lora_model, tokenizer = load_pipeline()

    if args.text:
        result = predict_pipeline(nlu_model, lora_model, tokenizer, args.text)
        print_result(result)
        return

    print("\n" + "=" * 60)
    print("  急救 NLU 完整流水线（清洗→拆分→追问→类型→组别→细分）")
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
            print_result(result)
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
