# Technical Architecture — First Aid AI NLU System

> **用途**：给另一个 AI 或评审专家快速上手的完整技术文档。读完能立刻理解架构、复现实验、接手开发。

---

## 0. Quick Handoff（另一个 AI 必读）

### 一句话描述
基于 Qwen2.5-1.5B-Instruct + LoRA 微调的**急救伤情三级级联 NLU 系统**，将自然语言描述解析为 `{type, subtype, location}` 结构化输出。

### 已完成 vs 待做
| 模块 | 状态 | LoRA 位置 |
|------|------|-----------|
| Step1: 受伤大类型（开放式/闭合式） | ✅ 可用 | `models/step1_type_lora/` |
| Step1.5: NLU 文本清洗（adapter 训练） | 🟡 脚本就绪，数据待生成 | `models/nlu_clean_lora/` |
| Step2a: 开放式损伤分组（7选1） | ✅ 可用 | `models/step2_open_group_lora/` |
| Step2b: 组内细分类型（7 个独立 LoRA） | 🟡 脚本就绪，数据待生成 | `models/step2_open_*_lora/` |
| Step2-Closed: 闭合式小类型 | ❌ 未开始 | — |
| Step3: 受伤部位 | ❌ 未开始 | — |
| 统一调度器 | ❌ 未开始 | — |

### 关键文件
| 文件 | 作用 |
|------|------|
| `src/train/step1_train.py` | 训练 Step1 类型分类 LoRA |
| `src/train/test_step1.py` | Step1 单步推理（旧版 5 层关键词混合流水线） |
| `src/train/test_pipeline.py` | **当前主力** — Step1 + Step2a 串联推理（双模型分离 + logprob margin 追问 + NLU 安全网） |
| `src/train/nlu_clean_train.py` | 训练 NLU 文本清洗 LoRA（纯指令格式，无示例） |
| `src/train/step2_open_group_train.py` | 训练 Step2a 分组器 |
| `src/train/test_step2_group.py` | Step2a 单步推理 |
| `src/train/step2_open_detail_train.py` | 训练 Step2b 细分（按 --group 参数） |
| `data/step1_nlu/type/` | Step1 训练数据 |
| `data/step1_nlu/subtype/group/` | Step2a 训练数据 |
| `data/step1_nlu/subtype/detail/{open|closed}/{group_en}/` | Step2b 按 type + group 分目录 |
| `data/nlu_clean/` | NLU 清洗训练数据 |

### 复现命令
```powershell
# 统一流水线测试（Step1 + Step2a 串联，含追问）— 当前主力
cd "D:\Trae CN\Trae Project\First AID Agent"
D:\python\python.exe src\train\test_pipeline.py

# Step1 训练
D:\python\python.exe src\train\step1_train.py --dataset type_train_1.json --force

# Step1 单步测试（旧版）
D:\python\python.exe src\train\test_step1.py

# NLU 清洗 adapter 训练
D:\python\python.exe src\train\nlu_clean_train.py --epochs 8 --batch_size 16 --force

# Step2a 训练
D:\python\python.exe src\train\step2_open_group_train.py --dataset subtype_group_train_balanced.json --force

# Step2a 测试
D:\python\python.exe src\train\test_step2_group.py
```

### 环境
```
Python:     D:\python\python.exe
PyTorch:    2.x (CUDA 13.2)
Transformers: 5.14.1
PEFT:       latest
GPU:        NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
Base Model: Qwen/Qwen2.5-1.5B-Instruct (本地缓存)
```

---

## 1. Problem Statement

### 场景
急救分诊场景中，调度员接收自然语言伤情描述（如"被车撞了腿摔断了骨头戳出来流血了"），需要快速确定：
1. 伤口是否开放（决定是否需要立即止血）
2. 损伤机制（决定后续处理方向）
3. 受伤部位（决定检查优先级）

### 通用大模型的缺陷
| 问题 | 示例 |
|------|------|
| **输出不稳定** | 同一句输入，两次可能输出不同的标签 |
| **不会主动追问** | 输入"腿骨折了"（没说皮肤状态），模型总会瞎猜一个 |
| **标签体系混乱** | 可能输出"外伤"、"伤口"、"破皮"等非标准词 |
| **无法处理复合伤情** | "手割破了同时腿骨折" → 只能输出一个 |
| **医学常识错误** | 开放性骨折和闭合性骨折互斥，但模型可能同时输出 |

### 本系统的解决方案
**级联架构 + NLU/关键词混合推理 + 主动信息索取**

---

## 2. Core Architecture

### 三级级联流水线（Cascaded Pipeline）

```
自然语言输入
  │
  ▼
┌───────────────────────────────────────────────────────┐
│ Pre — NLU 清洗 + 多伤情拆分 (纯 base model, 无 adapter) │
│ Prompt: "删除脏话、感叹词、求助语、时间地点，只保留伤病" │
│         "内容，用逗号分隔，只输出清理结果：{text}"       │
│ 安全网: 清理后无伤情关键词但原文有 → 回退原文             │
│ 备选: 训练 nlu_clean LoRA 替换纯 base model（待做）       │
└───────────────────────────────────────────────────────┘
  │ 对每条伤情
  ▼
┌───────────────────────────────────────────────────────┐
│ Step 1 — 受伤大类型分类器 (LoRA #1: step1_type_lora)    │
│ 输出: "开放式" / "闭合式"                               │
│ 技术:                                                 │
│   · logprob margin 不确定性检测（见 §3.2 核心算法）      │
│   · 不确定性驱动追问（骨折/烧烫/化学/电击 → 强制追问）   │
│   · 用户回答补全皮肤状态 → 再次分类                      │
└───────────────────────────────────────────────────────┘
  │ 输出记为 type ∈ {open, closed} + 补全后的伤情描述
  ▼
┌───────────────────────────────────────────────────────┐
│ Step 2a — 损伤组别分类器 (LoRA #2: step2_open_group_lora)│
│ 前提: type == "开放式"（闭合式直接跳过）                 │
│ 输出: "机械性损伤"/"热力损伤"/... (7选1)                  │
│ 技术: 精简 Prompt 纯 NLU 分类（43 token/sample, 0.53s/step）│
└───────────────────────────────────────────────────────┘
  │ 输出记为 group
  ▼
┌───────────────────────────────────────────────────────┐
│ Step 2b — 组内细分 (7 个独立 LoRA, 按 type 分支)         │
│ LoRA 选择: step2_open_{group}_lora / step2_closed_*_lora│
│ 输入: (补全后的伤情描述) 单参数                          │
│ 输出: 具体损伤类型（如"切割伤"、"闭合性骨折"）            │
│ 关键: 同一 group 在 open/closed 下类型完全不同            │
└───────────────────────────────────────────────────────┘
  │
  ▼
┌───────────────────────────────────────────────────────┐
│ Step 3 — 部位识别 (LoRA, 待做)                          │
│ 输出: ["手臂"] / ["腿部","头部"]                        │
└───────────────────────────────────────────────────────┘
  │
  ▼
最终 JSON:
{
  "original": "卧槽骨折了烫伤",
  "injuries": [
    {"injury": "骨折", "resolved": "骨折 皮肤粘膜有破损",
     "type": "开放式", "group": "机械性损伤", "margin": 0.97},
    {"injury": "烫伤", "resolved": "烫伤 皮肤粘膜有破损",
     "type": "开放式", "group": "热力损伤", "margin": 1.23}
  ]
}
```

### 双模型分离架构（关键设计）

推理时需要**两个独立的 `AutoModelForCausalLM` 实例**，不能用一个 PeftModel 同时做 NLU 清洗和分类：

```
实例 A: 纯 Qwen base model（永远不挂 adapter）
  └─ 专做 NLU 清洗：删脏话、拆多伤情
  └─ 为什么纯 base？adapter 权重会拉偏模型分布，
     导致清洗时输出训练数据里见过的 token（如"闭合式伤口没破皮没流血"重复）

实例 B: PeftModel，同时挂 step1 + step2 两个 adapter
  └─ 用 set_adapter("step1") 切换 → 做类型分类 + 追问判断
  └─ 用 set_adapter("step2") 切换 → 做组别分类
```

### 新旧两套流水线并存

| 对比项 | `test_step1.py`（旧版） | `test_pipeline.py`（当前主力） |
|--------|------------------------|-------------------------------|
| 拆分方式 | 5 层关键词混合流水线 | 纯 NLU base model 清洗 |
| 追问触发 | 6 种硬编码歧义模板 | logprob margin + 高风险关键词双条件 |
| 分类方式 | 单 adapter + 关键词兜底 | 双 adapter 动态切换 + loss 比较 |
| 多伤情 | 支持，但按关键词边界切 | 支持，NLU 按逗号/换行切 |
| 追问粒度 | 全句一次追问 | 每条伤情独立追问 |
| 用途 | 单步独立测试 | 串联 Step1 + Step2a 的完整流程 |

### 为什么用级联而不是单一模型

| 方案 | 优点 | 缺点 |
|------|------|------|
| **单一多标签模型** | 只需一次推理 | 标签空间太大（type×subtype×location），数据量指数增长，模型容易混淆边界 |
| **级联多模型** ✅ | 每步标签空间小（2 / 7 / 3-11 / 8），精度更高；可独立升级每步；可以按需裁剪 | 需要多次推理 |

### 为什么用 LoRA 而不是全量微调

| 对比项 | 全量微调 | LoRA |
|--------|---------|------|
| 可训练参数量 | 1.5B (100%) | ~1M (0.07%) |
| 显存占用 | ~6GB | ~3GB |
| 训练速度 | 慢 | 快 |
| 部署灵活性 | 需要存一份完整模型 | 只需存 adapter 权重 (4-70MB) |
| 多任务扩展 | 每个任务一份完整模型 | 每个任务一个 adapter，base model 共享 |

**关键**：1.5B 模型 + 多个 LoRA adapter 可以在 8GB 显存上共存，部署时按需切换 adapter。

---

## 3. Technology Deep Dive

### 3.1 LoRA (Low-Rank Adaptation)

**原理**：在 Transformer 的注意力投影层（$W_q, W_v$）旁插入两个低秩矩阵 $A$ 和 $B$，训练 $\Delta W = BA$ 而不是原始权重：

```
原始: h = W_q x
LoRA: h = W_q x + (BA) x
         ↑ 冻结      ↑ 训练 (rank r=8, alpha=32)
```

**本系统配置**：
```python
LoraConfig(
    r=8,                  # 低秩维度
    lora_alpha=32,        # 缩放因子，effective_lr = lr * alpha/r
    target_modules=["q_proj", "v_proj"],  # 只训练注意力的 Q 和 V 投影
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**为什么只选 q_proj 和 v_proj**：
- 全量 7 个模块（q/k/v/o/gate/up/down）→ LoRA 权重 70MB
- 只选 q/v → LoRA 权重 4MB
- 二分类/七分类任务不需要模型生成能力，只需要注意力模式的微调
- 经验证效果不降，速度提升显著

### 3.2 Step1 混合推理流水线（核心创新）

当前主力实现在 `test_pipeline.py`，采用 **双模型分离 + logprob margin 追问** 架构。
旧版 `test_step1.py` 的 5 层关键词流水线仍保留作独立单步测试用。

#### 3.2.1 Logprob Margin 不确定性检测（核心算法）

**问题**：纯 NLU 分类无法判断"什么时候该追问"。模型要么盲目输出，要么需要关键词触发。

**解决方案**：利用 Causal LM 的 loss 值作为模型自信度指标。对同一输入，分别计算模型生成"开放式"和"闭合式"的 forward loss：

```python
def _type_forward_loss(model, tokenizer, text, target_type):
    prompt = f'判断受伤类型：{text}\n{STEP1_RULE}'
    # 跑一次 forward，计算 prompt + target_type 的 loss
    # loss 越低 = 模型越倾向于这个类型
    full_ids = cat([prompt_ids, target_ids])
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100  # 只对 target 部分计算 loss
    outputs = model(full_ids, labels=labels)
    return outputs.loss.item()
```

**关键公式**：
```
loss_open   = forward_loss(prompt + "开放式")
loss_closed = forward_loss(prompt + "闭合式")

predicted = argmin(loss_open, loss_closed)
margin    = |loss_open - loss_closed|   # 两个 loss 的差距

margin 小 → 模型犹豫 → 追问
margin 大 → 模型自信 → 直接输出
```

**实测数据**（来自 `test_pipeline.py` 运行日志）：
```
"骨折"            → loss_open=0.053, loss_closed=0.234  → margin=0.181 → 追问（模型不确定）
"手擦破皮"         → loss_open=0.006, loss_closed=1.778  → margin=1.772 → 自信（开放式）
"脚扭伤"           → loss_open=8.094, loss_closed=0.0    → margin=8.094 → 自信（闭合式）
"辐射袭击"         → loss_open=0.053, loss_closed=1.023  → margin=0.97  → 自信（开放式）
```

**阈值选择**：`TYPE_AMBIGUITY_THRESHOLD = 0.5`。margin < 0.5 时模型犹豫，触发追问。

#### 3.2.2 追问触发双条件

即使模型很自信，某些伤情的皮肤状态直接决定类型，必须追问：

| 条件 | 说明 | 示例 |
|------|------|------|
| `ambiguous=True` | margin < 0.5，模型自己犹豫 | "骨折"（margin=0.181） |
| `must_ask=True` | 高风险伤情，不管模型多自信都要问 | "骨折"、"烧伤"、"化学腐蚀"、"电击" |

**MUST_ASK_KEYWORDS**：
```python
['骨折', '骨头断', '骨头裂', '脱臼', '扭伤',
 '烧', '烫', '冻伤', '冻',
 '化学', '腐蚀', '酸碱',
 '电击', '电伤', '触电']
```

**追问模板**（按伤情类型定制）：
```
骨折/断 → "[骨折] 皮肤有没有破？骨头有没有戳出来？（有/没有）"
烧/烫   → "[烫伤] 皮肤有没有破损、水泡或发黑？（有/没有）"
冻伤    → "[冻伤] 皮肤有没有破皮、溃烂或发黑？（有/没有）"
化学    → "[化学腐蚀] 皮肤有没有溃烂、水泡或发黑？（有=开放式/没有=闭合式）"
电击    → "[电击] 皮肤有没有伤口或焦痂？（有/没有）"
其他    → "[伤情] 皮肤有没有破损？（有=开放式/没有=闭合式）"
```

**回答解析**（_append_skin_status）：
```
有/是/对/破了/流血/骨头戳 → 追加 " 皮肤粘膜有破损"
没有/无/没/没破/完好       → 追加 " 皮肤粘膜完整无损"
跳过                      → 不追加，用原分类结果
```

**追问流程**：
```
1. 初始分类（无皮肤信息）→ 得到 pred_type + margin
2. ambiguous OR must_ask?
   ├─ 是 → 弹出追问 → 用户回答 → 补全伤情描述
   │         → 用补全后的文本再次分类 → 得到最终 pred_type
   └─ 否 → 直接用初始分类结果
```

#### 3.2.3 NLU 安全网（防 base model 删丢伤情）

纯 base model 做 NLU 清洗时，可能在删脏话时把伤情也删了：
```
输入:    "我去你大爷。砍死我了"
NLU输出: "我大爷"   ← 把"砍死我了"丢了！
```

**安全网机制**：
```python
def parse_injuries(nlu_model, tokenizer, text):
    items = _nlu_parse(nlu_model, tokenizer, text)

    # 条件1: NLU 有输出
    # 条件2: 清理后每条都没伤情关键词
    # 条件3: 原文有伤情关键词
    # → 说明删过了 → 回退原文
    if items and not any(_extract_keywords(i) for i in items):
        if _extract_keywords(text):
            items = None  # 触发回退

    if not items:
        items = [text]  # 回退原文（或 NLU 失败时）
    return [{"injury": s} for s in items]
```

**安全网只做回退，不做修正**——回退到原文后，追问和分类会在原文上继续执行，不会丢信息。

#### 3.2.4 动态 Adapter 切换

推理时一个 PeftModel 同时挂 step1 和 step2 两个 adapter：

```python
# 加载两个 adapter
lora_model = PeftModel.from_pretrained(base, STEP1_LORA, adapter_name="step1")
lora_model.load_adapter(STEP2_LORA, adapter_name="step2")

# 分类时切换
lora_model.set_adapter("step1")   # 做 Step1 类型分类
pred_type, margin, _, losses = predict_type_with_confidence(lora_model, ...)

# 组别分类时切换
lora_model.set_adapter("step2")   # 做 Step2a 组别分类
group = predict_group(lora_model, ...)
```

**为什么用 `set_adapter` 而不是手动 merge**：`set_adapter` 是零开销的 Python 属性切换，不涉及模型权重合并/拷贝，切换延迟 < 1ms。

#### 3.2.5 NLU 解析 — 多伤情拆分

纯 base model 做 NLU 清洗时，输出可能有多种格式，解析逻辑需要容错：

```python
def _nlu_parse(nlu_model, tokenizer, text):
    raw = _generate(nlu_model, tokenizer, CLEAN_PROMPT)

    # 1. 先按换行拆
    # 2. 去掉 "1." "1、" 等编号前缀
    # 3. 行内按逗号分
    # 4. 去重（保持顺序）
    # 5. 长度过滤（≥2 字符）

    # 兜底：NLU 完全失败时，用连词/关键词边界拆分
    if not items:
        items = _fallback_split(text)  # 连词 + 伤情关键词边界
```

**Prompt 设计约束**：
```
不能给示例 → 防止幻觉（模型会照着示例格式输出无关内容）
不能说 JSON → base model 可能输出 markdown code block
纯指令: "删除脏话、感叹词、求助语、时间地点，只保留伤病内容，用逗号分隔，只输出清理结果：{text}"
```

---

#### 旧版：5 层关键词混合流水线（test_step1.py，遗留参考）

以下是 `test_step1.py` 的架构，保留作单步独立测试。与 `test_pipeline.py` 相比：
- 追问用 6 种硬编码模板（不是 logprob margin）
- 拆分用连词正则 + 关键词检测（不是 NLU 清洗）
- 分类后有关键词兜底和互斥消歧（`test_pipeline.py` 暂未迁移这两层）

```
输入文本
  │
  ├─ Layer 1: 连词拆分 (split_input)
  │   正则匹配连词切分多伤情
  │   "手割破了同时腿骨折" → ["手割破了", "腿骨折"]
  │
  │   对每个子短语：
  │   ├─ Layer 2: 歧义检测 (check_ambiguity)
  │   │   6 种歧义模板（骨折/断肢/烧烫伤/化学/冻伤/电击）
  │   │   → 判断信息是否充足
  │   │   → 不足则主动追问
  │   │   → 用户回答后 resolve_ambiguity() 补全文本
  │   │
  │   ├─ Layer 3: NLU 分类 (predict_single)
  │   │   LoRA 模型推理 → "开放式" / "闭合式" / None
  │   │
  │   └─ Layer 4: 关键词兜底 (post_process_type)
  │       NLU 输出异常时按皮肤关键词硬判
  │
  └─ Layer 5: 互斥消歧 (resolve_mutual_exclusion)
      医学硬约束：开放型骨折 ≠ 闭合型骨折
```

#### Layer 1: 连词拆分（旧版）

**算法**：正则 `[，,。；;、和加伴有伴还有同时此外另外然后接着]` 切分 + 二次检测。

如果某段同时命中 OPEN_KEYWORDS 和 CLOSED_KEYWORDS，生成一个"去掉开放关键词"的版本，两句分别送入 NLU。

**复杂度**：O(n)，n 为输入长度。

#### Layer 2: 歧义检测与主动追问（旧版）

**算法**：模板匹配 + 状态机。

每个歧义模板定义：
```python
{
    "triggers": ["骨折", "骨头断", ...],      # 触发词
    "skin_positive": ["破皮", "流血", ...],   # 皮肤破损信号
    "skin_negative": ["没破皮", ...],          # 皮肤完好信号
    "question": "皮肤有没有被骨头戳破？",        # 一级追问
    "followup_question": "骨头戳的还是擦伤？",   # 二级追问（仅骨折）
    "bone_pierce_keywords": ["骨头戳", ...],    # 确认骨头穿透的关键词
}
```

**流程**：
```
triggers 命中？
  ├─ 是 → skin_positive 或 skin_negative 是否提及？
  │       ├─ 都没提 → 触发一级追问
  │       ├─ 有 skin_positive 但没 bone_pierce → 触发二级追问（仅骨折模板）
  │       └─ 有明确皮肤信息 → 跳过追问
  └─ 否 → 所有模板都没命中？
          ├─ 是 且 文本没提皮肤状态 → 通用追问
          └─ 否 → 跳过追问
```

#### Layer 3: NLU 分类（旧版）

**Prompt 设计决策**：精简 vs 冗长（已迁移到 test_pipeline.py）。

#### Layer 4: 关键词兜底（旧版）

**算法**：优先级匹配。

```
模型输出不在 ["开放式", "闭合式"]？
  ├─ 文本含皮肤破损关键词 → "开放式"
  ├─ 文本含皮肤完好关键词 → "闭合式"
  ├─ 文本含非机械损伤关键词 → "闭合式"（中毒/中暑/溺水等默认皮肤完好）
  └─ 都没匹配 → "未知"
```

#### Layer 5: 互斥消歧（旧版）

**医学约束编码**：
```
if (骨头戳破/骨头露/骨头顶) AND (骨折/断了/折了/裂了):
    结果 = 结果 - {"闭合式"}    # 开放性骨折绝对不是闭合性
if (骨折) AND (破皮/流血/皮肤粘膜有破损):
    结果 = 结果 - {"闭合式"}
```

**为什么代码硬编码而不是训练**：医学互斥规则是**确定性逻辑**，概率模型无法保证 100% 不违反。

### 3.3 Step2 两阶段分类

**架构决策**：先分组再细分，而不是直接做 30+ 类的多标签分类。

| 方案 | 问题 |
|------|------|
| 直接 30 类 | 爆炸碎片伤 vs 火器伤 vs 开放性骨折，边界模糊，需要大量样本区分 |
| **先组后分** ✅ | 第一步判断"机械性损伤还是热力损伤"（因果机制差异大），第二步在组内做细分（同机制下的差异） |

7 组的因果机制定义：
```
机械性损伤 ← 外力物理作用（切/割/砸/爆炸碎片/骨折端刺破）
热力损伤  ← 温度极端（火焰/蒸汽/低温冻伤）
电能损伤  ← 电流/电弧（电烧伤/电弧烧伤）
化学腐蚀  ← 化学反应（酸/碱/腐蚀剂）
辐射损伤  ← 能量辐射（放疗/紫外线/电光性眼炎）
生物毒素  ← 生物因素（动物咬/毒虫/感染）
病理破溃  ← 内部病变（压疮/糖尿病足/肿瘤破溃）
```

---

## 4. Training Methodology

### 4.1 训练参数（已统一优化）

| 参数 | Step1 (type) | Step2a (group) | Step2b (detail) |
|------|-------------|----------------|-----------------|
| base model | Qwen2.5-1.5B-Instruct | 同左 | 同左 |
| LoRA r | 8 | 8 | 8 |
| lora_alpha | 32 | 32 | 32 |
| target_modules | q_proj, v_proj | q_proj, v_proj | q_proj, v_proj |
| batch_size | 8 | 16 | 16 |
| max_length | 128 | 128 | 128 |
| epochs | 8 | 8 | 8 |
| lr | 2e-4 | 2e-4 | 2e-4 |
| fp16 | ✅ | ✅ | ✅ |
| LoRA 权重大小 | ~4 MB | ~4 MB | ~4 MB |

**显存计算**：
```
Base model fp16:   1.5B × 2 bytes = 3 GB
LoRA params:       ~1M × 2 bytes = 2 MB (negligible)
KV cache (128 tok): batch × max_len × layers × dim × 2 bytes
                   = 16 × 128 × 28 × 1536 × 2 ≈ 168 MB
Optimizer states:  ~4 MB × 2 (AdamW) = 8 MB
Total:             ~3.2 GB → 8GB 显卡够用
```

### 4.2 数据构造

**数据格式**：
```json
[
  {"input": "手指被刀切了流血", "output": "开放式"},
  {"input": "脚崴了肿了", "output": "闭合式"}
]
```

**Step2a balanced 数据分布**：
```
生物毒素-感染: 300 (原始 2084, 下采样)
化学腐蚀:     300 (原始 304, 下采样)
机械性损伤:   300 (原始 563, 下采样)
热力损伤:     218 (全部保留)
病理自发性破溃: 140 (全部保留)
电能损伤:      94 (全部保留)
辐射损伤:      85 (全部保留)
总计:        1437
```

**类别不平衡问题**：原始数据生物毒素-感染占 60%（2084/3488），少数类被淹没。采用 downsampling 多数类 + 全保留少数类的策略，每类 ≤300 条。

### 4.3 数据质量控制

**踩过的坑**：AI 生成的训练数据存在跨文件标签不一致。

```
train_1.json: 烟花爆炸 → 机械性损伤 ✅ (11 条)
train_2.json: 烟花爆炸 → 辐射损伤   ❌ (50 条)
train_3.json: 烟花爆炸 → 辐射损伤   ❌ (52 条)
train_4.json: 烟花爆炸 → 辐射损伤   ❌ (449 条)
```

**根因**：爆炸碎片造成的是物理破损（机械性损伤），但 AI 生成数据时把"爆炸"关联到"核辐射/电离辐射"。跨文件复制扩展数据时没有重新校验标签。

**修复流程**：
```
1. 关键词匹配定位错误样本
   （烟花/爆炸/爆竹/鞭炮/炸药/锅炉/煤气罐 + 辐射损伤标签）
2. 批量改标签为"机械性损伤"
3. 人工抽查辐射损伤样本不被误改
   （放疗/紫外线/电光性灼伤 → 这些才是真辐射）
4. 合并去重 + 平衡
5. 删除 _formatted.json 缓存，让训练脚本重新生成
```

**教训**：AI 生成训练数据后必须做 **一致性校验**——同一 input 在不同文件里的 output 必须相同；关键词与标签的对应关系必须符合业务逻辑。

---

## 5. Critical Bugs & Solutions

### Bug 1: `ValueError: Can't find 'adapter_config.json'`
**现象**：加载 LoRA 时报 adapter_config.json 不存在。
**根因**：训练脚本 `--force` 清空了输出目录但训练中途失败，没有生成新权重。
**修复**：训练完成后再加载；或手动检查目录内容。

### Bug 2: GPU 利用率接近 0%
**现象**：`nvidia-smi` 显示 GPU 几乎空闲。
**根因**：`device_map="auto"` 与 `PeftModel.from_pretrained()` 冲突，模型被放在 CPU 上。
**修复**：显式 `model = model.to("cuda")`；TrainingArguments 设 `fp16=True`。

### Bug 3: `group_by_length` 参数报错
**现象**：`TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'group_by_length'`
**根因**：transformers 5.14.1 删除了此参数（移到 DataCollator 层面）。
**修复**：从所有训练脚本中移除该参数。

### Bug 4: 训练速度 13-15 秒/step
**现象**：step2 训练极慢。
**根因**：Prompt 模板里塞了 7 组别的长篇定义，每条样本打满 max_length=256。
**修复**：精简 prompt 为一行，max_length 256→128，batch 8→16。
**效果**：25× 加速。

### Bug 5: `bitsandbytes` CUDA 库缺失
**现象**：`RuntimeError: Configured CUDA binary not found at ... libbitsandbytes_cuda132.dll`
**根因**：bitsandbytes 没正确安装 CUDA 版本。
**影响**：不影响 fp16 训练（本系统用 fp16 不用 4bit 量化），只影响 flash attention 等可选优化。
**修复**：忽略此警告，系统正常运行。

### Bug 6: 标签错误导致分类错误（烟花→辐射）
**现象**：模型把烟花爆炸归类为辐射损伤。
**根因**：训练数据中 551 条烟花/爆炸样本被错误标注为"辐射损伤"。
**修复**：批量修正 + 重新训练。

### Bug 7: Adapter 污染 base model 输出
**现象**：用挂了 step1 adapter 的 PeftModel 做 NLU 清洗，模型输出 `"闭合式伤口没破皮没流血 流血不止 破皮了..."` 重复内容。
**根因**：step1 adapter 是在 `"判断受伤类型：xxx → 开放式/闭合式"` 格式上训练的，adapter 权重把模型分布拉向分类输出。做 NLU 清洗时，模型会"幻觉"出训练数据里见过的 token。
**修复**：双模型分离——两个独立的 `AutoModelForCausalLM` 实例。实例 A 纯 Qwen 做 NLU，实例 B 挂 adapter 做分类。
**教训**：不同任务的 LoRA adapter 会互相干扰，需要物理隔离模型实例。

### Bug 8: Base model 删丢伤情
**现象**：`"我去你大爷。砍死我了"` → NLU 清理后变成 `"我大爷"`（删脏话时把"砍死我了"也丢了）。
**根因**：1.5B base model 的理解能力有限，删脏话时可能过度删除相邻的伤情内容。
**修复**：加安全网——清理后所有条目都没伤情关键词但原文有时，回退到原文。
**教训**：base model 做 NLU 清洗需要安全网兜底。长期方案是训练专用的 nlu_clean LoRA adapter。

---

## 6. Performance Baseline

### Step1 (type)
| 指标 | 值 |
|------|-----|
| 训练样本 | ~1000 条 |
| LoRA 参数量 | ~1M |
| LoRA 权重 | 4 MB |
| 训练时间 | ~15 min |
| 训练集拟合率 | 97%+ |
| 待验证 | 测试集准确率 |

### Step2a (group)
| 指标 | 值 |
|------|-----|
| 训练样本 | 1437 条（balanced） |
| LoRA 参数量 | ~1M |
| LoRA 权重 | 4 MB |
| 训练时间 | ~10 min |
| 训练集拟合率 | 97%（修复标签错误后重训） |
| 待验证 | 测试集准确率 |

### 推理延迟（单次）
```
NLU 清洗 (base model): ~80ms
Step1 类型分类 (2× forward loss): ~200ms
Step2a 组别分类:        ~100ms
Step1 + Step2a 串行:   ~380ms（含 NLU 清洗）
追问延迟:               +用户思考时间
```

---

## 7. Design Decisions & Trade-offs

### Q: 为什么不用更大的模型（7B/14B）？
A: 显存限制（8GB）+ 推理延迟要求。1.5B + LoRA 在 8GB 上流畅运行，精度足够分类任务。

### Q: 为什么不做 4bit 量化？
A: 二分类/七分类任务对精度敏感，4bit 可能损失关键区分能力。fp16 已经够用。

### Q: 为什么关键词不全部用 NLU 替代？
A: 4 个原因——
1. 拆分：连词组合无限多，无法穷举进训练集（但用 base model NLU 清洗替代了关键词拆分，见 test_pipeline.py）
2. 追问：NLU 无法判断"信息够不够"，追问是主动索取不是被动分类。我们用 logprob margin 量化模型自信度，这是概率模型的内在属性
3. 兜底：模型偶尔幻觉，关键词/安全网是最后防线
4. 互斥：医学硬约束，概率模型无法保证 100% 不违反

### Q: 为什么需要双模型分离（base model + PeftModel）？
A: LoRA adapter 会拉偏模型的 token 分布。如果用分类 adapter 做 NLU 清洗，模型会"幻觉"出训练数据里的 token（如"闭合式伤口没破皮没流血"重复）。纯 base model 不受 adapter 干扰，清洗结果纯净。代价是多占 3GB 显存（两个 fp16 模型）。

### Q: 追问为什么用 logprob margin 而不是关键词？
A: 关键词只能触发，不能量化。"骨折"这个关键词总是触发追问，但"手擦破皮"不触发——模型实际上在某些歧义场景下即使没有高风险关键词也会犹豫（margin 小）。logprob margin 是**模型自身的不确定感**，比人工枚举的关键词覆盖更全面。

### Q: 为什么 Step2 先分组再细分，不直接 30 分类？
A: 边界模糊的类别（如"爆炸碎片伤"vs"开放性骨折"）直接分类准确率低。先按因果机制分组（差异大），再组内细分（差异小），更符合人类诊断思路。

### Q: 训练数据自己生成还是用公开数据集？
A: 自己生成。公开医学数据集（如 MedQA）是英文 + 诊断级，不适合急救场景的粗粒度分类。

---

## 8. Next Steps

| 优先级 | 任务 | 依赖 |
|--------|------|------|
| P0 | 生成 NLU 清洗训练数据（300-500 条）并训练 nlu_clean LoRA | AI 生成 + 人工校验 |
| P0 | 生成 Step2b 7 组细分训练数据 | 需要 AI 生成 + 人工校验 |
| P0 | 训练 7 个 Step2b LoRA | 数据就绪 |
| P1 | 将 test_step1.py 的互斥消歧/关键词兜底迁移到 test_pipeline.py | 无 |
| P1 | 设计 Step3 部位标签体系 | 无 |
| P1 | 生成 Step3 训练数据 | 无 |
| P2 | 编写统一调度器 | Step1+Step2 全部就绪 |
| P2 | 写测试集评估整体准确率 | 调度器就绪 |
| P3 | 探索 LoRA 合并（多 adapter 权重合并为一个） | 研究可行性 |

---

## 9. 给评审的 Elevator Pitch

> 我们做了一个**急救分诊 NLU 系统**，核心技术是**三级级联架构** + **双模型分离 + logprob margin 不确定性追问**。
>
> 为什么要这么设计？纯大模型在急救场景有三个致命问题：
> 1. **不会主动追问**——用户说"腿骨折了"（没说皮肤状态），模型总会瞎猜一个
> 2. **会犯医学常识错误**——开放性骨折和闭合性骨折互斥，但模型可能同时输出
> 3. **输出不稳定**——同一句输入可能输出不同标签
>
> 我们的解决方案：
> - **三级级联**：先分开放/闭合（2 类），再分 7 组损伤机制，最后组内细分。每步标签空间小，精度更高
> - **Logprob margin 追问**：对"开放式"和"闭合式"分别算 forward loss，margin 小 = 模型犹豫 → 主动追问用户皮肤状态。这是概率模型的内在属性，比关键词覆盖更全面
> - **双模型分离**：纯 base model 做 NLU 清洗（去脏话、拆多伤情），挂 LoRA 的 PeftModel 做分类。防止 adapter 权重污染 NLU 输出
> - **医学硬约束**：开放性骨折绝对不是闭合性，代码硬编码互斥消歧，概率模型无法保证 100% 不违反
>
> 训练上：级联架构 + 精简 Prompt（25× 加速）+ 数据平衡（解决类别不平衡）。每个 LoRA adapter 只有 4MB。
>
> 目前已完成 Step1（开放/闭合 + 追问）和 Step2a（7 组损伤机制），开发阶段用 RTX 5060 训练和调试，目标部署平台是 Orange Pi 5 Pro 16GB（全离线运行，预计推理延迟 1-2s）。统一流水线 `test_pipeline.py` 支持交互式对话测试，实时反馈模型自信度（margin）和追问判断。
