# 急救 AI 模型训练计划

## 项目概述
基于 Qwen2.5-1.5B-Instruct + LoRA 微调的急救指导模型。
自然语言理解拆为 3 个串行步骤，分别训练 LoRA adapter，部署时按需切换。
统一测试流水线 `test_pipeline.py` 串联 Step1 + Step2a，支持多伤情拆分、NLU 清洗、不确定性驱动追问。

---

## Step 1 — 受伤大类型

### 输出
`开放式` / `闭合式`（可逗号分隔多值）

### 判定规则（test_pipeline.py STEP1_RULE，推理时附加在 prompt 后）
```
核心原则：以皮肤/粘膜是否完整为准
- 开放式：皮肤或粘膜有破损、有伤口、有外出血、有溃烂、有焦痂、有水泡破溃
- 闭合式：皮肤和粘膜完整无损（包括骨折、脱臼、扭伤、内出血、中毒、中暑、溺水、电击、不破皮的烧烫伤、冻伤等）

只输出一个类型词
```

### 定义
| 类型 | 判定标准 | 常见伤情 |
|------|---------|---------|
| **开放式** | 皮肤或粘膜有破损 | 流血、破皮、割伤、烧伤破溃、化学腐蚀溃烂、骨折端戳破皮肤、断肢、咬伤 |
| **闭合式** | 皮肤和粘膜完整无损 | 骨折、脱臼、扭伤、内出血、中毒、中暑、溺水、触电（无伤口）、不破皮的烧烫伤、冻伤（苍白僵硬期） |

### 文件结构
| 文件 | 路径 |
|------|------|
| 训练脚本 | `src/train/step1_train.py` |
| 单步测试脚本 | `src/train/test_step1.py` |
| 统一流水线测试 | `src/train/test_pipeline.py` |
| LoRA 权重 | `models/step1_type_lora/` |
| 训练数据 | `data/step1_nlu/type/type_train_*.json` |

### 训练参数
```
base model:   Qwen2.5-1.5B-Instruct
LoRA r:       8
lora_alpha:   32
target:       q_proj, v_proj
batch_size:   8
max_length:   128
epochs:       8
lr:           2e-4
```

---

### 统一流水线推理架构（test_pipeline.py）

`test_pipeline.py` 是当前主力测试脚本，串联 NLU 清洗 → Step1 分类 → Step2a 分组。
与 `test_step1.py`（单步独立测试，5 层关键词混合流水线）不同，流水线版本采用 **纯 NLU + LoRA** 架构。

#### 整体流程

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ ① NLU 清洗 + 拆分 (base model, 无 adapter)           │
│   纯 Qwen 理解：删脏话/感叹词/时间地点               │
│   伤情用逗号或换行分隔                                │
│   安全网: 清理后无伤情关键词 → 回退原文               │
└─────────────────────────────────────────────────────┘
  │ 对每条伤情
  ▼
┌─────────────────────────────────────────────────────┐
│ ② Step1 LoRA 分类 + 不确定性判断                     │
│   分别算"开放式"和"闭合式"的 forward loss             │
│   margin = |loss_open - loss_closed|                 │
│   margin < 0.5 → 模型犹豫 → 追问                     │
│   高风险伤情（骨折/烧烫/化学/电击）→ 强制追问          │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ ③ 交互式追问补全                                     │
│   追问皮肤状态 → 用户回答 → 追加到伤情描述             │
│   再次分类 → 得到最终 type                            │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ ④ Step2a LoRA 分组（仅开放式）                        │
│   7 组: 机械/热力/电能/化学/辐射/生物/病理             │
│   闭合式 → 跳过分组                                   │
└─────────────────────────────────────────────────────┘
  │
  ▼
  输出每条伤情: {injury, resolved, type, group, margin, losses}
```

#### 关键设计：双模型分离

```python
# 实例 A: 纯 Qwen，永远不挂 adapter，专做 NLU 清洗
nlu_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, ...).to(device)

# 实例 B: 挂 step1 + step2 两个 adapter，专做分类
lora_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, ...).to(device)
lora_model = PeftModel.from_pretrained(lora_model, STEP1_LORA, adapter_name="step1")
lora_model.load_adapter(STEP2_LORA, adapter_name="step2")
lora_model.set_adapter("step1")   # 分类时切换
lora_model.set_adapter("step2")
```

**为什么不能用一个 PeftModel 做所有事**：step1 adapter 是在 `"判断受伤类型：xxx → 开放式/闭合式"` 格式上训练的，adapter 权重把模型分布拉向分类输出。如果用它做 NLU 清洗，模型会输出训练数据里见过的 token（如"闭合式伤口没破皮没流血"重复），造成幻觉。

#### 追问触发机制（双条件）

| 条件 | 说明 |
|------|------|
| `ambiguous=True` | LoRA 的 type margin < 0.5，模型自己犹豫 |
| `must_ask=True` | 高风险伤情，不管模型多自信都要问 |

**MUST_ASK_KEYWORDS**:
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

#### NLU 安全网（防 base model 删丢伤情）

base model 可能在删脏话时把伤情也丢了（如 `"我去你大爷。砍死我了"` → `"我大爷"`）。

```python
# 清理后每条都没伤情关键词，但原文有 → 说明删过了 → 回退原文
if items and not any(_extract_keywords(i) for i in items):
    if _extract_keywords(text):
        items = None  # 触发回退
```

#### NLU 清理 Prompt

```
删除脏话、感叹词、求助语、时间地点，只保留伤病内容，用逗号分隔，只输出清理结果：{text}
```

---

### 单步推理架构（test_step1.py — 保留，独立测试用）

5 层关键词 + NLU 混合流水线，详见 TECHNICAL.md §3.2。
与 `test_pipeline.py` 不同，`test_step1.py` 的追问是 6 种硬编码歧义模板驱动，不是 logprob margin。

---

### 运行命令

```powershell
# 统一流水线测试（当前主力）
cd "D:\Trae CN\Trae Project\First AID Agent"
D:\python\python.exe src\train\test_pipeline.py

# Step1 训练（覆盖旧权重需 --force）
D:\python\python.exe src\train\step1_train.py --dataset type_train_1.json --force

# Step1 单步测试
D:\python\python.exe src\train\test_step1.py

# Step1 批量测试
D:\python\python.exe src\train\test_step1.py --batch test_cases.json
```

---

## Step 1.5 — NLU 文本清洗（adapter 训练）

### 目的
让 base model 的 NLU 清洗能力更稳定可控，替代当前纯 base model + 安全网的方案。
训练一个轻量 adapter，任务是：输入带噪声的报案文本 → 输出清理后用逗号分隔的伤情列表。

### 文件结构
| 文件 | 路径 |
|------|------|
| 训练脚本 | `src/train/nlu_clean_train.py` |
| LoRA 输出 | `models/nlu_clean_lora/` |
| 训练数据 | `data/nlu_clean/train.json` |

### 训练参数
```
与 step2_open_group 完全一致
base model:   Qwen2.5-1.5B-Instruct
LoRA r:       8, alpha: 32, target: q_proj/v_proj
batch_size:   16, epochs: 8, max_length: 128, lr: 2e-4
```

### Prompt 格式
```
<|im_start|>user
清理报案噪声，保留伤情，用逗号分隔：{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

### 训练数据格式
```json
[
  {"input": "卧槽骨折了还烫伤了", "output": "骨折了，烫伤了"},
  {"input": "我刚才在厨房手被菜刀切了一道口子吓死我了同时腿也骨折了", "output": "手被菜刀切伤了，腿也骨折了"},
  {"input": "操他妈动脉劈裂", "output": "动脉劈裂"},
  {"input": "手臂被毒蛇咬了头晕恶心", "output": "手臂被毒蛇咬伤，头晕恶心"},
  {"input": "辐射袭击，我眼睛瞎了", "output": "辐射伤，眼睛失明"}
]
```

### 运行命令
```powershell
cd "D:\Trae CN\Trae Project\First AID Agent"
D:\python\python.exe src\train\nlu_clean_train.py --epochs 8 --batch_size 16 --force
```

---

## Step 2 — 受伤小类型（两阶段：先分组再细分）

### Step2a — 开放式损伤分组（7选1）
- **前提**：Step1 判定为「开放式」
- **7组**：机械性损伤、热力损伤、电能损伤、化学腐蚀、辐射损伤、生物毒素-感染、病理自发性破溃
- **训练数据**：`data/step1_nlu/subtype/group/subtype_group_train_balanced.json`
- **训练脚本**：`src/train/step2_open_group_train.py`
- **LoRA 输出**：`models/step2_open_group_lora/`
- **状态**：✅ 已完成训练（balanced 数据集，1437 条）

#### 训练数据分布（balanced）
| 组别 | 样本数 |
|------|--------|
| 生物毒素-感染 | 300 |
| 化学腐蚀 | 300 |
| 机械性损伤 | 300 |
| 热力损伤 | 218 |
| 病理自发性破溃 | 140 |
| 电能损伤 | 94 |
| 辐射损伤 | 85 |

#### Prompt 格式（精简版）
```
<|im_start|>user
判断受伤组别（机械/热力/电能/化学/辐射/生物/病理）：{伤情描述}<|im_end|>
<|im_start|>assistant
```

#### 运行命令
```powershell
# 训练
D:\python\python.exe src\train\step2_open_group_train.py --dataset subtype_group_train_balanced.json --force

# 交互式测试
D:\python\python.exe src\train\test_step2_group.py
```

---

### Step2b — 开放式损伤具体类型（按组分训）
7 个独立训练脚本，每个只处理该组内的细分类型。通过 `--group` 参数选择：

| group_en | 组别中文名 | 类型数 | LoRA 输出 |
|----------|-----------|--------|-----------|
| `mechanical` | 机械性损伤 | 11 | `step2_open_mechanical_detail_lora` |
| `thermal` | 热力损伤 | 3 | `step2_open_thermal_detail_lora` |
| `electrical` | 电能损伤 | 2 | `step2_open_electrical_detail_lora` |
| `chemical` | 化学腐蚀 | 3 | `step2_open_chemical_detail_lora` |
| `radiation` | 辐射损伤 | 2 | `step2_open_radiation_detail_lora` |
| `bioinfection` | 生物毒素-感染 | 4 | `step2_open_bioinfection_detail_lora` |
| `pathological` | 病理自发性破溃 | 6 | `step2_open_pathological_detail_lora` |

- **训练脚本**：`src/train/step2_detail_train.py --type open --group mechanical`
- **数据目录**：`data/step1_nlu/subtype/detail/{open|closed}/{group_en}/`
- **闭合式细分**：同脚本 `--type closed --group mechanical`，LoRA 命名 `step2_closed_{group}_detail_lora`
- **状态**：🟡 脚本已就位，数据待生成

---

### Step2-Closed — 闭合式小类型（待做）
- **前提**：Step1 判定为「闭合式」
- **候选类型**：骨折、关节脱位/半脱位、扭伤/拉伤、脑震荡、内脏损伤、内出血、挫伤/血肿
- **训练数据**：待生成
- **脚本**：待创建
- **LoRA 输出**：`models/step2_closed_subtype_lora/`

---

## Step 3 — 受伤部位（待做）
- **输出**：头部、颈部、躯干、手臂、手部、腿部、足部、多部位等
- **训练数据**：`data/step1_nlu/location/train.json`（空）
- **脚本**：待创建
- **LoRA 输出**：`models/step3_location_lora/`

---

## 回退指令对照
| 你说 | 意思 |
|------|------|
| 「回退到第一步」 | 训练/测试受伤大类型（type） |
| 「回退到第二步开放式」 | 训练/测试开放式小类型 |
| 「回退到第二步闭合式」 | 训练/测试闭合式小类型 |
| 「回退到第三步」 | 训练/测试受伤部位 |

---

## 训练脚本 GPU 配置规范（已统一）
```python
# 正确做法
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR,
    local_files_only=True,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model = get_peft_model(model, lora_config)
if torch.cuda.is_available():
    model = model.to("cuda")
# TrainingArguments 里 fp16=torch.cuda.is_available()

# 错误做法（旧版）
# device_map="auto" + get_peft_model 冲突
# fp16=False 硬编码
```

## LoRA 训练保护机制
训练脚本默认不覆盖已有 `adapter_config.json` 的目录。
要强制重训加 `--force` 参数。
每次训练后手动备份 `models/backups/stepX_xxx_lora_YYYYMMDD_HHMMSS/`。

---

## 已知问题与排查记录

### Step2 训练速度极慢（13-15 秒/step）
- **根因**：Prompt 模板里塞了 7 个组别的长篇定义，每条样本 258-261 token（打满 max_length=256）
- **修复**：精简 prompt 为一行（如"判断受伤组别（机械/热力/电能/化学/辐射/生物/病理）：{伤情}"），max_length 从 256→128，batch 从 8→16
- **效果**：step 耗时 13-15s → 0.53s，约 25× 加速

### `group_by_length` 参数报错
- **根因**：transformers 5.14.1 删除了这个参数（移到 DataCollator 层面）
- **修复**：从所有训练脚本里去掉该参数

### Step2 烟花/爆炸 → 辐射损伤
- **根因**：AI 生成训练数据时，train_2/3/4.json 中共 551 条烟花/锅炉爆炸样本被错误标为"辐射损伤"（爆炸碎片造成的物理破损应归"机械性损伤"）
- **修复**：批量修正标签 + 合并去重 + 平衡数据集（每类 ≤300 条）
- **教训**：AI 生成训练数据需要人工抽查标签正确性，特别是跨文件复制扩展时容易产生一致性错误

### Adapter 污染 base model 输出（NLU 解析时生成分类训练数据内容）
- **现象**：用挂了 step1 adapter 的 PeftModel 做 NLU 清理，模型输出 `"闭合式伤口没破皮没流血 流血不止 破皮了..."` 重复内容
- **根因**：step1 adapter 是在 `"判断受伤类型：xxx → 开放式/闭合式"` 格式上训练的，adapter 权重把模型分布拉向分类输出，做 NLU 时模型会生成训练数据里见过的 token
- **修复**：用两个独立的 `AutoModelForCausalLM` 实例——实例 A 纯 Qwen 做 NLU，实例 B 挂 adapter 做分类
- **教训**：不同任务的 LoRA adapter 会互相干扰，需要物理隔离模型实例

### Base model 删丢伤情
- **现象**：`"我去你大爷。砍死我了"` → NLU 清理后变成 `"我大爷"`（删脏话时把"砍死我了"也丢了）
- **根因**：1.5B base model 的理解能力有限，删脏话时可能过度
- **修复**：加安全网——清理后无伤情关键词但原文有 → 回退原文
- **计划**：训练 nlu_clean LoRA adapter 来替代纯 base model，数据量约 300-500 条
