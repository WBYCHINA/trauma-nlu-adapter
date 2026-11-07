# 急救伤情理解 —— Orange Pi 5 Pro 部署说明

## 一、硬件
- Orange Pi 5 Pro（8GB 内存）
- 无 NVIDIA GPU，纯 CPU 推理

## 二、目录结构
```
deploy/
├── pi_inference.py      # 推理主程序
├── models/              # 5 个 Q4_K_M GGUF 模型（每个约 940MB）
│   ├── base_clean-q4_k_m.gguf
│   ├── step1_type-q4_k_m.gguf
│   ├── step2_group-q4_k_m.gguf
│   ├── step2_detail-q4_k_m.gguf
│   └── step3_location-q4_k_m.gguf
├── install.sh           # 一键安装
├── run.sh               # 运行脚本
└── requirements.txt
```

## 三、安装
```bash
cd deploy
bash install.sh
```
`install.sh` 会：
1. 安装 `python3-pip build-essential cmake`
2. 升级 pip
3. 用 CPU 配置编译安装 `llama-cpp-python`（约 5-10 分钟）

> 如果 install.sh 失败，手动执行：
> ```bash
> sudo apt-get install -y python3-pip build-essential cmake
> CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_BLAS=OFF" python3 -m pip install numpy llama-cpp-python
> ```

## 四、运行
```bash
# 单句测试
bash run.sh --text "被刀割伤直流血"

# 交互式
bash run.sh
```

## 五、作为 HTTP API 服务运行（供其他模型调用）
```bash
# 默认监听 0.0.0.0:8080
bash api.sh

# 自定义端口
bash api.sh 9000
```

### 接口
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查，返回 `{"status":"ok"}` |
| POST | `/classify` | 伤情分类 |

### 调用示例
**Python：**
```python
import requests
r = requests.post("http://<pi_ip>:8080/classify",
                  json={"text": "被刀割伤直流血"})
print(r.json())
```

**curl：**
```bash
curl -X POST http://<pi_ip>:8080/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"被刀割伤直流血"}'
```

### 返回示例
```json
{
  "original": "被刀割伤直流血",
  "injuries": [{
    "idx": 1,
    "injury": "被刀割伤直流血",
    "type": "开放式",
    "group": "机械性损伤",
    "detail": "切割伤",
    "location": "躯干"
  }]
}
```

## 六、与 Ollama（另一个模型）对接
本系统是 5 个 GGUF 模型级联的分类流水线，**Ollama 无法直接运行**（Ollama 只能跑单个模型做对话）。
正确架构：**本系统用 `api.sh` 跑成 HTTP 服务，Ollama 上的主模型通过 HTTP 调用它。**

1. Orange Pi 上启动急救分类 API：
   ```bash
   cd ~/deploy && bash api.sh
   ```
2. 在跑 Ollama 的机器上，让主对话模型具备调用本接口的能力（function calling 或工具调用）：
   - 工具名：`injury_classify`
   - 工具描述：对急救报案文本进行伤情分类，返回开放式/闭合式、受伤原因、细分类型、部位。
   - 实现：`POST http://<pi_ip>:8080/classify`，body `{"text":"<报案原文>"}`
3. 主模型拿到返回的结构化 JSON 后，结合知识库生成急救处置建议。

## 七、输出示例
```
  原始: 被刀割伤直流血
    伤情1: 被刀割伤直流血
          type=开放式  group=机械性损伤  detail=切割伤  location=躯干
```

## 八、流水线
```
报案原文
  → NLU 清洗（base_clean，只删噪声，防幻觉）→ 拆多伤情
  → 追问（原文未交代皮肤状态就问）
  → Step1 类型（开放式/闭合式）
  → Step2a 组别（受伤原因 7 选 1，开放/闭合皆判）
  → Step2b 细分（机械组 11 选 1，仅开放式+机械）
  → Step3 部位（8 选 1）
  → JSON 输出
```

## 九、内存与性能
- 5 个模型按需顺序加载，同一时刻只驻留一个，峰值内存约 1.5GB。
- 单句推理（含 4 个分类阶段）在 Pi 5 Pro 上约几秒。

## 十、已知限制
- `step2_group` 目前是旧版（只学过开放式组别），**闭合式伤情的「受伤原因」会判错**。待重训 Step2a（开放+闭合数据）后，重新 merge + 转 GGUF 替换 `step2_group-q4_k_m.gguf` 即可。

## 十一、重新生成 GGUF（在有 GPU/足够内存的 x86 机器上）
```bash
# 1. 合并 LoRA（见 src/deploy/merge_loras.py）
python src/deploy/merge_loras.py
# 2. 转 fp16 GGUF
python llama.cpp/convert_hf_to_gguf.py models/merged/step1_type \
  --outfile models/gguf/step1_type-f16.gguf --outtype f16
# 3. Q4_K_M 量化
llama-quantize models/gguf/step1_type-f16.gguf \
  models/gguf/step1_type-q4_k_m.gguf q4_k_m
```
