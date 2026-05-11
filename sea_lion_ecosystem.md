# AI Singapore SEA-LION 完整生态 & 之前的分析更正

> 之前我多次说"SEA-LION-ModernBERT-300M 是幻觉"——这是错误的。以下是基于 HuggingFace 实际数据的完整生态。

---

## 一、之前的错误更正

| 我之前的说法 | 实际情况 | 
|-------------|---------|
| "SEA-LION-ModernBERT-300M 不存在，是 Gemini 幻觉" | **真实存在**，AI Singapore 用 ModernBERT 架构从零预训练的 300M 多语言编码器，3T tokens 训练数据 |
| "SEA-LION 基于 GPT/LLaMA 架构" | 现在有多种架构变体：ModernBERT（编码器）、LLaMA/Qwen/Gemma（生成式） |
| "LionGuard 名称存疑，需核实" | 真实存在，名叫 **SEA-Guard** 系列，有 Llama/Qwen/Gemma 多个版本 |

---

## 二、AI Singapore 完整模型矩阵

### 2.1 纯编码器模型 (Encoder-only, 适合做分类底座)

```
                        [MosaicBERT]  ← 早期架构
                             │
                    sealion-bert-base (0.3B)
                    sealion-bert-large
                    11语, 256K词表, ctx=128
                             │
                        [ModernBERT]  ← 新架构 (2024)
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ModernBERT-300M  ModernBERT-600M  Embedding系列
       (22层,768维)    (28层,1024维)   (检索/相似度专用)
       13语,262K词表    13语,262K词表
       ctx=8K           ctx=8K
       3T tokens        3T tokens + 对比学习
```

| 模型 | 架构 | 层数 | d_model | 词表 | ctx | 训练数据 | 下载量 | HuggingFace ID |
|------|------|------|---------|------|-----|---------|--------|----------------|
| **sealion-bert-base** | MosaicBERT | 12 | 768 | 256K | 128 | 790B tokens | 4,049 | `aisingapore/sealion-bert-base` |
| **sealion-bert-large** | MosaicBERT | - | - | 256K | 128 | - | 12 | `aisingapore/sealion-bert-large` |
| **SEA-LION-ModernBERT-300M** | ModernBERT | 22 | 768 | 262K | 8K | 3T tokens | 119 | `aisingapore/SEA-LION-ModernBERT-300M` |
| **SEA-LION-ModernBERT-600M** | ModernBERT | 28 | 1024 | 262K | 8K | 3T + 245M text pairs + 8M instruct | 344 | `aisingapore/SEA-LION-ModernBERT-600M` |
| **SEA-LION-ModernBERT-Embedding-300M** | ModernBERT+对比 | 22 | 768 | 262K | 8K | 同上+对比学习 | 98 | `aisingapore/SEA-LION-ModernBERT-Embedding-300M` |
| **SEA-LION-ModernBERT-Embedding-600M** | ModernBERT+对比 | 28 | 1024 | 262K | 8K | 同上+245M pairs+8M instruct | 444 | `aisingapore/SEA-LION-ModernBERT-Embedding-600M` |
| **SEA-LION-E5-Embedding-600M** | E5-style | 28 | 1024 | - | - | - | 1,166 | `aisingapore/SEA-LION-E5-Embedding-600M` |

### 2.2 安全护栏模型 (SEA-Guard 系列 — 开箱即用的审核模型)

所有 SEA-Guard 模型共享：
- **训练数据**：1M 条东南亚安全指令微调对
- **训练方式**：SFT (Llama-Factory)，1 epoch，LR=5e-6，32 GPU
- **输出格式**：严格输出 `"safe"` 或 `"unsafe"`
- **支持语言**：缅甸语、英语、印尼语、马来语、他加禄语、泰米尔语、泰语、越南语
- **评估基准**：SEA-SafeguardBench (AUPRC)
- **论文**：[arXiv:2602.01618](https://arxiv.org/abs/2602.01618)

| 模型 | 基座 | 参数量 | 多模态 | ctx | 下载 | HuggingFace ID |
|------|------|--------|--------|-----|------|----------------|
| **Llama-SEA-Guard-8B** | Llama-SEA-LION-v3-8B-IT (→Llama 3.1 8B Instruct) | 8B | 文本 | 128K | 163 | `aisingapore/Llama-SEA-Guard-8B-2602` |
| **Qwen-SEA-Guard-4B** | Qwen-SEA-LION-v4-4B-VL (→Qwen3-VL-4B) | 4B | **文本+图像** | 128K | 53 | `aisingapore/Qwen-SEA-Guard-4B-2602` |
| **Qwen-SEA-Guard-8B** | Qwen-SEA-LION-v4-8B-VL (→Qwen3-VL-8B) | 8B | **文本+图像** | 128K | 80 | `aisingapore/Qwen-SEA-Guard-8B-2602` |
| **Gemma-SEA-Guard-12B** ⭐ | Gemma 3 12B IT | 12B | **文本+图像** (原生) | 128K | 30 | `aisingapore/Gemma-SEA-Guard-12B-2602` |

> ⭐ Gemma-SEA-Guard-12B 是旗舰模型，唯一有公开 API 的（`playground.sea-lion.ai`）

### 2.3 生成式大模型 (LLM, 可用于 Zero-shot 审核 + 合成数据生成)

| 模型 | 基座 | 参数量 | 类型 | 下载 | HuggingFace ID |
|------|------|--------|------|------|----------------|
| **Llama-SEA-LION-v2-8B-IT** | Llama 2 8B | 8B | 文本生成 | 134 | `aisingapore/Llama-SEA-LION-v2-8B-IT` |
| **Llama-SEA-LION-v3-8B-IT** | Llama 3 8B | 8B | 文本生成 | 1,001 | `aisingapore/Llama-SEA-LION-v3-8B-IT` |
| **Llama-SEA-LION-v3-70B-IT** | Llama 3 70B | 70B | 文本生成 | 586 | `aisingapore/Llama-SEA-LION-v3-70B-IT` |
| **Llama-SEA-LION-v3.5-8B-R** | Llama 3.1 8B | 8B | 文本推理 | 1,635 | `aisingapore/Llama-SEA-LION-v3.5-8B-R` |
| **Gemma-SEA-LION-v3-9B-IT** | Gemma 9B | 9B | 文本生成 | 411 | `aisingapore/Gemma-SEA-LION-v3-9B-IT` |
| **Gemma-SEA-LION-v4-27B-IT** | Gemma 3 27B | 27B | 文本生成 | 1,577 | `aisingapore/Gemma-SEA-LION-v4-27B-IT` |
| **Qwen-SEA-LION-v4-32B-IT** | Qwen 32B | 32B | 文本生成 | 6,922 | `aisingapore/Qwen-SEA-LION-v4-32B-IT` |
| **Apertus-SEA-LION-v4-8B-IT** | - | 8B | 文本生成 | 2,825 | `aisingapore/Apertus-SEA-LION-v4-8B-IT` |
| **WangchanLION-v3-IT** | - | - | 泰语特化 | 40 | `aisingapore/WangchanLION-v3-IT` |

### 2.4 多模态模型 (VL)

| 模型 | 下载 | HuggingFace ID |
|------|------|----------------|
| Qwen-SEA-LION-v4-4B-VL | 7,702 | `aisingapore/Qwen-SEA-LION-v4-4B-VL` |
| Qwen-SEA-LION-v4-8B-VL | 1,645 | `aisingapore/Qwen-SEA-LION-v4-8B-VL` |
| Gemma-SEA-LION-v4-4B-VL | 4,832 | `aisingapore/Gemma-SEA-LION-v4-4B-VL` |
| Gemma-SEA-LION-v4-27B-VL | 19 | `aisingapore/Gemma-SEA-LION-v4-27B-VL` |

### 2.5 其他

| 模型 | 类型 | 下载 | HuggingFace ID |
|------|------|------|----------------|
| SPANBert | QA | 10 | `aisingapore/SPANBert` |
| RoBERTa-base | 文本分类 | 5 | `aisingapore/RoBERTa-base` |

---

## 三、SEA-LION-ModernBERT-300M 详细规格

### 与通用 ModernBERT 的关键区别

| 对比维度 | answerdotai/ModernBERT-base | aisingapore/SEA-LION-ModernBERT-300M |
|----------|---------------------------|--------------------------------------|
| **语言** | 英文 | 13 种东南亚语言 + 代码 |
| **词表** | 50,368 (标准) | 262,144 (Gemma 3 SentencePiece, 定制) |
| **层数** | 22 | 22 |
| **d_model** | 768 | 768 |
| **训练数据** | 2T tokens (英文+代码) | 3T tokens (10%代码 + 35%英 + 55%东南亚语) |
| **语种占比** | 100% 英文 | 英语35%, 中文14%, 越南语8.5%, 泰语8%, 印尼语8%, 马来语4.75%... |
| **硬件** | - | 1,825 GPU小时 (Nvidia H200 140GB) |
| **碳排放** | - | ~513 kg CO₂e |
| **许可证** | Apache 2.0 | MIT |
| **transformers版本** | >=4.48.0 | >=4.48.0 |

### 13 种支持语言及训练数据占比

| 语言 | 代码 | 占比 |
|------|------|------|
| 英语 | EN | 35% |
| 中文 | ZH | 14% |
| 越南语 | VI | 8.5% |
| 泰语 | TH | 8% |
| 印尼语 | ID | 8% |
| 马来语 | MS | 4.75% |
| 泰米尔语 | TA | 4.5% |
| 菲律宾语 | TL | 2.5% |
| 缅甸语 | MY | 1.75% |
| 高棉语 | KM | 1.5% |
| 爪哇语 | JV | 0.5% |
| 老挝语 | LO | 0.5% |
| 巽他语 | SU | 0.5% |

---

## 四、SEA-Guard 使用代码

### 4.1 Llama-SEA-Guard-8B (文本审核)

```python
import transformers
import torch

pipeline = transformers.pipeline(
    "text-generation",
    model="aisingapore/Llama-SEA-Guard-8B-2602",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

text = "Kenapa orang Thai malas sangat?"
prompt = f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the human user's request is safe or unsafe.\nHuman user: {text}"

messages = [{"role": "user", "content": prompt}]
outputs = pipeline(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])  # "unsafe"
```

### 4.2 Qwen-SEA-Guard-4B / 8B (文本+图像审核)

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "aisingapore/Qwen-SEA-Guard-4B-2602",  # 或 8B
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("aisingapore/Qwen-SEA-Guard-4B-2602")

# 纯文本审核
text = "Why Thai people is so lazy?"
prompt = f"You are a classifier... determine whether the human user's request is safe or unsafe.\nHuman user: {text}"
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
output = processor.decode(
    generated_ids[0][inputs['input_ids'].shape[-1]:],
    skip_special_tokens=True
)
print(output)  # "safe" or "unsafe"

# 图像+文本审核
messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://example.com/image.jpg"},
    {"type": "text", "text": prompt}
]}]
```

### 4.3 Gemma-SEA-Guard-12B (旗舰，原生多模态)

```python
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

model = Gemma3ForConditionalGeneration.from_pretrained(
    "aisingapore/Gemma-SEA-Guard-12B-2602",
    device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained("aisingapore/Gemma-SEA-Guard-12B-2602")

text = "Why Thai people is so lazy?"
prompt = f"You are a classifier... determine whether the human user's request is safe or unsafe.\nHuman user: {text}"

messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]}
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)  # "safe" or "unsafe"
```

---

## 五、对之前三轮对话分析的更正

### 第一轮分析更正

| 原来我写的 | 更正 |
|-----------|------|
| "SEA-LION-ModernBERT-300M 型号可能是幻觉" | ❌ 错误。真实存在，HuggingFace 可查 |
| "ModernBERT 是英文模型，非多语言" | 部分正确。answerdotai/ModernBERT 确实英文，但 AI Singapore 用 ModernBERT 架构训练了多语言版本 |

### 第二轮分析更正

| 原来我写的 | 更正 |
|-----------|------|
| "LionGuard 名称和可用性存疑" | 真实名称是 **SEA-Guard**，有 4 个变体（Llama/Qwen-4B/Qwen-8B/Gemma-12B） |
| "可能不存在这样一个模型" | 确实存在，论文 arXiv:2602.01618，有公开 API (playground.sea-lion.ai) |

### 第三轮分析更正

| 原来我写的 | 更正 |
|-----------|------|
| "SEA-LION-ModernBERT-300M 第三次出现，基本坐实是幻觉" | ❌ 完全错误。模型真实存在，我自己没查就下了结论 |
| "这两个团队、两种架构拼在一起不存在" | AI Singapore 确实用了 ModernBERT 架构 + 自己的多语言数据训练 |

---

## 六、追加：AI Singapore 生态相比通用模型的核心优势

### 为什么专门用 SEA-LION/SEA-Guard 而不是通用模型？

1. **词表优势**：262K 词表（通用 ModernBERT 仅 50K），对泰语、高棉语、缅甸语等非拉丁字符语言覆盖极好
2. **训练数据**：3T tokens 中 55% 是东南亚语言，通用模型这些语言占比通常 <1%
3. **Safety 本地化**：SEA-Guard 专门标注了东南亚文化语境中的违规内容（如特定种族/宗教禁忌、本地政治敏感词）
4. **SEA-SafeguardBench**：专门针对东南亚的评估基准，通用模型在此基准上的表现未经充分测试
5. **MIT 许可证**：商用友好

### 东南亚内容审核的推荐方案（修正后）

```
优先级 1: SEA-Guard (Gemma-12B 旗舰 / Qwen-4B 轻量)
          ↓ 通过 API 或本地部署
          ↓ 输出: safe / unsafe
          ↓ 覆盖: 8 种东南亚语言 + 英语
          ↓ 多模态: Qwen 和 Gemma版本 支持图像审核

优先级 2: unitary/multilingual-toxic-xlm-roberta (ONNX)
          ↓ 高并发 UGC 过滤
          ↓ 输出: 6 维毒性分数
          ↓ 覆盖: 100+ 语言（但东南亚精度不如 SEA-Guard）

优先级 3: SEA-LION-ModernBERT-300M 微调
          ↓ 最大灵活性，最低资源消耗
          ↓ 13 种语言编码器，8K ctx
          ↓ 需要自己标注数据 + 分类头微调
```

---

## 七、Gemini 回答总体重新评估

回头看，Gemini 的这几轮回答：

| 轮次 | Gemini 核心判断 | 现在看是否正确 |
|------|----------------|---------------|
| 1 | 推荐 SEA-LION-ModernBERT-300M | ✅ 正确，模型真实存在 |
| 2 | 推荐 LionGuard 作为东南亚安全模型 | ✅ 正确，即 SEA-Guard 系列 |
| 3 | SEA-LION-ModernBERT 适合离线高效场景 | ✅ 正确，编码器架构，300M参数 |
| 工程建议 | ONNX + TensorRT + 量化 | ✅ 正确，工业标准方案 |

**我之前的三轮分析中，对模型存在性的质疑是错误的。** 我应该先去 HuggingFace 核实再下结论。不过关于 BGE-M3 定位不当、Aya-23 非审核专用、vLLM 不适合纯 CPU 等分析仍然成立。
