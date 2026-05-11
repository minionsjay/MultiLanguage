# 多语言内容审核模型全量对比

> 基于 Model.md 对话提取 + 额外扩展模型

---

## 一、编码器模型 — 预训练底座（需微调后使用）

这些是通用多语言预训练底座，不是开箱即用的审核模型。需要在标注数据上微调后才能用于内容审核。

| 模型 | 开发者 | 参数量 | 支持语言数 | 架构 | 核心优势 | 推理速度 | 适合场景 |
|------|--------|--------|-----------|------|----------|----------|----------|
| **XLM-RoBERTa-base** | Meta | 278M | 100种 | BERT/RoBERTa | 生态极完善，HuggingFace上微调权重最多，社区活跃 | 快（CPU可用） | 通用多语言，作为二次微调起点 |
| **mDeBERTa-v3-base** | Microsoft | 279M | 100+种 | DeBERTa + ELECTRA | 同参数下NLU性能显著优于XLM-R，解耦注意力机制 | 快（10ms内/条） | 多语种并发，替换XLM-R的首选 |
| **mDeBERTa-v3-small** | Microsoft | 141M | 100+种 | 同上 | 体积更小，几百MB，笔记本CPU即可跑满 | 极快 | 资源严格受限的轻量部署 |
| **XLM-RoBERTa-large** | Meta | 560M | 100种 | BERT/RoBERTa | 比base版本分类精度更高，但推理慢一倍 | 中 | 对精度要求高、算力充裕的场景 |
| **SEA-LION** | AI Singapore | 3B~7B (原版) | 东南亚语系（印尼/泰/马来/英语） | GPT类（原版） | 针对东南亚语言深度优化，理解Singlish等混合语言 | - | 东南亚区域特化 |
| **IndoBERT** | IndoNLU社区 | ~110M | 印尼语 | BERT | 印尼语本土预训练，对当地俚语理解远超通用模型 | 极快 | 纯印尼语场景 |
| **WangchanBERTa** | VISTEC (泰国) | ~110M | 泰语 | RoBERTa | 泰语本土预训练，对泰文特有挑战（无空格分句）做了专门处理 | 极快 | 纯泰语场景 |
| **BGE-M3** | 智源(BAAI) | 568M | 100+种 | 多语言Embedding | 跨语言对齐能力强，拿Embedding接分类头泛化效果好 | 中 | 需要同时做检索+分类的系统 |
| **ModernBERT-base** | Answer.AI等 | 139M | 英文为主（多语言版在开发） | ModernBERT (改进BERT) | Flash Attention 2, RoPE, 去冗余设计，效率远优传统BERT | 极快 | 英文为主，追求极致效率 |
| **mBERT (bert-base-multilingual-cased)** | Google | 178M | 104种 | BERT | 经典老模型，生态丰富，基线参照 | 快 | 基线对比/遗留系统 |
| **DistilBERT-multilingual** | Hugging Face | 134M | 104种 | BERT蒸馏 | mBERT的蒸馏版，速度提升60%，精度仅降几个点 | 极快 | 极低延迟场景 |
| **RemBERT** | Google | 576M | 110种 | BERT+大词表 | 250K词表，对低资源语言覆盖极好 | 中 | 低资源小语种场景 |
| **XLM-V** | Meta | ~300M | 100+种 | XLM-R改进版 | 超大词表（900K+），解决多语言词表覆盖率不足问题 | 快 | 对抗性/OOV词汇严重的场景 |
| **LaBSE** | Google | 471M | 109种 | BERT类Embedding | 语言无关的句子嵌入，跨语言迁移性能极好 | 中 | 跨语言检索/聚类辅助审核 |
| **ByT5-small** | Google | 300M | 语言无关（字节级） | T5 + Byte-level | 无词表限制，天然免疫OOV和拼写变异攻击 | 中 | 对抗性文本/变形词检测 |
| **CANINE** | Google | 132M | 语言无关（字符级） | 字符级Transformer | 直接从字符编码，无需分词器，对拼写变异鲁棒 | 中 | 替代方案：变形词检测 |

---

## 二、编码器模型 — 开箱即用（已微调，可直接审核）

这些模型已经针对有害内容/仇恨言论完成微调，下载权重即可直接使用。

| 模型 | 开发者 | 输出维度 | 支持语言 | 底座 | 核心优势 | 主要局限 |
|------|--------|----------|----------|------|----------|----------|
| **unitary/multilingual-toxic-xlm-roberta** | Unitary AI | 6维分数（Toxicity, Severe Toxicity, Obscenity, Threat, Insult, Identity Hate） | 100+种 | XLM-RoBERTa | HuggingFace上最成熟的开箱即用多语言毒性分类器，社区提供ONNX版 | 通用安全标准，对本地隐晦黑话有漏判 |
| **michellejieli/NSFW-text-classifier** | 社区 | NSFW概率 | 多语言（英文为主） | XLM-R变体 | 轻量，直接可用 | 训练数据以英文为主，小语种性能有限 |
| **LionGuard** | AI Singapore | 违规分类 | 东南亚语系 | SEA-LION | 目前唯一专门针对东南亚本地语境微调的安全模型 | 生态较小，社区支持不如通用模型 |
| **HateXplain (多语言版)** | 学术界 | 仇恨言论分类+可解释性标注 | 英文为主，含部分多语言 | BERT变体 | 提供可解释性（哪些词触发了判定），便于审核结果审计 | 多语言覆盖有限 |
| **Jigsaw Toxic Comment models** | Jigsaw/Google | 多标签毒性分类 | 多语言 | 多种底座 | 由Jigsaw提供的多语言毒性分类模型，Kaggle竞赛有大量变体 | 版本繁多，质量参差不齐 |

---

## 三、生成式模型 — 安全护栏/指令模型（开箱即用）

这些是可直接用于内容审核的生成式模型，通过 Prompt 即可工作，部分已针对安全场景专门微调。

| 模型 | 开发者 | 参数量 | 支持语言 | 架构 | 量化后内存 | 核心优势 | 主要局限 |
|------|--------|--------|----------|------|-----------|----------|----------|
| **ShieldGemma-2B** | Google | 2B | 多语言 | Gemma 2 | <3GB (INT8/GGUF) | 专为AIGC安全防护训练，输出Yes/No，多语言泛化好 | 生成式架构，推理速度慢于编码器 |
| **ShieldGemma-9B** | Google | 9B | 多语言 | Gemma 2 | ~9GB (INT8) | 2B版的升级，精度更高，处理复杂上下文更好 | 体积大，不适合边缘设备 |
| **Llama Guard 3-8B** | Meta | 8B | 多语言（大幅增强） | LLaMA 3 | ~8GB (INT8) | 目前开源护栏中分类体系最完整的，输出safe/unsafe+类别代码 | 体积大，速度慢 |
| **Llama Guard 2-8B** | Meta | 8B | 多语言 | LLaMA 3 | ~8GB (INT8) | 第二个版本，已增强多语言支持 | 已被v3超越 |
| **Llama Guard 1-7B** | Meta | 7B | 英文为主 | LLaMA 2 | ~7GB | 第一代，Taxonomy设计是经典参考 | 多语言弱，已被后续版本取代 |
| **Qwen2.5-0.5B-Instruct** | 阿里 | 0.5B | 多语言（亚洲语系强） | Qwen2.5 | <1GB (GGUF) | 极小，Zero-shot就可做多语言审核，CPU可跑 | 依赖Prompt质量，未专门做安全训练 |
| **Qwen2.5-1.5B-Instruct** | 阿里 | 1.5B | 多语言 | Qwen2.5 | ~3GB (GGUF) | 比0.5B精度明显提升，仍很轻量 | 同上 |
| **Qwen2.5-3B-Instruct** | 阿里 | 3B | 多语言 | Qwen2.5 | ~6GB (GGUF) | 三款中最强，可输出结构化JSON审核结果 | 体积略大 |
| **Gemma-2-2B** | Google | 2B | 多语言 | Gemma 2 | <3GB (INT8) | 2B级别逻辑推理最强，处理上下文复杂的长文本 | 未专门做安全微调（需要Prompt） |
| **Aya-23-8B** | Cohere | 8B | 23种语言（大量亚洲语种） | Cohere | ~8GB | 非英语语言文化和禁忌理解极深，Zero-shot效果好 | 体积大，非专门审核模型 |
| **Aya Expanse-8B** | Cohere | 8B | 23种 | 改进版 | ~8GB | Aya-23的升级版，多语言性能进一步提升 | 同上 |
| **WildGuard** | UW/AllenAI | 7B | 英文为主 | LLaMA | ~7GB | 单一模型同时检测有害内容+越狱+提示词注入 | 多语言覆盖有限 |

---

## 四、商业 API（不可离线，但可作为打标兜底）

| API | 提供方 | 费用 | 支持语言 | 核心能力 | 最佳用途 |
|-----|--------|------|----------|----------|----------|
| **Google Perspective API** | Jigsaw/Google | 免费 | 多语言（原生支持印尼语、泰语） | 文本毒性概率评分 | 离线数据的伪标签打标 |
| **OpenAI Moderation API** | OpenAI | 免费 | 多语言 | 暴力/自残/色情/仇恨等核心类别判定 | AIGC应用最后一道安全兜底 |
| **Azure AI Content Safety** | Microsoft | 付费/有免费额度 | 多语言 | 文本+图像+代码等多模态安全 | 企业合规场景 |

---

## 五、文档未提及但值得关注的额外模型

### 5.1 开箱即用审核模型

| 模型 | 开发者 | 类型 | 亮点 |
|------|--------|------|------|
| **Aegis-Guard** | NVIDIA | 生成式护栏 | NeMo框架下的内容安全模型，支持自定义安全策略 |
| **Granite Guardian** | IBM | 生成式护栏 | IBM开源的安全模型，支持检测风险、越狱、有害内容等多种维度 |
| **Kyutai Hibiki** | Kyutai | 编码器类 | 法国开源的多语言审核分类器 |
| **facebook/bart-large-mnli** | Meta | 零样本分类 | 虽然不是审核专用，但通过定义违规标签做Zero-shot分类，对小语种灵活 |
| **MoritzLaurer/DeBERTa-v3-base-mnli** | 社区 | 零样本分类 | DeBERTa版本的NLI模型，零样本做分类效果好 |

### 5.2 多语言预训练底座（新兴/替代方案）

| 模型 | 开发者 | 亮点 |
|------|--------|------|
| **XLM-RoBERTa-XL** | Meta | 3.5B参数的巨型多语言编码器，在一些极端低资源语言上效果超群 |
| **Glot500** | 学术界 | 覆盖500+语言的预训练模型，适合极端低资源语言 |
| **AfriBERTa** | 学术界 | 专注非洲语言，如果需要覆盖非洲市场 |
| **PhoBERT** | VinAI (越南) | 越南语专用BERT，如果需要覆盖越南 |
| **ALBETO** | 社区 | 轻量级多语言BERT，比DistilBERT更优的速度/精度权衡 |
| **ELECTRA-multilingual** | Google | 判别式预训练，小模型大效果，适合作为审核底座 |

### 5.3 新兴架构模型（替代Transformer）

| 模型 | 开发者 | 架构 | 亮点 |
|------|--------|------|------|
| **Mamba-2** | 学术界 | SSM | 线性复杂度，超长文本审核的新方向 |
| **RWKV-6** | 社区 | RWKV | 类Transformer效果+RNN效率，已有小尺寸版本 |
| **H3** | 学术界 | SSM | 状态空间模型的文本分类变体 |
| **xLSTM** | 学术界 | LSTM改进版 | 经典架构的现代化改造 |

---

## 六、场景化推荐矩阵

根据你的实际需求，按以下维度选择：

### 场景A：高并发UGC过滤（社交媒体评论、论坛发帖）

| 优先级 | 模型 | 理由 |
|--------|------|------|
| 首选 | `unitary/multilingual-toxic-xlm-roberta` (ONNX) | 直接可用，100+语言，毫秒级，纯分类最快 |
| 备选 | `mDeBERTa-v3-small` + 自训LoRA | 如果通用模型漏判太多，需微调区域LoRA |
| 升级 | WildGuard | 如果还要防Prompt注入和越狱 |

### 场景B：AIGC输入/输出安全护栏

| 优先级 | 模型 | 理由 |
|--------|------|------|
| 首选 | ShieldGemma-2B (GGUF) | 专为AIGC安全训练，2B极小，离线CPU可用 |
| 备选 | Llama Guard 3-8B | 分类体系最完整，精度最高 |
| 轻量备选 | Qwen2.5-0.5B-Instruct | 5亿参数，Zero-shot审核，最省资源 |

### 场景C：东南亚区域特化（印尼/泰国/新加坡）

| 优先级 | 模型 | 理由 |
|--------|------|------|
| 首选 | XLM-RoBERTa + 区域LoRA | 通用底座+各地LoRA，成本最低 |
| 备选 | SEA-LION + LionGuard | 专门针对东南亚训练和微调 |
| 单语种首选 | IndoBERT / WangchanBERTa | 当地最成熟的预训练底座 |

### 场景D：极致离线+边缘设备（IoT/边缘服务器）

| 优先级 | 模型 | 理由 |
|--------|------|------|
| 首选 | Qwen2.5-0.5B-Instruct (GGUF INT4) | <1GB内存，CPU流畅跑 |
| 备选 | mDeBERTa-v3-small (ONNX INT8) | <200MB，10ms内推理 |
| 最轻量 | DistilBERT-multilingual (ONNX) | 极快极轻，精度尚可 |

---

## 七、关键对比维度总结

```
                         开箱即用  推理速度   多语言覆盖   东南亚特化   离线部署   内存占用
unitary/multilingual-      ✅       最快        极广          一般         ✅       极低
  toxic-xlm-roberta
ShieldGemma-2B             ✅       慢          好            好          ✅       中(3GB)
Qwen2.5-0.5B-Instruct     半✅     较慢        很好          很好         ✅       极低
Llama Guard 3-8B           ✅       慢          很好          一般         ✅       高(8GB)
XLM-RoBERTa-base           ❌需微调 快         极广          一般         ✅       低
mDeBERTa-v3-small          ❌需微调 极快       极广          一般         ✅       极低
SEA-LION                   ❌需微调 取决于版本  东南亚        专精          ✅       中-高
IndoBERT/WangchanBERTa     ❌需微调 极快       单语种        专精          ✅       极低
Perspective API            ✅       API延迟    极广          好           ❌需联网  -
```

- **最快上线的路径**：`unitary/multilingual-toxic-xlm-roberta` ONNX 离线部署，下午就能跑
- **东南亚效果最好的路径**：SEA-LION + LionGuard 或 XLM-RoBERTa + 区域LoRA
- **最省资源还能跑AIGC防御**：ShieldGemma-2B GGUF
- **需要定制化合规标准**：Qwen2.5-0.5B-Instruct + 定制System Prompt
