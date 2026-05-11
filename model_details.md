# 多语言内容审核模型详细档案

> 每个模型包含：基本信息、技术架构、训练数据、性能特点、使用方式、HuggingFace地址、优缺点

---

# 第一部分：预训练编码器底座（需微调）

---

## 1. XLM-RoBERTa (XLM-R)

| 维度 | 详情 |
|------|------|
| **全称** | Cross-lingual Language Model - RoBERTa |
| **开发者** | Meta AI (原Facebook AI) |
| **发布时间** | 2019年11月 |
| **论文** | [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) |
| **架构** | Transformer Encoder (RoBERTa变体)，12层(base)/24层(large)，768维(base)/1024维(large) |
| **参数量** | base: 278M / large: 560M |
| **词表大小** | 250K (SentencePiece) |
| **支持语言** | 100种语言 |
| **预训练数据** | 2.5TB CommonCrawl多语言语料（过滤后），覆盖100种语言 |
| **预训练任务** | Masked Language Modeling (MLM)，多语言混合训练 |
| **推理速度** | base版本在CPU上单条文本~10-20ms |
| **内存占用** | base: ~1.1GB (FP32) / ~280MB (INT8) / large: ~2.2GB (FP32) |
| **HuggingFace** | `FacebookAI/xlm-roberta-base` / `FacebookAI/xlm-roberta-large` |
| **关键优势** | ① 生态最完善 — HuggingFace上有数千个基于它微调的模型 ② 对100种语言支持稳定，社区验证充分 ③ 大量现成的多语言仇恨言论/毒性分类微调权重 ④ ONNX导出支持成熟 |
| **关键局限** | ① 2024年已被mDeBERTa-v3在多语言NLU上超越 ② 词表250K对低资源语言仍不够 ③ 训练语料偏正式文本，对网络黑话/俚语理解一般 |
| **适合场景** | 二次微调的起点，需要生态支持丰富的项目 |

---

## 2. mDeBERTa-v3

| 维度 | 详情 |
|------|------|
| **全称** | Multilingual Decoding-enhanced BERT with Disentangled Attention v3 |
| **开发者** | Microsoft |
| **发布时间** | 2023年 |
| **论文** | [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training](https://arxiv.org/abs/2111.09543) |
| **架构** | Transformer Encoder + 解耦注意力机制 (Disentangled Attention) + ELECTRA式RTD预训练 + GD (Gradient Disentangled Embedding Sharing) |
| **参数量** | base: 279M / small: 141M |
| **词表大小** | 250K |
| **支持语言** | 100+种语言 |
| **预训练数据** | 多语言CommonCrawl + Wikipedia + 书籍 |
| **预训练任务** | Replaced Token Detection (RTD) — 判别式任务，比MLM更高效 |
| **推理速度** | small版CPU上单条~5-10ms，base版~10-15ms |
| **内存占用** | small: ~560MB (FP32) / ~140MB (INT8) / base: ~1.1GB (FP32) |
| **HuggingFace** | `microsoft/mdeberta-v3-base` / `microsoft/mdeberta-v3-small` |
| **关键优势** | ① 相同参数量下多语言NLU性能显著优于XLM-R ② small版本仅141M参数，笔记本CPU即可满负载跑 ③ 适合作为审核模型微调的底座 |
| **关键局限** | ① 生态不如XLM-R丰富，社区微调权重较少 ② 预训练数据仍是通用语料，缺少安全/毒性领域知识 ③ 没有真正"开箱即用"的安全版本 |
| **适合场景** | 作为新一代多语言审核底座的默认选择，替代XLM-R |

---

## 3. SEA-LION 系列

| 维度 | 详情 |
|------|------|
| **全称** | Southeast Asian Languages In One Network |
| **开发者** | AI Singapore |
| **发布时间** | 2023年开始，持续更新 |
| **论文** | [SEA-LION Technical Report](https://arxiv.org/abs/2310.16124) |
| **架构** | 基于GPT类/LLaMA架构的Decoder-only模型（原版）；后续有基于其它架构的变体 |
| **参数量** | 原版: 3B / 7B；传闻有300M级别的小型变体 |
| **词表大小** | 针对东南亚语言定制，覆盖马来语、印尼语、泰语、越南语、他加禄语、缅甸语等 |
| **支持语言** | 东南亚11种官方语言 + Singlish等混合语 |
| **预训练数据** | 东南亚语系专用语料：本地新闻、社交媒体、政府文件、维基百科，包含大量Singlish和Code-Switching文本 |
| **预训练任务** | Causal Language Modeling |
| **推理速度** | 取决于版本大小，3B/7B需要GPU |
| **内存占用** | 3B: ~6GB (FP16) / 7B: ~14GB (FP16) |
| **HuggingFace** | `aisingapore/sea-lion-7b-instruct` 等 |
| **关键优势** | ① 唯一真正针对东南亚语系深度优化的多语言大模型 ② 对Singlish（新加坡式英语）理解远超任何通用模型 ③ 理解东南亚特有的文化语境和禁忌 ④ 有AI Singapore官方持续维护 |
| **关键局限** | ① 参数量偏大（3B起步），不适合极轻量离线部署 ② 原版是生成式，不是编码器，推理速度不如BERT类 ③ 生态远小于XLM-R等通用模型 ④ 300M版本的存在性需核实 |
| **适合场景** | 东南亚语种为主的内容审核底座，尤其新加坡/马来西亚市场 |

> ⚠️ Gemini提到的 "SEA-LION-ModernBERT-300M" 型号需要去AI Singapore官方核实，ModernBERT和SEA-LION来自不同团队，该组合可能是幻觉。

---

## 4. IndoBERT

| 维度 | 详情 |
|------|------|
| **全称** | IndoBERT (Indonesian BERT) |
| **开发者** | IndoNLU社区（印尼多所大学联合） |
| **发布时间** | 2020年 |
| **论文** | [IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding](https://arxiv.org/abs/2009.05387) |
| **架构** | BERT-base (Transformer Encoder)，12层，768维 |
| **参数量** | ~110M (base) / ~130M (large) |
| **词表大小** | 32K，专门针对印尼语训练 |
| **支持语言** | 印尼语（Bahasa Indonesia） |
| **预训练数据** | 印尼语维基百科 + 新闻语料 + 社交媒体 + 政府文档，总计约23GB文本 |
| **预训练任务** | Masked Language Modeling + Next Sentence Prediction |
| **推理速度** | CPU上单条~5ms |
| **内存占用** | ~440MB (FP32) / ~110MB (INT8) |
| **HuggingFace** | `indobenchmark/indobert-base-p1` / `indobenchmark/indobert-large-p2` |
| **关键优势** | ① 印尼语最成熟的预训练底座 ② 对印尼语网络俚语（Bahasa Gaul）理解远超通用多语言模型 ③ 极小且快，适合高并发 |
| **关键局限** | ① 仅支持印尼语，无法跨语言 ② 基于老BERT架构，不如ModernBERT效率高 ③ 需要自己微调做审核 |
| **适合场景** | 纯印尼语内容审核 |

---

## 5. WangchanBERTa

| 维度 | 详情 |
|------|------|
| **全称** | WangchanBERTa (Thai BERT) |
| **开发者** | VISTEC（泰国国家科技发展署） |
| **发布时间** | 2020年 |
| **论文** | [WangchanBERTa: Pretraining Transformer-Based Thai Language Models](https://arxiv.org/abs/2101.09635) |
| **架构** | RoBERTa-base (Transformer Encoder)，12层，768维 |
| **参数量** | ~110M |
| **词表大小** | 25K，专门针对泰语训练 |
| **支持语言** | 泰语（Thai） |
| **预训练数据** | 泰语维基百科 + 新闻 + 社交媒体 + 论坛（Pantip等） + 图书，约30GB |
| **预训练任务** | Masked Language Modeling |
| **推理速度** | CPU上单条~5ms |
| **内存占用** | ~440MB (FP32) / ~110MB (INT8) |
| **HuggingFace** | `airesearch/wangchanberta-base-attack-on-train` |
| **关键优势** | ① 泰语最成熟的预训练底座 ② 对泰文的特殊挑战（无空格分词、复杂正字法）做了专门优化 ③ 训练数据包含泰国本地论坛，覆盖网络用语 |
| **关键局限** | ① 仅支持泰语 ② 架构较老 ③ 需要微调 |
| **适合场景** | 纯泰语内容审核 |

---

## 6. BGE-M3

| 维度 | 详情 |
|------|------|
| **全称** | BAAI General Embedding - Multilingual/Multi-Granularity/Multi-Function |
| **开发者** | 智源研究院 (BAAI) |
| **发布时间** | 2024年 |
| **论文** | [BGE M3-Embedding: Multi-Lingual, Multi-Granularity, Multi-Functionality](https://arxiv.org/abs/2402.03216) |
| **架构** | XLM-RoBERTa-like Encoder |
| **参数量** | 568M |
| **词表大小** | 250K |
| **支持语言** | 100+种语言 |
| **预训练数据** | 多语言语料 + 大规模检索/分类数据 |
| **预训练任务** | 多阶段训练：Retrieval + Clustering + Classification |
| **推理速度** | CPU上单条~20ms |
| **内存占用** | ~2.3GB (FP32) |
| **HuggingFace** | `BAAI/bge-m3` |
| **关键优势** | ① 极优秀的跨语言语义对齐能力 ② 同时支持Dense Retrieval和Sparse Retrieval ③ 同一模型可用于检索和分类 |
| **关键局限** | ① 本质是Embedding模型，不是为分类设计的 — 用于内容审核需要额外接分类头+微调 ② 参数量比BERT-base大一倍 ③ 纯分类场景不如专用模型高效 |
| **适合场景** | 需要审核+检索并行（如先检索敏感文档再判断上下文）的系统 |

> ⚠️ Gemini将其作为"通用多语言编码器"与mDeBERTa-v3并列推荐用于分类，实际上它的主战场是Embedding/检索，用于分类是迂回方案。

---

## 7. ModernBERT

| 维度 | 详情 |
|------|------|
| **全称** | ModernBERT |
| **开发者** | Answer.AI + LightOn + 多机构联合 |
| **发布时间** | 2024年12月 |
| **论文** | [ModernBERT: Smarter, Better, Faster, Longer](https://arxiv.org/abs/2412.01890) |
| **架构** | BERT Encoder现代改进版：Flash Attention 2 + RoPE + 偏置丢弃 + 无位置偏移 + GeGLU激活 + 交替全局/局部注意力 |
| **参数量** | base: 139M / large: 395M |
| **词表大小** | 50368 (英文为主) |
| **支持语言** | 目前主要为英文，多语言版本开发中 |
| **预训练数据** | 2T tokens英文数据（含C4、ArXiv、代码等），序列长度支持8192 |
| **预训练任务** | Masked Language Modeling |
| **推理速度** | 极快 — 比经典BERT快2倍以上，支持长文本处理 |
| **内存占用** | base: ~560MB (FP32) |
| **HuggingFace** | `answerdotai/ModernBERT-base` / `answerdotai/ModernBERT-large` |
| **关键优势** | ① BERT架构的最新进化，效率极优 ② 原生支持Flash Attention 2，GPU友好 ③ 8192 context window远超传统BERT的512 ④ 训练数据包含代码，对变形文本可能更鲁棒 |
| **关键局限** | ① 目前主要是英文模型，多语言版仍在开发 ② 生态极新（2024年底），社区微调权重几乎没有 ③ 非多语言模型，直接用于多语言场景效果未知 |
| **适合场景** | 英文内容审核的极致高效底座；等中文/多语言版发布后再评估多语言场景 |

> ⚠️ Gemini提到的"SEA-LION-ModernBERT-300M"中的ModernBERT部分需谨慎对待。ModernBERT确实是好架构，但它目前是英文模型，不会成为东南亚专用的底座。

---

## 8. mBERT (bert-base-multilingual-cased)

| 维度 | 详情 |
|------|------|
| **全称** | Multilingual BERT |
| **开发者** | Google Research |
| **发布时间** | 2018年11月 |
| **论文** | 原BERT论文 (Devlin et al., 2019) |
| **架构** | BERT-base (Transformer Encoder)，12层，768维 |
| **参数量** | 178M |
| **词表大小** | 119K，覆盖104种语言 |
| **支持语言** | 104种语言 |
| **预训练数据** | 104种语言的维基百科（总计约10GB文本） |
| **预训练任务** | Masked Language Modeling + Next Sentence Prediction |
| **推理速度** | CPU上单条~5-8ms |
| **内存占用** | ~710MB (FP32) |
| **HuggingFace** | `google-bert/bert-base-multilingual-cased` / `google-bert/bert-base-multilingual-uncased` |
| **关键优势** | ① 最经典、被引用最多的多语言模型 ② 生态极丰富 ③ 适合作为基线（baseline）进行对比实验 |
| **关键局限** | ① 发布于2018年，技术上已落后于XLM-R和mDeBERTa ② 训练数据量小（仅Wiki），缺网络语料 ③ 词表小导致一些低资源语言覆盖率差 ④ 多语言表现显著弱于XLM-R |
| **适合场景** | 学术基线对比，遗留系统维护 |

---

## 9. DistilBERT-multilingual

| 维度 | 详情 |
|------|------|
| **全称** | DistilBERT Multilingual |
| **开发者** | Hugging Face |
| **发布时间** | 2020年 |
| **论文** | [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) |
| **架构** | 从mBERT蒸馏而来，6层（为mBERT的一半），768维 |
| **参数量** | 134M |
| **词表大小** | 119K |
| **支持语言** | 104种语言 |
| **预训练数据** | 同mBERT（知识蒸馏，非重新预训练） |
| **预训练任务** | 蒸馏训练（Student-Teacher） |
| **推理速度** | 比mBERT快60%；CPU上单条~3-5ms |
| **内存占用** | ~540MB (FP32) / ~135MB (INT8) |
| **HuggingFace** | `distilbert/distilbert-base-multilingual-cased` |
| **关键优势** | ① 比mBERT快60%，精度仅损失3-5% ② 极省资源 ③ 适合边缘设备 |
| **关键局限** | ① 基于mBERT蒸馏，继承了mBERT的局限（训练数据少、词表小） ② 精度天花板低于XLM-R ③ 缺少社区安全审核微调权重 |
| **适合场景** | 极致轻量、对精度容忍度高的场景 |

---

## 10. RemBERT

| 维度 | 详情 |
|------|------|
| **全称** | Rebalanced Multilingual BERT |
| **开发者** | Google Research |
| **发布时间** | 2020年 |
| **论文** | [Rethinking Embedding Coupling in Pre-trained Language Models](https://arxiv.org/abs/2010.12821) |
| **架构** | BERT改进版，解耦输入Embedding和输出层，大词表（250K） |
| **参数量** | 576M |
| **词表大小** | 250K |
| **支持语言** | 110种语言 |
| **预训练数据** | 多语言维基百科 + CommonCrawl |
| **预训练任务** | MLM |
| **推理速度** | CPU上单条~20ms |
| **内存占用** | ~2.3GB (FP32) |
| **HuggingFace** | `google/rembert` |
| **关键优势** | ① 更大的词表对低资源语言覆盖更好 ② 解耦设计平衡了高资源和低资源语言的性能 |
| **关键局限** | ① 参数量大但生态很小 ② 社区关注度低，几乎无安全审核相关微调权重 |
| **适合场景** | 需要覆盖小众低资源语言的场景 |

---

## 11. XLM-V

| 维度 | 详情 |
|------|------|
| **全称** | XLM-V (XLM with Very Large Vocabulary) |
| **开发者** | Meta AI |
| **发布时间** | 2023年 |
| **论文** | [XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models](https://arxiv.org/abs/2301.10472) |
| **架构** | XLM-R改进版，超大词表 |
| **参数量** | ~300M |
| **词表大小** | 901K (对比XLM-R的250K) |
| **支持语言** | 100+种语言 |
| **预训练数据** | 同XLM-R语料 |
| **预训练任务** | MLM + 词表扩展训练 |
| **推理速度** | 略慢于XLM-R-base（因大词表） |
| **内存占用** | ~1.5GB (FP32) |
| **HuggingFace** | `facebook/xlm-v-base` |
| **关键优势** | ① 超大词表极大提升低资源语言和OOV词汇的覆盖 ② 对对抗性变形词的鲁棒性可能更好（好分词=好表征） |
| **关键局限** | ① 生态几乎没有，社区微调权重稀少 ② 大词表带来额外内存开销 ③ 知名度低 |
| **适合场景** | 低资源语言或变形词严重的对抗性审核场景 |

---

## 12. LaBSE

| 维度 | 详情 |
|------|------|
| **全称** | Language-agnostic BERT Sentence Embedding |
| **开发者** | Google Research |
| **发布时间** | 2020年 |
| **论文** | [Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852) |
| **架构** | BERT-base encoder + 双塔翻译Ranking训练 |
| **参数量** | 471M |
| **词表大小** | 500K |
| **支持语言** | 109种语言 |
| **预训练数据** | 多语言维基百科 + 翻译平行语料 |
| **预训练任务** | 翻译Ranking（Translation Ranking Task）+ MLM |
| **推理速度** | CPU上单条~15ms |
| **内存占用** | ~1.9GB (FP32) |
| **HuggingFace** | `sentence-transformers/LaBSE` |
| **关键优势** | ① 语言无关的句子嵌入（同一语义的不同语言句子产生相似向量） ② 跨语言零样本迁移能力极强 ③ 可用于审核系统中找相似的违规文本（多语言相似度匹配） |
| **关键局限** | ① Embedding模型，非分类模型 ② 需要额外分类头+微调才能在审核中使用 |
| **适合场景** | 跨语言违规文本检索/相似度匹配/聚类 |

---

## 13. XLM-RoBERTa-XL / XXL

| 维度 | 详情 |
|------|------|
| **全称** | XLM-RoBERTa Extra Large |
| **开发者** | Meta AI |
| **发布时间** | 2021年 |
| **架构** | 同XLM-R，深度扩展（36层/48层） |
| **参数量** | XL: 3.5B / XXL: 10.7B |
| **词表大小** | 250K |
| **支持语言** | 100+种语言 |
| **预训练数据** | 同XLM-R（更大规模） |
| **关键优势** | 低资源语言下游任务性能极强 |
| **关键局限** | 体积巨大，不适合离线小模型场景 |
| **适合场景** | 仅当精度要求极高且有GPU集群时考虑 |

---

## 14. mT5

| 维度 | 详情 |
|------|------|
| **全称** | Multilingual T5 |
| **开发者** | Google Research |
| **发布时间** | 2020年 |
| **论文** | [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/abs/2010.11934) |
| **架构** | T5 Encoder-Decoder (Transformer) |
| **参数量** | small: 300M / base: 580M / large: 1.2B / xl: 3.7B / xxl: 13B |
| **支持语言** | 101种语言 |
| **关键优势** | Encoder-Decoder架构，可同时做分类和生成 |
| **关键局限** | 比纯Encoder更重，推理更慢 |
| **适合场景** | 审核 + 解释生成一体化 |

---

## 15. ELECTRA-multilingual

| 维度 | 详情 |
|------|------|
| **全称** | ELECTRA Multilingual |
| **开发者** | Google Research |
| **发布时间** | 2020年 |
| **架构** | 判别式预训练（Generator + Discriminator），推理时只用Discriminator |
| **参数量** | base: ~110M |
| **支持语言** | 100+种语言 |
| **关键优势** | 判别式训练效率远高于MLM，同样参数下性能更强 |
| **关键局限** | 多语言版本不如原始英文版流行 |
| **适合场景** | 追求极端性价比的底座 |

---

## 16. Glot500

| 维度 | 详情 |
|------|------|
| **全称** | Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages |
| **开发者** | 学术界（多机构联合） |
| **发布时间** | 2023年 |
| **架构** | XLM-R兼容 |
| **支持语言** | 500+种语言 |
| **关键优势** | 覆盖全球500+语言的极端多语言模型，包含大量非洲、东南亚低资源语言 |
| **关键局限** | 性能在每个语言上不如特化模型 |
| **适合场景** | 需要覆盖极端低资源小语种 |

---

## 17. PhoBERT

| 维度 | 详情 |
|------|------|
| **全称** | PhoBERT (Vietnamese BERT) |
| **开发者** | VinAI Research (越南) |
| **发布时间** | 2020年 |
| **架构** | RoBERTa-base |
| **参数量** | 135M |
| **支持语言** | 越南语 |
| **HuggingFace** | `vinai/phobert-base` / `vinai/phobert-large` |
| **关键优势** | 越南语最成熟的预训练模型 |
| **适合场景** | 越南市场内容审核 |

---

## 18. AfriBERTa

| 维度 | 详情 |
|------|------|
| **全称** | AfriBERTa |
| **开发者** | 学术界 |
| **发布时间** | 2021年 |
| **支持语言** | 非洲语言（主要是尼日尔-刚果语系） |
| **关键优势** | 非洲语言覆盖 |
| **适合场景** | 非洲市场内容审核 |
| **HuggingFace** | `castorini/afriberta_large` |

---

## 19. ALBETO

| 维度 | 详情 |
|------|------|
| **全称** | A Lite BERT |
| **开发者** | 社区 |
| **架构** | BERT轻量化变体 |
| **关键优势** | 比DistilBERT更优的速度/精度权衡 |
| **适合场景** | 轻量部署 |

---

# 第二部分：开箱即用审核模型（编码器类）

---

## 20. unitary/multilingual-toxic-xlm-roberta

| 维度 | 详情 |
|------|------|
| **全称** | Multilingual Toxic Comment Classification (XLM-RoBERTa) |
| **开发者** | Unitary AI |
| **发布时间** | 2021年 |
| **架构** | XLM-RoBERTa-base + 6分类头 |
| **参数量** | ~278M |
| **支持语言** | 100+种语言 |
| **训练数据** | 多语言有毒评论数据，包含英文Jigsaw数据 + 多语言翻译/对齐 |
| **输出格式** | 6维概率数组：[Toxicity, Severe Toxicity, Obscenity, Threat, Insult, Identity Hate] |
| **推理速度** | CPU上单条~10-20ms；ONNX版~5-10ms |
| **内存占用** | ~1.1GB (PyTorch) / ~300MB (ONNX优化) |
| **HuggingFace** | `unitary/multilingual-toxic-xlm-roberta` |
| **ONNX版本** | HuggingFace社区有现成导出 |
| **是否可直接用** | ✅ 是 |
| **是否需要联网** | ❌ 否 |
| **关键优势** | ① 整个文档最重要、最直接可用的模型 ② 100+语言，6维度全面 ③ ONNX版离线即插即用 ④ 推理速度极快，高并发友好 ⑤ 社区验证充分 |
| **关键局限** | ① 基于通用国际安全标准，对东南亚隐晦黑话有漏判 ② 不输出违规理由，只输出分数 ③ XLM-R底座相对老 |
| **适合场景** | 高并发UGC评论审核的首选方案 |

---

## 21. michellejieli/NSFW-text-classifier

| 维度 | 详情 |
|------|------|
| **全称** | NSFW Text Classifier |
| **开发者** | Michelle Li (社区) |
| **发布时间** | ~2022年 |
| **架构** | 基于XLM-R或类似多语言底座的分类器 |
| **输出格式** | NSFW概率 |
| **支持语言** | 多语言（英文为主，跨语言迁移） |
| **HuggingFace** | `michellejieli/NSFW-text-classifier` |
| **是否可直接用** | ✅ 是 |
| **关键优势** | 轻量，社区验证 |
| **关键局限** | ① 训练数据以英文为主，小语种效果有限 ② 只输出NSFW二分类，不如unitary模型6维度全面 ③ 主要用于色情内容检测，仇恨言论覆盖不全面 |
| **适合场景** | 辅助NSFW过滤，作为unitary模型的补充 |

---

## 22. LionGuard

| 维度 | 详情 |
|------|------|
| **全称** | LionGuard (推测名称，需核实) |
| **开发者** | AI Singapore |
| **发布时间** | 2024年（推测） |
| **架构** | 基于SEA-LION底座微调的安全分类模型 |
| **支持语言** | 东南亚语系 |
| **训练数据** | 东南亚区域仇恨言论、敏感话题、侮辱性词汇数据 |
| **是否可直接用** | ✅ 是（Gemini宣称） |
| **关键优势** | ① 目前唯一专门针对东南亚本地语境微调的开源安全模型 ② 对Singlish、印尼/泰本地黑话理解好 |
| **关键局限** | ① 生态极小，社区认知度低 ② 实际模型名称和可用性需到AI Singapore确认 ③ Gemini的原文说"这几乎是目前开源界唯一可以直接拿来用的本地化成模" — 暗示选择极少 |
| **适合场景** | 东南亚语种内容审核的首选成品模型 |

> ⚠️ 这个模型名称和是否存在需要去AI Singapore的HuggingFace或GitHub仓库确认。

---

## 23. HateXplain

| 维度 | 详情 |
|------|------|
| **全称** | HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection |
| **开发者** | 学术界（多机构） |
| **发布时间** | 2020年 |
| **架构** | BERT-based |
| **参数量** | ~110M |
| **关键优势** | 不仅输出分类，还输出哪些词触发了判定（可解释性） |
| **关键局限** | 训练数据主要是英文，多语言支持有限 |
| **适合场景** | 需要对审核结果进行审计/可解释性分析的场景 |
| **HuggingFace** | `Hate-speech-CNERG/bert-base-uncased-hatexplain` |

---

## 24. Jigsaw Toxic Comment Models

| 维度 | 详情 |
|------|------|
| **开发商** | Jigsaw / Google / Kaggle竞赛社区 |
| **发布时间** | 2018年至今持续更新 |
| **架构** | 多种（BERT、LSTM、CNN等），Kaggle上有数百个开源变体 |
| **关键优势** | 由Google旗下的Jigsaw推动，数据质量高，竞赛催生了大量优化方案 |
| **关键局限** | 版本繁多、质量参差不齐，需要自己筛选评估 |
| **适合场景** | 寻找不同底座和优化策略的灵感来源 |
| **Kaggle** | [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification) |

---

# 第三部分：生成式安全护栏模型

---

## 25. ShieldGemma-2B

| 维度 | 详情 |
|------|------|
| **全称** | ShieldGemma |
| **开发者** | Google DeepMind |
| **发布时间** | 2024年 |
| **论文** | [ShieldGemma: Generative AI Content Moderation Based on Gemma](https://arxiv.org/abs/2407.21772) |
| **架构** | Gemma 2 (Decoder-only Transformer) |
| **参数量** | 2B |
| **支持语言** | 多语言（Gemma 2底座的多语言能力） |
| **训练数据** | 安全策略/准则 + 违规/合规样本对 + 强化学习对齐 |
| **输出格式** | `Yes`（违规）或 `No`（合规） |
| **推理速度** | 生成式，需要逐token输出，单条文本~0.2-0.5秒（CPU） |
| **内存占用** | FP16: ~4GB / INT8: <3GB / GGUF INT4: ~1.5GB |
| **HuggingFace** | `google/shieldgemma-2b` |
| **是否可直接用** | ✅ 是 |
| **是否需要联网** | ❌ 否 |
| **关键优势** | ① 专为AI安全防护训练，开箱即用 ② 多语言泛化能力好 ③ 量化后资源极省 ④ 直接输出Yes/No，简单可靠 |
| **关键局限** | ① 生成式推理速度远慢于编码器 ② 只输出合规/违规二元判断，无细分维度 ③ 对东南亚隐晦黑话性能未知 ④ 通用安全标准，非本地化 |
| **适合场景** | AIGC输入/输出安全护栏；离线CPU部署的安全防线 |

---

## 26. ShieldGemma-9B

| 维度 | 详情 |
|------|------|
| **全称** | 同上，大版本 |
| **参数量** | 9B |
| **内存占用** | FP16: ~18GB / INT8: ~9GB |
| **HuggingFace** | `google/shieldgemma-9b` |
| **关键优势** | 精度高于2B版，处理复杂上下文更准确 |
| **关键局限** | 体积大，不适合边缘设备 |
| **适合场景** | GPU服务器部署，追求最高精度 |

---

## 27. Llama Guard 3

| 维度 | 详情 |
|------|------|
| **全称** | Llama Guard 3 |
| **开发者** | Meta |
| **发布时间** | 2024年 |
| **论文** | [Llama Guard: LLM-based Input-Output Safeguard](https://arxiv.org/abs/2312.06674) |
| **架构** | LLaMA 3 (Decoder-only Transformer) |
| **参数量** | 8B |
| **支持语言** | 多语言（v3大幅增强） |
| **训练数据** | 安全分类体系（Taxonomy） + 违规样本 + 指令微调 |
| **输出格式** | `safe` 或 `unsafe` + 违规类别代码（如 S1=暴力, S2=性犯罪, S3=犯罪策划, ...） |
| **推理速度** | CPU不可用，GPU上单条~0.3-1秒 |
| **内存占用** | FP16: ~16GB / INT8: ~8GB |
| **HuggingFace** | `meta-llama/Llama-Guard-3-8B` |
| **是否可直接用** | ✅ 是 |
| **是否需要联网** | ❌ 否 |
| **关键优势** | ① 开源护栏中安全分类体系最完整最严谨 ② 输出违规类别代码，便于自动化后处理 ③ 多语言支持在v3大幅增强 |
| **关键局限** | ① 8B参数需要GPU，离线CPU部署困难 ② 生成式推理慢 ③ 规则体系和美国安全标准对齐，本地化需适配 |
| **适合场景** | 有GPU的服务器端AIGC安全护栏 |

---

## 28. Llama Guard 2

| 维度 | 详情 |
|------|------|
| **全称** | Llama Guard 2 |
| **参数量** | 8B |
| **支持语言** | 多语言（v2开始增强） |
| **HuggingFace** | `meta-llama/Llama-Guard-2-8B` |
| **关键优势** | v2已开始支持多语言 |
| **关键局限** | 已被v3超越 |
| **适合场景** | 已被v3取代，一般不再推荐 |

---

## 29. Llama Guard 1

| 维度 | 详情 |
|------|------|
| **参数量** | 7B |
| **支持语言** | 英文为主 |
| **HuggingFace** | `meta-llama/Llama-Guard-7b` |
| **关键优势** | 分类体系（Taxonomy）设计是经典参考 |
| **关键局限** | 多语言支持弱，已被v2/v3取代 |
| **适合场景** | 研究其分类体系设计思路 |

---

## 30. Qwen2.5-0.5B-Instruct

| 维度 | 详情 |
|------|------|
| **全称** | Qwen2.5-0.5B-Instruct |
| **开发者** | 阿里巴巴（通义千问） |
| **发布时间** | 2024年 |
| **论文** | [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.04749) |
| **架构** | Decoder-only Transformer (Qwen2.5) |
| **参数量** | 0.5B (5亿) |
| **支持语言** | 多语言（亚洲语系占比极高：中文、日文、韩文、印尼文、泰文、越南文等） |
| **训练数据** | 18T tokens预训练数据，亚洲语言占比远超LLaMA/Gemma系列 |
| **输出格式** | 文本输出，可通过System Prompt定制为JSON |
| **推理速度** | CPU上~20-50 tokens/s |
| **内存占用** | FP16: ~1GB / GGUF INT4: ~350MB |
| **HuggingFace** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **是否可直接用** | ✅ 是（通过Zero-shot Prompt） |
| **是否需要联网** | ❌ 否 |
| **关键优势** | ① 全文档中最小且可直接用的模型 ② 亚洲语系原生支持极好（训练数据占比高） ③ 通过System Prompt灵活定制审核规则 ④ GGUF版可在无GPU服务器流畅跑 |
| **关键局限** | ① 0.5B参数在复杂上下文上能力有限 ② 未专门做安全微调，依赖Prompt ③ 生成式推理比编码器慢 ④ Prompt设计好坏对效果影响大 |
| **适合场景** | 资源极度受限的离线审核 + 需要灵活调整审核规则 |

---

## 31. Qwen2.5-1.5B-Instruct

| 维度 | 详情 |
|------|------|
| **参数量** | 1.5B |
| **内存占用** | FP16: ~3GB / GGUF INT4: ~1GB |
| **HuggingFace** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **关键优势** | 比0.5B精度明显提升，推理速度仍可接受 |
| **关键局限** | 比编码器慢 |
| **适合场景** | 0.5B精度不够时的升级选择 |

---

## 32. Qwen2.5-3B-Instruct

| 维度 | 详情 |
|------|------|
| **参数量** | 3B |
| **内存占用** | FP16: ~6GB / GGUF INT4: ~2GB |
| **HuggingFace** | `Qwen/Qwen2.5-3B-Instruct` |
| **关键优势** | 可输出结构化JSON审核结果（分类+置信度+理由） |
| **适合场景** | 需要结构化审核输出的场景 |

---

## 33. Gemma-2-2B

| 维度 | 详情 |
|------|------|
| **全称** | Gemma 2 2B |
| **开发者** | Google DeepMind |
| **发布时间** | 2024年 |
| **论文** | [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) |
| **架构** | Decoder-only Transformer (Gemma 2) |
| **参数量** | 2.6B（官方称2B级别） |
| **支持语言** | 多语言 |
| **训练数据** | 多语言语料 |
| **关键优势** | 2B级别逻辑推理能力最强，处理上下文复杂的长文本好 |
| **关键局限** | ① 未专门做安全微调（除非用ShieldGemma） ② 需要Prompt来引导审核行为 |
| **HuggingFace** | `google/gemma-2-2b-it` |
| **适合场景** | 需要强推理的复杂长文本审核 |

---

## 34. Aya-23-8B

| 维度 | 详情 |
|------|------|
| **全称** | Aya-23 |
| **开发者** | Cohere For AI |
| **发布时间** | 2024年 |
| **论文** | [Aya Model: An Open-Access Collection and Multilingual Instruction Tuned Models](https://arxiv.org/abs/2402.07818) |
| **架构** | Cohere Command R-based (Decoder-only Transformer) |
| **参数量** | 8B |
| **支持语言** | 23种语言，大量亚洲语种（印尼语、泰语、马来语、越南语、阿拉伯语等） |
| **训练数据** | 大规模多语言指令微调数据（Aya Dataset） |
| **关键优势** | ① 非英语语言（尤其亚洲语言）的文化禁忌理解对这类模型最好 ② 零样本多语言分类能力强 ③ 覆盖23种语言是精挑细选的（非XLM-R那样100种但多数浅尝辄止） |
| **关键局限** | ① 8B需要GPU ② 非专门审核模型 |
| **HuggingFace** | `CohereForAI/aya-23-8b` |
| **适合场景** | 亚洲小语种Zero-shot审核 |

---

## 35. Aya Expanse-8B

| 维度 | 详情 |
|------|------|
| **全称** | Aya Expanse |
| **开发者** | Cohere For AI |
| **发布时间** | 2024年底 |
| **参数量** | 8B |
| **支持语言** | 23种语言 |
| **关键优势** | Aya-23的升级版，多语言性能进一步提升 |
| **HuggingFace** | `CohereForAI/aya-expanse-8b` |

---

## 36. WildGuard

| 维度 | 详情 |
|------|------|
| **全称** | WildGuard: Open Safety Moderation |
| **开发者** | Allen Institute for AI (AllenAI) + 华盛顿大学 |
| **发布时间** | 2024年 |
| **论文** | [WildGuard: Open One-Stop Moderation Tools](https://arxiv.org/abs/2406.18495) |
| **架构** | 基于LLaMA的Decoder-only |
| **参数量** | 7B |
| **训练数据** | 覆盖有害内容、越狱、提示词注入的标注数据 |
| **输出格式** | 三合一：有害内容检测 + 越狱检测 + 提示词注入检测 |
| **关键优势** | ① 单一模型覆盖三种安全威胁 ② 开源 ③ 学术界前沿方案 |
| **关键局限** | ① 多语言覆盖有限（英文为主） ② 7B需GPU |
| **HuggingFace** | `allenai/wildguard` |
| **适合场景** | Agent/AI系统的分层防御架构 |

---

## 37. Aegis-Guard

| 维度 | 详情 |
|------|------|
| **全称** | NVIDIA Aegis Content Safety |
| **开发者** | NVIDIA |
| **发布时间** | 2024年 |
| **架构** | NeMo Guardrails框架下的内容安全模型 |
| **参数量** | 多种配置 |
| **关键优势** | ① NVIDIA官方支持 ② 可自定义安全策略 ③ 与NeMo生态集成 |
| **HuggingFace** | `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0` |
| **适合场景** | NVIDIA生态下的内容安全 |

---

## 38. Granite Guardian

| 维度 | 详情 |
|------|------|
| **全称** | Granite Guardian |
| **开发者** | IBM |
| **发布时间** | 2024年 |
| **架构** | Granite系列 |
| **参数量** | 多种尺寸（含小模型） |
| **关键优势** | ① IBM开源 ② 覆盖风险检测、越狱、有害内容、RAG安全等多个维度 ③ 有较小版本 |
| **HuggingFace** | `ibm-granite/granite-guardian` 系列 |
| **适合场景** | 企业级多维度安全防护 |

---

## 39. Kyutai Hibiki

| 维度 | 详情 |
|------|------|
| **全称** | Hibiki (Kyutai) |
| **开发者** | Kyutai (法国非营利AI实验室) |
| **发布** | 2024-2025年 |
| **架构** | 编码器类多语言审核 |
| **关键优势** | 欧洲视角的安全标准，与美中模型形成互补 |
| **适合场景** | 需要欧洲安全合规视角 |

---

## 40. facebook/bart-large-mnli (零样本分类)

| 维度 | 详情 |
|------|------|
| **全称** | BART Large MNLI |
| **开发者** | Meta |
| **架构** | BART Encoder-Decoder，在MNLI上微调 |
| **参数量** | 407M |
| **支持语言** | 英文为主，多语言泛化有限 |
| **使用方式** | Zero-shot Classification：定义审核标签（如"hate speech", "violence", "sexual content"），模型直接给每个标签打分 |
| **HuggingFace** | `facebook/bart-large-mnli` |
| **关键优势** | ① 不需要审核训练数据，定义标签即可 ② 灵活应对新的审核维度 |
| **关键局限** | ① 多语言效果差（英文训练的NLI） ② 不适合高精度场景 |
| **适合场景** | 快速PoC / 临时审核需求 / 灵活标签探索 |

---

## 41. MoritzLaurer/DeBERTa-v3-base-mnli (零样本分类)

| 维度 | 详情 |
|------|------|
| **全称** | DeBERTa-v3-base MNLI |
| **开发者** | Moritz Laurer (社区) |
| **架构** | DeBERTa-v3-base + MNLI微调 |
| **参数量** | 184M |
| **HuggingFace** | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| **关键优势** | ① DeBERTa架构比BART快 ② Zero-shot分类 ③ 精度在NLI模型中较高 |
| **适合场景** | 同上，比BART-MNLI更高效的零样本分类 |

---

# 第四部分：新架构模型（替代Transformer的方向）

---

## 42. Mamba-2

| 维度 | 详情 |
|------|------|
| **全称** | Mamba-2 (Structured State Space Duality) |
| **开发者** | 学术界（Tri Dao / Albert Gu 等） |
| **发布时间** | 2024年 |
| **论文** | [Transformers are SSMs: Generalized Models and Efficient Algorithms](https://arxiv.org/abs/2405.21060) |
| **架构** | State Space Model (SSM)，线性复杂度 |
| **关键优势** | 推理时间与文本长度呈线性关系，超长文本审核有潜力 |
| **关键局限** | ① 文本分类的SSM应用仍在探索阶段 ② 预训练模型少 |
| **适合场景** | 超长文本有害内容审核的研究方向 |

---

## 43. RWKV-6

| 维度 | 详情 |
|------|------|
| **全称** | RWKV-6 (Receptance Weighted Key Value) |
| **架构** | 线性Transformer (RNN式高效推理 + Transformer式并行训练) |
| **关键优势** | 小尺寸版本适合边缘设备 |
| **适合场景** | 边缘设备文本审核 |

---

## 44. ByT5

| 维度 | 详情 |
|------|------|
| **全称** | ByT5: Towards a Token-Free Future |
| **开发者** | Google Research |
| **发布时间** | 2021年 |
| **论文** | [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://arxiv.org/abs/2105.13626) |
| **架构** | T5但输入/输出直接从UTF-8字节序列（无Tokenizer） |
| **参数量** | small: 300M / base: 580M / large: 1.2B |
| **支持语言** | 语言无关（字节级，天然多语言） |
| **关键优势** | ① **无词表限制，天然免疫OOV和拼写变异攻击** ② 任何语言/符号/Emoji都能处理 ③ 对对抗性文本（"@buse"代替"abuse"）极鲁棒 |
| **关键局限** | ① 字节序列长，推理比词表模型慢 ② 精度在常规文本上可能略低于词表模型 |
| **HuggingFace** | `google/byt5-small` / `google/byt5-base` |
| **适合场景** | 对抗性/变异文本检测的场景 |

---

## 45. CANINE

| 维度 | 详情 |
|------|------|
| **全称** | CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation |
| **开发者** | Google Research |
| **发布时间** | 2021年 |
| **论文** | [CANINE: Pre-training an Efficient Tokenization-Free Encoder](https://arxiv.org/abs/2103.06874) |
| **架构** | 字符级Transformer Encoder（无分词器） |
| **参数量** | 132M |
| **支持语言** | 语言无关 |
| **关键优势** | ① 字符级编码，直接处理Unicode字符 ② 比ByT5更高效（非Encoder-Decoder） ③ 免疫OOV |
| **HuggingFace** | `google/canine-s` / `google/canine-c` |
| **适合场景** | 替代方案：对抗性变形词检测 |

---

# 第五部分：商业API

---

## 46. Google Perspective API

| 维度 | 详情 |
|------|------|
| **开发商** | Jigsaw (Google子公司) |
| **费用** | 免费（有QPS限制） |
| **支持语言** | 多语言（原生支持印尼语、泰语等） |
| **输出格式** | 0-1之间的毒性概率分数 |
| **是否可离线** | ❌ 需联网 |
| **关键优势** | ① 完全免费 ② 多语言原生支持 ③ 可用来给历史数据打伪标签 |
| **关键局限** | ① 不可离线 ② API有速率限制 ③ 规则不透明 |
| **适合场景** | 离线模型的训练数据打标；线上兜底 |

---

## 47. OpenAI Moderation API

| 维度 | 详情 |
|------|------|
| **开发商** | OpenAI |
| **费用** | 免费 |
| **支持语言** | 多语言 |
| **输出格式** | 类别分数 + 整体判定 |
| **是否可离线** | ❌ 需联网 |
| **关键优势** | ① 底层多语言分类模型精度极高 ② AIGC生态首选的兜底方案 |
| **关键局限** | ① 不可离线 ② 数据发送到OpenAI服务器（隐私考虑） |
| **适合场景** | AIGC应用最后一道安全防线 |

---

## 48. Azure AI Content Safety

| 维度 | 详情 |
|------|------|
| **开发商** | Microsoft |
| **费用** | 付费（有免费额度） |
| **支持语言** | 多语言，多模态（文本+图像+代码） |
| **关键优势** | 企业合规级别，可自定义安全策略 |
| **关键局限** | 付费，不可离线 |
| **适合场景** | 企业级合规场景 |

---

# 第六部分：全量模型快速索引

| # | 模型 | 类型 | 可直接用 | 可离线 | 推荐优先级 |
|---|------|------|----------|--------|-----------|
| 1 | unitary/multilingual-toxic-xlm-roberta | 编码器成品 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| 2 | ShieldGemma-2B | 生成式护栏 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| 3 | Qwen2.5-0.5B-Instruct | 生成式通用 | ✅ (Zero-shot) | ✅ | ⭐⭐⭐⭐ |
| 4 | mDeBERTa-v3-small/base | 编码器底座 | ❌ 需微调 | ✅ | ⭐⭐⭐⭐ |
| 5 | XLM-RoBERTa-base | 编码器底座 | ❌ 需微调 | ✅ | ⭐⭐⭐⭐ |
| 6 | Llama Guard 3-8B | 生成式护栏 | ✅ | ✅ (需GPU) | ⭐⭐⭐ |
| 7 | ModernBERT-base | 编码器底座 | ❌ 需微调 | ✅ | ⭐⭐⭐ |
| 8 | IndoBERT / WangchanBERTa | 编码器底座 | ❌ 需微调 | ✅ | ⭐⭐⭐ (单语种) |
| 9 | LionGuard | 编码器成品 | ✅ | ✅ | ⭐⭐⭐ (需核实) |
| 10 | Granite Guardian | 生成式护栏 | ✅ | ✅ | ⭐⭐⭐ |
| 11 | Aegis-Guard | 生成式护栏 | ✅ | ✅ (需GPU) | ⭐⭐⭐ |
| 12 | Perspective API | API | ✅ | ❌ | ⭐⭐ |
| 13 | OpenAI Moderation API | API | ✅ | ❌ | ⭐⭐ |
| 14 | mBERT | 编码器底座 | ❌ | ✅ | ⭐⭐ (基线) |
| 15 | DistilBERT-multilingual | 编码器底座 | ❌ | ✅ | ⭐⭐ |
| 16 | ByT5 / CANINE | 编码器底座 | ❌ | ✅ | ⭐⭐ (对抗性) |
| 17 | BGE-M3 | Embedding | ❌ | ✅ | ⭐⭐ |
| 18 | LaBSE | Embedding | ❌ | ✅ | ⭐⭐ |
| 19 | BART-MNLI / DeBERTa-MNLI | 零样本分类 | ✅ | ✅ | ⭐⭐ |
| 20 | WildGuard | 生成式护栏 | ✅ | ✅ (需GPU) | ⭐⭐ |
| 21 | Aya-23-8B | 生成式通用 | ✅ (Zero-shot) | ✅ (需GPU) | ⭐⭐ |
| 22 | Qwen2.5-1.5B/3B | 生成式通用 | ✅ (Zero-shot) | ✅ | ⭐⭐ |
| 23 | Gemma-2-2B | 生成式通用 | ✅ (Zero-shot) | ✅ | ⭐⭐ |
| 24 | RemBERT / XLM-V | 编码器底座 | ❌ | ✅ | ⭐ |
| 25 | Glot500 / AfriBERTa / PhoBERT | 编码器底座 | ❌ | ✅ | ⭐ (特殊语言) |
| 26 | Mamba-2 / RWKV-6 | 新架构 | ❌ | ✅ | ⭐ (探索性) |
