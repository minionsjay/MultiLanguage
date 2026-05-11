# 多语言内容审核 — 最新论文综述 (2024-2026)

> 与 Gemini 第五轮推荐的论文相比，以下是基于实际检索到的、更完整的最新论文列表。
> 标注 ⭐ 的是 Gemini 没提到的重要论文。

---

## 一、安全护栏 & 审核模型（核心必读）

### 1.1 ⭐ SEA-Guard: Culturally Grounded Multilingual Safeguard for Southeast Asia
- **时间**: 2026年2月
- **作者**: Panuthep Tasawong, Jian Gang Ngui, Alham Fikri Aji, Trevor Cohn, Peerat Limkonchotiwat
- **论文**: [arXiv:2602.01618](https://arxiv.org/abs/2602.01618)
- **解决了什么问题**: 现有安全护栏依赖英文数据的机器翻译，无法捕捉东南亚文化细微差异
- **核心贡献**:
  - 提出 agentic data-generation framework，自动化生成符合东南亚文化背景的安全数据
  - 发布 SEA-Guard 模型家族（Llama-8B / Qwen-4B / Qwen-8B / Gemma-12B）
  - 在检测区域敏感内容上一致优于现有护栏模型，同时保持通用安全性能
- **与你需求的关联**: 这是目前唯一针对东南亚 8 种语言的开源安全护栏

### 1.2 ⭐ WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs
- **时间**: 2024年6月 (NeurIPS 2024)
- **作者**: Seungju Han, Kavel Rao, Allyson Ettinger 等 (AllenAI + 华盛顿大学)
- **论文**: [arXiv:2406.18495](https://arxiv.org/abs/2406.18495)
- **解决了什么问题**: 之前需要三个独立模型分别检测有害内容、越狱攻击和拒绝回答，WildGuard 用一个轻量模型统一搞定
- **核心贡献**:
  - 单一模型同时做三件事：检测 prompt 恶意、检测 response 安全风险、判断模型是否拒绝
  - 发布 WildGuardMix（92K 标注样本，含直接 prompt 和对抗越狱）
  - 将越狱攻击成功率从 79.8% 降到 2.4%
  - 覆盖 13 个风险类别
- **与你需求的关联**: 如果你的审核需要防御 Prompt 注入 + 越狱 + 有害内容，这是一个模型搞定三个任务的方案

### 1.3 ShieldGemma: Generative AI Content Moderation Based on Gemma
- **时间**: 2024年7月
- **作者**: Wenjun Zeng, Yuchi Liu, Ryan Mullins 等 (Google)
- **论文**: [arXiv:2407.21772](https://arxiv.org/abs/2407.21772)
- **核心贡献**:
  - 基于 Gemma2 的内容安全分类器，覆盖色情/危险内容/骚扰/仇恨言论
  - 提出 LLM-based 数据合成 pipeline，主要用合成数据训练
  - 相比 Llama Guard 提升 10.8% AU-PRC，相比 WildCard 提升 4.3%
  - 2B 版本可量化到 <3GB 离线部署
- **与你需求的关联**: 目前最小的专用安全护栏（2B），适合离线 CPU 部署

### 1.4 ⭐ Llama Guard 3 (及 v1/v2 系列)
- **时间**: v1: 2023年12月 / v2: 2024年 / v3: 2024年底
- **作者**: Hakan Inan, Kartikeya Upasani 等 (Meta)
- **论文 (v1)** : [arXiv:2312.06674](https://arxiv.org/abs/2312.06674)
- **核心贡献**:
  - 将内容审核定义为 LLM 可理解的分类任务
  - 定义了安全风险分类树（Taxonomy）：暴力(S1)、非暴力犯罪(S2)、性犯罪(S3)等 14 个类别
  - v3 大幅增强了多语言支持
  - 输出 `safe` 或 `unsafe` + 违规类别代码
- **与你需求的关联**: 安全分类体系设计的教科书级参考

---

## 二、东南亚语种评估基准 & 数据集

### 2.1 ⭐ SEAHateCheck: Functional Tests for Detecting Hate Speech in Low-Resource Languages of Southeast Asia
- **时间**: 2026年3月 (TALLIP 接收)
- **作者**: Ri Chi Ng, Aditi Kumaresan, Yujia Hu, Roy Ka-Wei Lee
- **论文**: [arXiv:2603.16070](https://arxiv.org/abs/2603.16070)
- **语言**: 印尼语、他加禄语、泰语、越南语
- **核心贡献**:
  - 首个针对东南亚低资源语言的仇恨言论功能测试套件
  - 基于 HateCheck 框架，由 LLM 扩充 + 本地专家验证
  - **关键发现**: 他加禄语模型准确率最低；俚语测试最难；模型在隐性仇恨和反讽上普遍失败
- **与你需求的关联**: 如果你要评估你的审核模型在东南亚语种上的真正能力，这是必用基准

### 2.2 ⭐ SEA-SafeguardBench: Evaluating AI Safety in SEA Languages and Cultures
- **时间**: 2025年12月
- **作者**: Panuthep Tasawong, Jian Gang Ngui 等 (AI Singapore)
- **论文**: [arXiv:2512.05501](https://arxiv.org/abs/2512.05501)
- **语言**: 8 种东南亚语言
- **核心贡献**:
  - 首个经人工验证的东南亚安全基准，21,640 个样本，3 个子集 (General / In-the-wild / Content Generation)
  - 发现即使顶尖 LLM 和护栏模型在东南亚文化场景中也会显著失败
  - 证明机器翻译的英文数据无法替代本地标注
- **与你需求的关联**: 评估任何多语言安全模型在东南亚市场表现的权威基准

### 2.3 ⭐ SEA-BED: How Do Embedding Models Represent Southeast Asian Languages?
- **时间**: 2025年8月
- **作者**: Wuttikorn Ponwitayarat, Peerat Limkonchotiwat 等 (AI Singapore)
- **论文**: [arXiv:2508.12243](https://arxiv.org/abs/2508.12243)
- **语言**: 10 种东南亚语言，169 个数据集
- **核心贡献**:
  - 大规模东南亚语言嵌入基准，覆盖检索/相似度/聚类/分类
  - **关键发现**: 没有一个模型在所有东南亚语言上表现一致；任务难度在语言间差异极大
- **与你需求的关联**: 如果你的审核系统包含跨语言检索或相似度匹配，必看

### 2.4 ⭐ Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection
- **时间**: 2024年11月 (NAACL 2025)
- **作者**: Minh Duc Bui, Katharina von der Wense, Anne Lauscher
- **论文**: [arXiv:2411.03888](https://arxiv.org/abs/2411.03888)
- **语言**: 英语、德语、西班牙语、印地语、中文
- **核心贡献**:
  - 首个多模态 + 多语言 + 多文化并行仇恨言论数据集
  - **关键发现**: 不同文化背景的标注者之间平均一致性仅 74%（美国-印度低至 67%）
  - VLM 零样本检测存在美国中心偏见（US-centric bias）
- **与你需求的关联**: 证明了"文化背景决定仇恨言论判断"——这正是为什么你需要区域特化模型

---

## 三、高效架构 & 跨语言迁移

### 3.1 ⭐ Boosting Accuracy and Interpretability in Multilingual Hate Speech Detection Through Layer Freezing and Explainable AI
- **时间**: 2026年1月
- **作者**: Meysam Shirdel Bilehsavar 等
- **论文**: [arXiv:2601.02697](https://arxiv.org/abs/2601.02697)
- **语言**: 英语、韩语、日语、中文、法语
- **核心贡献**:
  - 冻结前 8 层的参数效率策略在 mBERT/RoBERTa/XLM-R 上的系统评估
  - 使用 LIME 进行可解释性分析，展示哪些词触发了判定
  - 证明冻结底层可以保持甚至提升性能，同时大幅减少计算成本
- **与你需求的关联**: 如果你要用 LoRA 或冻结策略微调多语言底座，这篇提供了参数效率的实证依据

### 3.2 ⭐ Comparison of Modern Multilingual Text Embedding Techniques
- **时间**: 2026年4月
- **论文**: [arXiv:2604.14907](https://arxiv.org/abs/2604.14907)
- **核心贡献**: 对当前主流多语言文本嵌入技术的系统性比较
- **与你需求的关联**: 如果你的审核系统需要 Embedding 做跨语言相似度检索

---

## 四、Gemini 第五轮提到的论文（核实 & 补充）

| Gemini 推荐的论文 | 核实状态 | 评注 |
|------------------|---------|------|
| SEA-LION Technical Report | ✅ 真实 | AI Singapore 官方技术报告，确实详细描述了分词器构建和语料清洗 |
| NusaX | ✅ 真实 | 印尼语多语言平行语料基准，对印尼方言覆盖好 |
| HateBR | ✅ 真实 | 巴西葡语仇恨言论数据集，但偏向拉美，与东南亚关联不大 |
| Llama Guard 论文 | ✅ 真实 | 如上所述，安全分类体系经典 |
| ShieldGemma 论文 | ✅ 真实 | 如上所述，小模型安全护栏 |
| "Cross-Lingual Transfer Learning for Hate Speech Detection" | ⚠️ 模糊 | 非具体论文，是研究方向名称。建议替换为：SEAHateCheck (2603.16070) 或上面第 3.1 节 |
| Aya Model 论文 | ✅ 真实 | Cohere 多语言指令微调，数据合成 pipeline 有参考价值 |
| Jailbreaking Black Box LLMs | ✅ 真实 | 但偏向 LLM 攻击，与多语言仇恨文本检测关联较弱 |

---

## 五、按场景的论文阅读路线

### 如果你要做东南亚多语言审核系统
```
必读路线:
1. SEA-Guard (2602.01618) → 了解怎么做东南亚安全对齐
2. SEA-SafeguardBench (2512.05501) → 了解用什么基准评估
3. SEAHateCheck (2603.16070) → 了解模型在哪些方面会失败
4. SEA-BED (2508.12243) → 了解 Embedding 在东南亚语言的真实表现
```

### 如果你要构建通用的多语言安全护栏
```
必读路线:
1. WildGuard (2406.18495) → 统一护栏设计的参考
2. ShieldGemma (2407.21772) → 小模型护栏+合成数据 pipeline
3. Llama Guard (2312.06674) → 安全分类体系设计
4. Multi3Hate (2411.03888) → 理解文化差异对标注的影响
```

### 如果你要高效微调小模型做审核
```
必读路线:
1. 2601.02697 → 层冻结策略的实证
2. SEA-LION Technical Report → 分词器构建
3. Aya Model (2402.07818) → 数据合成+蒸馏
```

---

## 六、Gemini 第五轮回答的最终评价

| 维度 | 评价 |
|------|------|
| 分类框架 | ✅ 四个方向划分合理 |
| 论文具体性 | ❌ "Cross-Lingual Transfer Learning for Hate Speech Detection" 不是具体论文 |
| 完整性 | ❌ 漏了 SEA-Guard、SEAHateCheck、SEA-SafeguardBench、WildGuard、Multi3Hate 等核心论文 |
| 相关性 | ⚠️ HateBR（巴西）和 Jailbreaking（LLM攻击）与多语言文本审核关联较弱 |
| 可操作性 | ❌ 没有给出论文之间的阅读顺序或选择指南 |
