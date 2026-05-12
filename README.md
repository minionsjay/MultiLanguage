# 多语言内容审核 — 小模型研究知识库

一个专注于**多语言文本内容审核**的研究知识库，覆盖小模型选型、离线部署策略、代码示例、前沿论文，面向印尼语、泰语、马来语、新加坡英语（Singlish）、巴西葡萄牙语等非英语语种的有害/仇恨言论检测。

## 目录结构

```
.
├── Model.md                    # 核心知识文档 — 结构化 Q&A：模型选型、部署方案、研究方向
├── model_comparison.md         # 全量模型对比表：编码器、生成式、开箱即用模型
├── model_details.md            # 模型详细档案：架构、训练数据、性能、HuggingFace 地址
├── model_code.md               # 完整代码：推理、微调、ONNX 导出、部署示例
├── model_relationship.md       # 模型演化树 & 家族关系图谱
├── recent_papers.md            # 最新论文综述（2024–2026）：安全护栏、数据合成、对抗防御
├── sea_lion_ecosystem.md       # AI Singapore SEA-LION 生态深度分析
├── charts.py                   # 可视化图表生成脚本
├── charts/                     # 11 张生成的可视化 PNG 图表
└── CLAUDE.md                   # Claude Code 项目配置
```

## 覆盖主题

### 模型选型
- **编码器模型（Encoder-only）** — XLM-RoBERTa、mDeBERTa-v3、ModernBERT、SEA-LION、IndoBERT、WangchanBERTa
- **生成式安全护栏模型** — ShieldGemma（2B）、Qwen2.5（0.5B–3B）、Llama Guard 3、WildGuard
- **开箱即用分类器** — `unitary/multilingual-toxic-xlm-roberta`、LionGuard/SEA-Guard

### 离线部署与效率优化
- ONNX 导出，脱离 PyTorch 环境运行
- GGUF 量化，基于 llama.cpp 在纯 CPU 环境推理
- 动态量化（FP32 → INT8），精度损失 <1%，速度提升 2–3 倍
- LoRA 区域适配器，实现文化感知分类

### 数据工程
- 基于大模型（LLM）的合成数据蒸馏流水线
- LLM-as-a-Judge 自动化数据打标
- Focal Loss 处理极度不平衡数据集
- 对抗性数据增强，防御语码切换和符号替换

### 前沿研究方向
- 文化感知的跨语言安全对齐
- Late Chunking 与 Matryoshka 表示学习（MRL）
- 字节级/字符级编码器防御对抗性文本变异
- 统一安全护栏模型（单一模型同时检测有害内容 + 越狱攻击 + 提示词注入）

## 快速开始

浏览文档：

```bash
# 从核心 Q&A 文档开始
cat Model.md

# 快速对比所有模型
cat model_comparison.md

# 查看模型详细档案
cat model_details.md

# 运行代码示例
cat model_code.md
```

生成可视化图表：

```bash
pip install matplotlib numpy
python charts.py
# 输出 11 张 PNG 到 charts/ 目录
```

## 场景推荐速查表

| 场景 | 推荐模型 | 部署要点 |
|------|----------|----------|
| 离线 CPU、覆盖 100+ 语言 | `unitary/multilingual-toxic-xlm-roberta`（ONNX） | ~280MB，<10ms/条 |
| 离线 CPU、需要输出违规理由 | `Qwen2.5-0.5B-Instruct`（GGUF INT8） | ~1GB，零样本 Prompt |
| 东南亚语种（印尼/泰/马来/新加坡） | SEA-LION-ModernBERT-300M + LoRA | ~600MB，本地文化感知 |
| AIGC 安全护栏 | `ShieldGemma-2B`（GGUF INT8） | <3GB，拦截提示词注入 |
| 高并发 GPU 批量审核 | `mDeBERTa-v3-small` → ONNX → TensorRT | 极致吞吐量 |

## 参考文献

本知识库综合了以下研究：

- **SEA-Guard**（AI Singapore, 2026） — 面向东南亚的文化感知多语言安全护栏
- **WildGuard**（AllenAI, NeurIPS 2024） — 三合一安全审核：有害内容 + 越狱 + 拒绝检测
- **ShieldGemma**（Google, 2024） — 2B 级生成式内容安全分类器
- **Llama Guard 1/2/3**（Meta, 2023–2024） — 基于分类树的安全检测体系
- **NusaX** — 印尼语多方言平行语料库
- **HateBR** — 巴西葡萄牙语仇恨言论检测数据集

完整论文列表见 `recent_papers.md`。

## 许可

本项目为研究资源。使用时请参考各模型的原始许可协议（Apache 2.0、MIT、Gemma License、Llama Community License 等）。
