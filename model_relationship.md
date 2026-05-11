# 模型关系图谱 & 全景对比

---

## 一、模型家族演化树

```
                         Transformer (2017)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   [Encoder-only]        [Encoder-Decoder]     [Decoder-only]
   纯分类/理解             分类+生成              纯生成
        │                     │                     │
   ┌────┴────┐           ┌────┴────┐           ┌────┴─────┐
  BERT    RoBERTa         T5       BART       GPT      LLaMA
 (2018)   (2019)        (2019)    (2019)     (2018)    (2023)
   │         │              │                    │         │
   ├─mBERT   ├─XLM-R        ├─mT5               ├─Qwen    ├─Llama Guard 1/2/3
   │ (104语) │ (100语)      │ (101语)            │ (亚洲)   │ (安全护栏)
   │         │              │                    │         │
   ├─Distil-│ ├─XLM-R-large│ ├─ByT5            ├─SEA-LION ├─WildGuard
   │ BERT    │ │ (560M)     │ │ (字节级,无词表) │ (东南亚) │ (三合一检测)
   │ (蒸馏)  │ │            │                    │         │
   │         │ ├─XLM-V     │                    │         │
   ├─IndoBERT│ │ (901K词表)│                    │         │
   │ (印尼语)│ │            │                    │         │
   │         │ ├─BGE-M3    │                    │
   ├─Wang-  │ │ (Embedding)│                    │
   │ chanBERTa│ │           │                    │
   │ (泰语)  │ │            │                    │
   │         │ ├─unitary/  │                    │
   ├─PhoBERT│ │ toxic-xlm  │                    │
   │ (越南语)│ │ (开箱即用) │                    │
   │         │             │                    │
   ├─AfriBERTa│            │                    │
   │ (非洲语)│             │                    │
   │         │             │                    │
   ├─HateXplain│          │                    │
   │ (可解释) │             │                    │
   │         │             │                    │
   └─CANINE │              │                    │
    (字符级) │              │                    │
             │              │                    │
        DeBERTa         ModernBERT              Gemma
        (2020)           (2024)                (2024)
           │                │                    │
        mDeBERTa-v3      [多语言版               ├─Gemma-2-2B
        (100+语, 最优)   开发中]                 │  (逻辑强)
           │                                     │
        ┌──┴──┐                                ShieldGemma
      base  small                               (2B/9B,安全)
     (279M) (141M)                                │
                                               Aya-23/Expanse
                                                  │
                                               (Cohere,23语)


                   [新架构 — 替代Transformer]
                          │
              ┌───────────┼───────────┐
             SSM          RWKV        xLSTM
              │            │            │
          Mamba-2      RWKV-6      (探索中)
        (线性复杂度)  (边缘设备)
```

---

## 二、按使用方式分类

```
                        你要做内容审核
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         有标注数据?      没标注数据?      API兜底/打标
              │               │               │
         [微调底座]      [开箱即用]         [商业API]
              │               │               │
    ┌─────────┼─────────┐     │        ┌──────┼──────┐
    │         │         │     │        │      │      │
  通用      区域      对抗性   │    Perspective OpenAI Azure
  底座      底座      底座     │      API免费   API免费  付费
    │         │         │     │
 XLM-R    IndoBERT   ByT5    ├── 编码器成品 ──┐
 mDeBERTa Wangchan   CANINE  │                │
 mBERT    BERTa               │                │
 RemBERT  SEA-LION            │                │
 XLM-V    PhoBERT             │                │
    │         │               │                │
    └────┬────┘         ┌─────┴──────┐  ┌──────┴──────┐
         │              │            │  │              │
    微调后自己用    unitary/     LionGuard   HateXplain
                  toxic-xlm    (东南亚)     (可解释性)
                  (100+语)

                    [零样本方案]
                         │
              ┌──────────┼──────────┐
              │          │          │
         BART-MNLI  DeBERTa-MNLI  Qwen(Zero-shot)
         (英文为主)  (英文为主)     (亚洲语强)

                    [安全护栏 — 生成式]
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ShieldGemma    Llama Guard 3    Granite Guardian
      (2B/9B)         (8B)            (IBM)
    Google安全      Meta分类体系      企业合规

                    [三合一威胁检测]
                         │
                    WildGuard
                  (有害+越狱+注入)
```

---

## 三、核心模型能力雷达图（文字版）

### 3.1 编码器底座对比

```
                   多语言覆盖  推理速度  东南亚适配  生态完善度  资源消耗
                   (0-10)     (0-10)     (0-10)     (0-10)     (越低越好)
XLM-RoBERTa-base   ████████   ████████   ██████     ██████████  ████
                   (9)        (8)        (6)        (10)        (4)

mDeBERTa-v3-small  ████████   ██████████ ██████     ██████      ██
                   (9)        (10)       (6)        (6)         (2)

mBERT              ████████   ████████   █████      ████████    ███
                   (8)        (8)        (5)        (8)         (3)

ModernBERT-base    ████       ██████████ ██         ███         ██
                   (4)        (10)       (2)        (3)         (2)

SEA-LION-7B        █████      ████       ██████████ █████       ██████████
                   (5)        (4)        (10)       (5)         (10)

IndoBERT           ██         ██████████ ██████████ ██████      ██
                   (2)        (10)       (10)       (6)         (2)

BGE-M3             ████████   ██████     ██████     ██████      ██████
                   (9)        (6)        (6)        (6)         (6)
```

### 3.2 开箱即用模型对比

```
                   多语言  精度  速度  可解释性  离线可用  易部署性
                   (0-10)  (0-10) (0-10) (0-10)  (0-10)   (0-10)
unitary/toxic-     ████████ ██████ ██████████ ████  ██████████ ██████████
xlm-roberta        (9)      (7)    (10)     (4)    (10)       (10)

ShieldGemma-2B     ███████  ████████ ████   ████  ██████████ ██████
                   (7)      (8)    (4)      (4)   (10)       (6)

Qwen2.5-0.5B       ████████ ██████  ███    █████ ██████████ ████████
-Instruct          (8)      (6)    (3)     (5)   (10)       (8)

Llama Guard 3-8B   ████████ █████████ ███  ██████████ ██████  ████
                   (8)      (9)    (3)     (10)  (7)       (4)

LionGuard          █████    ███████  ███████ ████  ████████  ███████
                   (5)      (7)    (7)     (4)   (8)       (7)

WildGuard           ████    ████████ ███  █████  ██████   ████
                   (4)      (8)    (3)     (5)   (7)       (4)
```

---

## 四、模型选择决策树

```
                        开始: 选择内容审核模型
                              │
              ┌───────────────┼───────────────┐
              │               │               │
          需要离线?      需要输出理由?    需要多快?
              │               │               │
    ┌──── yes └── no → API    │        ┌─ <10ms/条
    │             方案         │        │
    │                         │        ├─ 编码器
    │              ┌── yes ───┘        │   (unitary/toxic,
    │              │                   │    mDeBERTa)
    │              │ 生成式模型         │
    │              ├─ ShieldGemma       ├─ <100ms/条
    │              ├─ Qwen              │
    │              └─ Llama Guard       ├─ GGUF量化生成式
    │                                  │   (Qwen 0.5B,
    │              ┌── no               │    ShieldGemma)
    │              │                   │
    │              └─ 编码器输出分数     └─ >100ms/条
    │                 (unitary/toxic)       │
    │                                       └─ 完整生成式
    │                                          (Llama Guard)
    │
    ├── 资源有多紧?
    │
    ├── <2GB内存, 纯CPU ──→ Qwen2.5-0.5B GGUF 或 mDeBERTa-v3-small ONNX INT8
    │
    ├── <4GB内存, 纯CPU ──→ ShieldGemma-2B GGUF INT4 或 unitary/toxic ONNX
    │
    ├── 有GPU (≥8GB)  ──→ Llama Guard 3 或 ShieldGemma-9B
    │
    └── 有GPU集群     ──→ vLLM + 任意大模型批量推理
    │
    ├── 覆盖哪些语言?
    │
    ├── 全球100+语言   ──→ unitary/toxic-xlm 或 mDeBERTa-v3
    │
    ├── 东南亚专精     ──→ SEA-LION + LionGuard 或 XLM-R + 区域LoRA
    │
    ├── 纯印尼语       ──→ IndoBERT
    │
    ├── 纯泰语         ──→ WangchanBERTa
    │
    ├── 纯越南语       ──→ PhoBERT
    │
    └── 混合/变形文本  ──→ ByT5 或 CANINE
    │
    ├── 多久上线?
    │
    ├── 今天下午       ──→ unitary/toxic ONNX 离线部署
    │
    ├── 这周内         ──→ Qwen Zero-shot + 调Prompt
    │
    ├── 这个月         ──→ mDeBERTa + LoRA微调
    │
    └── 长期迭代       ──→ 合成数据蒸馏 → 小模型 (完整流水线)
```

---

## 五、全量模型关系矩阵

```
                        ┌─────────── 预训练底座 (需微调) ───────────┐
                        │                                           │
                  [BERT系]                                    [RoBERTa系]
                   mBERT                                      XLM-R-base
                    │                                           │
            ┌───────┼───────┐                          ┌────────┼────────┐
            │       │       │                          │        │        │
        Distil-  IndoBERT CANINE                   XLM-R-lg  XLM-V   BGE-M3
        BERT     (印尼语) (字符级)                  (560M)  (901K词表) (检索)
                  │
            ┌─────┴─────┐
        WangchanBERTa  PhoBERT
          (泰语)        (越南语)


              [DeBERTa系]           [T5系]           [Modern系]
           mDeBERTa-v3              mT5             ModernBERT
           ┌─────┴─────┐        ┌───┴───┐          (英文,8192ctx)
         base        small     ByT5    T5
        (279M)      (141M)   (字节级)


        ┌─────────── 开箱即用 (直接可用) ───────────┐
        │                                            │
   [纯编码器]                              [生成式安全护栏]
   unitary/toxic-xlm                        │
   (6维毒性分数,100+语)            ┌─────────┼─────────┐
   NSFW-text-classifier            │         │         │
   HateXplain (可解释)        ShieldGemma Llama Guard WildGuard
   LionGuard (东南亚)          (2B/9B)   (1/2/3,8B)  (三合一)


        ┌─────────── 生成式通用 (Zero-shot可用) ────────┐
        │                                                │
      [Qwen系]          [Gemma系]        [Cohere系]
   Qwen2.5系列          Gemma-2-2B        Aya-23-8B
   ├─ 0.5B-Instruct    (2B级逻辑最强)    (23语,亚洲强)
   ├─ 1.5B-Instruct                      Aya Expanse-8B
   └─ 3B-Instruct                        (升级版)


        ┌─────────── 零样本方案 ────────────┐
        │                                    │
   BART-MNLI (英文)        DeBERTa-MNLI (更准更快)
   Zero-shot分类            Zero-shot分类


        ┌─────────── 新架构 (探索性) ────────────┐
        │                                        │
   Mamba-2 (SSM)        RWKV-6             xLSTM
   线性复杂度            边缘友好            经典复兴


        ┌─────────── 商业API ────────────┐
        │                                │
   Perspective API    OpenAI Moderation    Azure Content Safety
   (Google,免费)      (OpenAI,免费)       (Microsoft,付费)
```

---

## 六、按场景的最佳组合方案

```
┌─────────────────────────────────────────────────────────────────┐
│                    场景1: 社交平台 UGC 高并发审核                │
│                                                                  │
│  主模型: unitary/multilingual-toxic-xlm-roberta (ONNX)          │
│  架构:   Encoder-only, 6维毒性分数                               │
│  语言:   100+ 语言                                               │
│  速度:   <10ms/条 (CPU)                                         │
│  部署:   C++ ONNX Runtime + FastAPI                              │
│                                                                  │
│  兜底:   Perspective API (打标评估)                              │
│  升级:   区域LoRA (mDeBERTa-v3 + 各地LoRA)                      │
│                                                                  │
│  架构图:                                                         │
│  ┌─────────┐   ┌──────────┐   ┌──────┐   ┌───────────┐         │
│  │ 用户文本 │──▶│ 文本归一化 │──▶│ ONNX │──▶│ 6维毒性分 │         │
│  └─────────┘   │ (去零宽等) │   │ 推理  │   │ 数 > 阈值?│         │
│                └──────────┘   └──────┘   └─────┬─────┘         │
│                                                │                │
│                                     ┌──────────┴──────────┐    │
│                                   放行                   拦截    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 场景2: AIGC 应用安全护栏 (双层防御)              │
│                                                                  │
│  第一层 (输入): ShieldGemma-2B (GGUF INT4)                      │
│  检测:   Prompt注入, 越狱攻击, 有害请求                          │
│  速度:   ~200ms/条 (CPU)                                        │
│                                                                  │
│  第二层 (输出): Llama Guard 3-8B (vLLM, GPU)                    │
│  检测:   AI生成内容是否安全 (S1-S14分类)                         │
│  速度:   ~300ms/条 (GPU批量)                                    │
│                                                                  │
│  架构图:                                                         │
│  ┌─────────┐   ┌──────────────┐   ┌───────────┐   ┌─────────┐  │
│  │ 用户输入 │──▶│ShieldGemma-2B│──▶│  大模型应用 │──▶│ Llama   │  │
│  └─────────┘   │ 输入过滤      │   │ (Agent/RAG)│   │ Guard 3 │  │
│                │ Safe/Unsafe   │   └───────────┘   │ 输出过滤 │  │
│                └──────────────┘                    └────┬────┘  │
│                    │                                   │        │
│              Unsafe → 拦截                       Safe → 返回用户 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 场景3: 东南亚多国出海审核                         │
│                                                                  │
│  底座:   mDeBERTa-v3-small (共享, ONNX导出)                      │
│  适配:   各国独立LoRA模块 (每个~6MB)                              │
│                                                                  │
│  印尼LoRA ─── 理解: Bahasa Gaul, 本地侮辱词汇                    │
│  泰国LoRA ─── 理解: 泰文拼写变异, 皇室敏感话题                    │
│  新加坡LoRA ── 理解: Singlish, 种族宗教融合语境                   │
│  巴西LoRA ─── 理解: 政客侮辱俚语, 区域性骚扰词汇                  │
│                                                                  │
│  架构图:                                                         │
│                    ┌──────────────────┐                          │
│                    │ mDeBERTa-v3-small │                          │
│                    │  (共享底座, 冻结)  │                          │
│                    └────────┬─────────┘                          │
│                             │                                    │
│            ┌────────────────┼────────────────┐                   │
│            │                │                │                   │
│       ┌────┴────┐     ┌────┴────┐     ┌────┴────┐              │
│       │ 印尼LoRA │     │ 泰国LoRA │     │ 新加坡  │   ...        │
│       │  (6MB)   │     │  (6MB)   │     │ LoRA    │              │
│       └────┬────┘     └────┬────┘     └────┬────┘              │
│            │                │                │                   │
│       ┌────┴────┐     ┌────┴────┐     ┌────┴────┐              │
│       │ 违规分类 │     │ 违规分类 │     │ 违规分类 │              │
│       │ (印尼)   │     │ (泰国)   │     │ (新加坡) │              │
│       └─────────┘     └─────────┘     └─────────┘              │
│                                                                  │
│  路由: 根据用户IP/语言标签 → 加载对应LoRA → 推理                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                场景4: 极低成本离线边缘设备                        │
│                                                                  │
│  (IoT网关 / 树莓派 / 旧笔记本 / 无GPU服务器)                     │
│                                                                  │
│  方案A: Qwen2.5-0.5B-Instruct GGUF INT4                          │
│  - 内存: ~350MB                                                  │
│  - 语言: 亚洲语系优秀                                             │
│  - 方式: Zero-shot Prompt                                        │
│  - 特点: 灵活调整审核规则                                         │
│                                                                  │
│  方案B: mDeBERTa-v3-small ONNX INT8                              │
│  - 内存: ~140MB                                                  │
│  - 语言: 100+语                                                  │
│  - 方式: 微调后导出                                               │
│  - 特点: 极速推理(几ms)                                           │
│                                                                  │
│  方案C: DistilBERT-multilingual ONNX                             │
│  - 内存: ~135MB                                                  │
│  - 语言: 104语                                                   │
│  - 方式: 微调后导出                                               │
│  - 特点: 最快但精度最低                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              场景5: 对抗性变形文本检测                            │
│                                                                  │
│  问题: 用户用 "@", "4", 零宽字符, 同形异字 绕过审核              │
│  例: "@bu$e" 代替 "abuse", "bunuh" → "ßûñüh"                    │
│                                                                  │
│  流水线:                                                         │
│  ┌─────────┐   ┌──────────────┐   ┌──────────┐   ┌───────────┐ │
│  │ 原始输入 │──▶│ 文本归一化    │──▶│ ByT5 或  │──▶│ 分类结果   │ │
│  │          │   │ - 去零宽字符  │   │ CANINE   │   │           │ │
│  │          │   │ - 全角转半角  │   │ 字节级   │   │           │ │
│  │          │   │ - 同形字映射  │   │ 编码器   │   │           │ │
│  │          │   │ - Emoji翻译   │   │          │   │           │ │
│  └─────────┘   └──────────────┘   └──────────┘   └───────────┘ │
│                                                                  │
│  增强: 训练时注入对抗样本 (随机删元音, 形似符号替换, 混合语种)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 七、模型参数规模 — 性能 — 易用性 三维对比

```
        易用性 (开箱即用程度)
        │
     10 ┤   ★ Perspective API
        │   ★ OpenAI Moderation API
        │
      9 ┤   ★ unitary/multilingual-toxic-xlm-roberta
        │
      8 ┤   ★ ShieldGemma-2B
        │
      7 ┤   ★ LionGuard    ★ Qwen2.5-0.5B-Instruct (Zero-shot)
        │
      6 ┤   ★ WildGuard (7B)          ★ Aya-23 (Zero-shot)
        │
      5 ┤   ★ Llama Guard 3 (8B)
        │
      4 ┤                            ★ Granite Guardian
        │
      3 ┤   ★ BART-MNLI (零样本)
        │
      2 ┤   ★ XLM-R ★ mDeBERTa ★ ModernBERT
        │       (需要自己微调)
      1 ┤   ★ ByT5 ★ CANINE
        │
      0 ┴────┬────┬────┬────┬────┬────┬────┬────┬────┬───▶
             0.1  0.3  0.5  1    2    3    5    8    10+
                    参数量 (B)

        说明:
        - 越靠近左上角 = 越小越好用 (unitary/toxic)
        - 越靠近右上角 = 越大越好用 (API)
        - 下方区域 = 需要投入微调成本
        - 编码器类 (XLM-R等) 集中在左下: 小但需微调
        - 安全护栏类 (ShieldGemma等) 集中在中右: 中等大小,直接可用
```

---

## 八、速度 — 精度 权衡图

```
        精度 (违规内容召回率)
        │
    95% ┤                          ★ Llama Guard 3 (GPU)
        │                     ★ ShieldGemma-9B
    90% ┤              ★ Qwen2.5-3B
        │         ★ ShieldGemma-2B
    85% ┤    ★ Qwen2.5-0.5B
        │         ★ LionGuard
    80% ┤──────────────────── ★ unitary/toxic ONNX ──────────
        │                               (高并发场景的工业基线)
    75% ┤              ★ BART-MNLI (零样本)
        │
    70% ┤
        │    ★ DistilBERT (未微调)
    65% ┤
        │
    60% ┤
        │
    55% ┴────┬────┬────┬────┬────┬────┬────┬────┬────┬───▶
            0.5  1    5   10   20   50  100  200  500 1000
                     推理延迟 (ms/条, 对数刻度)

    编码器 ──→ 快但精度中等 (unitary/toxic)
    量化生成式 → 中等速度和精度 (ShieldGemma-2B GGUF)
    完整生成式 → 慢但精度高 (Llama Guard 3)

    最佳性价比区间:
    ┌──────────────────────────┐
    │  ★ unitary/toxic ONNX    │  <10ms,   ~80% 召回率
    │  ★ ShieldGemma-2B GGUF   │  ~200ms,  ~88% 召回率
    │  ★ Qwen2.5-0.5B GGUF     │  ~100ms,  ~80% 召回率
    └──────────────────────────┘
```

---

## 九、按模型家族的能力继承关系

```
                    [Transformer 预训练范式的演进]

  第一代 (2018-2019)          第二代 (2020-2021)        第三代 (2023-2024)
  ────────────────            ────────────────        ────────────────
  BERT                        XLM-RoBERTa             mDeBERTa-v3
  mBERT (多语言BERT)          XLM-R-large              ModernBERT
  GPT-1/2                     mT5                      SEA-LION
  T5                          ELECTRA                   ShieldGemma
  RoBERTa                     DistilBERT               Llama Guard
                              IndoBERT                 WildGuard
                              WangchanBERTa            BGE-M3
                              RemBERT                  Qwen2.5
                              PhoBERT                  Gemma-2
                              ByT5                     Aya-23
                              CANINE                   Mamba-2
                              LaBSE


    关键演进方向:
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  多语言覆盖: 英文 → 100语 → 500+语 (Glot500)             │
    │  效率提升:   平方注意力 → 解耦注意力 → Flash Attention   │
    │  安全性:     通用预训练 → 安全对齐RLHF → 专用安全护栏    │
    │  部署友好:   PyTorch → ONNX → GGUF → TensorRT           │
    │  可定制性:   全量微调 → LoRA → Adapter → Zero-shot       │
    │  对抗鲁棒:   词表模型 → 大词表(XLM-V) → 字节级(ByT5)     │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

---

## 十、总结：一图看全部模型关系

```
                           内容审核模型宇宙
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   [预训练底座]            [开箱即用]                [商业API]
   需要自己微调             下载直接用                 不可离线
        │                        │                        │
   ┌────┴────┐          ┌───────┴───────┐          ┌────┴────┐
   │         │          │               │          │         │
 通用底座  区域底座   编码器成品       生成式护栏  Google   OpenAI
   │         │          │               │       Perspective Moderation
 XLM-R   IndoBERT   unitary/toxic  ShieldGemma    API      API
 mDeBERTa Wangchan   NSFW-class    Llama Guard
 mBERT   BERTa      HateXplain     WildGuard                Azure
 RemBERT PhoBERT    LionGuard      Granite Guardian       Content
 XLM-V   SEA-LION                  Aegis-Guard            Safety
 Modern  AfriBERTa
 BERT                                      [零样本方案]
 ByT5    ALBETO                        ┌────────┴────────┐
 CANINE  Glot500                    BART-MNLI     DeBERTa-MNLI
 LaBSE
 BGE-M3                            [生成式通用 Zero-shot]
 mT5                               ┌────────┼────────┐
                                Qwen2.5  Gemma-2  Aya系列
                                0.5B/1.5B/3B 2B   23-8B

        [新架构 — 探索性]
        ┌────────┼────────┐
      Mamba-2  RWKV-6   xLSTM
```

---

## 使用建议

- **想最快上手**: 从 `unitary/multilingual-toxic-xlm-roberta` 开始，第14节代码，ONNX部署，下午就能跑
- **想最高精度**: `Llama Guard 3-8B` + `ShieldGemma-9B` 双层，但需要GPU
- **东南亚专精**: `mDeBERTa-v3-small` + 各国LoRA微调（第2.2节代码）
- **最省资源**: `Qwen2.5-0.5B-Instruct GGUF` (~350MB) 或 `mDeBERTa-v3-small ONNX INT8` (~140MB)
- **对抗文本**: `ByT5` 替代传统BERT，对变形词天然免疫
