"""
多语言内容审核模型 — 可视化图表生成
运行: pip install matplotlib numpy adjustText && python charts.py
输出: charts/ 目录下的多个PNG图片
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np
import os

os.makedirs("charts", exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Droid Sans Fallback"] + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# %% 图表1: 模型家族演化树 ——————————————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(28, 18))
ax.set_xlim(0, 28)
ax.set_ylim(0, 18)
ax.axis("off")
ax.set_title("Content Moderation Model Family Tree", fontsize=22, fontweight="bold", pad=20)

def draw_box(ax, x, y, w, h, text, color, fontsize=8, text_color="black"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor="#333", linewidth=1.2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, color="#666"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, connectionstyle="arc3,rad=0"))

# Root
draw_box(ax, 14, 17, 3.5, 1.0, "Transformer\n(2017)", "#2c3e50", 9, "white")

# Branch 1: Encoder-only
draw_box(ax, 5, 15, 3.0, 0.9, "Encoder-only\n纯分类", "#3498db", 8, "white")
draw_arrow(ax, 13, 16.5, 6, 15.4)

draw_box(ax, 2, 13, 2.2, 0.7, "BERT\n(2018)", "#5dade2")
draw_box(ax, 5, 13, 2.2, 0.7, "RoBERTa\n(2019)", "#5dade2")
draw_box(ax, 8, 13, 2.2, 0.7, "DeBERTa\n(2020)", "#5dade2")

draw_arrow(ax, 4.2, 14.5, 1.5, 13.4)
draw_arrow(ax, 4.8, 14.5, 5, 13.4)
draw_arrow(ax, 5.5, 14.5, 8, 13.4)

# BERT children
for i, (name, xx) in enumerate([("mBERT (104语)", 0.5), ("DistilBERT", 2), ("IndoBERT (印尼)", 3.5)]):
    draw_box(ax, xx, 11, 1.8, 0.6, name, "#aed6f1", 6)
    draw_arrow(ax, 1.5, 12.6, xx, 11.4)

# RoBERTa children
for i, (name, xx) in enumerate([("XLM-R (100语)", 3.5), ("XLM-V (901K词表)", 5.5), ("BGE-M3 (检索)", 7.5)]):
    draw_box(ax, xx, 11, 1.8, 0.6, name, "#aed6f1", 6)
    draw_arrow(ax, 5, 12.6, xx, 11.4)

# DeBERTa children
draw_box(ax, 8, 11, 2.2, 0.6, "mDeBERTa-v3\n(100+语, 最优)", "#aed6f1", 7)

# ModernBERT
draw_box(ax, 10.5, 13, 2.0, 0.7, "ModernBERT\n(2024)", "#5dade2")
draw_arrow(ax, 5.8, 14.7, 10.5, 13.4)

# Branch 2: Decoder-only
draw_box(ax, 18, 15, 3.0, 0.9, "Decoder-only\n纯生成", "#e74c3c", 8, "white")
draw_arrow(ax, 15, 16.5, 17, 15.4)

draw_box(ax, 16, 13, 2.0, 0.7, "GPT系", "#f1948a")
draw_box(ax, 19, 13, 2.0, 0.7, "LLaMA系", "#f1948a")
draw_box(ax, 22, 13, 2.0, 0.7, "Gemma系", "#f1948a")

draw_arrow(ax, 18, 14.5, 16, 13.4)
draw_arrow(ax, 18.5, 14.5, 19, 13.4)
draw_arrow(ax, 19, 14.5, 22, 13.4)

# GPT children
for i, (name, xx) in enumerate([("Qwen2.5\n(亚洲语强)", 14.5), ("SEA-LION\n(东南亚)", 17.5)]):
    draw_box(ax, xx, 11, 1.8, 0.7, name, "#f5b7b1", 6)
    draw_arrow(ax, 16, 12.6, xx, 11.4)

# LLaMA children
for i, (name, xx) in enumerate([("Llama Guard 3\n(安全护栏)", 18), ("WildGuard\n(三合一)", 20.5)]):
    draw_box(ax, xx, 11, 1.8, 0.7, name, "#f5b7b1", 6)
    draw_arrow(ax, 19, 12.6, xx, 11.4)

# Gemma children
draw_box(ax, 21, 11, 2.0, 0.7, "Gemma-2-2B", "#f5b7b1", 6)
draw_box(ax, 23.5, 11, 2.0, 0.7, "ShieldGemma\n(2B/9B 安全)", "#f5b7b1", 6)
draw_arrow(ax, 22, 12.6, 21, 11.4)
draw_arrow(ax, 22.5, 12.6, 23.5, 11.4)

# Branch 3: Encoder-Decoder
draw_box(ax, 25, 15, 2.5, 0.9, "Encoder-\nDecoder", "#27ae60", 8, "white")
draw_arrow(ax, 14.5, 16.5, 25, 15.4)

draw_box(ax, 24, 13, 2.0, 0.7, "T5 (2019)", "#82e0aa")
draw_box(ax, 26.5, 13, 2.0, 0.7, "BART (2019)", "#82e0aa")

draw_box(ax, 23, 11, 2.0, 0.7, "mT5 (101语)", "#abebc6", 6)
draw_box(ax, 25.5, 11, 2.0, 0.7, "ByT5\n(字节级, 无词表)", "#abebc6", 6)
draw_arrow(ax, 24, 12.6, 23, 11.4)
draw_arrow(ax, 24.5, 12.6, 25.5, 11.4)

# New arch
draw_box(ax, 14, 8.5, 3.5, 0.8, "新架构 (替代Transformer)", "#8e44ad", 8, "white")
draw_arrow(ax, 14, 16.2, 14, 8.9, "#999")

for i, (name, xx) in enumerate([("Mamba-2 (SSM)", 11), ("RWKV-6", 14), ("xLSTM", 17)]):
    draw_box(ax, xx, 7, 2.2, 0.6, name, "#d2b4de", 7)
    draw_arrow(ax, 14, 8.2, xx, 7.4)

plt.tight_layout()
fig.savefig("charts/01_model_family_tree.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 01_model_family_tree.png")

# %% 图表2: 模型能力雷达图 ——————————————————————————————————————
categories = ["多语言覆盖", "推理速度", "东南亚适配", "开箱可用", "资源消耗低"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

models_radar = {
    "XLM-RoBERTa-base": [9, 8, 6, 4, 6],
    "mDeBERTa-v3-small": [9, 10, 6, 4, 9],
    "SEA-LION-7B": [5, 4, 10, 5, 2],
    "unitary/toxic-xlm": [9, 10, 6, 10, 9],
    "ShieldGemma-2B": [7, 5, 6, 9, 6],
    "Qwen2.5-0.5B": [8, 5, 8, 8, 9],
    "Llama Guard 3-8B": [8, 3, 5, 9, 3],
}
colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

fig, axes = plt.subplots(2, 4, figsize=(20, 11), subplot_kw=dict(polar=True))
axes = axes.flatten()

for idx, (name, values) in enumerate(models_radar.items()):
    ax = axes[idx]
    values_plot = values + values[:1]
    ax.fill(angles, values_plot, alpha=0.25, color=colors[idx])
    ax.plot(angles, values_plot, color=colors[idx], linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=6)
    ax.set_title(name, fontsize=9, fontweight="bold", pad=12)

# 空白子图不需要
axes[-1].set_visible(False)

fig.suptitle("Model Capability Radar Comparison", fontsize=18, fontweight="bold", y=0.98)
plt.tight_layout()
fig.savefig("charts/02_radar_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 02_radar_comparison.png")

# %% 图表3: 叠加雷达图 ——————————————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))
key_models = {
    "unitary/toxic-xlm\n(开箱即用)": ([9, 10, 6, 10, 9], "#f39c12"),
    "mDeBERTa-v3-small\n(最佳底座)": ([9, 10, 6, 4, 9], "#2ecc71"),
    "ShieldGemma-2B\n(安全护栏)": ([7, 5, 6, 9, 6], "#9b59b6"),
    "Llama Guard 3-8B\n(最高精度)": ([8, 3, 5, 9, 3], "#e67e22"),
}

for name, (values, color) in key_models.items():
    vals = values + values[:1]
    ax.fill(angles, vals, alpha=0.1, color=color)
    ax.plot(angles, vals, color=color, linewidth=2.5, label=name)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8)
ax.set_title("Model Comparison: Top 4 Contenders", fontsize=16, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
fig.savefig("charts/02b_overlay_radar.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 02b_overlay_radar.png")

# %% 图表4: 模型散点图 (速度 vs 精度) ——————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(16, 10))

models_scatter = [
    # (name, speed_ms, recall, mem_mb, category)
    ("unitary/toxic-xlm\n(ONNX)", 5, 0.80, 300, "编码器成品"),
    ("mDeBERTa-v3-small\n(ONNX INT8)", 5, 0.78, 140, "编码器底座(微调后)"),
    ("XLM-R-base", 12, 0.79, 1100, "编码器底座(微调后)"),
    ("DistilBERT-multi", 4, 0.72, 540, "编码器底座(微调后)"),
    ("ShieldGemma-2B\n(GGUF INT4)", 200, 0.87, 1500, "生成式量化"),
    ("ShieldGemma-9B\n(vLLM)", 150, 0.92, 9000, "生成式量化"),
    ("Qwen2.5-0.5B\n(GGUF INT4)", 80, 0.79, 350, "生成式量化"),
    ("Qwen2.5-3B\n(GGUF INT4)", 300, 0.86, 2000, "生成式量化"),
    ("Llama Guard 3-8B\n(vLLM)", 350, 0.93, 8000, "生成式安全护栏"),
    ("Llama Guard 3-8B\n(GGUF INT4 CPU)", 800, 0.91, 5000, "生成式安全护栏"),
    ("Gemma-2-2B", 250, 0.83, 4000, "生成式通用"),
    ("WildGuard-7B", 400, 0.85, 7000, "生成式安全护栏"),
    ("Aya-23-8B", 500, 0.84, 8000, "生成式通用"),
    ("BART-MNLI\n(零样本)", 50, 0.65, 1600, "零样本方案"),
]

cat_colors = {
    "编码器成品": "#f39c12",
    "编码器底座(微调后)": "#2ecc71",
    "生成式量化": "#3498db",
    "生成式安全护栏": "#e74c3c",
    "生成式通用": "#9b59b6",
    "零样本方案": "#95a5a6",
}

for name, speed, recall, mem, cat in models_scatter:
    color = cat_colors[cat]
    size = np.sqrt(mem) * 2  # 气泡大小代表内存
    ax.scatter(speed, recall * 100, s=size, c=color, edgecolors="#333",
               linewidth=1, alpha=0.85, zorder=5)
    offset = (15, 5) if "unitary" in name else (10, -8) if "Qwen" in name else (10, 5)
    ax.annotate(name, (speed, recall * 100), textcoords="offset points",
                xytext=offset, fontsize=7.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.8))

ax.set_xscale("log")
ax.set_xlabel("Inference Latency (ms/text, log scale)", fontsize=12, fontweight="bold")
ax.set_ylabel("Harmful Content Recall (%)", fontsize=12, fontweight="bold")
ax.set_title("Speed vs Accuracy: Content Moderation Models\n(bubble size = memory footprint)",
             fontsize=16, fontweight="bold")

# 最佳性价比区域
from matplotlib.patches import Rectangle
rect = Rectangle((2, 73), 80, 15, linewidth=2, edgecolor="#f39c12",
                  facecolor="#f39c12", alpha=0.08, linestyle="--")
ax.add_patch(rect)
ax.annotate("Best Value Zone\n<50ms, >75% recall", (25, 85), fontsize=10,
            fontweight="bold", color="#e67e22", ha="center")

ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(1.5, 1500)
ax.set_ylim(55, 98)

legend_elements = [mpatches.Patch(facecolor=c, edgecolor="#333", label=cat)
                   for cat, c in cat_colors.items()]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, title="Model Category")

plt.tight_layout()
fig.savefig("charts/03_speed_vs_accuracy.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 03_speed_vs_accuracy.png")

# %% 图表5: 堆叠柱状图 — 按架构类型 ——————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

categories_bar = ["Encoder-only\n(分类专用)", "Encoder-only\n(开箱即用)", "Decoder-only\n(生成式通用)",
                  "Decoder-only\n(安全护栏)", "商业API", "新架构\n(探索)"]
counts = [12, 4, 5, 6, 3, 4]
colors_bar = ["#2ecc71", "#27ae60", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

bars = ax.bar(categories_bar, counts, color=colors_bar, edgecolor="#333", linewidth=1.2, width=0.6)

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(count), ha="center", fontsize=14, fontweight="bold")

ax.set_ylabel("Number of Models", fontsize=12, fontweight="bold")
ax.set_title("Model Distribution by Architecture Type", fontsize=16, fontweight="bold")
ax.set_ylim(0, max(counts) + 2)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# 添加具体模型名注释
annotations = {
    "Encoder-only\n(分类专用)": "XLM-R, mDeBERTa, mBERT,\nModernBERT, BGE-M3...",
    "Encoder-only\n(开箱即用)": "unitary/toxic, LionGuard,\nHateXplain, NSFW-classifier",
    "Decoder-only\n(生成式通用)": "Qwen2.5, Gemma-2, Aya,\nSEA-LION (原版)",
    "Decoder-only\n(安全护栏)": "ShieldGemma, Llama Guard,\nWildGuard, Granite Guardian",
    "商业API": "Perspective, OpenAI Mod,\nAzure Safety",
    "新架构\n(探索)": "Mamba-2, RWKV-6, ByT5,\nCANINE",
}
for i, (cat, txt) in enumerate(annotations.items()):
    ax.text(i, counts[i] + 1.2, txt, ha="center", fontsize=7.5, color="#555", va="bottom")

plt.tight_layout()
fig.savefig("charts/04_architecture_distribution.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 04_architecture_distribution.png")

# %% 图表6: 决策流程图 ——————————————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis("off")
ax.set_title("Content Moderation Model Selection Decision Tree", fontsize=20, fontweight="bold", pad=15)

def dbox(x, y, w, h, text, color, fs=8, tc="black", bold=True):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.15",
                        facecolor=color, edgecolor="#333", linewidth=1.5)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc)

def darrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2 + 0.3), xytext=(x1, y1 - 0.3),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.8, connectionstyle="arc3,rad=0"))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.4, my, label, fontsize=8, fontweight="bold", color="#c0392b", va="center")

# Root
dbox(11, 15, 5, 1.0, "Starting: Choose Moderation Model", "#2c3e50", 11, "white")

# Level 1
dbox(3, 13, 3.5, 0.9, "Need Offline?", "#34495e", 9, "white")
dbox(11, 13, 3.5, 0.9, "Need Reason\nOutput?", "#34495e", 9, "white")
dbox(19, 13, 3.5, 0.9, "Need Latency\n<10ms?", "#34495e", 9, "white")

darrow(10, 14.5, 3, 13.4)
darrow(11, 14.5, 11, 13.4)
darrow(12, 14.5, 19, 13.4)

# Level 2 - Offline branch
dbox(1.5, 11, 2.5, 0.8, "Yes:\nLocal Deploy", "#e74c3c", 7, "white")
dbox(5, 11, 2.5, 0.8, "No:\nAPI", "#27ae60", 7, "white")
darrow(3, 12.5, 1.5, 11.4)
darrow(3, 12.5, 5, 11.4)

# Offline -> sub options
dbox(0.5, 9, 2.2, 0.8, "How much RAM?", "#c0392b", 7, "white")
darrow(1.5, 10.5, 0.5, 9.4)

dbox(0.2, 6.5, 1.8, 0.7, "<2GB\nCPU Only", "#f1948a", 6, "white")
dbox(2.2, 6.5, 1.8, 0.7, "<4GB\nCPU Only", "#f1948a", 6, "white")
darrow(0.5, 8.5, 0.2, 6.9)
darrow(0.5, 8.5, 2.2, 6.9)

dbox(0.2, 4.5, 2.0, 0.8, "Qwen2.5-0.5B\nGGUF INT4\n(~350MB)", "#f9e79f", 6)
dbox(2.2, 4.5, 2.0, 0.8, "ShieldGemma-2B\nGGUF INT4\n(~1.5GB)", "#f9e79f", 6)
darrow(0.2, 6.1, 0.2, 5.0)
darrow(2.2, 6.1, 2.2, 5.0)

# API branch
dbox(5, 9, 2.5, 0.8, "Perspective API\n(Free)", "#82e0aa", 7)
dbox(7.5, 9, 2.5, 0.8, "OpenAI Mod API\n(Free)", "#82e0aa", 7)
darrow(5, 10.5, 5, 9.4)
darrow(5.5, 10.5, 7.5, 9.4)

# Reason branch
dbox(9, 11, 2.5, 0.8, "Yes:\nNeed Reasons", "#e74c3c", 7, "white")
dbox(13, 11, 2.5, 0.8, "No:\nScore Only", "#27ae60", 7, "white")
darrow(11, 12.5, 9, 11.4)
darrow(11, 12.5, 13, 11.4)

dbox(9, 9, 2.5, 0.8, "ShieldGemma\nQwen2.5\nLlama Guard", "#f5b7b1", 7)
dbox(13, 9, 2.5, 0.8, "unitary/toxic\nONNX (Fast)", "#abebc6", 7)
darrow(9, 10.5, 9, 9.4)
darrow(13, 10.5, 13, 9.4)

# Speed branch
dbox(17, 11, 2.5, 0.8, "Yes:\n<10ms/req", "#e74c3c", 7, "white")
dbox(21, 11, 2.5, 0.8, "No:\nTolerable", "#27ae60", 7, "white")
darrow(19, 12.5, 17, 11.4)
darrow(19, 12.5, 21, 11.4)

dbox(17, 9, 2.5, 0.8, "Encoder ONNX\nmDeBERTa\nunitary/toxic", "#abebc6", 7)
dbox(21, 9, 2.5, 0.8, "GGUF Quants\nShieldGemma\nQwen", "#82e0aa", 7)
darrow(17, 10.5, 17, 9.4)
darrow(21, 10.5, 21, 9.4)

# Language branch (independent)
dbox(11, 7.5, 4, 0.9, "Language Coverage?", "#34495e", 9, "white")

dbox(5, 5.5, 2.2, 0.7, "100+ langs", "#5dade2", 7)
dbox(9, 5.5, 2.2, 0.7, "Southeast Asia", "#5dade2", 7)
dbox(13, 5.5, 2.2, 0.7, "Single Lang", "#5dade2", 7)
dbox(17, 5.5, 2.2, 0.7, "Adversarial", "#5dade2", 7)

darrow(10, 7.0, 5, 5.9)
darrow(10.5, 7.0, 9, 5.9)
darrow(11.5, 7.0, 13, 5.9)
darrow(12, 7.0, 17, 5.9)

dbox(5, 3.5, 2.2, 0.7, "XLM-R/mDeBERTa\nunitary/toxic", "#aed6f1", 6)
dbox(9, 3.5, 2.2, 0.7, "SEA-LION+LoRA\nLionGuard", "#aed6f1", 6)
dbox(13, 3.5, 2.2, 0.7, "IndoBERT\nWangchanBERTa", "#aed6f1", 6)
dbox(17, 3.5, 2.2, 0.7, "ByT5/CANINE\n+文本归一化", "#aed6f1", 6)

darrow(5, 5.1, 5, 3.9)
darrow(9, 5.1, 9, 3.9)
darrow(13, 5.1, 13, 3.9)
darrow(17, 5.1, 17, 3.9)

plt.tight_layout()
fig.savefig("charts/05_decision_tree.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 05_decision_tree.png")

# %% 图表7: 横向对比柱状图 — 内存占用 ——————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

models_mem = [
    ("mDeBERTa-v3-small ONNX INT8", 140, "#2ecc71"),
    ("DistilBERT-multilingual ONNX", 135, "#27ae60"),
    ("IndoBERT/WangchanBERTa FP32", 440, "#27ae60"),
    ("Qwen2.5-0.5B GGUF INT4", 350, "#3498db"),
    ("unitary/toxic-xlm ONNX", 300, "#f39c12"),
    ("ModernBERT-base FP32", 560, "#2ecc71"),
    ("XLM-RoBERTa-base FP32", 1100, "#2ecc71"),
    ("XLM-RoBERTa-large FP32", 2200, "#2ecc71"),
    ("ShieldGemma-2B GGUF INT4", 1500, "#e74c3c"),
    ("LaBSE FP32", 1900, "#e67e22"),
    ("BGE-M3 FP32", 2300, "#e67e22"),
    ("Qwen2.5-3B GGUF INT4", 2000, "#3498db"),
    ("ByT5-small FP32", 1200, "#9b59b6"),
    ("Gemma-2-2B FP16", 4000, "#3498db"),
    ("Llama Guard 3-8B GGUF INT4", 5000, "#c0392b"),
    ("ShieldGemma-9B FP16", 18000, "#c0392b"),
    ("Llama Guard 3-8B FP16", 16000, "#c0392b"),
]

models_mem.sort(key=lambda x: x[1])

names = [m[0] for m in models_mem]
mem_vals = [m[1] for m in models_mem]
colors_mem = [m[2] for m in models_mem]

bars = ax.barh(names, mem_vals, color=colors_mem, edgecolor="#333", linewidth=0.8)

for bar, val in zip(bars, mem_vals):
    label = f"{val}MB" if val < 1000 else f"{val/1000:.1f}GB"
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8, fontweight="bold")

ax.set_xlabel("Memory (MB)", fontsize=12, fontweight="bold")
ax.set_title("Model Memory Footprint Comparison\n(FP32 / INT8 / INT4 quantized)", fontsize=16, fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(50, 30000)
ax.grid(axis="x", alpha=0.3, linestyle="--")

legend_elements = [
    mpatches.Patch(facecolor="#2ecc71", label="Encoder Backbone"),
    mpatches.Patch(facecolor="#f39c12", label="Encoder Ready-to-use"),
    mpatches.Patch(facecolor="#3498db", label="Generative (Quantized)"),
    mpatches.Patch(facecolor="#e74c3c", label="Guardrail Model"),
    mpatches.Patch(facecolor="#e67e22", label="Embedding Model"),
    mpatches.Patch(facecolor="#9b59b6", label="New Architecture"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

plt.tight_layout()
fig.savefig("charts/06_memory_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 06_memory_comparison.png")

# %% 图表8: 场景推荐矩阵热力图 ——————————————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(16, 9))

scenarios = [
    "UGC High\nConcurrency",
    "AIGC Safety\nGuardrail",
    "SE Asia\nLocalized",
    "Edge Device\nOffline",
    "Adversarial\nText",
    "Zero-shot\nQuick Start",
    "Explainable\nAudit",
    "Long Text\nModeration",
]

models_hm = [
    "unit./toxic",
    "mDeBERTa-s",
    "XLM-R-base",
    "ShieldGemma-2B",
    "LlamaGuard-3",
    "Qwen0.5B",
    "Qwen3B",
    "ByT5",
    "CANINE",
    "LionGuard",
    "WildGuard",
    "GraniteGuard",
]

# 评分矩阵 (0-10)
matrix = np.array([
    # uni mDe XLM Shd LlG Q05 Q3B ByT CAN Lio Wil Grn
    [10,  9,  8,  3,  2,  3,  2,  2,  2,  5,  2,  3],  # UGC高并发
    [ 4,  3,  3,  9, 10,  6,  7,  2,  2,  5,  9,  8],  # AIGC护栏
    [ 5,  7,  5,  6,  4,  7,  7,  4,  4, 10,  4,  5],  # 东南亚本地化
    [ 8,  9,  7,  8,  1, 10,  6,  5,  5,  6,  2,  5],  # 边缘离线
    [ 2,  3,  3,  4,  4,  4,  5, 10,  9,  5,  5,  5],  # 对抗文本
    [10,  4,  3,  8,  7,  9,  8,  3,  3,  5,  7,  7],  # 零样本快速
    [ 3,  2,  2,  5,  8,  6,  8,  3,  3,  4,  6,  7],  # 可解释审计
    [ 4,  7,  5,  5,  5,  5,  6,  8,  7,  5,  5,  5],  # 长文本
])

im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=10)

ax.set_xticks(range(len(models_hm)))
ax.set_xticklabels(models_hm, fontsize=8, rotation=45, ha="right")
ax.set_yticks(range(len(scenarios)))
ax.set_yticklabels(scenarios, fontsize=9, fontweight="bold")

for i in range(len(scenarios)):
    for j in range(len(models_hm)):
        val = matrix[i, j]
        color = "white" if val >= 7 else "black"
        ax.text(j, i, str(val), ha="center", va="center", fontsize=8, fontweight="bold", color=color)

ax.set_title("Scenario × Model Recommendation Heatmap\n(10 = best fit, ⬛ = top recommendation)",
             fontsize=15, fontweight="bold", pad=15)
fig.colorbar(im, ax=ax, shrink=0.8, label="Fit Score")

plt.tight_layout()
fig.savefig("charts/07_scenario_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 07_scenario_heatmap.png")

# %% 图表9: 模型时间线 ——————————————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(18, 8))

timeline = [
    (2018, "BERT", "Google", "Encoder"),
    (2018, "mBERT", "Google", "Encoder"),
    (2019, "XLM-RoBERTa", "Meta", "Encoder"),
    (2019, "RoBERTa", "Meta", "Encoder"),
    (2019, "T5/mT5", "Google", "Enc-Dec"),
    (2020, "DeBERTa", "Microsoft", "Encoder"),
    (2020, "IndoBERT", "IndoNLU", "Encoder"),
    (2020, "WangchanBERTa", "VISTEC", "Encoder"),
    (2020, "RemBERT", "Google", "Encoder"),
    (2020, "LaBSE", "Google", "Embedding"),
    (2020, "DistilBERT-multi", "HF", "Encoder"),
    (2021, "ByT5", "Google", "Enc-Dec"),
    (2021, "CANINE", "Google", "Encoder"),
    (2023, "mDeBERTa-v3", "Microsoft", "Encoder"),
    (2023, "SEA-LION", "AI Singapore", "Decoder"),
    (2023, "XLM-V", "Meta", "Encoder"),
    (2023, "Llama Guard 1", "Meta", "Guardrail"),
    (2024, "ShieldGemma", "Google", "Guardrail"),
    (2024, "Qwen2.5", "Alibaba", "Decoder"),
    (2024, "Gemma-2", "Google", "Decoder"),
    (2024, "ModernBERT", "Answer.AI", "Encoder"),
    (2024, "BGE-M3", "BAAI", "Embedding"),
    (2024, "WildGuard", "AllenAI", "Guardrail"),
    (2024, "Aya-23", "Cohere", "Decoder"),
    (2024, "Llama Guard 3", "Meta", "Guardrail"),
    (2024, "Granite Guardian", "IBM", "Guardrail"),
    (2025, "Mamba-2", "Academia", "SSM"),
    (2025, "Aya Expanse", "Cohere", "Decoder"),
]

cat_color = {"Encoder": "#2ecc71", "Encoder": "#2ecc71", "Enc-Dec": "#3498db",
              "Decoder": "#e74c3c", "Embedding": "#f39c12", "Guardrail": "#c0392b",
              "SSM": "#9b59b6"}

for year, name, org, cat in timeline:
    color = cat_color[cat]
    y_offset = {
        "Encoder": 0,
        "Enc-Dec": 1.8,
        "Decoder": -1.8,
        "Guardrail": -3.6,
        "Embedding": 3.6,
        "SSM": 5.2,
    }[cat]
    ax.scatter(year, y_offset, s=120, c=color, edgecolors="#333", linewidth=1, zorder=5)
    ax.annotate(f"{name}\n({org})", (year, y_offset), textcoords="offset points",
                xytext=(0, 12 if y_offset >= 0 else -16), fontsize=6.5, ha="center",
                fontweight="bold", alpha=0.85)

ax.set_yticks([])
ax.set_xlim(2017.5, 2025.5)
ax.set_xlabel("Year", fontsize=12, fontweight="bold")
ax.set_title("Multilingual Content Moderation Model Timeline", fontsize=16, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--")

legend_elements = [mpatches.Patch(facecolor=c, label=cat)
                   for cat, c in [("Encoder (分类)", "#2ecc71"),
                                  ("Encoder-Decoder", "#3498db"),
                                  ("Decoder (生成)", "#e74c3c"),
                                  ("Safety Guardrail", "#c0392b"),
                                  ("Embedding", "#f39c12"),
                                  ("SSM (New)", "#9b59b6")]]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8, title="Architecture")

plt.tight_layout()
fig.savefig("charts/08_timeline.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 08_timeline.png")

# %% 图表10: 最终推荐流程图 ——————————————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Quick Model Selection Roadmap", fontsize=18, fontweight="bold", pad=10)

# Top: question
dbox(8, 9.2, 6, 0.8, "What is your PRIMARY constraint?", "#2c3e50", 10, "white")

# Four paths
path_colors = ["#e74c3c", "#f39c12", "#3498db", "#27ae60"]
paths = [
    (2, 7.5, "SPEED\n(must be <10ms)", "#e74c3c"),
    (6, 7.5, "LOCALIZATION\n(SE Asia accuracy)", "#f39c12"),
    (10, 7.5, "ACCURACY\n(best recall)", "#3498db"),
    (14, 7.5, "RESOURCES\n(ultra-low memory)", "#27ae60"),
]

for x, y, label, color in paths:
    dbox(x, y, 3.2, 0.9, label, color, 8, "white")
    darrow(8, 8.8 + 0.3, x, y + 0.45)

# Answers per path
# Speed
dbox(2, 5.5, 3.2, 0.8, "1. unitary/toxic-xlm\n   ONNX (<10ms)", "#f1948a", 7)
dbox(2, 4.2, 3.2, 0.8, "2. mDeBERTa-v3-small\n   ONNX INT8 (~5ms)", "#f1948a", 7)
dbox(2, 2.9, 3.2, 0.8, "3. DistilBERT-multi\n   ONNX (~3ms)", "#f1948a", 7)
darrow(2, 7.0, 2, 5.9)

# Localization
dbox(6, 5.5, 3.2, 0.8, "1. XLM-R + Local LoRA\n   (Per-country, ~6MB)", "#fad7a0", 7)
dbox(6, 4.2, 3.2, 0.8, "2. SEA-LION + LionGuard\n   (SE Asia native)", "#fad7a0", 7)
dbox(6, 2.9, 3.2, 0.8, "3. Qwen2.5 + Custom\n   System Prompt", "#fad7a0", 7)
darrow(6, 7.0, 6, 5.9)

# Accuracy
dbox(10, 5.5, 3.2, 0.8, "1. Llama Guard 3-8B\n   (GPU, 93% recall)", "#85c1e9", 7)
dbox(10, 4.2, 3.2, 0.8, "2. ShieldGemma-9B\n   (GPU, 92% recall)", "#85c1e9", 7)
dbox(10, 2.9, 3.2, 0.8, "3. Double Guard:\n   ShieldGemma+LlamaGuard", "#85c1e9", 7)
darrow(10, 7.0, 10, 5.9)

# Resources
dbox(14, 5.5, 3.2, 0.8, "1. Qwen2.5-0.5B GGUF\n   INT4 (~350MB)", "#a9dfbf", 7)
dbox(14, 4.2, 3.2, 0.8, "2. mDeBERTa-v3-small\n   ONNX INT8 (~140MB)", "#a9dfbf", 7)
dbox(14, 2.9, 3.2, 0.8, "3. DistilBERT-multi\n   ONNX (~135MB)", "#a9dfbf", 7)
darrow(14, 7.0, 14, 5.9)

# Bottom: universal tip
dbox(8, 0.8, 10, 0.7, "Universal fallback: Perspective API (free, multilingual, for pseudo-labeling)",
     "#2c3e50", 8, "white")

plt.tight_layout()
fig.savefig("charts/09_quick_selection.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 09_quick_selection.png")

# %% 图表11: 模型分类桑基图风格 ——————————————————————————————————————
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis("off")
ax.set_title("Model Architecture → Use Case → Final Deployment Flow", fontsize=18, fontweight="bold", pad=15)

# Column 1: Architecture
col1 = [
    (2, 10.5, "Encoder-only", "#2ecc71"),
    (2, 8.5, "Decoder-only", "#e74c3c"),
    (2, 6.5, "Enc-Dec / SSM", "#3498db"),
    (2, 4.5, "API / Other", "#f39c12"),
]
for x, y, txt, c in col1:
    dbox(x, y, 3.5, 0.8, txt, c, 9, "white")

# Column 2: Models
col2 = [
    (7, 11.2, "XLM-R / mDeBERTa / mBERT", "#aed6f1", 7),
    (7, 10.2, "ModernBERT / DistilBERT", "#aed6f1", 7),
    (7, 9.2, "IndoBERT / WangchanBERTa / PhoBERT", "#aed6f1", 6),
    (7, 7.8, "Qwen2.5 / Gemma-2 / Aya", "#f5b7b1", 7),
    (7, 6.8, "SEA-LION / ShieldGemma / LlamaGuard", "#f5b7b1", 6),
    (7, 5.2, "mT5 / ByT5 / CANINE", "#85c1e9", 7),
    (7, 4.2, "Mamba-2 / RWKV-6", "#d2b4de", 7),
    (7, 2.8, "Perspective / OpenAI Mod API", "#fad7a0", 7),
]
for x, y, txt, c, fs in col2:
    dbox(x, y, 3.8, 0.7, txt, c, fs)

# Column 3: Use case
col3 = [
    (13, 11.2, "Multi-lang UGC Filtering", "#abebc6", 7),
    (13, 10, "Single-lang Precision", "#abebc6", 7),
    (13, 8, "AIGC Safety Guardrail", "#f1948a", 7),
    (13, 6.8, "Flexible Zero-shot", "#f1948a", 7),
    (13, 5.2, "Adversarial Text Defense", "#85c1e9", 7),
    (13, 4.2, "Long Text / Research", "#d2b4de", 7),
    (13, 2.8, "Baseline / Pseudo-labeling", "#fad7a0", 7),
]
for x, y, txt, c, fs in col3:
    dbox(x, y, 3.8, 0.7, txt, c, fs)

# Column 4: Deployment
col4 = [
    (18, 11.2, "ONNX Runtime (C++/Rust)", "#2c3e50", 6, "white"),
    (18, 9.6, "ONNX + FastAPI", "#2c3e50", 6, "white"),
    (18, 7.6, "GGUF + llama.cpp", "#2c3e50", 6, "white"),
    (18, 6.4, "vLLM GPU Batching", "#2c3e50", 6, "white"),
    (18, 5.0, "Docker + K8s", "#2c3e50", 6, "white"),
    (18, 3.6, "Edge / IoT Device", "#2c3e50", 6, "white"),
    (18, 2.4, "HTTP API Call", "#2c3e50", 6, "white"),
]
for x, y, txt, c, fs, tc in col4:
    dbox(x, y, 2.5, 0.6, txt, c, fs, tc)

# Draw connections (simplified)
for y1, y2 in [(10.9, 11.5), (10.5, 10.5), (8.9, 9.5), (7, 8.2), (6.5, 7.2), (5, 5.5), (4.5, 4.5), (3, 3.1)]:
    darrow(3.9, y1, 4.9, y2)
for y1, y2 in [(11.5, 11.5), (10.5, 10.3), (8.2, 8.3), (7.2, 7.1), (5.5, 5.5), (4.5, 4.5), (3.1, 3.1)]:
    darrow(11, y1, 11, y2)
for y1, y2 in [(11.5, 11.5), (9.9, 9.9), (7.9, 7.9), (6.7, 6.7), (5.2, 5.3), (4.2, 3.9), (3, 2.7)]:
    darrow(16.9, y1, 16.9, y2)

plt.tight_layout()
fig.savefig("charts/10_deployment_flow.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] 10_deployment_flow.png")

print(f"\n{'='*60}")
print("All 11 charts generated successfully in charts/ directory:")
for f in sorted(os.listdir("charts")):
    print(f"  charts/{f}")
print(f"{'='*60}")
