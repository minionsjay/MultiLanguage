# Multilingual Content Moderation — Small Models & Research Knowledge Base

A research knowledge base for **multilingual text content moderation** using small, offline-deployable models. Covers model selection, deployment strategies, code examples, and cutting-edge research for detecting toxic/hate speech across multiple languages (Indonesian, Thai, Malay, Singlish, Brazilian Portuguese, and more).

## Repository Structure

```
.
├── Model.md                    # Core knowledge document — structured Q&A on model selection & deployment
├── model_comparison.md         # Full comparison tables: encoder, generative, and ready-to-use models
├── model_details.md            # Detailed model profiles (architecture, data, performance, HF links)
├── model_code.md               # Complete code: inference, fine-tuning, ONNX export, deployment
├── model_relationship.md       # Model evolution tree & cross-family relationship mapping
├── recent_papers.md            # Latest papers (2024–2026): safety guardrails, data synthesis, adversarial defense
├── sea_lion_ecosystem.md       # Deep dive into AI Singapore's SEA-LION ecosystem
├── charts.py                   # Visualization generator (model trees, radar charts, speed/accuracy plots)
├── charts/                     # 11 generated PNG visualizations
└── CLAUDE.md                   # Claude Code project instructions
```

## Topics Covered

### Model Selection
- **Encoder-only models** — XLM-RoBERTa, mDeBERTa-v3, ModernBERT, SEA-LION, IndoBERT, WangchanBERTa
- **Generative guard models** — ShieldGemma (2B), Qwen2.5 (0.5B–3B), Llama Guard 3, WildGuard
- **Ready-to-use classifiers** — `unitary/multilingual-toxic-xlm-roberta`, LionGuard/SEA-Guard

### Deployment & Efficiency
- ONNX export for framework-free offline inference
- GGUF quantization for llama.cpp deployment on CPU-only hardware
- Dynamic quantization (FP32 → INT8) with <1% accuracy loss
- LoRA adapters for culture-specific classification

### Data Engineering
- LLM-driven synthetic data distillation pipelines
- LLM-as-a-Judge for automated data labeling
- Handling imbalanced datasets with Focal Loss
- Adversarial data augmentation for code-switching & symbol substitution

### Cutting-Edge Research
- Culture-aware cross-lingual safety alignment
- Late Chunking & Matryoshka Representation Learning (MRL)
- Byte-level / character-level defenses against adversarial text
- Unified guardrail models (single model for toxicity + jailbreak + prompt injection)

## Quick Start

Browse the content:

```bash
# Start with the core Q&A document
cat Model.md

# Compare all models at a glance
cat model_comparison.md

# Dive into detailed model profiles
cat model_details.md

# Run code examples for your chosen model
cat model_code.md
```

Generate visualizations:

```bash
pip install matplotlib numpy
python charts.py
# Outputs 11 PNGs to charts/
```

## Key Recommendations at a Glance

| Scenario | Recommended Model | Deployment |
|----------|------------------|------------|
| Offline, CPU-only, 100+ languages | `unitary/multilingual-toxic-xlm-roberta` (ONNX) | ~280 MB, <10 ms/text |
| Offline, CPU-only, reasoning needed | `Qwen2.5-0.5B-Instruct` (GGUF, INT8) | ~1 GB, zero-shot Prompt |
| Southeast Asian languages (ID/TH/MY/SG) | SEA-LION-ModernBERT-300M + LoRA | ~600 MB, culture-aware |
| AIGC safety guardrail | `ShieldGemma-2B` (GGUF, INT8) | <3 GB, blocks prompt injection |
| High-volume, GPU batching | `mDeBERTa-v3-small` → ONNX → TensorRT | Massively parallel |

## References

This knowledge base synthesizes research from:
- **SEA-Guard** (AI Singapore, 2026) — Culturally grounded multilingual safeguard for Southeast Asia
- **WildGuard** (AllenAI, NeurIPS 2024) — Unified moderation for safety risks, jailbreaks, and refusals
- **ShieldGemma** (Google, 2024) — 2B generative content safety classifier
- **Llama Guard 1/2/3** (Meta, 2023–2024) — Taxonomy-based safety classification
- **NusaX** — Multilingual parallel corpus for Indonesian languages
- **HateBR** — Brazilian Portuguese abusive language detection

See `recent_papers.md` for the full annotated bibliography.

## License

This is a research resource. Refer to each model's license (Apache 2.0, MIT, Gemma license, Llama Community License, etc.) for usage terms.
