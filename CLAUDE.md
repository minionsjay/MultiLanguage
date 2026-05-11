# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research knowledge base for **multilingual text content moderation** using small, offline-deployable models. The primary content is in `Model.md`, which is a structured Q&A covering model selection, deployment strategies, and research directions for detecting toxic/hate speech across multiple languages (Indonesian, Thai, Malay, Singlish, Brazilian Portuguese, and others).

## Repository Structure

- `Model.md` — The sole knowledge document. Contains deep-dive comparisons of encoder-only models (XLM-RoBERTa, mDeBERTa-v3, ModernBERT), small generative models (Qwen2.5, ShieldGemma, Gemma), ONNX/GGUF deployment, data synthesis strategies for low-resource languages, and cultural-aware safety alignment approaches.

## Core Architectural Patterns (from Model.md)

When building code for this domain, the recommended architecture is:

1. **Unified backbone + regional LoRA adapters** — Freeze a multilingual encoder (e.g., `mDeBERTa-v3-small` or `XLM-RoBERTa-base`), attach per-country LoRA modules for culture-specific classification, and route at inference time based on locale.
2. **ONNX export for offline deployment** — Export fine-tuned encoder models to ONNX format for framework-free inference on CPU-only or edge hardware.
3. **LLM-driven data distillation pipeline** — Use large multilingual models (GPT-4o, Claude) as data generators and judges to synthesize labeled training data for low-resource languages, then distill into sub-1B encoder models.
4. **Character-level defenses** — Augment subword tokenizers with character-level CNNs or use byte-level encoders (ByT5) to handle adversarial text (symbol substitution, code-switching, zero-width characters).

## Key Terminology

- **UGC** — User-Generated Content
- **AIGC** — AI-Generated Content
- **SLM** — Small Language Model (typically <3B parameters)
- **LoRA** — Low-Rank Adaptation (efficient fine-tuning method)
- **GGUF** — Quantized model format for llama.cpp inference
- **ONNX** — Open Neural Network Exchange format for framework-agnostic deployment
- **Focal Loss** — Loss function that down-weights easy examples, critical for imbalanced moderation datasets
- **MRL** — Matryoshka Representation Learning (flexible embedding dimensions)
- **Late Chunking** — Context-aware text chunking for retrieval
