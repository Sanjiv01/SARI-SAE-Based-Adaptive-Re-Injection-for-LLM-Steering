# SARI: SAE-Based Adaptive Re-Injection for LLM Steering

**Boston University CS505 NLP Final Project**
Sanjiv Sridhar, Tarushi Gandhi, Amartya Mannava

---

## Overview

SARI is a closed-loop proportional controller for SAE-based activation steering in large language models. Instead of injecting a fixed steering vector at every token, SARI monitors the target feature's live activation and only injects when it falls below a set threshold - preventing the fluency collapse that plagues constant-alpha steering.

---

## Setup

```bash
pip install torch transformers huggingface_hub openai scipy numpy matplotlib
```

Requires a GPU with ~48GB VRAM (tested on NVIDIA L40S). Model and SAE weights are downloaded automatically via HuggingFace Hub on first run.

---

## Key Results

| Method | Concept | Fluency |
|---|---|---|
| No Steering | 1.22 | 4.38 |
| Prompting | **4.33** | **4.43** |
| Constant SAE | 2.40 | 1.47 |
| Decay SAE | 2.71 | 2.69 |
| **SARI (ours)** | **2.98** | **3.37** |

SARI vs. Constant SAE: Wilcoxon p < 0.001, Cohen's d = +0.40

---

## Model and SAE

- **Model**: `google/gemma-2-2b-it`
- **SAE**: `google/gemma-scope-2b-pt-res`, layer 20, width 16K (`average_l0_71`)
- **Steering layer**: 20 (of 26)
- **Decoding**: greedy, seed 42, max 200 tokens
