# CSE 151B Spring 2026 — Math Reasoning Competition

**Model**: `Qwen/Qwen3-4B-Thinking-2507` (no fine-tuning)  
**Leaderboard Score**: 0.671 (n=1) → ~0.72 expected (n=3 voting)

---

## GPU & Runtime

| Item | Details |
|------|---------|
| GPU | NVIDIA A100 40GB (UCSD DataHub) |
| Inference time (n=1) | ~90 minutes (943 questions) |
| Inference time (n=3) | ~8 hours (943 questions) |
| Quantization | INT8 (BitsAndBytes via vLLM) |

---

## Strategy

### Key Insight
The baseline (~40%) used `enable_thinking=False`, disabling the core reasoning capability
of Qwen3-4B-**Thinking**. Enabling thinking with a controlled budget was the primary driver of improvement.

### Improvement Journey

| Configuration | 200-sample Accuracy | Leaderboard |
|---|---|---|
| Baseline: `thinking=OFF, max=2000` | ~15% | ~40% |
| `budget=512, max=4000` | ~30% | — |
| `budget=2048, max=7000, n=1` | 60.0% | **67.1%** |
| `budget=2048, max=7000, n=3 voting` | 65.0% | *~72% expected* |

### Final Strategy
- `enable_thinking=True`, `thinking_budget=2048`, `max_tokens=7000`
- **n=3 self-consistency majority voting** — generate 3 responses, take the most common answer
- **Forced `\boxed{` prefix retry** — for responses with no `\boxed{}`, retry with a prefix completion

---

## Setup

### 1. Install dependencies

```bash
pip install vllm transformers bitsandbytes pandas
```

### 2. Download dataset

Place competition data files in the `data/` directory:

```
data/
  private.jsonl
  public.jsonl
```

### 3. Model weights

Model weights are downloaded automatically from HuggingFace on first run:

```
Qwen/Qwen3-4B-Thinking-2507
```

No manual download required.

---

## Run Inference

### Option A: Python import

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_path="submission.csv",
)
```

### Option B: Command line

```bash
python run_inference.py
```

Both produce `submission.csv` with columns `id` and `response`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | `data/private.jsonl` | Input dataset path |
| `output_path` | `submission.csv` | Output CSV path |
| `model_id` | `Qwen/Qwen3-4B-Thinking-2507` | HuggingFace model ID |
| `n_samples` | `3` | Samples per question (majority voting) |
| `thinking_budget` | `2048` | Max thinking tokens |
| `max_tokens` | `7000` | Max total tokens per response |
| `temperature` | `0.7` | Sampling temperature |
| `gpu_memory_utilization` | `0.55` | Fraction of GPU VRAM to use |

---

## Repository Structure

```
├── README.md
├── run_inference.py          # Single entry point — run_inference()
└── notebook_final_en.ipynb   # Full experiment notebook with improvement journey
```

---

## Reproducibility Notes

- Results may vary slightly due to sampling randomness
- Overall accuracy should be consistent with leaderboard (within ~3-5%)
- For exact reproduction, use `n_samples=3, thinking_budget=2048, temperature=0.7`
