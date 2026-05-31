#!/usr/bin/env python
# coding: utf-8

# # CSE 151B Competition — Math Reasoning with Qwen3-4B-Thinking
# 
# This notebook documents the complete pipeline for the **CSE 151B Spring 2026 Math Reasoning Competition**.
# 
# ## Improvement Journey
# 
# | Strategy | 200-sample Accuracy | Leaderboard Score |
# |---|---|---|
# | Baseline: `thinking=OFF, max=2000` | ~15% | ~40% |
# | `budget=1024, max=2500` | ~15% | — |
# | `budget=512, max=4000` | ~30% | — |
# | `budget=256, max=5000` | ~35% | — |
# | **`budget=2048, max=7000, n=1`** | **60.0%** | **67.1%** |
# | **`budget=2048, max=7000, n=3 voting`** | **65.0%** | *~72% expected* |
# 
# ### Key Insight
# The baseline scored ~40% because `enable_thinking=False` disabled the core reasoning
# capability of Qwen3-4B-**Thinking**.
# 
# Enabling `thinking_budget=2048` with `max_tokens=7000` was the single biggest improvement.
# Without a budget cap, thinking tokens consume all available space and the model never
# writes `\boxed{}`. The budget forces the model to conclude its reasoning and write an answer.
# 

# ## 1. Environment Setup
# 
# Install dependencies using `pip`. After running, restart the kernel to pick up `vllm` and `transformers`.

# ### Comment Out the cell below after first installation.

# In[2]:


# Install uv
get_ipython().system('wget -qO- https://astral.sh/uv/install.sh | sh')

# Create a virtual environment
get_ipython().system('python -m venv .venv')

# Install dependencies — this is fast thanks to uv's parallel resolver
get_ipython().system('.venv/bin/python -m pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter')

# Install Jupyter Kernel
get_ipython().system('.venv/bin/python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"')

print("Done. Restart the kernel before proceeding.")
print("Selection process: on top right, click on current kernel '(ususally named python)' -> 'select another kernel' -> 'Jupyter Kernel' -> 'Python (cse151b)'.")


# In[ ]:


# Install uv
get_ipython().system('wget -qO- https://astral.sh/uv/install.sh | sh')

# Create a virtual environment
get_ipython().system('python -m venv .venv')

# Install dependencies — this is fast thanks to uv's parallel resolver
get_ipython().system('.venv/bin/python -m pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter')

# Install Jupyter Kernel
get_ipython().system('.venv/bin/python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"')

print("Done. Restart the kernel before proceeding.")
print("Selection process: on top right, click on current kernel '(ususally named python)' -> 'select another kernel' -> 'Jupyter Kernel' -> 'Python (cse151b)'.")


# ### Run the cell below every time to activate the installed environment. 

# In[2]:


# activate venv after installation. This needs to be run everytime.
get_ipython().system('source ./.venv/bin/activate')


# ## 2. Imports & Configuration
# 
# All key settings are collected in one place.  
# - `DATA_PATH` — public dataset with ground-truth answers (use this to measure accuracy)
# - `OUTPUT_PATH` — where per-question results will be written
# - `GPU_ID` — which GPU to use (update if your machine has a different device index)
# - `MAX_TOKENS` — maximum tokens the model may generate per response

# In[3]:


get_ipython().system('pip install huggingface-hub==0.34.4 --force-reinstall')


# In[1]:


import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


# In[2]:


import json
import os

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "1"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/starter_results.jsonl"
MAX_TOKENS  = 512

# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ## 3. Load the Dataset
# 
# The dataset is stored as newline-delimited JSON (`.jsonl`). Each line is one question with the following fields:
# 
# | Field | Description |
# |---|---|
# | `id` | Unique question identifier |
# | `question` | Problem statement |
# | `options` | List of answer choices — present for **MCQ**, absent for **free-form** |
# | `answer` | Ground-truth answer (letter for MCQ, value/list for free-form) |

# In[3]:


data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

# Preview one MCQ and one free-form item
mcq_sample  = next(d for d in data if d.get("options"))
free_sample = next(d for d in data if not d.get("options"))

print("\n── MCQ sample ──")
print(json.dumps(mcq_sample, indent=2))
print("\n── Free-form sample ──")
print(json.dumps(free_sample, indent=2))


# ## 4. Prompt Construction
# 
# We use two system prompts depending on the question type:
# 
# - **MCQ** — the model must select the best answer letter and wrap it in `\boxed{}`
# - **Free-form** — the model solves step-by-step and puts the final answer in `\boxed{}`
# 
# `build_prompt()` returns the appropriate `(system, user)` pair for each item.

# In[4]:


SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
        """
    Solve the problem efficiently.
    Do not over-verify or restart the solution.
    When you find a plausible answer, immediately finish.
    You must always end with exactly one final line:
    \\boxed{answer}
    If uncertain, still provide your best guess in \\boxed{}.
    For MCQ, answer only one capital letter inside \\boxed{}.
    For multiple [ANS] blanks, separate answers with commas in order.
    Do not write anything after \\boxed{}.
    
    Do not round or approximate numerical answers.
    Keep exact fractions or expressions when possible.
    
    """
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
    "\n\nIMPORTANT: Keep reasoning under 10 lines. "
    "Write \\boxed{answer} immediately when you reach the answer. "
    "Do NOT write anything after \\boxed{}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
        """
    Solve the problem efficiently.
    Do not over-verify or restart the solution.
    When you find a plausible answer, immediately finish.
    You must always end with exactly one final line:
    \\boxed{answer}
    If uncertain, still provide your best guess in \\boxed{}.
    For MCQ, answer only one capital letter inside \\boxed{}.
    For multiple [ANS] blanks, separate answers with commas in order.
    Do not write anything after \\boxed{}.
    
    Do not round or approximate numerical answers.
    Keep exact fractions or expressions when possible.

    Before your final answer, write no more than 8 lines of reasoning.
    You must end with \boxed{answer}.

    """
    "You are an expert mathematician. "
    "Select the single best answer. Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}."
    "\n\nIMPORTANT: 3 lines of reasoning max, then immediately \\boxed{letter}. Nothing after."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# Verify with samples
for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
    sys_p, usr_p = build_prompt(item["question"], item.get("options"))
    print(f"── {label} user prompt (first 200 chars) ──")
    print(usr_p[:200], "...\n")


# ## 5b. Load Model — Transformers (DataHub fallback)
# 
# Alternative to vLLM using INT4 quantization. Slower but works on more environments.

# In[5]:


from vllm import LLM, SamplingParams
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    enable_prefix_caching=False,
    gpu_memory_utilization=0.55,
    max_model_len=8192,
    trust_remote_code=True,
    max_num_seqs=32,
    max_num_batched_tokens=8192,
    enforce_eager=True,
)

MAX_TOKENS = 8192  

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.3,   # 0.6 → 0.3
    top_p=0.9,         # 0.95 → 0.9
    top_k=20,
    repetition_penalty=1.05,
)
print("Model loaded.")


# In[12]:


get_ipython().system('nvidia-smi')


# ## 5. TEST 

# In[10]:


data = [json.loads(line) for line in open("data/public.jsonl")]  # Verify file path!
print(data[0].keys())


# In[ ]:


# ─── Quick Test: thinking_budget=1024 vs baseline (enable_thinking=False) ───
import re
from collections import defaultdict

data = [json.loads(line) for line in open("data/public.jsonl")]
print(f"public set {len(data)} items loaded, keys: {list(data[0].keys())}")

TEST_N = 20
test_data = data[:TEST_N]

# ── 공통 extract_boxed (think 태그 제거 포함) ──
def extract_boxed_v2(text):
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for t in [clean, text]:
        key = r"\boxed{"
        start = t.rfind(key)
        if start == -1:
            continue
        i, depth, answer = start + len(key), 1, ""
        while i < len(t) and depth > 0:
            ch = t[i]
            if ch == "{":   depth += 1; answer += ch
            elif ch == "}":
                depth -= 1
                if depth > 0: answer += ch
            else: answer += ch
            i += 1
        if depth == 0:
            return answer.strip()
    return ""

# ── 실험 함수 ──
def run_experiment(label, enable_thinking, thinking_budget=None, max_tokens=2500, temperature=0.6):
    params = dict(max_tokens=max_tokens, temperature=temperature, top_p=0.95, top_k=20)
    sp = SamplingParams(**params)

    prompts = []
    for item in test_data:
        system, user = build_prompt(item["question"], item.get("options"))
        kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs
        ))

    outputs = llm.generate(prompts, sp)
    responses = [o.outputs[0].text.strip() for o in outputs]

    no_box, correct = 0, 0
    for item, resp in zip(test_data, responses):
        pred = extract_boxed_v2(resp)
        if not pred:
            no_box += 1
        # MCQ
        if item.get("options"):
            gold = str(item["answer"]).strip().upper()
            m = re.search(r"\\boxed\{([A-Za-z])\}", resp)
            if m and m.group(1).upper() == gold:
                correct += 1
        else:
            # free-form: 단순 string match (빠른 확인용)
            gold_list = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
            preds = [p.strip() for p in pred.split(",")]
            if len(preds) == len(gold_list) and all(p == g for p, g in zip(preds, gold_list)):
                correct += 1

    print(f"\n{'='*50}")
    print(f"[{label}]")
    print(f"  No \\boxed{{}} : {no_box}/{TEST_N} ({no_box/TEST_N*100:.1f}%)")
    print(f"  Accuracy     : {correct}/{TEST_N} ({correct/TEST_N*100:.1f}%)")
    print(f"  tokens       : max={max_tokens}, thinking={'budget='+str(thinking_budget) if thinking_budget else 'OFF'}")
    return responses

# ── 실험 A: 베이스라인 (thinking OFF, max_tokens=2000) ──
resp_A = run_experiment("Baseline (thinking=OFF, max=2000)", enable_thinking=False, max_tokens=2000, temperature=0.3)

# ── 실험 B: thinking ON + budget 1024 ──
resp_B = run_experiment("Thinking ON (budget=1024, max=2500)", enable_thinking=True, thinking_budget=1024, max_tokens=2500, temperature=0.6)

# ── 틀린 케이스 비교 ──
print("\n── 틀린 케이스 샘플 (Thinking ON 기준) ──")
for i, (item, resp) in enumerate(zip(test_data, resp_B)):
    pred = extract_boxed_v2(resp)
    gold = item["answer"]
    if not pred:
        print(f"[{i}] NO BOXED | gold={gold}")
        print("  last 200:", resp[-200:])
    elif pred != (gold[0] if isinstance(gold, list) else gold):
        print(f"[{i}] WRONG | pred={pred} | gold={gold}")

# ── 실험 C: budget 줄이고 답 쓸 공간 확보 ──
resp_C = run_experiment(
    "Thinking ON (budget=512, max=4000)",
    enable_thinking=True,
    thinking_budget=512,
    max_tokens=4000,
    temperature=0.6
)

# Inspect NO-BOXED cases
print("\n── NO BOXED 케이스 (실험 C) ──")
for i, (item, resp) in enumerate(zip(test_data, resp_C)):
    pred = extract_boxed_v2(resp)
    if not pred:
        print(f"[{i}] gold={item['answer']} | last 150: {resp[-150:]}")
        print()


# In[14]:


# ── 실험 D: budget 더 줄이기 ──
resp_D = run_experiment(
    "Thinking ON (budget=256, max=5000)",
    enable_thinking=True,
    thinking_budget=256,
    max_tokens=5000,
    temperature=0.6
)

print("\n── NO BOXED 케이스 (실험 D) ──")
for i, (item, resp) in enumerate(zip(test_data, resp_D)):
    pred = extract_boxed_v2(resp)
    if not pred:
        print(f"[{i}] gold={item['answer']} | last 150: {resp[-150:]}")
        print()


# In[15]:


# ── 실험 D 결과에 retry 추가 ──
retry_indices = [i for i, resp in enumerate(resp_D) if not extract_boxed_v2(resp)]
print(f"retry 대상: {retry_indices}")

if retry_indices:
    retry_sp = SamplingParams(max_tokens=1500, temperature=0.1, top_p=0.9)
    retry_prompts = []
    for i in retry_indices:
        item = test_data[i]
        system, user = build_prompt(item["question"], item.get("options"))
        system += "\n\nIMPORTANT: Write ONLY \\boxed{answer} at the end. No further text after \\boxed{}."
        retry_prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        ))

    retry_outputs = llm.generate(retry_prompts, retry_sp)
    for j, out in enumerate(retry_outputs):
        resp_D[retry_indices[j]] = out.outputs[0].text.strip()

# Re-score results
no_box2, correct2 = 0, 0
for item, resp in zip(test_data, resp_D):
    pred = extract_boxed_v2(resp)
    if not pred:
        no_box2 += 1
    if item.get("options"):
        gold = str(item["answer"]).strip().upper()
        m = re.search(r"\\boxed\{([A-Za-z])\}", resp)
        if m and m.group(1).upper() == gold:
            correct2 += 1
    else:
        gold_list = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        preds = [p.strip() for p in pred.split(",")]
        if len(preds) == len(gold_list) and all(p == g for p, g in zip(preds, gold_list)):
            correct2 += 1

print(f"retry 후 No \\boxed{{}}: {no_box2}/20")
print(f"retry 후 Accuracy: {correct2}/20 ({correct2/20*100:.1f}%)")


# In[16]:


retry_indices = [i for i, resp in enumerate(resp_D) if not extract_boxed_v2(resp)]

if retry_indices:
    retry_sp = SamplingParams(max_tokens=200, temperature=0.1)  # 짧게 강제
    retry_prompts = []
    for i in retry_indices:
        item = test_data[i]
        if item.get("options"):
            labels = [chr(65+j) for j in range(len(item["options"]))]
            opts = "\n".join(f"{l}. {o}" for l, o in zip(labels, item["options"]))
            user_msg = f"{item['question']}\n\nOptions:\n{opts}\n\nAnswer with ONE letter only:"
        else:
            gold_len = len(item["answer"]) if isinstance(item["answer"], list) else 1
            user_msg = f"{item['question']}\n\nProvide only the final answer(s) {'comma-separated ' if gold_len > 1 else ''}in \\boxed{{}}:"

        retry_prompts.append(tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Output ONLY \\boxed{answer}. No reasoning. No explanation. Just \\boxed{}."},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        ))

    retry_outputs = llm.generate(retry_prompts, retry_sp)
    for j, out in enumerate(retry_outputs):
        new_resp = out.outputs[0].text.strip()
        print(f"[{retry_indices[j]}] retry → {new_resp[:100]}")
        resp_D[retry_indices[j]] = new_resp


# In[17]:


retry_indices = [i for i, resp in enumerate(resp_D) if not extract_boxed_v2(resp)]

if retry_indices:
    # MCQ: 아주 짧게 (letter 하나만 생성)
    # Free-form: 조금 더 허용
    retry_prompts = []
    retry_is_mcq = []

    for i in retry_indices:
        item = test_data[i]
        system, user = build_prompt(item["question"], item.get("options"))
        
        base = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        # \boxed{ 까지 붙여서 모델이 답만 완성하게
        retry_prompts.append(base + "\\boxed{")
        retry_is_mcq.append(bool(item.get("options")))

    mcq_sp   = SamplingParams(max_tokens=5,   temperature=0.1, stop=["}"])
    free_sp  = SamplingParams(max_tokens=100, temperature=0.1, stop=["}"])

    for j, (prompt, is_mcq) in enumerate(zip(retry_prompts, retry_is_mcq)):
        sp = mcq_sp if is_mcq else free_sp
        out = llm.generate([prompt], sp)
        raw = out[0].outputs[0].text.strip()
        full_resp = f"\\boxed{{{raw}}}"
        print(f"[{retry_indices[j]}] gold={test_data[retry_indices[j]]['answer']} → {full_resp}")
        resp_D[retry_indices[j]] = full_resp


# In[18]:


# ── 200개 샘플로 accuracy 추정 ──
import json, re, random
from tqdm import tqdm

# Load data
data = [json.loads(line) for line in open("data/public.jsonl")]

# 랜덤 200개 샘플
random.seed(42)
sample_200 = random.sample(data, 200)
TEST_N = 200
test_data = sample_200

print(f"샘플 {TEST_N}개 준비 (전체 {len(data)}개 중)")
mcq_cnt  = sum(1 for d in test_data if d.get("options"))
free_cnt = TEST_N - mcq_cnt
print(f"MCQ: {mcq_cnt}개 / Free-form: {free_cnt}개")

# ── 1. 응답 생성 ──
resp_200 = run_experiment(
    "200-sample (budget=256, max=4000)",
    enable_thinking=True,
    thinking_budget=256,
    max_tokens=4000,
    temperature=0.6
)

# ── 2. no-boxed retry ──
retry_indices = [i for i, r in enumerate(resp_200) if not extract_boxed_v2(r)]
print(f"\nno-boxed {len(retry_indices)}개 retry 중...")

if retry_indices:
    retry_prompts, retry_is_mcq = [], []
    for i in retry_indices:
        item = test_data[i]
        system, user = build_prompt(item["question"], item.get("options"))
        base = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        retry_prompts.append(base + "\\boxed{")
        retry_is_mcq.append(bool(item.get("options")))

    for j, (prompt, is_mcq) in enumerate(zip(retry_prompts, retry_is_mcq)):
        sp = SamplingParams(max_tokens=5 if is_mcq else 100, temperature=0.1, stop=["}"])
        out = llm.generate([prompt], sp)
        raw = out[0].outputs[0].text.strip()
        resp_200[retry_indices[j]] = f"\\boxed{{{raw}}}"

    remaining = sum(1 for r in resp_200 if not extract_boxed_v2(r))
    print(f"retry 후 no-boxed: {remaining}개")

# ── 3. judger로 채점 ──
import sys
sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

results_200 = []
for item, resp in tqdm(zip(test_data, resp_200), total=TEST_N, desc="채점"):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]

    if is_mcq:
        m = re.search(r"\\boxed\{([A-Za-z])\}", resp)
        ok = (m.group(1).upper() == str(gold).strip().upper()) if m else False
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            ok = judger.auto_judge(pred=resp, gold=gold_list, options=[[]] * len(gold_list))
        except Exception:
            ok = False

    results_200.append({"id": item["id"], "is_mcq": is_mcq, "gold": gold, "response": resp, "correct": ok})

# ── 4. 결과 출력 ──
mcq_res  = [r for r in results_200 if r["is_mcq"]]
free_res = [r for r in results_200 if not r["is_mcq"]]

def pct(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

print(f"\n{'='*50}")
print(f"MCQ       : {sum(r['correct'] for r in mcq_res):3d}/{len(mcq_res):3d} ({pct(mcq_res):.1f}%)")
print(f"Free-form : {sum(r['correct'] for r in free_res):3d}/{len(free_res):3d} ({pct(free_res):.1f}%)")
print(f"Overall   : {sum(r['correct'] for r in results_200):3d}/{TEST_N:3d} ({pct(results_200):.1f}%)")


# In[19]:


# ── budget=2048 vs budget=256 비교 실험 ──
import random

data = [json.loads(line) for line in open("data/public.jsonl")]
random.seed(42)
test_data = random.sample(data, 200)
TEST_N = 200

print(f"샘플 {TEST_N}개 (seed=42, 이전과 동일)")

# ── 생성 ──
resp_2048 = run_experiment(
    "200-sample (budget=2048, max=7000)",
    enable_thinking=True,
    thinking_budget=2048,
    max_tokens=7000,
    temperature=0.6
)

# ── 배치 retry ──
retry_indices = [i for i, r in enumerate(resp_2048) if not extract_boxed_v2(r)]
print(f"\nno-boxed {len(retry_indices)}개 배치 retry 중...")

if retry_indices:
    retry_prompts = []
    for i in retry_indices:
        item = test_data[i]
        system, user = build_prompt(item["question"], item.get("options"))
        base = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        retry_prompts.append(base + "\\boxed{")

    retry_sp = SamplingParams(max_tokens=100, temperature=0.1, stop=["}"])
    retry_outputs = llm.generate(retry_prompts, retry_sp)
    for j, out in enumerate(retry_outputs):
        raw = out.outputs[0].text.strip()
        resp_2048[retry_indices[j]] = f"\\boxed{{{raw}}}"

print(f"retry 후 no-boxed: {sum(1 for r in resp_2048 if not extract_boxed_v2(r))}개")

# ── judger 채점 ──
from judger import Judger
judger = Judger(strict_extract=False)

results_2048 = []
for item, resp in zip(test_data, resp_2048):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    if is_mcq:
        m  = re.search(r"\\boxed\{([A-Za-z])\}", resp)
        ok = (m.group(1).upper() == str(gold).strip().upper()) if m else False
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            ok = judger.auto_judge(pred=resp, gold=gold_list, options=[[]] * len(gold_list))
        except Exception:
            ok = False
    results_2048.append({"is_mcq": is_mcq, "correct": ok})

mcq_res  = [r for r in results_2048 if r["is_mcq"]]
free_res = [r for r in results_2048 if not r["is_mcq"]]

print(f"\n{'='*50}")
print(f"[budget=2048, max=7000]")
print(f"MCQ       : {sum(r['correct'] for r in mcq_res)}/{len(mcq_res)} ({sum(r['correct'] for r in mcq_res)/len(mcq_res)*100:.1f}%)")
print(f"Free-form : {sum(r['correct'] for r in free_res)}/{len(free_res)} ({sum(r['correct'] for r in free_res)/len(free_res)*100:.1f}%)")
print(f"Overall   : {sum(r['correct'] for r in results_2048)}/{TEST_N} ({sum(r['correct'] for r in results_2048)/TEST_N*100:.1f}%)")
print(f"\n이전 budget=256: 55.5% → 이번:")


# In[7]:


# ── private set 제출 파일 생성 (budget=2048) ──
import json, re, csv
import pandas as pd

data_priv = [json.loads(line) for line in open("data/private.jsonl")]
test_data  = data_priv
TEST_N     = len(data_priv)
print(f"private set: {TEST_N}개")

resp_priv = run_experiment(
    "Private (budget=2048, max=7000)",
    enable_thinking=True,
    thinking_budget=2048,
    max_tokens=7000,
    temperature=0.6
)

# 배치 retry
retry_indices = [i for i, r in enumerate(resp_priv) if not extract_boxed_v2(r)]
print(f"no-boxed {len(retry_indices)}개 배치 retry...")

if retry_indices:
    retry_prompts = []
    for i in retry_indices:
        item = test_data[i]
        system, user = build_prompt(item["question"], item.get("options"))
        base = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        retry_prompts.append(base + "\\boxed{")
    retry_sp = SamplingParams(max_tokens=100, temperature=0.1, stop=["}"])
    retry_outputs = llm.generate(retry_prompts, retry_sp)
    for j, out in enumerate(retry_outputs):
        raw = out.outputs[0].text.strip()
        resp_priv[retry_indices[j]] = f"\\boxed{{{raw}}}"

print(f"최종 no-boxed: {sum(1 for r in resp_priv if not extract_boxed_v2(r))}개")

df = pd.DataFrame([{"id": item["id"], "response": resp}
                   for item, resp in zip(data_priv, resp_priv)])
df.to_csv("submission_budget2048.csv", index=False, quoting=csv.QUOTE_ALL)
print(f"저장 완료: submission_budget2048.csv ({len(df)}행)")


# In[6]:


# ── 함수 정의만 ──
import re, json
from vllm import SamplingParams

def extract_boxed_v2(text):
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for t in [clean, text]:
        key = r"\boxed{"
        start = t.rfind(key)
        if start == -1:
            continue
        i, depth, answer = start + len(key), 1, ""
        while i < len(t) and depth > 0:
            ch = t[i]
            if ch == "{":   depth += 1; answer += ch
            elif ch == "}":
                depth -= 1
                if depth > 0: answer += ch
            else: answer += ch
            i += 1
        if depth == 0:
            return answer.strip()
    return ""

def run_experiment(label, enable_thinking, thinking_budget=None, max_tokens=2500, temperature=0.6):
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95, top_k=20)
    prompts = []
    for item in test_data:
        system, user = build_prompt(item["question"], item.get("options"))
        kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs
        ))
    outputs = llm.generate(prompts, sp)
    responses = [o.outputs[0].text.strip() for o in outputs]
    no_box = sum(1 for r in responses if not extract_boxed_v2(r))
    print(f"[{label}] no-boxed: {no_box}/{len(test_data)}, tokens: max={max_tokens}, thinking={'budget='+str(thinking_budget) if thinking_budget else 'OFF'}")
    return responses

print("함수 정의 완료: extract_boxed_v2, run_experiment")


# ## 5. Load Model with Transformers (alternative to vLLM for DataHub)
# 
# We load **Qwen3-4B-Thinking-2507** with **INT4 quantization** via BitsAndBytes.  
# 
# Key parameters:
# - `load_in_4bit` — quantization strategy of INT4

# In[ ]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
)


# ## 6. Generate Responses
# 
# We format every question into a chat-template prompt, then call `llm.generate()` in one batched pass.  
# vLLM handles batching and scheduling internally — no manual batching needed.

# ### Generate with vLLM

# In[ ]:


MAX_TOKENS = 2000
RETRY_MAX_TOKENS = 4000

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.3,
    top_p=0.9,
    top_k=20,
    repetition_penalty=1.05,
)

retry_sampling_params = SamplingParams(
    max_tokens=RETRY_MAX_TOKENS,
    temperature=0.25,
    top_p=0.9,
    top_k=20,
    repetition_penalty=1.1,
)

# Build prompts
prompts = []
for item in data:
    system, user = build_prompt(item["question"], item.get("options"))

    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompts.append(prompt_text)

print(f"Generating responses for {len(prompts)} questions...")
outputs = llm.generate(prompts, sampling_params=sampling_params)
responses = [out.outputs[0].text.strip() for out in outputs]

# -------------------------
# Extract boxed answers
# -------------------------
def extract_boxed(text):
    key = r"\boxed{"
    start = text.rfind(key)
    if start == -1:
        return ""
    i = start + len(key)
    depth = 1
    answer = ""
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            answer += ch
        elif ch == "}":
            depth -= 1
            if depth > 0:
                answer += ch
        else:
            answer += ch
        i += 1
    if depth == 0:
        return answer.strip()
    return ""

# -------------------------
# Retry
# -------------------------
retry_indices = [i for i, r in enumerate(responses) if not extract_boxed(r)]
print("No boxed before retry:", retry_indices)
print("Count:", len(retry_indices))

if retry_indices:
    retry_prompts_no_think = []
    for i in retry_indices:
        item = data[i]
        system, user = build_prompt(item["question"], item.get("options"))

        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        retry_prompts_no_think.append(prompt_text)

    retry_outputs = llm.generate(retry_prompts_no_think, sampling_params=retry_sampling_params)
    for j, out in enumerate(retry_outputs):
        original_idx = retry_indices[j]
        responses[original_idx] = out.outputs[0].text.strip()

# -------------------------
# Final predictions
# -------------------------
predictions = [extract_boxed(r) for r in responses]
failed_indices = [i for i, p in enumerate(predictions) if not p]
print("Still no boxed answer:", failed_indices)
print("Count:", len(failed_indices))
for i, p in enumerate(predictions[:10]):
    print(i, p)
for i in range(min(3, len(responses))):
    print(f"\n── Response {i} (id={data[i].get('id')}) ──")
    print(responses[i][-800:])



# In[ ]:


import pandas as pd
from tqdm import tqdm
import csv

results5 = []

for i, (item, response) in enumerate(tqdm(
    zip(data[:len(responses)], responses),
    total=len(responses),
    desc="Building results5"
)):
    prediction = extract_boxed(response)

    results5.append({
        "id": item.get("id"),
        "is_mcq": bool(item.get("options")),
        "response": response,
        "prediction": prediction,
    })

print(f"Done. {len(results5)} results.")

# Kaggle submission: id + response columns
df5 = pd.DataFrame([
    {
        "id": r["id"],
        "response": r["response"]
    }
    for r in results5
])

df5.to_csv("submission5.csv", index=False, quoting=csv.QUOTE_ALL)
print(f"Saved {len(df5)} rows to submission5.csv")

# Debug file including predictions
debug_df5 = pd.DataFrame(results5)
debug_df5.to_csv("debug_results5.csv", index=False, quoting=csv.QUOTE_ALL)
print(f"Saved debug file to debug_results5.csv")


# ### Generate with Transformers (for Datahub)

# In[16]:


pip install pandas tqdm


# In[24]:


MAX_TOKENS = 2048
RETRY_MAX_TOKENS = 4096

prompts = []

for item in data:
    system, user = build_prompt(item["question"], item.get("options"))

    system += """
Solve the problem efficiently.

Do not over-verify or restart the solution.
When you find a plausible answer, immediately finish.

You must always end with exactly one final line:
\\boxed{answer}

If uncertain, still provide your best guess in \\boxed{}.

For MCQ, answer only one capital letter inside \\boxed{}.
For multiple [ANS] blanks, separate answers with commas in order.

Do not write anything after \\boxed{}.
"""

    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt_text)

print(f"Generating responses for {len(prompts)} questions...")

inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=16384,
).to(llm.device)

with torch.no_grad():
    output_ids = llm.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        top_k=20,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

responses = []

prompt_len = inputs["input_ids"].shape[1]

for i, out in enumerate(output_ids):
    actual_prompt_len = inputs["attention_mask"][i].sum().item()
    new_tokens = out[actual_prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    responses.append(text)


# -------------------------
# Extract boxed answers
# -------------------------

import re

def extract_boxed(text):
    key = r"\boxed{"
    start = text.rfind(key)
    if start == -1:
        return ""

    i = start + len(key)
    depth = 1
    answer = ""

    while i < len(text) and depth > 0:
        ch = text[i]

        if ch == "{":
            depth += 1
            answer += ch
        elif ch == "}":
            depth -= 1
            if depth > 0:
                answer += ch
        else:
            answer += ch

        i += 1

    if depth == 0:
        return answer.strip()

    return ""


# -------------------------
# Retry only responses without boxed answer
# -------------------------

retry_indices = [i for i, r in enumerate(responses) if not extract_boxed(r)]

print("No boxed before retry:", retry_indices)
print("Count:", len(retry_indices))

if retry_indices:
    retry_prompts = [prompts[i] for i in retry_indices]

    retry_inputs = tokenizer(
        retry_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16384,
    ).to(llm.device)

    with torch.no_grad():
        retry_output_ids = llm.generate(
            **retry_inputs,
            max_new_tokens=RETRY_MAX_TOKENS,
            do_sample=True,
            temperature=0.25,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            thinking_budget=1024
        )

    retry_prompt_len = retry_inputs["input_ids"].shape[1]

    for j, out in enumerate(retry_output_ids):
        original_idx = retry_indices[j]
        actual_prompt_len = retry_inputs["attention_mask"][j].sum().item()  # ← 수정
        new_tokens = out[actual_prompt_len:]
        retry_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses[original_idx] = retry_text


# -------------------------
# Final predictions
# -------------------------

predictions = [extract_boxed(r) for r in responses]

failed_indices = [i for i, p in enumerate(predictions) if not p]

print("Still no boxed answer:", failed_indices)
print("Count:", len(failed_indices))

for i, p in enumerate(predictions[:10]):
    print(i, p)

for i in range(min(3, len(responses))):
    print(f"\n── Response {i} (id={data[i].get('id')}) ──")
    print(responses[i][-800:])


# In[19]:


print(responses[5])


# In[15]:


import re

failed_indices = []

for i, r in enumerate(responses):
    boxes = re.findall(r"\\boxed\{([^}]*)\}", r)
    if not boxes:
        failed_indices.append(i)

print("No boxed answer:", failed_indices)
print("Count:", len(failed_indices))


# In[19]:





# In[20]:


# Re-check boxed answers
predictions = [extract_boxed(r) for r in responses]

failed_indices = []

for i, p in enumerate(predictions):
    if not p:
        failed_indices.append(i)

print("Still no boxed answer:", failed_indices)
print("Count:", len(failed_indices))

for i, p in enumerate(predictions[:10]):
    print(i, p)


# ## 7. Score Responses
# 
# Scoring differs by question type:
# 
# - **MCQ**: extract the predicted letter from `\boxed{}` and compare to the gold letter (exact match).
# - **Free-form**: use `Judger.auto_judge()` which handles symbolic and numeric equivalence.
# 
# Each result record contains `{id, is_mcq, gold, response, correct}`.

# In[8]:


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


# Load Judger for free-form scoring
sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

results = []
for item, response in tqdm(
    zip(data[:len(responses)], responses),
    total=len(responses),
    desc="Scoring"
):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]

    if is_mcq:
        correct = score_mcq(response, str(gold))
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception:
            correct = False

    results.append({
        "id":       item.get("id"),
        "is_mcq":   is_mcq,
        "gold":     gold,
        "response": response,
        "correct":  correct,
    })

print(f"Scoring complete. {len(results)} results.")


# ## 8. Summary
# 
# Print accuracy broken down by question type.

# In[9]:


results2 = []
for i, (item, response) in enumerate(tqdm(
    zip(data[:len(responses)], responses),
    total=len(responses),
    desc="Building results2"
)):
    prediction = extract_boxed(response)
    if not prediction:
        prediction = fallback_answer(item, response)
        print(f"  [fallback] idx={i} → '{prediction}'")
    
    results2.append({
        "id":         item.get("id"),
        "is_mcq":     bool(item.get("options")),
        "response":   response,
        "prediction": prediction,
    })

print(f"Done. {len(results2)} results.")

df2 = pd.DataFrame([{"id": r["id"], "response": r["response"]} for r in results2])
df2.to_csv("submission2.csv", index=False, quoting=1)
print(f"Saved {len(df2)} rows to submission2.csv")


# In[22]:


mcq_res  = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]

def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
print("=" * 50)


# In[23]:


for r in results:
    if not r["correct"]:
        print("="*50)
        print("ID:", r["id"])
        print("MCQ:", r["is_mcq"])
        print("GOLD:", r["gold"])
        print("RESPONSE:", r["response"][:500])


# ## 9. Save Results
# 
# Results are written as newline-delimited JSON.
# 
# **With evaluation** (public set — you have ground-truth):  
# Each line: `{id, is_mcq, gold, response, correct}`
# 
# **Without evaluation** (private test set — no ground-truth available):  
# Each line: `{id, is_mcq, response}` — omit `gold` and `correct`.
# 
# Toggle `SAVE_EVAL` below accordingly.

# In[12]:


SAVE_EVAL = False   # Set to False when running on the private test set

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    for r in results:
        if SAVE_EVAL:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                      "response": r["response"], "correct": r["correct"]}
        else:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
        f.write(json.dumps(record) + "\n")

print(f"Saved {len(results)} records to {out_path}")


# ## Next Steps
# 
# This notebook gives you a working baseline. Here are directions to improve your score:
# 
# - **Prompt engineering** — try different system prompts or few-shot examples inside the user turn
# - **Sampling parameters** — adjust `temperature`, `top_p`, or use majority voting across multiple samples
# - **Fine-tuning** — the competition allows model fine-tuning; see the course resources for guidance
# 
# Good luck!

# In[44]:


import torch, gc

del inputs
del output_ids
gc.collect()
torch.cuda.empty_cache()


# In[45]:


import torch, gc
gc.collect()
torch.cuda.empty_cache()


# In[10]:


print(len(results))
print(len(responses))
print(len(predictions))


# In[ ]:





# ---
# ## Section 10: Improvement Journey — Thinking Budget Experiments
# 
# We systematically tested different `thinking_budget` values to find the optimal balance between reasoning depth and answer completeness.

# ### Why No-Boxed Failures Happen
# 
# When `enable_thinking=True` without a budget, the model uses all available tokens for
# reasoning and never writes the final `\boxed{}` answer.
# 
# **Fix**: Set `thinking_budget` to cap the thinking phase, leaving room for the answer.
# 
# | Config | No-boxed Rate | Notes |
# |---|---|---|
# | `thinking=OFF` | 60% | Model writes too much explanation |
# | `budget=1024, max=2500` | 55% | Budget too high for remaining tokens |
# | `budget=512, max=4000` | 35% | Better balance |
# | `budget=256, max=5000` | 30% | More answer space |
# | **`budget=2048, max=7000`** | **16.5%** | Optimal — enough thinking + enough answer space |
# | budget=2048 + n=3 voting | 9.5% | Multiple samples reduce format failures |
# 

# In[ ]:


# ── Helper Functions (run before any experiment) ──
import re, json, random
from collections import Counter
from vllm import SamplingParams

def extract_boxed_v2(text):
    """Extract last \\boxed{} answer, stripping <think>...</think> first."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for t in [clean, text]:
        key = r"\boxed{"
        start = t.rfind(key)
        if start == -1:
            continue
        i, depth, answer = start + len(key), 1, ""
        while i < len(t) and depth > 0:
            ch = t[i]
            if ch == "{":   depth += 1; answer += ch
            elif ch == "}":
                depth -= 1
                if depth > 0: answer += ch
            else: answer += ch
            i += 1
        if depth == 0:
            return answer.strip()
    return ""

def run_experiment(label, enable_thinking, thinking_budget=None, max_tokens=2500, temperature=0.6):
    """Run inference on global test_data and report no-boxed rate."""
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95, top_k=20)
    prompts = []
    for item in test_data:
        system, user = build_prompt(item["question"], item.get("options"))
        kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs
        ))
    outputs = llm.generate(prompts, sp)
    responses = [o.outputs[0].text.strip() for o in outputs]
    no_box = sum(1 for r in responses if not extract_boxed_v2(r))
    print(f"[{label}]")
    print(f"  No-boxed : {no_box}/{len(test_data)} ({no_box/len(test_data)*100:.1f}%)")
    print(f"  Tokens   : max={max_tokens}, thinking={'budget='+str(thinking_budget) if thinking_budget else 'OFF'}")
    return responses

print("Functions ready: extract_boxed_v2, run_experiment")


# ### Experiment Results: Budget Tuning (20 samples)
# 
# Run the cell below to reproduce the budget tuning experiments.

# In[ ]:


# ── Budget Tuning Experiments (20 samples) ──
data      = [json.loads(line) for line in open("data/public.jsonl")]
TEST_N    = 20
test_data = data[:TEST_N]

# Experiment A: Baseline — thinking disabled
resp_A = run_experiment("A: Baseline (thinking=OFF, max=2000)",
                        enable_thinking=False, max_tokens=2000, temperature=0.3)

# Experiment B: budget=1024 — too little answer space
resp_B = run_experiment("B: budget=1024, max=2500",
                        enable_thinking=True, thinking_budget=1024, max_tokens=2500, temperature=0.6)

# Experiment C: budget=512 — better balance
resp_C = run_experiment("C: budget=512, max=4000",
                        enable_thinking=True, thinking_budget=512,  max_tokens=4000, temperature=0.6)

# Experiment D: budget=256 — even more answer space
resp_D = run_experiment("D: budget=256, max=5000",
                        enable_thinking=True, thinking_budget=256,  max_tokens=5000, temperature=0.6)

# Winner: budget=2048, max=7000 (tested on 200 samples below)


# ### Retry Strategy: Forced `\boxed{` Prefix
# 
# For responses that have no `\boxed{}`, we retry by appending `\boxed{` to the prompt.
# This forces the model to complete the answer in one token (for MCQ) or a few tokens (free-form).
# 
# This approach is much faster than re-running the full inference.
# 

# In[ ]:


# ── Batch Retry for No-Boxed Responses ──
def batch_retry(test_data, responses):
    """Retry no-boxed responses using forced \\boxed{ prefix completion."""
    retry_indices = [i for i, r in enumerate(responses) if not extract_boxed_v2(r)]
    if not retry_indices:
        return responses
    print(f"Retrying {len(retry_indices)} no-boxed responses...")

    retry_prompts = []
    for i in retry_indices:
        item = test_data[i]
        system, user = build_prompt(item["question"], item.get("options"))
        base = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        retry_prompts.append(base + "\\boxed{")

    retry_sp  = SamplingParams(max_tokens=100, temperature=0.1, stop=["}"])
    retry_out = llm.generate(retry_prompts, retry_sp)
    for j, out in enumerate(retry_out):
        raw = out.outputs[0].text.strip()
        responses[retry_indices[j]] = f"\\boxed{{{raw}}}"

    remaining = sum(1 for r in responses if not extract_boxed_v2(r))
    print(f"After retry — no-boxed: {remaining}/{len(responses)}")
    return responses


# ---
# ## Section 11: Best Single-Pass Strategy
# 
# `thinking_budget=2048, max_tokens=7000, temperature=0.6`
# 
# **200-sample accuracy: 60.0% | Leaderboard: 67.1%**

# In[ ]:


# ── 200-Sample Evaluation: budget=2048, max=7000 ──
import sys
from tqdm import tqdm

data      = [json.loads(line) for line in open("data/public.jsonl")]
random.seed(42)
test_data = random.sample(data, 200)
TEST_N    = 200
print(f"Sample: {TEST_N} items (seed=42)")

resp_200 = run_experiment("200-sample (budget=2048, max=7000)",
                          enable_thinking=True, thinking_budget=2048,
                          max_tokens=7000, temperature=0.6)
resp_200 = batch_retry(test_data, resp_200)

# Score with judger
sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

results = []
for item, resp in tqdm(zip(test_data, resp_200), total=TEST_N, desc="Scoring"):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    if is_mcq:
        m  = re.search(r"\\boxed\{([A-Za-z])\}", resp)
        ok = (m.group(1).upper() == str(gold).strip().upper()) if m else False
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:    ok = judger.auto_judge(pred=resp, gold=gold_list, options=[[]] * len(gold_list))
        except: ok = False
    results.append({"is_mcq": is_mcq, "correct": ok})

mcq_r  = [r for r in results if r["is_mcq"]]
free_r = [r for r in results if not r["is_mcq"]]
print(f"\n{'='*50}")
print(f"MCQ       : {sum(r['correct'] for r in mcq_r)}/{len(mcq_r)} ({sum(r['correct'] for r in mcq_r)/len(mcq_r)*100:.1f}%)")
print(f"Free-form : {sum(r['correct'] for r in free_r)}/{len(free_r)} ({sum(r['correct'] for r in free_r)/len(free_r)*100:.1f}%)")
print(f"Overall   : {sum(r['correct'] for r in results)}/200 ({sum(r['correct'] for r in results)/2:.1f}%)")
# Result: MCQ 62.4% | Free-form 58.3% | Overall 60.0% → Leaderboard 67.1%


# ---
# ## Section 12: Self-Consistency (n=3 Majority Voting)
# 
# Generate 3 independent responses per question and select the most common answer.
# 
# **Results on 200-sample benchmark (seed=42)**:
# 
# | Method | MCQ | Free-form | Overall |
# |---|---|---|---|
# | n=1, budget=2048 | 62.4% | 58.3% | 60.0% |
# | **n=3, budget=2048** | **69.4%** | **61.7%** | **65.0%** |
# 
# MCQ improved **+7%** with majority voting. Expected leaderboard score: **~72%**.
# 
# vLLM's `n=3` parameter generates all 3 samples in a single batched call — efficient.
# 

# In[ ]:


# ── Self-Consistency Test: n=3 Voting, 200 samples ──
from tqdm import tqdm

data      = [json.loads(line) for line in open("data/public.jsonl")]
random.seed(42)
test_data = random.sample(data, 200)
TEST_N    = 200
print(f"Sample: {TEST_N} items | n=3 majority voting...")

sp_vote = SamplingParams(n=3, max_tokens=7000, temperature=0.7, top_p=0.95, top_k=20)

prompts = []
for item in test_data:
    system, user = build_prompt(item["question"], item.get("options"))
    prompts.append(tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=True, thinking_budget=2048
    ))

outputs = llm.generate(prompts, sp_vote)

final_responses = []
for out in outputs:
    candidates = [o.text.strip() for o in out.outputs]
    answers    = [extract_boxed_v2(c) for c in candidates]
    valid      = [(a, c) for a, c in zip(answers, candidates) if a]
    if not valid:
        final_responses.append(candidates[0])
    else:
        best_ans  = Counter(a for a, _ in valid).most_common(1)[0][0]
        best_resp = next(c for a, c in valid if a == best_ans)
        final_responses.append(best_resp)

final_responses = batch_retry(test_data, final_responses)

# Score
sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

results = []
for item, resp in tqdm(zip(test_data, final_responses), total=TEST_N, desc="Scoring"):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    if is_mcq:
        m  = re.search(r"\\boxed\{([A-Za-z])\}", resp)
        ok = (m.group(1).upper() == str(gold).strip().upper()) if m else False
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:    ok = judger.auto_judge(pred=resp, gold=gold_list, options=[[]] * len(gold_list))
        except: ok = False
    results.append({"is_mcq": is_mcq, "correct": ok})

mcq_r  = [r for r in results if r["is_mcq"]]
free_r = [r for r in results if not r["is_mcq"]]
print(f"\n{'='*50}")
print(f"[n=3 Self-Consistency, budget=2048]")
print(f"MCQ       : {sum(r['correct'] for r in mcq_r)}/{len(mcq_r)} ({sum(r['correct'] for r in mcq_r)/len(mcq_r)*100:.1f}%)")
print(f"Free-form : {sum(r['correct'] for r in free_r)}/{len(free_r)} ({sum(r['correct'] for r in free_r)/len(free_r)*100:.1f}%)")
print(f"Overall   : {sum(r['correct'] for r in results)}/200 ({sum(r['correct'] for r in results)/2:.1f}%)")
# Result: MCQ 69.4% | Free-form 61.7% | Overall 65.0%


# ---
# ## Section 13: Private Set Submission
# 
# ### Submission History
# | File | Strategy | Leaderboard |
# |---|---|---|
# | `submission_budget2048.csv` | n=1, budget=2048, max=7000 | **67.1%** |
# | `submission_n3_voting.csv` | n=3, budget=2048, max=7000 | *in progress* |
# 
# Run **Option A** or **Option B** below to generate a submission file.
# 

# ### Option A: n=1, budget=2048 (Leaderboard: **67.1%**)

# In[ ]:


# ── Private Set Submission: n=1, budget=2048, max=7000 ──
import csv
import pandas as pd

data_priv = [json.loads(line) for line in open("data/private.jsonl")]
test_data  = data_priv
TEST_N     = len(data_priv)
print(f"Private set: {TEST_N} questions (~90 min)")

sp = SamplingParams(max_tokens=7000, temperature=0.6, top_p=0.95, top_k=20)
prompts = []
for item in test_data:
    system, user = build_prompt(item["question"], item.get("options"))
    prompts.append(tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=True, thinking_budget=2048
    ))

outputs   = llm.generate(prompts, sp)
responses = [o.outputs[0].text.strip() for o in outputs]
responses = batch_retry(test_data, responses)

df = pd.DataFrame([{"id": item["id"], "response": resp}
                   for item, resp in zip(data_priv, responses)])
df.to_csv("submission_budget2048.csv", index=False, quoting=csv.QUOTE_ALL)
print(f"Saved: submission_budget2048.csv ({len(df)} rows)")


# ### Option B: n=3 Voting, budget=2048 (Expected: **~72%**)

# In[ ]:


# ── Private Set Submission: n=3 Self-Consistency Voting ──
import csv
import pandas as pd

data_priv = [json.loads(line) for line in open("data/private.jsonl")]
test_data  = data_priv
TEST_N     = len(data_priv)
print(f"Private set: {TEST_N} questions, n=3 voting (~8 hours)")

sp_vote = SamplingParams(n=3, max_tokens=7000, temperature=0.7, top_p=0.95, top_k=20)
prompts = []
for item in test_data:
    system, user = build_prompt(item["question"], item.get("options"))
    prompts.append(tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=True, thinking_budget=2048
    ))

outputs = llm.generate(prompts, sp_vote)

final_responses = []
for out in outputs:
    candidates = [o.text.strip() for o in out.outputs]
    answers    = [extract_boxed_v2(c) for c in candidates]
    valid      = [(a, c) for a, c in zip(answers, candidates) if a]
    if not valid:
        final_responses.append(candidates[0])
    else:
        best_ans  = Counter(a for a, _ in valid).most_common(1)[0][0]
        best_resp = next(c for a, c in valid if a == best_ans)
        final_responses.append(best_resp)

final_responses = batch_retry(test_data, final_responses)

df = pd.DataFrame([{"id": item["id"], "response": resp}
                   for item, resp in zip(data_priv, final_responses)])
df.to_csv("submission_n3_voting.csv", index=False, quoting=csv.QUOTE_ALL)
print(f"Saved: submission_n3_voting.csv ({len(df)} rows)")


# In[2]:


get_ipython().system('jupyter nbconvert --to script competition.ipynb')


# In[ ]:




