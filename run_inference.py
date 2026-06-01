import json
import re
import csv
import os
import pandas as pd

def run_inference(
    data_path: str = "data/private.jsonl",
    output_path: str = "submission.csv",
    model_id: str = "Qwen/Qwen3-4B-Thinking-2507",
    thinking_budget: int = 2048,
    max_tokens: int = 7000,
    temperature: float = 0.6,
    gpu_memory_utilization: float = 0.55,
):
    """
    End-to-End Inference Pipeline for CSE 151B Math Reasoning Competition.

    This function automatically handles model loading, prompt formatting, 
    batched generation via vLLM with thinking logic enabled, post-processing 
    retry structures for failed formats, and final CSV formatting.

    Final Leaderboard Score: 0.671
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # ── Load Tokenizer & Model Engine ──
    print(f"[-] Initializing tokenizer and model from Hub: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Instantiate the engine using INT8 BitsAndBytes quantization via vLLM
    llm = LLM(
        model=model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=8192,
        trust_remote_code=True,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        enforce_eager=True,
    )
    print("[+] Model stack loaded successfully.")

    # ── Dataset Verification & Loading ──
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Target evaluation file path does not exist: {data_path}")
        
    data = [json.loads(line) for line in open(data_path)]
    print(f"[–] Successfully parsed {len(data)} problems from {data_path}")

    # ── Strict Structural Prompt Configurations ──
    SYSTEM_MATH = (
        "You are an expert mathematician. Solve the problem step-by-step. "
        "Put your final answer inside \\boxed{}. "
        "If the problem has multiple sub-answers, separate them by commas "
        "inside a single \\boxed{}, e.g. \\boxed{3, 7}. "
        "IMPORTANT: Write \\boxed{answer} immediately when done. "
        "Do not write anything after \\boxed{}."
    )
    SYSTEM_MCQ = (
        "You are an expert mathematician. Select the single best answer. "
        "Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}. "
        "IMPORTANT: Brief reasoning, then immediately \\boxed{letter}. Nothing after."
    )

    def build_prompt(question, options):
        if options:
            labels = [chr(65 + i) for i in range(len(options))]
            opts = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
            return SYSTEM_MCQ, f"{question}\n\nOptions:\n{opts}"
        return SYSTEM_MATH, question

    def extract_boxed(text):
        """Extracts the final valid \\boxed{} sequence after scrubbing inner <think> blocks."""
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        for t in [clean, text]:
            key = r"\boxed{"
            start = t.rfind(key)
            if start == -1:
                continue
            i, depth, answer = start + len(key), 1, ""
            while i < len(t) and depth > 0:
                ch = t[i]
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

    # ── Build Chat-Template Sequences ──
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True, # Critical parameter mapped to the 67.1% setup
            thinking_budget=thinking_budget,
        ))

    # ── Primary Pipeline Execution (Pass 1) ──
    print(f"[-] Executing primary inference pass (budget={thinking_budget}, max_tokens={max_tokens}, temp={temperature})...")
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=20,
    )
    outputs = llm.generate(prompts, sp)
    responses = [o.outputs[0].text.strip() for o in outputs]

    no_box = sum(1 for r in responses if not extract_boxed(r))
    print(f"[+] Primary pass complete. Formatting failures (missing \\boxed{{}}): {no_box}/{len(data)}")

    # ── Post-Processing Pipeline: Forced \boxed{ Prefix Retry ──
    retry_indices = [i for i, r in enumerate(responses) if not extract_boxed(r)]
    if retry_indices:
        print(f"[-] Routing {len(retry_indices)} malformed generations to the post-processing retry loop.")
        retry_prompts = []
        for i in retry_indices:
            item = data[i]
            system, user = build_prompt(item["question"], item.get("options"))
            base = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False, # Disable thinking during retry to force immediate structural completion
            )
            retry_prompts.append(base + "\\boxed{")

        retry_sp  = SamplingParams(max_tokens=100, temperature=0.1, stop=["}"])
        retry_out = llm.generate(retry_prompts, retry_sp)
        for j, out in enumerate(retry_out):
            raw = out.outputs[0].text.strip()
            responses[retry_indices[j]] = f"\\boxed{{{raw}}}"

    final_no_box = sum(1 for r in responses if not extract_boxed(r))
    print(f"[+] Post-processing sequence finalized. Structural anomalies remaining: {final_no_box}/{len(data)}")

    # ── CSV Serialization ──
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    df = pd.DataFrame([
        {"id": item["id"], "response": resp}
        for item, resp in zip(data, responses)
    ])
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"[+] Final submission file successfully stored to: {output_path} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    # Standard submission run when executing script directly via CLI
    run_inference(
        data_path="data/private.jsonl",
        output_path="submission.csv",
    )
