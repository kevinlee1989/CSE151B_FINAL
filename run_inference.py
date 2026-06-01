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

    Replicates the exact pipeline that achieved 0.671 on the private leaderboard.
    Uses Qwen3-4B-Thinking-2507 with thinking_budget=2048, max_tokens=7000.

    Args:
        data_path: Path to private.jsonl
        output_path: Output CSV file path
        model_id: HuggingFace model ID
        thinking_budget: Max tokens for the thinking phase
        max_tokens: Max total tokens per response
        temperature: Sampling temperature
        gpu_memory_utilization: Fraction of GPU VRAM to use
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # ── Load Tokenizer & Model ──
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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
    print("Model loaded.")

    # ── Load Dataset ──
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = [json.loads(line) for line in open(data_path)]
    print(f"Loaded {len(data)} questions from {data_path}")

    # ── System Prompts (exact prompts used for 0.671 submission) ──
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

    def build_prompt(question, options):
        if options:
            labels = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
        return SYSTEM_PROMPT_MATH, question

    # ── Answer Extraction ──
    def extract_boxed(text):
        """Extract last \\boxed{} answer, stripping <think> tags first."""
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

    # ── Build Prompts ──
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            thinking_budget=thinking_budget,
        ))

    # ── Primary Generation Pass ──
    print(f"Generating (budget={thinking_budget}, max_tokens={max_tokens}, temp={temperature})...")
    sp = SamplingParams(
        n=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=20,
    )
    outputs = llm.generate(prompts, sp)
    responses = [o.outputs[0].text.strip() for o in outputs]

    no_box = sum(1 for r in responses if not extract_boxed(r))
    print(f"Generation complete. No-boxed: {no_box}/{len(data)}")

    # ── Batch Retry for No-Boxed Responses ──
    retry_indices = [i for i, r in enumerate(responses) if not extract_boxed(r)]
    if retry_indices:
        print(f"Retrying {len(retry_indices)} no-boxed responses...")
        retry_prompts = []
        for i in retry_indices:
            item = data[i]
            system, user = build_prompt(item["question"], item.get("options"))
            base = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            retry_prompts.append(base + "\\boxed{")

        retry_sp  = SamplingParams(max_tokens=100, temperature=0.1, stop=["}"])
        retry_out = llm.generate(retry_prompts, retry_sp)
        for j, out in enumerate(retry_out):
            raw = out.outputs[0].text.strip()
            responses[retry_indices[j]] = f"\\boxed{{{raw}}}"

    final_no_box = sum(1 for r in responses if not extract_boxed(r))
    print(f"After retry — no-boxed: {final_no_box}/{len(data)}")

    # ── Save CSV ──
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame([
        {"id": item["id"], "response": resp}
        for item, resp in zip(data, responses)
    ])
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved: {output_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    run_inference(
        data_path="data/private.jsonl",
        output_path="submission.csv",
    )
