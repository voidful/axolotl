#!/usr/bin/env python3
"""
MATH-500 Pass@k Evaluation via vLLM.

Evaluates test-time scaling by sampling k responses per problem and computing
Pass@1, Pass@16, Pass@64. Uses vLLM for efficient batched generation.

This is the core Battle B evaluation script for the Drift-Trust paper.

Usage:
    python eval_math_passk.py \
        --model_path ./outputs/paper/drift-math-4b/checkpoint-final \
        --k 64 \
        --temperature 0.7 \
        --output_dir ./results/paper \
        --run_name drift_math_4b
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np


def load_math500():
    """Load MATH-500 benchmark."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return ds


def extract_answer(text: str) -> str:
    """Extract the final boxed answer from a MATH-style response."""
    # Look for \boxed{...}
    boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if boxed:
        return boxed[-1].strip()

    # Look for "The answer is ..." patterns
    answer_patterns = [
        r'[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)',
        r'[Aa]nswer[:\s]*(.+?)(?:\.|$)',
        r'= *(.+?)$',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1).strip()

    # Return last line as fallback
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove LaTeX formatting
    answer = answer.replace("\\$", "").replace("$", "")
    answer = answer.replace("\\%", "%")
    answer = answer.replace("\\text{", "").replace("}", "")
    answer = answer.replace("\\mathrm{", "").replace("\\mathbf{", "")
    answer = answer.replace("\\frac{", "").replace("\\dfrac{", "")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\", "")
    answer = answer.strip()

    # Try to evaluate simple fractions
    frac_match = re.match(r'^(\d+)/(\d+)$', answer)
    if frac_match:
        try:
            return str(int(frac_match.group(1)) / int(frac_match.group(2)))
        except (ValueError, ZeroDivisionError):
            pass

    return answer


def is_correct(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        return abs(pred_num - gold_num) < 1e-6
    except (ValueError, TypeError):
        pass

    return pred_norm.lower() == gold_norm.lower()


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute Pass@k.

    n: total number of samples
    c: number of correct samples
    k: number of samples to consider
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(math.comb(n - c, k)) / float(math.comb(n, k))


def generate_with_vllm(model_path: str, prompts: list[str], k: int,
                        temperature: float = 0.7, max_tokens: int = 2048,
                        tensor_parallel: int = 1):
    """Generate k responses per prompt using vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        n=k,
    )

    print(f"Generating {k} samples per prompt for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        samples = [o.text for o in output.outputs]
        results.append(samples)

    return results


def main():
    parser = argparse.ArgumentParser(description="MATH-500 Pass@k evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--k", type=int, default=64,
                        help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./results/paper")
    parser.add_argument("--run_name", type=str, default="model")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load MATH-500
    print("Loading MATH-500...")
    ds = load_math500()
    print(f"Loaded {len(ds)} problems")

    # Prepare prompts
    prompts = []
    gold_answers = []
    for sample in ds:
        problem = sample["problem"]
        prompt = f"Solve the following math problem step by step. Show your work and put your final answer in \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"
        prompts.append(prompt)
        gold_answers.append(sample["answer"])

    # Generate
    all_responses = generate_with_vllm(
        model_path=args.model_path,
        prompts=prompts,
        k=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel=args.tensor_parallel,
    )

    # Evaluate
    k_values = [1, 2, 4, 8, 16, 32, 64]
    k_values = [k for k in k_values if k <= args.k]

    per_problem = []
    for i, (responses, gold) in enumerate(zip(all_responses, gold_answers)):
        correct_count = 0
        per_sample = []
        for resp in responses:
            pred = extract_answer(resp)
            c = is_correct(pred, gold)
            correct_count += int(c)
            per_sample.append({
                "response": resp[:500],  # Truncate for storage
                "predicted": pred,
                "correct": c,
            })

        per_problem.append({
            "problem_idx": i,
            "gold_answer": gold,
            "n_samples": len(responses),
            "n_correct": correct_count,
            "samples": per_sample,
        })

    # Compute Pass@k for each k
    results = {}
    for k in k_values:
        pass_k_values = []
        for prob in per_problem:
            n = prob["n_samples"]
            c = prob["n_correct"]
            pk = pass_at_k(n, c, k)
            pass_k_values.append(pk)
        results[f"pass@{k}"] = float(np.mean(pass_k_values))

    # Print results
    print("\n" + "=" * 50)
    print(f"MATH-500 Pass@k Results — {args.run_name}")
    print("=" * 50)
    for k_label, score in results.items():
        print(f"  {k_label}: {score:.4f} ({score*100:.2f}%)")
    print("=" * 50)

    # Save results
    output_file = Path(args.output_dir) / f"math500_passk_{args.run_name}.json"
    with open(output_file, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "run_name": args.run_name,
            "k": args.k,
            "temperature": args.temperature,
            "pass_at_k": results,
            "per_problem_summary": [
                {"idx": p["problem_idx"], "n": p["n_samples"], "c": p["n_correct"]}
                for p in per_problem
            ],
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Save detailed results (large file)
    detail_file = Path(args.output_dir) / f"math500_passk_{args.run_name}_detail.json"
    with open(detail_file, "w") as f:
        json.dump(per_problem, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {detail_file}")


if __name__ == "__main__":
    main()
