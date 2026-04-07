#!/usr/bin/env python3
"""
Sample 10k examples from NuminaMath-CoT for long chain-of-thought training.

This dataset is used for Battle B: Test-Time Scaling.
We want to test whether Drift-Trust preserves sampling diversity (Pass@k)
better than standard CE which causes mode collapse.

Usage:
    python sample_numina_math.py \
        --output_dir ./data/paper \
        --num_samples 10000 \
        --seed 42
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Sample NuminaMath-CoT")
    parser.add_argument("--output_dir", type=str, default="./data/paper")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--min_solution_len", type=int, default=200,
                        help="Minimum solution length in chars (filter short CoT)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading AI-MO/NuminaMath-CoT from HuggingFace...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    print(f"Full dataset: {len(ds)} samples")

    # Filter for quality: keep samples with substantial chain-of-thought
    def is_quality_cot(sample):
        solution = sample.get("solution", "")
        return len(solution) >= args.min_solution_len

    ds_filtered = ds.filter(is_quality_cot)
    print(f"After quality filter (>= {args.min_solution_len} chars): {len(ds_filtered)} samples")

    # Shuffle and sample
    ds_filtered = ds_filtered.shuffle(seed=args.seed)
    if len(ds_filtered) > args.num_samples:
        ds_filtered = ds_filtered.select(range(args.num_samples))

    # Convert to chat template format
    output_path = Path(args.output_dir) / f"numina_math_cot_{args.num_samples // 1000}k.jsonl"

    stats = {"total": 0, "avg_solution_len": 0}
    total_len = 0

    with open(output_path, "w") as f:
        for sample in ds_filtered:
            problem = sample.get("problem", "")
            solution = sample.get("solution", "")

            conversations = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]

            record = {
                "conversations": conversations,
                "source": sample.get("source", "unknown"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats["total"] += 1
            total_len += len(solution)

    stats["avg_solution_len"] = total_len / max(stats["total"], 1)

    print(f"\nDataset saved to {output_path}")
    print(f"Total samples: {stats['total']}")
    print(f"Average solution length: {stats['avg_solution_len']:.0f} chars")


if __name__ == "__main__":
    main()
