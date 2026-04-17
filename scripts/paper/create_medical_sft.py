#!/usr/bin/env python3
"""
Create a medical SFT dataset from medical_meadow_medical_flashcards.

Converts to Axolotl chat_template format for training.

Usage:
    python create_medical_sft.py \
        --output_dir ./data/paper \
        --num_samples 10000 \
        --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Create medical SFT dataset")
    parser.add_argument("--output_dir", type=str, default="./data/paper")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading medical_meadow_medical_flashcards from HuggingFace...")
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    print(f"Loaded {len(ds)} samples")

    # Shuffle and sample
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if len(indices) > args.num_samples:
        indices = indices[:args.num_samples]

    output_path = Path(args.output_dir) / f"medical_flashcards_{args.num_samples // 1000}k.jsonl"

    count = 0
    with open(output_path, "w") as f:
        for i in indices:
            sample = ds[i]

            # medical_meadow format: input (question) + output (answer)
            question = sample.get("input", "").strip()
            answer = sample.get("output", "").strip()

            if not question or not answer:
                continue

            # Build system prompt for medical context
            messages = [
                {
                    "role": "system",
                    "content": "You are a knowledgeable medical assistant. Provide accurate, helpful medical information based on established medical knowledge."
                },
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

            record = {"conversations": messages}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"\nDataset saved to {output_path}")
    print(f"Total samples: {count}")


if __name__ == "__main__":
    main()
