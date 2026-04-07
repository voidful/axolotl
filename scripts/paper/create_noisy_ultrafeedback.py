#!/usr/bin/env python3
"""
Create a 10k UltraFeedback dataset with 25% realistic synthetic noise.

Noise types injected into the 25% noisy portion:
  1. Hallucinated facts  — Swap key entities/numbers in the response
  2. Refusal injection   — Replace response with refusal template
  3. Logic chain break   — Truncate or shuffle reasoning steps
  4. Style degradation   — Inject repetitive filler or off-topic tangents

Output: JSONL in Axolotl chat_template format with a `is_noisy` metadata field.

Usage:
    python create_noisy_ultrafeedback.py \
        --output_dir ./data/paper \
        --num_samples 10000 \
        --noise_ratio 0.25 \
        --seed 42
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Noise injection functions
# ---------------------------------------------------------------------------

REFUSAL_TEMPLATES = [
    "I'm sorry, but I cannot assist with that request.",
    "I appreciate your question, but I'm unable to provide that information.",
    "Thank you for your inquiry. However, I must decline to answer this question as it falls outside my guidelines.",
    "I understand your curiosity, but I'm not able to help with this particular topic.",
    "I apologize, but providing this information would not be appropriate.",
]

FILLER_PHRASES = [
    "As we all know, this is a very important topic. ",
    "It's worth noting that there are many perspectives on this. ",
    "Generally speaking, and this is well-established, ",
    "From a holistic perspective, one might argue that ",
    "Interestingly enough, and this cannot be overstated, ",
]


def inject_hallucination(text: str, rng: random.Random) -> str:
    """Swap numbers and key nouns to create plausible-looking but wrong facts."""
    # Swap numbers
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    if len(numbers) >= 2:
        a, b = rng.sample(numbers, 2)
        text = text.replace(a, "__SWAP__", 1)
        text = text.replace(b, a, 1)
        text = text.replace("__SWAP__", b, 1)

    # Swap capitalized words (likely named entities)
    caps = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    if len(caps) >= 2:
        unique_caps = list(set(caps))
        if len(unique_caps) >= 2:
            a, b = rng.sample(unique_caps, 2)
            text = text.replace(a, "__SWAP__")
            text = text.replace(b, a)
            text = text.replace("__SWAP__", b)

    return text


def inject_refusal(text: str, rng: random.Random) -> str:
    """Replace the response with a refusal template."""
    return rng.choice(REFUSAL_TEMPLATES)


def inject_logic_break(text: str, rng: random.Random) -> str:
    """Break the logical chain by truncating or shuffling sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 2:
        return text

    strategy = rng.choice(["truncate", "shuffle", "repeat"])
    if strategy == "truncate":
        # Cut off at 40-60% of the response
        cut = rng.randint(len(sentences) * 2 // 5, len(sentences) * 3 // 5)
        return " ".join(sentences[:cut])
    elif strategy == "shuffle":
        # Keep first and last, shuffle middle
        middle = sentences[1:-1]
        rng.shuffle(middle)
        return " ".join([sentences[0]] + middle + [sentences[-1]])
    else:
        # Repeat a random sentence 2-3 times
        idx = rng.randint(0, len(sentences) - 1)
        repeat_count = rng.randint(2, 3)
        sentences.insert(idx + 1, " ".join([sentences[idx]] * repeat_count))
        return " ".join(sentences)


def inject_style_degradation(text: str, rng: random.Random) -> str:
    """Add verbose filler that degrades signal-to-noise ratio."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return rng.choice(FILLER_PHRASES) + text

    # Insert 2-4 filler phrases at random positions
    num_fillers = rng.randint(2, 4)
    for _ in range(num_fillers):
        pos = rng.randint(0, len(sentences))
        filler = rng.choice(FILLER_PHRASES)
        sentences.insert(pos, filler)

    return " ".join(sentences)


NOISE_FUNCTIONS = [
    (inject_hallucination, "hallucination"),
    (inject_refusal, "refusal"),
    (inject_logic_break, "logic_break"),
    (inject_style_degradation, "style_degradation"),
]


# ---------------------------------------------------------------------------
# Data loading and processing
# ---------------------------------------------------------------------------

def load_ultrafeedback(num_samples: int, seed: int):
    """Load and sample from UltraFeedback dataset."""
    print("Loading UltraFeedback from HuggingFace...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_sft")

    # Shuffle and sample
    ds = ds.shuffle(seed=seed)
    if len(ds) > num_samples:
        ds = ds.select(range(num_samples))

    print(f"Loaded {len(ds)} samples from UltraFeedback")
    return ds


def extract_conversation(sample) -> list[dict]:
    """Extract messages from UltraFeedback format into chat template format."""
    messages = []
    if "chosen" in sample and sample["chosen"]:
        for msg in sample["chosen"]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
    return messages


def apply_noise(messages: list[dict], rng: random.Random) -> tuple[list[dict], str]:
    """Apply a random noise function to the assistant's response."""
    noise_fn, noise_type = rng.choice(NOISE_FUNCTIONS)

    noisy_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            noisy_content = noise_fn(msg["content"], rng)
            noisy_messages.append({"role": "assistant", "content": noisy_content})
        else:
            noisy_messages.append(msg.copy())

    return noisy_messages, noise_type


def main():
    parser = argparse.ArgumentParser(description="Create noisy UltraFeedback dataset")
    parser.add_argument("--output_dir", type=str, default="./data/paper",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Total number of samples")
    parser.add_argument("--noise_ratio", type=float, default=0.25,
                        help="Fraction of samples to inject noise into")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    ds = load_ultrafeedback(args.num_samples, args.seed)

    # Determine which samples get noise
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    num_noisy = int(len(indices) * args.noise_ratio)
    noisy_indices = set(indices[:num_noisy])

    print(f"Clean samples: {len(indices) - num_noisy}, Noisy samples: {num_noisy}")

    # Process
    output_path = Path(args.output_dir) / f"ultrafeedback_noisy_{int(args.noise_ratio*100)}pct_{args.num_samples // 1000}k.jsonl"
    noise_stats = {"hallucination": 0, "refusal": 0, "logic_break": 0, "style_degradation": 0}

    with open(output_path, "w") as f:
        for i, sample in enumerate(ds):
            messages = extract_conversation(sample)
            if not messages:
                continue

            is_noisy = i in noisy_indices
            noise_type = None

            if is_noisy:
                messages, noise_type = apply_noise(messages, rng)
                noise_stats[noise_type] += 1

            record = {
                "conversations": messages,
                "is_noisy": is_noisy,
                "noise_type": noise_type,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDataset saved to {output_path}")
    print(f"Noise distribution: {json.dumps(noise_stats, indent=2)}")
    print(f"Total: {len(ds)} samples ({num_noisy} noisy, {len(ds) - num_noisy} clean)")


if __name__ == "__main__":
    main()
