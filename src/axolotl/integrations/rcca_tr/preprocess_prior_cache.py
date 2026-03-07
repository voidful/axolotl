# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Offline prior cache preprocessing for RCCA-TR (A+ variant).

Runs the frozen base model once over the entire dataset and caches per-token:
  - prior_target_logp: log p_0(y_t)
  - prior_top1_logp:   log p_0(ŷ^(1)_t)
  - prior_margin:       log p_0(ŷ^(1)) - log p_0(y_t)

This eliminates the need for a live frozen model during training.

Usage:
    python -m axolotl.integrations.rcca_tr.preprocess_prior_cache \
        --base_model Qwen/Qwen3.5-9B \
        --dataset_path voidful/gemini-3.1-opus-4.6-reasoning-merged \
        --output_path ./prior_cache \
        --batch_size 4 \
        --max_length 8000
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_prior_logits_for_batch(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute prior cache values for a batch.

    Args:
        model: Frozen base model.
        input_ids: (B, T) input token IDs.
        attention_mask: (B, T) attention mask.
        labels: (B, T) ground-truth labels. -100 = ignore.

    Returns:
        Dict with prior_target_logp, prior_top1_logp, prior_margin, each (B, T).
    """
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)

    safe_labels = labels.clamp(min=0)
    valid_mask = labels != -100

    # log p_0(y_t)
    prior_target_logp = log_probs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)
    prior_target_logp = prior_target_logp * valid_mask.float()

    # log p_0(ŷ^(1))
    prior_top1_logp = log_probs.max(dim=-1).values  # (B, T)
    prior_top1_logp = prior_top1_logp * valid_mask.float()

    # margin
    prior_margin = prior_top1_logp - prior_target_logp  # (B, T)
    prior_margin = prior_margin * valid_mask.float()

    return {
        "prior_target_logp": prior_target_logp.cpu(),
        "prior_top1_logp": prior_top1_logp.cpu(),
        "prior_margin": prior_margin.cpu(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute frozen model prior cache for RCCA-TR"
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=8000)
    parser.add_argument("--chat_template", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path, split=args.dataset_split)

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(dataset)} samples...")

    all_target_logp = []
    all_top1_logp = []
    all_margin = []

    # Process in batches
    for start_idx in tqdm(range(0, len(dataset), args.batch_size)):
        end_idx = min(start_idx + args.batch_size, len(dataset))
        batch_samples = dataset[start_idx:end_idx]

        # Tokenize
        if "messages" in batch_samples:
            # Chat format — apply chat template
            texts = []
            for i in range(len(batch_samples["messages"])):
                msgs = batch_samples["messages"][i]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
        elif "text" in batch_samples:
            texts = batch_samples["text"]
        else:
            raise ValueError(
                f"Dataset must have 'messages' or 'text' column. "
                f"Found: {list(batch_samples.keys())}"
            )

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        # Labels = input_ids shifted (for causal LM)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        cache = compute_prior_logits_for_batch(
            model, input_ids, attention_mask, labels
        )

        # Store per-sample (variable length)
        for i in range(input_ids.size(0)):
            seq_len = attention_mask[i].sum().item()
            all_target_logp.append(cache["prior_target_logp"][i, :seq_len])
            all_top1_logp.append(cache["prior_top1_logp"][i, :seq_len])
            all_margin.append(cache["prior_margin"][i, :seq_len])

    # Save
    cache_data = {
        "prior_target_logp": all_target_logp,
        "prior_top1_logp": all_top1_logp,
        "prior_margin": all_margin,
    }
    cache_path = output_dir / "prior_cache.pt"
    torch.save(cache_data, cache_path)
    print(f"Prior cache saved to {cache_path} ({len(all_target_logp)} samples)")


if __name__ == "__main__":
    main()
