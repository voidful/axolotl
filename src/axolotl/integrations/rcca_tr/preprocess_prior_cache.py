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

    # Apply standard causal shift: logits[t] predicts token at position t+1
    shift_log_probs = log_probs[:, :-1, :]  # (B, T-1, V)
    shift_labels = input_ids[:, 1:]          # (B, T-1) — next tokens
    shift_safe_labels = shift_labels.clamp(min=0)

    # Valid mask: both source and target positions must be valid
    shift_valid = (labels[:, 1:] != -100) & attention_mask[:, :-1].bool()

    # log p_0(y_{t+1} | context through t)
    shifted_target_logp = shift_log_probs.gather(
        dim=-1, index=shift_safe_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)
    shifted_target_logp = shifted_target_logp * shift_valid.float()

    # log p_0(top-1 token) at each shifted position
    shifted_top1_logp = shift_log_probs.max(dim=-1).values  # (B, T-1)
    shifted_top1_logp = shifted_top1_logp * shift_valid.float()

    # margin
    shifted_margin = (shifted_top1_logp - shifted_target_logp) * shift_valid.float()

    # Pad back to length T: position 0 = 0.0 (no valid prior for first token)
    # After this: prior_target_logp[t] = log p_0(input_ids[t] | context through t-1)
    # for t >= 1, and 0.0 for t = 0.
    prior_target_logp = F.pad(shifted_target_logp, (1, 0), value=0.0)  # (B, T)
    prior_top1_logp = F.pad(shifted_top1_logp, (1, 0), value=0.0)      # (B, T)
    prior_margin = F.pad(shifted_margin, (1, 0), value=0.0)             # (B, T)

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
    parser.add_argument("--merge_dir", type=str, default=None, help="If provided, merges all rank cache chunks in dir to output_path and exits.")
    parser.add_argument("--chat_template", type=str, default=None)
    parser.add_argument(
        "--axolotl_config", type=str, default=None,
        help="Path to axolotl YAML config to extract chat_template_jinja from."
    )
    parser.add_argument(
        "--chat_template_jinja", type=str, default=None,
        help="Jinja2 template string for chat formatting (overrides tokenizer default)."
    )
    args = parser.parse_args()

    if args.merge_dir:
        print(f"Merging cache chunks from {args.merge_dir} into {args.output_path}...")
        import glob
        chunks = glob.glob(os.path.join(args.merge_dir, "prior_cache_rank_*.pt"))
        all_tgt, all_top1, all_mrgn = [], [], []
        # sort by rank numerically carefully
        chunks = sorted(chunks, key=lambda x: int(os.path.basename(x).split('_rank_')[-1].split('.pt')[0]))
        for f in tqdm(chunks, desc="Merging"):
            c = torch.load(f, weights_only=False)
            all_tgt.extend(c['prior_target_logp'])
            all_top1.extend(c['prior_top1_logp'])
            all_mrgn.extend(c['prior_margin'])
        torch.save({
            "prior_target_logp": all_tgt,
            "prior_top1_logp": all_top1,
            "prior_margin": all_mrgn,
        }, args.output_path)
        print(f"Merged {len(all_tgt)} samples to {args.output_path}.")
        return

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

    print(f"[Rank {rank}/{world_size}] Loading model {args.base_model} on cuda:{local_rank}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Apply custom chat template if specified
    custom_template = None
    if args.axolotl_config:
        import yaml
        with open(args.axolotl_config) as f:
            ax_cfg = yaml.safe_load(f)
        custom_template = ax_cfg.get("chat_template_jinja")
        if custom_template and rank == 0:
            print(f"Using chat_template_jinja from axolotl config: {args.axolotl_config}")
    elif args.chat_template_jinja:
        custom_template = args.chat_template_jinja

    if custom_template:
        tokenizer.chat_template = custom_template

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()

    output_dir = Path(args.output_path).parent if Path(args.output_path).suffix else Path(args.output_path)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Rank {rank}] Dataset total length: {len(dataset)}")
    
    if world_size > 1:
        chunk_size = (len(dataset) + world_size - 1) // world_size
        start_sample = rank * chunk_size
        end_sample = min((rank + 1) * chunk_size, len(dataset))
        dataset = dataset.select(range(start_sample, end_sample))
    
    print(f"[Rank {rank}] Processing {len(dataset)} samples...")

    all_target_logp = []
    all_top1_logp = []
    all_margin = []

    # Process in batches
    for start_idx in tqdm(range(0, len(dataset), args.batch_size)):
        end_idx = min(start_idx + args.batch_size, len(dataset))
        batch_samples = dataset[start_idx:end_idx]

        messages_col = None
        for col in ["messages", "conversations"]:
            if col in batch_samples:
                messages_col = col
                break

        # Tokenize
        if messages_col:
            # Chat format — apply chat template
            texts = []
            for i in range(len(batch_samples[messages_col])):
                msgs = batch_samples[messages_col][i]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
        elif "text" in batch_samples:
            texts = batch_samples["text"]
        else:
            raise ValueError(
                f"Dataset must have 'messages', 'conversations', or 'text' column. "
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
    if world_size > 1:
        cache_path = output_dir / f"prior_cache_rank_{rank}.pt"
    else:
        cache_path = Path(args.output_path)
        if cache_path.is_dir():
            cache_path = cache_path / "prior_cache.pt"
            
    torch.save(cache_data, cache_path)
    print(f"[Rank {rank}] Prior cache saved to {cache_path} ({len(all_target_logp)} samples)")


if __name__ == "__main__":
    main()
