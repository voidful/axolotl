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

import gc
import glob
import json
import os
import struct
import threading
import time

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import safetensors
import transformers.modeling_utils


# ---------------------------------------------------------------------------
# Lustre/SLURM mmap Bypass — SafeOpenStreamer
# ---------------------------------------------------------------------------
# Lustre rejects 30-node concurrent mmap on the same 5 GB safetensor files.
# Allocating a contiguous 5 GB numpy buffer also fails due to strict cluster
# Virtual Address Space (VMAS) limits per CGroup.
#
# Solution: stream tensor-by-tensor via readinto() into a numpy buffer sized
# to exactly ONE tensor (max ~2.5 GB for lm_head).  Since DeepSpeed ZeRO-3
# is removed, these CPU buffers move straight into VRAM via Accelerate's
# device_map="auto" and get properly GC'd.  No memory blobs, no mmap.
# ---------------------------------------------------------------------------

_global_stream_lock = threading.Lock()


class SafeOpenStreamer:
    """Drop-in replacement for ``safetensors.safe_open`` that avoids mmap."""

    def __init__(self, filename, framework="pt", device="cpu"):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.filename = filename
        self.device = device
        with open(self.filename, "rb") as f:
            header_size_bytes = f.read(8)
            self.header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_json = f.read(self.header_size).decode("utf-8")
            self.metadata = json.loads(header_json)

        self.data_offset = 8 + self.header_size
        self.tensor_keys = [k for k in self.metadata.keys() if k != "__metadata__"]

    def keys(self):
        return self.tensor_keys

    def get_tensor(self, key):
        info = self.metadata[key]
        dtype_str = info["dtype"]
        shape = info["shape"]
        offsets = info["data_offsets"]

        dtype_map = {
            "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
            "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
            "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
        }
        dt = dtype_map[dtype_str]
        byte_size = offsets[1] - offsets[0]

        with _global_stream_lock:
            t_bytes_np = np.empty(byte_size, dtype=np.uint8)
            m_view = memoryview(t_bytes_np)

            with open(self.filename, "rb") as f:
                f.seek(self.data_offset + offsets[0])
                bytes_read = 0
                while bytes_read < byte_size:
                    n = f.readinto(m_view[bytes_read:])
                    if n == 0:
                        raise EOFError(f"Unexpected EOF while streaming tensor '{key}'")
                    bytes_read += n

            t_bytes = torch.from_numpy(t_bytes_np)
            return t_bytes.view(dtype=dt).reshape(shape)

    def get_slice(self, key):
        class _SliceProxy:
            def __init__(self, parent, k):
                self.parent = parent
                self.key = k
            def __getitem__(self, idx):
                return self.parent.get_tensor(self.key)[idx]
        return _SliceProxy(self, key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metadata = None
        self.tensor_keys = None


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
    B, T = input_ids.shape

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        shifted_target_logp = torch.zeros(B, T - 1, device=input_ids.device)
        shifted_top1_logp = torch.zeros(B, T - 1, device=input_ids.device)

        # Sub-batch to stay under PyTorch's int32 indexing limit (2.14 B elements).
        mini_batch_size = 4
        chunk_size = 1024  # [1, 1024, 151936] ≈ 600 MB

        for mb_start in range(0, B, mini_batch_size):
            mb_end = min(B, mb_start + mini_batch_size)
            mb_input_ids = input_ids[mb_start:mb_end]
            mb_attention_mask = attention_mask[mb_start:mb_end]

            # Forward through the transformer body (without lm_head)
            base_model = (
                getattr(model, model.base_model_prefix)
                if hasattr(model, "base_model_prefix")
                else model.model
            )
            base_outputs = base_model(input_ids=mb_input_ids, attention_mask=mb_attention_mask)
            hidden_states = (
                base_outputs[0]
                if isinstance(base_outputs, tuple)
                else base_outputs.last_hidden_state
            )

            # Chunk the sequence through lm_head to cap VRAM usage
            for local_b in range(mb_end - mb_start):
                global_b = mb_start + local_b
                for i in range(0, T - 1, chunk_size):
                    i_end = min(i + chunk_size, T - 1)

                    h_chunk = hidden_states[local_b : local_b + 1, i:i_end, :]  # [1, C, D]
                    logits_chunk = model.lm_head(h_chunk)                         # [1, C, V]
                    log_probs_chunk = F.log_softmax(logits_chunk.float(), dim=-1) # [1, C, V]

                    # Align labels/masks to whichever GPU the pipeline placed the output on
                    chunk_dev = log_probs_chunk.device
                    labels_chunk = labels[global_b : global_b + 1, i + 1 : i_end + 1].to(chunk_dev)
                    safe_labels_chunk = labels_chunk.clamp(min=0)
                    valid_chunk = (
                        (labels_chunk != -100)
                        & attention_mask[global_b : global_b + 1, i:i_end].to(chunk_dev).bool()
                    )

                    target_lp = log_probs_chunk.gather(
                        dim=-1, index=safe_labels_chunk.unsqueeze(-1)
                    ).squeeze(-1)
                    top1_lp = log_probs_chunk.max(dim=-1).values

                    shifted_target_logp[global_b : global_b + 1, i:i_end] = (
                        (target_lp * valid_chunk.float()).to(shifted_target_logp.device)
                    )
                    shifted_top1_logp[global_b : global_b + 1, i:i_end] = (
                        (top1_lp * valid_chunk.float()).to(shifted_top1_logp.device)
                    )

                    del h_chunk, logits_chunk, log_probs_chunk
                    del labels_chunk, safe_labels_chunk, valid_chunk, target_lp, top1_lp

            del base_outputs, hidden_states
            torch.cuda.empty_cache()

        shifted_margin = shifted_top1_logp - shifted_target_logp

    # Pad position 0 with 0.0 (no valid prior for the first token)
    prior_target_logp = F.pad(shifted_target_logp, (1, 0), value=0.0)
    prior_top1_logp = F.pad(shifted_top1_logp, (1, 0), value=0.0)
    prior_margin = F.pad(shifted_margin, (1, 0), value=0.0)

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
    parser.add_argument(
        "--merge_dir", type=str, default=None,
        help="If provided, merges all rank cache chunks in dir to output_path and exits.",
    )
    parser.add_argument("--chat_template", type=str, default=None)
    parser.add_argument(
        "--axolotl_config", type=str, default=None,
        help="Path to axolotl YAML config to extract chat_template_jinja from.",
    )
    parser.add_argument(
        "--chat_template_jinja", type=str, default=None,
        help="Jinja2 template string for chat formatting (overrides tokenizer default).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Fast-path: merge previously generated rank chunks and exit
    # ------------------------------------------------------------------
    if args.merge_dir:
        print(f"Merging cache chunks from {args.merge_dir} into {args.output_path}...")
        chunks = sorted(
            glob.glob(os.path.join(args.merge_dir, "prior_cache_rank_*.pt")),
            key=lambda x: int(os.path.basename(x).split("_rank_")[-1].split(".pt")[0]),
        )
        all_tgt, all_top1, all_mrgn = [], [], []
        for f in tqdm(chunks, desc="Merging"):
            c = torch.load(f, weights_only=False)
            all_tgt.extend(c["prior_target_logp"])
            all_top1.extend(c["prior_top1_logp"])
            all_mrgn.extend(c["prior_margin"])
        torch.save(
            {"prior_target_logp": all_tgt, "prior_top1_logp": all_top1, "prior_margin": all_mrgn},
            args.output_path,
        )
        print(f"Merged {len(all_tgt)} samples → {args.output_path}")
        return

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

    print(f"[Rank {rank}/{world_size}] Loading model {args.base_model} on cuda:{local_rank}")

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

    # Stagger model loading across nodes to avoid Lustre NFS overload
    num_nodes = int(os.environ.get("NNODES", world_size // 4 if world_size >= 4 else 1))
    node_id = rank // (world_size // num_nodes) if num_nodes > 0 else 0
    time.sleep(node_id * 3)

    # Lightweight gloo process group for the final save-barrier only
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(hours=4))

    # Download / verify model files (huggingface_hub handles file locking)
    from huggingface_hub import snapshot_download
    print(f"[Rank {rank}] Downloading/verifying model files via snapshot_download...")
    local_model_path = snapshot_download(
        repo_id=args.base_model,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.bin", "*.txt"],
        max_workers=1,
    )
    print(f"[Rank {rank}] Model files ready at: {local_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, trust_remote_code=True, local_files_only=True,
    )

    # Lock down to offline mode from here on
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    if dist.is_initialized():
        dist.barrier()

    # ------------------------------------------------------------------
    # Patch safetensors to bypass Lustre mmap
    # ------------------------------------------------------------------
    safetensors.safe_open = SafeOpenStreamer
    transformers.modeling_utils.safe_open = SafeOpenStreamer
    print(f"[Rank {rank}] Applied SafeOpenStreamer (tensor-by-tensor, zero mmap)")

    # ------------------------------------------------------------------
    # Load model — single-node multi-GPU pipeline via Accelerate
    # ------------------------------------------------------------------
    print(f"[Rank {rank}/{world_size}] Loading {local_model_path} via device_map='auto'...")

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    gc.collect()

    print(f"[Rank {rank}/{world_size}] Model loaded successfully!")

    # ------------------------------------------------------------------
    # Dataset sharding — each rank processes its own chunk independently
    # ------------------------------------------------------------------
    output_dir = Path(args.output_path).parent if Path(args.output_path).suffix else Path(args.output_path)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_path, split=args.dataset_split)
    print(f"[Rank {rank}] Dataset total length: {len(dataset)}")

    if world_size > 1:
        shard_size = (len(dataset) + world_size - 1) // world_size
        start_sample = rank * shard_size
        end_sample = min((rank + 1) * shard_size, len(dataset))
        dataset = dataset.select(range(start_sample, end_sample))

    print(f"[Rank {rank}] Processing {len(dataset)} samples...")

    all_target_logp = []
    all_top1_logp = []
    all_margin = []

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    for start_idx in tqdm(
        range(0, len(dataset), args.batch_size),
        desc=f"Cache Gen [Rank {rank}]",
        disable=(rank != 0),
        mininterval=2.0,
    ):
        end_idx = min(start_idx + args.batch_size, len(dataset))
        batch_samples = dataset[start_idx:end_idx]

        # Detect message column
        messages_col = None
        for col in ["messages", "conversations"]:
            if col in batch_samples:
                messages_col = col
                break

        # Tokenize
        if messages_col:
            texts = []
            for i in range(len(batch_samples[messages_col])):
                msgs = batch_samples[messages_col][i]

                # Auto-convert ShareGPT ↔ standard HuggingFace messages format
                standard_msgs = []
                if isinstance(msgs, list):
                    for m in msgs:
                        if isinstance(m, dict):
                            if "from" in m and "value" in m:
                                role = "user" if m["from"] in ("human", "user") else "assistant"
                                content = m["value"] if m["value"] is not None else ""
                                standard_msgs.append({"role": role, "content": content})
                            elif "role" in m and "content" in m:
                                content = m["content"] if m["content"] is not None else ""
                                standard_msgs.append({"role": m["role"], "content": content})
                            else:
                                standard_msgs.append(m)
                        else:
                            standard_msgs.append(m)
                else:
                    standard_msgs = msgs

                text = tokenizer.apply_chat_template(
                    standard_msgs, tokenize=False, add_generation_prompt=False,
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

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        cache = compute_prior_logits_for_batch(model, input_ids, attention_mask, labels)

        for i in range(input_ids.size(0)):
            seq_len = attention_mask[i].sum().item()
            all_target_logp.append(cache["prior_target_logp"][i, :seq_len])
            all_top1_logp.append(cache["prior_top1_logp"][i, :seq_len])
            all_margin.append(cache["prior_margin"][i, :seq_len])

    # ------------------------------------------------------------------
    # Save rank-local cache
    # ------------------------------------------------------------------
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
    print(f"[Rank {rank}] Prior cache saved → {cache_path} ({len(all_target_logp)} samples)")


if __name__ == "__main__":
    main()
