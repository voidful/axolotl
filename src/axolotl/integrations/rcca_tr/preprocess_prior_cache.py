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

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
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
    B, T = input_ids.shape
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        shifted_target_logp = torch.zeros(B, T-1, device=input_ids.device)
        shifted_top1_logp = torch.zeros(B, T-1, device=input_ids.device)
        
        # 1. PyTorch C++ kernels (like conv1d in Mamba/Qwen3.5-A3B) use 32-bit int math for indexing.
        # If B=32, T=8000, D=16384, elements = 4.19 Billion > 2.14 Billion (int32 limit).
        # We must sub-batch the forward pass to guarantee we never hit this catastrophic PyTorch limitation.
        mini_batch_size = 4
        chunk_size = 1024 # [1, 1024, 151936] = ~600MB
        
        for mb_start in range(0, B, mini_batch_size):
            mb_end = min(B, mb_start + mini_batch_size)
            mb_input_ids = input_ids[mb_start:mb_end]
            mb_attention_mask = attention_mask[mb_start:mb_end]
            
            # Forward base model without lm_head
            base_model = getattr(model, model.base_model_prefix) if hasattr(model, "base_model_prefix") else model.model
            base_outputs = base_model(input_ids=mb_input_ids, attention_mask=mb_attention_mask)
            hidden_states = base_outputs[0] if isinstance(base_outputs, tuple) else base_outputs.last_hidden_state
            
            # 2. Iterate items in this mini-batch and chunk sequence carefully for lm_head
            for local_b in range(mb_end - mb_start):
                global_b = mb_start + local_b
                for i in range(0, T - 1, chunk_size):
                    i_end = min(i + chunk_size, T - 1)
                    
                    h_chunk = hidden_states[local_b:local_b+1, i:i_end, :] # [1, C, D]
                    logits_chunk = model.lm_head(h_chunk)      # [1, C, V]
                    log_probs_chunk = F.log_softmax(logits_chunk.float(), dim=-1) # [1, C, V]
                    
                    # shift targets and masks
                    labels_chunk = labels[global_b:global_b+1, i+1:i_end+1] # [1, C]
                    safe_labels_chunk = labels_chunk.clamp(min=0)
                    valid_chunk = (labels_chunk != -100) & attention_mask[global_b:global_b+1, i:i_end].bool()
                    
                    target_lp = log_probs_chunk.gather(dim=-1, index=safe_labels_chunk.unsqueeze(-1)).squeeze(-1) # [1, C]
                    top1_lp = log_probs_chunk.max(dim=-1).values # [1, C]
                    
                    shifted_target_logp[global_b:global_b+1, i:i_end] = target_lp * valid_chunk.float()
                    shifted_top1_logp[global_b:global_b+1, i:i_end] = top1_lp * valid_chunk.float()
                    
                    del h_chunk, logits_chunk, log_probs_chunk
            
            del base_outputs, hidden_states
            torch.cuda.empty_cache()
                
        shifted_margin = (shifted_top1_logp - shifted_target_logp)

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

    # NOTE: Do NOT set OFFLINE envvars here!
    # They must be set AFTER snapshot_download on Rank 0 so files can be fetched.
    
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))) 
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

    # --- HPC NFS DDoS Bypass Removed ---
    # We no longer copy to `/tmp` because `/tmp` is typically a RAM-backed tmpfs on SLURM, 
    # and copying a 54GB model into it instantly exhausts 54GB of Physical RAM before inference even starts!
    # Instead, our sequential staggered loading (implemented below) sufficiently protects the NFS from DDoS
    # without sacrificing Node Virtual Memory.
    pass

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

    import time
    import fcntl
    
    # Stagger model loading across nodes by 3 seconds each to avoid Lustre NFS overload
    num_nodes = int(os.environ.get("NNODES", world_size // 4 if world_size >= 4 else 1))
    node_id = rank // (world_size // num_nodes) if num_nodes > 0 else 0
    time.sleep(node_id * 3)
    
    # We rely on Accelerate / DeepSpeed zero-3 for memory-efficient loading.
    # When zero-3 is enabled, transformers automatically uses deepspeed.zero.Init()
    # to partition the model across GPUs WITHOUT loading the full 50GB into CPU RAM.
    # Add a barrier to ensure only Rank 0 pulls HuggingFace files to cache first to avoid race conditions 
    import torch.distributed as dist
    import datetime
    
    # Initialize basic gloo process group just for this barrier if not initialized
    if dist.is_available() and not dist.is_initialized():
         dist.init_process_group(backend="gloo", timeout=datetime.timedelta(hours=4))
         
    
    # Every rank calls snapshot_download independently.
    # huggingface_hub handles file locking internally, so concurrent calls are safe.
    # This ensures each node has the files in its local HF cache before loading.
    from huggingface_hub import snapshot_download
    print(f"[Rank {rank}] Downloading/verifying model files via snapshot_download...")
    local_model_path = snapshot_download(
        repo_id=args.base_model,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.bin", "*.txt"],
        max_workers=1,  # conservative to avoid Lustre overload
    )
    print(f"[Rank {rank}] Model files ready at: {local_model_path}")
    
    # Load tokenizer from the local path to avoid any HF Hub resolution issues
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True)
    
    # Now set offline mode to prevent any further network calls
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    if dist.is_initialized():
        dist.barrier()
        
    print(f"[Rank {rank}/{world_size}] Loading {local_model_path} using Accelerate/DeepSpeed...")
    
    print(f"[Rank {rank}/{world_size}] Loading {local_model_path} using Accelerate/DeepSpeed...")
    
    # -------------------------------------------------------------------------
    # Lustre/SLURM mmap Bypass Patch (Ultimate `SafeOpenRam` Version)
    # -------------------------------------------------------------------------
    # Lustre absolutely rejects 30-node concurrent `mmap` calls on the same 5GB 
    # safetensor file (`Cannot allocate memory` / `ENOMEM`). 
    # But allocating `torch.empty` for EVERY tensor causes 35GB+ Python Heap leaks
    # dragging GC to its knees and blowing up the CGroup limits!
    # 
    # SOLUTION: Allocate exactly ONE `torch.empty(5GB)` matching the exact file size.
    # Read the whole file strictly sequentially using `f.readinto()` (Hyper fast, 
    # no Lustre locks). Then behave exactly like native safetensors: dole out 
    # `view()` slices that point right into that giant contiguous buffer!
    
    import safetensors
    import transformers.modeling_utils
    import json
    import struct
    import torch

    class SafeOpenRam:
        def __init__(self, filename, framework="pt", device="cpu"):
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.filename = filename
            self.device = device
            
            file_size = os.path.getsize(filename)
            self._buffer = torch.empty(file_size, dtype=torch.uint8, device="cpu")
            m_view = memoryview(self._buffer.numpy())
            
            with open(self.filename, 'rb') as f:
                bytes_read = 0
                while bytes_read < file_size:
                    n = f.readinto(m_view[bytes_read:])
                    if n == 0:
                        raise EOFError(f"Unexpected EOF while streaming {filename}")
                    bytes_read += n
                    
            header_size_bytes = self._buffer[:8].numpy().tobytes()
            self.header_size = struct.unpack('<Q', header_size_bytes)[0]
            
            header_json_bytes = self._buffer[8:8+self.header_size].numpy().tobytes()
            header_json = header_json_bytes.decode('utf-8')
            self.metadata = json.loads(header_json)
            
            self.data_offset = 8 + self.header_size
            self.tensor_keys = [k for k in self.metadata.keys() if k != '__metadata__']
            
        def keys(self):
            return self.tensor_keys
            
        def get_tensor(self, key):
            info = self.metadata[key]
            dtype_str = info['dtype']
            shape = info['shape']
            offsets = info['data_offsets']
            
            dtype_map = {
                "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
                "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
                "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool
            }
            dt = dtype_map[dtype_str]
            start_idx = self.data_offset + offsets[0]
            end_idx = self.data_offset + offsets[1]
            
            # Memory View, 0 overhead. Exactly like mmap.
            t_slice = self._buffer[start_idx:end_idx]
            return t_slice.view(dtype=dt).reshape(shape)
            
        def get_slice(self, key):
            class _SliceProxy:
                def __init__(self, parent, key):
                    self.parent = parent
                    self.key = key
                def __getitem__(self, idx):
                    t = self.parent.get_tensor(self.key)
                    return t[idx]
            return _SliceProxy(self, key)
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._buffer = None
            self.metadata = None
            self.tensor_keys = None

    safetensors.safe_open = SafeOpenRam
    transformers.modeling_utils.safe_open = SafeOpenRam
    print(f"[Rank {rank}] Applied SafeOpenRam (Native MMap emulation via 5GB Block Read!)")
    # -------------------------------------------------------------------------
    
    # Force DeepSpeed ZeRO-3 partitioned loading manually
    # Since this is a standalone script, we must initialize the zero.Init context explicitly
    try:
        import deepspeed
        from deepspeed.runtime.zero.partition_parameters import ZeroParamType
        
        print(f"[Rank {rank}] Forcing DeepSpeed ZeRO-3 Init context to partition model weights...")
        
        # Determine deepspeed config path from environment, or use default zero3
        ds_config_path = os.environ.get("DEEPSPEED_CONFIG", "deepspeed_configs/zero3_custom.json")
        if os.path.exists(ds_config_path):
             import json
             with open(ds_config_path, "r") as f:
                 ds_config = json.load(f)
             # DeepSpeed standalone Init() doesn't understand "auto"
             if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
                 ds_config["train_micro_batch_size_per_gpu"] = 1
             if ds_config.get("train_batch_size") == "auto":
                 ds_config["train_batch_size"] = 1
             if ds_config.get("gradient_accumulation_steps") == "auto":
                 ds_config["gradient_accumulation_steps"] = 1
        else:
             # minimal inline config if file not found
             ds_config = {"train_micro_batch_size_per_gpu": 1, "train_batch_size": 1, "zero_optimization": {"stage": 3}}
             
        with deepspeed.zero.Init(config_dict_or_path=ds_config, remote_device=None):
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,  # Use local path, NOT HF repo ID
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa",
                local_files_only=True,  # Force local-only, no HF Hub resolution
                low_cpu_mem_usage=True,  # Critical for ZeRO-3
                ignore_mismatched_sizes=True,  # Prevent shape mismatch failures
            )
        print(f"[Rank {rank}] Model loaded successfully via ZeRO-3 Init!")
    except ImportError:
        # Fallback to device_map if deepspeed is not installed
        print(f"[Rank {rank}] WARNING: deepspeed not found, falling back to device_map. May OOM on Lustre.")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            trust_remote_code=True,
            attn_implementation="sdpa",
            local_files_only=True,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
        )
    model.eval()
    import gc
    gc.collect()

    print(f"[Rank {rank}/{world_size}] Model loaded successfully!")

    # Global barrier: wait for ALL ranks to finish loading before any inference begins
    # Use gloo backend (CPU-only) to avoid allocating extra CUDA VAS
    import torch.distributed as dist
    import datetime
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            timeout=datetime.timedelta(hours=4)
        )
    dist.barrier()
    print(f"[Rank {rank}] All ranks loaded. Starting inference...")

    output_dir = Path(args.output_path).parent if Path(args.output_path).suffix else Path(args.output_path)

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    dataset = load_dataset(args.dataset_path, split=args.dataset_split)
    
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
    for start_idx in tqdm(
        range(0, len(dataset), args.batch_size),
        desc="Cache Gen [Rank 0]",
        disable=(rank != 0),
        mininterval=2.0
    ):
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
                
                # Auto-convert ShareGPT format to standard HuggingFace messages format
                standard_msgs = []
                if isinstance(msgs, list):
                    for m in msgs:
                        if isinstance(m, dict):
                            if "from" in m and "value" in m:
                                role = "user" if m["from"] in ["human", "user"] else "assistant"
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
                    standard_msgs, tokenize=False, add_generation_prompt=False
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

    if world_size > 1:
        print(f"[Rank {rank}] Waiting for all ranks to finish...")
        dist.barrier()
        print(f"[Rank {rank}] All ranks completed!")


if __name__ == "__main__":
    main()
