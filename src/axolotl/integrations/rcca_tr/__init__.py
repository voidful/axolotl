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
Plugin init for RCCA-TR A+ variant.

Provides token-wise adaptive trust-region fine-tuning with:
  - Offline prior cache (no live frozen model)
  - Drift buffer (no live EMA model)
  - Only the active model in GPU memory
"""

import torch
from transformers import Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .args import RCCATRArgs as RCCATRArgs

LOG = get_logger(__name__)


class RCCATRPlugin(BasePlugin):
    """
    Plugin for RCCA-TR A+ support in Axolotl.

    Memory: ~1× model size (just the active model).
    Prior information is pre-computed offline and loaded as cached values.
    Evidence drift is tracked via a lightweight statistical buffer.
    """

    def get_input_args(self):
        return "axolotl.integrations.rcca_tr.RCCATRArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.rcca_tr.args.RCCATRTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.rcca_tr_trainer:
            from .trainer import AxolotlRCCATRTrainer

            return AxolotlRCCATRTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "rcca_tr_conflict_lambda1": cfg.rcca_tr_conflict_lambda1,
            "rcca_tr_conflict_lambda2": cfg.rcca_tr_conflict_lambda2,
            "rcca_tr_conflict_tau": cfg.rcca_tr_conflict_tau,
            "rcca_tr_reliability_beta": cfg.rcca_tr_reliability_beta,
            "rcca_tr_reliability_tau": cfg.rcca_tr_reliability_tau,
            "rcca_tr_epsilon_min": cfg.rcca_tr_epsilon_min,
            "rcca_tr_epsilon_max": cfg.rcca_tr_epsilon_max,
            "rcca_tr_kl_lambda": cfg.rcca_tr_kl_lambda,
            "rcca_tr_use_smooth_objective": cfg.rcca_tr_use_smooth_objective,
            "rcca_tr_ema_decay": cfg.rcca_tr_ema_decay,
            "rcca_tr_drift_gamma": cfg.rcca_tr_drift_gamma,
            "rcca_tr_prior_cache_path": cfg.rcca_tr_prior_cache_path,
        }

    def get_collator_cls_and_kwargs(self, cfg, is_eval=False):
        if not cfg.rcca_tr_trainer:
            return None, None

        from .collator import DataCollatorForRCCATR

        return DataCollatorForRCCATR, {}

    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer):
        cache_path = getattr(cfg, "rcca_tr_prior_cache_path", None)
        if not cache_path:
            if getattr(cfg, "rcca_tr_trainer", False):
                raise ValueError(
                    "rcca_tr_prior_cache_path is required when rcca_tr_trainer is True. "
                    "Run `python -m axolotl.integrations.rcca_tr.preprocess_prior_cache` "
                    "to generate the prior cache first."
                )
        import os
        import torch.distributed as dist
        import hashlib

        assert cache_path is not None, "cache_path must be a string here"
        
        if not os.path.exists(cache_path):
            raise ValueError(
                f"Prior cache file not found at '{cache_path}'. "
                f"You must generate it before training! Run:\n"
                f"python -m axolotl.integrations.rcca_tr.preprocess_prior_cache "
                f"--base_model <YOUR_MODEL> --dataset_path <YOUR_DATA> --output_path {os.path.dirname(cache_path) or './prior_cache'}"
            )

        is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

        # Create a deterministic fingerprint for the mapped dataset so ranks 1-N can hit the cache
        # Note: We incorporate the file modification time so that if the prior is regenerated, it remaps.
        mtime = os.path.getmtime(cache_path)
        fingerprint = hashlib.md5(f"rcca_tr_prior_{cache_path}_{mtime}".encode()).hexdigest()

        if is_main_process:
            import time
            t0 = time.time()
            LOG.info("Loading prior cache from %s", cache_path)
            cache = torch.load(cache_path, weights_only=True)
            num_cache_samples = len(cache["prior_target_logp"])
            num_dataset_samples = len(trainer.train_dataset)
            LOG.info("Prior cache loaded: %d samples (dataset: %d samples) in %.1fs", 
                     num_cache_samples, num_dataset_samples, time.time() - t0)

            # Convert tensors to Python lists up-front (avoids repeated .tolist() calls)
            t1 = time.time()
            prior_logp_lists = [t.tolist() for t in cache["prior_target_logp"]]
            prior_margin_lists = [t.tolist() for t in cache["prior_margin"]]
            del cache
            LOG.info("Tensor to list conversion done in %.1fs", time.time() - t1)

            # Build the columns: trim/pad each entry to match the tokenized sequence length
            t2 = time.time()
            logp_column = []
            margin_column = []
            for idx in range(num_dataset_samples):
                seq_len = len(trainer.train_dataset[idx]["input_ids"])
                if idx < num_cache_samples:
                    logp = prior_logp_lists[idx][:seq_len]
                    margin = prior_margin_lists[idx][:seq_len]
                    if len(logp) < seq_len:
                        logp += [0.0] * (seq_len - len(logp))
                        margin += [0.0] * (seq_len - len(margin))
                else:
                    logp = [0.0] * seq_len
                    margin = [0.0] * seq_len
                logp_column.append(logp)
                margin_column.append(margin)
                if (idx + 1) % 10000 == 0:
                    LOG.info("Prior injection progress: %d/%d samples (%.1fs)", 
                             idx + 1, num_dataset_samples, time.time() - t2)

            del prior_logp_lists, prior_margin_lists
            LOG.info("Column building done: %d samples in %.1fs", num_dataset_samples, time.time() - t2)

            # Use add_column which is much faster than .map()
            t3 = time.time()
            trainer.train_dataset = trainer.train_dataset.add_column("prior_target_logp", logp_column)
            trainer.train_dataset = trainer.train_dataset.add_column("prior_margin", margin_column)
            del logp_column, margin_column

            # Save with fingerprint so other ranks can load from cache
            trainer.train_dataset.save_to_disk(f"/tmp/rcca_tr_dataset_{fingerprint}")
            LOG.info("Prior cache injected and saved in %.1fs (total: %.1fs)", 
                     time.time() - t3, time.time() - t0)

        if dist.is_initialized():
            dist.barrier()

        if not is_main_process:
            from datasets import load_from_disk
            LOG.info("Loading injected dataset from shared cache...")
            trainer.train_dataset = load_from_disk(f"/tmp/rcca_tr_dataset_{fingerprint}")
