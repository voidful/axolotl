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
            LOG.info("No prior cache path specified. Prior values will be zeros.")
            return

        LOG.info("Loading prior cache from %s", cache_path)
        cache = torch.load(cache_path, weights_only=True)
        num_cache_samples = len(cache["prior_target_logp"])
        LOG.info("Prior cache loaded: %d samples", num_cache_samples)

        def inject_prior(example, idx):
            seq_len = len(example["input_ids"])
            if idx < num_cache_samples:
                cached_logp = cache["prior_target_logp"][idx]
                cached_margin = cache["prior_margin"][idx]
                logp_list = cached_logp[:seq_len].tolist()
                margin_list = cached_margin[:seq_len].tolist()
                # Pad if cache sequence is shorter than tokenized sequence
                if len(logp_list) < seq_len:
                    logp_list += [0.0] * (seq_len - len(logp_list))
                    margin_list += [0.0] * (seq_len - len(margin_list))
                example["prior_target_logp"] = logp_list
                example["prior_margin"] = margin_list
            else:
                example["prior_target_logp"] = [0.0] * seq_len
                example["prior_margin"] = [0.0] * seq_len
            return example

        trainer.train_dataset = trainer.train_dataset.map(
            inject_prior, with_indices=True
        )
        LOG.info("Prior cache values injected into training dataset.")
