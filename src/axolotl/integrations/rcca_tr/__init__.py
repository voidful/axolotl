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
Plugin init to add Reliability-Calibrated Conflict-Aware Trust-Region
Fine-Tuning (RCCA-TR) support to Axolotl.
"""

from typing import Any

from transformers import Trainer

from axolotl.integrations.base import BasePlugin

from .args import RCCATRArgs as RCCATRArgs


class RCCATRPlugin(BasePlugin):
    """
    Plugin for RCCA-TR support in Axolotl.

    Provides a custom trainer that implements a token-wise adaptive trust region
    for fine-tuning, controlled by two signals:
      1. Conflict score: whether the supervision contradicts the prior.
      2. Reliability score: whether the prior is stable and trustworthy.
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
            "rcca_tr_num_perturbations": cfg.rcca_tr_num_perturbations,
            "rcca_tr_stability_update_interval": cfg.rcca_tr_stability_update_interval,
        }

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Trainer) -> list:
        """
        Adds the EMA update callback to the Trainer instance.

        Args:
            cfg: Configuration object.
            trainer: Huggingface Trainer instance.

        Returns:
            list: List containing the EMA update callback.
        """
        if not cfg.rcca_tr_trainer:
            return []

        from .callbacks import EMAUpdateCallback

        ema_decay = cfg.rcca_tr_ema_decay if cfg.rcca_tr_ema_decay is not None else 0.999
        return [EMAUpdateCallback(ema_decay=ema_decay)]
