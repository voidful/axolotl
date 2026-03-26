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
Plugin init for RCCA-TR (no-cache variant).

Provides token-wise adaptive trust-region fine-tuning with:
  - Drift buffer for temporal evidence tracking
  - Only the active model in GPU memory
  - No frozen prior cache required
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import RCCATRArgs as RCCATRArgs

LOG = get_logger(__name__)


class RCCATRPlugin(BasePlugin):
    """
    Plugin for RCCA-TR support in Axolotl (no-cache variant).

    Memory: ~1× model size (just the active model).
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
            "rcca_tr_reliability_beta": cfg.rcca_tr_reliability_beta,
            "rcca_tr_reliability_tau": cfg.rcca_tr_reliability_tau,
            "rcca_tr_epsilon_min": cfg.rcca_tr_epsilon_min,
            "rcca_tr_epsilon_max": cfg.rcca_tr_epsilon_max,
            "rcca_tr_kl_lambda": cfg.rcca_tr_kl_lambda,
            "rcca_tr_use_smooth_objective": cfg.rcca_tr_use_smooth_objective,
            "rcca_tr_ema_decay": cfg.rcca_tr_ema_decay,
            "rcca_tr_drift_gamma": cfg.rcca_tr_drift_gamma,
        }
