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
Plugin init for drift-loss.

drift-loss keeps CE as the base loss, and only uses drift to measure
improvement versus a reference state.

Default:
  - scalar EMA reference
  - detached mean-preserving focal weights
  - one primary tuning parameter: drift_gamma

Appendix-style variant:
  - token-wise prior reference via `reference_target_logp`
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import DriftArgs as DriftArgs

LOG = get_logger(__name__)


class DriftPlugin(BasePlugin):
    """
    Plugin for drift-loss.

    Memory: ~1× model size (just the active model).
    Default temporal state is a single scalar EMA reference.
    """

    def get_input_args(self):
        return "axolotl.integrations.drift.DriftArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.drift.args.DriftTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.drift_trainer:
            from .trainer import AxolotlDriftTrainer

            return AxolotlDriftTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "drift_reference_mode": cfg.drift_reference_mode,
            "drift_reference_key": cfg.drift_reference_key,
            "drift_ema_decay": cfg.drift_ema_decay,
            "drift_gamma": cfg.drift_gamma,
            "drift_detach_weights": cfg.drift_detach_weights,
            "drift_eps": cfg.drift_eps,
            # Legacy trust-region args are still forwarded for compatibility.
            "drift_reliability_beta": cfg.drift_reliability_beta,
            "drift_reliability_tau": cfg.drift_reliability_tau,
            "drift_epsilon_min": cfg.drift_epsilon_min,
            "drift_epsilon_max": cfg.drift_epsilon_max,
            "drift_kl_lambda": cfg.drift_kl_lambda,
            "drift_anchor_weight": cfg.drift_anchor_weight,
            "drift_use_smooth_objective": cfg.drift_use_smooth_objective,
            "drift_per_sample": cfg.drift_per_sample,
        }
