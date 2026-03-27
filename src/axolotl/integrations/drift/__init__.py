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
Plugin init for Drift variant.

Drift-based trust-region fine-tuning with:
  - DriftBuffer for temporal evidence tracking
  - Only the active model in GPU memory
  - No frozen prior cache required

  drift_t = 0.5 · CE_t + 0.5 · d_running
  r_t = σ((exp(-γ · drift_t) - μ) / τ)
  L = λ · r_t · CE_t
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import DriftArgs as DriftArgs

LOG = get_logger(__name__)


class DriftPlugin(BasePlugin):
    """
    Plugin for Drift variant.

    Memory: ~1× model size (just the active model).
    Evidence drift is tracked via a lightweight statistical buffer.
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
            "drift_reliability_beta": cfg.drift_reliability_beta,
            "drift_reliability_tau": cfg.drift_reliability_tau,
            "drift_epsilon_min": cfg.drift_epsilon_min,
            "drift_epsilon_max": cfg.drift_epsilon_max,
            "drift_kl_lambda": cfg.drift_kl_lambda,
            "drift_use_smooth_objective": cfg.drift_use_smooth_objective,
            "drift_ema_decay": cfg.drift_ema_decay,
            "drift_gamma": cfg.drift_gamma,
        }
