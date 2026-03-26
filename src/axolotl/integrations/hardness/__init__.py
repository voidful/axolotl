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
Plugin init for Hardness variant.

Self-paced hardness suppression: tokens with high CE are suppressed,
tokens with low CE are learned normally. No frozen prior cache required.

  h_t = σ((CE_t − τ_p) / T_p)
  w_t = w_min + (1 − w_min) · (1 − h_t)
  L = Σ w_t · CE_t / |V|
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import HardnessArgs as HardnessArgs

LOG = get_logger(__name__)


class HardnessPlugin(BasePlugin):
    """
    Plugin for Hardness variant.

    Memory: ~1× model size (just the active model).
    Token weighting based purely on the active model's cross-entropy.
    """

    def get_input_args(self):
        return "axolotl.integrations.hardness.HardnessArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.hardness.args.HardnessTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.hardness_trainer:
            from .trainer import AxolotlHardnessTrainer

            return AxolotlHardnessTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "hardness_tau_p": cfg.hardness_tau_p,
            "hardness_T_p": cfg.hardness_T_p,
            "hardness_w_min": cfg.hardness_w_min,
        }
