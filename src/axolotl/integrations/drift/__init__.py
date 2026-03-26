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
Plugin init for Drift: Unified Risk Score.

Combines instantaneous hardness and historical drift into a single
token risk score for adaptive weighted CE fine-tuning.
Single model, single forward, no cache.
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import DriftArgs as DriftArgs

LOG = get_logger(__name__)


class DriftPlugin(BasePlugin):
    """
    Plugin for Drift (unified risk score).

    Memory: ~1× model size. No teacher, no cache.
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
            "drift_rho": cfg.drift_rho,
            "drift_beta": cfg.drift_beta,
            "drift_tau": cfg.drift_tau,
            "drift_lambda": cfg.drift_lambda,
        }
