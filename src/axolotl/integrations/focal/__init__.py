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
Plugin init for the token-level focal-loss baseline.

This baseline reshapes token CE by down-weighting confident targets:

  p_t = p_theta(y_t)
  w_t = (1 - p_t) ^ gamma
  L = mean(w_t * CE_t)
"""

from axolotl.integrations.base import BasePlugin

from .args import FocalArgs as FocalArgs


class FocalPlugin(BasePlugin):
    """
    Token-level focal loss baseline plugin.
    """

    def get_input_args(self):
        return "axolotl.integrations.focal.FocalArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.focal.args.FocalTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.focal_trainer:
            from .trainer import AxolotlFocalTrainer

            return AxolotlFocalTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "focal_gamma": cfg.focal_gamma,
            "focal_eps": cfg.focal_eps,
        }
