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
Plugin init for the L2-SP forgetting baseline.
"""

from axolotl.integrations.base import BasePlugin

from .args import L2SPArgs as L2SPArgs


class L2SPPlugin(BasePlugin):
    """
    L2-SP regularization plugin.
    """

    def get_input_args(self):
        return "axolotl.integrations.l2sp.L2SPArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.l2sp.args.L2SPTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.l2sp_trainer:
            from .trainer import AxolotlL2SPTrainer

            return AxolotlL2SPTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "l2sp_lambda": cfg.l2sp_lambda,
        }
