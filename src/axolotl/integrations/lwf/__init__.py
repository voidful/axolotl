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
Plugin init for the Learning without Forgetting baseline.
"""

from axolotl.integrations.base import BasePlugin

from .args import LWFArgs as LWFArgs


class LWFPlugin(BasePlugin):
    """
    Learning without Forgetting plugin.
    """

    def get_input_args(self):
        return "axolotl.integrations.lwf.LWFArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.lwf.args.LWFTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.lwf_trainer:
            from .trainer import AxolotlLWFTrainer

            return AxolotlLWFTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "lwf_teacher_model": cfg.lwf_teacher_model,
            "lwf_ce_alpha": cfg.lwf_ce_alpha,
            "lwf_alpha": cfg.lwf_alpha,
            "lwf_temperature": cfg.lwf_temperature,
        }
