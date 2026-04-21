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
Plugin init for entropy-focus baselines.

This family keeps only one half of tokens, split by batch-mean predictive entropy:

  - high: keep tokens with entropy >= mean entropy
  - low:  keep tokens with entropy < mean entropy
"""

from axolotl.integrations.base import BasePlugin

from .args import EntropyFocusArgs as EntropyFocusArgs


class EntropyFocusPlugin(BasePlugin):
    """
    High-entropy / low-entropy token selection baseline plugin.
    """

    def get_input_args(self):
        return "axolotl.integrations.entropy_focus.EntropyFocusArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.entropy_focus.args.EntropyFocusTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.entropy_focus_trainer:
            from .trainer import AxolotlEntropyFocusTrainer

            return AxolotlEntropyFocusTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "entropy_focus_mode": cfg.entropy_focus_mode,
            "entropy_focus_eps": cfg.entropy_focus_eps,
        }
