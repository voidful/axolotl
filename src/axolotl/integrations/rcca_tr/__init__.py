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
Plugin init for RCCA-TR: Suppress-by-Default, Rescue-if-Useful.

Provides token-wise adaptive weighted CE fine-tuning with:
  - Offline prior cache (no live frozen model)
  - Hardness gate h_t (suppress high-perplexity tokens by default)
  - Useful-hard gate q_t (rescue hard tokens with positive improvement evidence)
  - Only the active model in GPU memory
"""

import torch
from transformers import Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .args import RCCATRArgs as RCCATRArgs

LOG = get_logger(__name__)


class RCCATRPlugin(BasePlugin):
    """
    Plugin for RCCA-TR support in Axolotl.

    Memory: ~1× model size (just the active model).
    Prior information is pre-computed offline and loaded as cached values.
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
            "rcca_tr_tau_p": cfg.rcca_tr_tau_p,
            "rcca_tr_T_p": cfg.rcca_tr_T_p,
            "rcca_tr_w_min": cfg.rcca_tr_w_min,
        }

    def get_collator_cls_and_kwargs(self, cfg, is_eval=False):
        return None, None

    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer):
        pass

