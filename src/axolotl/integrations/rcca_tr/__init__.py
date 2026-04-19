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
Plugin init for Drift-Trust.

A token-level regularization family built on a shared drift signal
and two transfer functions for post-training knowledge preservation.

Four modes (controlled by ``rcca_tr_mode``):

  ce            — Standard cross-entropy baseline.
  hardness      — Self-paced hardness weighting only (ablation).
  drift_trust_s — Suppressive mapping (best for noisy alignment).
  drift_trust_a — Anchoring mapping (best for clean domain specialization).

Shared drift signal:
  d_t = log p_θ(y_t) − running_mean     [temporal drift]

Drift-Trust-S (suppressive):
  w_t = w_min + (1 - w_min) · (β · s_t + (1 - β) · r_t)
  w_t ∈ [0.05, 1.0]

Drift-Trust-A (anchoring):
  w_t = w_0 + λ · r_t
  w_t ∈ [0.1, 4.1]
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import RCCATRArgs as RCCATRArgs

LOG = get_logger(__name__)


class RCCATRPlugin(BasePlugin):
    """
    Plugin for Drift-Trust.

    Memory: ~1× model size (just the active model).
    Evidence drift is tracked via a lightweight scalar EMA buffer.
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
            "rcca_tr_mode": cfg.rcca_tr_mode,
            "rcca_tr_ema_decay": cfg.rcca_tr_ema_decay,
            "rcca_tr_self_tau": cfg.rcca_tr_self_tau,
            "rcca_tr_drift_tau": cfg.rcca_tr_drift_tau,
            "rcca_tr_w_min": cfg.rcca_tr_w_min,
            "rcca_tr_beta": cfg.rcca_tr_beta,
            "rcca_tr_drift_gamma": cfg.rcca_tr_drift_gamma,
            "rcca_tr_anchor_base": cfg.rcca_tr_anchor_base,
            "rcca_tr_anchor_lambda": cfg.rcca_tr_anchor_lambda,
            "rcca_tr_reliability_tau": cfg.rcca_tr_reliability_tau,
        }
