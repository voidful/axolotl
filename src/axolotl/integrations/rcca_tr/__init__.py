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
Plugin init for RCCA-TR (Reliability-Calibrated, Curriculum-Aware Trust Region).

Token-level dense weighting for knowledge-preserving post-training.

Four modes (controlled by ``rcca_tr_mode``):
  - ce:         Standard cross-entropy baseline.
  - hardness:   Self-paced hardness weighting only.
  - drift_only: Self-paced + drift regularization (MAIN METHOD).
  - drift:      Legacy trust-region formulation (ablation).

Main method formula (drift_only):
  w_t = w_min + (1 - w_min) · (β · s_t + (1 - β) · r_t)
  L   = Σ w_t · CE_t / N

Where:
  s_t = σ((log p_θ(y_t) − μ_s) / τ_s)   [self-paced score]
  r_t = σ((d_t − μ_r) / τ_r)             [drift score]
  d_t = log p_θ(y_t) − running_mean       [temporal drift]
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import RCCATRArgs as RCCATRArgs

LOG = get_logger(__name__)


class RCCATRPlugin(BasePlugin):
    """
    Plugin for RCCA-TR.

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
            "rcca_tr_tau_p": cfg.rcca_tr_tau_p,
            "rcca_tr_T_p": cfg.rcca_tr_T_p,
            "rcca_tr_tau_delta": cfg.rcca_tr_tau_delta,
            "rcca_tr_T_delta": cfg.rcca_tr_T_delta,
            "rcca_tr_w_min": cfg.rcca_tr_w_min,
            "rcca_tr_beta": cfg.rcca_tr_beta,
            "rcca_tr_self_tau": cfg.rcca_tr_self_tau,
            "rcca_tr_drift_tau": cfg.rcca_tr_drift_tau,
            "rcca_tr_drift_gamma": cfg.rcca_tr_drift_gamma,
            "rcca_tr_ema_decay": cfg.rcca_tr_ema_decay,
            "rcca_tr_kl_lambda": cfg.rcca_tr_kl_lambda,
            "rcca_tr_anchor_weight": cfg.rcca_tr_anchor_weight,
            "rcca_tr_reliability_tau": cfg.rcca_tr_reliability_tau,
        }
