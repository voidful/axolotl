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
Plugin init for RCCA-TR.

The current research path is full fine-tuning with base-aware token role
triage and module-routed gradients. ``module_aware_retention`` and
``fullft_module_aware_retention`` use a frozen reference model to assign token
roles, then route acquisition-heavy loss into attention and retention-heavy
loss into MLP/other parameters.
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import RCCATRArgs as RCCATRArgs

LOG = get_logger(__name__)


class RCCATRPlugin(BasePlugin):
    """
    Plugin for RCCA-TR.
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
            "rcca_tr_tau_old": cfg.rcca_tr_tau_old,
            "rcca_tr_tau_new": cfg.rcca_tr_tau_new,
            "rcca_tr_tau_noise": cfg.rcca_tr_tau_noise,
            "rcca_tr_gate_temperature": cfg.rcca_tr_gate_temperature,
            "rcca_tr_old_quantile": cfg.rcca_tr_old_quantile,
            "rcca_tr_new_quantile": cfg.rcca_tr_new_quantile,
            "rcca_tr_noise_quantile": cfg.rcca_tr_noise_quantile,
            "rcca_tr_stm_keep_ratio": cfg.rcca_tr_stm_keep_ratio,
            "rcca_tr_lambda_acquire": cfg.rcca_tr_lambda_acquire,
            "rcca_tr_mu_noise": cfg.rcca_tr_mu_noise,
            "rcca_tr_rho_retention": cfg.rcca_tr_rho_retention,
            "rcca_tr_triage_w_floor": cfg.rcca_tr_triage_w_floor,
            "rcca_tr_triage_w_max": cfg.rcca_tr_triage_w_max,
            "rcca_tr_kl_beta": cfg.rcca_tr_kl_beta,
            "rcca_tr_kl_chunk_size": cfg.rcca_tr_kl_chunk_size,
            "rcca_tr_reference_model": cfg.rcca_tr_reference_model,
            "rcca_tr_attn_lambda_acquire": cfg.rcca_tr_attn_lambda_acquire,
            "rcca_tr_attn_mu_noise": cfg.rcca_tr_attn_mu_noise,
            "rcca_tr_attn_rho_retention": cfg.rcca_tr_attn_rho_retention,
            "rcca_tr_attn_kl_beta": cfg.rcca_tr_attn_kl_beta,
            "rcca_tr_mlp_lambda_acquire": cfg.rcca_tr_mlp_lambda_acquire,
            "rcca_tr_mlp_mu_noise": cfg.rcca_tr_mlp_mu_noise,
            "rcca_tr_mlp_rho_retention": cfg.rcca_tr_mlp_rho_retention,
            "rcca_tr_mlp_kl_beta": cfg.rcca_tr_mlp_kl_beta,
            "rcca_tr_other_lambda_acquire": cfg.rcca_tr_other_lambda_acquire,
            "rcca_tr_other_mu_noise": cfg.rcca_tr_other_mu_noise,
            "rcca_tr_other_rho_retention": cfg.rcca_tr_other_rho_retention,
            "rcca_tr_other_kl_beta": cfg.rcca_tr_other_kl_beta,
        }
