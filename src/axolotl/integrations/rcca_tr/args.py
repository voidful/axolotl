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
Plugin args for RCCA-TR (Reliability-Calibrated, Curriculum-Aware Trust Region).

Four explicit modes:
  - ce:         Standard cross-entropy (baseline).
  - hardness:   Self-paced hardness weighting only.
  - drift_only: Self-paced + drift regularization (main method).
  - drift:      Full drift + hardness + anchor (ablation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel


class RCCATRArgs(BaseModel):
    """
    Input args for RCCA-TR fine-tuning.
    """

    rcca_tr_trainer: Optional[bool] = None

    # --- Method mode ---
    rcca_tr_mode: Optional[Literal["ce", "hardness", "drift_only", "drift"]] = (
        "drift_only"
    )

    # --- Self-paced (hardness) hyperparameters ---
    rcca_tr_tau_p: Optional[float] = 2.0   # temperature for hardness sigmoid
    rcca_tr_T_p: Optional[float] = 1.0     # hardness threshold (unused in drift_only)

    # --- Drift hyperparameters ---
    rcca_tr_tau_delta: Optional[float] = 0.8  # temperature for drift score sigmoid
    rcca_tr_T_delta: Optional[float] = 1.0    # drift threshold (unused in drift_only)

    # --- Weighting ---
    rcca_tr_w_min: Optional[float] = 0.05  # minimum token weight floor
    rcca_tr_beta: Optional[float] = 0.5    # balance: β·s_t + (1-β)·r_t

    # --- Drift buffer ---
    rcca_tr_self_tau: Optional[float] = 1.0    # self-paced score temperature
    rcca_tr_drift_tau: Optional[float] = 1.0   # drift score temperature
    rcca_tr_drift_gamma: Optional[float] = 1.0 # drift scaling (legacy compat)
    rcca_tr_ema_decay: Optional[float] = 0.999 # EMA decay for running mean

    # --- Legacy drift mode (ablation) ---
    rcca_tr_kl_lambda: Optional[float] = 4.0
    rcca_tr_anchor_weight: Optional[float] = 0.1
    rcca_tr_reliability_tau: Optional[float] = 1.0


@dataclass
class RCCATRTrainingArgsMixin:
    """
    Additional training args injected into HuggingFace TrainingArguments.
    """

    rcca_tr_mode: Optional[str] = "drift_only"
    rcca_tr_tau_p: Optional[float] = 2.0
    rcca_tr_T_p: Optional[float] = 1.0
    rcca_tr_tau_delta: Optional[float] = 0.8
    rcca_tr_T_delta: Optional[float] = 1.0
    rcca_tr_w_min: Optional[float] = 0.05
    rcca_tr_beta: Optional[float] = 0.5
    rcca_tr_self_tau: Optional[float] = 1.0
    rcca_tr_drift_tau: Optional[float] = 1.0
    rcca_tr_drift_gamma: Optional[float] = 1.0
    rcca_tr_ema_decay: Optional[float] = 0.999
    rcca_tr_kl_lambda: Optional[float] = 4.0
    rcca_tr_anchor_weight: Optional[float] = 0.1
    rcca_tr_reliability_tau: Optional[float] = 1.0
