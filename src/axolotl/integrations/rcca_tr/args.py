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
Plugin args for RCCA-TR (no-cache variant).
"""

from dataclasses import dataclass

from pydantic import BaseModel


class RCCATRArgs(BaseModel):
    """
    Input args for RCCA-TR fine-tuning (no-cache variant).
    """

    rcca_tr_trainer: bool | None = None  # whether to use RCCA-TR trainer

    # --- Reliability score hyperparameters ---
    rcca_tr_reliability_beta: float | None = (
        0.5  # balance between stability and evidence reliability
    )
    rcca_tr_reliability_tau: float | None = (
        1.0  # temperature for sigmoid mapping of reliability score
    )

    # --- Trust-region hyperparameters ---
    rcca_tr_epsilon_min: float | None = 0.01  # minimum trust-region radius
    rcca_tr_epsilon_max: float | None = 1.0  # maximum trust-region radius
    rcca_tr_kl_lambda: float | None = 1.0  # Lagrange multiplier for KL penalty
    rcca_tr_use_smooth_objective: bool | None = (
        True  # use smooth g(r_t)*KL vs hinge max(0, KL - epsilon_t)
    )

    # --- Drift buffer (replaces EMA model) ---
    rcca_tr_ema_decay: float | None = 0.999  # decay rate for drift buffer
    rcca_tr_drift_gamma: float | None = (
        1.0  # scaling factor for drift → reliability mapping
    )


@dataclass
class RCCATRTrainingArgsMixin:
    """
    Additional training args for RCCA-TR (no-cache variant).
    """

    rcca_tr_reliability_beta: float | None = 0.5
    rcca_tr_reliability_tau: float | None = 1.0
    rcca_tr_epsilon_min: float | None = 0.01
    rcca_tr_epsilon_max: float | None = 1.0
    rcca_tr_kl_lambda: float | None = 1.0
    rcca_tr_use_smooth_objective: bool | None = True
    rcca_tr_ema_decay: float | None = 0.999
    rcca_tr_drift_gamma: float | None = 1.0
