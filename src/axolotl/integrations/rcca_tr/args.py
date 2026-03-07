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
Plugin args for Reliability-Calibrated Conflict-Aware Trust-Region Fine-Tuning (RCCA-TR).
"""

from dataclasses import dataclass

from pydantic import BaseModel


class RCCATRArgs(BaseModel):
    """
    Input args for RCCA-TR fine-tuning.
    """

    rcca_tr_trainer: bool | None = None  # whether to use RCCA-TR trainer

    # --- Conflict score hyperparameters ---
    rcca_tr_conflict_lambda1: float | None = (
        1.0  # weight for surprisal component in conflict score
    )
    rcca_tr_conflict_lambda2: float | None = (
        0.5  # weight for margin-based conflict component
    )
    rcca_tr_conflict_tau: float | None = (
        1.0  # temperature for sigmoid mapping of conflict score
    )

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

    # --- EMA hyperparameters ---
    rcca_tr_ema_decay: float | None = 0.999  # EMA decay rate for slow evidence model

    # --- Stability estimation ---
    rcca_tr_num_perturbations: int | None = (
        3  # number of dropout perturbations for stability
    )
    rcca_tr_stability_update_interval: int | None = (
        50  # steps between stability cache refreshes
    )


@dataclass
class RCCATRTrainingArgsMixin:
    """
    Additional training args for RCCA-TR.
    """

    rcca_tr_conflict_lambda1: float | None = 1.0
    rcca_tr_conflict_lambda2: float | None = 0.5
    rcca_tr_conflict_tau: float | None = 1.0
    rcca_tr_reliability_beta: float | None = 0.5
    rcca_tr_reliability_tau: float | None = 1.0
    rcca_tr_epsilon_min: float | None = 0.01
    rcca_tr_epsilon_max: float | None = 1.0
    rcca_tr_kl_lambda: float | None = 1.0
    rcca_tr_use_smooth_objective: bool | None = True
    rcca_tr_ema_decay: float | None = 0.999
    rcca_tr_num_perturbations: int | None = 3
    rcca_tr_stability_update_interval: int | None = 50
