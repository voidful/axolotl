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
Plugin args for Drift-Only variant.
"""

from dataclasses import dataclass

from pydantic import BaseModel


class DriftOnlyArgs(BaseModel):
    """
    Input args for Drift-Only fine-tuning.
    """

    drift_only_trainer: bool | None = None

    # --- Reliability score hyperparameters ---
    drift_only_reliability_beta: float | None = (
        0.5  # balance between stability and evidence reliability
    )
    drift_only_reliability_tau: float | None = (
        1.0  # temperature for sigmoid mapping of reliability score
    )

    # --- Trust-region hyperparameters ---
    drift_only_epsilon_min: float | None = 0.01
    drift_only_epsilon_max: float | None = 1.0
    drift_only_kl_lambda: float | None = 1.0
    drift_only_use_smooth_objective: bool | None = True

    # --- Drift buffer ---
    drift_only_ema_decay: float | None = 0.999
    drift_only_gamma: float | None = 1.0


@dataclass
class DriftOnlyTrainingArgsMixin:
    """
    Additional training args for Drift-Only.
    """

    drift_only_reliability_beta: float | None = 0.5
    drift_only_reliability_tau: float | None = 1.0
    drift_only_epsilon_min: float | None = 0.01
    drift_only_epsilon_max: float | None = 1.0
    drift_only_kl_lambda: float | None = 1.0
    drift_only_use_smooth_objective: bool | None = True
    drift_only_ema_decay: float | None = 0.999
    drift_only_gamma: float | None = 1.0
