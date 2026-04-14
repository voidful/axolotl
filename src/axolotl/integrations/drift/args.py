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
Plugin args for Drift variant.
"""

from dataclasses import dataclass

from pydantic import BaseModel


class DriftArgs(BaseModel):
    """
    Input args for Drift fine-tuning.
    """

    drift_trainer: bool | None = None

    # --- Reliability score hyperparameters ---
    drift_reliability_beta: float | None = (
        0.5  # balance between stability and evidence reliability
    )
    drift_reliability_tau: float | None = (
        1.0  # temperature for sigmoid mapping of reliability score
    )

    # --- Trust-region hyperparameters ---
    drift_epsilon_min: float | None = 0.01
    drift_epsilon_max: float | None = 1.0
    drift_kl_lambda: float | None = 4.0  # modulation strength for clean tokens
    drift_anchor_weight: float | None = 0.1  # near-zero gradient floor for noisy tokens
    drift_use_smooth_objective: bool | None = True

    # --- Drift buffer ---
    drift_per_sample: bool | None = False  # Per-sample drift (avg CE per sample)
    drift_ema_decay: float | None = 0.99  # faster adaptation
    drift_gamma: float | None = 3.0  # sensitive to CE deviation


@dataclass
class DriftTrainingArgsMixin:
    """
    Additional training args for Drift.
    """

    drift_reliability_beta: float | None = 0.5
    drift_reliability_tau: float | None = 1.0
    drift_epsilon_min: float | None = 0.01
    drift_epsilon_max: float | None = 1.0
    drift_kl_lambda: float | None = 4.0
    drift_anchor_weight: float | None = 0.1
    drift_use_smooth_objective: bool | None = True
    drift_per_sample: bool | None = False
    drift_ema_decay: float | None = 0.99
    drift_gamma: float | None = 3.0
