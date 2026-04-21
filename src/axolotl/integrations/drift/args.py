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
Plugin args for drift-loss.

The new method centers on a single focal-style tuning parameter, `drift_gamma`,
while keeping the reference path explicit:
  - `ema`: scalar running-mean reference (paper default)
  - `prior`: token-wise frozen-prior reference (appendix-style)

Legacy trust-region fields are retained for config compatibility but ignored by
the drift-loss trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


class DriftArgs(BaseModel):
    """
    Input args for Drift fine-tuning.
    """

    drift_trainer: bool | None = None

    # --- drift-loss core ---
    drift_reference_mode: Literal["ema", "prior"] | None = "ema"
    drift_reference_key: str | None = "reference_target_logp"
    drift_ema_decay: float | None = 0.999
    drift_gamma: float | None = 2.0
    drift_detach_weights: bool | None = True
    drift_eps: float | None = 1e-6

    # --- Legacy trust-region args (ignored; kept for compatibility) ---
    drift_reliability_beta: float | None = (
        0.5
    )
    drift_reliability_tau: float | None = (
        1.0
    )
    drift_epsilon_min: float | None = 0.01
    drift_epsilon_max: float | None = 1.0
    drift_kl_lambda: float | None = 4.0
    drift_anchor_weight: float | None = 0.1
    drift_use_smooth_objective: bool | None = True
    drift_per_sample: bool | None = False


@dataclass
class DriftTrainingArgsMixin:
    """
    Additional training args for Drift.
    """

    drift_reference_mode: Literal["ema", "prior"] | None = "ema"
    drift_reference_key: str | None = "reference_target_logp"
    drift_ema_decay: float | None = 0.999
    drift_gamma: float | None = 2.0
    drift_detach_weights: bool | None = True
    drift_eps: float | None = 1e-6

    drift_reliability_beta: float | None = 0.5
    drift_reliability_tau: float | None = 1.0
    drift_epsilon_min: float | None = 0.01
    drift_epsilon_max: float | None = 1.0
    drift_kl_lambda: float | None = 4.0
    drift_anchor_weight: float | None = 0.1
    drift_use_smooth_objective: bool | None = True
    drift_per_sample: bool | None = False
