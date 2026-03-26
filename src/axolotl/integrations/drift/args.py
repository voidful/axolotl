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
Plugin args for Drift: Unified Risk Score.

Combines instantaneous hardness (CE_t) and historical drift (d_t)
into a single token risk score z_t, then maps to reliability r_t.
"""

from dataclasses import dataclass

from pydantic import BaseModel


class DriftArgs(BaseModel):
    """
    Input args for Drift fine-tuning (unified risk score).
    """

    drift_trainer: bool | None = None

    # --- EMA drift ---
    drift_rho: float | None = (
        0.999  # EMA decay for drift: d_t = ρ·d_{t-1} + (1-ρ)·CE_t
    )

    # --- Unified risk score ---
    drift_beta: float | None = (
        0.5  # balance: β·z_CE + (1-β)·z_drift (0=all drift, 1=all hardness)
    )
    drift_tau: float | None = (
        1.0  # temperature for reliability sigmoid: r_t = σ(-z_t/τ)
    )

    # --- Loss ---
    drift_lambda: float | None = (
        1.0  # overall loss multiplier: L = λ·r_t·CE_t
    )


@dataclass
class DriftTrainingArgsMixin:
    """
    Additional training args for Drift (unified risk score).
    """

    drift_rho: float | None = 0.999
    drift_beta: float | None = 0.5
    drift_tau: float | None = 1.0
    drift_lambda: float | None = 1.0
