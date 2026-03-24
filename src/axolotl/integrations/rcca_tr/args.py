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
Plugin args for RCCA-TR: Suppress-by-Default, Rescue-if-Useful.
"""

from dataclasses import dataclass

from pydantic import BaseModel


class RCCATRArgs(BaseModel):
    """
    Input args for RCCA-TR fine-tuning.
    """

    rcca_tr_trainer: bool | None = None  # whether to use RCCA-TR trainer

    # --- Challenge gate (α_t) ---
    rcca_tr_conflict_lambda1: float | None = (
        1.0  # weight for prior surprisal component
    )
    rcca_tr_conflict_lambda2: float | None = (
        0.5  # weight for margin-based conflict component
    )
    rcca_tr_conflict_tau: float | None = (
        1.0  # temperature for challenge gate sigmoid
    )

    # --- Hardness gate (h_t) ---
    rcca_tr_tau_p: float | None = (
        2.0  # hardness threshold: CE level above which suppression kicks in
    )
    rcca_tr_T_p: float | None = (
        1.0  # hardness sigmoid temperature
    )

    # --- Useful-hard gate (q_t) ---
    rcca_tr_tau_delta: float | None = (
        0.8  # improvement threshold for Δ_t⁺
    )
    rcca_tr_T_delta: float | None = (
        1.0  # improvement sigmoid temperature
    )

    # --- Weight ---
    rcca_tr_w_min: float | None = (
        0.05  # weight floor to prevent zero gradients
    )

    # --- Prior cache ---
    rcca_tr_prior_cache_path: str | None = (
        None  # path to pre-computed prior cache (.pt file)
    )


@dataclass
class RCCATRTrainingArgsMixin:
    """
    Additional training args for RCCA-TR.
    """

    rcca_tr_conflict_lambda1: float | None = 1.0
    rcca_tr_conflict_lambda2: float | None = 0.5
    rcca_tr_conflict_tau: float | None = 1.0
    rcca_tr_tau_p: float | None = 2.0
    rcca_tr_T_p: float | None = 1.0
    rcca_tr_tau_delta: float | None = 0.8
    rcca_tr_T_delta: float | None = 1.0
    rcca_tr_w_min: float | None = 0.05
    rcca_tr_prior_cache_path: str | None = None
