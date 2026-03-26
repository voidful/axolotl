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

    # --- Hardness gate (h_t) ---
    rcca_tr_tau_p: float | None = (
        2.0  # hardness threshold: CE level above which suppression kicks in
    )
    rcca_tr_T_p: float | None = (
        1.0  # hardness sigmoid temperature
    )

    # --- Weight ---
    rcca_tr_w_min: float | None = (
        0.05  # weight floor for hard tokens to prevent zero gradients
    )


@dataclass
class RCCATRTrainingArgsMixin:
    """
    Additional training args for RCCA-TR.
    """

    rcca_tr_tau_p: float | None = 2.0
    rcca_tr_T_p: float | None = 1.0
    rcca_tr_w_min: float | None = 0.05
