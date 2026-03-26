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
Plugin args for Hardness variant.
"""

from dataclasses import dataclass

from pydantic import BaseModel


class HardnessArgs(BaseModel):
    """
    Input args for Hardness fine-tuning.
    """

    hardness_trainer: bool | None = None

    # --- Hardness gate (h_t) ---
    hardness_tau_p: float | None = (
        2.0  # hardness threshold: CE level above which suppression kicks in
    )
    hardness_T_p: float | None = (
        1.0  # hardness sigmoid temperature
    )

    # --- Weight ---
    hardness_w_min: float | None = (
        0.05  # weight floor for hard tokens to prevent zero gradients
    )


@dataclass
class HardnessTrainingArgsMixin:
    """
    Additional training args for Hardness.
    """

    hardness_tau_p: float | None = 2.0
    hardness_T_p: float | None = 1.0
    hardness_w_min: float | None = 0.05
