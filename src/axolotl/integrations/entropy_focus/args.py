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
Plugin args for entropy-focus baselines.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


class EntropyFocusArgs(BaseModel):
    """
    Input args for entropy-focus fine-tuning.
    """

    entropy_focus_trainer: bool | None = None
    entropy_focus_mode: Literal["high", "low"] | None = "high"
    entropy_focus_eps: float | None = 1e-6


@dataclass
class EntropyFocusTrainingArgsMixin:
    """
    Additional training args for entropy-focus baselines.
    """

    entropy_focus_mode: Literal["high", "low"] | None = "high"
    entropy_focus_eps: float | None = 1e-6
