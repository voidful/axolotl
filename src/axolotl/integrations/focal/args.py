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
Plugin args for the token-level focal-loss baseline.
"""

from dataclasses import dataclass

from pydantic import BaseModel


class FocalArgs(BaseModel):
    """
    Input args for focal-loss fine-tuning.
    """

    focal_trainer: bool | None = None
    focal_gamma: float | None = 2.0
    focal_eps: float | None = 1e-6


@dataclass
class FocalTrainingArgsMixin:
    """
    Additional training args for focal loss.
    """

    focal_gamma: float | None = 2.0
    focal_eps: float | None = 1e-6
