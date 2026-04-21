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
Plugin args for Learning without Forgetting.
"""

from dataclasses import dataclass

from pydantic import BaseModel


class LWFArgs(BaseModel):
    """
    Input args for Learning without Forgetting.
    """

    lwf_trainer: bool | None = None
    lwf_teacher_model: str | None = None
    lwf_ce_alpha: float | None = 1.0
    lwf_alpha: float | None = 1.0
    lwf_temperature: float | None = 2.0


@dataclass
class LWFTrainingArgsMixin:
    """
    Additional training args for Learning without Forgetting.
    """

    lwf_teacher_model: str | None = None
    lwf_ce_alpha: float | None = 1.0
    lwf_alpha: float | None = 1.0
    lwf_temperature: float | None = 2.0
