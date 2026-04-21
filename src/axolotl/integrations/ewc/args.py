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
Plugin args for EWC regularization.
"""

from dataclasses import dataclass

from pydantic import BaseModel


class EWCArgs(BaseModel):
    """
    Input args for EWC fine-tuning.
    """

    ewc_trainer: bool | None = None
    ewc_lambda: float | None = 1e-4
    ewc_fisher_n_batches: int | None = 32


@dataclass
class EWCTrainingArgsMixin:
    """
    Additional training args for EWC.
    """

    ewc_lambda: float | None = 1e-4
    ewc_fisher_n_batches: int | None = 32
