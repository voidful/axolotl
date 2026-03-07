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
Training callbacks for RCCA-TR.

EMAUpdateCallback: Updates the EMA model parameters after each training step
using exponential moving average of the active model.
"""

from transformers import TrainerCallback

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class EMAUpdateCallback(TrainerCallback):
    """
    Callback to update the EMA model after each training step.

    The EMA model serves as a slow-moving evidence estimator in RCCA-TR.
    It accumulates training signal over time and is used to detect when
    the frozen prior has become outdated.
    """

    def __init__(self, ema_decay: float = 0.999):
        super().__init__()
        self.ema_decay = ema_decay

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Update EMA model parameters after each training step."""
        from .ema import update_ema_model

        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return

        if not hasattr(trainer, "ema_model") or trainer.ema_model is None:
            return

        # Get the underlying model (unwrap DDP/FSDP if needed)
        active_model = model
        if hasattr(active_model, "module"):
            active_model = active_model.module

        update_ema_model(trainer.ema_model, active_model, self.ema_decay)
