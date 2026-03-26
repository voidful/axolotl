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
Hardness Trainer: Self-Paced Hardness Suppression.

Token weights are determined solely by the active model's cross-entropy.

Single gate per token:
  Hardness gate h_t  — is this token hard enough to warrant suppression?

Final weight: w_t = w_min + (1-w_min) · (1 - h_t)
Loss: L = Σ w_t · CE_t / |V|
"""

import torch
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .loss import compute_weighted_ce_loss

LOG = get_logger(__name__)


class AxolotlHardnessTrainer(AxolotlTrainer):
    """
    Hardness Trainer: Self-Paced Hardness Suppression.

    Memory footprint: only the active model (1× model size).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LOG.info("Initializing Hardness trainer...")

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Compute the Hardness weighted CE loss.

        During training: uses self-paced hardness suppression.
        During eval: falls back to standard CE loss for proper eval_loss metric.
        """
        # Handle sample packing
        if (
            self.args.sample_packing
            and "attention_mask" in inputs
            and "position_ids" in inputs
        ):
            del inputs["attention_mask"]

        if num_items_in_batch is None and "labels" in inputs:
            num_items_in_batch = (inputs["labels"] != -100).sum().item()

        # Forward pass through active model
        outputs = model(**inputs)

        # During eval, use standard CE loss
        if not model.training:
            loss = outputs.loss if outputs.loss is not None else outputs[0]
            return (loss, outputs) if return_outputs else loss

        active_logits = outputs.logits  # (B, T, V)

        labels = inputs.get("labels", None)
        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        # Compute unified weighted CE loss
        loss, intermediates = compute_weighted_ce_loss(
            active_logits=active_logits,
            labels=labels,
            tau_p=getattr(self.args, "hardness_tau_p", 2.0) or 2.0,
            T_p=getattr(self.args, "hardness_T_p", 1.0) or 1.0,
            w_min=getattr(self.args, "hardness_w_min", 0.05) or 0.05,
            num_items_in_batch=num_items_in_batch,
        )

        return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Override prediction_step to use standard CE loss during eval.
        """
        model_inputs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "position_ids", "labels")
        }

        with torch.no_grad():
            model.eval()
            try:
                outputs = model(**model_inputs)
            finally:
                model.train()

        loss = outputs.loss if outputs.loss is not None else outputs[0]

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = outputs.logits if hasattr(outputs, "logits") else None
        labels = inputs.get("labels", None)
        return (
            loss.detach(),
            logits.detach() if logits is not None else None,
            labels.detach() if labels is not None else None,
        )
