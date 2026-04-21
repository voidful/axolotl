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
Trainer for the token-level focal-loss baseline.
"""

import torch
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer

from .loss import compute_focal_loss


class AxolotlFocalTrainer(AxolotlTrainer):
    """
    Token-level focal-loss trainer.
    """

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        if (
            self.args.sample_packing
            and "attention_mask" in inputs
            and "position_ids" in inputs
        ):
            del inputs["attention_mask"]

        if num_items_in_batch is None and "labels" in inputs:
            num_items_in_batch = (inputs["labels"] != -100).sum().item()

        outputs = model(**inputs)

        if not model.training:
            loss = outputs.loss if outputs.loss is not None else outputs[0]
            return (loss, outputs) if return_outputs else loss

        labels = inputs.get("labels", None)
        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        loss, _ = compute_focal_loss(
            active_logits=outputs.logits,
            labels=labels,
            gamma=getattr(self.args, "focal_gamma", 2.0) or 2.0,
            eps=getattr(self.args, "focal_eps", 1e-6) or 1e-6,
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
        model_inputs = {
            k: v
            for k, v in inputs.items()
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
