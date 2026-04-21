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
Trainer for the EWC forgetting baseline.
"""

from __future__ import annotations

import torch
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.integrations.forgetting_common import (
    compute_next_token_ce_loss,
    iter_named_trainable_params,
    quadratic_reference_penalty,
    snapshot_trainable_params,
)


class AxolotlEWCTrainer(AxolotlTrainer):
    """
    EWC trainer using a diagonal Fisher estimate gathered at train start.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_params = snapshot_trainable_params(self.model)
        self.fisher_diagonal: dict[str, torch.Tensor] | None = None

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

        labels = inputs.get("labels", None)
        if num_items_in_batch is None and labels is not None:
            num_items_in_batch = (labels != -100).sum().item()

        outputs = model(**inputs)

        if not model.training:
            loss = outputs.loss if outputs.loss is not None else outputs[0]
            return (loss, outputs) if return_outputs else loss

        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        ce_loss, _ = compute_next_token_ce_loss(
            logits=outputs.logits,
            labels=labels,
            num_items_in_batch=num_items_in_batch,
        )

        if self.fisher_diagonal:
            penalty = quadratic_reference_penalty(
                named_params=dict(iter_named_trainable_params(model)),
                reference_params=self.reference_params,
                fisher_diagonal=self.fisher_diagonal,
            )
            loss = ce_loss + ((getattr(self.args, "ewc_lambda", 1e-4) or 1e-4) * penalty)
        else:
            loss = ce_loss

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
