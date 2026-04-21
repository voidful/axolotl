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
drift-loss trainer.

Default mode uses a scalar EMA reference over recent target log-probabilities.
An appendix-style token-wise prior reference is also supported when the batch
contains a `reference_target_logp` tensor (or a custom key).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .drift import DriftMeanBuffer
from .loss import compute_drift_focal_loss

LOG = get_logger(__name__)


class AxolotlDriftTrainer(AxolotlTrainer):
    """
    drift-loss trainer with EMA or token-wise prior references.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ema_decay = getattr(self.args, "drift_ema_decay", None)
        if ema_decay is None:
            ema_decay = 0.999

        reference_mode = getattr(self.args, "drift_reference_mode", None) or "ema"
        drift_gamma = getattr(self.args, "drift_gamma", None)
        if drift_gamma is None:
            drift_gamma = 2.0
        detach_weights = getattr(self.args, "drift_detach_weights", True)

        self.drift_buffer = DriftMeanBuffer(decay=ema_decay)

        LOG.info(
            "Drift trainer initialized "
            "(reference_mode=%s, drift_decay=%.4f, drift_gamma=%.2f, detach_weights=%s)",
            reference_mode,
            ema_decay,
            drift_gamma,
            detach_weights,
        )

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Compute drift-loss during training and standard CE during eval.
        """
        if (
            self.args.sample_packing
            and "attention_mask" in inputs
            and "position_ids" in inputs
        ):
            del inputs["attention_mask"]

        labels = inputs.pop("labels", None)
        reference_key = getattr(self.args, "drift_reference_key", None)
        if not reference_key:
            reference_key = "reference_target_logp"
        reference_target_logp = inputs.pop(reference_key, None)

        if num_items_in_batch is None and labels is not None:
            num_items_in_batch = (labels != -100).sum().item()

        import accelerate.utils.operations as accel_ops
        original_convert = accel_ops.convert_to_fp32
        accel_ops.convert_to_fp32 = lambda x: x
        try:
            outputs = model(**inputs)
        finally:
            accel_ops.convert_to_fp32 = original_convert

        if not model.training:
            if labels is not None:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = outputs.loss if outputs.loss is not None else outputs[0]
            return (loss, outputs) if return_outputs else loss

        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        reference_mode = getattr(self.args, "drift_reference_mode", None) or "ema"
        if reference_mode == "ema":
            reference = self.drift_buffer.get_reference()
        elif reference_mode == "prior":
            if reference_target_logp is None:
                raise ValueError(
                    "drift_reference_mode='prior' requires a token-wise "
                    f"reference tensor in inputs['{reference_key}']"
                )
            reference = reference_target_logp
        else:
            raise ValueError(
                f"Unsupported drift_reference_mode={reference_mode!r}; "
                "expected 'ema' or 'prior'"
            )

        drift_gamma = getattr(self.args, "drift_gamma", None)
        if drift_gamma is None:
            drift_gamma = 2.0
        drift_eps = getattr(self.args, "drift_eps", None)
        if drift_eps is None:
            drift_eps = 1e-6

        loss, stats = compute_drift_focal_loss(
            active_logits=outputs.logits,
            labels=labels,
            reference_target_logp=reference,
            gamma=drift_gamma,
            eps=drift_eps,
            detach_weights=getattr(self.args, "drift_detach_weights", True),
            num_items_in_batch=num_items_in_batch,
        )

        if reference_mode == "ema":
            self.drift_buffer.step(
                active_target_logp=stats["active_target_logp"],  # type: ignore[arg-type]
                valid_mask=stats["shift_mask"],  # type: ignore[arg-type]
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
        Use standard CE loss during eval for an apples-to-apples eval metric.
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
