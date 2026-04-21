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
Trainer for the Learning without Forgetting baseline.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .loss import compute_lwf_loss

LOG = get_logger(__name__)


class AxolotlLWFTrainer(AxolotlTrainer):
    """
    LwF trainer with a frozen teacher initialized from the base model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = None

    def _ensure_teacher_loaded(self, model):
        if self.teacher_model is not None:
            return

        teacher_name = getattr(self.args, "lwf_teacher_model", None)
        if not teacher_name:
            teacher_name = self.axolotl_cfg.base_model if self.axolotl_cfg else None
        if not teacher_name:
            raise ValueError("LwF requires `lwf_teacher_model` or `base_model`")

        dtype = next(model.parameters()).dtype
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        teacher.to(next(model.parameters()).device)
        teacher.eval()
        teacher.requires_grad_(False)
        self.teacher_model = teacher
        LOG.info("Loaded LwF teacher model from %s", teacher_name)

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

        self._ensure_teacher_loaded(model)
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)

        loss, _ = compute_lwf_loss(
            student_logits=outputs.logits,
            teacher_logits=teacher_outputs.logits,
            labels=labels,
            ce_alpha=getattr(self.args, "lwf_ce_alpha", 1.0) or 1.0,
            alpha=getattr(self.args, "lwf_alpha", 1.0) or 1.0,
            temperature=getattr(self.args, "lwf_temperature", 2.0) or 2.0,
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
