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
Plugin init for the EWC forgetting baseline.
"""

from transformers.trainer_callback import TrainerCallback

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.forgetting_common import (
    compute_next_token_ce_loss,
    iter_named_trainable_params,
)
from axolotl.utils.logging import get_logger

from .args import EWCArgs as EWCArgs

LOG = get_logger(__name__)


class EWCFisherCallback(TrainerCallback):
    """
    Estimate a diagonal Fisher matrix from a small prefix of train batches.
    """

    def __init__(self, trainer, max_batches: int = 32):
        super().__init__()
        self.trainer = trainer
        self.max_batches = max_batches

    def on_train_begin(self, args, state, control, **kwargs):
        if self.trainer.fisher_diagonal is not None:
            return control

        model = kwargs.get("model", self.trainer.model)
        was_training = model.training
        model.eval()

        fisher = {
            name: torch.zeros_like(param.detach(), dtype=torch.float32)
            for name, param in iter_named_trainable_params(model)
        }
        if not fisher:
            self.trainer.fisher_diagonal = {}
            return control

        dataloader = self.trainer.get_train_dataloader()
        num_batches = 0

        for batch in dataloader:
            if num_batches >= self.max_batches:
                break
            prepared = self.trainer._prepare_inputs(batch)
            if (
                self.trainer.args.sample_packing
                and "attention_mask" in prepared
                and "position_ids" in prepared
            ):
                del prepared["attention_mask"]
            labels = prepared.get("labels")
            if labels is None:
                continue

            model.zero_grad(set_to_none=True)
            outputs = model(**prepared)
            loss, _ = compute_next_token_ce_loss(outputs.logits, labels)
            loss.backward()

            for name, param in iter_named_trainable_params(model):
                if param.grad is not None:
                    fisher[name] += param.grad.detach().float().pow(2)

            num_batches += 1

        if num_batches > 0:
            for name in fisher:
                fisher[name] /= float(num_batches)

            total_sum = sum(t.sum() for t in fisher.values())
            total_count = sum(t.numel() for t in fisher.values())
            scale = (total_sum / max(float(total_count), 1.0)).clamp(min=1e-8)
            for name in fisher:
                fisher[name] /= scale

        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

        self.trainer.fisher_diagonal = fisher
        LOG.info(
            "EWC Fisher estimation complete using %d batch(es)",
            num_batches,
        )
        return control


class EWCPlugin(BasePlugin):
    """
    EWC regularization plugin.
    """

    def get_input_args(self):
        return "axolotl.integrations.ewc.EWCArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.ewc.args.EWCTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.ewc_trainer:
            from .trainer import AxolotlEWCTrainer

            return AxolotlEWCTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "ewc_lambda": cfg.ewc_lambda,
            "ewc_fisher_n_batches": cfg.ewc_fisher_n_batches,
        }

    def add_callbacks_post_trainer(self, cfg, trainer):
        if not cfg.ewc_trainer:
            return []
        return [EWCFisherCallback(trainer, max_batches=cfg.ewc_fisher_n_batches or 32)]
