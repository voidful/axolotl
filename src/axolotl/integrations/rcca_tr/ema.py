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
EMA (Exponential Moving Average) model utilities for RCCA-TR.

The EMA model provides a slow-moving evidence estimator. When new data
consistently gives answers that differ from the frozen prior, the EMA model
gradually drifts away from p_0, signaling that the prior may be outdated.
"""

import copy

import torch
from torch import nn


def create_ema_model(model: nn.Module) -> nn.Module:
    """
    Create an EMA copy of the model.

    The EMA model is a deep copy with all gradients disabled and set to eval mode.
    It is used as a slow-moving evidence estimator that accumulates information
    from the training stream without backpropagation.

    Args:
        model: The active model to copy from.

    Returns:
        A deep copy of the model with gradients disabled.
    """
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    return ema_model


def create_frozen_model(model: nn.Module) -> nn.Module:
    """
    Create a frozen copy of the base model.

    The frozen model preserves the original prior p_0 throughout training.
    It is never updated and serves as the reference for both conflict scores
    and KL divergence constraints in the trust-region objective.

    Args:
        model: The base model to freeze.

    Returns:
        A deep copy of the model with gradients disabled.
    """
    frozen_model = copy.deepcopy(model)
    frozen_model.requires_grad_(False)
    frozen_model.eval()
    return frozen_model


@torch.no_grad()
def update_ema_model(
    ema_model: nn.Module, active_model: nn.Module, decay: float = 0.999
) -> None:
    """
    Update EMA model parameters with exponential moving average.

    For each parameter θ_ema:
        θ_ema = decay * θ_ema + (1 - decay) * θ_active

    Args:
        ema_model: The EMA model to update.
        active_model: The current active (training) model.
        decay: EMA decay rate. Higher = slower update. Default: 0.999.
    """
    for ema_param, active_param in zip(
        ema_model.parameters(), active_model.parameters()
    ):
        ema_param.data.mul_(decay).add_(active_param.data, alpha=1.0 - decay)
