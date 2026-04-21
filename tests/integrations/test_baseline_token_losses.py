from __future__ import annotations

import torch
import torch.nn.functional as F

from axolotl.integrations.entropy_focus.loss import compute_entropy_focus_loss
from axolotl.integrations.focal.loss import compute_focal_loss


def _ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class TestFocalLoss:
    def test_gamma_zero_matches_cross_entropy(self):
        logits = torch.tensor(
            [[[3.0, 0.5], [0.1, 2.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0, 1, 0]])

        loss, _ = compute_focal_loss(
            active_logits=logits,
            labels=labels,
            gamma=0.0,
        )

        assert torch.allclose(loss, _ce_loss(logits, labels), atol=1e-6)

    def test_harder_token_gets_higher_weight(self):
        logits = torch.tensor(
            [[[4.0, 0.1], [0.2, 1.5], [1.5, 0.2]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0, 1, 0]])

        _, stats = compute_focal_loss(
            active_logits=logits,
            labels=labels,
            gamma=2.0,
        )

        weights = stats["focal_weight"][stats["shift_mask"]]
        assert weights.shape[0] == 2
        assert weights[0] > weights[1]


class TestEntropyFocusLoss:
    def test_high_and_low_select_disjoint_valid_tokens(self):
        logits = torch.tensor(
            [
                [
                    [3.0, 0.1, 0.1],
                    [1.0, 1.0, 1.0],
                    [2.8, 0.1, 0.1],
                    [1.2, 1.1, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0, 1, 0, 1]])

        _, high = compute_entropy_focus_loss(
            active_logits=logits,
            labels=labels,
            mode="high",
        )
        _, low = compute_entropy_focus_loss(
            active_logits=logits,
            labels=labels,
            mode="low",
        )

        shift_mask = high["shift_mask"]
        selected_high = high["selected_mask"][shift_mask]
        selected_low = low["selected_mask"][shift_mask]

        assert (selected_high & selected_low).sum().item() == 0
        assert selected_high.sum().item() >= 1
        assert selected_low.sum().item() >= 1

    def test_high_entropy_loss_uses_only_selected_tokens(self):
        logits = torch.tensor(
            [
                [
                    [4.0, 0.1],
                    [1.0, 1.0],
                    [4.0, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0, 1, 0]])

        loss, stats = compute_entropy_focus_loss(
            active_logits=logits,
            labels=labels,
            mode="high",
        )

        ce_t = stats["ce_t"]
        selected = stats["selected_mask"].float()
        expected = (ce_t * selected).sum() / selected.sum().clamp(min=1.0)
        assert torch.allclose(loss, expected, atol=1e-6)
