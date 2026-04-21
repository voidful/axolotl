from __future__ import annotations

import torch

from axolotl.integrations.forgetting_common import quadratic_reference_penalty
from axolotl.integrations.lwf.loss import compute_lwf_loss


class TestQuadraticReferencePenalty:
    def test_zero_when_current_equals_reference(self):
        params = {
            "w": torch.tensor([1.0, 2.0]),
            "b": torch.tensor([0.5]),
        }
        refs = {
            "w": torch.tensor([1.0, 2.0]),
            "b": torch.tensor([0.5]),
        }

        penalty = quadratic_reference_penalty(params, refs)
        assert torch.allclose(penalty, torch.tensor(0.0))

    def test_fisher_weights_scale_penalty(self):
        params = {"w": torch.tensor([2.0, 0.0])}
        refs = {"w": torch.tensor([1.0, 0.0])}
        fisher = {"w": torch.tensor([3.0, 1.0])}

        penalty = quadratic_reference_penalty(params, refs, fisher_diagonal=fisher)
        assert torch.allclose(penalty, torch.tensor(1.5))


class TestLWFLoss:
    def test_kd_loss_zero_when_teacher_matches_student(self):
        logits = torch.tensor(
            [[[2.0, 0.0], [0.1, 1.5], [1.0, 0.5]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0, 1, 0]])

        loss, stats = compute_lwf_loss(
            student_logits=logits,
            teacher_logits=logits.clone(),
            labels=labels,
            ce_alpha=0.0,
            alpha=1.0,
            temperature=2.0,
        )

        assert torch.allclose(stats["kd_loss"], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_total_loss_includes_ce_and_kd(self):
        student = torch.tensor(
            [[[2.5, 0.2], [0.2, 1.8], [1.1, 0.2]]],
            dtype=torch.float32,
        )
        teacher = torch.tensor(
            [[[1.7, 0.9], [0.8, 1.0], [0.7, 0.6]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0, 1, 0]])

        loss, stats = compute_lwf_loss(
            student_logits=student,
            teacher_logits=teacher,
            labels=labels,
            ce_alpha=1.0,
            alpha=1.0,
            temperature=2.0,
        )

        assert loss.item() > 0.0
        assert stats["ce_loss"].item() > 0.0
        assert stats["kd_loss"].item() > 0.0
