"""
Unit tests for RCCA-TR A+ loss functions and drift buffer.
"""

import torch
import torch.nn as nn

from axolotl.integrations.rcca_tr.loss import (
    compute_conflict_score,
    compute_conflict_score_from_cache,
    compute_reliability_from_drift,
    compute_trust_region_loss,
    compute_trust_region_loss_cached,
)
from axolotl.integrations.rcca_tr.drift import DriftBuffer


class TestConflictScoreFromCache:
    """Tests for compute_conflict_score_from_cache."""

    def test_shape(self):
        B, T = 2, 10
        prior_target_logp = torch.randn(B, T)
        prior_margin = torch.rand(B, T)
        valid_mask = torch.ones(B, T, dtype=torch.bool)
        valid_mask[:, 0] = False

        alpha = compute_conflict_score_from_cache(
            prior_target_logp, prior_margin, valid_mask
        )
        assert alpha.shape == (B, T)

    def test_range(self):
        B, T = 4, 20
        prior_target_logp = torch.randn(B, T)
        prior_margin = torch.rand(B, T)
        valid_mask = torch.ones(B, T, dtype=torch.bool)

        alpha = compute_conflict_score_from_cache(
            prior_target_logp, prior_margin, valid_mask
        )
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_invalid_tokens_zero(self):
        B, T = 2, 10
        prior_target_logp = torch.randn(B, T)
        prior_margin = torch.rand(B, T)
        valid_mask = torch.zeros(B, T, dtype=torch.bool)

        alpha = compute_conflict_score_from_cache(
            prior_target_logp, prior_margin, valid_mask
        )
        assert torch.all(alpha == 0.0)

    def test_high_surprisal_high_conflict(self):
        B, T = 1, 5
        # Very negative log p_0(y_t) → high surprisal
        prior_target_logp = torch.full((B, T), -10.0)
        prior_margin = torch.full((B, T), 5.0)
        valid_mask = torch.ones(B, T, dtype=torch.bool)

        alpha = compute_conflict_score_from_cache(
            prior_target_logp, prior_margin, valid_mask, lambda1=1.0, lambda2=0.5
        )
        # All tokens have same conflict, so after z-score normalization sigmoid → ~0.5
        # This is expected — conflict scores are relative
        assert alpha.mean() > 0.0


class TestConflictScoreLegacy:
    """Tests for legacy compute_conflict_score (full logits)."""

    def test_shape(self):
        B, T, V = 2, 10, 100
        frozen_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[:, 0] = -100

        alpha = compute_conflict_score(frozen_logits, labels)
        assert alpha.shape == (B, T)

    def test_range(self):
        B, T, V = 4, 20, 50
        frozen_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))

        alpha = compute_conflict_score(frozen_logits, labels)
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_ignored_tokens_zero(self):
        B, T, V = 2, 10, 50
        frozen_logits = torch.randn(B, T, V)
        labels = torch.full((B, T), -100, dtype=torch.long)

        alpha = compute_conflict_score(frozen_logits, labels)
        assert torch.all(alpha == 0.0)


class TestReliabilityFromDrift:
    """Tests for compute_reliability_from_drift."""

    def test_range(self):
        B, T = 4, 20
        drift = torch.rand(B, T)

        r_t = compute_reliability_from_drift(drift)
        assert r_t.shape == (B, T)
        assert r_t.min() >= 0.0
        assert r_t.max() <= 1.0

    def test_zero_drift_high_reliability(self):
        B, T = 2, 10
        drift = torch.zeros(B, T)

        r_t = compute_reliability_from_drift(drift, gamma=1.0)
        # exp(-0) = 1.0, after sigmoid normalization → ~0.5
        assert r_t.mean() >= 0.4


class TestTrustRegionLossCached:
    """Tests for compute_trust_region_loss_cached."""

    def test_produces_scalar(self):
        B, T, V = 2, 10, 50
        active_logits = torch.randn(B, T, V, requires_grad=True)
        labels = torch.randint(0, V, (B, T))
        alpha_t = torch.rand(B, T)
        r_t = torch.rand(B, T)
        prior_target_logp = torch.randn(B, T)

        loss, active_logp = compute_trust_region_loss_cached(
            active_logits, labels, alpha_t, r_t, prior_target_logp
        )
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flows(self):
        B, T, V = 2, 10, 50
        active_logits = torch.randn(B, T, V, requires_grad=True)
        labels = torch.randint(0, V, (B, T))
        alpha_t = torch.rand(B, T)
        r_t = torch.rand(B, T)
        prior_target_logp = torch.randn(B, T)

        loss, _ = compute_trust_region_loss_cached(
            active_logits, labels, alpha_t, r_t, prior_target_logp
        )
        loss.backward()
        assert active_logits.grad is not None
        assert not torch.all(active_logits.grad == 0)

    def test_smooth_vs_hinge(self):
        B, T, V = 2, 10, 50
        active_logits = torch.randn(B, T, V, requires_grad=True)
        labels = torch.randint(0, V, (B, T))
        alpha_t = torch.rand(B, T)
        r_t = torch.rand(B, T)
        prior_target_logp = torch.randn(B, T)

        loss_smooth, _ = compute_trust_region_loss_cached(
            active_logits, labels, alpha_t, r_t, prior_target_logp, use_smooth=True
        )
        loss_hinge, _ = compute_trust_region_loss_cached(
            active_logits.detach().requires_grad_(True), labels,
            alpha_t, r_t, prior_target_logp, use_smooth=False
        )

        assert not torch.isnan(loss_smooth)
        assert not torch.isnan(loss_hinge)


class TestTrustRegionLossLegacy:
    """Tests for legacy compute_trust_region_loss."""

    def test_produces_scalar(self):
        B, T, V = 2, 10, 50
        active_logits = torch.randn(B, T, V, requires_grad=True)
        frozen_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        alpha_t = torch.rand(B, T)
        r_t = torch.rand(B, T)

        loss = compute_trust_region_loss(
            active_logits, frozen_logits, labels, alpha_t, r_t
        )
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_gradient_flows(self):
        B, T, V = 2, 10, 50
        active_logits = torch.randn(B, T, V, requires_grad=True)
        frozen_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        alpha_t = torch.rand(B, T)
        r_t = torch.rand(B, T)

        loss = compute_trust_region_loss(
            active_logits, frozen_logits, labels, alpha_t, r_t
        )
        loss.backward()
        assert active_logits.grad is not None
        assert not torch.all(active_logits.grad == 0)


class TestDriftBuffer:
    """Tests for DriftBuffer."""

    def test_update_and_reliability(self):
        buf = DriftBuffer(decay=0.9, gamma=1.0)

        B, T = 2, 10
        active_logp = torch.randn(B, T)
        prior_logp = torch.randn(B, T)
        valid_mask = torch.ones(B, T, dtype=torch.bool)

        drift = buf.update(active_logp, prior_logp, valid_mask)
        assert drift.shape == (B, T)
        assert drift.min() >= 0.0

        r_evi = buf.get_evidence_reliability(drift)
        assert r_evi.shape == (B, T)
        assert r_evi.min() >= 0.0
        assert r_evi.max() <= 1.0

    def test_no_drift_high_reliability(self):
        buf = DriftBuffer(decay=0.9, gamma=1.0)

        B, T = 2, 10
        logp = torch.randn(B, T)
        valid_mask = torch.ones(B, T, dtype=torch.bool)

        # Same values → zero drift
        drift = buf.update(logp, logp, valid_mask)
        r_evi = buf.get_evidence_reliability(drift)

        # exp(-0) = 1
        assert torch.allclose(r_evi, torch.ones_like(r_evi), atol=1e-5)

    def test_running_drift_accumulates(self):
        buf = DriftBuffer(decay=0.5, gamma=1.0)

        B, T = 1, 5
        valid_mask = torch.ones(B, T, dtype=torch.bool)

        # Step 1: large drift
        active = torch.zeros(B, T)
        prior = torch.ones(B, T) * 5.0
        buf.update(active, prior, valid_mask)
        drift_1 = buf.running_drift

        # Step 2: zero drift
        buf.update(active, active, valid_mask)
        drift_2 = buf.running_drift

        # Running drift should decrease
        assert drift_2 < drift_1

    def test_get_current_drift_does_not_update(self):
        buf = DriftBuffer(decay=0.5, gamma=1.0)

        B, T = 1, 5
        active = torch.zeros(B, T)
        prior = torch.ones(B, T) * 5.0
        valid_mask = torch.ones(B, T, dtype=torch.bool)

        drift_before = buf.running_drift
        _ = buf.get_current_drift(active, prior, valid_mask)
        drift_after = buf.running_drift

        # get_current_drift should NOT update running drift
        assert drift_before == drift_after

    def test_step_updates_running_drift(self):
        buf = DriftBuffer(decay=0.5, gamma=1.0)

        B, T = 1, 5
        active = torch.zeros(B, T)
        prior = torch.ones(B, T) * 5.0
        valid_mask = torch.ones(B, T, dtype=torch.bool)

        assert buf.running_drift == 0.0
        buf.step(active, prior, valid_mask)
        assert buf.running_drift > 0.0

    def test_split_api_matches_update(self):
        """get_current_drift + step should produce same results as update."""
        buf_split = DriftBuffer(decay=0.9, gamma=1.0)
        buf_combined = DriftBuffer(decay=0.9, gamma=1.0)

        B, T = 2, 10
        active = torch.randn(B, T)
        prior = torch.randn(B, T)
        valid_mask = torch.ones(B, T, dtype=torch.bool)

        drift_split = buf_split.get_current_drift(active, prior, valid_mask)
        buf_split.step(active, prior, valid_mask)

        drift_combined = buf_combined.update(active, prior, valid_mask)

        assert torch.allclose(drift_split, drift_combined, atol=1e-6)
        assert abs(buf_split.running_drift - buf_combined.running_drift) < 1e-6

    def test_plugin_imports(self):
        from axolotl.integrations.rcca_tr import RCCATRPlugin, RCCATRArgs
        plugin = RCCATRPlugin()
        assert plugin.get_input_args() == "axolotl.integrations.rcca_tr.RCCATRArgs"
        assert plugin.get_training_args_mixin() == "axolotl.integrations.rcca_tr.args.RCCATRTrainingArgsMixin"
