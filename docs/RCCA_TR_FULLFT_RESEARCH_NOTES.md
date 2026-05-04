# RCCA-TR Full-FT Research Notes

## Working Thesis

The original Drift Loss was too coarse because it gated tokens from the
student's current confidence. That signal cannot distinguish old-known tokens,
new trustworthy tokens, and noisy or mislabeled tokens.

The stronger direction is token role triage under full fine-tuning:

- Preserve tokens: tokens the frozen base already models confidently.
- Acquire tokens: tokens the frozen base finds unfamiliar but are not extreme
  outliers.
- Ignore tokens: high-NLL outliers that are likely noise, formatting artifacts,
  hallucinations, or label errors.

The key claim is not simply token reweighting. The paper-worthy claim should be:

```text
Token-role-conditioned gradient routing for full-parameter fine-tuning:
different token roles induce different objectives, and those objectives are
routed to different transformer module families.
```

## Mechanistic Motivation

The method is motivated by a rough division of labor in transformer LMs:

- MLP/FFN layers behave like key-value memories and are strongly involved in
  factual associations.
- Attention heads are closer to retrieval, routing, topic extraction, and
  context selection.

This implies that old-known tokens should not freely rewrite MLP memory, while
new/domain tokens should still be learnable through acquisition-oriented
gradients.

## Token Role Gates

Let the frozen base model produce `base_nll_t`.

```text
R_t = sigmoid((tau_old - base_nll_t) / T)
N_t = sigmoid((base_nll_t - tau_noise) / T)
A_t = sigmoid((base_nll_t - tau_new) / T) * (1 - N_t)
```

Interpretation:

- `R_t`: old-known / retention gate.
- `A_t`: new-trustworthy / acquisition gate.
- `N_t`: noisy-outlier gate.

Thresholds can be explicit values or batch quantiles.

## Baseline Scalar Triage Objective

The scalar token-triage loss is:

```text
w_t = clip(1 + lambda * A_t - mu * N_t - rho * R_t, w_floor, w_max)
L = mean(w_t * CE_t) + beta * mean(R_t * KL(base || student))
```

This is a useful baseline but is not enough novelty by itself. It looks like a
combination of low-PPL masking and token-level distillation.

## FullFT Module-Aware Retention

The new implementation routes separate losses to separate parameter groups.

```text
Attention route:
  w_attn = clip(1 + lambda_attn * A_t - mu_attn * N_t - rho_attn * R_t)
  L_attn = mean(w_attn * CE_t) + beta_attn * mean(R_t * KL)

MLP route:
  w_mlp = clip(1 + lambda_mlp * A_t - mu_mlp * N_t - rho_mlp * R_t)
  L_mlp = mean(w_mlp * CE_t) + beta_mlp * mean(R_t * KL)

Other route:
  w_other = clip(1 + lambda_other * A_t - mu_other * N_t - rho_other * R_t)
  L_other = mean(w_other * CE_t) + beta_other * mean(R_t * KL)
```

Default setting:

```text
attention: acquisition-heavy CE, no KL
MLP: selective acquisition CE + retention KL
other: conservative CE + retention KL
```

The trainer performs full-parameter fine-tuning. It keeps all trainable
parameters in the optimizer, but each route is backpropagated only through its
matching parameter group.

## Why This Is More Novel Than LoRA LR Groups

Weak framing:

```text
Use LoRA on attention with higher LR and MLP with lower LR.
```

This is mostly engineering and is close to selective tuning.

Stronger framing:

```text
For the same full-FT model and the same batch, route token-role-specific
objectives into different parameter subspaces.
```

This is a training principle rather than a target-module recipe.

## Expected Ablations

Required baselines:

- FullFT_CE.
- Low-PPL masking / STM_top20.
- Soft_STM.
- Retention_KL.
- Learn_New scalar triage.
- Attention-only update or attention-only route.
- MLP conservative update / MLP low-LR if feasible.
- Replay or LwF-style distillation if compute allows.

Key ablations:

- Remove `R_t * KL`.
- Remove `A_t`.
- Remove `N_t`.
- Route all objectives to all parameters.
- Swap attention and MLP routes.
- Use student confidence gates instead of frozen-base gates.

## Evidence Needed For Main-Conference Submission

The current idea is plausible but not yet enough for a strong main-conference
claim. To target ACL/EMNLP/NAACL main, or ICLR/NeurIPS with more risk, the work
needs clear evidence:

- Pareto improvement: better learning-forgetting frontier than FullFT_CE,
  KL-only, low-PPL masking, and selective tuning.
- Cross-setting robustness: multiple model families, domains, noise levels,
  and training lengths.
- Mechanistic verification: MLP drift on old-known tokens is reduced, attention
  absorbs more acquisition/routing signal, and swapping routes hurts.
- Clean narrative: the contribution is gradient routing conditioned on token
  role, not a heuristic combination of token weights and KL.

## Current Novelty Assessment

Risks:

- Low-perplexity token learning already covers high-PPL masking for forgetting.
- KL retention and LwF are old ideas.
- Selective module tuning has recent related work.

Best positioning:

```text
Full-parameter token-role-conditioned module gradient routing for preserving
pretrained knowledge while acquiring new task/domain behavior.
```

Likely venue path:

- With only scalar token triage: workshop or Findings-level.
- With strong routed full-FT results and mechanistic ablations: plausible for
  ACL/EMNLP/NAACL main.
- For ICLR/NeurIPS: needs a very clean theory or unusually strong, broad
  empirical evidence.

## Immediate Experiment Plan

Run a small trusted matrix before spending more GPU:

```text
methods:
  ce
  retention_kl
  stm_top20
  fullft_module_aware_retention

regimes:
  medical
  noise25

eval:
  IFEval full
  short-answer or MC MMLU-Pro candidates
  training/eval loss
  old-task retention metrics
```

Stop long, low-signal CoT MMLU-Pro runs until the short-answer evaluation path
is stable.
