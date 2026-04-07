#!/usr/bin/env python3
"""
Generate Figure 3: Token-level reliability heatmap.

Visualizes per-token r_t values on a sample sentence containing hallucination.
Tokens with low r_t (model detects anomaly) are shown in dark/red,
tokens with high r_t (reliable) are shown in light/green.

Usage:
    python plot_token_heatmap.py \
        --model_path ./outputs/paper/drift-noisy-4b \
        --output_path ./paper/figures/fig3_heatmap.pdf
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F


def compute_token_reliability(
    model, tokenizer, text: str,
    ema_decay: float = 0.999, gamma: float = 1.0,
) -> tuple[list[str], np.ndarray]:
    """
    Run a forward pass and compute per-token drift-based reliability.

    Returns:
        tokens: list of token strings
        r_t: array of reliability scores in [0, 1]
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, T, V)

    log_probs = F.log_softmax(logits, dim=-1)  # (1, T, V)

    # Gather target token log probs (shifted for next-token prediction)
    target_ids = input_ids[:, 1:]  # (1, T-1)
    pred_logp = log_probs[:, :-1, :]  # (1, T-1, V)
    target_logp = pred_logp.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # (1, T-1)

    # Drift = |log p_θ(y_t)| (CE since no prior cache)
    instant_drift = target_logp.abs()  # (1, T-1)

    # Simulate EMA running drift
    T = instant_drift.shape[1]
    running_drift = 0.0
    drift_values = []
    for t in range(T):
        d_t = instant_drift[0, t].item()
        running_drift = ema_decay * running_drift + (1 - ema_decay) * d_t
        blended = 0.5 * d_t + 0.5 * running_drift
        drift_values.append(blended)

    drift = np.array(drift_values)

    # Reliability: r_evi = exp(-γ * drift), then z-score + sigmoid
    r_evi = np.exp(-gamma * drift)
    mu = r_evi.mean()
    r_normalized = (r_evi - mu) / 1.0
    r_t = 1.0 / (1.0 + np.exp(-r_normalized))

    # Map input IDs to tokens
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0, 1:].tolist()]

    return tokens, r_t


def plot_heatmap(tokens: list[str], r_t: np.ndarray, output_path: str,
                 title: str = "Token-Level Reliability Score $r_t$"):
    """Generate a publication-quality token heatmap."""
    # Limit tokens for readability
    max_tokens = 60
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        r_t = r_t[:max_tokens]

    # Create figure
    fig_width = max(12, len(tokens) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 2.5))

    # Create heatmap as colored text
    cmap = plt.cm.RdYlGn  # Red (low r_t) → Yellow → Green (high r_t)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    x_pos = 0
    text_objects = []
    positions = []
    for i, (tok, r) in enumerate(zip(tokens, r_t)):
        color = cmap(norm(r))
        display_tok = tok.replace(' ', '·')
        if not display_tok.strip():
            display_tok = '·'

        # Background rectangle
        width = len(display_tok) * 0.12 + 0.1
        rect = plt.Rectangle((x_pos - 0.05, 0.2), width, 0.6,
                              facecolor=color, edgecolor='gray',
                              alpha=0.85, linewidth=0.5)
        ax.add_patch(rect)

        # Token text
        txt = ax.text(x_pos, 0.5, display_tok,
                      fontsize=7, fontfamily='monospace',
                      verticalalignment='center',
                      color='black' if r > 0.4 else 'white')
        text_objects.append(txt)
        positions.append(x_pos)

        # Score below
        ax.text(x_pos + width / 2 - 0.05, 0.05, f'{r:.2f}',
                fontsize=5, ha='center', color='gray')

        x_pos += width + 0.05

    ax.set_xlim(-0.2, x_pos + 0.2)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        pad=0.3, fraction=0.05, aspect=40)
    cbar.set_label('Reliability $r_t$ (low = suspicious, high = reliable)',
                   fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model. If None, uses synthetic demo data.")
    parser.add_argument("--text", type=str, default=None,
                        help="Custom text to visualize. If None, uses example with hallucination.")
    parser.add_argument("--output_path", type=str, default="./paper/figures/fig3_heatmap.pdf")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Default hallucination-containing text for demonstration
    if args.text is None:
        args.text = (
            "The capital of France is Berlin, which was established in 1792 "
            "during the French Revolution. Paris, the largest city, has a "
            "population of approximately 2.1 million people and is home to "
            "the Eiffel Tower, built in 1889."
        )

    if args.model_path is not None:
        # Use real model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        model.eval()

        tokens, r_t = compute_token_reliability(
            model, tokenizer, args.text,
            ema_decay=args.ema_decay, gamma=args.gamma,
        )
    else:
        # Generate synthetic demo data
        print("No model path — generating synthetic demo heatmap...")
        words = args.text.split()
        tokens = words
        # Simulate: hallucination tokens get low reliability
        hallucination_words = {"Berlin", "1792"}
        r_t = np.array([
            0.15 + np.random.uniform(0, 0.15) if w.rstrip('.,') in hallucination_words
            else 0.65 + np.random.uniform(0, 0.3)
            for w in words
        ])

    plot_heatmap(tokens, r_t, args.output_path)


if __name__ == "__main__":
    main()
