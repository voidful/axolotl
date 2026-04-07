#!/usr/bin/env python3
"""
Generate Figure 1: The Teaser (dual panel).

Left panel: Efficiency vs. Robustness scatter
  - X-axis: Peak VRAM (GB)
  - Y-axis: MMLU-Pro retention rate

Right panel: Test-Time Scaling (Pass@k curves)
  - X-axis: k (log scale)
  - Y-axis: MATH-500 accuracy (Pass@k)

Usage:
    python plot_teaser.py \
        --results_dir ./results/paper \
        --output_path ./paper/figures/fig1_teaser.pdf
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# --- NeurIPS-quality plot style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color palette
COLORS = {
    'ce': '#E74C3C',        # Red
    'drift': '#2ECC71',     # Green
    'dpo': '#3498DB',       # Blue
    'ce_clean': '#95A5A6',  # Gray (clean baseline reference)
}

MARKERS = {
    'ce': 'X',
    'drift': '*',
    'dpo': 'D',
    'ce_clean': 'o',
}


def plot_left_panel(ax, data):
    """Efficiency vs. Robustness scatter."""
    for method, info in data.items():
        ax.scatter(
            info['vram_gb'], info['mmlu_pro'],
            color=COLORS.get(method, '#333'),
            marker=MARKERS.get(method, 'o'),
            s=180 if method == 'drift' else 120,
            edgecolors='black' if method == 'drift' else 'none',
            linewidths=1.5 if method == 'drift' else 0,
            zorder=10 if method == 'drift' else 5,
            label=info.get('label', method),
        )

    # Highlight Drift-Trust's golden position
    if 'drift' in data:
        drift = data['drift']
        ax.annotate(
            'Drift-Trust\n(Ours)',
            xy=(drift['vram_gb'], drift['mmlu_pro']),
            xytext=(drift['vram_gb'] + 1.5, drift['mmlu_pro'] - 2),
            fontsize=9, fontweight='bold', color=COLORS['drift'],
            arrowprops=dict(arrowstyle='->', color=COLORS['drift'], lw=1.5),
        )

    ax.set_xlabel('Peak VRAM (GB)')
    ax.set_ylabel('MMLU-Pro Retention (%)')
    ax.set_title('(a) Efficiency vs. Robustness', fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_right_panel(ax, data):
    """Test-Time Scaling Pass@k curves."""
    k_values = [1, 2, 4, 8, 16, 32, 64]

    for method, info in data.items():
        scores = info.get('pass_at_k', {})
        ks = []
        vals = []
        for k in k_values:
            key = f"pass@{k}"
            if key in scores:
                ks.append(k)
                vals.append(scores[key] * 100)

        if not ks:
            continue

        ax.plot(
            ks, vals,
            color=COLORS.get(method, '#333'),
            marker=MARKERS.get(method, 'o'),
            markersize=7,
            linewidth=2.5 if method == 'drift' else 1.8,
            label=info.get('label', method),
            zorder=10 if method == 'drift' else 5,
        )

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])

    ax.set_xlabel('Samples $k$')
    ax.set_ylabel('MATH-500 Accuracy (%)')
    ax.set_title('(b) Test-Time Scaling', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def load_results(results_dir: str):
    """Load results from JSON files. Returns placeholder data if files don't exist."""
    # Check for real results
    left_data = {}
    right_data = {}

    # Try loading real data
    eff_file = Path(results_dir) / "efficiency" / "efficiency_results.json"
    if eff_file.exists():
        with open(eff_file) as f:
            eff_data = json.load(f)
        for r in eff_data:
            name = r.get("run_name", "")
            if "ce" in name:
                left_data["ce"] = {"vram_gb": r.get("peak_vram_gb", 9.2), "label": "Standard CE"}
            elif "drift" in name:
                left_data["drift"] = {"vram_gb": r.get("peak_vram_gb", 9.5), "label": "Drift-Trust (Ours)"}

    # Load Pass@k results
    for method, label in [("ce_math_4b", "Standard CE"), ("drift_math_4b", "Drift-Trust (Ours)")]:
        pk_file = Path(results_dir) / f"math500_passk_{method}.json"
        if pk_file.exists():
            with open(pk_file) as f:
                pk_data = json.load(f)
            key = "ce" if "ce" in method else "drift"
            right_data[key] = {"pass_at_k": pk_data.get("pass_at_k", {}), "label": label}

    # If no real data, use placeholder data for figure layout
    if not left_data:
        print("Using placeholder data for left panel (run experiments first!)")
        left_data = {
            'ce_clean': {'vram_gb': 9.2, 'mmlu_pro': 48.5, 'label': 'CE (Clean Data)'},
            'ce': {'vram_gb': 9.2, 'mmlu_pro': 43.1, 'label': 'Standard CE (Noisy)'},
            'dpo': {'vram_gb': 18.4, 'mmlu_pro': 47.8, 'label': 'DPO'},
            'drift': {'vram_gb': 9.5, 'mmlu_pro': 47.6, 'label': 'Drift-Trust (Ours)'},
        }

    if not right_data:
        print("Using placeholder data for right panel (run experiments first!)")
        right_data = {
            'ce': {
                'label': 'Standard CE',
                'pass_at_k': {
                    'pass@1': 0.32, 'pass@2': 0.40, 'pass@4': 0.48,
                    'pass@8': 0.53, 'pass@16': 0.56, 'pass@32': 0.57, 'pass@64': 0.58,
                },
            },
            'drift': {
                'label': 'Drift-Trust (Ours)',
                'pass_at_k': {
                    'pass@1': 0.34, 'pass@2': 0.43, 'pass@4': 0.52,
                    'pass@8': 0.60, 'pass@16': 0.66, 'pass@32': 0.71, 'pass@64': 0.75,
                },
            },
        }

    return left_data, right_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results/paper")
    parser.add_argument("--output_path", type=str, default="./paper/figures/fig1_teaser.pdf")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    left_data, right_data = load_results(args.results_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.35)

    plot_left_panel(ax1, left_data)
    plot_right_panel(ax2, right_data)

    plt.savefig(args.output_path)
    plt.savefig(args.output_path.replace('.pdf', '.png'))
    print(f"Figure saved to {args.output_path}")


if __name__ == "__main__":
    main()
