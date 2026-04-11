#!/bin/bash
# ============================================================
# Drift-Trust NeurIPS 2026 — Master Execution Script
# ============================================================
#
# This script orchestrates the full paper pipeline:
#   Phase 1: Data preparation
#   Phase 2: Training (4 runs)
#   Phase 3: Evaluation
#   Phase 4: Figures
#
# Usage:
#   bash run_paper_pipeline.sh [phase]
#   bash run_paper_pipeline.sh data      # Phase 1 only
#   bash run_paper_pipeline.sh train     # Phase 2 only
#   bash run_paper_pipeline.sh eval      # Phase 3 only
#   bash run_paper_pipeline.sh figures   # Phase 4 only
#   bash run_paper_pipeline.sh all       # Everything
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAPER_SCRIPTS="${PROJECT_DIR}/scripts/paper"
PAPER_CONFIGS="${PROJECT_DIR}/examples/paper"
DATA_DIR="${PROJECT_DIR}/data/paper"
RESULTS_DIR="${PROJECT_DIR}/results/paper"
FIGURES_DIR="${PROJECT_DIR}/paper/figures"

# Prevent CUDA memory fragmentation on 48GB GPUs (critical for checkpoint save+resume)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PHASE="${1:-all}"

echo "============================================"
echo "  Drift-Trust NeurIPS 2026 Pipeline"
echo "  Phase: ${PHASE}"
echo "  Project: ${PROJECT_DIR}"
echo "============================================"

# ── Phase 1: Data Preparation ──────────────────────────────
if [[ "${PHASE}" == "data" || "${PHASE}" == "all" ]]; then
    echo ""
    echo "=== PHASE 1: Data Preparation ==="
    mkdir -p "${DATA_DIR}"

    echo "[1/2] Creating noisy UltraFeedback (10k, 25% noise)..."
    python "${PAPER_SCRIPTS}/create_noisy_ultrafeedback.py" \
        --output_dir "${DATA_DIR}" \
        --num_samples 10000 \
        --noise_ratio 0.25 \
        --seed 42

    echo "[2/2] Sampling NuminaMath-CoT (10k)..."
    python "${PAPER_SCRIPTS}/sample_numina_math.py" \
        --output_dir "${DATA_DIR}" \
        --num_samples 10000 \
        --seed 42

    echo "Phase 1 complete. Data saved to ${DATA_DIR}"
fi

# ── Phase 2: Training ──────────────────────────────────────
if [[ "${PHASE}" == "train" || "${PHASE}" == "all" ]]; then
    echo ""
    echo "=== PHASE 2: Training (6 runs) ==="

    echo "[1/6] CE on noisy UltraFeedback..."
    accelerate launch -m axolotl.cli.train "${PAPER_CONFIGS}/ce_noisy_4b.yaml"

    echo "[2/6] Drift-Trust on noisy UltraFeedback..."
    accelerate launch -m axolotl.cli.train "${PAPER_CONFIGS}/drift_noisy_4b.yaml"

    echo "[3/6] CE on NuminaMath-CoT..."
    accelerate launch -m axolotl.cli.train "${PAPER_CONFIGS}/ce_math_4b.yaml"

    echo "[4/6] Drift-Trust on NuminaMath-CoT..."
    accelerate launch -m axolotl.cli.train "${PAPER_CONFIGS}/drift_math_4b.yaml"

    echo "[5/6] Route C: Per-Sample Drift on noisy UltraFeedback..."
    accelerate launch -m axolotl.cli.train "${PAPER_CONFIGS}/drift_noisy_4b_persample.yaml"

    echo "[6/6] Route B: Aggressive Drift on noisy UltraFeedback..."
    accelerate launch -m axolotl.cli.train "${PAPER_CONFIGS}/drift_noisy_4b_aggressive.yaml"

    echo "Phase 2 complete. Checkpoints saved."
fi

# ── Phase 3: Evaluation ────────────────────────────────────
if [[ "${PHASE}" == "eval" || "${PHASE}" == "all" ]]; then
    echo ""
    echo "=== PHASE 3: Evaluation ==="
    mkdir -p "${RESULTS_DIR}"

    # --- Battle A: MMLU-Pro + IFEval ---
    echo "[Battle A] Evaluating base model..."
    bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" "Qwen/Qwen3.5-4B" "qwen35_4b_base"

    echo "[Battle A] Evaluating CE (noisy)..."
    bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" \
        "${PROJECT_DIR}/outputs/paper/ce-noisy-4b" "ce_noisy_4b"

    echo "[Battle A] Evaluating Drift-Trust (noisy)..."
    bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" \
        "${PROJECT_DIR}/outputs/paper/drift-noisy-4b" "drift_noisy_4b"

    echo "[Battle A] Evaluating Per-Sample Drift (noisy) — Route C..."
    bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" \
        "${PROJECT_DIR}/outputs/paper/drift-noisy-4b-persample" "drift_noisy_persample"

    echo "[Battle A] Evaluating Aggressive Drift (noisy) — Route B..."
    bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" \
        "${PROJECT_DIR}/outputs/paper/drift-noisy-4b-aggressive" "drift_noisy_aggressive"

    # --- Battle B: MATH-500 Pass@k ---
    echo "[Battle B] Evaluating base model Pass@k..."
    python "${PAPER_SCRIPTS}/eval_math_passk.py" \
        --model_path "Qwen/Qwen3.5-4B" \
        --k 64 --temperature 0.7 \
        --output_dir "${RESULTS_DIR}" \
        --run_name "qwen35_4b_base"

    echo "[Battle B] Evaluating CE (math) Pass@k..."
    python "${PAPER_SCRIPTS}/eval_math_passk.py" \
        --model_path "${PROJECT_DIR}/outputs/paper/ce-math-4b" \
        --k 64 --temperature 0.7 \
        --output_dir "${RESULTS_DIR}" \
        --run_name "ce_math_4b"

    echo "[Battle B] Evaluating Drift-Trust (math) Pass@k..."
    python "${PAPER_SCRIPTS}/eval_math_passk.py" \
        --model_path "${PROJECT_DIR}/outputs/paper/drift-math-4b" \
        --k 64 --temperature 0.7 \
        --output_dir "${RESULTS_DIR}" \
        --run_name "drift_math_4b"

    echo "Phase 3 complete. Results saved to ${RESULTS_DIR}"
fi

# ── Phase 4: Figures ───────────────────────────────────────
if [[ "${PHASE}" == "figures" || "${PHASE}" == "all" ]]; then
    echo ""
    echo "=== PHASE 4: Figure Generation ==="
    mkdir -p "${FIGURES_DIR}"

    echo "[1/2] Generating Figure 1 (Teaser)..."
    python "${PAPER_SCRIPTS}/plot_teaser.py" \
        --results_dir "${RESULTS_DIR}" \
        --output_path "${FIGURES_DIR}/fig1_teaser.pdf"

    echo "[2/2] Generating Figure 3 (Token Heatmap)..."
    python "${PAPER_SCRIPTS}/plot_token_heatmap.py" \
        --model_path "${PROJECT_DIR}/outputs/paper/drift-noisy-4b" \
        --output_path "${FIGURES_DIR}/fig3_heatmap.pdf"

    echo "Phase 4 complete. Figures saved to ${FIGURES_DIR}"
fi

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
