#!/usr/bin/env bash
# ============================================================
# NeurIPS 2026 drift-loss experiment suite on 1x A6000
# ============================================================
#
# This wraps run_fast_drift_submission.sh into a staged paper plan.
# It keeps the method simple: CE vs one drift-loss objective.
#
# Phases:
#   gate      1-seed smoke experiment; decide whether the story is alive.
#   main      3-seed main table over domain + noisy alignment regimes.
#   ablation  gamma sweep on the two most diagnostic regimes.
#   scale     data/step scaling stress test.
#   all       gate + main + ablation + scale.
#
# Usage:
#   bash scripts/paper/run_neurips_drift_suite.sh gate
#   bash scripts/paper/run_neurips_drift_suite.sh main
#   bash scripts/paper/run_neurips_drift_suite.sh ablation
#   bash scripts/paper/run_neurips_drift_suite.sh scale
#
# For final/full eval, override:
#   EVAL_LIMIT="" MMLU_FEWSHOT=5 EVAL_BACKEND=vllm MERGE_FOR_EVAL=1 \
#     bash scripts/paper/run_neurips_drift_suite.sh main
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE="${SCRIPT_DIR}/run_fast_drift_submission.sh"

phase="${1:-gate}"

run_gate() {
    RUN_TAG=neurips_gate \
    REGIMES="medical noise25" \
    METHODS="ce drift" \
    SEEDS="42" \
    N_TRAIN=2000 \
    MAX_STEPS=180 \
    EVAL_LIMIT="${EVAL_LIMIT:-200}" \
    DOMAIN_EVAL=1 \
    bash "${PIPELINE}" pilot
}

run_main() {
    RUN_TAG=neurips_main \
    REGIMES="medical math noise0 noise25 noise50" \
    METHODS="ce drift" \
    SEEDS="${SEEDS:-42 123 456}" \
    N_TRAIN="${N_TRAIN:-2000}" \
    MAX_STEPS="${MAX_STEPS:-240}" \
    EVAL_LIMIT="${EVAL_LIMIT:-500}" \
    DOMAIN_EVAL=1 \
    bash "${PIPELINE}" pilot
}

run_ablation() {
    for gamma in 0.5 1.0 2.0 4.0; do
        tag="neurips_ablation_gamma_${gamma/./p}"
        RUN_TAG="${tag}" \
        REGIMES="medical noise25" \
        METHODS="drift" \
        SEEDS="42" \
        N_TRAIN="${N_TRAIN:-2000}" \
        MAX_STEPS="${MAX_STEPS:-240}" \
        DRIFT_GAMMA="${gamma}" \
        EVAL_LIMIT="${EVAL_LIMIT:-300}" \
        DOMAIN_EVAL=1 \
        bash "${PIPELINE}" pilot
    done
}

run_scale() {
    for size_steps in "1000:120" "2000:240" "5000:480"; do
        size="${size_steps%%:*}"
        steps="${size_steps##*:}"
        RUN_TAG="neurips_scale_${size}" \
        REGIMES="medical noise25" \
        METHODS="ce drift" \
        SEEDS="42" \
        N_TRAIN="${size}" \
        MAX_STEPS="${steps}" \
        EVAL_LIMIT="${EVAL_LIMIT:-300}" \
        DOMAIN_EVAL=1 \
        bash "${PIPELINE}" pilot
    done
}

case "${phase}" in
    gate)
        run_gate
        ;;
    main)
        run_main
        ;;
    ablation)
        run_ablation
        ;;
    scale)
        run_scale
        ;;
    all)
        run_gate
        run_main
        run_ablation
        run_scale
        ;;
    *)
        echo "Unknown phase: ${phase}" >&2
        exit 1
        ;;
esac

