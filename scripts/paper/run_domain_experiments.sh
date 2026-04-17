#!/bin/bash
# ============================================================
# Drift-Trust NeurIPS 2026 — Domain SFT Experiments
# ============================================================
#
# Battle B: Math SFT → General Knowledge Retention
# Battle C: Medical SFT → Domain + General Knowledge
#
# Usage:
#   bash run_domain_experiments.sh              # Run everything
#   bash run_domain_experiments.sh eval_math    # Battle B eval only
#   bash run_domain_experiments.sh medical      # Battle C (data+train+eval)
#   bash run_domain_experiments.sh summary      # Aggregate results
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAPER_SCRIPTS="${PROJECT_DIR}/scripts/paper"
PAPER_CONFIGS="${PROJECT_DIR}/examples/paper"
DATA_DIR="${PROJECT_DIR}/data/paper"
RESULTS_DIR="${PROJECT_DIR}/results/paper"

PHASE="${1:-all}"

echo "============================================"
echo "  Drift-Trust — Domain SFT Experiments"
echo "  Phase: ${PHASE}"
echo "============================================"

# Helper: run eval if results don't exist
run_eval_if_needed() {
    local model_path="$1"
    local run_name="$2"
    local result_dir="${RESULTS_DIR}/benchmarks/${run_name}"

    if [ ! -d "${model_path}" ] && [[ "${model_path}" != Qwen/* ]]; then
        echo "[eval] ${run_name} — SKIP (no checkpoint at ${model_path})"
        return 1
    fi

    if [ -d "${result_dir}" ]; then
        local has_results=false
        for root_dir in "${result_dir}"; do
            if find "${root_dir}" -name 'results_*.json' 2>/dev/null | head -1 | grep -q .; then
                has_results=true
            fi
        done
        if [ "${has_results}" = true ]; then
            echo "[eval] ${run_name} — SKIP (results exist)"
            return 0
        fi
    fi

    echo "[eval] ${run_name}..."
    bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" "${model_path}" "${run_name}"
    echo "[eval] ${run_name} — DONE"
    return 0
}

# ══════════════════════════════════════════════
# Battle B: Math SFT → MMLU-Pro eval
# ══════════════════════════════════════════════
phase_eval_math() {
    echo ""
    echo "=== BATTLE B: Math SFT → General Knowledge Retention ==="
    echo "  (No training needed — using existing checkpoints)"
    echo ""

    run_eval_if_needed "./outputs/paper/ce-math-4b" "ce_math_4b"
    run_eval_if_needed "./outputs/paper/drift-math-4b" "drift_math_4b"

    echo ""
    echo "=== Battle B eval complete ==="
}

# ══════════════════════════════════════════════
# Battle C: Medical SFT
# ══════════════════════════════════════════════
phase_medical_data() {
    echo ""
    echo "=== BATTLE C: Medical Data Preparation ==="
    mkdir -p "${DATA_DIR}"

    local fname="medical_flashcards_10k.jsonl"
    if [ -f "${DATA_DIR}/${fname}" ]; then
        echo "[data] ${fname} already exists, skipping."
    else
        echo "[data] Creating medical SFT dataset (10k)..."
        python3 "${PAPER_SCRIPTS}/create_medical_sft.py" \
            --output_dir "${DATA_DIR}" \
            --num_samples 10000 \
            --seed 42
    fi

    echo "=== Medical data ready ==="
}

phase_medical_train() {
    echo ""
    echo "=== BATTLE C: Medical Training ==="

    # CE baseline
    local ce_out="./outputs/paper/ce-medical-4b"
    if [ -d "${ce_out}" ] && [ -f "${ce_out}/config.json" ]; then
        echo "[train] CE-Medical — SKIP (exists)"
    else
        echo "[train] CE-Medical..."
        python3 -m accelerate.commands.launch -m axolotl.cli.train \
            "${PAPER_CONFIGS}/ce_medical_4b.yaml"
    fi

    # Drift-Trust
    local drift_out="./outputs/paper/drift-medical-4b"
    if [ -d "${drift_out}" ] && [ -f "${drift_out}/config.json" ]; then
        echo "[train] Drift-Medical — SKIP (exists)"
    else
        echo "[train] Drift-Medical..."
        python3 -m accelerate.commands.launch -m axolotl.cli.train \
            "${PAPER_CONFIGS}/drift_medical_4b.yaml"
    fi

    echo "=== Medical training complete ==="
}

phase_medical_eval() {
    echo ""
    echo "=== BATTLE C: Medical Evaluation ==="

    # MMLU-Pro (general knowledge retention)
    run_eval_if_needed "./outputs/paper/ce-medical-4b" "ce_medical_4b"
    run_eval_if_needed "./outputs/paper/drift-medical-4b" "drift_medical_4b"

    # MedQA (domain performance) — using lm-eval-harness
    echo ""
    echo "[eval] Running MedQA for CE-Medical..."
    local ce_medqa="${RESULTS_DIR}/benchmarks/ce_medical_4b/medqa"
    if [ -d "${ce_medqa}" ] && find "${ce_medqa}" -name 'results_*.json' 2>/dev/null | head -1 | grep -q .; then
        echo "[eval] CE-Medical MedQA — SKIP (exists)"
    else
        mkdir -p "${RESULTS_DIR}/benchmarks/ce_medical_4b"
        python3 -m lm_eval --model vllm \
            --model_args "pretrained=./outputs/paper/ce-medical-4b,dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.9,max_model_len=4096" \
            --tasks medqa_4options \
            --batch_size auto \
            --num_fewshot 0 \
            --output_path "${RESULTS_DIR}/benchmarks/ce_medical_4b/medqa" \
            --log_samples 2>&1 | tee "${RESULTS_DIR}/benchmarks/ce_medical_4b/medqa.log"
    fi

    echo "[eval] Running MedQA for Drift-Medical..."
    local drift_medqa="${RESULTS_DIR}/benchmarks/drift_medical_4b/medqa"
    if [ -d "${drift_medqa}" ] && find "${drift_medqa}" -name 'results_*.json' 2>/dev/null | head -1 | grep -q .; then
        echo "[eval] Drift-Medical MedQA — SKIP (exists)"
    else
        mkdir -p "${RESULTS_DIR}/benchmarks/drift_medical_4b"
        python3 -m lm_eval --model vllm \
            --model_args "pretrained=./outputs/paper/drift-medical-4b,dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.9,max_model_len=4096" \
            --tasks medqa_4options \
            --batch_size auto \
            --num_fewshot 0 \
            --output_path "${RESULTS_DIR}/benchmarks/drift_medical_4b/medqa" \
            --log_samples 2>&1 | tee "${RESULTS_DIR}/benchmarks/drift_medical_4b/medqa.log"
    fi

    echo ""
    echo "=== Medical evaluation complete ==="
}

# ══════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════
phase_summary() {
    echo ""
    echo "=== Domain Experiments Summary ==="

    python3 << 'SUMMARY_EOF'
import json, os

results_dir = os.environ.get("RESULTS_DIR", "./results/paper")
bench_dir = os.path.join(results_dir, "benchmarks")

def load_results(run_name):
    """Load results from a run directory, handling hidden subdirs."""
    run_dir = os.path.join(bench_dir, run_name)
    data = {}
    if not os.path.isdir(run_dir):
        return data
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                try:
                    with open(os.path.join(root, f)) as fh:
                        jdata = json.load(fh)
                    for task, metrics in jdata.get("results", {}).items():
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)) and \
                               ("acc" in metric.lower() or "match" in metric.lower()):
                                data[f"{task}/{metric}"] = value
                except Exception:
                    pass
    return data

def fmt(v):
    return f"{v*100:.2f}%" if v is not None else "---"

print("\n" + "=" * 75)
print("  DOMAIN SFT EXPERIMENTS — CATASTROPHIC FORGETTING ANALYSIS")
print("=" * 75)

# Battle B: Math
print("\n--- Battle B: Math SFT ---\n")
print(f"{'Model':<25} {'MMLU-Pro':>10} {'MATH-500 P@1':>14} {'Δ MMLU':>10}")
print("-" * 65)

base = load_results("qwen35_4b_base")
ce_math = load_results("ce_math_4b")
drift_math = load_results("drift_math_4b")

base_mmlu = base.get("mmlu_pro/exact_match,custom-extract")
ce_math_mmlu = ce_math.get("mmlu_pro/exact_match,custom-extract")
drift_math_mmlu = drift_math.get("mmlu_pro/exact_match,custom-extract")

delta_ce = f"{(ce_math_mmlu - base_mmlu)*100:+.2f}pp" if (ce_math_mmlu and base_mmlu) else "---"
delta_dr = f"{(drift_math_mmlu - base_mmlu)*100:+.2f}pp" if (drift_math_mmlu and base_mmlu) else "---"

print(f"{'Base (no SFT)':<25} {fmt(base_mmlu):>10} {'48.50%':>14} {'---':>10}")
print(f"{'CE-Math':<25} {fmt(ce_math_mmlu):>10} {'52.29%':>14} {delta_ce:>10}")
print(f"{'Drift-Math':<25} {fmt(drift_math_mmlu):>10} {'50.27%':>14} {delta_dr:>10}")

# Battle C: Medical
print("\n--- Battle C: Medical SFT ---\n")
print(f"{'Model':<25} {'MMLU-Pro':>10} {'MedQA':>10} {'Δ MMLU':>10}")
print("-" * 60)

ce_med = load_results("ce_medical_4b")
drift_med = load_results("drift_medical_4b")

ce_med_mmlu = ce_med.get("mmlu_pro/exact_match,custom-extract")
drift_med_mmlu = drift_med.get("mmlu_pro/exact_match,custom-extract")
ce_med_medqa = ce_med.get("medqa_4options/acc,none") or ce_med.get("medqa/acc,none")
drift_med_medqa = drift_med.get("medqa_4options/acc,none") or drift_med.get("medqa/acc,none")

delta_ce_m = f"{(ce_med_mmlu - base_mmlu)*100:+.2f}pp" if (ce_med_mmlu and base_mmlu) else "---"
delta_dr_m = f"{(drift_med_mmlu - base_mmlu)*100:+.2f}pp" if (drift_med_mmlu and base_mmlu) else "---"

print(f"{'Base (no SFT)':<25} {fmt(base_mmlu):>10} {'---':>10} {'---':>10}")
print(f"{'CE-Medical':<25} {fmt(ce_med_mmlu):>10} {fmt(ce_med_medqa):>10} {delta_ce_m:>10}")
print(f"{'Drift-Medical':<25} {fmt(drift_med_mmlu):>10} {fmt(drift_med_medqa):>10} {delta_dr_m:>10}")

print("\n" + "=" * 75)

SUMMARY_EOF

    echo ""
    echo "=== Summary complete ==="
}

# ══════════════════════════════════════════════
# Main dispatch
# ══════════════════════════════════════════════
export RESULTS_DIR

case "${PHASE}" in
    eval_math)    phase_eval_math ;;
    medical_data) phase_medical_data ;;
    medical_train) phase_medical_train ;;
    medical_eval) phase_medical_eval ;;
    medical)
        phase_medical_data
        phase_medical_train
        phase_medical_eval
        ;;
    summary)      phase_summary ;;
    all)
        phase_eval_math
        phase_medical_data
        phase_medical_train
        phase_medical_eval
        phase_summary
        ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo "Usage: $0 {eval_math|medical|medical_data|medical_train|medical_eval|summary|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Done! (Phase: ${PHASE})"
echo "============================================"
