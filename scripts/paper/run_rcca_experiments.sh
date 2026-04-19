#!/bin/bash
# ============================================================
# RCCA-TR NeurIPS 2026 — Full Paper Experiments
# ============================================================
#
# Usage:
#   bash run_rcca_experiments.sh              # Run everything
#   bash run_rcca_experiments.sh train        # Train only
#   bash run_rcca_experiments.sh eval         # Eval only
#   bash run_rcca_experiments.sh summary      # Summary only
#   bash run_rcca_experiments.sh medical      # Medical only (train+eval)
#   bash run_rcca_experiments.sh math         # Math only (train+eval)
#   bash run_rcca_experiments.sh noise        # Noise sweep only (train+eval)
#
# On SLURM:
#   hrun -c 20 -m 50 -G -g RTX6000AdaGeneration \
#       bash scripts/paper/run_rcca_experiments.sh all
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAPER_SCRIPTS="${PROJECT_DIR}/scripts/paper"
CONFIG_DIR="${PROJECT_DIR}/examples/paper/rcca_tr/qwen4b"
DATA_DIR="${PROJECT_DIR}/data/paper"
RESULTS_DIR="${PROJECT_DIR}/results/paper"
EVAL_SCRIPT="${PAPER_SCRIPTS}/eval_benchmarks.sh"

PHASE="${1:-all}"
MODES=("ce" "hardness" "drift_only" "drift")
SEEDS=(42)  # Extend to (42 123 456) for 3-seed runs

echo "============================================"
echo "  RCCA-TR NeurIPS 2026"
echo "  Full Paper Experiments"
echo "  Phase: ${PHASE}"
echo "  Modes: ${MODES[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "============================================"

# ── Helpers ────────────────────────────────────────────────────────────

train_if_needed() {
    local config="$1"
    local output_dir="$2"
    local desc="$3"

    if [ -d "${output_dir}" ] && [ -f "${output_dir}/config.json" ]; then
        echo "[train] ${desc} — SKIP (exists: ${output_dir})"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "[train] ${desc}"
    echo "  Config: ${config}"
    echo "  Output: ${output_dir}"
    echo "============================================================"

    python3 -m accelerate.commands.launch -m axolotl.cli.train "${config}"
    echo "[train] ${desc} — DONE"
}

eval_if_needed() {
    local model_path="$1"
    local run_name="$2"
    local desc="$3"
    local result_dir="${RESULTS_DIR}/benchmarks/${run_name}"

    if [ ! -d "${model_path}" ] && [[ "${model_path}" != Qwen/* ]]; then
        echo "[eval] ${desc} — SKIP (no checkpoint)"
        return 0
    fi

    # Check if results already exist (handle hidden dirs from lm-eval)
    if [ -d "${result_dir}" ]; then
        local found_results
        found_results=$(find "${result_dir}" -name 'results_*.json' 2>/dev/null | head -1)
        if [ -n "${found_results}" ]; then
            echo "[eval] ${desc} — SKIP (results exist)"
            return 0
        fi
    fi

    echo "[eval] ${desc}..."
    bash "${EVAL_SCRIPT}" "${model_path}" "${run_name}"
    echo "[eval] ${desc} — DONE"
}

generate_config() {
    local mode="$1"
    local dataset_path="$2"
    local output_dir="$3"
    local config_path="$4"
    local prepared_path="$5"
    local seed="${6:-42}"

    cat > "${config_path}" << YAML
# Auto-generated RCCA-TR config
# Mode: ${mode} | Seed: ${seed}
base_model: Qwen/Qwen3.5-4B
low_cpu_mem_usage: true

plugins:
  - axolotl.integrations.rcca_tr.RCCATRPlugin
  - axolotl.integrations.liger.LigerPlugin

liger_rms_norm: true
liger_glu_activation: true

rcca_tr_trainer: true
rcca_tr_mode: ${mode}
rcca_tr_self_tau: 1.0
rcca_tr_drift_tau: 1.0
rcca_tr_w_min: 0.05
rcca_tr_beta: 0.5
rcca_tr_ema_decay: 0.999
rcca_tr_kl_lambda: 4.0
rcca_tr_anchor_weight: 0.1
rcca_tr_reliability_tau: 1.0

datasets:
  - path: ${dataset_path}
    ds_type: json
    type: chat_template
    field_messages: conversations
    split: train

dataset_prepared_path: ${prepared_path}
chat_template: qwen3_5

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: false

gradient_accumulation_steps: 8
micro_batch_size: 1
num_epochs: 3
optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 2e-5
warmup_ratio: 0.03
weight_decay: 0.01
seed: ${seed}

bf16: true
gradient_checkpointing: true
flash_attention: true
dataloader_num_workers: 4

val_set_size: 0.05
save_strategy: epoch
eval_strategy: epoch
save_total_limit: 1
output_dir: ${output_dir}

logging_steps: 1
report_to: wandb
wandb_project: rcca-tr-neurips2026
wandb_run_name: ${mode}-$(basename "${dataset_path}" .jsonl)-s${seed}
YAML
    echo "  Generated: ${config_path}"
}

# ══════════════════════════════════════════════════════════════════
# Phase: Data Preparation
# ══════════════════════════════════════════════════════════════════
phase_data() {
    echo ""
    echo "=== DATA PREPARATION ==="
    mkdir -p "${DATA_DIR}"

    # Medical dataset
    if [ ! -f "${DATA_DIR}/medical_flashcards_10k.jsonl" ]; then
        echo "[data] Creating medical SFT dataset..."
        python3 "${PAPER_SCRIPTS}/create_medical_sft.py" \
            --output_dir "${DATA_DIR}" --num_samples 10000 --seed 42
    else
        echo "[data] medical_flashcards_10k.jsonl — exists"
    fi

    # Noisy UltraFeedback (25% noise — main noise level)
    if [ ! -f "${DATA_DIR}/ultrafeedback_noisy_25pct_10k.jsonl" ]; then
        echo "[data] Creating noisy UltraFeedback (25%)..."
        python3 "${PAPER_SCRIPTS}/create_noisy_ultrafeedback.py" \
            --output_dir "${DATA_DIR}" --noise_fraction 0.25 \
            --num_samples 10000 --seed 42
    else
        echo "[data] ultrafeedback_noisy_25pct_10k.jsonl — exists"
    fi

    echo "=== Data ready ==="
}

# ══════════════════════════════════════════════════════════════════
# Phase: Training — Medical (Battle C)
# ══════════════════════════════════════════════════════════════════
phase_train_medical() {
    echo ""
    echo "=== TRAIN: Medical (Battle C) ==="

    local dataset="${DATA_DIR}/medical_flashcards_10k.jsonl"
    mkdir -p "${CONFIG_DIR}"

    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            local run="rcca-${mode}-medical-4b-s${seed}"
            local config="${CONFIG_DIR}/${mode}_medical_s${seed}.yaml"
            local output="./outputs/paper/${run}"
            local prepared="./prepared_data/paper/rcca_${mode}_medical_4b_s${seed}"

            generate_config "${mode}" "${dataset}" "${output}" "${config}" "${prepared}" "${seed}"
            train_if_needed "${config}" "${output}" "${mode} medical s${seed}"
        done
    done

    echo "=== Medical training complete ==="
}

# ══════════════════════════════════════════════════════════════════
# Phase: Training — Math (Battle B)
# ══════════════════════════════════════════════════════════════════
phase_train_math() {
    echo ""
    echo "=== TRAIN: Math (Battle B) ==="

    # Math uses existing NuminaMath-CoT configs — just need the RCCA versions
    # For now, use the same dataset path as the existing math configs
    local dataset
    if [ -f "${DATA_DIR}/numinamath_cot_10k.jsonl" ]; then
        dataset="${DATA_DIR}/numinamath_cot_10k.jsonl"
    else
        echo "[train] Math dataset not found at ${DATA_DIR}/numinamath_cot_10k.jsonl"
        echo "[train] Using existing ce-math-4b / drift-math-4b checkpoints for eval only"
        return 0
    fi

    mkdir -p "${CONFIG_DIR}"

    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            local run="rcca-${mode}-math-4b-s${seed}"
            local config="${CONFIG_DIR}/${mode}_math_s${seed}.yaml"
            local output="./outputs/paper/${run}"
            local prepared="./prepared_data/paper/rcca_${mode}_math_4b_s${seed}"

            generate_config "${mode}" "${dataset}" "${output}" "${config}" "${prepared}" "${seed}"
            train_if_needed "${config}" "${output}" "${mode} math s${seed}"
        done
    done

    echo "=== Math training complete ==="
}

# ══════════════════════════════════════════════════════════════════
# Phase: Training — Noisy Alignment (Battle A)
# ══════════════════════════════════════════════════════════════════
phase_train_noise() {
    echo ""
    echo "=== TRAIN: Noisy Alignment (Battle A) ==="

    local dataset="${DATA_DIR}/ultrafeedback_noisy_25pct_10k.jsonl"
    mkdir -p "${CONFIG_DIR}"

    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            local run="rcca-${mode}-noise25-4b-s${seed}"
            local config="${CONFIG_DIR}/${mode}_noise25_s${seed}.yaml"
            local output="./outputs/paper/${run}"
            local prepared="./prepared_data/paper/rcca_${mode}_noise25_4b_s${seed}"

            generate_config "${mode}" "${dataset}" "${output}" "${config}" "${prepared}" "${seed}"
            train_if_needed "${config}" "${output}" "${mode} noise25 s${seed}"
        done
    done

    echo "=== Noise training complete ==="
}

# ══════════════════════════════════════════════════════════════════
# Phase: Evaluation
# ══════════════════════════════════════════════════════════════════
phase_eval() {
    echo ""
    echo "=== EVALUATION ==="
    mkdir -p "${RESULTS_DIR}/benchmarks"

    # Base model (once)
    eval_if_needed "Qwen/Qwen3.5-4B" "qwen35_4b_base" "Base model"

    # All RCCA runs
    for regime in medical math noise25; do
        for mode in "${MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                local run="rcca-${mode}-${regime}-4b-s${seed}"
                local eval_name="rcca_${mode}_${regime}_s${seed}"
                eval_if_needed "./outputs/paper/${run}" "${eval_name}" "${mode} ${regime} s${seed}"
            done
        done
    done

    # Legacy checkpoints (from previous experiments)
    eval_if_needed "./outputs/paper/ce-math-4b" "ce_math_4b" "CE-Math (legacy)"
    eval_if_needed "./outputs/paper/drift-math-4b" "drift_math_4b" "Drift-Math (legacy)"
    eval_if_needed "./outputs/paper/ce-medical-4b" "ce_medical_4b" "CE-Medical (legacy)"
    eval_if_needed "./outputs/paper/drift-medical-4b" "drift_medical_4b" "Drift-Medical (legacy)"

    # MedQA for medical runs
    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            local run="rcca-${mode}-medical-4b-s${seed}"
            local model_path="./outputs/paper/${run}"
            local eval_name="rcca_${mode}_medical_s${seed}"
            local medqa_dir="${RESULTS_DIR}/benchmarks/${eval_name}/medqa"

            if [ ! -d "${model_path}" ]; then continue; fi
            if [ -d "${medqa_dir}" ] && find "${medqa_dir}" -name 'results_*.json' 2>/dev/null | head -1 | grep -q .; then
                echo "[eval] ${eval_name} MedQA — SKIP (exists)"
                continue
            fi

            echo "[eval] ${eval_name} MedQA..."
            mkdir -p "${RESULTS_DIR}/benchmarks/${eval_name}"
            python3 -m lm_eval --model vllm \
                --model_args "pretrained=${model_path},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.9,max_model_len=4096" \
                --tasks medqa_4options \
                --batch_size auto --num_fewshot 0 \
                --output_path "${medqa_dir}" \
                --log_samples 2>&1 | tee "${RESULTS_DIR}/benchmarks/${eval_name}/medqa.log"
        done
    done

    echo "=== Evaluation complete ==="
}

# ══════════════════════════════════════════════════════════════════
# Phase: Summary
# ══════════════════════════════════════════════════════════════════
phase_summary() {
    echo ""
    echo "=== RESULTS SUMMARY ==="

    python3 << 'PYEOF'
import json, os

results_dir = os.environ.get("RESULTS_DIR", "./results/paper")
bench_dir = os.path.join(results_dir, "benchmarks")

def load_metrics(run_name):
    run_dir = os.path.join(bench_dir, run_name)
    data = {}
    if not os.path.isdir(run_dir):
        return data
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                try:
                    with open(os.path.join(root, f)) as fh:
                        jd = json.load(fh)
                    for task, metrics in jd.get("results", {}).items():
                        for m, v in metrics.items():
                            if isinstance(v, (int, float)) and \
                               ("acc" in m.lower() or "match" in m.lower()):
                                data[f"{task}/{m}"] = v
                except Exception:
                    pass
    return data

def get_mmlu(d):
    return d.get("mmlu_pro/exact_match,custom-extract")

def get_medqa(d):
    return d.get("medqa_4options/acc,none") or d.get("medqa/acc,none")

def fmt(v):
    return f"{v*100:.2f}%" if v is not None else "---"

def delta(v, base):
    if v is None or base is None: return "---"
    return f"{(v-base)*100:+.2f}pp"

print("\n" + "=" * 80)
print("  RCCA-TR FULL PAPER RESULTS")
print("=" * 80)

base = load_metrics("qwen35_4b_base")
base_mmlu = get_mmlu(base)
print(f"\nBase Model MMLU-Pro: {fmt(base_mmlu)}")

# ── Medical (Battle C) ──
print("\n" + "-" * 70)
print("  Battle C: Medical SFT — Catastrophic Forgetting")
print("-" * 70)
print(f"{'Mode':<18} {'MMLU-Pro':>10} {'MedQA':>10} {'Δ MMLU':>12}")
print("-" * 55)
print(f"{'Base (no SFT)':<18} {fmt(base_mmlu):>10} {'---':>10} {'---':>12}")

for mode in ["ce", "hardness", "drift_only", "drift"]:
    d = load_metrics(f"rcca_{mode}_medical_s42")
    if not d:
        d = load_metrics(f"rcca_{mode}_medical_4b")  # static config fallback
    mmlu = get_mmlu(d)
    medqa = get_medqa(d)
    marker = " ★" if mode == "drift_only" else ""
    print(f"{mode + marker:<18} {fmt(mmlu):>10} {fmt(medqa):>10} {delta(mmlu, base_mmlu):>12}")

# Also show legacy results
for name, label in [("ce_medical_4b","CE (legacy)"),("drift_medical_4b","Drift (legacy)")]:
    d = load_metrics(name)
    if d:
        mmlu = get_mmlu(d)
        medqa = get_medqa(d)
        print(f"{label:<18} {fmt(mmlu):>10} {fmt(medqa):>10} {delta(mmlu, base_mmlu):>12}")

# ── Math (Battle B) ──
print("\n" + "-" * 70)
print("  Battle B: Math SFT — Reasoning Trade-off")
print("-" * 70)
print(f"{'Mode':<18} {'MMLU-Pro':>10} {'Δ MMLU':>12}")
print("-" * 45)
print(f"{'Base (no SFT)':<18} {fmt(base_mmlu):>10} {'---':>12}")

for mode in ["ce", "hardness", "drift_only", "drift"]:
    d = load_metrics(f"rcca_{mode}_math_s42")
    mmlu = get_mmlu(d)
    if mmlu:
        print(f"{mode:<18} {fmt(mmlu):>10} {delta(mmlu, base_mmlu):>12}")

for name, label in [("ce_math_4b","CE-Math (legacy)"),("drift_math_4b","Drift-Math (legacy)")]:
    d = load_metrics(name)
    if d:
        mmlu = get_mmlu(d)
        print(f"{label:<18} {fmt(mmlu):>10} {delta(mmlu, base_mmlu):>12}")

# ── Noise (Battle A) ──
print("\n" + "-" * 70)
print("  Battle A: Noisy Alignment (25% noise)")
print("-" * 70)
print(f"{'Mode':<18} {'MMLU-Pro':>10} {'Δ MMLU':>12}")
print("-" * 45)
print(f"{'Base (no SFT)':<18} {fmt(base_mmlu):>10} {'---':>12}")

for mode in ["ce", "hardness", "drift_only", "drift"]:
    d = load_metrics(f"rcca_{mode}_noise25_s42")
    mmlu = get_mmlu(d)
    if mmlu:
        print(f"{mode:<18} {fmt(mmlu):>10} {delta(mmlu, base_mmlu):>12}")

print("\n" + "=" * 80)

# Save JSON
all_data = {}
for d in os.listdir(bench_dir) if os.path.isdir(bench_dir) else []:
    m = load_metrics(d)
    if m:
        all_data[d] = m

out_path = os.path.join(results_dir, "rcca_full_results.json")
with open(out_path, "w") as f:
    json.dump(all_data, f, indent=2)
print(f"Full results saved to: {out_path}")

PYEOF

    echo "=== Summary complete ==="
}

# ══════════════════════════════════════════════════════════════════
# Main Dispatch
# ══════════════════════════════════════════════════════════════════
export RESULTS_DIR

case "${PHASE}" in
    data)       phase_data ;;
    train)
        phase_data
        phase_train_medical
        phase_train_math
        phase_train_noise
        ;;
    medical)
        phase_data
        phase_train_medical
        phase_eval
        phase_summary
        ;;
    math)
        phase_data
        phase_train_math
        phase_eval
        phase_summary
        ;;
    noise)
        phase_data
        phase_train_noise
        phase_eval
        phase_summary
        ;;
    eval)       phase_eval ;;
    summary)    phase_summary ;;
    all)
        phase_data
        phase_train_medical
        phase_train_math
        phase_train_noise
        phase_eval
        phase_summary
        ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo ""
        echo "Usage: $0 {all|train|eval|summary|medical|math|noise|data}"
        echo ""
        echo "Phases:"
        echo "  all      — Full pipeline: data → train → eval → summary"
        echo "  train    — Train all 4 modes × 3 regimes"
        echo "  eval     — Evaluate all checkpoints (MMLU-Pro + MedQA)"
        echo "  summary  — Aggregate and print results table"
        echo "  medical  — Medical only (train + eval + summary)"
        echo "  math     — Math only (train + eval + summary)"
        echo "  noise    — Noisy alignment only (train + eval + summary)"
        echo "  data     — Prepare datasets only"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Done! (Phase: ${PHASE})"
echo "============================================"
