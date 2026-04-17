#!/bin/bash
# ============================================================
# Drift-Trust NeurIPS 2026 — Complete Experiment Pipeline
# ============================================================
#
# One-click script to run ALL missing experiments:
#   Phase 1: Data — Create noisy datasets (0%, 10%, 50%, 75%)
#   Phase 2: Train — Train CE + Drift at each noise level
#   Phase 3: Eval  — MMLU-Pro + IFEval for all checkpoints
#   Phase 4: Efficiency — Wall time, VRAM measurement
#   Phase 5: Summary — Aggregate results into tables
#
# Usage:
#   bash run_full_experiments.sh                # Run everything
#   bash run_full_experiments.sh data           # Phase 1 only
#   bash run_full_experiments.sh train          # Phase 2 only
#   bash run_full_experiments.sh eval           # Phase 3 only
#   bash run_full_experiments.sh efficiency     # Phase 4 only
#   bash run_full_experiments.sh summary        # Phase 5 only
#
# Environment:
#   Uses hrun for GPU job submission.
#   Expects vLLM to be pre-patched for Qwen3.5 visual weights issue.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAPER_SCRIPTS="${PROJECT_DIR}/scripts/paper"
PAPER_CONFIGS="${PROJECT_DIR}/examples/paper"
DATA_DIR="${PROJECT_DIR}/data/paper"
RESULTS_DIR="${PROJECT_DIR}/results/paper"

PHASE="${1:-all}"

# Noise levels to sweep
NOISE_LEVELS=(0 10 25 50 75)

# GPU settings for hrun
HRUN_ARGS="-c 20 -m 50 -G -g RTX6000AdaGeneration"

echo "============================================"
echo "  Drift-Trust NeurIPS 2026"
echo "  Full Experiment Pipeline"
echo "  Phase: ${PHASE}"
echo "  Noise levels: ${NOISE_LEVELS[*]}"
echo "============================================"

# ══════════════════════════════════════════════
# Phase 1: Data Preparation
# ══════════════════════════════════════════════
phase_data() {
    echo ""
    echo "=== PHASE 1: Data Preparation ==="
    mkdir -p "${DATA_DIR}"

    for noise in "${NOISE_LEVELS[@]}"; do
        local pct="${noise}"
        local fname="ultrafeedback_noisy_${pct}pct_10k.jsonl"

        if [ -f "${DATA_DIR}/${fname}" ]; then
            echo "[data] ${fname} already exists, skipping."
            continue
        fi

        echo "[data] Creating UltraFeedback with ${pct}% noise..."
        local ratio=$(awk "BEGIN {printf \"%.2f\", ${pct}/100}")
        python "${PAPER_SCRIPTS}/create_noisy_ultrafeedback.py" \
            --output_dir "${DATA_DIR}" \
            --num_samples 10000 \
            --noise_ratio "${ratio}" \
            --seed 42

        # Handle 0% edge case (noise_ratio=0.0)
        if [ "${pct}" -eq 0 ]; then
            # Rename if script produces different filename
            local expected="${DATA_DIR}/ultrafeedback_noisy_0pct_10k.jsonl"
            if [ ! -f "${expected}" ]; then
                # The script uses int(0.0*100)=0, so should be fine
                echo "[data] WARNING: 0% noise file not found at expected path"
            fi
        fi
    done

    echo "Phase 1 complete."
}

# ══════════════════════════════════════════════
# Phase 2: Training
# ══════════════════════════════════════════════

# Generate a YAML config on the fly
generate_config() {
    local noise_pct="$1"
    local method="$2"  # "ce" or "drift"
    local seed="${3:-42}"
    local config_dir="${PAPER_CONFIGS}/sweep"
    mkdir -p "${config_dir}"

    local data_file="ultrafeedback_noisy_${noise_pct}pct_10k.jsonl"
    local config_path="${config_dir}/${method}_noise${noise_pct}_seed${seed}.yaml"
    local output_dir="./outputs/paper/${method}-noise${noise_pct}-seed${seed}"
    local prep_dir="./prepared_data/paper/${method}_noise${noise_pct}_seed${seed}"

    if [ "${method}" == "drift" ]; then
        cat > "${config_path}" << EOF
# Auto-generated: Drift-Trust, ${noise_pct}% noise, seed ${seed}
base_model: Qwen/Qwen3.5-4B
low_cpu_mem_usage: true

plugins:
  - axolotl.integrations.drift.DriftPlugin
  - axolotl.integrations.liger.LigerPlugin

liger_rms_norm: true
liger_glu_activation: true

# Drift-Trust hyperparameters (recommended)
drift_trainer: true
drift_reliability_beta: 0.5
drift_reliability_tau: 1.0
drift_epsilon_min: 0.01
drift_epsilon_max: 1.0
drift_kl_lambda: 4.0
drift_use_smooth_objective: true
drift_ema_decay: 0.99
drift_gamma: 3.0
drift_anchor_weight: 0.1

datasets:
  - path: ./data/paper/${data_file}
    ds_type: json
    type: chat_template
    field_messages: conversations
    split: train

dataset_prepared_path: ${prep_dir}
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
output_dir: ${output_dir}

logging_steps: 1
EOF
    else
        cat > "${config_path}" << EOF
# Auto-generated: CE baseline, ${noise_pct}% noise, seed ${seed}
base_model: Qwen/Qwen3.5-4B
low_cpu_mem_usage: true

plugins:
  - axolotl.integrations.liger.LigerPlugin

liger_rms_norm: true
liger_glu_activation: true

datasets:
  - path: ./data/paper/${data_file}
    ds_type: json
    type: chat_template
    field_messages: conversations
    split: train

dataset_prepared_path: ${prep_dir}
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
output_dir: ${output_dir}

logging_steps: 1
EOF
    fi

    echo "${config_path}"
}

phase_train() {
    echo ""
    echo "=== PHASE 2: Training ==="
    echo "  Noise levels: ${NOISE_LEVELS[*]}"
    echo "  Methods: CE, Drift-Trust"
    echo ""

    local total_runs=$(( ${#NOISE_LEVELS[@]} * 2 ))
    local run_num=0

    for noise in "${NOISE_LEVELS[@]}"; do
        for method in ce drift; do
            run_num=$((run_num + 1))
            local output_dir="./outputs/paper/${method}-noise${noise}-seed42"

            # Skip if checkpoint already exists
            if [ -d "${output_dir}" ] && [ -f "${output_dir}/config.json" ]; then
                echo "[${run_num}/${total_runs}] ${method} noise=${noise}% — SKIP (exists)"
                continue
            fi

            echo "[${run_num}/${total_runs}] Training ${method} with ${noise}% noise..."

            local config=$(generate_config "${noise}" "${method}" 42)
            echo "  Config: ${config}"
            echo "  Output: ${output_dir}"

            python3 -m accelerate.commands.launch -m axolotl.cli.train "${config}"

            echo "[${run_num}/${total_runs}] ${method} noise=${noise}% — DONE"
            echo ""
        done
    done

    echo ""
    echo "=== PHASE 2: Training complete ==="
    echo "Trained ${total_runs} models across ${#NOISE_LEVELS[@]} noise levels."
}

# ══════════════════════════════════════════════
# Phase 3: Evaluation
# ══════════════════════════════════════════════
phase_eval() {
    echo ""
    echo "=== PHASE 3: Evaluation ==="
    mkdir -p "${RESULTS_DIR}/benchmarks"

    # 1. Base model (only needs to be run once)
    local base_result="${RESULTS_DIR}/benchmarks/base"
    if [ ! -d "${base_result}" ] || [ -z "$(find ${base_result} -name 'results_*.json' 2>/dev/null)" ]; then
        echo "[eval] Base model..."
        bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" "Qwen/Qwen3.5-4B" "base"
    else
        echo "[eval] Base model — SKIP (exists)"
    fi

    # 2. All noise-level × method combinations
    for noise in "${NOISE_LEVELS[@]}"; do
        for method in ce drift; do
            local run_name="${method}_noise${noise}"
            local model_path="./outputs/paper/${method}-noise${noise}-seed42"
            local result_dir="${RESULTS_DIR}/benchmarks/${run_name}"

            if [ ! -d "${model_path}" ]; then
                echo "[eval] ${run_name} — SKIP (no checkpoint at ${model_path})"
                continue
            fi

            if [ -d "${result_dir}" ] && [ -n "$(find ${result_dir} -name 'results_*.json' 2>/dev/null)" ]; then
                echo "[eval] ${run_name} — SKIP (results exist)"
                continue
            fi

            echo "[eval] ${run_name}..."
            bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" "${model_path}" "${run_name}"
            echo "[eval] ${run_name} — DONE"
        done
    done

    # 3. Ablation: conservative + per-sample (already trained)
    for ablation in "drift-noisy-4b:drift_conservative" "drift-noisy-4b-persample:drift_persample"; do
        local model_path="./outputs/paper/${ablation%%:*}"
        local run_name="${ablation##*:}"
        local result_dir="${RESULTS_DIR}/benchmarks/${run_name}"

        if [ ! -d "${model_path}" ]; then
            echo "[eval] ${run_name} — SKIP (no checkpoint)"
            continue
        fi

        if [ -d "${result_dir}" ] && [ -n "$(find ${result_dir} -name 'results_*.json' 2>/dev/null)" ]; then
            echo "[eval] ${run_name} — SKIP (results exist)"
            continue
        fi

        echo "[eval] ${run_name} (ablation)..."
        bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" "${model_path}" "${run_name}"
    done

    echo ""
    echo "=== PHASE 3: Evaluation complete ==="
}

# ══════════════════════════════════════════════
# Phase 4: Efficiency Measurement
# ══════════════════════════════════════════════
phase_efficiency() {
    echo ""
    echo "=== PHASE 4: Efficiency Measurement ==="
    mkdir -p "${RESULTS_DIR}/efficiency"

    # Build list of configs to measure (CE + Drift default at 25% noise)
    local ce_config="${PAPER_CONFIGS}/ce_noisy_4b.yaml"
    local drift_config="${PAPER_CONFIGS}/drift_noisy_4b.yaml"

    if [ -f "${PAPER_SCRIPTS}/measure_efficiency.py" ]; then
        python3 "${PAPER_SCRIPTS}/measure_efficiency.py" \
            --configs "${ce_config}" "${drift_config}" \
            --max_steps 50 \
            --output_dir "${RESULTS_DIR}/efficiency"
    else
        echo "WARNING: measure_efficiency.py not found, skipping."
    fi

    echo "=== PHASE 4 complete ==="
}

# ══════════════════════════════════════════════
# Phase 5: Summary — Aggregate all results
# ══════════════════════════════════════════════
phase_summary() {
    echo ""
    echo "=== PHASE 5: Results Summary ==="
    mkdir -p "${RESULTS_DIR}"

    python3 << 'SUMMARY_EOF'
import json, os, glob

results_dir = os.environ.get("RESULTS_DIR", "./results/paper")
bench_dir = os.path.join(results_dir, "benchmarks")

print("\n" + "=" * 80)
print("  DRIFT-TRUST FULL EXPERIMENT RESULTS")
print("=" * 80)

# Debug: show what directories exist
print(f"\nScanning: {bench_dir}")
if os.path.isdir(bench_dir):
    subdirs = sorted(os.listdir(bench_dir))
    print(f"Found {len(subdirs)} result directories: {subdirs}")
else:
    print(f"WARNING: {bench_dir} does not exist!")
    subdirs = []

# Collect all results — search recursively for results_*.json
all_results = {}
for run_dir in sorted(glob.glob(os.path.join(bench_dir, "*"))):
    if not os.path.isdir(run_dir):
        continue
    run_name = os.path.basename(run_dir)
    run_data = {}

    for task in ["mmlu_pro", "ifeval"]:
        # Search recursively: results might be nested
        task_dir = os.path.join(run_dir, task)
        result_files = glob.glob(os.path.join(task_dir, "**", "results_*.json"), recursive=True)
        if not result_files:
            # Also try flat structure
            result_files = glob.glob(os.path.join(task_dir, "results_*.json"))
        if not result_files:
            continue
        try:
            with open(result_files[0]) as f:
                data = json.load(f)
            results = data.get("results", {})
            for task_name, metrics in results.items():
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and ("acc" in metric.lower() or "score" in metric.lower()):
                        run_data[f"{task_name}/{metric}"] = value
        except Exception as e:
            print(f"  WARNING: Failed to parse {result_files[0]}: {e}")

    if run_data:
        all_results[run_name] = run_data
        print(f"  ✓ {run_name}: {len(run_data)} metrics loaded")
    else:
        print(f"  ✗ {run_name}: no metrics found")

if not all_results:
    print("\nNo results found! Check that benchmark results exist in:")
    print(f"  {bench_dir}/<run_name>/mmlu_pro/results_*.json")
    print(f"  {bench_dir}/<run_name>/ifeval/results_*.json")

# --- Main Table: all configs ---
print("\n--- Main Results: MMLU-Pro + IFEval ---\n")
header = f"{'Config':<30} {'MMLU-Pro':>10} {'IFEval Strict':>15} {'IFEval Loose':>15}"
print(header)
print("-" * len(header))

for name, data in sorted(all_results.items()):
    mmlu = data.get("mmlu_pro/exact_match,custom-extract", None)
    ifeval_s = data.get("ifeval/prompt_level_strict_acc,none", None)
    ifeval_l = data.get("ifeval/prompt_level_loose_acc,none", None)
    mmlu_str = f"{mmlu:.4f}" if mmlu is not None else "N/A"
    ifeval_s_str = f"{ifeval_s:.4f}" if ifeval_s is not None else "N/A"
    ifeval_l_str = f"{ifeval_l:.4f}" if ifeval_l is not None else "N/A"
    print(f"{name:<30} {mmlu_str:>10} {ifeval_s_str:>15} {ifeval_l_str:>15}")

# --- Noise Sweep Table ---
# Map old naming conventions to noise levels
NAME_TO_NOISE = {
    # New sweep naming
    "ce_noise0": ("ce", 0), "drift_noise0": ("drift", 0),
    "ce_noise10": ("ce", 10), "drift_noise10": ("drift", 10),
    "ce_noise25": ("ce", 25), "drift_noise25": ("drift", 25),
    "ce_noise50": ("ce", 50), "drift_noise50": ("drift", 50),
    "ce_noise75": ("ce", 75), "drift_noise75": ("drift", 75),
    # Old naming (from previous runs, all 25% noise)
    "ce_noisy_4b": ("ce", 25),
    "qwen35_4b_base": ("base", -1),
    # Drift-Trust variants at 25% noise
    "drift_noisy_4b": ("drift_conservative", 25),
    "drift_noisy_aggressive": ("drift", 25),
    "drift_noisy_persample": ("drift_persample", 25),
    "drift_conservative": ("drift_conservative", 25),
    "drift_persample": ("drift_persample", 25),
}

# Group results by (method, noise_level)
grouped = {}
for name, data in all_results.items():
    method, noise = NAME_TO_NOISE.get(name, (name, -1))
    grouped[(method, noise)] = data

print("\n--- Noise Level Sweep ---\n")
print(f"{'Noise %':<10} {'CE MMLU':>10} {'Drift MMLU':>12} {'CE IFEval':>12} {'Drift IFEval':>14} {'Δ MMLU':>8} {'Δ IFEval':>10}")
print("-" * 80)

def fmt(v):
    return f"{v:.4f}" if v is not None else "---"

for noise in [0, 10, 25, 50, 75]:
    ce = grouped.get(("ce", noise), {})
    dr = grouped.get(("drift", noise), {})

    ce_mmlu = ce.get("mmlu_pro/exact_match,custom-extract")
    dr_mmlu = dr.get("mmlu_pro/exact_match,custom-extract")
    ce_if = ce.get("ifeval/prompt_level_strict_acc,none")
    dr_if = dr.get("ifeval/prompt_level_strict_acc,none")

    delta_mmlu = f"{(dr_mmlu - ce_mmlu):+.4f}" if (dr_mmlu is not None and ce_mmlu is not None) else "---"
    delta_if = f"{(dr_if - ce_if):+.4f}" if (dr_if is not None and ce_if is not None) else "---"

    print(f"{noise:>5}%    {fmt(ce_mmlu):>10} {fmt(dr_mmlu):>12} {fmt(ce_if):>12} {fmt(dr_if):>14} {delta_mmlu:>8} {delta_if:>10}")

# --- Ablation Table ---
print("\n--- Ablation (25% noise) ---\n")
print(f"{'Variant':<25} {'MMLU-Pro':>10} {'IFEval Strict':>15}")
print("-" * 55)

ablation_order = [("base", -1, "Base (no fine-tune)"), ("ce", 25, "Standard CE"), ("drift", 25, "Drift-Trust (Ours)"),
                  ("drift_conservative", 25, "  Conservative (w=0.5)"), ("drift_persample", 25, "  Sample-level")]
for method, noise, label in ablation_order:
    data = grouped.get((method, noise), {})
    mmlu = data.get("mmlu_pro/exact_match,custom-extract")
    ifeval_s = data.get("ifeval/prompt_level_strict_acc,none")
    if mmlu is not None or ifeval_s is not None:
        print(f"{label:<25} {fmt(mmlu):>10} {fmt(ifeval_s):>15}")

# Save to JSON
summary_path = os.path.join(results_dir, "full_results_summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nFull results saved to: {summary_path}")

SUMMARY_EOF

    echo ""
    echo "=== PHASE 5 complete ==="
}

# ══════════════════════════════════════════════
# Main dispatch
# ══════════════════════════════════════════════
export RESULTS_DIR

case "${PHASE}" in
    data)       phase_data ;;
    train)      phase_train ;;
    eval)       phase_eval ;;
    efficiency) phase_efficiency ;;
    summary)    phase_summary ;;
    all)
        phase_data
        phase_train
        phase_eval
        phase_efficiency
        phase_summary
        ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo "Usage: $0 {data|train|eval|efficiency|summary|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Pipeline complete! (Phase: ${PHASE})"
echo "============================================"
