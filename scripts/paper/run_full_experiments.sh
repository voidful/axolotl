#!/bin/bash
# ============================================================
# drift-loss paper runner
# ============================================================
#
# One script for the paper's three post-training regimes:
#   1. noisy alignment
#   2. factual / domain specialization
#   3. reasoning-path rewiring
#
# The runner covers:
#   - data preparation
#   - training for all configured baselines
#   - benchmark / task evaluation
#   - efficiency spot-check
#   - summary aggregation
#
# Usage:
#   bash run_full_experiments.sh
#   bash run_full_experiments.sh data
#   bash run_full_experiments.sh train
#   bash run_full_experiments.sh eval
#   bash run_full_experiments.sh summary
#   bash run_full_experiments.sh efficiency
#
# Useful overrides:
#   PAPER_SEEDS="42 43 44" bash run_full_experiments.sh train
#   NOISE_SWEEP_SEEDS="42" bash run_full_experiments.sh eval
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAPER_SCRIPTS="${PROJECT_DIR}/scripts/paper"
PAPER_CONFIGS="${PROJECT_DIR}/examples/paper"
GENERATED_CONFIG_DIR="${PAPER_CONFIGS}/generated"
DATA_DIR="${PROJECT_DIR}/data/paper"
RESULTS_DIR="${PROJECT_DIR}/results/paper"

PHASE="${1:-all}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
PRIMARY_NOISE_LEVEL="${PRIMARY_NOISE_LEVEL:-25}"

read -r -a PAPER_SEEDS <<< "${PAPER_SEEDS:-42 43 44}"
read -r -a NOISE_SWEEP_SEEDS <<< "${NOISE_SWEEP_SEEDS:-42}"
read -r -a NOISE_SWEEP_LEVELS <<< "${NOISE_SWEEP_LEVELS:-0 10 25 50 75}"

NOISY_BASELINES=(ce hardness focal eaft entropy_high entropy_low l2sp ewc lwf drift_loss)
DOMAIN_BASELINES=(ce hardness focal eaft entropy_high entropy_low l2sp ewc lwf drift_loss)
REASONING_BASELINES=(ce hardness focal eaft entropy_high entropy_low l2sp ewc lwf drift_loss)
DRIFT_ABLATIONS=(drift_loss_conservative drift_loss_attached)
NOISE_SWEEP_METHODS=(ce drift_loss)

mkdir -p "${GENERATED_CONFIG_DIR}" "${DATA_DIR}" "${RESULTS_DIR}"

echo "============================================"
echo "  drift-loss paper runner"
echo "  Phase: ${PHASE}"
echo "  Base model: ${BASE_MODEL}"
echo "  Paper seeds: ${PAPER_SEEDS[*]}"
echo "  Noise sweep seeds: ${NOISE_SWEEP_SEEDS[*]}"
echo "  Primary noise: ${PRIMARY_NOISE_LEVEL}%"
echo "============================================"

slugify_run_name() {
    local run_name="$1"
    echo "${run_name//__/-}"
}

config_path_for_run() {
    local run_name="$1"
    echo "${GENERATED_CONFIG_DIR}/$(slugify_run_name "${run_name}").yaml"
}

output_dir_for_run() {
    local run_name="$1"
    echo "./outputs/paper/$(slugify_run_name "${run_name}")"
}

prepared_dir_for_run() {
    local run_name="$1"
    echo "./prepared_data/paper/$(slugify_run_name "${run_name}")"
}

display_name_for_method() {
    case "$1" in
        ce) echo "CE" ;;
        hardness) echo "Hardness" ;;
        focal) echo "Focal" ;;
        eaft) echo "EAFT" ;;
        entropy_high) echo "High-Entropy Only" ;;
        entropy_low) echo "Low-Entropy Only" ;;
        l2sp) echo "L2-SP" ;;
        ewc) echo "EWC" ;;
        lwf) echo "LwF" ;;
        drift_loss) echo "drift-loss" ;;
        drift_loss_conservative) echo "drift-loss conservative" ;;
        drift_loss_attached) echo "drift-loss attached" ;;
        *) echo "$1" ;;
    esac
}

emit_method_block() {
    case "$1" in
        ce)
            cat <<'EOF'
plugins:
  - axolotl.integrations.liger.LigerPlugin
EOF
            ;;
        hardness)
            cat <<'EOF'
plugins:
  - axolotl.integrations.hardness.HardnessPlugin
  - axolotl.integrations.liger.LigerPlugin

hardness_trainer: true
hardness_tau_p: 2.0
hardness_T_p: 1.0
hardness_w_min: 0.05
EOF
            ;;
        focal)
            cat <<'EOF'
plugins:
  - axolotl.integrations.focal.FocalPlugin
  - axolotl.integrations.liger.LigerPlugin

focal_trainer: true
focal_gamma: 2.0
EOF
            ;;
        eaft)
            cat <<'EOF'
plugins:
  - axolotl.integrations.liger.LigerPlugin

use_eaft: true
eaft_alpha: 1.0
eaft_k: 20
EOF
            ;;
        entropy_high)
            cat <<'EOF'
plugins:
  - axolotl.integrations.entropy_focus.EntropyFocusPlugin
  - axolotl.integrations.liger.LigerPlugin

entropy_focus_trainer: true
entropy_focus_mode: high
EOF
            ;;
        entropy_low)
            cat <<'EOF'
plugins:
  - axolotl.integrations.entropy_focus.EntropyFocusPlugin
  - axolotl.integrations.liger.LigerPlugin

entropy_focus_trainer: true
entropy_focus_mode: low
EOF
            ;;
        l2sp)
            cat <<'EOF'
plugins:
  - axolotl.integrations.l2sp.L2SPPlugin
  - axolotl.integrations.liger.LigerPlugin

l2sp_trainer: true
l2sp_lambda: 1.0e-4
EOF
            ;;
        ewc)
            cat <<'EOF'
plugins:
  - axolotl.integrations.ewc.EWCPlugin
  - axolotl.integrations.liger.LigerPlugin

ewc_trainer: true
ewc_lambda: 1.0e-4
ewc_fisher_n_batches: 32
EOF
            ;;
        lwf)
            cat <<EOF
plugins:
  - axolotl.integrations.lwf.LWFPlugin
  - axolotl.integrations.liger.LigerPlugin

lwf_trainer: true
lwf_teacher_model: ${BASE_MODEL}
lwf_ce_alpha: 1.0
lwf_alpha: 1.0
lwf_temperature: 2.0
EOF
            ;;
        drift_loss)
            cat <<'EOF'
plugins:
  - axolotl.integrations.drift.DriftPlugin
  - axolotl.integrations.liger.LigerPlugin

drift_trainer: true
drift_reference_mode: ema
drift_ema_decay: 0.99
drift_gamma: 2.0
drift_detach_weights: true
EOF
            ;;
        drift_loss_conservative)
            cat <<'EOF'
plugins:
  - axolotl.integrations.drift.DriftPlugin
  - axolotl.integrations.liger.LigerPlugin

drift_trainer: true
drift_reference_mode: ema
drift_ema_decay: 0.999
drift_gamma: 1.0
drift_detach_weights: true
EOF
            ;;
        drift_loss_attached)
            cat <<'EOF'
plugins:
  - axolotl.integrations.drift.DriftPlugin
  - axolotl.integrations.liger.LigerPlugin

drift_trainer: true
drift_reference_mode: ema
drift_ema_decay: 0.99
drift_gamma: 2.0
drift_detach_weights: false
EOF
            ;;
        *)
            echo "Unknown method: $1" >&2
            return 1
            ;;
    esac
}

generate_config() {
    local run_name="$1"
    local regime="$2"
    local method="$3"
    local dataset_path="$4"
    local seed="$5"

    local config_path
    local output_dir
    local prepared_dir

    config_path="$(config_path_for_run "${run_name}")"
    output_dir="$(output_dir_for_run "${run_name}")"
    prepared_dir="$(prepared_dir_for_run "${run_name}")"

    {
        cat <<EOF
# Auto-generated for ${regime}
# Method: $(display_name_for_method "${method}")
# Run: ${run_name}

base_model: ${BASE_MODEL}
low_cpu_mem_usage: true

EOF
        emit_method_block "${method}"
        cat <<EOF

liger_rms_norm: true
liger_glu_activation: true

datasets:
  - path: ${dataset_path}
    ds_type: json
    type: chat_template
    field_messages: conversations
    split: train

dataset_prepared_path: ${prepared_dir}
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
    } > "${config_path}"

    echo "${config_path}"
}

train_config_if_needed() {
    local config_path="$1"
    local output_dir="$2"
    local label="$3"

    if [ -d "${PROJECT_DIR}/${output_dir#./}" ] && [ -f "${PROJECT_DIR}/${output_dir#./}/config.json" ]; then
        echo "[train] ${label} — SKIP (checkpoint exists)"
        return 0
    fi

    echo "[train] ${label}"
    python3 -m accelerate.commands.launch -m axolotl.cli.train "${config_path}"
}

run_bench_eval_if_needed() {
    local model_path="$1"
    local run_name="$2"
    local result_dir="${RESULTS_DIR}/benchmarks/${run_name}"

    if [ ! -d "${model_path}" ] && [[ "${model_path}" != Qwen/* ]]; then
        echo "[eval] ${run_name} — SKIP (missing model)"
        return 0
    fi

    if [ -d "${result_dir}" ] && find "${result_dir}" -name 'results_*.json' 2>/dev/null | head -1 | grep -q .; then
        echo "[eval] ${run_name} — SKIP (benchmarks exist)"
        return 0
    fi

    echo "[eval] ${run_name} — benchmarks"
    bash "${PAPER_SCRIPTS}/eval_benchmarks.sh" "${model_path}" "${run_name}"
}

run_medqa_eval_if_needed() {
    local model_path="$1"
    local run_name="$2"
    local output_path="${RESULTS_DIR}/benchmarks/${run_name}/medqa"

    if [ ! -d "${model_path}" ] && [[ "${model_path}" != Qwen/* ]]; then
        echo "[eval] ${run_name} — SKIP MedQA (missing model)"
        return 0
    fi

    if [ -d "${output_path}" ] && find "${output_path}" -name 'results_*.json' 2>/dev/null | head -1 | grep -q .; then
        echo "[eval] ${run_name} — SKIP MedQA (exists)"
        return 0
    fi

    mkdir -p "${RESULTS_DIR}/benchmarks/${run_name}"
    echo "[eval] ${run_name} — MedQA"
    python3 -m lm_eval --model vllm \
        --model_args "pretrained=${model_path},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.9,max_model_len=4096" \
        --tasks medqa_4options \
        --batch_size auto \
        --num_fewshot 0 \
        --output_path "${output_path}" \
        --log_samples 2>&1 | tee "${RESULTS_DIR}/benchmarks/${run_name}/medqa.log"
}

run_math_eval_if_needed() {
    local model_path="$1"
    local run_name="$2"
    local result_file="${RESULTS_DIR}/math500_passk_${run_name}.json"

    if [ ! -d "${model_path}" ] && [[ "${model_path}" != Qwen/* ]]; then
        echo "[eval] ${run_name} — SKIP MATH-500 (missing model)"
        return 0
    fi

    if [ -f "${result_file}" ]; then
        echo "[eval] ${run_name} — SKIP MATH-500 (exists)"
        return 0
    fi

    echo "[eval] ${run_name} — MATH-500 Pass@k"
    python3 "${PAPER_SCRIPTS}/eval_math_passk.py" \
        --model_path "${model_path}" \
        --k 64 \
        --temperature 0.7 \
        --output_dir "${RESULTS_DIR}" \
        --run_name "${run_name}"
}

phase_data() {
    echo ""
    echo "=== DATA ==="

    for noise in "${NOISE_SWEEP_LEVELS[@]}"; do
        local ratio
        ratio="$(awk "BEGIN {printf \"%.2f\", ${noise}/100}")"
        local noisy_file="${DATA_DIR}/ultrafeedback_noisy_${noise}pct_10k.jsonl"
        if [ -f "${noisy_file}" ]; then
            echo "[data] noisy ${noise}% — SKIP"
        else
            echo "[data] noisy alignment ${noise}%"
            python3 "${PAPER_SCRIPTS}/create_noisy_ultrafeedback.py" \
                --output_dir "${DATA_DIR}" \
                --num_samples 10000 \
                --noise_ratio "${ratio}" \
                --seed 42
        fi
    done

    if [ -f "${DATA_DIR}/medical_flashcards_10k.jsonl" ]; then
        echo "[data] medical — SKIP"
    else
        echo "[data] domain specialization"
        python3 "${PAPER_SCRIPTS}/create_medical_sft.py" \
            --output_dir "${DATA_DIR}" \
            --num_samples 10000 \
            --seed 42
    fi

    if [ -f "${DATA_DIR}/numina_math_cot_10k.jsonl" ]; then
        echo "[data] reasoning — SKIP"
    else
        echo "[data] reasoning rewiring"
        python3 "${PAPER_SCRIPTS}/sample_numina_math.py" \
            --output_dir "${DATA_DIR}" \
            --num_samples 10000 \
            --seed 42
    fi
}

phase_train_noisy() {
    echo ""
    echo "=== TRAIN: noisy alignment ==="

    local dataset_rel="./data/paper/ultrafeedback_noisy_${PRIMARY_NOISE_LEVEL}pct_10k.jsonl"
    for seed in "${PAPER_SEEDS[@]}"; do
        for method in "${NOISY_BASELINES[@]}"; do
            local run_name="noisy__${method}__seed${seed}"
            local config_path
            local output_dir
            config_path="$(generate_config "${run_name}" "noisy alignment" "${method}" "${dataset_rel}" "${seed}")"
            output_dir="$(output_dir_for_run "${run_name}")"
            train_config_if_needed "${config_path}" "${output_dir}" "${run_name}"
        done
    done

    for seed in "${NOISE_SWEEP_SEEDS[@]}"; do
        for noise in "${NOISE_SWEEP_LEVELS[@]}"; do
            local sweep_dataset="./data/paper/ultrafeedback_noisy_${noise}pct_10k.jsonl"
            for method in "${NOISE_SWEEP_METHODS[@]}"; do
                local run_name="noisy_sweep__${method}__noise${noise}__seed${seed}"
                local config_path
                local output_dir
                config_path="$(generate_config "${run_name}" "noisy alignment sweep" "${method}" "${sweep_dataset}" "${seed}")"
                output_dir="$(output_dir_for_run "${run_name}")"
                train_config_if_needed "${config_path}" "${output_dir}" "${run_name}"
            done
        done
    done

    local ablation_seed="${PAPER_SEEDS[0]}"
    for method in "${DRIFT_ABLATIONS[@]}"; do
        local run_name="noisy_ablation__${method}__seed${ablation_seed}"
        local config_path
        local output_dir
        config_path="$(generate_config "${run_name}" "noisy alignment ablation" "${method}" "${dataset_rel}" "${ablation_seed}")"
        output_dir="$(output_dir_for_run "${run_name}")"
        train_config_if_needed "${config_path}" "${output_dir}" "${run_name}"
    done
}

phase_train_domain() {
    echo ""
    echo "=== TRAIN: factual / domain specialization ==="

    local dataset_rel="./data/paper/medical_flashcards_10k.jsonl"
    for seed in "${PAPER_SEEDS[@]}"; do
        for method in "${DOMAIN_BASELINES[@]}"; do
            local run_name="domain__${method}__seed${seed}"
            local config_path
            local output_dir
            config_path="$(generate_config "${run_name}" "domain specialization" "${method}" "${dataset_rel}" "${seed}")"
            output_dir="$(output_dir_for_run "${run_name}")"
            train_config_if_needed "${config_path}" "${output_dir}" "${run_name}"
        done
    done
}

phase_train_reasoning() {
    echo ""
    echo "=== TRAIN: reasoning-path rewiring ==="

    local dataset_rel="./data/paper/numina_math_cot_10k.jsonl"
    for seed in "${PAPER_SEEDS[@]}"; do
        for method in "${REASONING_BASELINES[@]}"; do
            local run_name="reasoning__${method}__seed${seed}"
            local config_path
            local output_dir
            config_path="$(generate_config "${run_name}" "reasoning rewiring" "${method}" "${dataset_rel}" "${seed}")"
            output_dir="$(output_dir_for_run "${run_name}")"
            train_config_if_needed "${config_path}" "${output_dir}" "${run_name}"
        done
    done
}

phase_train() {
    phase_train_noisy
    phase_train_domain
    phase_train_reasoning
}

phase_eval_base() {
    echo ""
    echo "=== EVAL: base model anchors ==="
    run_bench_eval_if_needed "${BASE_MODEL}" "base__qwen35_4b"
    run_medqa_eval_if_needed "${BASE_MODEL}" "base__qwen35_4b"
    run_math_eval_if_needed "${BASE_MODEL}" "base__qwen35_4b"
}

phase_eval_noisy() {
    echo ""
    echo "=== EVAL: noisy alignment ==="

    for seed in "${PAPER_SEEDS[@]}"; do
        for method in "${NOISY_BASELINES[@]}"; do
            local run_name="noisy__${method}__seed${seed}"
            local model_path="${PROJECT_DIR}/$(output_dir_for_run "${run_name}" | sed 's#^\./##')"
            run_bench_eval_if_needed "${model_path}" "${run_name}"
        done
    done

    for seed in "${NOISE_SWEEP_SEEDS[@]}"; do
        for noise in "${NOISE_SWEEP_LEVELS[@]}"; do
            for method in "${NOISE_SWEEP_METHODS[@]}"; do
                local run_name="noisy_sweep__${method}__noise${noise}__seed${seed}"
                local model_path="${PROJECT_DIR}/$(output_dir_for_run "${run_name}" | sed 's#^\./##')"
                run_bench_eval_if_needed "${model_path}" "${run_name}"
            done
        done
    done

    local ablation_seed="${PAPER_SEEDS[0]}"
    for method in "${DRIFT_ABLATIONS[@]}"; do
        local run_name="noisy_ablation__${method}__seed${ablation_seed}"
        local model_path="${PROJECT_DIR}/$(output_dir_for_run "${run_name}" | sed 's#^\./##')"
        run_bench_eval_if_needed "${model_path}" "${run_name}"
    done
}

phase_eval_domain() {
    echo ""
    echo "=== EVAL: domain specialization ==="

    for seed in "${PAPER_SEEDS[@]}"; do
        for method in "${DOMAIN_BASELINES[@]}"; do
            local run_name="domain__${method}__seed${seed}"
            local model_path="${PROJECT_DIR}/$(output_dir_for_run "${run_name}" | sed 's#^\./##')"
            run_bench_eval_if_needed "${model_path}" "${run_name}"
            run_medqa_eval_if_needed "${model_path}" "${run_name}"
        done
    done
}

phase_eval_reasoning() {
    echo ""
    echo "=== EVAL: reasoning-path rewiring ==="

    for seed in "${PAPER_SEEDS[@]}"; do
        for method in "${REASONING_BASELINES[@]}"; do
            local run_name="reasoning__${method}__seed${seed}"
            local model_path="${PROJECT_DIR}/$(output_dir_for_run "${run_name}" | sed 's#^\./##')"
            run_bench_eval_if_needed "${model_path}" "${run_name}"
            run_math_eval_if_needed "${model_path}" "${run_name}"
        done
    done
}

phase_eval() {
    phase_eval_base
    phase_eval_noisy
    phase_eval_domain
    phase_eval_reasoning
}

phase_efficiency() {
    echo ""
    echo "=== EFFICIENCY ==="

    local seed="${NOISE_SWEEP_SEEDS[0]}"
    local ce_run="noisy_sweep__ce__noise${PRIMARY_NOISE_LEVEL}__seed${seed}"
    local drift_run="noisy_sweep__drift_loss__noise${PRIMARY_NOISE_LEVEL}__seed${seed}"
    local ce_config="${GENERATED_CONFIG_DIR}/$(slugify_run_name "${ce_run}").yaml"
    local drift_config="${GENERATED_CONFIG_DIR}/$(slugify_run_name "${drift_run}").yaml"

    if [ ! -f "${ce_config}" ]; then
        generate_config "${ce_run}" "noisy alignment sweep" "ce" "./data/paper/ultrafeedback_noisy_${PRIMARY_NOISE_LEVEL}pct_10k.jsonl" "${seed}" >/dev/null
    fi
    if [ ! -f "${drift_config}" ]; then
        generate_config "${drift_run}" "noisy alignment sweep" "drift_loss" "./data/paper/ultrafeedback_noisy_${PRIMARY_NOISE_LEVEL}pct_10k.jsonl" "${seed}" >/dev/null
    fi

    if [ -f "${PAPER_SCRIPTS}/measure_efficiency.py" ]; then
        python3 "${PAPER_SCRIPTS}/measure_efficiency.py" \
            --configs "${ce_config}" "${drift_config}" \
            --max_steps 50 \
            --output_dir "${RESULTS_DIR}/efficiency"
    else
        echo "[efficiency] measure_efficiency.py missing — SKIP"
    fi
}

phase_summary() {
    echo ""
    echo "=== SUMMARY ==="

    export RESULTS_DIR PRIMARY_NOISE_LEVEL
    export PAPER_SEEDS_STR="${PAPER_SEEDS[*]}"
    export NOISE_SWEEP_SEEDS_STR="${NOISE_SWEEP_SEEDS[*]}"
    export NOISE_SWEEP_LEVELS_STR="${NOISE_SWEEP_LEVELS[*]}"

    python3 <<'PY'
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev

RESULTS_DIR = os.environ["RESULTS_DIR"]
PRIMARY_NOISE_LEVEL = int(os.environ["PRIMARY_NOISE_LEVEL"])
PAPER_SEEDS = [int(x) for x in os.environ["PAPER_SEEDS_STR"].split() if x]
NOISE_SWEEP_SEEDS = [int(x) for x in os.environ["NOISE_SWEEP_SEEDS_STR"].split() if x]
NOISE_SWEEP_LEVELS = [int(x) for x in os.environ["NOISE_SWEEP_LEVELS_STR"].split() if x]

METHOD_LABELS = {
    "ce": "CE",
    "hardness": "Hardness",
    "focal": "Focal",
    "eaft": "EAFT",
    "entropy_high": "High-Entropy Only",
    "entropy_low": "Low-Entropy Only",
    "l2sp": "L2-SP",
    "ewc": "EWC",
    "lwf": "LwF",
    "drift_loss": "drift-loss",
    "drift_loss_conservative": "drift-loss conservative",
    "drift_loss_attached": "drift-loss attached",
}


def parse_run_name(name: str):
    parts = name.split("__")
    if len(parts) == 2 and parts[0] == "base":
        return {"scenario": "base", "method": parts[1], "seed": None, "noise": None}
    if len(parts) == 3 and parts[0] in {"noisy", "domain", "reasoning"}:
        return {
            "scenario": parts[0],
            "method": parts[1],
            "seed": int(parts[2].replace("seed", "")),
            "noise": None,
        }
    if len(parts) == 4 and parts[0] == "noisy_sweep":
        return {
            "scenario": "noisy_sweep",
            "method": parts[1],
            "seed": int(parts[3].replace("seed", "")),
            "noise": int(parts[2].replace("noise", "")),
        }
    if len(parts) == 3 and parts[0] == "noisy_ablation":
        return {
            "scenario": "noisy_ablation",
            "method": parts[1],
            "seed": int(parts[2].replace("seed", "")),
            "noise": PRIMARY_NOISE_LEVEL,
        }
    return None


def load_benchmark_metrics(run_name: str):
    run_dir = os.path.join(RESULTS_DIR, "benchmarks", run_name)
    data = {}
    if not os.path.isdir(run_dir):
        return data
    for root, _, files in os.walk(run_dir):
        for filename in files:
            if not filename.startswith("results_") or not filename.endswith(".json"):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path) as f:
                    payload = json.load(f)
            except Exception:
                continue
            for task_name, metrics in payload.get("results", {}).items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        data[f"{task_name}/{metric_name}"] = value
    return data


def load_math_metrics(run_name: str):
    path = os.path.join(RESULTS_DIR, f"math500_passk_{run_name}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        payload = json.load(f)
    return payload.get("pass_at_k", {})


def metric_value(run_name: str, metric_key: str):
    benchmarks = load_benchmark_metrics(run_name)
    return benchmarks.get(metric_key)


def medqa_value(run_name: str):
    benchmarks = load_benchmark_metrics(run_name)
    return benchmarks.get("medqa_4options/acc,none") or benchmarks.get("medqa/acc,none")


def math_value(run_name: str, key: str):
    return load_math_metrics(run_name).get(key)


def summarize(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "---"
    if len(vals) == 1:
        return f"{vals[0] * 100:.2f}"
    return f"{mean(vals) * 100:.2f}±{pstdev(vals) * 100:.2f}"


def avg_delta(a_vals, b_vals):
    pairs = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
    if not pairs:
        return "---"
    deltas = [a - b for a, b in pairs]
    if len(deltas) == 1:
        return f"{deltas[0] * 100:+.2f}"
    return f"{mean(deltas) * 100:+.2f}"


run_names = []
bench_root = os.path.join(RESULTS_DIR, "benchmarks")
if os.path.isdir(bench_root):
    run_names.extend(sorted(os.listdir(bench_root)))
for path in glob.glob(os.path.join(RESULTS_DIR, "math500_passk_*.json")):
    run_names.append(os.path.basename(path)[len("math500_passk_"):-len(".json")])
run_names = sorted(set(run_names))

parsed = {name: parse_run_name(name) for name in run_names}
parsed = {name: info for name, info in parsed.items() if info is not None}

print("\n" + "=" * 92)
print(" drift-loss paper summary")
print("=" * 92)

base_name = "base__qwen35_4b"
base_mmlu = metric_value(base_name, "mmlu_pro/exact_match,custom-extract")
base_ifeval = metric_value(base_name, "ifeval/prompt_level_strict_acc,none")
base_medqa = medqa_value(base_name)
base_pass1 = math_value(base_name, "pass@1")
base_pass64 = math_value(base_name, "pass@64")

print("\nBase anchors")
print(f"  MMLU-Pro: {summarize([base_mmlu])}")
print(f"  IFEval strict: {summarize([base_ifeval])}")
print(f"  MedQA: {summarize([base_medqa])}")
print(f"  MATH-500 Pass@1: {summarize([base_pass1])}")
print(f"  MATH-500 Pass@64: {summarize([base_pass64])}")

print("\nNoisy alignment baselines")
print(f"{'Method':<24} {'MMLU-Pro':>12} {'IFEval':>12} {'ΔMMLU vs CE':>14}")
print("-" * 66)
ce_mmlu_vals = [
    metric_value(f"noisy__ce__seed{seed}", "mmlu_pro/exact_match,custom-extract")
    for seed in PAPER_SEEDS
]
for method in ["ce", "hardness", "focal", "eaft", "entropy_high", "entropy_low", "l2sp", "ewc", "lwf", "drift_loss"]:
    run_vals_mmlu = [
        metric_value(f"noisy__{method}__seed{seed}", "mmlu_pro/exact_match,custom-extract")
        for seed in PAPER_SEEDS
    ]
    run_vals_if = [
        metric_value(f"noisy__{method}__seed{seed}", "ifeval/prompt_level_strict_acc,none")
        for seed in PAPER_SEEDS
    ]
    print(
        f"{METHOD_LABELS[method]:<24} {summarize(run_vals_mmlu):>12} "
        f"{summarize(run_vals_if):>12} {avg_delta(run_vals_mmlu, ce_mmlu_vals):>14}"
    )

print("\nNoisy alignment sweep")
print(f"{'Noise %':<10} {'CE MMLU':>12} {'drift-loss MMLU':>18} {'Δ':>10}")
print("-" * 54)
for noise in NOISE_SWEEP_LEVELS:
    ce_vals = [
        metric_value(
            f"noisy_sweep__ce__noise{noise}__seed{seed}",
            "mmlu_pro/exact_match,custom-extract",
        )
        for seed in NOISE_SWEEP_SEEDS
    ]
    drift_vals = [
        metric_value(
            f"noisy_sweep__drift_loss__noise{noise}__seed{seed}",
            "mmlu_pro/exact_match,custom-extract",
        )
        for seed in NOISE_SWEEP_SEEDS
    ]
    print(
        f"{str(noise) + '%':<10} {summarize(ce_vals):>12} "
        f"{summarize(drift_vals):>18} {avg_delta(drift_vals, ce_vals):>10}"
    )

print("\nNoisy drift-loss ablations")
print(f"{'Variant':<28} {'MMLU-Pro':>12} {'IFEval':>12}")
print("-" * 54)
for method in ["drift_loss", "drift_loss_conservative", "drift_loss_attached"]:
    if method == "drift_loss":
        names = [f"noisy__{method}__seed{seed}" for seed in PAPER_SEEDS]
    else:
        names = [f"noisy_ablation__{method}__seed{PAPER_SEEDS[0]}"]
    print(
        f"{METHOD_LABELS[method]:<28} "
        f"{summarize([metric_value(n, 'mmlu_pro/exact_match,custom-extract') for n in names]):>12} "
        f"{summarize([metric_value(n, 'ifeval/prompt_level_strict_acc,none') for n in names]):>12}"
    )

print("\nDomain specialization")
print(f"{'Method':<24} {'MMLU-Pro':>12} {'MedQA':>12} {'ΔMMLU vs CE':>14}")
print("-" * 66)
ce_domain_mmlu = [
    metric_value(f"domain__ce__seed{seed}", "mmlu_pro/exact_match,custom-extract")
    for seed in PAPER_SEEDS
]
for method in ["ce", "hardness", "focal", "eaft", "entropy_high", "entropy_low", "l2sp", "ewc", "lwf", "drift_loss"]:
    mmlu_vals = [
        metric_value(f"domain__{method}__seed{seed}", "mmlu_pro/exact_match,custom-extract")
        for seed in PAPER_SEEDS
    ]
    medqa_vals = [medqa_value(f"domain__{method}__seed{seed}") for seed in PAPER_SEEDS]
    print(
        f"{METHOD_LABELS[method]:<24} {summarize(mmlu_vals):>12} "
        f"{summarize(medqa_vals):>12} {avg_delta(mmlu_vals, ce_domain_mmlu):>14}"
    )

print("\nReasoning-path rewiring")
print(f"{'Method':<24} {'MMLU-Pro':>12} {'Pass@1':>12} {'Pass@64':>12}")
print("-" * 64)
for method in ["ce", "hardness", "focal", "eaft", "entropy_high", "entropy_low", "l2sp", "ewc", "lwf", "drift_loss"]:
    mmlu_vals = [
        metric_value(f"reasoning__{method}__seed{seed}", "mmlu_pro/exact_match,custom-extract")
        for seed in PAPER_SEEDS
    ]
    pass1_vals = [math_value(f"reasoning__{method}__seed{seed}", "pass@1") for seed in PAPER_SEEDS]
    pass64_vals = [math_value(f"reasoning__{method}__seed{seed}", "pass@64") for seed in PAPER_SEEDS]
    print(
        f"{METHOD_LABELS[method]:<24} {summarize(mmlu_vals):>12} "
        f"{summarize(pass1_vals):>12} {summarize(pass64_vals):>12}"
    )

summary_path = os.path.join(RESULTS_DIR, "paper_experiment_summary.json")
with open(summary_path, "w") as f:
    json.dump(
        {
            "base": {
                "mmlu_pro": base_mmlu,
                "ifeval_strict": base_ifeval,
                "medqa": base_medqa,
                "pass@1": base_pass1,
                "pass@64": base_pass64,
            },
            "paper_seeds": PAPER_SEEDS,
            "noise_sweep_seeds": NOISE_SWEEP_SEEDS,
            "noise_sweep_levels": NOISE_SWEEP_LEVELS,
        },
        f,
        indent=2,
    )
print(f"\nSummary saved to: {summary_path}")
PY
}

case "${PHASE}" in
    all)
        phase_data
        phase_train
        phase_eval
        phase_efficiency
        phase_summary
        ;;
    data) phase_data ;;
    train) phase_train ;;
    eval) phase_eval ;;
    summary) phase_summary ;;
    efficiency) phase_efficiency ;;
    train_noisy) phase_train_noisy ;;
    train_domain) phase_train_domain ;;
    train_reasoning) phase_train_reasoning ;;
    eval_noisy) phase_eval_noisy ;;
    eval_domain) phase_eval_domain ;;
    eval_reasoning) phase_eval_reasoning ;;
    *)
        echo "Unknown phase: ${PHASE}" >&2
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  runner complete"
echo "============================================"
