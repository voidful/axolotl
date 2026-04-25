#!/usr/bin/env bash
# ============================================================
# Fast single-GPU drift-loss submission pipeline
# ============================================================
#
# Goal:
#   Get a small but paper-shaped CE vs Drift result quickly on 1x A6000.
#
# Default experiment:
#   - Model: Qwen/Qwen3.5-4B
#   - Adapter: QLoRA
#   - Methods: CE vs single drift-loss only
#   - Regimes: medical narrow-domain SFT + noisy UltraFeedback 25%
#   - Scale: 2k examples, 180 steps, 1 seed
#   - Eval: Base/CE/Drift on IFEval + MMLU-Pro with lm-eval --limit
#
# Usage:
#   bash scripts/paper/run_fast_drift_submission.sh pilot
#   bash scripts/paper/run_fast_drift_submission.sh data
#   bash scripts/paper/run_fast_drift_submission.sh train
#   bash scripts/paper/run_fast_drift_submission.sh eval
#   bash scripts/paper/run_fast_drift_submission.sh summary
#
# Useful overrides:
#   SEEDS="42 123" MAX_STEPS=240 EVAL_LIMIT=500 bash scripts/paper/run_fast_drift_submission.sh pilot
#   REGIMES="noise25" N_TRAIN=1000 MAX_STEPS=120 bash scripts/paper/run_fast_drift_submission.sh pilot
#   EVAL_BACKEND=vllm MERGE_FOR_EVAL=1 bash scripts/paper/run_fast_drift_submission.sh eval
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

PHASE="${1:-pilot}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
RUN_TAG="${RUN_TAG:-fast_drift}"
N_TRAIN="${N_TRAIN:-2000}"
NOISE_RATIO="${NOISE_RATIO:-0.25}"
NOISE_PCT="${NOISE_PCT:-25}"
SEEDS_STR="${SEEDS:-42}"
REGIMES_STR="${REGIMES:-medical noise25}"

SEQ_LEN="${SEQ_LEN:-2048}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-180}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
DRIFT_GAMMA="${DRIFT_GAMMA:-2.0}"
DRIFT_EMA_DECAY="${DRIFT_EMA_DECAY:-0.99}"

REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-drift-loss-neurips2026-fast}"

EVAL_BACKEND="${EVAL_BACKEND:-hf_peft}"  # hf_peft or vllm
MERGE_FOR_EVAL="${MERGE_FOR_EVAL:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"          # empty string means full eval
MMLU_FEWSHOT="${MMLU_FEWSHOT:-0}"        # use 5 for final/full runs
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"

DATA_DIR="${PROJECT_DIR}/data/paper/fast"
CONFIG_DIR="${PROJECT_DIR}/examples/paper/fast_drift"
OUTPUT_BASE="${PROJECT_DIR}/outputs/paper/${RUN_TAG}"
PREPARED_BASE="${PROJECT_DIR}/prepared_data/paper/${RUN_TAG}"
RESULTS_BASE="${PROJECT_DIR}/results/paper/${RUN_TAG}"
CACHE_DIR="${PROJECT_DIR}/tmp/hf_datasets_cache_${RUN_TAG}"

read -r -a SEED_LIST <<< "${SEEDS_STR}"
read -r -a REGIME_LIST <<< "${REGIMES_STR}"

mkdir -p "${DATA_DIR}" "${CONFIG_DIR}" "${OUTPUT_BASE}" "${PREPARED_BASE}" "${RESULTS_BASE}" "${CACHE_DIR}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_DIR}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

log() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

limit_args() {
    if [[ -n "${EVAL_LIMIT}" ]]; then
        printf -- "--limit %s" "${EVAL_LIMIT}"
    fi
}

dataset_for_regime() {
    local regime="$1"
    case "${regime}" in
        medical)
            echo "${DATA_DIR}/medical_flashcards_$((N_TRAIN / 1000))k.jsonl"
            ;;
        noise25)
            echo "${DATA_DIR}/ultrafeedback_noisy_${NOISE_PCT}pct_$((N_TRAIN / 1000))k.jsonl"
            ;;
        *)
            echo "Unknown regime: ${regime}" >&2
            return 1
            ;;
    esac
}

run_name() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    echo "${RUN_TAG}-${method}-${regime}-s${seed}"
}

config_path() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    echo "${CONFIG_DIR}/$(run_name "${method}" "${regime}" "${seed}").yaml"
}

output_dir() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    echo "${OUTPUT_BASE}/$(run_name "${method}" "${regime}" "${seed}")"
}

prepared_dir() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    echo "${PREPARED_BASE}/$(run_name "${method}" "${regime}" "${seed}")"
}

phase_data() {
    log "Preparing ${N_TRAIN}-sample datasets"

    local medical_path
    medical_path="$(dataset_for_regime medical)"
    if [[ ! -f "${medical_path}" ]]; then
        python3 "${SCRIPT_DIR}/create_medical_sft.py" \
            --output_dir "${DATA_DIR}" \
            --num_samples "${N_TRAIN}" \
            --seed 42
    else
        echo "[data] exists: ${medical_path}"
    fi

    local noise_path
    noise_path="$(dataset_for_regime noise25)"
    if [[ ! -f "${noise_path}" ]]; then
        python3 "${SCRIPT_DIR}/create_noisy_ultrafeedback.py" \
            --output_dir "${DATA_DIR}" \
            --num_samples "${N_TRAIN}" \
            --noise_ratio "${NOISE_RATIO}" \
            --seed 42
    else
        echo "[data] exists: ${noise_path}"
    fi
}

write_config() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    local dataset="$4"
    local cfg="$5"
    local out="$6"
    local prepared="$7"

    local plugins_block
    local drift_block

    if [[ "${method}" == "drift" ]]; then
        plugins_block="plugins:
  - axolotl.integrations.drift.DriftPlugin"
        drift_block="drift_trainer: true
drift_reference_mode: ema
drift_ema_decay: ${DRIFT_EMA_DECAY}
drift_gamma: ${DRIFT_GAMMA}
drift_detach_weights: true"
    else
        plugins_block="# CE baseline: no custom trainer plugin"
        drift_block="# CE baseline: standard next-token cross entropy"
    fi

    cat > "${cfg}" << YAML
# Auto-generated by scripts/paper/run_fast_drift_submission.sh
# Method: ${method}
# Regime: ${regime}
# Seed: ${seed}

base_model: ${MODEL}
trust_remote_code: true
strict: false
low_cpu_mem_usage: true

${plugins_block}

${drift_block}

datasets:
  - path: ${dataset}
    ds_type: json
    type: chat_template
    field_messages: conversations
    split: train

dataset_prepared_path: ${prepared}
chat_template: qwen3_5

sequence_len: ${SEQ_LEN}
sample_packing: true
pad_to_sequence_len: false

load_in_4bit: true
adapter: qlora
lora_model_dir:
lora_r: ${LORA_R}
lora_alpha: ${LORA_ALPHA}
lora_dropout: 0.05
lora_target_linear: true

gradient_accumulation_steps: ${GRAD_ACCUM}
micro_batch_size: ${MICRO_BATCH}
max_steps: ${MAX_STEPS}
num_epochs: 1
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: ${LEARNING_RATE}
warmup_ratio: 0.06
weight_decay: 0.0
seed: ${seed}

bf16: auto
tf32: true
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
flash_attention: true
dataloader_num_workers: 4

val_set_size: 0.05
eval_strategy: steps
eval_steps: 60
save_strategy: steps
save_steps: ${MAX_STEPS}
save_total_limit: 1
output_dir: ${out}

logging_steps: 5
report_to: ${REPORT_TO}
wandb_project: ${WANDB_PROJECT}
wandb_run_name: $(basename "${out}")
YAML
}

train_one() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    local dataset
    local cfg
    local out
    local prepared

    dataset="$(dataset_for_regime "${regime}")"
    cfg="$(config_path "${method}" "${regime}" "${seed}")"
    out="$(output_dir "${method}" "${regime}" "${seed}")"
    prepared="$(prepared_dir "${method}" "${regime}" "${seed}")"

    if [[ ! -f "${dataset}" ]]; then
        echo "[train] missing dataset: ${dataset}" >&2
        echo "        run: bash scripts/paper/run_fast_drift_submission.sh data" >&2
        return 1
    fi

    write_config "${method}" "${regime}" "${seed}" "${dataset}" "${cfg}" "${out}" "${prepared}"

    if [[ -f "${out}/adapter_config.json" || -f "${out}/config.json" ]]; then
        echo "[train] skip existing: ${out}"
        return 0
    fi

    log "Training ${method} / ${regime} / seed ${seed}"
    python3 -m accelerate.commands.launch -m axolotl.cli.train "${cfg}"
}

phase_train() {
    log "Training CE and single drift-loss"
    for regime in "${REGIME_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            train_one "ce" "${regime}" "${seed}"
            train_one "drift" "${regime}" "${seed}"
        done
    done
}

merge_one() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    local cfg
    local out

    cfg="$(config_path "${method}" "${regime}" "${seed}")"
    out="$(output_dir "${method}" "${regime}" "${seed}")"

    if [[ -d "${out}/merged" ]]; then
        echo "[merge] skip existing: ${out}/merged"
        return 0
    fi
    if [[ ! -f "${out}/adapter_config.json" ]]; then
        echo "[merge] skip missing adapter: ${out}"
        return 0
    fi

    log "Merging LoRA for ${method} / ${regime} / seed ${seed}"
    python3 -m axolotl.cli.merge_lora "${cfg}"
}

phase_merge() {
    log "Merging adapters for vLLM/full-model evaluation"
    for regime in "${REGIME_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            merge_one "ce" "${regime}" "${seed}"
            merge_one "drift" "${regime}" "${seed}"
        done
    done
}

eval_model_args() {
    local model_or_adapter="$1"
    if [[ "${EVAL_BACKEND}" == "vllm" ]]; then
        echo "pretrained=${model_or_adapter},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${SEQ_LEN}"
    else
        echo "pretrained=${MODEL},peft=${model_or_adapter},dtype=bfloat16,trust_remote_code=True"
    fi
}

eval_base_args() {
    if [[ "${EVAL_BACKEND}" == "vllm" ]]; then
        echo "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${SEQ_LEN}"
    else
        echo "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True"
    fi
}

lm_eval_one_task() {
    local run="$1"
    local model_args="$2"
    local task="$3"
    local fewshot="$4"
    local out_dir="${RESULTS_BASE}/benchmarks/${run}/${task}"
    local lm_backend
    local gen_kwargs

    if [[ "${EVAL_BACKEND}" == "vllm" ]]; then
        lm_backend="vllm"
    else
        lm_backend="hf"
    fi
    if [[ "${task}" == "ifeval" ]]; then
        gen_kwargs="max_gen_toks=512"
    else
        gen_kwargs="max_gen_toks=256"
    fi

    if find "${out_dir}" -name 'results_*.json' 2>/dev/null | grep -q .; then
        echo "[eval] skip existing: ${run}/${task}"
        return 0
    fi

    mkdir -p "${out_dir}"
    log "Evaluating ${run} on ${task}"

    # shellcheck disable=SC2046
    python3 -m lm_eval \
        --model "${lm_backend}" \
        --model_args "${model_args}" \
        --tasks "${task}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_fewshot "${fewshot}" \
        --gen_kwargs "${gen_kwargs}" \
        $(limit_args) \
        --output_path "${out_dir}" \
        --log_samples 2>&1 | tee "${out_dir}.log"
}

eval_one() {
    local run="$1"
    local model_path="$2"
    local model_args

    model_args="$(eval_model_args "${model_path}")"
    lm_eval_one_task "${run}" "${model_args}" "ifeval" 0
    lm_eval_one_task "${run}" "${model_args}" "mmlu_pro" "${MMLU_FEWSHOT}"
}

phase_eval() {
    log "Evaluation backend=${EVAL_BACKEND}, limit=${EVAL_LIMIT:-full}"

    if [[ "${MERGE_FOR_EVAL}" == "1" || "${EVAL_BACKEND}" == "vllm" ]]; then
        phase_merge
    fi

    local base_args
    base_args="$(eval_base_args)"
    lm_eval_one_task "base" "${base_args}" "ifeval" 0
    lm_eval_one_task "base" "${base_args}" "mmlu_pro" "${MMLU_FEWSHOT}"

    for regime in "${REGIME_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            for method in ce drift; do
                local run
                local out
                local eval_path

                run="$(run_name "${method}" "${regime}" "${seed}")"
                out="$(output_dir "${method}" "${regime}" "${seed}")"

                if [[ "${EVAL_BACKEND}" == "vllm" ]]; then
                    eval_path="${out}/merged"
                else
                    eval_path="${out}"
                fi

                if [[ ! -d "${eval_path}" ]]; then
                    echo "[eval] skip missing: ${eval_path}"
                    continue
                fi
                eval_one "${run}" "${eval_path}"
            done
        done
    done
}

phase_summary() {
    log "Summarizing fast drift results"
    python3 "${SCRIPT_DIR}/summarize_fast_drift.py" \
        --run_tag "${RUN_TAG}" \
        --outputs "${OUTPUT_BASE}" \
        --results "${RESULTS_BASE}" \
        --regimes ${REGIMES_STR} \
        --seeds ${SEEDS_STR}
}

case "${PHASE}" in
    pilot)
        phase_data
        phase_train
        phase_eval
        phase_summary
        ;;
    data)
        phase_data
        ;;
    train)
        phase_train
        ;;
    merge)
        phase_merge
        ;;
    eval)
        phase_eval
        ;;
    summary)
        phase_summary
        ;;
    *)
        echo "Unknown phase: ${PHASE}" >&2
        exit 1
        ;;
esac
