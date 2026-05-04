#!/usr/bin/env bash
# ============================================================
# Token-role triage experiment pipeline
# ============================================================
#
# Methods:
#   ce
#   stm_top20
#   soft_stm
#   retention_kl
#   learn_new
#   module_aware_retention
#   fullft_module_aware_retention
#   attention_only_new
#
# Default run is a fast single-GPU pilot:
#   REGIMES="medical noise25" SEEDS="42" MAX_STEPS=180
#
# Usage:
#   bash scripts/paper/run_token_triage_experiments.sh pilot
#   bash scripts/paper/run_token_triage_experiments.sh data
#   bash scripts/paper/run_token_triage_experiments.sh config
#   bash scripts/paper/run_token_triage_experiments.sh train
#   bash scripts/paper/run_token_triage_experiments.sh eval
#   bash scripts/paper/run_token_triage_experiments.sh summary
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

PHASE="${1:-pilot}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
RUN_TAG="${RUN_TAG:-token_triage}"
RESULTS_TAG="${RESULTS_TAG:-${RUN_TAG}}"
N_TRAIN="${N_TRAIN:-2000}"
SEEDS_STR="${SEEDS:-42}"
REGIMES_STR="${REGIMES:-medical noise25}"
METHODS_STR="${METHODS:-ce stm_top20 soft_stm retention_kl learn_new fullft_module_aware_retention}"

SEQ_LEN="${SEQ_LEN:-2048}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-180}"
EVAL_STEPS="${EVAL_STEPS:-60}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-token-triage-neurips2026-fast}"

TRIAGE_GATE_T="${TRIAGE_GATE_T:-1.0}"
TRIAGE_OLD_Q="${TRIAGE_OLD_Q:-0.40}"
TRIAGE_NEW_Q="${TRIAGE_NEW_Q:-0.60}"
TRIAGE_NOISE_Q="${TRIAGE_NOISE_Q:-0.95}"
STM_KEEP_RATIO="${STM_KEEP_RATIO:-0.80}"
LAMBDA_ACQUIRE="${LAMBDA_ACQUIRE:-1.0}"
MU_NOISE="${MU_NOISE:-1.0}"
RHO_RETENTION="${RHO_RETENTION:-0.5}"
TRIAGE_W_FLOOR="${TRIAGE_W_FLOOR:-0.1}"
TRIAGE_W_MAX="${TRIAGE_W_MAX:-3.0}"
KL_BETA="${KL_BETA:-0.05}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-128}"
ATTN_LAMBDA_ACQUIRE="${ATTN_LAMBDA_ACQUIRE:-1.0}"
ATTN_MU_NOISE="${ATTN_MU_NOISE:-1.0}"
ATTN_RHO_RETENTION="${ATTN_RHO_RETENTION:-0.0}"
ATTN_KL_BETA="${ATTN_KL_BETA:-0.0}"
MLP_LAMBDA_ACQUIRE="${MLP_LAMBDA_ACQUIRE:-0.5}"
MLP_MU_NOISE="${MLP_MU_NOISE:-1.0}"
MLP_RHO_RETENTION="${MLP_RHO_RETENTION:-0.5}"
MLP_KL_BETA="${MLP_KL_BETA:-0.05}"
OTHER_LAMBDA_ACQUIRE="${OTHER_LAMBDA_ACQUIRE:-0.25}"
OTHER_MU_NOISE="${OTHER_MU_NOISE:-1.0}"
OTHER_RHO_RETENTION="${OTHER_RHO_RETENTION:-0.75}"
OTHER_KL_BETA="${OTHER_KL_BETA:-0.05}"

EVAL_BACKEND="${EVAL_BACKEND:-hf}"
MERGE_FOR_EVAL="${MERGE_FOR_EVAL:-0}"
EVAL_TASKS_STR="${EVAL_TASKS:-ifeval mmlu_pro}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
MMLU_FEWSHOT="${MMLU_FEWSHOT:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"
EVAL_INCLUDE_PATH="${EVAL_INCLUDE_PATH:-}"
IFEVAL_MAX_GEN_TOKS="${IFEVAL_MAX_GEN_TOKS:-512}"
MMLU_MAX_GEN_TOKS="${MMLU_MAX_GEN_TOKS:-256}"
MMLU_DIRECT_MAX_GEN_TOKS="${MMLU_DIRECT_MAX_GEN_TOKS:-8}"
DOMAIN_EVAL="${DOMAIN_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

DATA_DIR="${PROJECT_DIR}/data/paper/fast"
CONFIG_DIR="${PROJECT_DIR}/examples/paper/token_triage/qwen4b"
OUTPUT_BASE="${PROJECT_DIR}/outputs/paper/${RUN_TAG}"
EVAL_OUTPUT_BASE="${EVAL_OUTPUT_BASE:-${OUTPUT_BASE}}"
PREPARED_BASE="${PROJECT_DIR}/prepared_data/paper/${RUN_TAG}"
RESULTS_BASE="${PROJECT_DIR}/results/paper/${RESULTS_TAG}"
CACHE_DIR="${PROJECT_DIR}/tmp/hf_datasets_cache_${RUN_TAG}"

read -r -a SEED_LIST <<< "${SEEDS_STR}"
read -r -a REGIME_LIST <<< "${REGIMES_STR}"
read -r -a METHOD_LIST <<< "${METHODS_STR}"
read -r -a EVAL_TASK_LIST <<< "${EVAL_TASKS_STR}"

mkdir -p "${DATA_DIR}" "${CONFIG_DIR}" "${OUTPUT_BASE}" "${PREPARED_BASE}" "${RESULTS_BASE}" "${CACHE_DIR}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_DIR}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

log() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

limit_args() {
    case "${EVAL_LIMIT}" in
        ""|0|none|full)
            return 0
            ;;
    esac
    if [[ -n "${EVAL_LIMIT}" ]]; then
        printf -- "--limit %s" "${EVAL_LIMIT}"
    fi
}

include_path_args() {
    if [[ -n "${EVAL_INCLUDE_PATH}" ]]; then
        printf -- "--include_path %s" "${EVAL_INCLUDE_PATH}"
    fi
}

gen_kwargs_args() {
    local gen_kwargs="$1"
    if [[ -n "${gen_kwargs}" ]]; then
        printf -- "--gen_kwargs %s" "${gen_kwargs}"
    fi
}

dataset_for_regime() {
    local regime="$1"
    local suffix="$((N_TRAIN / 1000))k"
    case "${regime}" in
        medical)
            echo "${DATA_DIR}/medical_flashcards_${suffix}.jsonl"
            ;;
        math)
            echo "${DATA_DIR}/numina_math_cot_${suffix}.jsonl"
            ;;
        noise*)
            local pct="${regime#noise}"
            echo "${DATA_DIR}/ultrafeedback_noisy_${pct}pct_${suffix}.jsonl"
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

eval_output_dir() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    echo "${EVAL_OUTPUT_BASE}/$(run_name "${method}" "${regime}" "${seed}")"
}

prepared_dir() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    echo "${PREPARED_BASE}/$(run_name "${method}" "${regime}" "${seed}")"
}

phase_data() {
    log "Preparing ${N_TRAIN}-sample datasets"

    for regime in "${REGIME_LIST[@]}"; do
        local data_path
        data_path="$(dataset_for_regime "${regime}")"
        if [[ -f "${data_path}" ]]; then
            echo "[data] exists: ${data_path}"
            continue
        fi

        case "${regime}" in
            medical)
                python3 "${SCRIPT_DIR}/create_medical_sft.py" \
                    --output_dir "${DATA_DIR}" \
                    --num_samples "${N_TRAIN}" \
                    --seed 42
                ;;
            math)
                python3 "${SCRIPT_DIR}/sample_numina_math.py" \
                    --output_dir "${DATA_DIR}" \
                    --num_samples "${N_TRAIN}" \
                    --seed 42
                ;;
            noise*)
                local pct="${regime#noise}"
                local ratio
                ratio="$(python3 -c "print(${pct} / 100.0)")"
                python3 "${SCRIPT_DIR}/create_noisy_ultrafeedback.py" \
                    --output_dir "${DATA_DIR}" \
                    --num_samples "${N_TRAIN}" \
                    --noise_ratio "${ratio}" \
                    --seed 42
                ;;
        esac
    done
}

write_config() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    local dataset="$4"
    local cfg="$5"
    local out="$6"
    local prepared="$7"

    cat > "${cfg}" << YAML
# Auto-generated by scripts/paper/run_token_triage_experiments.sh
# Method: ${method}
# Regime: ${regime}
# Seed: ${seed}

base_model: ${MODEL}
trust_remote_code: true
strict: false
low_cpu_mem_usage: true

plugins:
  - axolotl.integrations.rcca_tr.RCCATRPlugin

rcca_tr_trainer: true
rcca_tr_mode: ${method}
rcca_tr_gate_temperature: ${TRIAGE_GATE_T}
rcca_tr_old_quantile: ${TRIAGE_OLD_Q}
rcca_tr_new_quantile: ${TRIAGE_NEW_Q}
rcca_tr_noise_quantile: ${TRIAGE_NOISE_Q}
rcca_tr_stm_keep_ratio: ${STM_KEEP_RATIO}
rcca_tr_lambda_acquire: ${LAMBDA_ACQUIRE}
rcca_tr_mu_noise: ${MU_NOISE}
rcca_tr_rho_retention: ${RHO_RETENTION}
rcca_tr_triage_w_floor: ${TRIAGE_W_FLOOR}
rcca_tr_triage_w_max: ${TRIAGE_W_MAX}
rcca_tr_kl_beta: ${KL_BETA}
rcca_tr_kl_chunk_size: ${KL_CHUNK_SIZE}
rcca_tr_reference_model: ${MODEL}
rcca_tr_attn_lambda_acquire: ${ATTN_LAMBDA_ACQUIRE}
rcca_tr_attn_mu_noise: ${ATTN_MU_NOISE}
rcca_tr_attn_rho_retention: ${ATTN_RHO_RETENTION}
rcca_tr_attn_kl_beta: ${ATTN_KL_BETA}
rcca_tr_mlp_lambda_acquire: ${MLP_LAMBDA_ACQUIRE}
rcca_tr_mlp_mu_noise: ${MLP_MU_NOISE}
rcca_tr_mlp_rho_retention: ${MLP_RHO_RETENTION}
rcca_tr_mlp_kl_beta: ${MLP_KL_BETA}
rcca_tr_other_lambda_acquire: ${OTHER_LAMBDA_ACQUIRE}
rcca_tr_other_mu_noise: ${OTHER_MU_NOISE}
rcca_tr_other_rho_retention: ${OTHER_RHO_RETENTION}
rcca_tr_other_kl_beta: ${OTHER_KL_BETA}

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

YAML

    cat >> "${cfg}" << YAML

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
dataset_num_proc: ${DATASET_NUM_PROC}

val_set_size: 0.05
eval_strategy: steps
eval_steps: ${EVAL_STEPS}
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
        echo "        run: bash scripts/paper/run_token_triage_experiments.sh data" >&2
        return 1
    fi

    write_config "${method}" "${regime}" "${seed}" "${dataset}" "${cfg}" "${out}" "${prepared}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] wrote config: ${cfg}"
        return 0
    fi

    if [[ -f "${out}/config.json" ]]; then
        echo "[train] skip existing: ${out}"
        return 0
    fi

    log "Training ${method} / ${regime} / seed ${seed}"
    axolotl train "${cfg}" --launcher python
}

phase_train() {
    log "Training token-triage matrix"
    for regime in "${REGIME_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            for method in "${METHOD_LIST[@]}"; do
                train_one "${method}" "${regime}" "${seed}"
            done
        done
    done
}

phase_config() {
    log "Generating token-triage configs"
    DRY_RUN=1 phase_train
}

merge_one() {
    local method="$1"
    local regime="$2"
    local seed="$3"
    local cfg
    local out

    cfg="$(config_path "${method}" "${regime}" "${seed}")"
    out="$(output_dir "${method}" "${regime}" "${seed}")"

    echo "[merge] no-op for full-FT checkpoint: ${out}"
}

phase_merge() {
    log "Merge phase is a no-op for full-FT checkpoints"
    for regime in "${REGIME_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            for method in "${METHOD_LIST[@]}"; do
                merge_one "${method}" "${regime}" "${seed}"
            done
        done
    done
}

eval_model_args() {
    local model_or_adapter="$1"
    if [[ "${EVAL_BACKEND}" == "vllm" ]]; then
        echo "pretrained=${model_or_adapter},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${SEQ_LEN}"
    else
        echo "pretrained=${model_or_adapter},dtype=bfloat16,trust_remote_code=True"
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
    case "${task}" in
        ifeval)
            gen_kwargs="max_gen_toks=${IFEVAL_MAX_GEN_TOKS}"
            ;;
        mmlu_pro_direct)
            gen_kwargs="max_gen_toks=${MMLU_DIRECT_MAX_GEN_TOKS}"
            ;;
        mmlu_pro_mc)
            gen_kwargs=""
            ;;
        mmlu_pro*)
            gen_kwargs="max_gen_toks=${MMLU_MAX_GEN_TOKS}"
            ;;
        *)
            gen_kwargs="max_gen_toks=256"
            ;;
    esac

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
        $(gen_kwargs_args "${gen_kwargs}") \
        $(include_path_args) \
        $(limit_args) \
        --output_path "${out_dir}" \
        --log_samples 2>&1 | tee "${out_dir}.log"
}

fewshot_for_task() {
    local task="$1"
    case "${task}" in
        mmlu_pro*)
            echo "${MMLU_FEWSHOT}"
            ;;
        *)
            echo 0
            ;;
    esac
}

eval_one() {
    local run="$1"
    local model_path="$2"
    local regime="$3"
    local model_args

    model_args="$(eval_model_args "${model_path}")"
    for task in "${EVAL_TASK_LIST[@]}"; do
        lm_eval_one_task "${run}" "${model_args}" "${task}" "$(fewshot_for_task "${task}")"
    done
    if [[ "${DOMAIN_EVAL}" == "1" && "${regime}" == "medical" ]]; then
        lm_eval_one_task "${run}" "${model_args}" "medqa_4options" 0
    fi
}

phase_eval() {
    log "Evaluation backend=${EVAL_BACKEND}, tasks=${EVAL_TASKS_STR}, results=${RESULTS_TAG}, limit=${EVAL_LIMIT:-full}"

    if [[ "${MERGE_FOR_EVAL}" == "1" ]]; then
        phase_merge
    fi

    local base_args
    base_args="$(eval_base_args)"
    for task in "${EVAL_TASK_LIST[@]}"; do
        lm_eval_one_task "base" "${base_args}" "${task}" "$(fewshot_for_task "${task}")"
    done

    for regime in "${REGIME_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            for method in "${METHOD_LIST[@]}"; do
                local run
                local out
                local eval_path

                run="$(run_name "${method}" "${regime}" "${seed}")"
                out="$(eval_output_dir "${method}" "${regime}" "${seed}")"

                eval_path="${out}"

                if [[ ! -d "${eval_path}" ]]; then
                    echo "[eval] skip missing: ${eval_path}"
                    continue
                fi
                eval_one "${run}" "${eval_path}" "${regime}"
            done
        done
    done
}

phase_summary() {
    log "Summarizing token triage results"
    python3 "${SCRIPT_DIR}/summarize_fast_drift.py" \
        --run_tag "${RUN_TAG}" \
        --outputs "${OUTPUT_BASE}" \
        --results "${RESULTS_BASE}" \
        --methods ${METHODS_STR} \
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
    config)
        phase_config
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
