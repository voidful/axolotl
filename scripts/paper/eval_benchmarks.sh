#!/bin/bash
# ============================================================
# Drift-Trust Paper — Benchmark Evaluation Script
# Battle A: MMLU-Pro + IFEval via lm-evaluation-harness
# Backend: vLLM (PagedAttention, ~5-10x faster than HF generate)
# ============================================================
#
# Usage:
#   bash eval_benchmarks.sh <model_path> <run_name>
#
# Examples:
#   bash eval_benchmarks.sh ./outputs/paper/ce-noisy-4b ce_noisy_4b
#   bash eval_benchmarks.sh ./outputs/paper/drift-noisy-4b drift_noisy_4b
#   bash eval_benchmarks.sh Qwen/Qwen3.5-4B qwen35_4b_base  # raw base model
# ============================================================

set -euo pipefail

MODEL_PATH="${1:?Usage: eval_benchmarks.sh <model_path> <run_name>}"
RUN_NAME="${2:?Usage: eval_benchmarks.sh <model_path> <run_name>}"
OUTPUT_DIR="./results/paper/benchmarks/${RUN_NAME}"
BATCH_SIZE="${3:-auto}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Evaluating: ${RUN_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Backend: vLLM"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Fix: vLLM treats Qwen3.5 as VLM, fails on missing visual weights.
# Patch vLLM's default_loader.py to tolerate missing weights (warning instead of error).
VLLM_LOADER=$(python3 -c "import vllm.model_executor.model_loader.default_loader as m; print(m.__file__)")
if [ -n "${VLLM_LOADER}" ]; then
    if ! [ -f "${VLLM_LOADER}.orig" ]; then
        cp "${VLLM_LOADER}" "${VLLM_LOADER}.orig"
    fi
    # Change "raise ValueError" to "pass  # patched" for uninitialized weights
    sed -i 's/raise ValueError(/import logging; logging.getLogger(__name__).warning(  # patched: was raise ValueError/' "${VLLM_LOADER}"
    echo "Patched vLLM default_loader.py to tolerate missing visual weights."
fi

# Restore config.json if previously patched
if [ -d "${MODEL_PATH}" ] && [ -f "${MODEL_PATH}/config.json.bak" ]; then
    cp "${MODEL_PATH}/config.json.bak" "${MODEL_PATH}/config.json"
    rm -f "${MODEL_PATH}/config.json.bak"
fi

VLLM_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.9,max_model_len=4096"

# --- MMLU-Pro ---
echo ""
echo "[1/2] Running MMLU-Pro..."
python -m lm_eval --model vllm \
    --model_args "${VLLM_ARGS}" \
    --tasks mmlu_pro \
    --batch_size "${BATCH_SIZE}" \
    --num_fewshot 5 \
    --gen_kwargs "max_gen_toks=256" \
    --output_path "${OUTPUT_DIR}/mmlu_pro" \
    --log_samples 2>&1 | tee "${OUTPUT_DIR}/mmlu_pro.log"

echo ""
echo "[1/2] MMLU-Pro complete."

# --- IFEval ---
echo ""
echo "[2/2] Running IFEval..."
python -m lm_eval --model vllm \
    --model_args "${VLLM_ARGS}" \
    --tasks ifeval \
    --batch_size "${BATCH_SIZE}" \
    --num_fewshot 0 \
    --gen_kwargs "max_gen_toks=512" \
    --output_path "${OUTPUT_DIR}/ifeval" \
    --log_samples 2>&1 | tee "${OUTPUT_DIR}/ifeval.log"

echo ""
echo "[2/2] IFEval complete."

# --- Summary ---
echo ""
echo "============================================"
echo "All benchmarks complete for: ${RUN_NAME}"
echo "Results: ${OUTPUT_DIR}"
echo "============================================"

# Extract key scores from JSON outputs
echo ""
echo "Summary:"
for task_dir in mmlu_pro ifeval; do
    result_file=$(find "${OUTPUT_DIR}/${task_dir}" -name "results_*.json" 2>/dev/null | head -1)
    if [ -n "${result_file}" ]; then
        echo "  ${task_dir}:"
        python3 -c "
import json
with open('${result_file}') as f:
    data = json.load(f)
results = data.get('results', {})
for task, metrics in results.items():
    for metric, value in metrics.items():
        if 'acc' in metric.lower() or 'score' in metric.lower():
            if isinstance(value, (int, float)):
                print(f'    {task}/{metric}: {value:.4f}')
" 2>/dev/null || echo "    (parse results manually)"
    fi
done
