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

# vLLM model args:
#   gpu_memory_utilization=0.9  — use 90% of GPU memory for KV cache
#   max_model_len=4096          — cap context length to fit in single GPU
#   trust_remote_code=True      — required for Qwen3.5
#   hf_overrides — force text-only CausalLM (vLLM may default to VLM for Qwen3.5)
VLLM_ARGS='pretrained='"${MODEL_PATH}"',dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.9,max_model_len=4096,hf_overrides={"architectures":["Qwen3_5ForCausalLM"]}'

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
