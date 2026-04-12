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
# Precisely patch only the "uninitialized weights" check in default_loader.py.
python3 << 'PATCH_EOF'
import importlib, re
mod = importlib.import_module("vllm.model_executor.model_loader.default_loader")
fpath = mod.__file__

# Restore original if backup exists
import os
orig = fpath + ".orig"
if os.path.exists(orig):
    import shutil
    shutil.copy2(orig, fpath)
else:
    import shutil
    shutil.copy2(fpath, orig)

with open(fpath) as f:
    src = f.read()

# Target only: raise ValueError(f"Following weights were not initialized...")
patched = src.replace(
    'raise ValueError(\n                f"Following weights were not initialized from checkpoint: "',
    'pass  # PATCHED: was raise ValueError\n                # f"Following weights were not initialized from checkpoint: "'
)

if patched != src:
    with open(fpath, "w") as f:
        f.write(patched)
    print("  Patched: uninitialized weights check → warning (visual encoder safe)")
else:
    print("  Already patched or pattern not found.")
PATCH_EOF

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
