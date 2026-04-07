#!/usr/bin/env python3
"""
Measure training efficiency: Peak VRAM and Throughput.

Runs a short training check (50 steps) with each config and records:
  - Peak GPU memory (VRAM) in GB
  - Training throughput in tokens/second
  - Wall-clock time per step

Used to fill Table 2 in the Drift-Trust paper.

Usage:
    python measure_efficiency.py \
        --configs examples/paper/ce_noisy_4b.yaml examples/paper/drift_noisy_4b.yaml \
        --max_steps 50 \
        --output_dir ./results/paper/efficiency
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run_training_check(config_path: str, max_steps: int = 50, output_dir: str = "/tmp/efficiency_check"):
    """Run a short training and capture efficiency metrics."""
    run_name = Path(config_path).stem

    # Override output dir and steps for efficiency measurement
    cmd = [
        "accelerate", "launch", "-m", "axolotl.cli.train",
        config_path,
        "--max_steps", str(max_steps),
        "--output_dir", os.path.join(output_dir, run_name),
        "--save_strategy", "no",
        "--eval_strategy", "no",
        "--report_to", "none",
    ]

    print(f"\n{'='*60}")
    print(f"Measuring efficiency: {run_name}")
    print(f"Config: {config_path}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min timeout
    )
    wall_time = time.time() - start_time

    # Parse GPU memory from output
    peak_vram_gb = None
    tokens_per_sec = None

    for line in result.stdout.split("\n") + result.stderr.split("\n"):
        # Look for memory usage in log
        if "peak" in line.lower() and ("memory" in line.lower() or "vram" in line.lower()):
            import re
            numbers = re.findall(r'[\d.]+', line)
            if numbers:
                peak_vram_gb = float(numbers[-1])

        # Look for throughput
        if "tokens" in line.lower() and ("sec" in line.lower() or "/s" in line.lower()):
            import re
            numbers = re.findall(r'[\d.]+', line)
            if numbers:
                tokens_per_sec = float(numbers[0])

    # If not found in logs, try nvidia-smi
    if peak_vram_gb is None:
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            vram_values = [float(x) / 1024 for x in smi.stdout.strip().split("\n") if x.strip()]
            peak_vram_gb = max(vram_values) if vram_values else None
        except (FileNotFoundError, ValueError):
            pass

    return {
        "config": config_path,
        "run_name": run_name,
        "max_steps": max_steps,
        "wall_time_seconds": wall_time,
        "time_per_step": wall_time / max_steps,
        "peak_vram_gb": peak_vram_gb,
        "tokens_per_sec": tokens_per_sec,
        "exit_code": result.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure training efficiency")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Training YAML config paths")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./results/paper/efficiency")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for config in args.configs:
        try:
            metrics = run_training_check(config, args.max_steps, args.output_dir)
            results.append(metrics)
            print(f"\n  Peak VRAM: {metrics['peak_vram_gb']} GB")
            print(f"  Throughput: {metrics['tokens_per_sec']} tokens/s")
            print(f"  Time/step: {metrics['time_per_step']:.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"config": config, "error": str(e)})

    # Save results
    output_file = Path(args.output_dir) / "efficiency_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Config':<30} {'VRAM (GB)':>12} {'Tok/s':>12} {'s/step':>10}")
    print("-" * 70)
    for r in results:
        name = r.get("run_name", "?")[:28]
        vram = f"{r['peak_vram_gb']:.1f}" if r.get('peak_vram_gb') else "N/A"
        tps = f"{r['tokens_per_sec']:.0f}" if r.get('tokens_per_sec') else "N/A"
        sps = f"{r['time_per_step']:.2f}" if r.get('time_per_step') else "N/A"
        print(f"{name:<30} {vram:>12} {tps:>12} {sps:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
