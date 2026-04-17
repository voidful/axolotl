#!/usr/bin/env python3
"""
Measure training efficiency: Peak VRAM and Throughput.

Runs a short training check (50 steps) with each config and records:
  - Peak GPU memory (VRAM) in GB
  - Training throughput in tokens/second
  - Wall-clock time per step

Used to fill the efficiency table in the Drift-Trust paper.

Usage:
    python measure_efficiency.py \
        --configs examples/paper/ce_noisy_4b.yaml examples/paper/drift_noisy_4b.yaml \
        --max_steps 50 \
        --output_dir ./results/paper/efficiency
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


def poll_gpu_memory(stop_event, results_dict, interval=2.0):
    """Background thread: poll nvidia-smi for peak GPU VRAM usage."""
    peak_mb = 0
    while not stop_event.is_set():
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            for line in smi.stdout.strip().split("\n"):
                if line.strip():
                    mb = float(line.strip())
                    peak_mb = max(peak_mb, mb)
        except Exception:
            pass
        stop_event.wait(interval)
    results_dict["peak_vram_mb"] = peak_mb


def run_training_check(config_path: str, max_steps: int = 50, output_dir: str = "/tmp/efficiency_check"):
    """Run a short training and capture efficiency metrics via streaming output."""
    run_name = Path(config_path).stem
    run_output_dir = os.path.join(output_dir, run_name)
    log_path = os.path.join(output_dir, f"{run_name}.log")

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "-m", "axolotl.cli.train",
        config_path,
        "--max_steps", str(max_steps),
        "--output_dir", run_output_dir,
        "--save_strategy", "no",
        "--eval_strategy", "no",
        "--report_to", "none",
    ]

    print(f"\n{'='*60}")
    print(f"Measuring efficiency: {run_name}")
    print(f"Config: {config_path}")
    print(f"Max steps: {max_steps}")
    print(f"Log: {log_path}")
    print(f"{'='*60}")

    # Start GPU memory polling in background
    stop_event = threading.Event()
    gpu_results = {}
    gpu_thread = threading.Thread(target=poll_gpu_memory, args=(stop_event, gpu_results), daemon=True)
    gpu_thread.start()

    # Run training with streaming output (no capture deadlock)
    tokens_per_sec = None
    loss_values = []
    start_time = time.time()

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            # Write to log file
            logf.write(line)
            logf.flush()

            # Print to console (so user can see progress)
            sys.stdout.write(line)
            sys.stdout.flush()

            # Parse throughput from training output
            # Typical format: "{'loss': 1.23, ... 'tokens_per_second': 4567, ...}"
            if "tokens_per_second" in line.lower() or "tok/s" in line.lower():
                numbers = re.findall(r'tokens_per_second[\'"\s:]+(\d+\.?\d*)', line, re.IGNORECASE)
                if numbers:
                    tokens_per_sec = float(numbers[-1])

            # Also try to catch throughput from different log formats
            if "samples/s" in line.lower() or "it/s" in line.lower():
                match = re.search(r'(\d+\.?\d*)\s*(?:samples/s|it/s)', line)
                if match and tokens_per_sec is None:
                    # Rough estimate: samples/s × seq_len ≈ tokens/s
                    pass  # Can't reliably convert without knowing seq_len

            # Parse loss
            loss_match = re.search(r"'loss':\s*(\d+\.?\d*)", line)
            if loss_match:
                loss_values.append(float(loss_match.group(1)))

        proc.wait()

    wall_time = time.time() - start_time

    # Stop GPU polling
    stop_event.set()
    gpu_thread.join(timeout=5)

    peak_vram_gb = gpu_results.get("peak_vram_mb", 0) / 1024 if gpu_results.get("peak_vram_mb") else None

    # If tokens_per_sec not found in streaming output, scan log file
    if tokens_per_sec is None:
        try:
            with open(log_path) as f:
                for line in f:
                    tps_match = re.search(r"tokens_per_second['\"\s:]+(\d+\.?\d*)", line, re.IGNORECASE)
                    if tps_match:
                        tokens_per_sec = float(tps_match.group(1))
        except Exception:
            pass

    return {
        "config": config_path,
        "run_name": run_name,
        "max_steps": max_steps,
        "wall_time_seconds": round(wall_time, 1),
        "time_per_step": round(wall_time / max_steps, 2) if max_steps > 0 else None,
        "peak_vram_gb": round(peak_vram_gb, 2) if peak_vram_gb else None,
        "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else None,
        "final_loss": round(loss_values[-1], 4) if loss_values else None,
        "exit_code": proc.returncode,
        "log_path": log_path,
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
            print(f"  Time/step: {metrics['time_per_step']}s")
            print(f"  Wall time: {metrics['wall_time_seconds']}s")
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
