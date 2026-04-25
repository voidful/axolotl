#!/usr/bin/env python3
"""
Summarize the fast CE vs single drift-loss pipeline.

Inputs:
  outputs/paper/<run_tag>/<run_name>/trainer_state.json
  results/paper/<run_tag>/benchmarks/<run_name>/<task>/results_*.json

Outputs:
  results/paper/<run_tag>/fast_drift_summary.json
  results/paper/<run_tag>/fast_drift_summary.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any


METRIC_PRIORITY = {
    "ifeval": [
        "prompt_level_strict_acc,none",
        "inst_level_strict_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none",
    ],
    "mmlu_pro": [
        "exact_match,custom-extract",
        "acc,none",
        "acc_norm,none",
        "exact_match,none",
    ],
}


def load_json(path: str | Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def final_eval_loss(output_dir: Path) -> float | None:
    state = load_json(output_dir / "trainer_state.json")
    if not state:
        return None
    vals: list[float] = []
    for entry in state.get("log_history", []):
        val = entry.get("eval_loss")
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            vals.append(float(val))
    return vals[-1] if vals else None


def find_result_json(task_dir: Path) -> Path | None:
    matches = sorted(glob.glob(str(task_dir / "**" / "results_*.json"), recursive=True))
    if not matches:
        return None
    return Path(matches[-1])


def pick_task_metric(task: str, metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    for key in METRIC_PRIORITY.get(task, []):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return key, float(val)

    for key, val in metrics.items():
        key_l = key.lower()
        if isinstance(val, (int, float)) and (
            "acc" in key_l or "exact_match" in key_l or "score" in key_l
        ):
            return key, float(val)

    return None, None


def load_benchmark(results_root: Path, run: str, task: str) -> dict[str, Any]:
    result_file = find_result_json(results_root / "benchmarks" / run / task)
    if not result_file:
        return {"metric": None, "score": None, "file": None}

    data = load_json(result_file)
    if not data:
        return {"metric": None, "score": None, "file": str(result_file)}

    results = data.get("results", {})
    task_metrics: dict[str, Any] | None = None
    if task in results and isinstance(results[task], dict):
        task_metrics = results[task]
    else:
        for name, metrics in results.items():
            if task in name and isinstance(metrics, dict):
                task_metrics = metrics
                break

    if not task_metrics:
        return {"metric": None, "score": None, "file": str(result_file)}

    metric, score = pick_task_metric(task, task_metrics)
    return {"metric": metric, "score": score, "file": str(result_file)}


def run_name(run_tag: str, method: str, regime: str, seed: str) -> str:
    return f"{run_tag}-{method}-{regime}-s{seed}"


def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    clean = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def fmt_loss(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val:.4f}"


def fmt_pct(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val * 100:.2f}"


def fmt_delta(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val * 100:+.2f}"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["eval_loss", "ifeval", "mmlu_pro", "ifeval_delta", "mmlu_pro_delta"]:
        mean, std = mean_std([r[key] for r in rows if r.get(key) is not None])
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", default="fast_drift")
    parser.add_argument("--outputs", default="./outputs/paper/fast_drift")
    parser.add_argument("--results", default="./results/paper/fast_drift")
    parser.add_argument("--regimes", nargs="+", default=["medical", "noise25"])
    parser.add_argument("--seeds", nargs="+", default=["42"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs_root = Path(args.outputs)
    results_root = Path(args.results)
    results_root.mkdir(parents=True, exist_ok=True)

    base_ifeval = load_benchmark(results_root, "base", "ifeval")
    base_mmlu = load_benchmark(results_root, "base", "mmlu_pro")
    base = {
        "run": "base",
        "ifeval": base_ifeval["score"],
        "mmlu_pro": base_mmlu["score"],
        "ifeval_metric": base_ifeval["metric"],
        "mmlu_pro_metric": base_mmlu["metric"],
    }

    rows: list[dict[str, Any]] = []
    for regime in args.regimes:
        for seed in args.seeds:
            for method in ["ce", "drift"]:
                run = run_name(args.run_tag, method, regime, seed)
                output_dir = outputs_root / run
                ifeval = load_benchmark(results_root, run, "ifeval")
                mmlu = load_benchmark(results_root, run, "mmlu_pro")
                row = {
                    "run": run,
                    "method": method,
                    "regime": regime,
                    "seed": seed,
                    "eval_loss": final_eval_loss(output_dir),
                    "ifeval": ifeval["score"],
                    "mmlu_pro": mmlu["score"],
                    "ifeval_metric": ifeval["metric"],
                    "mmlu_pro_metric": mmlu["metric"],
                    "ifeval_delta": (
                        ifeval["score"] - base["ifeval"]
                        if ifeval["score"] is not None and base["ifeval"] is not None
                        else None
                    ),
                    "mmlu_pro_delta": (
                        mmlu["score"] - base["mmlu_pro"]
                        if mmlu["score"] is not None and base["mmlu_pro"] is not None
                        else None
                    ),
                }
                rows.append(row)

    grouped: dict[str, dict[str, Any]] = {}
    for regime in args.regimes:
        for method in ["ce", "drift"]:
            key = f"{method}:{regime}"
            grouped[key] = aggregate(
                [r for r in rows if r["method"] == method and r["regime"] == regime]
            )

    summary = {
        "run_tag": args.run_tag,
        "base": base,
        "rows": rows,
        "grouped": grouped,
    }

    json_path = results_root / "fast_drift_summary.json"
    md_path = results_root / "fast_drift_summary.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines: list[str] = []
    lines.append("# Fast Single Drift-Loss Summary")
    lines.append("")
    lines.append(f"- Base IFEval: {fmt_pct(base['ifeval'])} ({base['ifeval_metric'] or 'missing'})")
    lines.append(f"- Base MMLU-Pro: {fmt_pct(base['mmlu_pro'])} ({base['mmlu_pro_metric'] or 'missing'})")
    lines.append("")
    lines.append("| Regime | Method | Target eval loss | IFEval | IFEval Δ | MMLU-Pro | MMLU-Pro Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for regime in args.regimes:
        for method in ["ce", "drift"]:
            agg = grouped[f"{method}:{regime}"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        regime,
                        method.upper(),
                        fmt_loss(agg["eval_loss_mean"]),
                        fmt_pct(agg["ifeval_mean"]),
                        fmt_delta(agg["ifeval_delta_mean"]),
                        fmt_pct(agg["mmlu_pro_mean"]),
                        fmt_delta(agg["mmlu_pro_delta_mean"]),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Per-Run")
    lines.append("")
    lines.append("| Run | Target eval loss | IFEval | IFEval Δ | MMLU-Pro | MMLU-Pro Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"],
                    fmt_loss(row["eval_loss"]),
                    fmt_pct(row["ifeval"]),
                    fmt_delta(row["ifeval_delta"]),
                    fmt_pct(row["mmlu_pro"]),
                    fmt_delta(row["mmlu_pro_delta"]),
                ]
            )
            + " |"
        )

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

