#!/bin/bash
# ============================================================
# Collect & Summarize Training Results
# ============================================================
# Pulls training metrics from wandb (API) and cross-references
# with local output directories to build a full status report.
#
# Usage:
#   bash scripts/paper/collect_training_results.sh
#   bash scripts/paper/collect_training_results.sh ./outputs/paper
# ============================================================

set -uo pipefail

OUTPUT_BASE="${1:-./outputs/paper}"
WANDB_PROJECT="${2:-drift-trust-neurips2026}"

python3 << 'PYEOF'
import json, os, sys, glob
from collections import defaultdict

output_base = sys.argv[1] if len(sys.argv) > 1 else "./outputs/paper"
wandb_project = os.environ.get("WANDB_PROJECT", "drift-trust-neurips2026")

# ── 1. Scan local output directories ─────────────────────────────
runs = {}
for d in sorted(glob.glob(os.path.join(output_base, "rcca-*"))):
    name = os.path.basename(d)
    info = {"dir": d, "name": name, "complete": False}

    if os.path.isfile(os.path.join(d, "config.json")):
        info["complete"] = True
    elif os.path.isfile(os.path.join(d, "adapter_config.json")):
        info["complete"] = True

    # Parse run name: rcca-{mode}-{regime}-4b-s{seed}
    parts = name.replace("rcca-", "").replace("-4b-", "|").split("|")
    if len(parts) == 2:
        mode_regime = parts[0]
        seed_str = parts[1]
        info["seed"] = seed_str.replace("s", "")

        for regime in ["medical", "math", "noise0", "noise10", "noise25", "noise50", "noise75"]:
            if mode_regime.endswith(f"-{regime}"):
                info["mode"] = mode_regime[: -(len(regime) + 1)]
                info["regime"] = regime
                break
        else:
            info["mode"] = mode_regime
            info["regime"] = "unknown"

    # Disk size
    size = 0
    for root, dirs, files in os.walk(d):
        for f in files:
            try:
                size += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    info["size_gib"] = size / (1024**3)

    runs[name] = info

# ── 2. Pull metrics from wandb ───────────────────────────────────
wandb_metrics = {}
wandb_available = False

try:
    import wandb
    api = wandb.Api()

    # Get entity from wandb settings
    entity = api.default_entity
    project_path = f"{entity}/{wandb_project}" if entity else wandb_project

    print(f"  Fetching from wandb: {project_path} ...")
    api_runs = api.runs(project_path, per_page=200)

    for wr in api_runs:
        wname = wr.name  # auto-generated name like "absurd-music-31"
        wstate = wr.state
        wsummary = wr.summary._json_dict if hasattr(wr.summary, '_json_dict') else dict(wr.summary)
        wconfig = wr.config if hasattr(wr, 'config') else {}

        entry = {
            "wandb_id": wr.id,
            "wandb_name": wname,
            "wandb_state": wstate,
        }
        try:
            entry["wandb_url"] = wr.url
        except Exception:
            pass

        # Extract output_dir and configured run name from config
        if isinstance(wconfig, dict):
            if "output_dir" in wconfig:
                entry["output_dir"] = wconfig["output_dir"]
            # The configured run name (what we set in yaml)
            for key in ["wandb_run_name", "_wandb", "run_name"]:
                if key in wconfig and isinstance(wconfig[key], str):
                    entry["config_run_name"] = wconfig[key]
                    break

        # Extract key metrics from summary
        for key_map in [
            ("train/train_loss", "final_train_loss"),
            ("train_loss", "final_train_loss"),
            ("loss", "final_train_loss"),
            ("eval/loss", "final_eval_loss"),
            ("eval_loss", "final_eval_loss"),
            ("epoch", "final_epoch"),
            ("train/epoch", "final_epoch"),
        ]:
            src, dst = key_map
            if src in wsummary and dst not in entry:
                entry[dst] = wsummary[src]

        # eval ppl
        for k in ["eval/ppl", "eval_ppl", "eval/perplexity"]:
            if k in wsummary:
                entry["eval_ppl"] = wsummary[k]
                break

        # Steps
        entry["total_steps"] = wsummary.get("_step", wsummary.get("train/global_step", 0))

        wandb_metrics[wname] = entry

    wandb_available = True
    print(f"  Found {len(wandb_metrics)} wandb runs.")
    # Diagnostic: show first 5 with output_dir
    sample_entries = list(wandb_metrics.values())[:5]
    for e in sample_entries:
        od = e.get("output_dir", "N/A")
        loss = e.get("final_train_loss", "---")
        crn = e.get("config_run_name", "N/A")
        print(f"    {e['wandb_name']:<35} output_dir={od}  config_run_name={crn}  loss={loss}")
    print()

except ImportError:
    print("  ⚠️  wandb not installed. Install with: pip install wandb")
    print("  Falling back to local wandb log files...\n")
except Exception as e:
    print(f"  ⚠️  wandb API error: {e}")
    print("  Falling back to local wandb log files...\n")

# ── 2b. Fallback: local wandb logs ───────────────────────────────
if not wandb_available:
    wandb_dirs = sorted(glob.glob("./wandb/run-*"))
    if wandb_dirs:
        print(f"  Found {len(wandb_dirs)} local wandb run dirs.")

        # Diagnostic: show first dir's structure
        sample_wd = wandb_dirs[0]
        print(f"  Sample dir: {sample_wd}")
        files_dir = os.path.join(sample_wd, "files")
        if os.path.isdir(files_dir):
            print(f"  Files: {', '.join(sorted(os.listdir(files_dir))[:15])}")
        # Check root of wandb dir too
        root_files = [f for f in os.listdir(sample_wd) if os.path.isfile(os.path.join(sample_wd, f))]
        if root_files:
            print(f"  Root files: {', '.join(sorted(root_files)[:10])}")
        print()

    for wd in wandb_dirs:
        try:
            wname = os.path.basename(wd)  # default fallback name
            entry = {"wandb_state": "finished", "wandb_dir": wd}
            metrics_found = False

            # ── Try to get the run name ──
            # Method 1: wandb-metadata.json
            meta_path = os.path.join(wd, "files", "wandb-metadata.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                wname = meta.get("displayName", meta.get("codePath", wname))

            # Method 2: config.yaml for wandb_run_name / output_dir
            config_yaml_path = os.path.join(wd, "files", "config.yaml")
            output_dir_from_config = None
            if os.path.isfile(config_yaml_path):
                try:
                    import yaml
                    with open(config_yaml_path) as f:
                        wcfg = yaml.safe_load(f)
                    # wandb config stores values as {value: X}
                    if isinstance(wcfg, dict):
                        for key in ["wandb_run_name", "run_name"]:
                            if key in wcfg:
                                val = wcfg[key]
                                if isinstance(val, dict) and "value" in val:
                                    wname = val["value"]
                                elif isinstance(val, str):
                                    wname = val
                                break
                        # Also get output_dir for direct matching
                        if "output_dir" in wcfg:
                            od = wcfg["output_dir"]
                            if isinstance(od, dict) and "value" in od:
                                output_dir_from_config = od["value"]
                            elif isinstance(od, str):
                                output_dir_from_config = od
                except ImportError:
                    # No yaml module, try simple parsing
                    with open(config_yaml_path) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("wandb_run_name:") or line.startswith("run_name:"):
                                # Handle both "key: value" and "key:\n  value: X"
                                parts = line.split(":", 1)
                                if len(parts) == 2 and parts[1].strip():
                                    wname = parts[1].strip().strip("'\"")
                            if line.startswith("output_dir:"):
                                parts = line.split(":", 1)
                                if len(parts) == 2 and parts[1].strip():
                                    output_dir_from_config = parts[1].strip().strip("'\"")
                except Exception:
                    pass

            entry["wandb_name"] = wname
            if output_dir_from_config:
                entry["output_dir"] = output_dir_from_config

            # ── Get metrics ──
            # Method 1: wandb-summary.json
            summary_path = os.path.join(wd, "files", "wandb-summary.json")
            if os.path.isfile(summary_path):
                with open(summary_path) as f:
                    ws = json.load(f)
                for key_map in [
                    ("train/train_loss", "final_train_loss"),
                    ("train_loss", "final_train_loss"),
                    ("loss", "final_train_loss"),
                    ("eval/loss", "final_eval_loss"),
                    ("eval_loss", "final_eval_loss"),
                    ("epoch", "final_epoch"),
                    ("train/epoch", "final_epoch"),
                ]:
                    src, dst = key_map
                    if src in ws and dst not in entry:
                        entry[dst] = ws[src]
                for k in ["eval/ppl", "eval_ppl"]:
                    if k in ws:
                        entry["eval_ppl"] = ws[k]
                        break
                entry["total_steps"] = ws.get("_step", 0)
                if entry.get("final_train_loss") is not None:
                    metrics_found = True

            # Method 2: Parse wandb history jsonl (last lines)
            if not metrics_found:
                history_path = os.path.join(wd, "files", "wandb-history.jsonl")
                if os.path.isfile(history_path):
                    try:
                        last_lines = []
                        with open(history_path) as f:
                            for line in f:
                                last_lines.append(line)
                                if len(last_lines) > 50:
                                    last_lines.pop(0)
                        for line in reversed(last_lines):
                            try:
                                row = json.loads(line)
                                if "loss" in row and "final_train_loss" not in entry:
                                    entry["final_train_loss"] = row["loss"]
                                    entry["final_epoch"] = row.get("epoch")
                                if "eval_loss" in row and "final_eval_loss" not in entry:
                                    entry["final_eval_loss"] = row["eval_loss"]
                                    entry["eval_ppl"] = row.get("eval_ppl")
                            except json.JSONDecodeError:
                                continue
                    except Exception:
                        pass

            wandb_metrics[wname] = entry
        except Exception:
            pass

    if wandb_metrics:
        wandb_available = True
        # Print first 5 discovered names for debugging
        names = sorted(wandb_metrics.keys())[:5]
        print(f"  Parsed {len(wandb_metrics)} wandb runs. First 5 names:")
        for n in names:
            od = wandb_metrics[n].get("output_dir", "")
            loss = wandb_metrics[n].get("final_train_loss", "---")
            print(f"    {n:<55} loss={loss}  output_dir={od}")
        print()

# ── 3. Match wandb runs → local runs ─────────────────────────────
# wandb_run_name pattern from config: {mode}-{dataset_basename}-s{seed}
# Example: ce-medical_flashcards_10k-s42
# Map dataset basenames to regimes
DATASET_TO_REGIME = {
    "medical_flashcards_10k": "medical",
    "numina_math_cot_10k": "math",
    "ultrafeedback_noisy_0pct_10k": "noise0",
    "ultrafeedback_noisy_10pct_10k": "noise10",
    "ultrafeedback_noisy_25pct_10k": "noise25",
    "ultrafeedback_noisy_50pct_10k": "noise50",
    "ultrafeedback_noisy_75pct_10k": "noise75",
}

matched = 0
for wname, winfo in wandb_metrics.items():
    already_matched = False

    # Strategy A: match by wandb run name pattern: {mode}-{dataset}-s{seed}
    for dataset_base, regime in DATASET_TO_REGIME.items():
        if dataset_base in wname:
            prefix = wname.split(f"-{dataset_base}")[0] if f"-{dataset_base}" in wname else None
            suffix = wname.split(f"{dataset_base}-")[-1] if f"{dataset_base}-" in wname else None

            if prefix and suffix and suffix.startswith("s"):
                seed = suffix[1:]
                mode = prefix
                local_name = f"rcca-{mode}-{regime}-4b-s{seed}"

                if local_name in runs:
                    for k, v in winfo.items():
                        if k not in runs[local_name] or runs[local_name].get(k) is None:
                            runs[local_name][k] = v
                    runs[local_name]["metrics_source"] = "wandb"
                    matched += 1
                    already_matched = True
            break

    # Strategy B: match by output_dir from wandb config
    if not already_matched and "output_dir" in winfo:
        od = winfo["output_dir"]
        # output_dir looks like: ./outputs/paper/rcca-ce-medical-4b-s42
        od_basename = os.path.basename(od.rstrip("/"))
        if od_basename in runs:
            for k, v in winfo.items():
                if k not in runs[od_basename] or runs[od_basename].get(k) is None:
                    runs[od_basename][k] = v
            runs[od_basename]["metrics_source"] = "wandb(output_dir)"
            matched += 1

if wandb_available:
    print(f"  Matched {matched}/{len(wandb_metrics)} wandb runs to local output dirs.")

# ── 4. Pretty Print ──────────────────────────────────────────────
MODE_LABELS = {
    "ce": "CE",
    "hardness": "Hardness",
    "drift_trust_s": "DT-S",
    "drift_trust_a": "DT-A",
}

def fmt_loss(val):
    if val is None: return "---"
    return f"{val:.4f}"

total_complete = sum(1 for r in runs.values() if r["complete"])
total_incomplete = sum(1 for r in runs.values() if not r["complete"])
total_with_metrics = sum(1 for r in runs.values() if r.get("final_train_loss") is not None)

print()
print("=" * 110)
print("  DRIFT-TRUST TRAINING STATUS REPORT")
print(f"  Scanned: {output_base}")
print(f"  Total runs: {len(runs)}  |  ✅ Complete: {total_complete}  |  ❌ Incomplete: {total_incomplete}  |  📊 With metrics: {total_with_metrics}")
print("=" * 110)

REGIME_ORDER = ["medical", "math", "noise0", "noise10", "noise25", "noise50", "noise75"]
REGIME_TITLES = {
    "medical": "Battle C: Medical SFT",
    "math":    "Battle B: Math SFT",
    "noise0":  "Noise Sweep: 0% noise",
    "noise10": "Noise Sweep: 10% noise",
    "noise25": "Noise Sweep: 25% noise (primary)",
    "noise50": "Noise Sweep: 50% noise",
    "noise75": "Noise Sweep: 75% noise",
}
REGIME_MODES = {
    "medical": ["ce", "hardness", "drift_trust_s", "drift_trust_a"],
    "math":    ["ce", "hardness", "drift_trust_s", "drift_trust_a"],
    "noise25": ["ce", "hardness", "drift_trust_s", "drift_trust_a"],
}
DEFAULT_NOISE_MODES = ["ce", "drift_trust_s", "drift_trust_a"]
SEEDS = ["42", "123", "456"]

for regime in REGIME_ORDER:
    title = REGIME_TITLES.get(regime, regime)
    expected_modes = REGIME_MODES.get(regime, DEFAULT_NOISE_MODES)

    print(f"\n{'─' * 110}")
    print(f"  {title}")
    print(f"{'─' * 110}")
    print(f"  {'Mode':<16} {'Seed':>5}  {'Ckpt':>4}  {'Train Loss':>11} {'Eval Loss':>11} {'Eval PPL':>10} {'Epoch':>7}  {'Size':>7}  {'Source':<8}")
    print(f"  {'-'*16} {'-'*5}  {'-'*4}  {'-'*11} {'-'*11} {'-'*10} {'-'*7}  {'-'*7}  {'-'*8}")

    for mode in expected_modes:
        label = MODE_LABELS.get(mode, mode)
        for seed in SEEDS:
            run_name = f"rcca-{mode}-{regime}-4b-s{seed}"
            info = runs.get(run_name)

            if info is None:
                print(f"  {label:<16} {seed:>5}  {'⬜':>4}  {'---':>11} {'---':>11} {'---':>10} {'---':>7}  {'---':>7}  {'---':<8}")
            else:
                ckpt = "✅" if info["complete"] else "❌"
                train_loss = fmt_loss(info.get("final_train_loss"))
                eval_loss = fmt_loss(info.get("final_eval_loss"))
                eval_ppl = fmt_loss(info.get("eval_ppl"))
                epoch = info.get("final_epoch", "---")
                if isinstance(epoch, float):
                    epoch = f"{epoch:.1f}"
                size = f"{info.get('size_gib', 0):.1f}G"
                source = info.get("metrics_source", "---")
                if len(source) > 8:
                    source = source[:8]
                print(f"  {label:<16} {seed:>5}  {ckpt:>4}  {train_loss:>11} {eval_loss:>11} {eval_ppl:>10} {str(epoch):>7}  {size:>7}  {source:<8}")

# ── 5. Aggregate table ────────────────────────────────────────────
import statistics

print(f"\n{'=' * 110}")
print("  AGGREGATE: Mean ± Std of Eval Loss (across 3 seeds)")
print(f"{'=' * 110}")

print(f"\n  {'Mode':<16}", end="")
for r in REGIME_ORDER:
    print(f"  {r:>16}", end="")
print()
print(f"  {'-'*16}", end="")
for _ in REGIME_ORDER:
    print(f"  {'-'*16}", end="")
print()

all_modes = ["ce", "hardness", "drift_trust_s", "drift_trust_a"]
for mode in all_modes:
    label = MODE_LABELS.get(mode, mode)
    print(f"  {label:<16}", end="")
    for regime in REGIME_ORDER:
        expected_modes = REGIME_MODES.get(regime, DEFAULT_NOISE_MODES)
        if mode not in expected_modes:
            print(f"  {'n/a':>16}", end="")
            continue

        vals = []
        for seed in SEEDS:
            run_name = f"rcca-{mode}-{regime}-4b-s{seed}"
            info = runs.get(run_name)
            if info and info.get("final_eval_loss") is not None:
                vals.append(info["final_eval_loss"])

        if not vals:
            print(f"  {'---':>16}", end="")
        elif len(vals) == 1:
            print(f"  {vals[0]:>14.4f}  ", end="")
        else:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            print(f"  {m:.4f}±{s:.4f}".rjust(16), end="")
    print()

# ── 6. Disk usage summary ────────────────────────────────────────
print(f"\n{'─' * 110}")
print("  DISK USAGE SUMMARY")
print(f"{'─' * 110}")

total_size = sum(r.get("size_gib", 0) for r in runs.values())
complete_size = sum(r.get("size_gib", 0) for r in runs.values() if r["complete"])
incomplete_size = sum(r.get("size_gib", 0) for r in runs.values() if not r["complete"])

print(f"  Total:      {total_size:>8.1f} GiB ({len(runs)} runs)")
print(f"  Complete:   {complete_size:>8.1f} GiB ({total_complete} runs)")
print(f"  Incomplete: {incomplete_size:>8.1f} GiB ({total_incomplete} runs)")

# Spot 0-size runs (likely crashed/incomplete saves)
zero_runs = [n for n, r in runs.items() if r.get("size_gib", 0) < 1.0 and r["complete"]]
if zero_runs:
    print(f"\n  ⚠️  Suspiciously small 'complete' runs (< 1 GiB):")
    for n in zero_runs:
        print(f"     {n}  ({runs[n].get('size_gib', 0):.2f} GiB)")

# ── 7. Unmatched wandb runs ──────────────────────────────────────
if wandb_available:
    unmatched_wandb = {k: v for k, v in wandb_metrics.items()
                       if not any(r.get("wandb_name") == k or r.get("wandb_id") == v.get("wandb_id")
                                  for r in runs.values())}
    if unmatched_wandb:
        print(f"\n{'─' * 110}")
        print(f"  UNMATCHED WANDB RUNS ({len(unmatched_wandb)})")
        print(f"{'─' * 110}")
        for wname, winfo in sorted(unmatched_wandb.items()):
            loss = fmt_loss(winfo.get("final_train_loss"))
            state = winfo.get("wandb_state", "?")
            print(f"  {wname:<55} loss={loss}  state={state}")

# ── 8. Export JSON ────────────────────────────────────────────────
export_data = {}
for name, info in runs.items():
    export_data[name] = {
        "complete": info["complete"],
        "mode": info.get("mode"),
        "regime": info.get("regime"),
        "seed": info.get("seed"),
        "final_train_loss": info.get("final_train_loss"),
        "final_eval_loss": info.get("final_eval_loss"),
        "eval_ppl": info.get("eval_ppl"),
        "final_epoch": info.get("final_epoch"),
        "total_steps": info.get("total_steps"),
        "size_gib": round(info.get("size_gib", 0), 2),
        "metrics_source": info.get("metrics_source"),
    }

out_json = os.path.join(output_base, "training_status.json")
with open(out_json, "w") as f:
    json.dump(export_data, f, indent=2)
print(f"\n  Exported to: {out_json}")
print("=" * 110)
PYEOF
