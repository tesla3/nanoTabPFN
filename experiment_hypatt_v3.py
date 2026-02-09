"""
Hypernetwork Attention v3: Comprehensive 4-Phase Sweep
======================================================

Previous findings:
- v1: emb=512, 10M params, rms_head+mlp_silu scored 0.714 (overparameterized, LR too low)
- v2: emb=96, 4 attn types x 5 LRs, baseline 0.744

This experiment runs a fully automated 4-phase sweep:
  Phase 1: Architecture sweep (5 baseline + 7 hypatt archs x 4 LRs, 2500 steps)
  Phase 2: LR sweep (top 3 archs from P1 x 6 LRs x 2 attn types)
  Phase 3: Ablations (best config from P2 with mlp_linear, softmax, depth 2/4)
  Phase 4: Extended training (top 2 configs at 5000 steps)

Crash-safe: saves results after every run, skips completed configs on restart.
"""

import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import NanoTabPFNModel, NanoTabPFNClassifier
from train import PriorDumpDataLoader, get_default_device, get_openml_datasets, eval_model

import schedulefree
from sklearn.model_selection import StratifiedKFold
from torch import nn

# ============================================================================
# Constants
# ============================================================================

RESULTS_PATH = os.path.join(os.path.dirname(__file__) or ".", "experiment_results_v3.json")
PRIOR_PATH = os.path.join(os.path.dirname(__file__) or ".", "300k_150x5_2.h5")
SEED = 42
NUM_OUTPUTS = 2  # binary classification datasets

# ============================================================================
# Crash-safe results management
# ============================================================================

def load_results():
    """Load existing results from disk, or return empty dict."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {"runs": [], "phase_summaries": {}}


def save_results(results):
    """Save results to disk atomically (write to temp, then rename)."""
    tmp_path = RESULTS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, RESULTS_PATH)


def run_key(phase, config):
    """Create a unique key for a run to detect already-completed configs."""
    return (f"p{phase}_"
            f"tn{config['target_network']}_an{config['attn_norm']}_"
            f"e{config['embedding_size']}_h{config['num_heads']}_"
            f"l{config['num_layers']}_lr{config['lr']}_"
            f"s{config['num_steps']}_bs{config['batch_size']}_"
            f"acc{config.get('accumulation_steps', 1)}")


def get_completed_keys(results):
    """Return set of run keys that are already completed."""
    return {r["run_key"] for r in results["runs"]}


# ============================================================================
# Training with gradient accumulation
# ============================================================================

_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def evaluate_model_cv(model, device, datasets):
    """Evaluate using 5-fold CV ROC AUC on OpenML datasets."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    scores = {}
    for name, (X, y) in datasets.items():
        targets = []
        probabilities = []
        for train_idx, test_idx in _skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            targets.append(y_test)
            clf = NanoTabPFNClassifier(model, device)
            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)
            if prob.shape[1] == 2:
                prob = prob[:, 1]
            probabilities.append(prob)
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0)
        if np.isnan(probabilities).any():
            scores[name] = float("nan")
        else:
            try:
                scores[name] = float(roc_auc_score(targets, probabilities, multi_class="ovr"))
            except ValueError:
                scores[name] = float("nan")

    valid = [v for v in scores.values() if not np.isnan(v)]
    scores["MEAN"] = float(np.mean(valid)) if valid else float("nan")
    return scores


def train_and_eval(config, device, eval_datasets):
    """Train a model with gradient accumulation and periodic eval.

    Args:
        config: dict with keys: embedding_size, num_heads, mlp_hidden_size,
                num_layers, num_outputs, target_network, attn_norm,
                lr, num_steps, batch_size, accumulation_steps
        device: torch device
        eval_datasets: dict of OpenML datasets

    Returns:
        dict with final_scores, history, n_params, train_time
    """
    # Reset seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = NanoTabPFNModel(
        embedding_size=config["embedding_size"],
        num_attention_heads=config["num_heads"],
        mlp_hidden_size=config["mlp_hidden_size"],
        num_layers=config["num_layers"],
        num_outputs=config["num_outputs"],
        target_network=config["target_network"],
        attn_norm=config["attn_norm"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    model.to(device)

    accum_steps = config.get("accumulation_steps", 1)
    effective_bs = config["batch_size"] * accum_steps

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=config["lr"], weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    eval_every = config.get("eval_every", 500)
    num_steps = config["num_steps"]

    history = []
    train_time = 0.0
    step = 0
    accum_count = 0

    prior = PriorDumpDataLoader(
        PRIOR_PATH, num_steps=num_steps * accum_steps, batch_size=config["batch_size"], device=device
    )

    optimizer.zero_grad()
    for full_data in prior:
        if step >= num_steps:
            break

        t0 = time.time()
        train_test_split_index = full_data["train_test_split_index"]
        x_batch = full_data["x"].to(device)
        y_batch = full_data["y"].to(device)

        # Skip batches with NaN (corrupted entries in HDF5 at ~91k, 176k, etc.)
        if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
            train_time += time.time() - t0
            continue

        data = (x_batch, y_batch[:, :train_test_split_index])
        targets = y_batch

        output = model(data, train_test_split_index=train_test_split_index)
        targets = targets[:, train_test_split_index:]
        targets = targets.reshape((-1,)).to(torch.long)
        output = output.view(-1, output.shape[-1])

        loss = criterion(output, targets).mean()

        # Early exit on NaN loss (model divergence, not bad data)
        if torch.isnan(loss):
            print(f"    NaN loss at step {step}, aborting run")
            del model, optimizer, prior
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            nan_scores = {name: float("nan") for name in eval_datasets}
            nan_scores["MEAN"] = float("nan")
            return {
                "final_scores": nan_scores,
                "history": history,
                "n_params": n_params,
                "train_time": round(train_time, 1),
                "effective_batch_size": effective_bs,
            }

        # Scale loss for gradient accumulation
        (loss / accum_steps).backward()
        train_time += time.time() - t0

        accum_count += 1
        if accum_count < accum_steps:
            continue

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        accum_count = 0
        step += 1

        if step % eval_every == 0 or step == num_steps:
            model.eval()
            optimizer.eval()
            scores = evaluate_model_cv(model, device, eval_datasets)
            history.append({
                "step": step,
                "train_time": round(train_time, 1),
                "loss": round(loss.item(), 4),
                "mean_roc_auc": scores["MEAN"],
            })
            print(f"    step {step:5d} | loss {loss.item():.4f} | "
                  f"roc_auc {scores['MEAN']:.4f} | time {train_time:.1f}s")
            model.train()
            optimizer.train()

    # Final eval
    model.eval()
    optimizer.eval()
    final_scores = evaluate_model_cv(model, device, eval_datasets)

    # Cleanup
    del model, optimizer, prior
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "final_scores": final_scores,
        "history": history,
        "n_params": n_params,
        "train_time": round(train_time, 1),
        "effective_batch_size": effective_bs,
    }


# ============================================================================
# Phase 1: Architecture Sweep
# ============================================================================

def phase1_configs():
    """Architecture sweep with per-attention-type arch configs and LR sweep.

    Baseline uses standard head counts. Hypernetwork uses many small heads
    (head_dim=8 or 16) since each head independently applies attention weights
    twice, giving more hypernetwork capacity.
    """
    # Baseline: standard head sizes (head_dim=24-32)
    baseline_archs = [
        {"embedding_size": 96,  "num_heads": 4,  "mlp_hidden_size": 192},   # head_dim=24
        {"embedding_size": 128, "num_heads": 4,  "mlp_hidden_size": 256},   # head_dim=32
        {"embedding_size": 128, "num_heads": 8,  "mlp_hidden_size": 256},   # head_dim=16
        {"embedding_size": 192, "num_heads": 8,  "mlp_hidden_size": 384},   # head_dim=24
        {"embedding_size": 256, "num_heads": 8,  "mlp_hidden_size": 512},   # head_dim=32
    ]
    # Hypernetwork: many small heads (head_dim=8-16)
    hypatt_archs = [
        {"embedding_size": 96,  "num_heads": 12, "mlp_hidden_size": 192},   # head_dim=8
        {"embedding_size": 128, "num_heads": 16, "mlp_hidden_size": 256},   # head_dim=8
        {"embedding_size": 128, "num_heads": 8,  "mlp_hidden_size": 256},   # head_dim=16
        {"embedding_size": 192, "num_heads": 24, "mlp_hidden_size": 384},   # head_dim=8
        {"embedding_size": 192, "num_heads": 12, "mlp_hidden_size": 384},   # head_dim=16
        {"embedding_size": 256, "num_heads": 32, "mlp_hidden_size": 512},   # head_dim=8
        {"embedding_size": 256, "num_heads": 16, "mlp_hidden_size": 512},   # head_dim=16
    ]
    learning_rates = [5e-5, 2e-4, 1e-3, 4e-3]

    configs = []
    for arch in baseline_archs:
        for lr in learning_rates:
            configs.append({
                **arch,
                "num_layers": 3,
                "num_outputs": NUM_OUTPUTS,
                "target_network": "default",
                "attn_norm": "softmax",
                "attn_label": "baseline",
                "lr": lr,
                "num_steps": 2500,
                "batch_size": 32,
                "accumulation_steps": 4,  # effective bs=128
                "eval_every": 500,
            })
    for arch in hypatt_archs:
        for lr in learning_rates:
            configs.append({
                **arch,
                "num_layers": 3,
                "num_outputs": NUM_OUTPUTS,
                "target_network": "mlp_silu",
                "attn_norm": "rms_head",
                "attn_label": "hypatt",
                "lr": lr,
                "num_steps": 2500,
                "batch_size": 32,
                "accumulation_steps": 4,  # effective bs=128
                "eval_every": 500,
            })
    return configs


def phase1_select_top(results, n=3):
    """Select top N architectures from Phase 1.

    For each (arch, attn_type) pair, take the best LR. Then for each arch,
    average across attn types. Rank by that average.
    """
    # Group by (arch, attn_type, lr)
    arch_attn_lr = {}
    for r in results["runs"]:
        if not r["run_key"].startswith("p1_"):
            continue
        cfg = r["config"]
        arch_key = f"e{cfg['embedding_size']}_h{cfg['num_heads']}"
        attn_key = cfg["target_network"]
        group_key = (arch_key, attn_key)
        score = r["final_scores"]["MEAN"]
        if np.isnan(score):
            continue
        if group_key not in arch_attn_lr or score > arch_attn_lr[group_key]:
            arch_attn_lr[group_key] = score

    # For each arch, average the best score across attn types
    arch_best = {}
    for (arch_key, attn_key), score in arch_attn_lr.items():
        if arch_key not in arch_best:
            arch_best[arch_key] = []
        arch_best[arch_key].append(score)
    arch_avg = {k: np.mean(v) for k, v in arch_best.items()}
    sorted_archs = sorted(arch_avg.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Phase 1 architecture ranking (best LR per attn type, averaged):")
    for i, (arch, score) in enumerate(sorted_archs):
        marker = " <-- TOP" if i < n else ""
        print(f"    {i+1}. {arch}: {score:.4f}{marker}")

    # Parse back to configs
    top_archs = []
    for arch_key, _ in sorted_archs[:n]:
        parts = arch_key.split("_")
        emb = int(parts[0][1:])
        heads = int(parts[1][1:])
        top_archs.append({
            "embedding_size": emb,
            "num_heads": heads,
            "mlp_hidden_size": emb * 2,
        })
    return top_archs


# ============================================================================
# Phase 2: LR Sweep
# ============================================================================

def phase2_configs(top_archs):
    """Top 3 architectures x 6 LRs x 2 attention types."""
    learning_rates = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    attention_types = [
        {"target_network": "default",  "attn_norm": "softmax",  "label": "baseline"},
        {"target_network": "mlp_silu", "attn_norm": "rms_head", "label": "hypatt"},
    ]

    configs = []
    for arch in top_archs:
        for attn in attention_types:
            for lr in learning_rates:
                configs.append({
                    **arch,
                    "num_layers": 3,
                    "num_outputs": NUM_OUTPUTS,
                    "target_network": attn["target_network"],
                    "attn_norm": attn["attn_norm"],
                    "attn_label": attn["label"],
                    "lr": lr,
                    "num_steps": 2500,
                    "batch_size": 32,
                    "accumulation_steps": 4,
                    "eval_every": 500,
                })
    return configs


def phase2_select_best(results):
    """Select best overall config from Phase 2."""
    best = None
    best_score = -1
    for r in results["runs"]:
        if not r["run_key"].startswith("p2_"):
            continue
        score = r["final_scores"]["MEAN"]
        if score > best_score:
            best_score = score
            best = r
    return best


def phase2_select_top(results, n=2):
    """Select top N configs from Phase 2 by mean ROC AUC."""
    p2_runs = [r for r in results["runs"] if r["run_key"].startswith("p2_")]
    p2_runs.sort(key=lambda r: r["final_scores"]["MEAN"], reverse=True)
    return p2_runs[:n]


# ============================================================================
# Phase 3: Ablations
# ============================================================================

def phase3_configs(best_config):
    """Ablations on the best config from Phase 2."""
    cfg = best_config["config"]
    base = {
        "embedding_size": cfg["embedding_size"],
        "num_heads": cfg["num_heads"],
        "mlp_hidden_size": cfg["mlp_hidden_size"],
        "lr": cfg["lr"],
        "num_steps": 2500,
        "batch_size": 32,
        "accumulation_steps": 4,
        "eval_every": 500,
        "num_outputs": NUM_OUTPUTS,
    }

    configs = []

    # 1. mlp_linear (no activation between double attention)
    configs.append({
        **base, "num_layers": 3,
        "target_network": "mlp_linear", "attn_norm": "rms_head",
        "attn_label": "ablation_mlp_linear",
    })

    # 2. mlp_silu with softmax norm (instead of rms_head)
    configs.append({
        **base, "num_layers": 3,
        "target_network": "mlp_silu", "attn_norm": "softmax",
        "attn_label": "ablation_softmax_silu",
    })

    # 3. mlp_linear with softmax norm
    configs.append({
        **base, "num_layers": 3,
        "target_network": "mlp_linear", "attn_norm": "softmax",
        "attn_label": "ablation_softmax_linear",
    })

    # 4. default with rms_head (standard attn but rms norm)
    configs.append({
        **base, "num_layers": 3,
        "target_network": "default", "attn_norm": "rms_head",
        "attn_label": "ablation_rms_default",
    })

    # 5. Depth 2 layers - baseline
    configs.append({
        **base, "num_layers": 2,
        "target_network": "default", "attn_norm": "softmax",
        "attn_label": "ablation_depth2_baseline",
    })

    # 6. Depth 2 layers - hypatt
    configs.append({
        **base, "num_layers": 2,
        "target_network": "mlp_silu", "attn_norm": "rms_head",
        "attn_label": "ablation_depth2_hypatt",
    })

    # 7. Depth 4 layers - baseline
    configs.append({
        **base, "num_layers": 4,
        "target_network": "default", "attn_norm": "softmax",
        "attn_label": "ablation_depth4_baseline",
    })

    # 8. Depth 4 layers - hypatt
    configs.append({
        **base, "num_layers": 4,
        "target_network": "mlp_silu", "attn_norm": "rms_head",
        "attn_label": "ablation_depth4_hypatt",
    })

    return configs


# ============================================================================
# Phase 4: Extended Training
# ============================================================================

def phase4_configs(top_runs):
    """Top 2 configs at 5000 steps."""
    configs = []
    for r in top_runs:
        cfg = r["config"]
        configs.append({
            "embedding_size": cfg["embedding_size"],
            "num_heads": cfg["num_heads"],
            "mlp_hidden_size": cfg["mlp_hidden_size"],
            "num_layers": cfg["num_layers"],
            "num_outputs": NUM_OUTPUTS,
            "target_network": cfg["target_network"],
            "attn_norm": cfg["attn_norm"],
            "attn_label": cfg.get("attn_label", "extended"),
            "lr": cfg["lr"],
            "num_steps": 5000,
            "batch_size": 32,
            "accumulation_steps": 4,
            "eval_every": 500,
        })
    return configs


# ============================================================================
# Run a phase
# ============================================================================

def run_phase(phase_num, configs, device, eval_datasets, results):
    """Run all configs for a phase, skipping already-completed runs."""
    completed = get_completed_keys(results)
    total = len(configs)
    skipped = 0

    for i, cfg in enumerate(configs):
        key = run_key(phase_num, cfg)
        if key in completed:
            skipped += 1
            print(f"  [{i+1}/{total}] SKIP (already done): {key}")
            continue

        arch_str = f"e{cfg['embedding_size']}/h{cfg['num_heads']}/l{cfg['num_layers']}"
        attn_str = f"{cfg['attn_norm']}+{cfg['target_network']}"
        print(f"\n  [{i+1}/{total}] {arch_str} | {attn_str} | lr={cfg['lr']}")

        result = train_and_eval(cfg, device, eval_datasets)

        run = {
            "run_key": key,
            "phase": phase_num,
            "config": cfg,
            "n_params": result["n_params"],
            "final_scores": result["final_scores"],
            "history": result["history"],
            "train_time": result["train_time"],
            "effective_batch_size": result["effective_batch_size"],
        }
        results["runs"].append(run)
        save_results(results)

        print(f"    FINAL: roc_auc={result['final_scores']['MEAN']:.4f} | "
              f"params={result['n_params']:,} | time={result['train_time']:.1f}s")

    if skipped > 0:
        print(f"\n  Skipped {skipped}/{total} already-completed runs")


def print_phase_summary(phase_num, results):
    """Print a summary table for a phase."""
    runs = [r for r in results["runs"] if r["phase"] == phase_num]
    if not runs:
        return

    runs.sort(key=lambda r: r["final_scores"]["MEAN"], reverse=True)

    print(f"\n{'='*90}")
    print(f"PHASE {phase_num} SUMMARY (sorted by mean ROC AUC)")
    print(f"{'='*90}")
    print(f"{'Arch':<18s} {'Attn Type':<25s} {'LR':>8s} {'Params':>8s} {'ROC AUC':>8s} {'Time':>7s}")
    print(f"{'-'*90}")

    for r in runs:
        cfg = r["config"]
        arch = f"e{cfg['embedding_size']}/h{cfg['num_heads']}/l{cfg['num_layers']}"
        attn = f"{cfg['attn_norm']}+{cfg['target_network']}"
        print(f"{arch:<18s} {attn:<25s} {cfg['lr']:>8.1e} {r['n_params']:>8,d} "
              f"{r['final_scores']['MEAN']:>8.4f} {r['train_time']:>6.1f}s")

    # Store in results
    results["phase_summaries"][str(phase_num)] = {
        "best_run_key": runs[0]["run_key"],
        "best_score": runs[0]["final_scores"]["MEAN"],
        "num_runs": len(runs),
    }
    save_results(results)


# ============================================================================
# Main
# ============================================================================

def main():
    device = get_default_device()
    print(f"Device: {device}")
    print(f"Results file: {RESULTS_PATH}")

    if not os.path.exists(PRIOR_PATH):
        print(f"ERROR: Prior data not found at {PRIOR_PATH}")
        return

    # Load existing results (crash recovery)
    results = load_results()
    existing = len(results["runs"])
    if existing > 0:
        print(f"Resuming: {existing} completed runs found")

    # Load eval datasets
    print("Loading OpenML TabArena datasets...")
    eval_datasets = get_openml_datasets()
    print(f"Loaded {len(eval_datasets)} datasets: {list(eval_datasets.keys())}")

    # ==== Phase 1: Architecture Sweep ====
    print(f"\n{'#'*90}")
    print(f"# PHASE 1: Architecture Sweep (5 baseline + 7 hypatt archs x 4 LRs, 2500 steps)")
    print(f"{'#'*90}")

    p1_configs = phase1_configs()
    print(f"  Configs: {len(p1_configs)}")
    run_phase(1, p1_configs, device, eval_datasets, results)
    print_phase_summary(1, results)

    # Select top 3 architectures
    top_archs = phase1_select_top(results, n=3)
    if not top_archs:
        print("ERROR: No Phase 1 results found")
        return

    # ==== Phase 2: LR Sweep ====
    print(f"\n{'#'*90}")
    print(f"# PHASE 2: LR Sweep (top 3 archs x 6 LRs x 2 attn types)")
    print(f"{'#'*90}")

    p2_configs = phase2_configs(top_archs)
    print(f"  Configs: {len(p2_configs)}")
    run_phase(2, p2_configs, device, eval_datasets, results)
    print_phase_summary(2, results)

    # Select best config
    best_p2 = phase2_select_best(results)
    if not best_p2:
        print("ERROR: No Phase 2 results found")
        return
    print(f"\n  Best Phase 2: {best_p2['run_key']} -> {best_p2['final_scores']['MEAN']:.4f}")

    # ==== Phase 3: Ablations ====
    print(f"\n{'#'*90}")
    print(f"# PHASE 3: Ablations (best config with variants)")
    print(f"{'#'*90}")

    p3_configs = phase3_configs(best_p2)
    print(f"  Configs: {len(p3_configs)}")
    run_phase(3, p3_configs, device, eval_datasets, results)
    print_phase_summary(3, results)

    # ==== Phase 4: Extended Training ====
    print(f"\n{'#'*90}")
    print(f"# PHASE 4: Extended Training (top 2 configs at 5000 steps)")
    print(f"{'#'*90}")

    # Top 2 from Phase 2
    top_p2 = phase2_select_top(results, n=2)
    p4_configs = phase4_configs(top_p2)
    print(f"  Configs: {len(p4_configs)}")
    run_phase(4, p4_configs, device, eval_datasets, results)
    print_phase_summary(4, results)

    # ==== Final Summary ====
    print(f"\n{'#'*90}")
    print(f"# FINAL SUMMARY: All Phases")
    print(f"{'#'*90}")

    all_runs = sorted(results["runs"], key=lambda r: r["final_scores"]["MEAN"], reverse=True)
    print(f"\n{'Rank':<5s} {'Phase':<6s} {'Arch':<18s} {'Attn Type':<25s} "
          f"{'LR':>8s} {'Steps':>6s} {'Params':>8s} {'ROC AUC':>8s}")
    print(f"{'-'*90}")
    for rank, r in enumerate(all_runs[:20], 1):
        cfg = r["config"]
        arch = f"e{cfg['embedding_size']}/h{cfg['num_heads']}/l{cfg['num_layers']}"
        attn = f"{cfg['attn_norm']}+{cfg['target_network']}"
        print(f"{rank:<5d} {'P'+str(r['phase']):<6s} {arch:<18s} {attn:<25s} "
              f"{cfg['lr']:>8.1e} {cfg['num_steps']:>6d} {r['n_params']:>8,d} "
              f"{r['final_scores']['MEAN']:>8.4f}")

    # Per-dataset breakdown for overall best
    best_overall = all_runs[0]
    print(f"\nBest overall: {best_overall['run_key']}")
    print(f"  ROC AUC per dataset:")
    for name, score in best_overall["final_scores"].items():
        if name != "MEAN":
            print(f"    {name:30s}: {score:.4f}")
    print(f"    {'MEAN':30s}: {best_overall['final_scores']['MEAN']:.4f}")

    # Hypernetwork vs baseline comparison
    print(f"\nHypernetwork vs Baseline (best of each):")
    best_hyp = max([r for r in all_runs if r["config"]["target_network"] != "default"],
                   key=lambda r: r["final_scores"]["MEAN"], default=None)
    best_base = max([r for r in all_runs if r["config"]["target_network"] == "default"
                     and r["config"]["attn_norm"] == "softmax"],
                    key=lambda r: r["final_scores"]["MEAN"], default=None)
    if best_base:
        cfg = best_base["config"]
        print(f"  Baseline:    {best_base['final_scores']['MEAN']:.4f} "
              f"(e{cfg['embedding_size']}/h{cfg['num_heads']}/l{cfg['num_layers']}, "
              f"lr={cfg['lr']})")
    if best_hyp:
        cfg = best_hyp["config"]
        print(f"  Hypernetwork: {best_hyp['final_scores']['MEAN']:.4f} "
              f"(e{cfg['embedding_size']}/h{cfg['num_heads']}/l{cfg['num_layers']}, "
              f"{cfg['attn_norm']}+{cfg['target_network']}, lr={cfg['lr']})")
    if best_base and best_hyp:
        diff = best_hyp["final_scores"]["MEAN"] - best_base["final_scores"]["MEAN"]
        print(f"  Delta:       {diff:+.4f}")

    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total runs: {len(results['runs'])}")


if __name__ == "__main__":
    main()
