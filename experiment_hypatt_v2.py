"""
Hypernetwork Attention v2: Longer Training + Wider LR Sweep
============================================================

Key insight from v1: 250 steps is far too few for hypernetwork variants.
The loss plateaus at ~0.69 (near random) while softmax default already converges.

This experiment:
1. Trains for 2500 steps (10x more, matching nanoTabPFN default)
2. Wider LR sweep for hypernetwork (the double-application of attention
   changes the effective gradient magnitude, so optimal LR differs)
3. Focuses on the best architecture from v1 (base_96_3L) to save time
4. Adds SiLU variant (user request: replace ReLU with SiLU in hypernetwork)

Rationale from papers:
- HypAtt paper (2406.05816): rms_head + mlp_silu is best on all benchmarks
- The paper trained for full convergence (not 250 steps)
- RMS head normalization allows negative attention weights, which changes
  the optimization landscape significantly - may need lower LR
- SiLU (smooth gating) helps gradient flow vs ReLU (hard threshold)
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from model import NanoTabPFNModel, NanoTabPFNClassifier
from train import PriorDumpDataLoader, get_default_device

import schedulefree
from torch import nn


def load_eval_datasets(max_features=10, max_samples=200, seed=42):
    datasets = []
    for loader, name in [
        (load_breast_cancer, "breast_cancer"),
        (load_wine, "wine"),
        (load_iris, "iris"),
    ]:
        X, y = loader(return_X_y=True)
        if X.shape[1] > max_features:
            rng = np.random.RandomState(seed)
            feat_idx = rng.choice(X.shape[1], max_features, replace=False)
            X = X[:, feat_idx]
        if X.shape[0] > max_samples:
            rng = np.random.RandomState(seed)
            row_idx = rng.choice(X.shape[0], max_samples, replace=False)
            X, y = X[row_idx], y[row_idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=seed, stratify=y
        )
        datasets.append((name, X_train, X_test, y_train, y_test))

    X, y = load_digits(return_X_y=True)
    rng = np.random.RandomState(seed)
    row_idx = rng.choice(X.shape[0], max_samples, replace=False)
    X, y = X[row_idx], y[row_idx]
    feat_idx = rng.choice(X.shape[1], min(max_features, X.shape[1]), replace=False)
    X = X[:, feat_idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )
    datasets.append(("digits", X_train, X_test, y_train, y_test))
    return datasets


def evaluate_model(model, device, datasets):
    model.eval()
    results = {}
    for name, X_train, X_test, y_train, y_test in datasets:
        clf = NanoTabPFNClassifier(model, device)
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)
        pred = prob.argmax(axis=1)
        acc = accuracy_score(y_test, pred)
        bal_acc = balanced_accuracy_score(y_test, pred)
        n_classes = len(set(y_test))
        if n_classes == 2:
            roc = roc_auc_score(y_test, prob[:, 1] if prob.shape[1] >= 2 else prob[:, 0])
        else:
            try:
                roc = roc_auc_score(y_test, prob[:, :n_classes], multi_class="ovr")
            except ValueError:
                roc = float("nan")
        results[name] = {"acc": acc, "bal_acc": bal_acc, "roc_auc": roc}

    mean_metrics = {}
    for metric in ["acc", "bal_acc", "roc_auc"]:
        vals = [r[metric] for r in results.values() if not np.isnan(r[metric])]
        mean_metrics[metric] = np.mean(vals) if vals else float("nan")
    results["MEAN"] = mean_metrics
    return results


def train_model(model, prior, lr, device, num_steps, eval_datasets, eval_every=250):
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.train()

    history = []
    train_time = 0.0
    step = 0
    best_bal_acc = 0.0

    for full_data in prior:
        if step >= num_steps:
            break
        t0 = time.time()
        train_test_split_index = full_data["train_test_split_index"]
        data = (full_data["x"].to(device),
                full_data["y"][:, :train_test_split_index].to(device))
        targets = full_data["y"].to(device)
        output = model(data, train_test_split_index=train_test_split_index)
        targets = targets[:, train_test_split_index:]
        targets = targets.reshape((-1,)).to(torch.long)
        output = output.view(-1, output.shape[-1])
        loss = criterion(output, targets).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        train_time += time.time() - t0
        step += 1

        if step % eval_every == 0 or step == num_steps:
            model.eval()
            optimizer.eval()
            results = evaluate_model(model, device, eval_datasets)
            mean = results["MEAN"]
            best_bal_acc = max(best_bal_acc, mean["bal_acc"])
            history.append({
                "step": step, "train_time": train_time,
                "loss": loss.item(), "metrics": results,
            })
            print(f"  step {step:5d} | loss {loss.item():.4f} | "
                  f"acc {mean['acc']:.4f} | bal_acc {mean['bal_acc']:.4f} | "
                  f"roc_auc {mean['roc_auc']:.4f} | time {train_time:.1f}s")
            model.train()
            optimizer.train()

    model.eval()
    optimizer.eval()
    final = evaluate_model(model, device, eval_datasets)
    return model, history, final


def main():
    device = get_default_device()
    print(f"Device: {device}")

    seed = 42
    eval_datasets = load_eval_datasets()
    print(f"Evaluation datasets: {[d[0] for d in eval_datasets]}")

    prior_path = os.path.join(os.path.dirname(__file__), "300k_150x5_2.h5")
    if not os.path.exists(prior_path):
        print(f"ERROR: Prior data not found at {prior_path}")
        return

    NUM_STEPS = 2500
    BATCH_SIZE = 32
    EVAL_EVERY = 250

    # Focused experiment: base architecture, sweep attention type x LR
    # Include SiLU variant (user request)
    attention_configs = [
        {"target_network": "default",    "attn_norm": "softmax",  "label": "softmax_default"},
        {"target_network": "mlp_silu",   "attn_norm": "rms_head", "label": "rms_head_mlp_silu"},
        {"target_network": "mlp_silu",   "attn_norm": "softmax",  "label": "softmax_mlp_silu"},
        {"target_network": "mlp_linear", "attn_norm": "rms_head", "label": "rms_head_mlp_linear"},
    ]

    # LR sweep - wider range for hypernetwork variants
    learning_rates = [5e-4, 1e-3, 2e-3, 4e-3, 8e-3]

    experiments = []
    for attn_cfg in attention_configs:
        for lr in learning_rates:
            experiments.append({
                "target_network": attn_cfg["target_network"],
                "attn_norm": attn_cfg["attn_norm"],
                "attn_label": attn_cfg["label"],
                "embedding_size": 96,
                "num_layers": 3,
                "num_heads": 4,
                "mlp_hidden_size": 192,
                "lr": lr,
                "num_outputs": 10,
            })

    print(f"\nTotal experiments: {len(experiments)} (4 attention types x 5 LRs)")
    print(f"Steps per run: {NUM_STEPS}")

    all_results = []

    for i, cfg in enumerate(experiments):
        label = f"{cfg['attn_label']}_lr{cfg['lr']}"
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(experiments)}] {label}")
        print(f"  attn_norm={cfg['attn_norm']}, target_network={cfg['target_network']}, lr={cfg['lr']}")
        print(f"{'='*70}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = NanoTabPFNModel(
            embedding_size=cfg["embedding_size"],
            num_attention_heads=cfg["num_heads"],
            mlp_hidden_size=cfg["mlp_hidden_size"],
            num_layers=cfg["num_layers"],
            num_outputs=cfg["num_outputs"],
            target_network=cfg["target_network"],
            attn_norm=cfg["attn_norm"],
        )
        n_params = sum(p.numel() for p in model.parameters())

        prior = PriorDumpDataLoader(
            prior_path, num_steps=NUM_STEPS, batch_size=BATCH_SIZE, device=device
        )

        model, history, final = train_model(
            model, prior, lr=cfg["lr"], device=device,
            num_steps=NUM_STEPS, eval_datasets=eval_datasets, eval_every=EVAL_EVERY
        )

        result = {
            "label": label,
            "config": cfg,
            "n_params": n_params,
            "final_metrics": final,
        }
        all_results.append(result)

        for ds_name, metrics in final.items():
            print(f"  {ds_name:16s}: acc={metrics['acc']:.4f}  "
                  f"bal_acc={metrics['bal_acc']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

    # Summary
    print("\n" + "="*90)
    print("SUMMARY: All Configs (sorted by mean balanced accuracy)")
    print("="*90)
    print(f"{'Config':<45s} {'LR':>8s} {'Acc':>7s} {'BalAcc':>7s} {'ROCAUC':>7s}")
    print("-"*90)

    sorted_results = sorted(all_results, key=lambda r: r["final_metrics"]["MEAN"]["bal_acc"], reverse=True)
    for r in sorted_results:
        m = r["final_metrics"]["MEAN"]
        print(f"{r['label']:<45s} {r['config']['lr']:>8.4f} {m['acc']:>7.4f} {m['bal_acc']:>7.4f} {m['roc_auc']:>7.4f}")

    # Best per attention type
    print("\n" + "="*90)
    print("SUMMARY: Best LR per attention type")
    print("="*90)
    attn_groups = {}
    for r in all_results:
        key = r["config"]["attn_label"]
        if key not in attn_groups:
            attn_groups[key] = []
        attn_groups[key].append(r)

    for attn_type, group in attn_groups.items():
        best = max(group, key=lambda r: r["final_metrics"]["MEAN"]["bal_acc"])
        m = best["final_metrics"]["MEAN"]
        print(f"\n  {attn_type}:")
        print(f"    Best LR: {best['config']['lr']}")
        print(f"    Acc={m['acc']:.4f}  BalAcc={m['bal_acc']:.4f}  ROCAUC={m['roc_auc']:.4f}")
        for ds_name, metrics in best["final_metrics"].items():
            if ds_name != "MEAN":
                print(f"      {ds_name:16s}: acc={metrics['acc']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

    # Save
    results_path = os.path.join(os.path.dirname(__file__), "experiment_results_v2.json")
    json_results = [{"label": r["label"], "config": r["config"],
                     "n_params": r["n_params"], "final_metrics": r["final_metrics"]}
                    for r in all_results]
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
