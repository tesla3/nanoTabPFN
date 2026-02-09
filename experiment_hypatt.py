"""
Hypernetwork Attention vs Standard Softmax Attention: Comprehensive Comparison
==============================================================================

This experiment compares standard softmax attention against hypernetwork variants
(from "Attention as a Hypernetwork", arxiv 2406.05816) on nanoTabPFN.

The key hypothesis: applying attention weights twice (as a hypernetwork) gives
stronger in-context learning, which should show up as better generalization
on held-out tabular classification tasks.

Experiment design:
1. Train each config on the same synthetic prior data for a fixed number of steps
2. Evaluate on multiple sklearn datasets (breast_cancer, wine, iris, digits)
3. Perform architecture search over: embedding_size, num_layers, num_heads, lr
4. Report ROC AUC, accuracy, balanced accuracy
"""

import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import openml
from openml.tasks import TaskType
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer

sys.path.insert(0, os.path.dirname(__file__))
from model import NanoTabPFNModel, NanoTabPFNClassifier
from train import PriorDumpDataLoader, get_default_device

import schedulefree
from torch import nn


# ============================================================================
# Evaluation datasets - OpenML TabArena (from experiment.ipynb)
# ============================================================================

def get_feature_preprocessor(X):
    """Fits a preprocessor that encodes categorical features and removes constant features."""
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum()
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x)),
        ("to_numeric", FunctionTransformer(lambda x: x.apply(pd.to_numeric, errors='coerce').to_numpy())),
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor


def load_eval_datasets(max_features=20, max_samples=600, target_classes_filter=2):
    """Load OpenML TabArena datasets, filtered and subsampled."""
    task_ids = [
        363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
        363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
        363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
        363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
        363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
        363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
        363708, 363711, 363712
    ]  # TabArena v0.1
    datasets = {}
    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue
        dataset = task.get_dataset(download_data=False)

        if (dataset.qualities["NumberOfFeatures"] > max_features
                or dataset.qualities["NumberOfClasses"] > target_classes_filter
                or dataset.qualities["PercentageOfInstancesWithMissingValues"] > 0
                or dataset.qualities["MinorityClassPercentage"] < 2.5):
            continue
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )
        if max_samples < len(y):
            _, X_sub, _, y_sub = train_test_split(
                X, y, test_size=max_samples, stratify=y, random_state=0,
            )
        else:
            X_sub = X
            y_sub = y

        X_np = X_sub.to_numpy(copy=True)
        y_np = y_sub.to_numpy(copy=True)
        label_encoder = LabelEncoder()
        y_np = label_encoder.fit_transform(y_np)

        preprocessor = get_feature_preprocessor(X_np)
        X_np = preprocessor.fit_transform(X_np).astype(np.float64)
        datasets[dataset.name] = (X_np, y_np)

    return datasets


_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def evaluate_model(model, device, datasets):
    """Evaluate a trained model on multiple datasets using 5-fold CV. Returns per-dataset and mean ROC AUC."""
    model.eval()
    results = {}

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
        roc = roc_auc_score(targets, probabilities, multi_class="ovr")
        results[name] = {"roc_auc": roc}
        print(f"    {name:30s}: roc_auc={roc:.4f}")

    # Compute mean across datasets
    vals = [r["roc_auc"] for r in results.values()]
    mean_roc = np.mean(vals) if vals else float("nan")
    results["MEAN"] = {"roc_auc": mean_roc}
    print(f"    {'MEAN':30s}: roc_auc={mean_roc:.4f}")

    return results


# ============================================================================
# Training
# ============================================================================

def train_model(model, prior, lr, device, num_steps, eval_datasets, eval_every=50):
    """Train with periodic evaluation. Returns model and eval history."""
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    history = []
    train_time = 0.0
    step = 0

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
            history.append({
                "step": step,
                "train_time": train_time,
                "loss": loss.item(),
                "metrics": results,
            })
            mean = results["MEAN"]
            print(f"  step {step:4d} | loss {loss.item():.4f} | "
                  f"roc_auc {mean['roc_auc']:.4f} | time {train_time:.1f}s")
            model.train()
            optimizer.train()

    # Final eval
    model.eval()
    optimizer.eval()
    final = evaluate_model(model, device, eval_datasets)
    return model, history, final


# ============================================================================
# Experiment configs - architecture search
# ============================================================================

def get_experiment_configs():
    """
    Design rationale (from papers):

    1. Standard softmax attention (baseline) - the default TabPFN setup
    2. rms_head + mlp_silu - best combo from the HypAtt paper (Table 1, 2), with SiLU replacing ReLU
       RMS head normalization removes the softmax bottleneck, allowing negative
       attention weights. The double application of attention as a hypernetwork
       gives the model more expressive power for in-context learning.
    3. rms_head + mlp_linear - ablation: hypernetwork without activation.
       Tests whether the nonlinearity between the two applications matters.
    4. softmax + mlp_silu - ablation: hypernetwork with standard normalization.
       Tests whether RMS head norm is necessary for the hypernetwork to work.

    Architecture search: we vary embedding size, num_layers, num_heads, and lr.
    The prior data has max 5 features and 150 rows, so the model can be small.
    """
    # Focused: rms_head + mlp_silu, emb=512, heads=32, LR sweep 1e-5 to 5e-4
    #learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]
    learning_rates = [5e-5]

    experiments = []
    for lr in learning_rates:
        experiments.append({
            "target_network": "mlp_silu",
            "attn_norm": "rms_head",
            "attn_label": "rms_head_mlp_silu",
            "embedding_size": 512,
            "num_layers": 3,
            "num_heads": 32,
            "mlp_hidden_size": 1024,
            "lr": lr,
            "arch_label": f"emb256_16H_lr{lr:.0e}",
            "num_outputs": 2,
        })

    return experiments


# ============================================================================
# Main
# ============================================================================

def main():
    device = get_default_device()
    print(f"Device: {device}")

    # Seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load eval datasets (OpenML TabArena, max 20 features, 600 samples)
    print("Loading OpenML TabArena datasets...")
    eval_datasets = load_eval_datasets()
    print(f"Loaded {len(eval_datasets)} datasets: {list(eval_datasets.keys())}")
    for name, (X, y) in eval_datasets.items():
        print(f"  {name}: samples={len(y)}, features={X.shape[1]}, classes={len(set(y))}")

    # Load prior data
    prior_path = os.path.join(os.path.dirname(__file__), "300k_150x5_2.h5")
    if not os.path.exists(prior_path):
        print(f"ERROR: Prior data not found at {prior_path}")
        print("Download it: curl -L -o 300k_150x5_2.h5 'https://figshare.com/ndownloader/files/58932628?private_link=63fc1ada93e42e388e63'")
        return

    # Training config
    NUM_STEPS = 2500
    BATCH_SIZE = 32
    EVAL_EVERY = 50

    experiments = get_experiment_configs()
    print(f"\nTotal experiments: {len(experiments)}")

    all_results = []

    for i, cfg in enumerate(experiments):
        label = f"{cfg['attn_label']}_{cfg['arch_label']}"
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(experiments)}] {label}")
        print(f"  attn_norm={cfg['attn_norm']}, target_network={cfg['target_network']}")
        print(f"  emb={cfg['embedding_size']}, layers={cfg['num_layers']}, "
              f"heads={cfg['num_heads']}, mlp={cfg['mlp_hidden_size']}, lr={cfg['lr']}")
        print(f"{'='*70}")

        # Reset seed for fair comparison
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
        print(f"  Parameters: {n_params:,}")

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
            "history": history,
        }
        all_results.append(result)

        # Print final per-dataset results
        for ds_name, metrics in final.items():
            print(f"  {ds_name:20s}: roc_auc={metrics['roc_auc']:.4f}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY: Final Mean ROC AUC (sorted)")
    print("="*70)
    print(f"{'Config':<45s} {'Params':>8s} {'ROCAUC':>7s}")
    print("-"*70)

    sorted_results = sorted(all_results, key=lambda r: r["final_metrics"]["MEAN"]["roc_auc"], reverse=True)
    for r in sorted_results:
        m = r["final_metrics"]["MEAN"]
        print(f"{r['label']:<45s} {r['n_params']:>8,d} {m['roc_auc']:>7.4f}")

    # Per-dataset breakdown for best config
    best = sorted_results[0]
    print(f"\nBest: {best['label']} ({best['n_params']:,} params)")
    for ds_name, metrics in best["final_metrics"].items():
        if ds_name != "MEAN":
            print(f"  {ds_name:20s}: roc_auc={metrics['roc_auc']:.4f}")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "experiment_results.json")
    # Convert to JSON-serializable
    json_results = []
    for r in all_results:
        jr = {
            "label": r["label"],
            "config": r["config"],
            "n_params": r["n_params"],
            "final_metrics": r["final_metrics"],
        }
        json_results.append(jr)

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
