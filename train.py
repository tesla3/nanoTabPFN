import random
import time

import h5py
import numpy as np
import pandas as pd
import schedulefree
import torch
import openml
from openml.tasks import TaskType
from model import NanoTabPFNClassifier, NanoTabPFNModel
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer
from torch import nn
from torch.utils.data import DataLoader


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_randomness_seed(0)

def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    if torch.cuda.is_available(): device = "cuda"
    return device


def _get_feature_preprocessor(X):
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


def get_openml_datasets(max_features=20, new_instances=600, target_classes_filter=2):
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
        if new_instances < len(y):
            _, X_sub, _, y_sub = train_test_split(
                X, y, test_size=new_instances, stratify=y, random_state=0,
            )
        else:
            X_sub = X
            y_sub = y

        X_np = X_sub.to_numpy(copy=True)
        y_np = y_sub.to_numpy(copy=True)
        label_encoder = LabelEncoder()
        y_np = label_encoder.fit_transform(y_np)

        preprocessor = _get_feature_preprocessor(X_np)
        X_np = preprocessor.fit_transform(X_np).astype(np.float64)
        datasets[dataset.name] = (X_np, y_np)

    return datasets


_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def eval_model(classifier, datasets=None):
    """Evaluate classifier on datasets using 5-fold CV ROC AUC.
    If datasets is None, loads OpenML TabArena datasets."""
    if datasets is None:
        datasets = get_openml_datasets()
    scores = {}
    for name, (X, y) in datasets.items():
        targets = []
        probabilities = []
        for train_idx, test_idx in _skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            targets.append(y_test)
            classifier.fit(X_train, y_train)
            prob = classifier.predict_proba(X_test)
            if prob.shape[1] == 2:
                prob = prob[:, 1]
            probabilities.append(prob)
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0)
        scores[f"{name}/ROC AUC"] = float(roc_auc_score(targets, probabilities, multi_class="ovr"))

    avg = np.mean(list(scores.values()))
    scores["ROC AUC"] = float(avg)
    return scores

def train(model: NanoTabPFNModel, prior: DataLoader,
          lr: float = 1e-4, device: torch.device = None, steps_per_eval=10, eval_func=None):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        lr: (float) learning rate
        device: (torch.device) the device we are using
        steps_per_eval: (int) how many steps we wait before running evaluation again
        eval_func: a function that takes in a classifier and returns a dict containing the average scores
                   for some metrics and datasets

    Returns:
        (model) our trained numpy model
        (list) a list containing our eval history, each entry is the real time used for training so far together
               with a dict mapping metric names to their average values accross a list of datasets
    """
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    train_time = 0
    eval_history=[]
    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]
            #if (torch.isnan(data[0]).any() or torch.isnan(data[1]).any()):
            #    continue
            data = (full_data["x"].to(device),
                    full_data["y"][:, :train_test_split_index].to(device))
            targets = full_data["y"].to(device)

            output = model(data, train_test_split_index=train_test_split_index)
            targets = targets[:, train_test_split_index:]

            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, targets).mean()
            loss.backward()
            total_loss = loss.cpu().detach().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            # evaluate
            if step % steps_per_eval == steps_per_eval-1 and eval_func is not None:
                model.eval()
                optimizer.eval()

                classifier = NanoTabPFNClassifier(model, device)
                scores = eval_func(classifier)
                eval_history.append((train_time, scores))
                score_str = " | ".join([f"{k} {v:7.4f}" for k,v in scores.items()])
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")

                model.train()
                optimizer.train()
            elif step % steps_per_eval == steps_per_eval-1 and eval_func is None:
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f}")
    except KeyboardInterrupt:
        pass

    return model, eval_history


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename, num_steps, batch_size, device=None):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.pointer = 0
        if device is None:
            device = get_default_device()
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer : end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq_in_batch])
                train_test_split_index = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print("""Finished iteration over all stored datasets! """)
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    train_test_split_index=train_test_split_index[0].item(),
                )

    def __len__(self):
        return self.num_steps

if __name__ == "__main__":
    import argparse
    import functools
    import json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--steps_per_eval", type=int, default=50)
    args = parser.parse_args()

    device = get_default_device()
    print("Loading OpenML TabArena datasets...")
    datasets = get_openml_datasets()
    print(f"Loaded {len(datasets)} datasets")
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    eval_func = functools.partial(eval_model, datasets=datasets)
    prior = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=args.num_steps, batch_size=args.batch_size, device=device)
    model, history = train(model, prior, lr=args.lr, steps_per_eval=args.steps_per_eval, eval_func=eval_func)

    print("Final evaluation:")
    final_scores = eval_model(NanoTabPFNClassifier(model, device), datasets=datasets)
    print(final_scores)

    # Save results
    results = {
        "config": {
            "embedding_size": 96,
            "num_attention_heads": 4,
            "mlp_hidden_size": 192,
            "num_layers": 3,
            "num_outputs": 2,
            "lr": args.lr,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
        },
        "n_params": sum(p.numel() for p in model.parameters()),
        "final_metrics": final_scores,
        "history": [{"train_time": t, "scores": s} for t, s in history],
    }
    results_path = os.path.join(os.path.dirname(__file__) or ".", "train_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
