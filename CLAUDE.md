# Environment
- Always activate the micromamba environment before running any commands: `micromamba activate scale`

# nanoTabPFN

## Overview
Minimal (~380 LOC) educational reimplementation of TabPFN (Prior-Function Networks) for tabular data classification. Trains a transformer on synthetic prior data, then uses in-context learning at inference: concatenates train+test data, feeds through the model, and reads off predictions. Scikit-learn compatible interface. Paper: https://arxiv.org/abs/2511.03634

## Tech Stack
- **Framework**: PyTorch
- **Optimizer**: schedulefree (AdamWScheduleFree)
- **Data**: h5py (HDF5 prior dumps), scikit-learn datasets, OpenML
- **Interface**: scikit-learn compatible (fit/predict/predict_proba)

## Project Structure
```
model.py          (196 LOC) - Architecture + sklearn interface
train.py          (186 LOC) - Training loop + HDF5 data loader
experiment.ipynb  - Reproducible experiments and baselines
```

## Architecture (model.py)

### NanoTabPFNModel
Input: `(X, y_train)` + `train_test_split_index` -> Output: `[B, N_test, num_classes]` logits

Pipeline:
1. **FeatureEncoder**: per-feature z-score normalization (train stats only) -> Linear(1, E)
2. **TargetEncoder**: pad test targets with train mean -> Linear(1, E)
3. **Concat**: features + target embeddings -> `[B, R, C+1, E]`
4. **TransformerEncoderLayer** x num_layers:
   - Self-attention across features (column-wise): `[B*R, C, E]`
   - Self-attention across datapoints (row-wise, **asymmetric**):
     - Train attends to train
     - Test attends to train only (no data leakage)
   - MLP (GELU) + residual + LayerNorm after each
5. **Decoder**: 2-layer MLP -> class logits

### Key Design: Asymmetric Attention
```python
# Train self-attends
src_left = attn(train[:split], train[:split], train[:split])
# Test cross-attends to train only
src_right = attn(test[split:], train[:split], train[:split])
src = cat([src_left, src_right]) + src  # residual
```

### NanoTabPFNClassifier (sklearn interface)
- `fit(X_train, y_train)` - stores data
- `predict_proba(X_test)` - concat, forward, softmax, truncate to num_classes
- `predict(X_test)` - argmax of predict_proba

## Default Hyperparameters
| Parameter | Value |
|-----------|-------|
| embedding_size | 96 |
| num_attention_heads | 4 |
| mlp_hidden_size | 192 |
| num_layers | 3 |
| num_outputs | 2 |
| lr | 4e-3 |
| batch_size | 32 |
| num_steps | 2500 |
| grad_clip | 1.0 |

## Data Format
**Prior dump** (HDF5, ~1GB): `300k_150x5_2.h5`
- `X`: `[N_datasets, max_rows, max_features]` float32
- `y`: `[N_datasets, max_rows]` labels
- `num_features`, `num_datapoints`: actual sizes per dataset
- `single_eval_pos`: train/test split index
- `max_num_classes`: max classes across all datasets
- Download: https://figshare.com/s/63fc1ada93e42e388e63

## Running
```bash
# Download prior dump
curl -L -H "User-Agent: Mozilla/5.0" -H "Referer: https://figshare.com/" \
  -o 300k_150x5_2.h5 \
  "https://figshare.com/ndownloader/files/58932628?private_link=63fc1ada93e42e388e63"

# Train
python train.py
```

## Dependencies
numpy, torch, schedulefree, h5py, scikit-learn, openml, seaborn (for experiments)

## Relevance to Scale Project
Reference implementation of in-context learning on tabular data via attention. The asymmetric attention pattern (test attends to train) and prior-data pre-training are conceptually related to URM's approach of learning reasoning patterns from examples.
