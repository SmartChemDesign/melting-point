import math
import random
import pandas as pd
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
import scipy.special
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import time
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
# import shap # Unused in this snippet, kept if needed later

# Load dataset
joined_df = pd.read_csv('/home/dubonos/MELTING-POINT-1/emb_morgan_df.csv')

# >>> Dataset configuration.
TaskType = Literal['regression', 'binclass', 'multiclass']

# Regression task
task_type: TaskType = 'regression'
n_classes = None

# Prepare raw data arrays
X_num_raw = joined_df.drop(columns=['MP']).to_numpy().astype(np.float32)
Y_raw = joined_df['MP'].to_numpy().astype(np.float32)

n_num_features = X_num_raw.shape[1]
task_is_regression = task_type == 'regression'

# Define seeds for multiple runs
seeds = [42, 123, 777]
results_list = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# AMP settings
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
# Changing False to True can speed up training
# of large enough models on compatible hardware.
amp_enabled = False and amp_dtype is not None

# torch.compile setting
compile_model = False

print(f'Device:        {device.type.upper()}')
print(f'AMP:           {amp_enabled}{f" ({amp_dtype})" if amp_enabled else ""}')
print(f'torch.compile: {compile_model}')

for seed in seeds:
    # Start timer for this run
    start_time_run = time.time()
    
    print(f"\n{'='*20} Starting Run with Seed {seed} {'='*20}")
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # >>> Split the dataset (Redone for each seed)
    all_idx = np.arange(len(Y_raw))
    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8, random_state=seed
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=0.8, random_state=seed
    )
    
    data_numpy = {
        'train': {'x_num': X_num_raw[train_idx], 'y': Y_raw[train_idx]},
        'val': {'x_num': X_num_raw[val_idx], 'y': Y_raw[val_idx]},
        'test': {'x_num': X_num_raw[test_idx], 'y': Y_raw[test_idx]},
    }

    # >>> 1. Ensure all data is numpy arrays
    for part in data_numpy:
        for key in data_numpy[part]:
            value = data_numpy[part][key]
            if hasattr(value, 'to_numpy'):  # pandas Series or DataFrame
                data_numpy[part][key] = value.to_numpy()
            elif not isinstance(value, np.ndarray):
                data_numpy[part][key] = np.array(value)

    # >>> 2. QuantileTransformer for X_num
    x_num_train_numpy = data_numpy['train']['x_num'].copy()
    noise = (
        np.random.default_rng(seed)
        .normal(0.0, 1e-5, x_num_train_numpy.shape)
        .astype(x_num_train_numpy.dtype)
    )
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
    ).fit(x_num_train_numpy + noise)
    del x_num_train_numpy

    for part in data_numpy:
        data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])

    # >>> 3. Label preprocessing with update of all splits
    class RegressionLabelStats(NamedTuple):
        mean: float
        std: float

    # Convert train y to numpy
    Y_train_np = np.asarray(data_numpy['train']['y'], dtype=np.float32)

    if task_type == 'regression':
        regression_label_stats = RegressionLabelStats(
            float(Y_train_np.mean()), 
            float(Y_train_np.std())
        )
        
        # Normalize ALL splits
        for part in data_numpy:
            y_part = np.asarray(data_numpy[part]['y'], dtype=np.float32)
            data_numpy[part]['y'] = ((y_part - regression_label_stats.mean) / regression_label_stats.std).astype(np.float32)
    else:
        regression_label_stats = None
        for part in data_numpy:
            data_numpy[part]['y'] = np.asarray(data_numpy[part]['y'], dtype=np.int64)

    # >>> 4. Final check
    for part, part_data in data_numpy.items():
        for key, value in part_data.items():
            assert isinstance(value, np.ndarray), f"{part}/{key} is not numpy array! Got {type(value)}"

    # >>> 5. Conversion to torch tensors
    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }

    Y_train_tensor = data['train']['y']

    if task_type == 'regression':
        for part in data:
            data[part]['y'] = data[part]['y'].float()
        Y_train_tensor = Y_train_tensor.float()

    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

    # Periodic embeddings.
    num_embeddings = rtdl_num_embeddings.PeriodicEmbeddings(n_num_features, lite=False)

    # Initialize Model
    model = tabm.TabM.make(
        n_num_features=n_num_features,
        d_out=1 if n_classes is None else n_classes,
        num_embeddings=num_embeddings,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    gradient_clipping_norm: Optional[float] = 1.0

    if compile_model:
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    share_training_batches = True

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[part]['x_num'][idx],
                data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
            )
            .squeeze(-1)  # Remove the last dimension for regression tasks.
            .float()
        )

    base_loss_fn = (
        nn.functional.mse_loss if task_is_regression else nn.functional.cross_entropy
    )

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TabM produces k predictions. Each of them must be trained separately.
        y_pred = y_pred.flatten(0, 1)

        if share_training_batches:
            # (batch_size,) -> (batch_size * k,)
            y_true = y_true.repeat_interleave(model.backbone.k)
        else:
            # (batch_size, k) -> (batch_size * k,)
            y_true = y_true.flatten(0, 1)

        return base_loss_fn(y_pred, y_true)

    @evaluation_mode()
    def evaluate(part: str) -> float:
        model.eval()

        eval_batch_size = 256
        y_pred: np.ndarray = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        
        # Invert predictions back to original scale
        if task_type == 'regression':
            assert regression_label_stats is not None
            y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean

        # Compute the mean of the k predictions.
        if task_type != 'regression':
            y_pred = scipy.special.softmax(y_pred, axis=-1)
        y_pred = y_pred.mean(1)

        y_true = data[part]['y'].cpu().numpy()
        if task_type == 'regression':
            assert regression_label_stats is not None
            y_true = y_true * regression_label_stats.std + regression_label_stats.mean

        # Note: This function returns negative RMSE for optimization purposes (higher is better)
        score = (
            -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
            if task_type == 'regression'
            else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
        )
        
        return float(score)

    # Training Loop
    n_epochs = 1_000_000_000
    train_size = len(train_idx)
    batch_size = 256
    
    epoch = -1
    metrics = {'val': -math.inf, 'test': -math.inf}

    def make_checkpoint() -> dict[str, Any]:
        return deepcopy(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
            }
        )

    best_checkpoint = make_checkpoint()

    # Early stopping: the training stops if the validation score
    # does not improve for more than `patience` consecutive epochs.
    patience = 16
    remaining_patience = patience

    for epoch in range(n_epochs):
        batches = (
            torch.randperm(train_size, device=device).split(batch_size)
            if share_training_batches
            else (
                torch.rand((train_size, model.backbone.k), device=device)
                .argsort(dim=0)
                .split(batch_size, dim=0)
            )
        )
        for batch_idx in batches:
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train_tensor[batch_idx])

            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()

            if gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )

            if grad_scaler is None:
                optimizer.step()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()

        # Evaluate metrics
        metrics = {part: evaluate(part) for part in ['val', 'test']}
        val_score_improved = metrics['val'] > best_checkpoint['metrics']['val']

        # Print epoch info (using absolute value for RMSE display)
        if epoch % 10 == 0: # Print less frequently to keep console clean
            print(
                f'{"*" if val_score_improved else " "}'
                f' [epoch] {epoch:<3}'
                f' [val] {-metrics["val"]:.4f}'  # convert back to positive RMSE
                f' [test] {-metrics["test"]:.4f}'
            )

        if val_score_improved:
            best_checkpoint = make_checkpoint()
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience < 0:
            break
            
    # Load best model for final evaluation
    model.load_state_dict(best_checkpoint['model'])
    
    # >>> Calculate Final Metrics for CSV (RMSE, MAE, R^2)
    model.eval()
    eval_batch_size = 256
    
    with torch.no_grad():
        y_pred_test: np.ndarray = (
            torch.cat(
                [
                    apply_model('test', idx)
                    for idx in torch.arange(len(data['test']['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )

    # Invert predictions
    if task_type == 'regression':
        y_pred_test = y_pred_test * regression_label_stats.std + regression_label_stats.mean
        y_true_test = data['test']['y'].cpu().numpy() * regression_label_stats.std + regression_label_stats.mean
    else:
        y_pred_test = scipy.special.softmax(y_pred_test, axis=-1).mean(1)
        y_true_test = data['test']['y'].cpu().numpy()

    y_pred_test = y_pred_test.mean(1)
    assert y_pred_test.shape == y_true_test.shape, "f{y_pred_test.shape}"
    # Calculate metrics
    final_rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true_test, y_pred_test))
    final_mae = sklearn.metrics.mean_absolute_error(y_true_test, y_pred_test)
    final_r2 = sklearn.metrics.r2_score(y_true_test, y_pred_test)
    
    # End timer for this run
    end_time_run = time.time()
    duration_seconds = end_time_run - start_time_run
    duration_minutes = duration_seconds / 60
    
    # The variable 'epoch' holds the last index processed. 
    # Since range(n_epochs) starts at 0, the number of epochs trained is epoch + 1.
    epochs_trained = epoch + 1
    
    print(f'\n[Seed {seed} Summary]')
    print(f'Epochs trained: {epochs_trained}')
    print(f'Duration: {duration_seconds:.2f} seconds')
    print(f'Final Test RMSE: {final_rmse:.4f}')
    print(f'Final Test MAE:  {final_mae:.4f}')
    print(f'Final Test R^2:  {final_r2:.4f}')
    
    results_list.append({
        'seed': seed,
        'epochs_trained': epochs_trained,
        'duration_seconds': duration_seconds,
        'duration_minutes': duration_minutes,
        'rmse': final_rmse,
        'mae': final_mae,
        'r2': final_r2
    })

# Save results to CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv('results_tabm_seeds_morgan.csv', index=False)
print("\nResults saved to results_tabm_seeds_morgan.csv")
print(results_df)