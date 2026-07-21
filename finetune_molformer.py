import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, lr_scheduler
from torch import nn
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64
MAX_LEN = 256
EPOCHS = 40
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
SAVE_PATH = "finetuned_molformer"
LEARNING_RATE = 4e-5
WEIGHT_DECAY = 0.001
GRAD_CLIP = 1.0
PATIENCE = 5  # Early stopping patience
N_RUNS = 3    # Количество запусков с разными сидами

df = pd.read_csv("/home/dubonos/MELTING-POINT-1/main_mp_df.csv")

def compute_metrics(outputs, targets):
    mae = mean_absolute_error(targets, outputs)
    rmse = np.sqrt(mean_squared_error(targets, outputs))
    r2 = r2_score(targets, outputs)
    return {"mae": mae, "rmse": rmse, "r2": r2}

def run_single_run(seed, run_number):
    print(f"\n{'='*60}")
    print(f"RUN {run_number}/{N_RUNS} (seed={seed})")
    print(f"{'='*60}\n")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    class MPDataset(Dataset):
        def __init__(self, smiles, mp):
            self.smiles = smiles
            self.mp = mp

        def __len__(self):
            return len(self.mp)

        def __getitem__(self, idx):
            encoding = tokenizer(
                self.smiles[idx],
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "MP": torch.FloatTensor([self.mp[idx]]),
            }
    
    scaler = StandardScaler()
    scaler.fit(train_df["MP"].values.reshape(-1, 1))
    y_train = scaler.transform(train_df["MP"].values.reshape(-1, 1))
    y_val = scaler.transform(val_df["MP"].values.reshape(-1, 1))
    y_test = scaler.transform(test_df["MP"].values.reshape(-1, 1))
    
    train_dataset = MPDataset(train_df["SMILES"].values, y_train.flatten())
    val_dataset = MPDataset(val_df["SMILES"].values, y_val.flatten())
    test_dataset = MPDataset(test_df["SMILES"].values, y_test.flatten())
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
    )
    
    class MPModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.molformer = AutoModel.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )
            self.regressor = nn.Sequential(
                nn.Linear(self.molformer.config.hidden_size, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )
            # Freeze first 5 layers
            for param in self.molformer.encoder.layer[:5].parameters():
                param.requires_grad = False

        def forward(self, input_ids, attention_mask):
            outputs = self.molformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            pooled_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(1, keepdim=True)
            return self.regressor(pooled_output)
    
    model = MPModel().to(DEVICE)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 3, eta_min=LEARNING_RATE / 10
    )
    loss_fn = nn.HuberLoss()

    best_val_mae = float("inf")
    no_improve = 0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets = batch["MP"].to(DEVICE)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        model.eval()
        val_loss = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                targets = batch["MP"].to(DEVICE)
                
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, targets)
                
                val_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_metrics = compute_metrics(np.array(all_outputs), np.array(all_targets))
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f} | Val RMSE: {val_metrics['rmse']:.4f} | Val R²: {val_metrics['r2']:.4f}")
        
        # Early stopping check
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            no_improve = 0
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': best_val_mae,
            }, f"{SAVE_PATH}_run{run_number}_best.pt")
            print(f"✓ Saved best model with MAE: {best_val_mae:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n⊕ Early stopping triggered at epoch {epoch+1}")
                print(f"  Best epoch: {best_epoch} with MAE: {best_val_mae:.4f}")
                break
    
    print(f"\n{'='*40}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*40}")
    
    # Загружаем лучшую модель
    checkpoint = torch.load(f"{SAVE_PATH}_run{run_number}_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets = batch["MP"].to(DEVICE)
            
            outputs = model(input_ids, attention_mask)
            
            all_outputs.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    test_metrics = compute_metrics(np.array(all_outputs), np.array(all_targets))
    
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    
    # Очистка памяти
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    
    return test_metrics

if __name__ == "__main__":
    seeds = [42, 123, 2024]  # Три разных сида
    all_metrics = {"mae": [], "rmse": [], "r2": []}
    
    print(f"\n{'#'*60}")
    print(f"# STARTING {N_RUNS} RUNS WITH DIFFERENT SEEDS")
    print(f"{'#'*60}")
    
    for i, seed in enumerate(seeds, 1):
        test_metrics = run_single_run(seed, i)
        all_metrics["mae"].append(test_metrics["mae"])
        all_metrics["rmse"].append(test_metrics["rmse"])
        all_metrics["r2"].append(test_metrics["r2"])
    
    print(f"\n{'#'*60}")
    print(f"# FINAL RESULTS ACROSS {N_RUNS} RUNS")
    print(f"{'#'*60}\n")
    
    mae_mean = np.mean(all_metrics["mae"])
    mae_std = np.std(all_metrics["mae"])
    rmse_mean = np.mean(all_metrics["rmse"])
    rmse_std = np.std(all_metrics["rmse"])
    r2_mean = np.mean(all_metrics["r2"])
    r2_std = np.std(all_metrics["r2"])
    
    print("="*60)
    print("AGGREGATED TEST METRICS (mean ± std)")
    print("="*60)
    print(f"MAE:  {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"R²:   {r2_mean:.4f} ± {r2_std:.4f}")
    print("="*60)
    
    results_df = pd.DataFrame({
        'run': range(1, N_RUNS + 1),
        'seed': seeds,
        'MAE': all_metrics["mae"],
        'RMSE': all_metrics["rmse"],
        'R2': all_metrics["r2"]
    })
    results_df.to_csv(f"{SAVE_PATH}_results.csv", index=False)
    print(f"\nResults saved to {SAVE_PATH}_results.csv")

        ## PLEASE NOTE THAT METRICS SHOULD BE REMADE INTO ORIGINAL SCALE BY MULTIPLICATION ON STD OF TRAIN DF
