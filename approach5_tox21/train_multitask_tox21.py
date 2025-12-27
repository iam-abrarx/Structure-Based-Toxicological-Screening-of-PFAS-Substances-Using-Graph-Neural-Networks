import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import os

# Configs
INPUT_FILE = "data/processed/tox21_features_ready.csv"
RESULTS_DIR = "results"
ENDPOINTS = ["SR-p53", "SR-HSE", "SR-MMP", "NR-AR", "NR-ER", "NR-PPAR-gamma"]
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
HIDDEN_DIM = 512
DROPOUT = 0.3

class Tox21Dataset(Dataset):
    def __init__(self, X, y, masks):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.masks = torch.FloatTensor(masks)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tasks):
        super(MultiTaskModel, self).__init__()
        
        # Shared Encoder
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        # Task-Specific Heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_tasks)
        ])
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        shared_rep = self.shared(x)
        outputs = []
        for head in self.heads:
            outputs.append(self.sigmoid(head(shared_rep)))
        return torch.cat(outputs, dim=1)

def main():
    print("--- Starting Multi-Task Learning (MTL) Training ---")
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # 1. Load Data
    print("Loading Data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Features
    feat_cols = [c for c in df.columns if c.startswith('fp_') or c in ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds']]
    X = df[feat_cols].values
    
    # Impute Features
    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Targets & Masks
    # Construct a Y matrix (N, T) and Mask matrix (N, T)
    Y = []
    Masks = []
    
    for ep in ENDPOINTS:
        col = f'{ep}_Binary'
        # Labels: 0/1, NaNs become 0 but Mask will cover them
        y_col = df[col].fillna(0).values 
        mask_col = (~df[col].isna()).astype(float).values
        Y.append(y_col)
        Masks.append(mask_col)
        
    Y = np.stack(Y, axis=1) # Shape (N, 6)
    Masks = np.stack(Masks, axis=1)
    
    # Split
    # We split indicies to keep alignment
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    M_train, M_test = Masks[train_idx], Masks[test_idx]
    
    # Loaders
    train_dset = Tox21Dataset(X_train, Y_train, M_train)
    test_dset = Tox21Dataset(X_test, Y_test, M_test)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = MultiTaskModel(input_dim=X.shape[1], hidden_dim=HIDDEN_DIM, num_tasks=len(ENDPOINTS))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss(reduction='none') # We apply mask manually
    
    print(f"Training on {len(X_train)} samples for {EPOCHS} epochs...")
    
    # Train Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for xb, yb, mb in train_loader:
            optimizer.zero_grad()
            preds = model(xb) # (Batch, 6)
            
            # Loss per task
            loss = criterion(preds, yb) # (Batch, 6)
            # Mask
            masked_loss = loss * mb
            # Mean over valid samples only? 
            # Or sum and divide by total valid?
            # Standard: Sum losses, potentially weight them.
            # Here: Mean over batch elements.
            
            final_loss = masked_loss.sum() / (mb.sum() + 1e-6)
            
            final_loss.backward()
            optimizer.step()
            total_loss += final_loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
            
    # Evaluation
    print("\n--- Evaluation ---")
    model.eval()
    
    # Collect all preds
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for xb, yb, mb in test_loader:
            preds = model(xb)
            all_preds.append(preds.numpy())
            all_targets.append(yb.numpy())
            all_masks.append(mb.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0) # (N_test, 6)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    metrics = []
    
    for i, ep in enumerate(ENDPOINTS):
        # Filter valid test samples for this task
        valid_mask = all_masks[:, i] == 1
        y_true = all_targets[valid_mask, i]
        y_pred = all_preds[valid_mask, i]
        
        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true, y_pred)
            
        print(f"Endpoint: {ep} | Test AUROC: {auc:.4f}")
        
        metrics.append({
            'Endpoint': ep,
            'Model': 'Multi-Task_NN',
            'AUROC': auc
        })
        
    # Save Results
    res_df = pd.DataFrame(metrics)
    out_path = os.path.join(RESULTS_DIR, "mtl_metrics.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved MTL metrics to {out_path}")

if __name__ == "__main__":
    main()
