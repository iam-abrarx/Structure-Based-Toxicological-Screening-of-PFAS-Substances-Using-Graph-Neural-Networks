import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Configs
SPLIT_DIR = "data/processed/splits"
RESULTS_DIR = "results"
ENDPOINTS = ["SR-p53", "SR-HSE", "SR-MMP", "NR-AR", "NR-ER", "NR-PPAR-gamma"]

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_data():
    """Load train/val/test splits."""
    try:
        train = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
        val = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
        test = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))
        return train, val, test
    except FileNotFoundError:
        print("Error: Split files not found. Run split_data.py first.")
        return None, None, None

def get_features_and_targets(df):
    """Separate features and targets."""
    # Identify feature columns
    # RDKit descriptors
    desc_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds']
    # Fingerprints (fp_0 to fp_2047)
    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    # GNN embeddings (gnn_0 to gnn_...)
    gnn_cols = [c for c in df.columns if c.startswith("gnn_")]
    
    features = {
        'baseline': df[desc_cols + fp_cols].values,
        'gnn': df[gnn_cols].values if gnn_cols else None,
        'combined': df[desc_cols + fp_cols + gnn_cols].values if gnn_cols else None
    }
    
    # Impute NaNs (simple mean for descriptors)
    # Note: In a real pipeline, scaler/imputer should be fit on TRAIN only.
    # For simplicity here, we assume features are mostly clean or handle per-split.
    
    targets = {}
    for ep in ENDPOINTS:
        col_name = f"{ep}_Binary"
        if col_name in df.columns:
            targets[ep] = df[col_name].values
            
    return features, targets

def evaluate_model(model, X, y_true, name="Model"):
    """Evaluate model and return metrics."""
    # Handle NaNs in X if any
    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(X)
    
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    # Metrics
    # Handle cases with only 1 class in y_true
    if len(np.unique(y_true)) < 2:
        return {'ROC-AUC': np.nan, 'PR-AUC': np.nan, 'Acc': np.nan, 'F1': np.nan}
        
    metrics = {
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'Acc': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }
    
    return metrics, y_pred, y_prob

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("--- Tox21 Modeling Pipeline ---")
    
    train_df, val_df, test_df = load_data()
    if train_df is None: return

    # Prepare Data
    # For simplicity, we combine Train+Val for training or use Val for early stopping.
    # Let's train on Train, eval on Test.
    
    print("Preparing features...")
    X_train_dict, y_train_dict = get_features_and_targets(train_df)
    X_test_dict, y_test_dict = get_features_and_targets(test_df)
    
    # Preprocessing (Imputation/Scaling)
    # We'll do this inside the loop or use a pipeline.
    
    results = []
    
    for endpoint in ENDPOINTS:
        print(f"\nProcessing Endpoint: {endpoint}")
        
        if endpoint not in y_train_dict:
            print(f"  Skipping {endpoint} (no labels)")
            continue
            
        # Get valid masks (labels not NaN)
        # Our processing script kept NaNs for missing assays? 
        # Yes, merged dataset has NaNs. Drop them for training THIS endpoint.
        
        # Train Helper
        def get_clean_xy(X_dict, y_dict, feat_type):
            X = X_dict[feat_type]
            y = y_dict[endpoint]
            if X is None: return None, None
            
            mask = ~np.isnan(y)
            return X[mask], y[mask]

        # 1. Baseline Model (Random Forest on Descriptors+FP)
        print("  Training Baseline (Random Forest)...")
        X_tr, y_tr = get_clean_xy(X_train_dict, y_train_dict, 'baseline')
        X_te, y_te = get_clean_xy(X_test_dict, y_test_dict, 'baseline')
        
        if X_tr is not None and len(y_tr) > 50: # Minimum samples check
            # Impute
            imp = SimpleImputer(strategy='mean') # Fit on train
            X_tr = imp.fit_transform(X_tr)
            X_te = imp.transform(X_te)
            
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            rf.fit(X_tr, y_tr)
            
            metrics, y_pred, _ = evaluate_model(rf, X_te, y_te)
            metrics['Endpoint'] = endpoint
            metrics['Model'] = 'Baseline_RF'
            results.append(metrics)
            print(f"    Baseline ROC-AUC: {metrics['ROC-AUC']:.3f}")
            
            # Save Confusion Matrix
            cm_file = os.path.join(RESULTS_DIR, f"cm_{endpoint}_baseline.png")
            plot_confusion_matrix(y_te, y_pred, f"{endpoint} Baseline RF", cm_file)
        else:
            print("    Insufficient data for Baseline.")

        # 2. Advanced Model (MLP on Combined or GNN)
        # Use Combined if available, else GNN
        feat_type = 'combined' if X_train_dict['combined'] is not None else 'baseline'
        if feat_type == 'combined':
            print("  Training Advanced (MLP on GNN+Features)...")
            X_tr, y_tr = get_clean_xy(X_train_dict, y_train_dict, 'combined')
            X_te, y_te = get_clean_xy(X_test_dict, y_test_dict, 'combined')
            
            if X_tr is not None and len(y_tr) > 50:
                imp = SimpleImputer(strategy='mean')
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(imp.fit_transform(X_tr))
                X_te = scaler.transform(imp.transform(X_te))
                
                mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500, early_stopping=True, random_state=42)
                mlp.fit(X_tr, y_tr)
                
                metrics, y_pred, _ = evaluate_model(mlp, X_te, y_te)
                metrics['Endpoint'] = endpoint
                metrics['Model'] = 'Advanced_GNN_MLP'
                results.append(metrics)
                print(f"    Advanced ROC-AUC: {metrics['ROC-AUC']:.3f}")
                
                cm_file = os.path.join(RESULTS_DIR, f"cm_{endpoint}_advanced.png")
                plot_confusion_matrix(y_te, y_pred, f"{endpoint} Advanced MLP", cm_file)
            else:
                print("    Insufficient data for Advanced.")
                
    # Save all results
    if results:
        res_df = pd.DataFrame(results)
        res_file = os.path.join(RESULTS_DIR, "final_metrics.csv")
        res_df.to_csv(res_file, index=False)
        print(f"\nSaved Metrics to {res_file}")
        print(res_df)
    else:
        print("No models trained successfully.")

if __name__ == "__main__":
    main()
