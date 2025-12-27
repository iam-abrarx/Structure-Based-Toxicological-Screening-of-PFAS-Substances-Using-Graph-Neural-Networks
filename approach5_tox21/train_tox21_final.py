import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from collections import defaultdict
from tqdm import tqdm

# Configs
INPUT_FILE = "data/processed/tox21_features_ready.csv"
RESULTS_DIR = "results"
ENDPOINTS = ["SR-p53", "SR-HSE", "SR-MMP", "NR-AR", "NR-ER", "NR-PPAR-gamma"]
N_SEEDS = 5
SEEDS = [42, 123, 777, 2024, 999]

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def generate_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        return None

def get_scaffold_splits(df, seed):
    """Perform Scaffold Split (80/20 train/test)"""
    # Note: We combine Val+Test from previous logic into one Test set for simplicty of this comparison
    # or just do a clean 80/20 split.
    
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        s = generate_scaffold(row['Canonical_SMILES'])
        if s: scaffolds[s].append(idx)
        else: scaffolds[f"null_{idx}"].append(idx)
        
    scaffold_sets = sorted(list(scaffolds.values()), key=lambda x: len(x), reverse=True)
    
    train_inds, test_inds = [], []
    train_cutoff = 0.8 * len(df)
    
    # Simple consistent shuffle of scaffolds based on seed could be added, 
    # but standard scaffold split is often deterministic by size. 
    # To get "variance" in scaffold split is hard unless we shuffle the scaffold order.
    # Let's shuffle scaffold_sets for the 'seed' aspect if requested, 
    # BUT 'classic' scaffold split is deterministic.
    # User asked for 5-10 RF ensemble runs. Usually this means fixed split, random RF seeds.
    # If we want to test split stability, we'd shuffle scaffolds. 
    # Let's stick to Fixed Scaffold Split + Random RF Seeds for "Uncertainty Estimation".
    
    # However, for Random Split, we definitely change the split seeds.
    
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) <= train_cutoff:
            train_inds.extend(scaffold_set)
        else:
            test_inds.extend(scaffold_set)
            
    return df.loc[train_inds], df.loc[test_inds]

def get_features_and_target(df, endpoint):
    # Features: Descriptors + Fingerprints + GNN (if available)
    # Filter valid labels
    mask = ~df[f'{endpoint}_Binary'].isna()
    df_valid = df[mask].copy()
    
    # Feature Cols
    # fp_0...fp_2047, MW, LogP, etc.
    # gnn_0...
    feat_cols = [c for c in df.columns if c.startswith('fp_') or c.startswith('gnn_') or c in ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds']]
    
    X = df_valid[feat_cols].values
    y = df_valid[f'{endpoint}_Binary'].values
    
    # Impute
    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(X)
    
    return X, y, df_valid

def train_and_evaluate(X_train, y_train, X_test, y_test, seed):
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predict
    if len(np.unique(y_test)) < 2: return np.nan
    y_prob = rf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)

def main():
    print("--- Starting Rigorous Evaluation ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded Data: {len(df)} compounds")
    
    results = []
    
    for endpoint in tqdm(ENDPOINTS, desc="Endpoints"):
        # Prepare Data once per endpoint (subsetting to valid rows)
        # Note: Different endpoints have different valid rows (NaNs). 
        # So splitting must happen PER ENDPOINT or we take global intersection.
        # Per endpoint is standard.
        
        mask = ~df[f'{endpoint}_Binary'].isna()
        df_ep = df[mask].reset_index(drop=True)
        
        if len(df_ep) < 100: continue
        
        # 1. Random Split Comparison (5 runs)
        # Here we vary the SPLIT SEED to see data variance, or Model Seed?
        # "5-10 RF ensemble runs" usually implies fixing the split and varying model seed 
        # OR varying the split.
        # Let's vary the SPLIT seed for Random Split to get a better sense of generalization.
        # For Scaffold, the split is deterministic (usually), so we vary Model seed.
        
        # Actually, to make them comparable:
        # We should probably report "Mean Performance across 5 RF seeds on a fixed Split" 
        # OR "Mean Performance across 5 Splits".
        # Let's do: 5 Runs.
        # Random: 5 different random splits. Train RF(seed=42).
        # Scaffold: 1 deterministic split. Train 5 RFs(seeds=...).
        
        # RANDOM SPLIT LOOPS
        aucs_random = []
        for seed in SEEDS:
            train_df, test_df = train_test_split(df_ep, test_size=0.2, random_state=seed, stratify=df_ep[f'{endpoint}_Binary'])
            X_tr, y_tr, _ = get_features_and_target(train_df, endpoint)
            X_te, y_te, _ = get_features_and_target(test_df, endpoint)
            
            score = train_and_evaluate(X_tr, y_tr, X_te, y_te, 42)
            aucs_random.append(score)
            
        # SCAFFOLD SPLIT LOOPS (Fixed split, varying RF seed)
        train_df_sc, test_df_sc = get_scaffold_splits(df_ep, 42) # Seed doesn't affect deterministic split
        X_tr_sc, y_tr_sc, _ = get_features_and_target(train_df_sc, endpoint)
        X_te_sc, y_te_sc, _ = get_features_and_target(test_df_sc, endpoint)
        
        aucs_scaffold = []
        for seed in SEEDS:
            score = train_and_evaluate(X_tr_sc, y_tr_sc, X_te_sc, y_te_sc, seed)
            aucs_scaffold.append(score)
            
        # stats
        results.append({
            'Endpoint': endpoint,
            'Split': 'Random',
            'Mean_AUROC': np.mean(aucs_random),
            'Std_AUROC': np.std(aucs_random),
            'CI_Lower': np.mean(aucs_random) - 1.96 * np.std(aucs_random),
            'CI_Upper': np.mean(aucs_random) + 1.96 * np.std(aucs_random)
        })
        
        results.append({
            'Endpoint': endpoint,
            'Split': 'Scaffold',
            'Mean_AUROC': np.mean(aucs_scaffold),
            'Std_AUROC': np.std(aucs_scaffold),
            'CI_Lower': np.mean(aucs_scaffold) - 1.96 * np.std(aucs_scaffold),
            'CI_Upper': np.mean(aucs_scaffold) + 1.96 * np.std(aucs_scaffold)
        })
        
    # Save
    res_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    res_df.to_csv(out_path, index=False)
    
    print("\n--- Summary Table ---")
    print(res_df)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
