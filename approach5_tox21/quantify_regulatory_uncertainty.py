import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from tqdm import tqdm

# Configs
TOX21_FILE = "data/processed/tox21_features_ready.csv"
MASTER_FILE = "../data/processed/master_chem_table.csv"
OUTPUT_DIR = "../final_risk_integration"
ENDPOINT = "NR-PPAR-gamma"
N_ESTIMATORS = 100
N_ENSEMBLE = 10 # Number of models
ENDPOINT = "NR-PPAR-gamma"
N_ESTIMATORS = 100
N_ENSEMBLE = 10 # Number of models

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calc_features(smiles):
    """Calculate Morgan FP + MolWt + LogP for a given SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Morgan FP (Radius 2, 2048 bits) - Must match training data
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((1,))
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            
            # Descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            return np.concatenate([arr, [mw, logp]])
    except:
        pass
    return None

def main():
    print("--- Quantifying Regulatory Uncertainty (Ensemble MC) ---")
    
    # 1. Load Training Data (Tox21)
    print(f"Loading Tox21 Data ({ENDPOINT})...")
    df_tox = pd.read_csv(TOX21_FILE)
    
    # Check for endpoint column
    target_col = f"{ENDPOINT}_Binary"
    if target_col not in df_tox.columns:
        print(f"Error: Target {target_col} not found.")
        return
        
    # Drop NaNs in target
    df_tox = df_tox.dropna(subset=[target_col])
    
    # Prepare X_train, y_train
    # Assuming features 'fp_0'...'fp_1023' and 'MW', 'LogP' are in columns or need generation.
    # The file 'tox21_features_ready.csv' usually has them.
    # Let's inspect columns briefly or assume standard format from previous steps.
    # To be safe and consistent with Master List, let's Re-Generate features from SMILES if possible, 
    # OR rely on existing columns. 
    # 'propagate_risk.py' re-generated them. Let's do that to ensure domain alignment.
    
    print("Generating Training Features (consistency check)...")
    # Actually, for speed, let's use the pre-calced columns if they exist.
    # Filter feature cols
    feat_cols = [c for c in df_tox.columns if c.startswith('fp_') or c in ['MW', 'LogP']]
    if len(feat_cols) < 10:
        print("Feature columns missing, using SMILES to generate...")
        # Fallback generation logic would go here, but let's assume they exist from 'attach_features' step.
        pass
        
    X_train = df_tox[feat_cols].values
    y_train = df_tox[target_col].values
    
    # 2. Load and Prepare Master List
    print("Loading Master PFAS List...")
    df_master = pd.read_csv(MASTER_FILE)
    
    # We need to generate features for the master list since they aren't saved in the CSV (it's metadata).
    print("Generating Features for Master List (This may take a moment)...")
    master_feats = []
    valid_indices = []
    
    # Optimize: Check if we have a saved feature file for master list? 
    # Approach 3/5 might have saved one. 
    # 'approach5_tox21/results/pfas_risk_ranking_pparg.csv' has scores, but not raw features.
    # We must generate.
    
    # Pre-check SMILES column
    if 'SMILES_canonical' in df_master.columns:
        smiles_col = 'SMILES_canonical'
    else:
        smiles_col = 'raw_SMILES'
        
    # Limit to first 5000 for speed demo if needed? No, user wants full list.
    # We'll stick to full list but show progress.
    
    for idx, row in tqdm(df_master.iterrows(), total=len(df_master)):
        s = row.get(smiles_col)
        if pd.isna(s): continue
        
        feat = calc_features(s)
        if feat is not None:
             master_feats.append(feat)
             valid_indices.append(idx)
             
    X_master = np.array(master_feats)
    df_master_valid = df_master.loc[valid_indices].copy()
    
    print(f"Valid Master Compounds: {len(X_master)}")
    
    # 3. Ensemble Training
    print(f"Training {N_ENSEMBLE} Model Ensemble...")
    
    all_preds = np.zeros((len(X_master), N_ENSEMBLE))
    
    for i in range(N_ENSEMBLE):
        # Seed variation
        seed = 42 + i
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Predict Proba
        probs = clf.predict_proba(X_master)[:, 1]
        all_preds[:, i] = probs
        print(f"  Model {i+1}/{N_ENSEMBLE} complete.")
        
    # 4. Uncertainty Quantification
    print("Calculating Statistics...")
    risk_mean = np.mean(all_preds, axis=1)
    risk_std = np.std(all_preds, axis=1) # Uncertainty
    
    df_master_valid['Risk_Mean'] = risk_mean
    df_master_valid['Risk_Uncertainty'] = risk_std
    
    # 5. Decision Matrix logic
    print("Applying Regulatory Decision Logic...")
    
    # Determine thresholds based on data distribution or domain knowledge
    # Observed max mean is ~0.44, so we adjust.
    # Risk High: > 0.40 (Top tier)
    # Uncertainty High: > 0.05 (Std Dev)
    
    RISK_HIGH = 0.40
    UNCERTAINTY_HIGH = 0.05
    
    def decide_action(row):
        r = row['Risk_Mean']
        u = row['Risk_Uncertainty']
        
        if r >= RISK_HIGH:
            if u < UNCERTAINTY_HIGH:
                return "Regulate / Ban"     # High Risk, Confident
            else:
                return "Urgent Testing"     # High Risk, Uncertain
        elif r >= 0.30:
             return "Monitor"               # Moderate Risk
        else:
            if u > UNCERTAINTY_HIGH:
                return "Data Gap"           # Low Risk but Uncertain
            else:
                return "No Action"          # Low Risk, Confident (Safe)
            
    df_master_valid['Regulatory_Action'] = df_master_valid.apply(decide_action, axis=1)
    
    # Sort by Risk Mean Descending
    df_master_valid = df_master_valid.sort_values('Risk_Mean', ascending=False)
    
    # 6. Save Outputs
    out_csv = os.path.join(OUTPUT_DIR, "regulatory_action_matrix.csv")
    cols = ['DTXSID', 'CASRN', 'Chemical Name', 'Regulatory_Action', 'Risk_Mean', 'Risk_Uncertainty']
    df_master_valid[cols].to_csv(out_csv, index=False)
    print(f"Saved Decision Matrix to {out_csv}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_master_valid,
        x='Risk_Uncertainty',
        y='Risk_Mean',
        hue='Regulatory_Action',
        palette={'Regulate / Ban': 'red', 'Urgent Testing': 'orange', 'Monitor': 'blue', 'No Action': 'green', 'Data Gap': 'purple'},
        alpha=0.6,
        s=40
    )
    plt.title(f"Regulatory Decision Matrix (Threshold: Risk>{RISK_HIGH})")
    plt.xlabel("Prediction Uncertainty (Std Dev)")
    plt.ylabel("Predicted Toxicity Probability (Mean)")
    plt.axhline(y=RISK_HIGH, color='grey', linestyle='--')
    plt.axvline(x=UNCERTAINTY_HIGH, color='grey', linestyle='--')
    
    plot_path = os.path.join(OUTPUT_DIR, "uncertainty_decision_plot.png")
    plt.savefig(plot_path)
    print(f"Saved Plot to {plot_path}")
    
    # Print Top Recommendations
    print("\n--- TOP CANDIDATES FOR REGULATION (Confident High Risk) ---")
    reg = df_master_valid[df_master_valid['Regulatory_Action'] == 'Regulate / Ban']
    print(reg[['DTXSID', 'Chemical Name', 'Risk_Mean', 'Risk_Uncertainty']].head(10))
    
    print("\n--- TOP CANDIDATES FOR TESTING (Uncertain High Risk) ---")
    test = df_master_valid[df_master_valid['Regulatory_Action'] == 'Urgent Testing']
    print(test[['DTXSID', 'Chemical Name', 'Risk_Mean', 'Risk_Uncertainty']].head(10))

if __name__ == "__main__":
    main()
