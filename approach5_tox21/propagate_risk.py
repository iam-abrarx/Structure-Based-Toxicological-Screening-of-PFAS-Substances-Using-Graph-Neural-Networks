import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from tqdm import tqdm

INPUT_DATA_TOX = "data/processed/tox21_features_ready.csv"
INPUT_MASTER = "../data/processed/master_chem_table.csv"
RESULTS_DIR = "results"
ENDPOINT = "NR-PPAR-gamma"
FP_RADIUS = 2
FP_BITS = 2048

def get_descriptors(mol):
    if not mol:
        return [np.nan]*6
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol)
    ]

def get_fingerprint(mol):
    if not mol:
        return np.zeros(FP_BITS) # Or NaN
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    return np.array(fp)

def process_master_list(master_file):
    print(f"Loading Master List from {master_file}...")
    df = pd.read_csv(master_file)
    # We need SMILES_canonical or raw_SMILES
    if 'SMILES_canonical' not in df.columns:
         if 'raw_SMILES' in df.columns:
             df['SMILES'] = df['raw_SMILES']
         else:
             print("Error: No SMILES column in Master List.")
             return None
    else:
        df['SMILES'] = df['SMILES_canonical']
        
    valid_data = []
    
    print("Generating Features for components...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row.get('SMILES')
        if pd.isna(smiles):
            continue
            
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
            
        desc = get_descriptors(mol)
        fp = get_fingerprint(mol)
        
        # Combine
        # Standardize col names to match training
        # Training has: fp_0...fp_2047, MW, LogP, TPSA, HBD, HBA, RotBonds
        
        # We store as dict to build DF later
        row_data = {
            'DTXSID': row.get('DTXSID', f'Unk_{idx}'),
            'SMILES': smiles,
        }
        
        # Descriptors
        desc_names = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds']
        for n, v in zip(desc_names, desc):
            row_data[n] = v
            
        # FP
        for i, v in enumerate(fp):
            row_data[f'fp_{i}'] = v
            
        valid_data.append(row_data)
        
    return pd.DataFrame(valid_data)

def main():
    print("--- Risk Propagation: NR-PPAR-gamma ---")
    
    # 1. Train Model on Full Tox21 Data (Best quality)
    print("Training Model on Full Tox21 Data...")
    df_tox = pd.read_csv(INPUT_DATA_TOX)
    mask = ~df_tox[f'{ENDPOINT}_Binary'].isna()
    df_tox = df_tox[mask]
    
    # Define Features
    # Note: GNN embeddings might be missing for some Master List items if not aligned.
    # To be safe and consistent, we use RDKit + Morgan ONLY for this step 
    # to ensure coverage of the whole master list without dependency on specific GNN precalc.
    # The Random Forest baseline performed very well (0.82) with these.
    
    feat_cols = [c for c in df_tox.columns if c.startswith('fp_') or c in ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds']]
    target_col = f'{ENDPOINT}_Binary'
    
    X_train = df_tox[feat_cols].values
    y_train = df_tox[target_col].values
    
    # Impute
    imp = SimpleImputer(strategy='mean')
    X_train = imp.fit_transform(X_train)
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("Model Trained.")
    
    # 2. Process Master List
    df_master = process_master_list(INPUT_MASTER)
    if df_master is None or df_master.empty:
        print("Failed to process Master List.")
        return
        
    # Align features
    # Ensure columns match X_train order
    # master might have NaNs
    
    # Check for missing feat cols in master and add 0
    for c in feat_cols:
        if c not in df_master.columns:
            df_master[c] = 0
            
    X_pred = df_master[feat_cols].values
    X_pred = imp.transform(X_pred) # Use same imputer!
    
    print(f"Predicting Risk for {len(df_master)} compounds...")
    probs = rf.predict_proba(X_pred)[:, 1]
    
    df_master['Risk_Score_PPARg'] = probs
    
    # Rank
    ranked = df_master[['DTXSID', 'SMILES', 'Risk_Score_PPARg']].sort_values('Risk_Score_PPARg', ascending=False)
    
    out_file = os.path.join(RESULTS_DIR, "pfas_risk_ranking_pparg.csv")
    ranked.to_csv(out_file, index=False)
    
    print(f"\n--- Top 10 High Risk PFAS (PPAR-gamma) ---")
    print(ranked.head(10))
    print(f"\nSaved Full Ranking to {out_file}")

if __name__ == "__main__":
    main()
