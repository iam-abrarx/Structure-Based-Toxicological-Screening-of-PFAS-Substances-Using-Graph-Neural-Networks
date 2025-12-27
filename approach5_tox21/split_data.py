import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Files
INPUT_FILE = "data/processed/tox21_features_ready.csv"
OUTPUT_DIR = "data/processed/splits"

def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold
    except:
        return None

def scaffold_split(df, smiles_col='Canonical_SMILES', frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42):
    """
    Split a dataset by scaffold so that no molecules in the test set share a scaffold with the training set.
    """
    np.random.seed(seed)
    
    print("Generating scaffolds...")
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold = generate_scaffold(row[smiles_col])
        if scaffold:
            scaffolds[scaffold].append(idx)
        else:
            # Put null scaffolds in a separate bucket or treat as unique
            scaffolds[f"null_{idx}"].append(idx)
            
    # Sort scaffolds by size (largest to smallest) to ensure balanced split size if possible
    scaffold_sets = sorted(list(scaffolds.values()), key=lambda x: len(x), reverse=True)
    
    train_inds, valid_inds, test_inds = [], [], []
    train_cutoff = frac_train * len(df)
    valid_cutoff = (frac_train + frac_valid) * len(df)
    
    print("Assigning splits...")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds.extend(scaffold_set)
            else:
                valid_inds.extend(scaffold_set)
        else:
            train_inds.extend(scaffold_set)
            
    print(f"Split results: Train={len(train_inds)}, Valid={len(valid_inds)}, Test={len(test_inds)}")
    
    return df.loc[train_inds], df.loc[valid_inds], df.loc[test_inds]

def main():
    print("--- Starting Scaffold Split ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Complete feature attachment first.")
        return
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded Features: {len(df)} rows")
    
    # Check for SMILES column
    if 'Canonical_SMILES' not in df.columns:
        print("Error: 'Canonical_SMILES' column missing.")
        return
        
    # Perform Split
    train_df, valid_df, test_df = scaffold_split(df)
    
    # Save
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    
    print(f"Saved splits to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
