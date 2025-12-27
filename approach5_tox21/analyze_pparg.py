import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import mannwhitneyu
import os

INPUT_FILE = "data/processed/tox21_features_ready.csv"
RESULTS_DIR = "results"
ENDPOINT = "NR-PPAR-gamma"

def calculate_biology_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        
        # Fluorine count
        f_count = len([a for a in mol.GetAtoms() if a.GetSymbol() == 'F'])
        
        # Approx Chain Length (Longest Carbon Chain)
        # This is a heuristic. We can use Rotatable Bonds as proxy for flexibility/length
        # or just heavy atom count.
        # Let's use Heavy Atom Count as a proxy for size/length in PFAS context often correlated.
        heavy_atoms = mol.GetNumHeavyAtoms()
        
        mw = Descriptors.MolWt(mol)
        
        return pd.Series([f_count, heavy_atoms, mw], index=['Bio_F_Count', 'Bio_Heavy_Atoms', 'Bio_MW'])
    except:
        return pd.Series([np.nan, np.nan, np.nan], index=['Bio_F_Count', 'Bio_Heavy_Atoms', 'Bio_MW'])

def main():
    print(f"--- Analyzing Biology of {ENDPOINT} ---")
    
    if not os.path.exists(INPUT_FILE):
        print("Data not found.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # Filter for endpoint valid data
    binary_col = f'{ENDPOINT}_Binary'
    mask = ~df[binary_col].isna()
    df = df[mask].copy()
    
    print(f"Data Points: {len(df)}")
    
    # Calculate Features
    print("Calculating biological features...")
    feats = df['Canonical_SMILES'].apply(calculate_biology_features)
    # Avoid duplicate columns if re-running
    feats = feats[[c for c in feats.columns if c not in df.columns]] 
    # Actually simpler: just concat and use the NEW names. 
    # But wait, my replace block changed the names in calculate_biology_features to Bio_...
    # So collision with old 'MW' is gone.
    
    df = pd.concat([df, feats], axis=1)
    
    features_to_test = ['Bio_F_Count', 'Bio_Heavy_Atoms', 'Bio_MW']
    
    # Stats & Plots
    # Save boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    stats = []
    
    for i, feat in enumerate(features_to_test):
        # Drop NaNs
        clean = df.dropna(subset=[feat, binary_col])
        actives = clean[clean[binary_col] == 1][feat]
        inactives = clean[clean[binary_col] == 0][feat]
        
        # Test
        stat, pval = mannwhitneyu(actives, inactives, alternative='two-sided')
        stats.append({
            'Feature': feat,
            'Active_Mean': actives.mean(),
            'Inactive_Mean': inactives.mean(),
            'P-Value': pval
        })
        
        # Plot
        sns.boxplot(x=binary_col, y=feat, data=clean, ax=axes[i], palette="Set2")
        axes[i].set_title(f"{feat}\n(p={pval:.2e})")
        axes[i].set_xticklabels(['Inactive', 'Active'])
        
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "ppar_gamma_bio_analysis.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
    # Save Stats
    stats_df = pd.DataFrame(stats)
    stats_path = os.path.join(RESULTS_DIR, "ppar_gamma_bio_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print("\n--- Statistical Differences ---")
    print(stats_df)

if __name__ == "__main__":
    main()
