import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler

# File Paths
FILE_APP3_RISK = "approach3_risk/pfas_risk_ranked.csv"
FILE_APP5_RISK = "approach5_tox21/results/pfas_risk_ranking_pparg.csv"
FILE_MASTER = "data/processed/master_chem_table.csv"
OUTPUT_DIR = "final_risk_integration"
def calc_tk_features(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            # Proxy for chain length/persistence: FractionCSP3 or Heavy Atoms?
            # User suggested Chain Length (Heavy Atoms or F-count). 
            # Let's use simple HeavyAtomCount for now as robust proxy.
            heavy_atoms = mol.GetNumHeavyAtoms()
            # Fluorine count is also good
            f_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
            return logp, tpsa, f_count
    except:
        pass
    return None, None, None

def calculate_exposure_proxy(row):
    """
    Estimate Exposure (1-10) based on metadata proxies.
    Assumption: Listed chemicals have higher production/detection.
    """
    score = 1.0 # Baseline
    
    # 1. Regulatory/Inventory Presence
    if row.get('is_pfas_list', False):
        score += 4.0
        
    # 2. Tox21 Tested
    if row.get('is_tox21_list', False):
        score += 2.0
        
    # 3. Structural Heuristics
    smiles = str(row.get('SMILES', '')).upper()
    
    # Sulfonates (PFOS-like)
    if 'S(=O)(=O)' in smiles or 'S(O)(=O)' in smiles:
        score += 2.0
        
    # Carboxylates (PFOA-like)
    if 'C(=O)O' in smiles:
        score += 1.0
        
    # Cap at 10
    return min(score, 10.0)

def main():
    print("--- Calculating Real-World Risk (Hazard x Exposure x Persistence) ---")
    
    # 1. Load Data
    print("Loading datasets...")
    try:
        df_app3 = pd.read_csv(FILE_APP3_RISK) # DTXSID, Risk_Score (0-100)
        df_app5 = pd.read_csv(FILE_APP5_RISK) # DTXSID, Risk_Score_PPARg (0-1)
        df_master = pd.read_csv(FILE_MASTER)  # DTXSID, Metadata
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Key Standardization (Merge on DTXSID)
    # Rename cols to be clear
    df_app3 = df_app3[['DTXSID', 'risk_score']].rename(columns={'risk_score': 'Hazard_PBT'})
    df_app5 = df_app5[['DTXSID', 'Risk_Score_PPARg']].rename(columns={'Risk_Score_PPARg': 'Hazard_ToxML'})
    
    # Master: Needs SMILES and flags
    # Check master columns
    # We saw 'is_pfas_list' in previous view
    cols_needed = ['DTXSID', 'CASRN', 'Chemical Name', 'SMILES_canonical', 'is_pfas_list', 'is_tox21_list']
    # Handle missing cols gracefully using get
    df_meta = df_master.copy()
    if 'SMILES_canonical' not in df_meta.columns: df_meta['SMILES_canonical'] = df_meta.get('raw_SMILES', '')
    df_meta['SMILES'] = df_meta['SMILES_canonical']
    
    # 3. Merge
    print("Merging Hazard scores...")
    df_merged = pd.merge(df_meta, df_app3, on='DTXSID', how='left')
    df_merged = pd.merge(df_merged, df_app5, on='DTXSID', how='left')
    
    # Impute missing Hazards (Use median or 0? Use Median to avoid zeroing out potential risks)
    df_merged['Hazard_PBT'] = df_merged['Hazard_PBT'].fillna(df_merged['Hazard_PBT'].median())
    df_merged['Hazard_ToxML'] = df_merged['Hazard_ToxML'].fillna(0.0) 
    
    # 3.5 Calculate TK Features using RDKit
    print("Calculating TK Surrogates (LogP, TPSA, F-Count)...")
    tk_features = df_merged['SMILES'].apply(calc_tk_features)
    tk_df = pd.DataFrame(tk_features.tolist(), columns=['LogP', 'TPSA', 'F_Count'], index=df_merged.index)
    df_merged = pd.concat([df_merged, tk_df], axis=1)
    
    # Simple Imputation for TK feats
    df_merged.fillna({'LogP': 2.0, 'TPSA': 50.0, 'F_Count': 6.0}, inplace=True)

    # 4. Calculate Scores
    print("Calculating Persistence & Risk Index...")
    
    # Exposure
    df_merged['Exposure_Score'] = df_merged.apply(calculate_exposure_proxy, axis=1)
    
    # Persistence Score Calculation
    # Normalize features to 0-1 range for combination
    scaler = MinMaxScaler()
    feat_norm = scaler.fit_transform(df_merged[['LogP', 'TPSA', 'F_Count']])
    # Persistence = (LogP_norm + F_Count_norm) - TPSA_norm (High TPSA = Easy Clearance)
    # Re-scale to roughly 0.5 - 2.0 multiplier logic? Or 0-10?
    # Let's map to 0.5 - 1.5 range to start, being conservative?
    # Implementation Plan says: Persistence Score. Let's make it 1-10 like others.
    
    p_raw = (feat_norm[:, 0] + feat_norm[:, 2]) - feat_norm[:, 1]
    # Shift and Scale to 1-10
    # p_raw is approx -1 to 2 range.
    p_scaled = (p_raw - p_raw.min()) / (p_raw.max() - p_raw.min()) # 0-1
    df_merged['Persistence_Score'] = 1.0 + (p_scaled * 9.0) # 1-10
    
    # Combined Hazard (0-10 Scale)
    # PBT is 0-1 -> *10
    # ToxML is 0-1 -> *10
    df_merged['Hazard_Composite'] = ( (df_merged['Hazard_PBT'] * 10.0) + (df_merged['Hazard_ToxML'] * 10.0) ) / 2.0
    
    # Global Risk = Hazard * Exposure * Persistence (Normalizing factor /10 to keep scale readable?)
    # Or just keep raw product? 10*10*10 = 1000 max.
    df_merged['Global_Risk_Index'] = (df_merged['Hazard_Composite'] * df_merged['Exposure_Score'] * df_merged['Persistence_Score']) / 10.0
    
    # 5. Categorization
    def categorize(row):
        h = row['Hazard_Composite']
        e = row['Exposure_Score']
        p = row['Persistence_Score']
        
        # Critical: High Haz AND (High Exp OR High Pers)
        # Refined: High Haz (>6) + High Pers (>7) = "Forever Toxic"
        if h >= 6.0 and p >= 7.0:
            return "Critical Priority (Persistent)"
        elif h >= 6.0 and e >= 5.0:
             return "High Exposure Risk"
        elif h < 4.0 and p < 4.0 and e >= 5.0:
            return "Safe Substitute Candidate"
        # Moderate
        else:
            return "Moderate Concern"
            
    df_merged['Risk_Category'] = df_merged.apply(categorize, axis=1)
    
    # 6. Save & Plot
    # Sort
    df_final = df_merged.sort_values('Global_Risk_Index', ascending=False)
    
    out_csv = os.path.join(OUTPUT_DIR, "final_global_risk_matrix.csv")
    cols_out = ['DTXSID', 'CASRN', 'Chemical Name', 'Global_Risk_Index', 'Risk_Category', 'Hazard_Composite', 'Exposure_Score', 'Persistence_Score', 'LogP', 'TPSA']
    df_final[cols_out].to_csv(out_csv, index=False)
    print(f"Saved Global Risk Matrix to {out_csv}")
    
    # Plot 1: Hazard vs Exposure (Color by Persistence)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_final, 
        x='Persistence_Score', 
        y='Hazard_Composite', 
        hue='Risk_Category',
        palette='viridis',
        alpha=0.6
    )
    plt.title("PFAS Risk: Hazard vs. Persistence")
    plt.xlabel("Persistence Score (LogP + F-Count - TPSA)")
    plt.ylabel("Composite Hazard Score")
    plt.axhline(y=6, color='black', linestyle='--')
    plt.axvline(x=7, color='black', linestyle='--')
    
    plot_path = os.path.join(OUTPUT_DIR, "risk_matrix_plot.png")
    plt.savefig(plot_path)
    print(f"Saved Matrix Plot to {plot_path}")
    
    # Print Top 5 Critical
    print("\n--- TOP 5 CRITICAL PRIORITY PFAS ---")
    critical = df_final[df_final['Risk_Category'] == 'Critical Priority']
    print(critical[['DTXSID', 'Chemical Name', 'Global_Risk_Index']].head(5))

if __name__ == "__main__":
    main()
