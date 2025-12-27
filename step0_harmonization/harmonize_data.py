import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolStandardize
import os

# Configuration
PFAS_FILE = "../data/raw/EPA PFAS Master List V2.xlsx"
TOX21_FILE = "../data/raw/tx0c00264_si_003.xlsx"
OUTPUT_CSV = "../data/processed/master_chem_table.csv"
OUTPUT_PARQUET = "../data/processed/master_chem_table.parquet"

def clean_cas(cas_series):
    """Normalize CASRNs: strip whitespace, ensure dashes."""
    return cas_series.astype(str).str.strip().str.replace(' ', '', regex=False)

def standardize_smiles(smiles_series):
    """
    Standardize SMILES strings using RDKit:
    1. Sanitize
    2. Canonicalize
    Returns list of (canonical_smiles, is_valid)
    """
    clean_smiles = []
    valid_flags = []
    
    print(f"Standardizing {len(smiles_series)} SMILES strings...")
    
    for smi in smiles_series:
        if pd.isna(smi) or str(smi).strip() == '':
            clean_smiles.append(None)
            valid_flags.append(False)
            continue
            
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                # Optional: specific standardization pipeline can go here
                # For now, just canonicalize
                can_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                clean_smiles.append(can_smi)
                valid_flags.append(True)
            else:
                clean_smiles.append(None)
                valid_flags.append(False)
        except:
            clean_smiles.append(None)
            valid_flags.append(False)
            
    return clean_smiles, valid_flags

def main():
    print(f"Loading {PFAS_FILE}...")
    # Load EPA PFAS Master List
    df_pfas = pd.read_excel(PFAS_FILE, sheet_name='EPA PFAS Master List V2')
    
    # Selecting core columns
    # Adjust column names based on inspection: 'CASRN', 'Chemical Name', 'DTXSID', 'SMILES'
    df_pfas = df_pfas[['DTXSID', 'CASRN', 'Chemical Name', 'SMILES']].copy()
    df_pfas['is_pfas_list'] = True
    
    print(f"Loading {TOX21_FILE}...")
    # Load Tox21 Data
    # Sheet S2 has chemical info
    df_tox21 = pd.read_excel(TOX21_FILE, sheet_name='S2.TOX21S')
    # Sheet S4 has properties (labels)
    df_tox21_props = pd.read_excel(TOX21_FILE, sheet_name='S4.Predicted properties')
    
    # Merge Tox21 info and properties
    df_tox21_full = df_tox21.merge(df_tox21_props, on='DTXSID', how='left')
    
    # Rename/Select Tox21 columns to match or map
    # Tox21 columns: DTXSID, PREFERRED_NAME, CAS RN, Structure_SMILES, ... properties ...
    tox21_cols = {
        'DTXSID': 'DTXSID',
        'CAS RN': 'CASRN',
        'PREFERRED_NAME': 'Chemical Name',
        'Structure_SMILES': 'SMILES',
        'AMES_MUTAGENICITY_TEST_PRED': 'tox21_ames_mutagenicity',
        'RatCarc_DEREK': 'tox21_rat_carcinogenicity',
        'ORAL_RAT_LD50_MOL/KG_TEST_PRED': 'tox21_oral_rat_ld50_mol_kg',
        'VAPOR_PRESSURE_MMHG_OPERA_PRED': 'tox21_vapor_pressure'
    }
    
    # Filter for columns that actually exist
    available_cols = [c for c in tox21_cols.keys() if c in df_tox21_full.columns]
    
    df_tox21_subset = df_tox21_full[available_cols].copy()
    df_tox21_subset.rename(columns=tox21_cols, inplace=True)
    df_tox21_subset['is_tox21_list'] = True
    
    print("Merging datasets on DTXSID...")
    # Outer join to keep everything
    # We prioritize PFAS list info, but fill from Tox21 where missing?
    # Actually, we want a master list.
    
    df_master = pd.merge(
        df_pfas,
        df_tox21_subset,
        on='DTXSID',
        how='outer',
        suffixes=('_pfas', '_tox21')
    )
    
    # Coalesce columns
    # DTXSID is the key, so it's already merged.
    # CASRN
    df_master['CASRN'] = df_master['CASRN_pfas'].combine_first(df_master['CASRN_tox21'])
    # Chemical Name
    df_master['Chemical Name'] = df_master['Chemical Name_pfas'].combine_first(df_master['Chemical Name_tox21'])
    # SMILES
    df_master['raw_SMILES'] = df_master['SMILES_pfas'].combine_first(df_master['SMILES_tox21'])
    
    # Fill flags
    df_master['is_pfas_list'] = df_master['is_pfas_list'].fillna(False)
    df_master['is_tox21_list'] = df_master['is_tox21_list'].fillna(False)
    
    # Clean CAS
    print("Cleaning CASRNs...")
    df_master['CASRN'] = clean_cas(df_master['CASRN'])
    
    # Standardize SMILES
    print("Standardizing SMILES (this may take a moment)...")
    clean_smis, valid_mask = standardize_smiles(df_master['raw_SMILES'])
    df_master['SMILES_canonical'] = clean_smis
    df_master['SMILES_valid'] = valid_mask
    
    # Create final table structure
    final_cols = [
        'DTXSID', 'CASRN', 'Chemical Name', 
        'raw_SMILES', 'SMILES_canonical', 'SMILES_valid',
        'is_pfas_list', 'is_tox21_list',
        'tox21_ames_mutagenicity', 'tox21_rat_carcinogenicity', 'tox21_oral_rat_ld50_mol_kg', 'tox21_vapor_pressure'
    ]
    
    # Keep only columns that exist (some tox21 cols might have been missing)
    final_cols = [c for c in final_cols if c in df_master.columns]
    
    df_final = df_master[final_cols].copy()
    
    print(f"Final shape: {df_final.shape}")
    print(f"PFAS List count: {df_final['is_pfas_list'].sum()}")
    print(f"Tox21 List count: {df_final['is_tox21_list'].sum()}")
    print(f"Overlap count: {df_final[df_final['is_pfas_list'] & df_final['is_tox21_list']].shape[0]}")
    print(f"Valid SMILES count: {df_final['SMILES_valid'].sum()}")
    
    # Save
    print(f"Saving to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    if hasattr(pd.DataFrame, "to_parquet"):
        try:
            print(f"Saving to {OUTPUT_PARQUET}...")
            df_final.to_parquet(OUTPUT_PARQUET, index=False)
        except ImportError:
            print("pyarrow or fastparquet not installed, skipping parquet save.")
        except Exception as e:
            print(f"Error saving parquet: {e}")

if __name__ == "__main__":
    main()
