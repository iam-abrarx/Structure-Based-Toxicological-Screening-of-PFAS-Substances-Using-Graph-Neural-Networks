import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict
import pubchempy as pcp
from rdkit import Chem
import time

# Files
TOX21_FILE = "data/processed/tox21_pfas_matched.csv"
pfas_master_file = "../data/processed/master_chem_table.csv"
gnn_embeddings_file = "../approach1_gnn/embeddings_pfas.npy"
OUTPUT_FILE = "data/processed/tox21_features_ready.csv"

# Configurations
RAW_DATA_DIR = "data_scraping"
MASTER_LIST = "../data/processed/master_chem_table.csv"
OUTPUT_DIR = "data/processed"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TARGETS = ["SR-p53", "SR-HSE", "SR-MMP", "NR-AR", "NR-ER", "NR-PPAR-gamma"]

def canonicalize_smiles(smiles):
    """Canonicalize SMILES using RDKit."""
    if pd.isna(smiles) or smiles == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None

def fetch_cid_smiles(cids):
    """Fetch Isomeric or Canonical SMILES for a list of CIDs using PubChemPy."""
    print(f"Fetching SMILES for {len(cids)} CIDs...")
    cid_map = {}
    
    # Batch processing (PubChem limits)
    batch_size = 50 
    cids = list(cids)
    
    for i in range(0, len(cids), batch_size):
        batch = cids[i:i+batch_size]
        try:
            # Request both Isomeric and Canonical as fallback
            properties = pcp.get_properties(['IsomericSMILES', 'CanonicalSMILES'], batch, namespace='cid')
            
            for prop in properties:
                cid = int(prop.get('CID', 0))
                if cid == 0: continue
                
                # Check all possible keys because PUG REST/PubChemPy might vary
                smiles = prop.get('IsomericSMILES')
                if not smiles: smiles = prop.get('CanonicalSMILES')
                if not smiles: smiles = prop.get('SMILES')
                    
                if smiles:
                    cid_map[cid] = smiles
                    
        except Exception as e:
            print(f"  Error fetching batch {i}: {e}")
            # Don't sleep too long if it's just a key error, but here it's safer to sleep on network err
            time.sleep(1)
            
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(cids)}... (Found {len(cid_map)} SMILES)")
            # Intermediate save to cache to prevent total loss
            temp_df = pd.DataFrame(list(cid_map.items()), columns=['CID', 'SMILES'])
            temp_df.to_csv("cid_smiles_cache_partial.csv", index=False)
            
    return cid_map

def load_master_list():
    """Load PFAS master list for mapping."""
    print(f"Loading Master Keys from {MASTER_LIST}...")
    df = pd.read_csv(MASTER_LIST)
    # We need a lookup strategy.
    # We will try to map via DTXSID, then CASRN (if available in raw), then SMILES?
    # Actually raw PubChem usually has CID.
    # Our master list might not have CID populated for everything, but let's check.
    # For now, we return the whole df to merge on DTXSID if possible.
    return df

def process_endpoint(endpoint):
    """Process all files for a single endpoint."""
    print(f"\nProcessing Endpoint: {endpoint}")
    
    # PubChem CSVs use uppercase usually: PUBCHEM_CID, PUBCHEM_ACTIVITY_OUTCOME
    # AC50 might be 'PUBCHEM_ACTIVITY_SCORE' or 'AC50' or missing (replicates).
    # We will prioritize Outcome and CID.
    
    clean_rows = []
    
    files = glob.glob(os.path.join(RAW_DATA_DIR, f"{endpoint}*.csv"))
    if not files:
        # Fallback search if naming convention differs slightly
        files = glob.glob(os.path.join(RAW_DATA_DIR, f"*{endpoint}*.csv"))
        
    print(f"  Found {len(files)} files.")
    
    for f in files:
        try:
            # Read with low_memory=False to avoid DtypeBenchmarks
            df = pd.read_csv(f, low_memory=False)
            
            # Identify columns
            cols = df.columns.tolist()
            cid_col = next((c for c in cols if 'PUBCHEM_CID' in c.upper()), None)
            outcome_col = next((c for c in cols if 'ACTIVITY_OUTCOME' in c.upper()), None)
            
            # AC50/Score is trickier. Look for Score or AC50.
            ac50_col = next((c for c in cols if 'ACTIVITY_SCORE' in c.upper()), None)
            if not ac50_col:
                ac50_col = next((c for c in cols if 'AC50' in c.upper()), None)
            
            if not cid_col or not outcome_col:
                print(f"  Skipping {os.path.basename(f)}: Critical columns missing.")
                continue
                
            # Filter and Rename
            subset = df[[cid_col, outcome_col]].copy()
            subset.columns = ['PubChem_CID', 'Activity Outcome']
            
            if ac50_col:
                subset['AC50'] = df[ac50_col]
            else:
                subset['AC50'] = np.nan
                
            # Clean Data
            subset = subset.dropna(subset=['PubChem_CID', 'Activity Outcome'])
            subset = subset[subset['Activity Outcome'] != 'Inconclusive']
            
            # Normalize Outcome (Active -> 1, Inactive -> 0)
            # PubChem Outcome: 'Active', 'Inactive', 'Inconclusive', 'Unspecified'
            subset['Binary'] = subset['Activity Outcome'].apply(lambda x: 1 if str(x).lower() == 'active' else (0 if str(x).lower() == 'inactive' else -1))
            subset = subset[subset['Binary'] != -1]
            
            if not subset.empty:
                # Add to list
                for _, row in subset.iterrows():
                    clean_rows.append({
                        'PubChem_CID': row['PubChem_CID'],
                        f'{endpoint}_Binary': row['Binary'],
                        f'{endpoint}_AC50': row['AC50']
                    })
                    
        except Exception as e:
            print(f"  Error reading {f}: {e}")
            continue

    if not clean_rows:
        print(f"  No valid data found for {endpoint}")
        return None
        
    df_clean = pd.DataFrame(clean_rows)
    print(f"  Raw merged rows: {len(df_clean)}")
    
    df_clean = pd.DataFrame(clean_rows)
    print(f"  Raw merged rows: {len(df_clean)}")
    
    # RESOLVE CONFLICTS
    # Strategy: Group by CID
    # Active (1) > Inactive (0)
    # AC50: Median of entries where Binary=1 (if any), else NaN
    
    # 1. Aggregation for Binary: Max (1 overrides 0)
    # 2. Aggregation for AC50: Median (we'll do this carefully)
    
    binary_col = f'{endpoint}_Binary'
    ac50_col = f'{endpoint}_AC50'
    
    # Group By CID
    # We can do a custom agg
    grouped = df_clean.groupby('PubChem_CID')
    
    final_rows = []
    
    for cid, group in grouped:
        binaries = group[binary_col]
        # Max: 1 if any 1, else 0
        final_binary = binaries.max()
        
        final_ac50 = np.nan
        if final_binary == 1:
            # Get AC50s from rows that are Active(1)
            # Some active rows might have NaN AC50
            actives = group[group[binary_col] == 1]
            vals = actives[ac50_col].dropna()
            if not vals.empty:
                final_ac50 = vals.median()
        
        final_rows.append({
            'PubChem_CID': cid,
            binary_col: final_binary,
            ac50_col: final_ac50
        })
        
    df_final = pd.DataFrame(final_rows)
    print(f"  Unique chemicals count: {len(df_final)}")
    
    n_active = (df_final[binary_col] == 1).sum()
    print(f"  Actives: {n_active} | Inactives: {len(df_final) - n_active}")
    
    return df_final

def main():
    print("--- Starting Tox21 Data Processing ---")
    
    # 1. Process each endpoint
    endpoint_dfs = []
    all_cids = set()
    
    for target in TARGETS:
        df = process_endpoint(target)
        if df is not None:
            endpoint_dfs.append(df)
            all_cids.update(df['PubChem_CID'].dropna().astype(int).tolist())
            
    if not endpoint_dfs:
        print("No data found.")
        return

    # 2. Merge all endpoints
    print("\nMerging endpoints...")
    full_tox = endpoint_dfs[0]
    for df in endpoint_dfs[1:]:
        full_tox = pd.merge(full_tox, df, on='PubChem_CID', how='outer')
        
    print(f"Total Unique CIDs in Tox21: {len(full_tox)}")
    
    # 3. Fetch SMILES for CIDs
    print("\nFetching SMILES for mapping...")
    # Check if we have a cache to save time
    cache_file = "cid_smiles_cache.csv"
    if os.path.exists(cache_file):
        print("  Loading SMILES from cache...")
        cid_df = pd.read_csv(cache_file)
        cid_smiles_map = dict(zip(cid_df['CID'], cid_df['SMILES']))
        # Fetch missing
        missing_cids = [c for c in all_cids if c not in cid_smiles_map]
        if missing_cids:
            new_map = fetch_cid_smiles(missing_cids)
            cid_smiles_map.update(new_map)
            # Update cache
            pd.DataFrame(list(cid_smiles_map.items()), columns=['CID', 'SMILES']).to_csv(cache_file, index=False)
    else:
        cid_smiles_map = fetch_cid_smiles(all_cids)
        pd.DataFrame(list(cid_smiles_map.items()), columns=['CID', 'SMILES']).to_csv(cache_file, index=False)
        
    full_tox['SMILES_Tox21'] = full_tox['PubChem_CID'].map(cid_smiles_map)
    full_tox['Canonical_SMILES'] = full_tox['SMILES_Tox21'].apply(canonicalize_smiles)
    
    # 4. Map to Master List
    print("\nMapping to PFAS Master List...")
    master_df = load_master_list()
    # Check master columns
    if 'SMILES_canonical' not in master_df.columns:
        # Generate if missing (should exist from harmonize step)
        master_df['SMILES_canonical'] = master_df['raw_SMILES'].apply(canonicalize_smiles)
    
    # Merge on Canonical SMILES
    # master_df key: SMILES_canonical
    # tox key: Canonical_SMILES
    
    merged_data = pd.merge(full_tox, master_df, left_on='Canonical_SMILES', right_on='SMILES_canonical', how='inner')
    
    print(f"\n--- MATCHING RESULTS ---")
    print(f"Tox21 Total CIDs: {len(full_tox)}")
    print(f"PFAS Master List Size: {len(master_df)}")
    print(f"Matched PFAS with Tox21 Data: {len(merged_data)}")
    
    if len(merged_data) > 0:
        output_file = os.path.join(OUTPUT_DIR, "tox21_pfas_matched.csv")
        merged_data.to_csv(output_file, index=False)
        print(f"Saved matched dataset to {output_file}")
    else:
        print("WARNING: No overlap found between Tox21 and PFAS Master List.")
        # Fallback: Save full tox21 just in case
        full_tox.to_csv(os.path.join(OUTPUT_DIR, "tox21_full_unmatched.csv"), index=False)

if __name__ == "__main__":
    main()
