import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from tqdm import tqdm

# Files
TOX21_FILE = "data/processed/tox21_pfas_matched.csv"
pfas_master_file = "../data/processed/master_chem_table.csv"
gnn_embeddings_file = "../approach1_gnn/embeddings_pfas.npy"
OUTPUT_FILE = "data/processed/tox21_features_ready.csv"

def compute_rdkit_features(smiles):
    """Compute basic RDKit descriptors and Morgan Fingerprints."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None, None
            
        # Descriptors
        desc = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol)
        }
        
        # Morgan Fingerprint (2048 bit)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.zeros((1,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        return desc, fp_array
    except:
        return None, None

def main():
    print("--- Starting Feature Attachment ---")
    
    if not os.path.exists(TOX21_FILE):
        print(f"Error: {TOX21_FILE} not found. Ensure process_tox21.py completed.")
        return

    df = pd.read_csv(TOX21_FILE)
    print(f"Loaded Tox21 Data: {len(df)} samples")
    
    # 1. RDKit Features
    print("Computing RDKit Descriptors & Fingerprints...")
    descriptors_list = []
    fingerprints_list = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row.get('Canonical_SMILES')
        desc, fp = compute_rdkit_features(smiles)
        
        if desc:
            descriptors_list.append(desc)
            fingerprints_list.append(fp)
            valid_indices.append(idx)
    
    if not valid_indices:
        print("No valid SMILES found!")
        return
        
    # Create DF for descriptors
    desc_df = pd.DataFrame(descriptors_list)
    desc_df.index = valid_indices
    
    # Create DF for fingerprints
    # We might keep fingerprints as a list column or expand? 
    # For ML, expanding is often better but huge. Let's keep as numpy array object or specific columns?
    # For compatibility with simple sklearn, expanding into columns is safer.
    print("Expanding fingerprints...")
    fp_matrix = np.stack(fingerprints_list)
    fp_df = pd.DataFrame(fp_matrix, columns=[f'fp_{i}' for i in range(2048)])
    fp_df.index = valid_indices
    
    # Merge back to original DF
    df_features = df.iloc[valid_indices].copy()
    df_features = pd.concat([df_features, desc_df, fp_df], axis=1)
    
    # 2. Attach GNN Embeddings
    print("Attaching GNN Embeddings...")
    if os.path.exists(pfas_master_file) and os.path.exists(gnn_embeddings_file):
        try:
            master_df = pd.read_csv(pfas_master_file)
            embeddings = np.load(gnn_embeddings_file)
            
            if len(master_df) == len(embeddings):
                # Create a map: DTXSID -> Embedding
                # We need to store embedding as a list or string repr to fit in CSV
                # Or save as separate .npy matched to the CSV?
                # Let's save as columns 'gnn_0', 'gnn_1', ...
                
                print("  Mapping embeddings to Tox21 data...")
                # We can optimize this by merging.
                # Create a temp DF with DTXSID and Embeddings
                emb_dim = embeddings.shape[1]
                emb_cols = [f'gnn_{i}' for i in range(emb_dim)]
                emb_df = pd.DataFrame(embeddings, columns=emb_cols)
                emb_df['DTXSID'] = master_df['DTXSID']
                
                # Merge
                # tox21 data should have DTXSID from the previous step
                df_features = pd.merge(df_features, emb_df, on='DTXSID', how='left')
                
                print(f"  Attached embeddings. Shape: {df_features.shape}")
            else:
                print(f"  Warning: Master list length ({len(master_df)}) != Embeddings length ({len(embeddings)}). Skipping GNN attachment.")
        except Exception as e:
            print(f"  Error loading GNN embeddings: {e}")
    else:
        print("  GNN embeddings or Master List not found. Skipping.")
        
    # Standardize/Clean
    # Handle NaNs in GNN columns (if some Tox21 chems matched Master List but master list row order was messed up? 
    # No, merging on DTXSID is robust if unique).
    # If GNN columns are NaN (subset of Master List didn't cover them? Impossible if joined earlier).
    # IF DTXSID was null?
    
    df_features.fillna(0, inplace=True) # Naive imputation for now
    
    # Save
    df_features.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved Features to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
