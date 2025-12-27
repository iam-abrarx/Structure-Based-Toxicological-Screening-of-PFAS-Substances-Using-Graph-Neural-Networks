import pandas as pd
import os
import pubchempy as pcp
import time
from tqdm import tqdm

# Configs
INPUT_MASTER = "../data/processed/master_chem_table.csv"
INPUT_RISK = "../final_risk_integration/final_global_risk_matrix.csv"
OUTPUT_DIR = "."

def generate_batch_files(df):
    print("Generating Batch Search Files...")
    
    # 1. EPA CompTox (DTXSID)
    dtxsids = df['DTXSID'].dropna().unique().tolist()
    with open(os.path.join(OUTPUT_DIR, "epa_comptox_batch_input.txt"), "w") as f:
        f.write("\n".join(dtxsids))
    print(f"Created EPA CompTox ID list ({len(dtxsids)} IDs)")

    # 2. ECHA (CASRN)
    casrns = df['CASRN'].dropna().unique().tolist()
    # ECHA often needs EC number, but CASRN is good start.
    with open(os.path.join(OUTPUT_DIR, "echa_casrn_list.txt"), "w") as f:
        f.write("\n".join(casrns))
    print(f"Created ECHA CASRN list ({len(casrns)} IDs)")

def fetch_pubchem_bioassays(df_risk, top_n=5):
    print(f"Fetching PubChem BioAssays for Top {top_n} Risky Compounds...")
    # Map DTXSID to SMILES to CID
    # We need SMILES from master list or risk list if avail
    
    # In this pipeline, we often fetched CIDs before. 
    # Let's try to get CID from SMILES using pubchempy for just these few.
    
    results = []
    
    for idx, row in df_risk.head(top_n).iterrows():
        dtxsid = row.get('DTXSID')
        # Chemical Name often better for quick log
        name = row.get('Chemical Name', dtxsid)
        
        print(f"Querying: {name}...")
        try:
            # Assumes name lookup works, otherwise need SMILES
            # Use columns if available from merge
            if 'SMILES' in row and not pd.isna(row['SMILES']):
                 compounds = pcp.get_compounds(row['SMILES'], namespace='smiles')
            else:
                 compounds = pcp.get_compounds(name, namespace='name')
            
            if not compounds:
                print("  No compound found via PubChem.")
                continue
                
            cid = compounds[0].cid
            print(f"  CID: {cid}")
            
            # Fetch Assays (Limit to avoiding huge download)
            # This returns Assay Summaries
            assays = pcp.get_assays(cid, limit=5) 
            
            for a in assays:
                # API structure might vary
                # Try common attributes
                outcome = getattr(a, 'outcome', 'Unknown')
                score = getattr(a, 'score', None)
                
                results.append({
                    'DTXSID': dtxsid,
                    'CID': cid,
                    'Assay_ID': a.aid,
                    'Assay_Name': a.name,
                    'Outcome': outcome,
                    'Score': score
                })
                
            time.sleep(0.5) # Rate limit
                
        except Exception as e:
            print(f"  Error: {e}")
            
    if results:
        df_res = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, "pubchem_bioassay_sample.csv")
        df_res.to_csv(out_path, index=False)
        print(f"Saved PubChem BioAssay Sample to {out_path}")
    else:
        print("No BioAssay data found or fetched.")

def create_manual_template():
    print("Creating Manual Literature Template...")
    cols = ["DTXSID", "Chemical_Name", "CASRN", "Half_Life_Hours", "Species", "Source_DOI", "Notes"]
    df = pd.DataFrame(columns=cols)
    out_path = os.path.join(OUTPUT_DIR, "manual_halflife_template.csv")
    df.to_csv(out_path, index=False)
    print(f"Template saved to {out_path}")

def main():
    print("--- External Data Enrichment Tool ---")
    
    # Load Master
    if os.path.exists(INPUT_MASTER):
        df_master = pd.read_csv(INPUT_MASTER)
    else:
        print("Master List not found.")
        return

    # Load Risk List (for focused bioassay fetch)
    if os.path.exists(INPUT_RISK):
        df_risk = pd.read_csv(INPUT_RISK)
    else:
        df_risk = df_master.head(10) # Fallback

    # 1. Generate Batch Files
    generate_batch_files(df_master)
    
    # 2. BioAssay Fetch (Demo)
    # Only run for top critical risks to demonstrate
    # Filter for Critical Priority if column exists
    if 'Risk_Category' in df_risk.columns:
        crit = df_risk[df_risk['Risk_Category'] == 'Critical Priority']
        if not crit.empty:
            df_target = crit
        else:
            df_target = df_risk.sort_values('Global_Risk_Index', ascending=False)
    else:
        df_target = df_risk
        
    print("\n--- Trying Novel Candidates (Expect Sparse Data) ---")
    fetch_pubchem_bioassays(df_target, top_n=3)

    print("\n--- POSITIVE CONTROL (PFOA) ---")
    # Manually fetch PFOA (CID 9554) to prove tool works
    pfoa_control = pd.DataFrame([{'DTXSID': 'DTXSID8031865', 'Chemical Name': 'PFOA', 'SMILES': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F'}])
    fetch_pubchem_bioassays(pfoa_control, top_n=1)
    
    # 3. Manual Template
    create_manual_template()
    
    print("\n--- INSTRUCTIONS ---")
    print("1. Upload 'epa_comptox_batch_input.txt' to: https://comptox.epa.gov/dashboard/batch-search")
    print("2. Search CASRNs from 'echa_casrn_list.txt' at: https://echa.europa.eu/information-on-chemicals")
    print("3. Download OPERA QSARs from: https://github.com/USEPA/OPERA")
    print("4. Fill 'manual_halflife_template.csv' with data from literature (Li et al. 2018).")

if __name__ == "__main__":
    main()
