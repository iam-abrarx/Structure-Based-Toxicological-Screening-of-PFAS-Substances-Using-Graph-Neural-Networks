import requests
import json
import time
import os
import pandas as pd

# Keywords defined by the user
TARGETS = {
    "NR-PPAR-gamma": "Tox21 PPAR gamma",
    "SR-p53": "Tox21 p53",
    "SR-HSE": "Tox21 HSE",
    "SR-MMP": "Tox21 MMP",
    "NR-AR": "Tox21 AR",
    "NR-ER": "Tox21 ER"
}

OUTPUT_DIR = "data_scraping"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_aids(keyword):
    """Search for Assay IDs (AIDs) using NCBI E-Utilities."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pcassay",
        "term": keyword,
        "retmode": "json",
        "retmax": 20  # Limit to top 20 matches to avoid downloading random user assays
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "esearchresult" in data and "idlist" in data["esearchresult"]:
            return data["esearchresult"]["idlist"]
        else:
            return []
    except Exception as e:
        print(f"Error searching for {keyword}: {e}")
        return []

def download_csv(aid, prefix):
    """Download CSV data for a specific AID using PUG REST."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
    output_file = os.path.join(OUTPUT_DIR, f"{prefix}_AID_{aid}.csv")
    
    if os.path.exists(output_file):
        print(f"  - File already exists: {output_file}")
        return
    
    print(f"  - Downloading AID {aid}...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"    Saved to {output_file}")
        else:
            print(f"    Failed to download AID {aid} (Status: {response.status_code})")
    except Exception as e:
        print(f"    Error downloading AID {aid}: {e}")
        
    # Rate limit courtesy
    time.sleep(1)

def main():
    print("Starting PubChem Scraper for Tox21 Assays...")
    print(f"Saving to: {os.path.abspath(OUTPUT_DIR)}\n")
    
    for label, keyword in TARGETS.items():
        print(f"Processing Target: {label} (Query: '{keyword}')")
        
        # 1. Get AIDs
        aids = get_aids(keyword)
        if not aids:
            print(f"  No assays found for {keyword}")
            continue
            
        print(f"  Found {len(aids)} matching assays. Fetching data...")
        
        # 2. Download Data for each AID
        for aid in aids:
            download_csv(aid, label)
            
    print("\nScraping Completed.")

if __name__ == "__main__":
    main()
