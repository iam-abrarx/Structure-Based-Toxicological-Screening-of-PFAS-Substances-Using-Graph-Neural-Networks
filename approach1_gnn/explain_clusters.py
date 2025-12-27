import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def analyze():
    df = pd.read_csv("pfas_clusters.csv")
    
    print(f"Total entries: {len(df)}")
    print(f"Clusters found: {df['cluster'].unique()}")
    
    summary = []
    
    for c in sorted(df['cluster'].unique()):
        if c == -1: continue
        
        subset = df[df['cluster'] == c]
        smiles_list = subset['SMILES'].tolist()
        
        # Analyze structure
        sulfonates = 0
        carboxylates = 0
        fluorines = 0
        mws = []
        
        for s in smiles_list:
            if pd.isna(s): continue
            
            # Simple substring check for speed (approximate)
            if "S(=O)(=O)O" in s or "S(=O)(=O)[O-]" in s:
                sulfonates += 1
            if "C(=O)O" in s or "C(=O)[O-]" in s:
                carboxylates += 1
                
            mol = Chem.MolFromSmiles(s)
            if mol:
                mws.append(Descriptors.MolWt(mol))
                fluorines += s.count('F')
        
        avg_mw = sum(mws)/len(mws) if mws else 0
        avg_f = fluorines/len(smiles_list)
        
        label = "Unknown"
        if sulfonates / len(smiles_list) > 0.8:
            label = "Sulfonates"
        elif carboxylates / len(smiles_list) > 0.8:
            label = "Carboxylates"
        elif avg_mw > 600:
            label = "Large Polymers"
        elif avg_f > 15:
            label = "Perfluorinated (High F)"
        else:
            label = "Fluorotelomers/Mix"
            
        print(f"Cluster {c}: N={len(subset)} | Avg MW={avg_mw:.1f} | Label={label}")
        print(f"  Sample: {smiles_list[0] if smiles_list else 'None'}")

if __name__ == "__main__":
    analyze()
