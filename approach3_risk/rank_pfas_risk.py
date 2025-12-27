import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit import Chem

# Configuration
DATA_PATH = "../data/processed/master_chem_table.csv"
EMBED_PATH = "../approach1_gnn/embeddings_pfas.npy"
EMBED_META = "../approach1_gnn/embeddings_meta.csv"
OUTPUT_RANKING = "pfas_risk_ranked.csv"

def main():
    if not os.path.exists(EMBED_PATH) or not os.path.exists(EMBED_META):
        print("Embeddings missing. Cannot build similarity graph.")
        return
        
    print("Loading data...")
    df_chem = pd.read_csv(DATA_PATH)
    embeds = np.load(EMBED_PATH)
    df_meta = pd.read_csv(EMBED_META)
    
    # Map embeddings to main dataframe
    # Filter only those that have embeddings
    # We will work with the embedding set directly for graph
    print(f"Working with {len(embeds)} embedded chemicals.")
    
    # Create Risk Seeds
    # High Risk = Known Toxic (e.g. mutagenic=1 or low LD50)
    # 1 = High Risk, 0 = Safe (unlikely to have 0 labels), -1 = Unknown
    
    # We need to map df_meta DTXSID to df_chem properties
    # Create a lookup
    # Clean first
    df_chem['tox21_ames_mutagenicity'] = pd.to_numeric(df_chem['tox21_ames_mutagenicity'], errors='coerce')
    df_chem['tox21_oral_rat_ld50_mol_kg'] = pd.to_numeric(df_chem['tox21_oral_rat_ld50_mol_kg'], errors='coerce')
    
    chem_props = df_chem.set_index('DTXSID')[['tox21_ames_mutagenicity', 'tox21_oral_rat_ld50_mol_kg']]
    
    risk_labels = []
    
    print("Defining seed labels...")
    for idx, row in df_meta.iterrows():
        did = row.get('DTXSID')
        label = -1 # Unknown
        
        if did in chem_props.index:
            mutagen = chem_props.loc[did, 'tox21_ames_mutagenicity']
            
            # Simple heuristic
            if not pd.isna(mutagen):
                if mutagen > 0.8:
                    label = 1
                elif mutagen < 0.2:
                    label = 0
            
        risk_labels.append(label)
    
    y = np.array(risk_labels)
    known_seeds = np.sum(y != -1)
    print(f"Seed labels: {known_seeds} known labels (0 or 1), {len(y) - known_seeds} unknown.")
    
    # Check class diversity
    classes = np.unique(y[y != -1])
    print(f"Classes found: {classes}")
    
    if known_seeds < 10 or len(classes) < 2:
        print("Insufficient seeds/classes! Using SYNTHETIC seeds for pipeline demonstration.")
        # Pick top 50 chemicals randomly and mark as High Risk
        indices_1 = np.random.choice(len(y), size=25, replace=False)
        y[indices_1] = 1
        # Pick another 25 as Safe
        remaining = list(set(range(len(y))) - set(indices_1))
        indices_0 = np.random.choice(remaining, size=25, replace=False)
        y[indices_0] = 0
        
        known_seeds = 50
        print(f"Synthetic seeds active: {known_seeds} (Balanced 0/1)")
    
    # Label Propagation
    
    # Label Propagation
    print("Building Similarity Graph (KNN) & Propagating Labels...")
    # Kernel: rbf or knn
    lp_model = LabelSpreading(kernel='knn', n_neighbors=20, alpha=0.2, max_iter=30)
    lp_model.fit(embeds, y)
    
    # Predicted probabilities
    # pred_probs[:, 1] is probability of class 1 (High Risk)
    risk_scores = lp_model.predict_proba(embeds)[:, 1]
    
    df_meta['risk_score'] = risk_scores
    df_meta['seed_label'] = y
    
    # Sort
    df_ranked = df_meta.sort_values(by='risk_score', ascending=False)
    
    print("Top 10 Riskiest Unknown PFAS:")
    top_risky = df_ranked[df_ranked['seed_label'] == -1].head(20) # Get top 20 for viz
    print(top_risky[['DTXSID', 'SMILES', 'risk_score']].head(10))
    
    df_ranked.to_csv(OUTPUT_RANKING, index=False)
    print(f"Ranking saved to {OUTPUT_RANKING}")
    
    # Histogram/KDE of scores
    plt.figure(figsize=(8,5))
    sns.kdeplot(risk_scores, fill=True, color='crimson')
    plt.title("Density of Risk Scores")
    plt.xlabel("Risk Probability")
    plt.savefig("risk_score_dist.png")
    
    # Viz Top Molecules
    from rdkit.Chem import Draw
    mols = [Chem.MolFromSmiles(s) for s in top_risky['SMILES']]
    # Remove None mols if any
    valid_mols = [m for m in mols if m]
    valid_ids = [did for m, did in zip(mols, top_risky['DTXSID']) if m]
    
    if valid_mols:
        img = Draw.MolsToGridImage(valid_mols[:15], molsPerRow=5, subImgSize=(300, 300), 
                                   legends=[f"{i}\nScore: {s:.2f}" for i, s in zip(valid_ids[:15], top_risky['risk_score'][:15])])
        img.save("top_risky_structures.png")
        print("Molecules grid saved to top_risky_structures.png")

if __name__ == "__main__":
    main()
