import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.cluster import DBSCAN, KMeans
import os

# Configuration
EMBED_FILE = "embeddings_pfas.npy"
META_FILE = "embeddings_meta.csv"
OUTPUT_PLOT = "pfas_clusters_umap.png"

def main():
    if not os.path.exists(EMBED_FILE):
        print("Embeddings file not found.")
        return

    print("Loading embeddings...")
    embeddings = np.load(EMBED_FILE)
    df_meta = pd.read_csv(META_FILE)
    
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # UMAP Reduction
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    df_meta['umap_1'] = embedding_2d[:, 0]
    df_meta['umap_2'] = embedding_2d[:, 1]
    
    # Clustering (DBSCAN)
    print("Clustering...")
    # Adjust eps based on scale of UMAP
    clusterer = DBSCAN(eps=0.5, min_samples=10)
    df_meta['cluster'] = clusterer.fit_predict(embedding_2d)
    
    n_clusters = df_meta['cluster'].nunique()
    print(f"Found {n_clusters} clusters (including noise -1)")
    
    # Plot - Refined Aesthetics
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_meta,
        x='umap_1', 
        y='umap_2', 
        hue='cluster',
        palette='turbo', # Better palette for many clusters
        s=15,
        alpha=0.6,
        legend='full',
        edgecolor=None
    )
    plt.title(f"PFAS Chemical Space (UMAP)\nClusters: {n_clusters}", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved to {OUTPUT_PLOT}")
    
    # Save meta with clusters
    df_meta.to_csv("pfas_clusters.csv", index=False)
    
    # Save Cluster Stats
    stats = df_meta['cluster'].value_counts().reset_index()
    stats.columns = ['cluster', 'count']
    stats.to_csv("cluster_stats.csv", index=False)
    print("Cluster assignments and stats saved.")

if __name__ == "__main__":
    main()
