import os
import glob
import shutil

PROJECT_ROOT = "C:/Users/Abrar/Downloads/pfas_datasets"

def delete_patterns(base_dir, patterns):
    for root, dirs, files in os.walk(base_dir):
        # Skip raw data downloads to avoid re-scraping
        if "data_scraping" in root:
            continue
            
        for pattern in patterns:
            for filepath in glob.glob(os.path.join(root, pattern)):
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath}")
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

def main():
    print("--- STARTING HARD CLEANUP ---")
    
    # 1. Delete Models
    delete_patterns(PROJECT_ROOT, ["*.pth", "*.pt", "*.model", "*.pkl"])
    
    # 2. Delete Embeddings & Features
    delete_patterns(PROJECT_ROOT, ["*.npy"])
    
    # 3. Delete Plots
    delete_patterns(PROJECT_ROOT, ["*.png", "*.jpg", "*.svg"])
    
    # 4. Delete Results/CSVs (Be careful not to delete raw input)
    # We will delete specific result files known to be generated
    generated_files = [
        "embeddings_meta.csv",
        "pfas_clusters.csv",
        "cluster_stats.csv",
        "generated_safepfas_candidates.csv",
        "tox21_features_ready.csv",
        "final_global_risk_matrix.csv",
        "all_results_5_approaches.md", # Delete old report if exists
        "pipeline_execution.log"
    ]
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for f in files:
            if f in generated_files or (f.endswith(".csv") and "results" in root) or (f.endswith(".csv") and "processed" in root and "master_chem_table" not in f):
                 # Keep master_chem_table as it is the base
                 file_path = os.path.join(root, f)
                 try:
                     os.remove(file_path)
                     print(f"Deleted Generated CSV: {file_path}")
                 except Exception as e:
                     print(f"Error deleting {file_path}: {e}")

    print("--- CLEANUP COMPLETE ---")

if __name__ == "__main__":
    main()
