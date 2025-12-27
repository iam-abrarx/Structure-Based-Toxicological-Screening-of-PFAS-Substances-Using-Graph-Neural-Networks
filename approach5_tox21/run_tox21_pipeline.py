import subprocess
import os
import sys
import time

def run_step(script_name, description):
    print(f"\n{'='*50}")
    print(f"Running Step: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    try:
        # Use python from current environment
        cmd = [sys.executable, script_name]
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"\n[SUCCESS] {description} completed in {duration:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed with error code {e.returncode}.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to run {script_name}: {e}")
        sys.exit(1)

def main():
    print("Starting Tox21 PFAS Pipeline Execution...")
    
    # Check if process_tox21 is already generating output, 
    # but strictly we should run it.
    # Note: process_tox21.py might skip downloading if files exist, 
    # and skip API fetching if cache exists.
    
    # 1. Processing & Mapping
    run_step("process_tox21.py", "Data Processing & Mapping")
    
    # 2. Feature Attachment
    run_step("attach_features_tox21.py", "Feature Engineering (RDKit + GNN)")
    
    # 3. Splitting
    run_step("split_data.py", "Scaffold Train/Test Splitting")
    
    # 4. Modeling
    run_step("train_tox21_robust.py", "Model Training & Evaluation")
    
    print("\n\nPipeline Finished Successfully!")
    print("Results are in 'processed_tox21/results'")

if __name__ == "__main__":
    main()
