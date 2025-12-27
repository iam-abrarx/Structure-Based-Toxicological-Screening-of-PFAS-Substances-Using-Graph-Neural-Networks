import os
import subprocess
import sys
import time
from datetime import datetime

# Orchestration Script for PFAS Project Reset & Re-Run
# Covers Approaches 1-9

LOG_FILE = "pipeline_execution.log"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

def run_step(description, command, cwd):
    log(f"Starting: {description}")
    start_time = time.time()
    try:
        # Run command
        if sys.platform == "win32":
            shell_cmd = True
        else:
            shell_cmd = False
            
        subprocess.check_call(command, cwd=cwd, shell=shell_cmd)
        elapsed = time.time() - start_time
        log(f"Completed: {description} (Time: {elapsed:.2f}s)")
        return True
    except subprocess.CalledProcessError as e:
        log(f"FAILED: {description} (Exit Code: {e.returncode})")
        return False

def main():
    log("=== STARTING FULL PROJECT RE-RUN ===")
    
    # 1. Approach 1: GNN Pretraining
    if not run_step("App 1: GNN Pretraining (75 Epochs)", ["python", "pretrain_gnn.py"], cwd="approach1_gnn"):
        return
        
    if not run_step("App 1: Component Analysis", ["python", "analyze_embeddings.py"], cwd="approach1_gnn"):
        return

    # 2. Approach 2: Predictive Modeling
    if not run_step("App 2: Train Predictive Model", ["python", "train_tox_model.py"], cwd="approach2_tox_pred"):
        return

    # 3. Approach 3: Risk Ranking
    # ALREADY COMPLETED - SKIPPING (Wait, app 3 failed warning earlier? No, it said warning but proceeded. Let's re-run 3-9 to be safe, or just 5-9. App 3 relies on App 2 models. App 2 finished. App 3 creates files. Let's re-run 3-9.)
    
    # Actually, App 3 failed? log said "Warning: Approach 3 failed". Why? Maybe missing files too.
    # Safe bet: Re-run App 3 onwards.
    
    # 3. Approach 3: Risk Ranking
    if not run_step("App 3: Master Risk Ranking", ["python", "rank_pfas_risk.py"], cwd="approach3_risk"):
         print("Warning: Approach 3 failed, proceeding...")

    # 4. Approach 4: Generative Design
    if not run_step("App 4: Generative AI", ["python", "generate_safer_pfas.py"], cwd="approach4_gen"):
        print("Warning: Approach 4 failed, proceeding...")

    # 5. Approach 5: Tox21 Analysis (Robust)
    # CWD: approach5_tox21
    
    # RECOVERY: Re-download raw data (deleted by cleanup)
    if not run_step("App 5: Download Raw Data", ["python", "scrape_pubchem.py"], cwd="approach5_tox21"):
        return

    # Missing Data Processing Steps added back
    if not run_step("App 5: Process Raw Data", ["python", "process_tox21.py"], cwd="approach5_tox21"):
        return
        
    if not run_step("App 5: Attach Features", ["python", "attach_features_tox21.py"], cwd="approach5_tox21"):
        return

    if not run_step("App 5: Split Data", ["python", "split_data.py"], cwd="approach5_tox21"):
        return

    if not run_step("App 5: Train Robust Tox21 Models", ["python", "train_tox21_robust.py"], cwd="approach5_tox21"):
        return
        
    # App 5: Detailed Evaluation
    if not run_step("App 5: Final Evaluation", ["python", "train_tox21_final.py"], cwd="approach5_tox21"):
        return

    # App 5: Biological Analysis
    if not run_step("App 5: Bio Analysis", ["python", "analyze_pparg.py"], cwd="approach5_tox21"):
        return
        
    # App 5: Risk Propagation
    if not run_step("App 5: Risk Propagation", ["python", "propagate_risk.py"], cwd="approach5_tox21"):
        return

    # 6. Real World Risk
    # CWD: internal root
    if not run_step("Risk: Real World Integration", ["python", "calculate_real_world_risk.py"], cwd="."):
        return

    # 7. Phase 9: Uncertainty
    # CWD: approach5_tox21 (where script resides)
    if not run_step("Phase 9: Regulatory Uncertainty", ["python", "quantify_regulatory_uncertainty.py"], cwd="approach5_tox21"):
        return

    log("=== FULL PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
