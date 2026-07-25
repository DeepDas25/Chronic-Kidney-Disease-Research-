import os
import subprocess
import shutil
import sys

def run_script(script_path, cwd):
    print(f"\n[ORCHESTRATOR] Running: {script_path} in {cwd}...")
    # Run using sys.executable to ensure we use the same Python interpreter
    result = subprocess.run([sys.executable, script_path], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed running {script_path}!")
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        raise RuntimeError(f"Pipeline step failed: {script_path}")
    else:
        print(f"[SUCCESS] Completed: {script_path}")
        # Optionally show short output summary
        lines = result.stdout.strip().split("\n")
        if lines:
            print(f"Summary: {lines[-1]}")

def main():
    # Set script dir as base working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("=" * 80)
    print("NHANES CHRONIC KIDNEY DISEASE ML PIPELINE ORCHESTRATOR")
    print("=" * 80)
    print(f"Base Directory: {script_dir}")
    
    # 1. Clean Data & Create Splits
    run_script("clean_data.py", script_dir)
    
    # 2. EDA Analysis & Figures
    run_script("generate_eda_and_analysis_final.py", script_dir)
    
    # 3. Baseline Classifier Evaluation
    run_script("result_clean.py", script_dir)
    
    # 4. Hyperparameter Tuning
    run_script("tune_top_models.py", script_dir)
    
    # 5. ROC Curve Generation
    run_script("plot_roc_curves.py", script_dir)
    
    # 6. SHAP Interpretability Generation
    run_script("catBoost_newclean.py", script_dir)
    
    # 7. Actionable Clinical Counterfactuals
    run_script("counterfactual_explain.py", script_dir)
    
    # 8. Mixture & Ensemble Models
    mixture_dir = os.path.join(script_dir, "ensemble_mixture")
    run_script("clinical_cascade.py", mixture_dir)
    run_script("hard_voting.py", mixture_dir)
    run_script("soft_voting.py", mixture_dir)
    run_script("weighted_soft_voting.py", mixture_dir)
    run_script("stacking_ensemble.py", mixture_dir)
    run_script("tree_only_blend.py", mixture_dir)
    
    # 9. Jupyter Notebook Re-Generation
    run_script("create_ipynb_v2.py", script_dir)
    
    # =========================================================================
    # Aggregating and Syncing Outputs to outputs/
    # =========================================================================
    print("\n" + "=" * 80)
    print("[ORCHESTRATOR] Aggregating all output artifacts into outputs/...")
    print("=" * 80)
    
    gen_dir = os.path.join(script_dir, "outputs")
    os.makedirs(gen_dir, exist_ok=True)
    
    # Define files in MAIN and their desired names or copy targets
    main_files = [
        # EDA outputs
        "EDA_01_CKD_Class_Distribution.png",
        "EDA_02_Age_Distribution_CKD.png",
        "EDA_03_Diabetes_vs_CKD.png",
        "EDA_04_BUN_Boxplot.png",
        "EDA_05_UricAcid_Violin.png",
        "EDA_06_SystolicBP_Boxplot.png",
        "EDA_07_SHAP_Correlation_Heatmap.png",
        "EDA_08_Threshold_Tuning.png",
        # Evaluation outputs
        "Final_Model_Comparison_43Features.csv",
        "Accuracy_Comparison_43Features.png",
        "F1_Comparison_43Features.png",
        "Recall_Comparison_43Features.png",
        "ROC_AUC_Comparison_43Features.png",
        "Combined_Model_Comparison_43Features.png",
        # Tuning & ROC outputs
        "tuning_results_summary.txt",
        "ROC_Curves_Top3.png",
        # SHAP outputs from catBoost_newclean.py
        "SHAP_BarPlot_Final43.png",
        "SHAP_SummaryPlot_Final43.png",
        "Top10_SHAP_Features_Final43.png",
        "SHAP_Top20_Features_Final43.csv"
    ]
    
    for fname in main_files:
        src = os.path.join(script_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(gen_dir, fname)
            shutil.copy2(src, dst)
            print(f"Copied to outputs: {fname}")
            # Clean up duplicate file from root directory to keep it tidy
            os.remove(src)
            
    # Define files in Mixture directory to copy
    mixture_files = [
        "cascade_results.csv",
        "hard_voting_results.csv",
        "soft_voting_results.csv",
        "weighted_soft_voting_results.csv",
        "stacking_results.csv",
        "tree_only_blend_results.csv"
    ]
    
    for fname in mixture_files:
        src = os.path.join(mixture_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(gen_dir, fname)
            shutil.copy2(src, dst)
            print(f"Copied to outputs: ensemble_mixture/{fname}")
            # We keep the files in ensemble_mixture/ as well for local consistency
            
    # Counterfactual recommendations is directly written to outputs by the script,
    # let's verify it exists
    cf_path = os.path.join(gen_dir, "counterfactual_recommendations.csv")
    if os.path.exists(cf_path):
        print("Verified counterfactual recommendations in outputs/")
        
    print("\n" + "=" * 80)
    print("[SUCCESS] PIPELINE SYNCED successfully! All outputs reside in outputs/.")
    print("=" * 80)

if __name__ == "__main__":
    main()
