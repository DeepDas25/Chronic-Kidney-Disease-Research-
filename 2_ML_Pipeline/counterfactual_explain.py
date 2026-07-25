import os
def _find_data_file(filename):
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        for sub in [os.path.join("3_Datasets", "experiment_splits"), os.path.join("3_Datasets", "processed"), ""]:
            cand = os.path.join(current, sub, filename)
            if os.path.exists(cand):
                return cand
        current = os.path.dirname(current)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # -------------------------------------------------------------------------
    # 1. Load Datasets
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("[INFO] Loading datasets...")
    print("=" * 60)
    X_train = pd.read_csv(_find_data_file("X_train_experiment.csv"))
    X_test = pd.read_csv(_find_data_file("X_test_experiment.csv"))
    y_train = pd.read_csv(_find_data_file("y_train_experiment.csv")).squeeze()
    y_test = pd.read_csv(_find_data_file("y_test_experiment.csv")).squeeze()
    
    # Load original raw dataset to extract mean/std for de-scaling
    raw_df = pd.read_csv(_find_data_file("ckd_nhanes.csv"))
    
    # Consolidate raw blood pressure columns exactly as clean_data.py does
    systolic_cols = [c for c in raw_df.columns if c.startswith("BPXOSY")]
    diastolic_cols = [c for c in raw_df.columns if c.startswith("BPXODI")]
    if systolic_cols:
        raw_df["BPX_SYS_MEAN"] = raw_df[systolic_cols].mean(axis=1)
    if diastolic_cols:
        raw_df["BPX_DIA_MEAN"] = raw_df[diastolic_cols].mean(axis=1)
        
    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 2. Fit Best Model (CatBoost Classifier)
    # -------------------------------------------------------------------------
    print("\nTraining CatBoost Classifier (Tuned)...")
    model = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_state=42, verbose=0)
    model.fit(X_train, y_train)
    print("[OK] Model training complete.")
    
    # -------------------------------------------------------------------------
    # 3. Define Modifiable Features and Descaling Statistics
    # -------------------------------------------------------------------------
    # Replicate train_test_split and imputation to get scaling parameters from training split
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    
    X = raw_df.drop(columns=["CKD"])
    y = raw_df["CKD"]
    X_train_raw, _, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Impute modifiable columns to find exact mean and std used by StandardScaler
    imputer = SimpleImputer(strategy='median')
    cols_to_impute = ["LBXSBU", "BPX_SYS_MEAN", "LBXSUA", "LBXSGL"]
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_raw[cols_to_impute]), columns=cols_to_impute)
    
    modifiable_features = {
        "LBXSBU": {
            "name": "Blood Urea Nitrogen (BUN)",
            "unit": "mg/dL",
            "mean": X_train_imputed["LBXSBU"].mean(),
            "std": X_train_imputed["LBXSBU"].std(ddof=0),
            "step": 1.0,
            "max_reduction": 20.0
        },
        "BPX_SYS_MEAN": {
            "name": "Systolic Blood Pressure",
            "unit": "mmHg",
            "mean": X_train_imputed["BPX_SYS_MEAN"].mean(),
            "std": X_train_imputed["BPX_SYS_MEAN"].std(ddof=0),
            "step": 2.0,
            "max_reduction": 40.0
        },
        "LBXSUA": {
            "name": "Uric Acid",
            "unit": "mg/dL",
            "mean": X_train_imputed["LBXSUA"].mean(),
            "std": X_train_imputed["LBXSUA"].std(ddof=0),
            "step": 0.2,
            "max_reduction": 4.0
        },
        "LBXSGL": {
            "name": "Blood Glucose",
            "unit": "mg/dL",
            "mean": X_train_imputed["LBXSGL"].mean(),
            "std": X_train_imputed["LBXSGL"].std(ddof=0),
            "step": 5.0,
            "max_reduction": 60.0
        }
    }
    
    # -------------------------------------------------------------------------
    # 4. Search Counterfactuals for High-Risk Patients
    # -------------------------------------------------------------------------
    # Predict probabilities on test set
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Find patients predicted as CKD=1 with high confidence (>0.70)
    high_risk_indices = np.where((test_probs >= 0.70) & (y_test == 1))[0]
    print(f"\n[INFO] Found {len(high_risk_indices)} high-risk true positive patients in test set.")
    
    # Select a sample of 5 high-risk patients to explain
    sample_indices = high_risk_indices[:5]
    
    recommendations = []
    
    print("\n" + "="*80)
    print("COUNTERFACTUAL CLINICAL RECOMMENDATIONS FOR HIGH-RISK PATIENTS")
    print("="*80)
    
    for idx in sample_indices:
        patient_vector = X_test.iloc[idx].copy()
        orig_prob = test_probs[idx]
        
        # Descaled patient modifiable features
        patient_orig_vals = {}
        for feat, meta in modifiable_features.items():
            scaled_val = patient_vector[feat]
            orig_val = scaled_val * meta["std"] + meta["mean"]
            patient_orig_vals[feat] = orig_val
            
        # We search for the optimal reduction combination
        # Let's perform a grid search on reductions
        best_cost = float('inf')
        best_reductions = None
        best_new_prob = None
        
        # Build grid coordinates
        feat_keys = list(modifiable_features.keys())
        r_bu = np.arange(0, modifiable_features["LBXSBU"]["max_reduction"] + 1, modifiable_features["LBXSBU"]["step"])
        r_sys = np.arange(0, modifiable_features["BPX_SYS_MEAN"]["max_reduction"] + 1, modifiable_features["BPX_SYS_MEAN"]["step"])
        r_ua = np.arange(0, modifiable_features["LBXSUA"]["max_reduction"] + 1, modifiable_features["LBXSUA"]["step"])
        r_gl = np.arange(0, modifiable_features["LBXSGL"]["max_reduction"] + 1, modifiable_features["LBXSGL"]["step"])
        
        found = False
        
        for d_bu in r_bu:
            for d_sys in r_sys:
                for d_ua in r_ua:
                    for d_gl in r_gl:
                        # Skip no-change scenario
                        if d_bu == 0 and d_sys == 0 and d_ua == 0 and d_gl == 0:
                            continue
                            
                        # Calculate cost (normalized L1 distance of reductions)
                        cost = (d_bu / 5.0) + (d_sys / 10.0) + (d_ua / 1.0) + (d_gl / 20.0)
                        
                        if cost >= best_cost:
                            continue
                            
                        # Apply reduction in original units
                        new_bu_orig = max(0.0, patient_orig_vals["LBXSBU"] - d_bu)
                        new_sys_orig = max(0.0, patient_orig_vals["BPX_SYS_MEAN"] - d_sys)
                        new_ua_orig = max(0.0, patient_orig_vals["LBXSUA"] - d_ua)
                        new_gl_orig = max(0.0, patient_orig_vals["LBXSGL"] - d_gl)
                        
                        # Scale back to standard units
                        temp_patient = patient_vector.copy()
                        temp_patient["LBXSBU"] = (new_bu_orig - modifiable_features["LBXSBU"]["mean"]) / modifiable_features["LBXSBU"]["std"]
                        temp_patient["BPX_SYS_MEAN"] = (new_sys_orig - modifiable_features["BPX_SYS_MEAN"]["mean"]) / modifiable_features["BPX_SYS_MEAN"]["std"]
                        temp_patient["LBXSUA"] = (new_ua_orig - modifiable_features["LBXSUA"]["mean"]) / modifiable_features["LBXSUA"]["std"]
                        temp_patient["LBXSGL"] = (new_gl_orig - modifiable_features["LBXSGL"]["mean"]) / modifiable_features["LBXSGL"]["std"]
                        
                        # Get model prediction probability
                        prob = model.predict_proba(temp_patient.values.reshape(1, -1))[0, 1]
                        
                        # We want prediction to flip to low risk (<0.35 threshold)
                        if prob < 0.35:
                            best_cost = cost
                            best_reductions = {"LBXSBU": d_bu, "BPX_SYS_MEAN": d_sys, "LBXSUA": d_ua, "LBXSGL": d_gl}
                            best_new_prob = prob
                            found = True
                            
        # Print results
        print(f"\nPatient ID (Test Index: {idx})")
        print(f"  * Current CKD Risk Probability: {orig_prob*100:.2f}% (High-Risk Classified)")
        print(f"  * Baseline Modifiable Features:")
        for feat, meta in modifiable_features.items():
            print(f"    - {meta['name']}: {patient_orig_vals[feat]:.2f} {meta['unit']}")
            
        if found:
            print(f"  * Counterfactual Actionable Path (Optimized Clinical Target):")
            rec_str = []
            for feat, meta in modifiable_features.items():
                reduction = best_reductions[feat]
                if reduction > 0:
                    new_val = patient_orig_vals[feat] - reduction
                    print(f"    - Reduce {meta['name']} by {reduction:.1f} {meta['unit']} (Target: {new_val:.1f} {meta['unit']})")
                    rec_str.append(f"Reduce {meta['name']} by {reduction:.1f} {meta['unit']} to {new_val:.1f} {meta['unit']}")
            print(f"  * Target CKD Risk Probability: {best_new_prob*100:.2f}% (Low-Risk Classified)")
            
            recommendations.append({
                "Test_Patient_Index": idx,
                "Original_Probability": orig_prob,
                "BUN_Original": patient_orig_vals["LBXSBU"],
                "BUN_Reduction": best_reductions["LBXSBU"],
                "SBP_Original": patient_orig_vals["BPX_SYS_MEAN"],
                "SBP_Reduction": best_reductions["BPX_SYS_MEAN"],
                "UricAcid_Original": patient_orig_vals["LBXSUA"],
                "UricAcid_Reduction": best_reductions["LBXSUA"],
                "Glucose_Original": patient_orig_vals["LBXSGL"],
                "Glucose_Reduction": best_reductions["LBXSGL"],
                "Target_Probability": best_new_prob,
                "Recommendation_Text": " | ".join(rec_str)
            })
        else:
            print("  * No counterfactual path found within max allowed reductions.")
            recommendations.append({
                "Test_Patient_Index": idx,
                "Original_Probability": orig_prob,
                "BUN_Original": patient_orig_vals["LBXSBU"],
                "BUN_Reduction": 0.0,
                "SBP_Original": patient_orig_vals["BPX_SYS_MEAN"],
                "SBP_Reduction": 0.0,
                "UricAcid_Original": patient_orig_vals["LBXSUA"],
                "UricAcid_Reduction": 0.0,
                "Glucose_Original": patient_orig_vals["LBXSGL"],
                "Glucose_Reduction": 0.0,
                "Target_Probability": orig_prob,
                "Recommendation_Text": "No path found within limits"
            })
            
    # Save to CSV
    recs_df = pd.DataFrame(recommendations)
    out_csv = os.path.join(script_dir, "Generated_Outputs", "counterfactual_recommendations.csv")
    recs_df.to_csv(out_csv, index=False)
    print("\n" + "="*80)
    print(f"[SUCCESS] Saved counterfactual recommendations log to: {out_csv}")
    print("="*80)

if __name__ == "__main__":
    main()
