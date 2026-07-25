# generate_eda_and_analysis_final.py
"""Corrected script for CKD EDA and model‑analysis figures.

- Paper EDA uses the original unscaled dataset `ckd_nhanes.csv`.
- Model‑analysis uses the pre‑split cleaned files `X_train_experiment.csv`, `X_test_experiment.csv`,
  `y_train_experiment.csv`, and `y_test_experiment.csv`.
- The final dataset contains 43 features; legacy column names are mapped to the final names:
    * AGE → RIDAGEYR
    * BUN → LBXSBU
    * URIC_ACID → LBXSUA
- Blood‑pressure variables are **computed** from the raw NHANES columns as in the original preprocessing:
    ```python
    systolic_cols = [c for c in raw_df.columns if c.startswith("BPXOSY")]
    diastolic_cols = [c for c in raw_df.columns if c.startswith("BPXODI")]
    pulse_cols = [c for c in raw_df.columns if c.startswith("BPXOPLS")]
    raw_df["BPX_SYS_MEAN"] = raw_df[systolic_cols].mean(axis=1)
    raw_df["BPX_DIA_MEAN"] = raw_df[diastolic_cols].mean(axis=1)
    raw_df["BPX_PLS_MEAN"] = raw_df[pulse_cols].mean(axis=1)
    ```
- Diabetes coding follows NHANES definitions (1=Yes, 2=No, 3=Borderline, 7=Refused, 9=Don't Know).
- Model analysis uses the exact CatBoost hyper‑parameters from the research pipeline.
- All figures are saved under `Generated_Outputs/` with the `_43(new)` suffix; existing files are never overwritten.
"""

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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Plot style – publication quality
plt.rcParams["savefig.dpi"] = 300
sns.set_theme(style="whitegrid", palette="deep", rc={"figure.figsize": (10, 6)})

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Generated_Outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(fig, name):
    """Save figure with `_43(new)` suffix, adding a numeric suffix if the file already exists."""
    filename = OUTPUT_DIR / f"{name}_43(new).png"
    counter = 1
    while filename.exists():
        filename = OUTPUT_DIR / f"{name}_43(new)_{counter}.png"
        counter += 1
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

# ---------------------------------------------------------------------------
# 1. Load original raw data for Paper EDA
# ---------------------------------------------------------------------------
raw_path = _find_data_file("ckd_nhanes.csv")
raw_df = pd.read_csv(raw_path)
# Load final cleaned 43-feature dataset for analysis (train split)
cleaned_df = pd.read_csv(_find_data_file("X_train_experiment.csv"))

# ---------------------------------------------------------------------------
# 2. Apply preprocessing that matches the original pipeline
# ---------------------------------------------------------------------------
# Map legacy column names to final names (where they exist)
rename_map = {
    "AGE": "RIDAGEYR",
    "BUN": "LBXSBU",
    "URIC_ACID": "LBXSUA",
    "LBDSBUSI": None,
    "LBDSUASI": None,
    "LBDSCHSI": None,
    "LBDSIRSI": None,
    "LBDSTRSI": None,
}
for old, new in rename_map.items():
    if old in raw_df.columns:
        if new:
            raw_df = raw_df.rename(columns={old: new})
        else:
            raw_df = raw_df.drop(columns=[old])

# Compute blood‑pressure mean columns from the raw NHANES variables
systolic_cols = [c for c in raw_df.columns if c.startswith("BPXOSY")]
diastolic_cols = [c for c in raw_df.columns if c.startswith("BPXODI")]
pulse_cols = [c for c in raw_df.columns if c.startswith("BPXOPLS")]
if systolic_cols:
    raw_df["BPX_SYS_MEAN"] = raw_df[systolic_cols].mean(axis=1)
if diastolic_cols:
    raw_df["BPX_DIA_MEAN"] = raw_df[diastolic_cols].mean(axis=1)
if pulse_cols:
    raw_df["BPX_PLS_MEAN"] = raw_df[pulse_cols].mean(axis=1)

# CKD label mapping (0 → Non‑CKD, 1 → CKD)
if "CKD" in raw_df.columns:
    raw_df["CKD"] = raw_df["CKD"].map({0: "Non-CKD", 1: "CKD"})

# Diabetes mapping according to NHANES coding
if "DIQ010" in raw_df.columns:
    diabetes_map = {
        1: "Yes",
        2: "No",
        3: "Borderline",
        7: "Refused",
        9: "Don't Know",
    }
    raw_df["DIABETES"] = raw_df["DIQ010"].map(diabetes_map)

# ---------------------------------------------------------------------------
# 3. Paper EDA Figures (original data)
# ---------------------------------------------------------------------------
# 3.1 CKD Class Distribution
fig, ax = plt.subplots()
sns.countplot(data=raw_df, x="CKD", order=["Non-CKD", "CKD"], ax=ax)
ax.set_title("CKD Class Distribution")
ax.set_xlabel("CKD Status")
ax.set_ylabel("Count")
save_fig(fig, "EDA_01_CKD_Class_Distribution")

# 3.2 Age Distribution by CKD Status (RIDAGEYR)
fig, ax = plt.subplots()
sns.violinplot(data=raw_df, x="CKD", y="RIDAGEYR", hue="CKD", split=True, ax=ax)
ax.set_title("Age Distribution by CKD Status")
ax.set_xlabel("CKD Status")
ax.set_ylabel("Age (years)")
save_fig(fig, "EDA_02_Age_Distribution_CKD")

# 3.3 Diabetes vs CKD
if "DIABETES" in raw_df.columns:
    fig, ax = plt.subplots()
    sns.countplot(data=raw_df, x="DIABETES", hue="CKD", ax=ax)
    ax.set_title("Diabetes vs CKD")
    ax.set_xlabel("Diabetes")
    ax.set_ylabel("Count")
    save_fig(fig, "EDA_03_Diabetes_vs_CKD")

# 3.4 Blood Urea Nitrogen Distribution (LBXSBU)
if "LBXSBU" in raw_df.columns:
    fig, ax = plt.subplots()
    sns.boxplot(data=raw_df, y="LBXSBU", ax=ax)
    ax.set_title("Blood Urea Nitrogen Distribution")
    ax.set_ylabel("BUN (mg/dL)")
    save_fig(fig, "EDA_04_BUN_Boxplot")

# 3.5 Uric Acid Distribution (LBXSUA)
if "LBXSUA" in raw_df.columns:
    fig, ax = plt.subplots()
    sns.violinplot(data=raw_df, y="LBXSUA", ax=ax)
    ax.set_title("Uric Acid Distribution")
    ax.set_ylabel("Uric Acid (mg/dL)")
    save_fig(fig, "EDA_05_UricAcid_Violin")

# 3.6 Blood Pressure Distribution (SYS, DIA, PLS)
bp_cols = ["BPX_SYS_MEAN", "BPX_DIA_MEAN", "BPX_PLS_MEAN"]
available_bp = [c for c in bp_cols if c in raw_df.columns]
if available_bp:
    fig, ax = plt.subplots()
    raw_df_melt = raw_df[available_bp].melt(var_name="BP_Type", value_name="Value")
    sns.boxplot(data=raw_df_melt, x="BP_Type", y="Value", ax=ax)
    ax.set_title("Blood Pressure Distribution")
    ax.set_xlabel("Blood Pressure Type")
    ax.set_ylabel("Pressure (mm Hg)")
    save_fig(fig, "EDA_06_BP_Boxplot")

# 3.7 Correlation Heatmap (top 20 pairs among final 43 features)
numeric_df = cleaned_df.select_dtypes(include=[np.number])
# Drop identifier / leakage columns if present
identifier_cols = ["SEQN", "URXUMS", "SDMVSTRA", "SDMVPSU", "BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3", "BPXOPLS1", "BPXOPLS2", "BPXOPLS3"]
numeric_df = numeric_df.drop(columns=[c for c in identifier_cols if c in numeric_df.columns], errors='ignore')
feature_cols = [c for c in numeric_df.columns]
corr = numeric_df[feature_cols].corr().abs()
mask = np.triu(np.ones_like(corr, dtype=bool))
tri_corr = corr.where(~mask)
top_pairs = (
    tri_corr.stack()
    .sort_values(ascending=False)
    .head(20)
    .index
)
pair_mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
for i, j in top_pairs:
    pair_mask.loc[i, j] = True
    pair_mask.loc[j, i] = True
fig, ax = plt.subplots()
sns.heatmap(corr, mask=~pair_mask, cmap="coolwarm", center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
ax.set_title("Correlation Heatmap (Top 20 Pairs)")
save_fig(fig, "EDA_07_Correlation_Heatmap")

# ---------------------------------------------------------------------------
# 4. Load cleaned train/test data and labels for Model Analysis
# ---------------------------------------------------------------------------
X_train_path = _find_data_file("X_train_experiment.csv")
X_test_path = _find_data_file("X_test_experiment.csv")
y_train_path = _find_data_file("y_train_experiment.csv")
y_test_path = _find_data_file("y_test_experiment.csv")

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
# Labels are stored separately; squeeze to obtain a 1‑D array/Series
y_train = pd.read_csv(y_train_path).squeeze()
y_test = pd.read_csv(y_test_path).squeeze()

# ---------------------------------------------------------------------------
# 5. Model Analysis Figures (CatBoost)
# ---------------------------------------------------------------------------
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    auc,
    average_precision_score,
    f1_score,
)

# Train CatBoost with the exact hyper‑parameters from the project
model = CatBoostClassifier(
    iterations=200,
    depth=4,
    learning_rate=0.05,
    random_state=42,
    verbose=0,
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

# 5.1 Threshold Tuning Plot (F1 vs threshold)
thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores = [f1_score(y_test, (proba >= t).astype(int)) for t in thresholds]
fig, ax = plt.subplots()
ax.plot(thresholds, f1_scores, marker='o')
ax.set_xlabel('Probability Threshold')
ax.set_ylabel('F1 Score')
ax.set_title('Threshold Tuning (F1 Score)')
save_fig(fig, "Threshold_Tuning")

# 5.2 Feature Importance Plot (CatBoost)
importances = model.get_feature_importance()
indices = np.argsort(importances)[::-1]
top_n = 15
fig, ax = plt.subplots()
ax.barh(range(top_n), importances[indices[:top_n]][::-1])
ax.set_yticks(range(top_n))
ax.set_yticklabels([X_train.columns[i] for i in indices[:top_n]][::-1])
ax.set_xlabel('Importance')
ax.set_title('Top 15 Feature Importances (CatBoost)')
save_fig(fig, "Feature_Importance")

# 5.3 SHAP Summary Plot (CatBoost)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
shap_fig = plt.gcf()
save_fig(shap_fig, "SHAP_Summary")

# 5.4 ROC Curve
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
save_fig(fig, "ROC_Curve")

# 5.5 Precision‑Recall Curve
precision, recall, _ = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)
fig, ax = plt.subplots()
ax.plot(recall, precision, label=f'AP = {ap:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision‑Recall Curve')
ax.legend(loc='lower left')
save_fig(fig, "Precision_Recall_Curve")

# 5.6 Confusion Matrix (optimal threshold from F1)
optimal_thr = thresholds[np.argmax(f1_scores)]
y_pred = (proba >= optimal_thr).astype(int)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (Threshold={optimal_thr:.2f})')
save_fig(fig, "Confusion_Matrix")

print("All figures generated successfully.")
