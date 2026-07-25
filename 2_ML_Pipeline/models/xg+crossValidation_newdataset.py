# ============================================================
# XGBOOST 5-FOLD STRATIFIED LEAKAGE-FREE CROSS VALIDATION
# ============================================================

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
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load raw dataset (relative to script directory)
raw_csv_path = _find_data_file("ckd_nhanes.csv")
if not os.path.exists(raw_csv_path):
    raw_csv_path = os.path.join(script_dir, "../../../NHANES/ckd_nhanes.csv")

df = pd.read_csv(raw_csv_path)

# Average repeat exam measurements (Systolic/Diastolic BP and Pulse)
systolic_cols = [c for c in df.columns if c.startswith("BPXOSY")]
diastolic_cols = [c for c in df.columns if c.startswith("BPXODI")]
pulse_cols = [c for c in df.columns if c.startswith("BPXOPLS")]

if systolic_cols:
    df["BPX_SYS_MEAN"] = df[systolic_cols].mean(axis=1)
    df.drop(columns=systolic_cols, inplace=True)
if diastolic_cols:
    df["BPX_DIA_MEAN"] = df[diastolic_cols].mean(axis=1)
    df.drop(columns=diastolic_cols, inplace=True)
if pulse_cols:
    df["BPX_PLS_MEAN"] = df[pulse_cols].mean(axis=1)
    df.drop(columns=pulse_cols, inplace=True)

# Drop identifiers, leakage, and redundant columns
leakage_and_ids = [
    "SEQN", "eGFR", "ACR", "LBXSCR", "LBDSCRSI", "URXUMA", "URXUMS",
    "URXUCR", "URDACT", "URXCRS", "URDUMALC", "URDUCRLC",
    "WTMEC2YR", "WTINT2YR", "WTPH2YR", "SDMVPSU", "SDMVSTRA", "SDDSRVYR",
    "LBDSALSI", "LBDSBUSI", "LBDSGBSI", "LBDSGLSI", "LBDSIRSI", "LBDSPHSI",
    "LBDSTBSI", "LBDSCHSI", "LBDSTPSI", "LBDSTRSI", "LBDSUASI",
    "RIDSTATR", "RIDEXMON", "BPAOARM", "LBDSGTLC", "LBDSTBLC", "BMDSTATS"
]
df.drop(columns=leakage_and_ids, inplace=True, errors="ignore")

# Drop highly missing columns (>40% missingness)
missing_percent = (df.isnull().sum() / len(df)) * 100
high_missing_cols = missing_percent[missing_percent > 40].index.tolist()
df.drop(columns=high_missing_cols, inplace=True, errors="ignore")

# Drop highly collinear features (as identified by clean_data.py correlation filter)
collinear_cols = ['RIDRETH3', 'LBDSCASI', 'BMXBMI', 'BMXARMC', 'BMXWAIST', 'BMXHIP', 'BPAOCSZ']
df.drop(columns=collinear_cols, inplace=True, errors="ignore")


# Separate features and target
X = df.drop(columns=["CKD"])
y = df["CKD"]

# Identify numerical and categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("="*60)
print("LEAKAGE-FREE DATASET SHAPE")
print("="*60)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Build preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Create a unified pipeline wrapping preprocessing and the classifier
cv_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=300,
        random_state=42,
        eval_metric="logloss"
    ))
])

# Run 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

print("\n[OK] Running Stratified 5-Fold Leakage-Free Cross-Validation...")

scores = cross_validate(
    cv_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

print("\n" + "="*70)
print("XGBOOST 5-FOLD LEAKAGE-FREE CV RESULTS")
print("="*70)

for metric in scoring.keys():
    mean_score = scores[f"test_{metric}"].mean()
    std_score = scores[f"test_{metric}"].std()
    print(
        f"{metric.upper():<12}: "
        f"{mean_score:.4f} ± {std_score:.4f}"
    )
print("="*70)