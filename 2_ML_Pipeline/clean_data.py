# ============================================================
# PHASE 3: DATA CLEANING & PREPROCESSING (BUG FIXED)
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

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
processed_dir = os.path.join(root_dir, "3_Datasets", "processed")
splits_dir = os.path.join(root_dir, "3_Datasets", "experiment_splits")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

input_csv = _find_data_file("ckd_nhanes.csv")
if not os.path.exists(input_csv):
    input_csv = _find_data_file("ckd_nhanes.csv")

df = pd.read_csv(input_csv)

print("="*60)
print("ORIGINAL DATASET SHAPE:", df.shape)
print("="*60)

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

print("[OK] Consolidated repeat exam measurements (BP and Pulse).")

# ============================================================
# STEP 1 & 2: DROP IDENTIFIERS AND LEAKAGE FEATURES
# ============================================================

leakage_and_ids = [

    # Identifier
    "SEQN",

    # Derived CKD variables
    "eGFR",
    "ACR",

    # Creatinine variables used in eGFR
    "LBXSCR",
    "LBDSCRSI",

    # Albumin variables used in ACR
    "URXUMA",
    "URXUMS",

    # Urine Creatinine variables used in ACR
    "URXUCR",
    "URDACT",

    # SI-unit version of Urine Creatinine
    "URXCRS",

    # Albumin / Creatinine detection flags
    "URDUMALC",
    "URDUCRLC",

    # NHANES Survey Design Variables
    "WTMEC2YR",
    "WTINT2YR",
    "WTPH2YR",
    "SDMVPSU",
    "SDMVSTRA",
    "SDDSRVYR",

    # Collinear duplicate SI units (redundant with standard units)
    "LBDSALSI",
    "LBDSBUSI",
    "LBDSGBSI",
    "LBDSGLSI",
    "LBDSIRSI",
    "LBDSPHSI",
    "LBDSTBSI",
    "LBDSCHSI",
    "LBDSTPSI",
    "LBDSTRSI",
    "LBDSUASI",

    # Administrative, zero-variance, and procedural (Clinically Irrelevant)
    "RIDSTATR",
    "RIDEXMON",
    "BPAOARM",
    "LBDSGTLC",
    "LBDSTBLC",
    "BMDSTATS"
]

df.drop(
    columns=leakage_and_ids,
    inplace=True,
    errors="ignore"
)

print("\n[OK] Leakage & Identifier Features Removed.")

# ============================================================
# STEP 3: DROP HIGHLY MISSING COLUMNS (Threshold: 40%)
# ============================================================
missing_percent = (df.isnull().sum() / len(df)) * 100
high_missing_cols = missing_percent[missing_percent > 40].index.tolist()

print(f"\n[!] Removing Columns with >40% Missing Values ({len(high_missing_cols)} columns):")
print(high_missing_cols)

df.drop(columns=high_missing_cols, inplace=True, errors="ignore")

# ============================================================
# STEP 4: SEPARATE FEATURES & TARGET
# ============================================================
X = df.drop(columns=["CKD"])
y = df["CKD"]

# ============================================================
# STEP 5: IDENTIFY NUMERICAL & CATEGORICAL COLUMNS (FOOLPROOF)
# ============================================================
# Explicitly define known categorical columns in the dataset
known_categorical = ['RIAGENDR', 'RIDRETH1', 'DMQMILIZ', 'DMDBORN4', 'DMDEDUC2', 'DMDMARTZ', 'DIQ010', 'DIQ160', 'DIQ180']
categorical_cols = [col for col in known_categorical if col in X.columns]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

print(f"\nNumerical Features discovered: {len(numerical_cols)}")
print(f"Categorical Features discovered: {len(categorical_cols)}")

# ============================================================
# CRITICAL FIX: SPLIT FIRST TO PREVENT DATA LEAKAGE
# ============================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\n[OK] Initial Stratified Split Completed.")

# ============================================================
# STEP 6: BUILD ROBUST TRANSFORMERS
# ============================================================
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Option B: Mode imputation only, NO scaling, NO one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# ============================================================
# STEP 7: FIT ONLY ON TRAIN, TRANSFORM BOTH
# ============================================================
# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train_raw)

# ONLY transform the testing data (Leakage strictly prevented!)
X_test_processed = preprocessor.transform(X_test_raw)

# Extract correct feature names (no One-Hot Encoding applied in Option B)
all_features = numerical_cols + categorical_cols

# Convert back to beautiful DataFrames for models
X_train = pd.DataFrame(X_train_processed, columns=all_features)
X_test = pd.DataFrame(X_test_processed, columns=all_features)

print("[OK] Scaling and Imputation isolated to Training Split successfully.")

# ============================================================
# AUTOMATED FEATURE SELECTION (CORRELATION FILTER)
# ============================================================
# Identify and remove features with pairwise correlation > 0.85 on training set
corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]

print(f"\n[!] Removing highly collinear features (correlation > 0.85) ({len(to_drop)} columns):")
print(to_drop)

X_train.drop(columns=to_drop, inplace=True)
X_test.drop(columns=to_drop, inplace=True)

# ============================================================
# SAVE CLEANED SPLITS FOR TRAINING (Maintains state consistency)
# ============================================================
# Save combined train and test files for backup if needed
train_clean_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
test_clean_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

# Combined full processed set for tracking/correlation
full_clean_df = pd.concat([train_clean_df, test_clean_df], axis=0, ignore_index=True)
full_clean_df.to_csv(_find_data_file("ckd_nhanes_clean_experiment.csv"), index=False)
full_clean_df.to_csv(_find_data_file("ckd_nhanes_clean_experiment.csv"), index=False)

# ============================================================
# DATASET SHAPE VERIFICATION LOGS
# ============================================================
print("\n" + "="*60)
print("FINAL DATASET AND SHAPE VERIFICATION")
print("="*60)
print(f"Training Splits (X_train, y_train)       : {X_train.shape}, {y_train.shape}")
print(f"Testing Splits (X_test, y_test)         : {X_test.shape}, {y_test.shape}")

print("\n" + "-"*40)
print("Class Distribution in Training Set:")
print(y_train.value_counts())
print("\nClass Distribution in Testing Set:")
print(y_test.value_counts())
print("-"*40)

print("\nTOP 15 FEATURES CORRELATED WITH CKD (On Cleaned Set)")
print("="*60)
train_corr_df = pd.concat(
    [X_train, y_train.reset_index(drop=True)],
    axis=1
)
corr = train_corr_df.corr(numeric_only=True)
print(corr["CKD"].abs().sort_values(ascending=False).head(15))

print("\n" + "="*60)
print("[OK] SAFE CLEAN DATASET PROCESSED & VERIFIED")
print("="*60)

# Save exact train-test splits to experiment_splits directory
def save_split(df_obj, fname):
    df_obj.to_csv(os.path.join(splits_dir, fname), index=False)
    # Also keep copy in script_dir if needed for backward compatibility
    df_obj.to_csv(os.path.join(script_dir, fname), index=False)

save_split(X_train, "X_train_experiment.csv")
save_split(X_test, "X_test_experiment.csv")
save_split(y_train, "y_train_experiment.csv")
save_split(y_test, "y_test_experiment.csv")
save_split(train_clean_df, "train_clean_df_experiment.csv")
save_split(test_clean_df, "test_clean_df_experiment.csv")

print("\n[OK] Train/Test Splits Saved Successfully")

print("\nWT COLUMNS FOUND:")
print([col for col in df.columns if col.startswith("WT")])