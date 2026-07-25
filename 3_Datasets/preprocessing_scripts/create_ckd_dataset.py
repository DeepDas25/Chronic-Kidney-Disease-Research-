import pandas as pd
import numpy as np

import os

# Dynamic path resolution for raw and processed datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.dirname(script_dir)
raw_dir = os.path.join(datasets_dir, "raw") if os.path.exists(os.path.join(datasets_dir, "raw")) else script_dir

def get_raw_path(filename):
    p = os.path.join(raw_dir, filename)
    return p if os.path.exists(p) else filename

# Load files
demo   = pd.read_sas(get_raw_path("DEMO_L.xpt"))
biopro = pd.read_sas(get_raw_path("BIOPRO_L.xpt"))
alb    = pd.read_sas(get_raw_path("ALB_CR_L.xpt"))
bmx    = pd.read_sas(get_raw_path("BMX_L.xpt"))
bpxo   = pd.read_sas(get_raw_path("BPXO_L.xpt"))
diq    = pd.read_sas(get_raw_path("DIQ_L.xpt"))

# Merge (Using SEQN as key)
df = demo.merge(biopro, on="SEQN", how="inner")
df = df.merge(alb, on="SEQN", how="inner")
df = df.merge(bmx, on="SEQN", how="left")
df = df.merge(bpxo, on="SEQN", how="left")
df = df.merge(diq, on="SEQN", how="left")

# Keep adults only (Medical research standard for CKD-EPI)
df = df[df["RIDAGEYR"] >= 18].copy()

# ============================================================
# CORRECTED: Albumin-Creatinine Ratio (ACR) Clinical Formula (mg/g)
# ============================================================
# NHANES: URXUMA is ug/mL (mg/L), URXUCR is mg/dL.
# (URXUMA / (URXUCR * 10)) * 1000 simplifies mathematically to multiplying by 100.
df["ACR"] = (df["URXUMA"] / df["URXUCR"]) * 100

# ============================================================
# CKD-EPI 2021 Equation for eGFR (Refreshed without Race Parameter)
# ============================================================
def calculate_egfr(scr, age, sex):
    if pd.isna(scr) or pd.isna(age) or pd.isna(sex) or scr <= 0:
        return np.nan

    if sex == 2:  # Female
        k = 0.7
        alpha = -0.241
        sex_factor = 1.012
    else:         # Male
        k = 0.9
        alpha = -0.302
        sex_factor = 1.0

    egfr = (
        142
        * (min(scr / k, 1) ** alpha)
        * (max(scr / k, 1) ** (-1.200))
        * (0.9938 ** age)
        * sex_factor
    )
    return egfr

df["eGFR"] = df.apply(
    lambda row: calculate_egfr(
        row["LBXSCR"],
        row["RIDAGEYR"],
        row["RIAGENDR"]
    ),
    axis=1
)

# ============================================================
# CKD Label Creation (eGFR < 60 or ACR >= 30 implies Kidney Damage)
# ============================================================
# Drop those records where both clinical markers cannot be determined
df = df.dropna(subset=["eGFR", "ACR"])

df["CKD"] = np.where(
    (df["eGFR"] < 60) | (df["ACR"] >= 30),
    1,
    0
)

print("="*60)
print("FIXED DATASET SHAPE:", df.shape)
print("="*60)
print("\nCorrected CKD Distribution:")
print(df["CKD"].value_counts())

# Save the corrected base file
processed_dir = os.path.join(datasets_dir, "processed") if os.path.exists(os.path.join(datasets_dir, "processed")) else script_dir
out_file = os.path.join(processed_dir, "ckd_nhanes.csv")
df.to_csv(out_file, index=False)
print(f"\n[✓] Saved corrected file: {out_file}")