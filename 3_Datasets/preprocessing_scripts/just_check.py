import pandas as pd



df = pd.read_csv("ckd_nhanes.csv")


df["ACR"] = (df["URXUMA"] / df["URXUCR"]) * 100

df["CKD"] = (
    (df["eGFR"] < 60) |
    (df["ACR"] >= 30)
).astype(int)

print(df["CKD"].value_counts())
print(df["CKD"].value_counts(normalize=True)*100)