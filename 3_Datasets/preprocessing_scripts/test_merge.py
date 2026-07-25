import pandas as pd

# demo = pd.read_sas("DEMO_L.xpt")
# biopro = pd.read_sas("BIOPRO_L.xpt")
# alb = pd.read_sas("ALB_CR_L.xpt")
# bmx = pd.read_sas("BMX_L.xpt")
# bpxo = pd.read_sas("BPXO_L.xpt")
# diq = pd.read_sas("DIQ_L.xpt")

# print("DEMO:", demo.shape)
# print("BIOPRO:", biopro.shape)
# print("ALB_CR:", alb.shape)
# print("BMX:", bmx.shape)
# print("BPXO:", bpxo.shape)
# print("DIQ:", diq.shape)

# df = demo.merge(biopro, on="SEQN", how="inner")
# df = df.merge(alb, on="SEQN", how="inner")
# df = df.merge(bmx, on="SEQN", how="left")
# df = df.merge(bpxo, on="SEQN", how="left")
# df = df.merge(diq, on="SEQN", how="left")

# print("\nMerged Shape:", df.shape)

# print("\nFirst 50 Columns:")
# print(df.columns[:50].tolist())

# alb = pd.read_sas("ALB_CR_L.xpt")
# print(alb.columns.tolist())

# print("\n")
# biopro = pd.read_sas("BIOPRO_L.xpt")
# print("LBXSCR exists:", "LBXSCR" in biopro.columns)

# demo = pd.read_sas("DEMO_L.xpt")

# for col in demo.columns:
#     if "AGE" in col.upper() or "GEND" in col.upper():
#         print(col)

# biopro = pd.read_sas("BIOPRO_L.xpt")

# for col in biopro.columns:
#     if "SCR" in col.upper() or "CREAT" in col.upper():
#         print(col)        


demo = pd.read_sas("DEMO_L.xpt")
alb = pd.read_sas("ALB_CR_L.xpt")
biopro = pd.read_sas("BIOPRO_L.xpt")

print("Gender values:")
print(demo["RIAGENDR"].value_counts().sort_index())

print("\nAge summary:")
print(demo["RIDAGEYR"].describe())

print("\nSerum Creatinine summary:")
print(biopro["LBXSCR"].describe())

print("\nUrine Albumin summary:")
print(alb["URXUMA"].describe())

print("\nUrine Creatinine summary:")
print(alb["URXUCR"].describe())