import pandas as pd

files = {
    "DEMO": "DEMO_L.xpt",
    "BIOPRO": "BIOPRO_L.xpt",
    "ALB_CR": "ALB_CR_L.xpt",
    "BMX": "BMX_L.xpt",
    "BPXO": "BPXO_L.xpt",
    "DIQ": "DIQ_L.xpt"
}

for name, file in files.items():
    print(f"\n{'='*50}")
    print(name)
    print('='*50)

    df = pd.read_sas(file)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    for col in df.columns:
        print(col)