import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../data/processed/master_chem_table.csv"

df = pd.read_csv(DATA_PATH)
print(f"Total rows: {len(df)}")
cols = [c for c in df.columns if 'tox21' in c]
print(f"Prop columns: {cols}")

for c in cols:
    valid = df[c].dropna()
    print(f"\nColumn: {c}")
    print(f"Count: {len(valid)}")
    if len(valid) > 0:
        # Check types
        try:
            numeric = pd.to_numeric(valid, errors='coerce')
            print(f"Numeric count: {len(numeric.dropna())}")
            print(f"Min: {numeric.min()}, Max: {numeric.max()}")
            print(f"Value counts (top 5):\n{valid.value_counts().head()}")
        except:
            print(f"Value counts:\n{valid.value_counts().head()}")
