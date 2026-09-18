"""Step 3 - Build the compact BLS 4.0 nutrient table used by the kindergarten demo.

Input : data/raw/BLS_4_0_2025_DE.zip  (Max Rubner-Institut, CC BY 4.0, downloaded from https://blsdb.de/download)
Output: data/processed/bls40_subset.csv  - BLS code, German/English name and 12 nutrient columns per 100 g
Values such as "<LOD", "<LOQ", "TR" (below detection limit / trace) are kept verbatim here; kita_demo.load_bls()
maps them to 0.0 and keeps empty cells as unknown (NaN).
"""
import os, zipfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, "data", "raw", "BLS_4_0_2025_DE.zip")
OUT = os.path.join(ROOT, "data", "processed", "bls40_subset.csv")
KEEP = ("ENERCC", "WATER", "PROT625", "FAT", "FASAT", "CHO", "SUGAR", "FIBT", "NA ", "CA ", "CHOCAL", "CHORL")

with zipfile.ZipFile(ZIP) as z:
    with z.open("BLS_4_0_2025_DE/BLS_4_0_Daten_2025_DE.xlsx") as f:
        d = pd.read_excel(f)
cols = list(d.columns[:3]) + [c for c in d.columns[3:] if c.startswith(KEEP) and "Datenherkunft" not in c and "Referenz" not in c]
d[cols].to_csv(OUT, index=False)
print(d[cols].shape, "->", OUT)
