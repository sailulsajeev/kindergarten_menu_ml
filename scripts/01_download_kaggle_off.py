"""Step 1 - Download the Kaggle Open Food Facts snapshot and extract the columns used by the project.

Dataset : "Open Food Facts" on Kaggle - https://www.kaggle.com/datasets/openfoodfacts/world-food-facts
          (publisher: Open Food Facts; Kaggle dataset version 5; file en.openfoodfacts.org.products.tsv,
          356,027 products x 163 columns, tab-separated, 2017 snapshot). Licence: Open Database License (ODbL) v1.0.
Download: programmatic and anonymous via `kagglehub.dataset_download` (no Kaggle account or token needed
          for this public dataset). If kagglehub cannot reach Kaggle, download the file manually from the
          URL above and pass its path as the first argument.

Usage : python scripts/01_download_kaggle_off.py [path/to/en.openfoodfacts.org.products.tsv]
Output: data/raw/off_kaggle_extract.parquet  (all 356,027 rows, 20 columns, values unchanged)
        data/raw/extract_meta.json           (source size, SHA-256, row counts)
"""
import sys, os, csv, json, hashlib, time
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "off_kaggle_extract.parquet")
COLS = ["code", "product_name", "brands", "countries_tags", "categories_tags", "main_category", "ingredients_text",
        "allergens", "traces_tags", "pnns_groups_1", "pnns_groups_2", "serving_size", "last_modified_datetime",
        "energy_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g", "fiber_100g",
        "proteins_100g", "salt_100g", "sodium_100g"]

if len(sys.argv) > 1:
    src = sys.argv[1]
else:
    import kagglehub                                   # download utility only; not used for modelling
    path = kagglehub.dataset_download("openfoodfacts/world-food-facts")
    src = os.path.join(path, "en.openfoodfacts.org.products.tsv")
print("source:", src)

t0 = time.time()
parts = [ch for ch in pd.read_csv(src, sep="\t", usecols=COLS, chunksize=100_000, dtype=str, low_memory=False,
                                  quoting=csv.QUOTE_NONE, on_bad_lines="skip")]
df = pd.concat(parts, ignore_index=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"rows {len(df):,} cols {len(df.columns)} in {time.time() - t0:.0f}s -> {OUT}")

def sha256(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

meta = {"source_file": os.path.basename(src), "source_bytes": os.path.getsize(src), "source_sha256": sha256(src),
        "kaggle_dataset": "openfoodfacts/world-food-facts", "kaggle_version": 5,
        "extract_rows": int(len(df)), "extract_bytes": os.path.getsize(OUT), "extract_sha256": sha256(OUT), "columns": COLS}
json.dump(meta, open(os.path.join(ROOT, "data", "raw", "extract_meta.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
