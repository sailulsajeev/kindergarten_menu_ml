"""Step 4 - Write data/source_manifest.json with provenance, versions, licences and checksums."""
import os, json, hashlib, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def entry(rel):
    p = os.path.join(ROOT, rel)
    return {"path": rel, "bytes": os.path.getsize(p), "sha256": sha(p)} if os.path.exists(p) else {"path": rel, "note": "not present (local only or deleted)"}

meta = json.load(open(os.path.join(ROOT, "data", "raw", "extract_meta.json")))
manifest = {
    "generated": datetime.date.today().isoformat(),
    "sources": [
        {"name": "Open Food Facts (Kaggle dataset openfoodfacts/world-food-facts)", "publisher": "Open Food Facts, published on Kaggle",
         "url": "https://www.kaggle.com/datasets/openfoodfacts/world-food-facts", "retrieved": "2026-09-16 (kagglehub, anonymous)",
         "version": "Kaggle dataset version 5 (2017 snapshot)", "license": "Open Database License (ODbL) v1.0",
         "original_filename": meta["source_file"], "bytes": meta["source_bytes"], "sha256": meta["source_sha256"], "rows": meta["extract_rows"], "columns_in_file": 163},
        {"name": "OFF food-groups / PNNS documentation", "publisher": "Open Food Facts (openfoodfacts-server, main)",
         "url": "https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-server/main/taxonomies/food_groups.txt",
         "retrieved": "2026-09-16", "license": "ODbL / AGPL-3.0 (repository)", **entry("refs/food_groups.txt")},
        {"name": "Bundeslebensmittelschlüssel (BLS) 4.0", "publisher": "Max Rubner-Institut, Karlsruhe",
         "url": "https://blsdb.de/download", "retrieved": "2026-09-16", "version": "4.0 (Dezember 2025)",
         "license": "CC BY 4.0 - cite: Max Rubner-Institut (2025): Bundeslebensmittelschlüssel (BLS), Version 4.0 - Deutsche Nährstoffdatenbank. Karlsruhe. DOI: 10.25826/Data20251217-134202-0",
         "original_filename": "BLS_4_0_2025_DE.zip", **entry("data/raw/BLS_4_0_2025_DE.zip")},
        {"name": "DGE-Qualitätsstandard für die Verpflegung in Kitas, 6. Aufl., 2. korr. Nachdruck 2023", "publisher": "Deutsche Gesellschaft für Ernährung e. V.",
         "url": "https://www.dge.de/fileadmin/dok/gemeinschaftsgastronomie/dge-qualitaetsstandards/2023/230929-DGE-QST-Kita.pdf",
         "retrieved": "2026-09-16", "license": "copyright DGE - used as reference, not redistributed", **entry("refs/DGE-QST-Kita-2023.pdf")},
        {"name": "Regulation (EU) No 1169/2011, Annex II", "publisher": "EUR-Lex", "url": "https://eur-lex.europa.eu/eli/reg/2011/1169/oj",
         "retrieved": "2026-09-16", "license": "EU public documents"},
    ],
    "derived_files": [
        {**entry("data/raw/off_kaggle_extract.parquet"), "rows": meta["extract_rows"], "recipe": "scripts/01_download_kaggle_off.py (22 columns, values unchanged)", "license": "ODbL v1.0"},
        {**entry("data/processed/off_foodgroups_sample.csv"), "rows": 12001, "recipe": "scripts/02_build_subset.py (filters F1-F6, seed 42; energy kJ -> kcal)", "license": "ODbL v1.0 - attribution: Open Food Facts contributors"},
        {**entry("data/processed/bls40_subset.csv"), "rows": 7140, "recipe": "scripts/03_build_bls_subset.py", "license": "CC BY 4.0 - Max Rubner-Institut"},
        {**entry("data/processed/demo_fixtures.json"), "recipe": "self-authored demonstration fixtures (fictional)", "license": "project"},
    ],
    "alternatives_inspected": [
        {"name": "Hugging Face openfoodfacts/product-database (2026 Parquet, revision 8aea7223)", "rows": 4740368, "note": "used in the first project version (350,618 German-language products); replaced by the Kaggle snapshot at the student's request so that the dataset comes from Kaggle"},
        {"name": "Kaggle irkaal/foodcom-recipes-and-reviews", "rows": 522517, "note": "English recipes, 311 free-text categories, nutrition per recipe without per-100 g basis"},
        {"name": "Kaggle trolukovich/nutritional-values-for-common-foods-and-products", "rows": 8789, "note": "nutrients only, no ingredient text and no supervised label"},
    ],
}
json.dump(manifest, open(os.path.join(ROOT, "data", "source_manifest.json"), "w"), indent=2, ensure_ascii=False)
print("written data/source_manifest.json")
