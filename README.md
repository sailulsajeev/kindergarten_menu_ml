# Kindergarten menu ML prototype (M505 individual project)

Food-group classification of packaged food products (Open Food Facts, Kaggle snapshot) with scikit-learn,
plus a compact rule-based kindergarten menu demonstration (nutrition, allergens, selected DGE
criteria, substitutions). The notebook follows a step-by-step report layout (header block, INTRODUCTION with problem statement,
DATA COLLECTION STRATEGY, DATASET OVERVIEW, Steps 1–9 with a takeaway after each, Final Discussion, References). The authoritative report is the executed notebook
`kindergarten_menu_ml.ipynb`, exported as `kindergarten_menu_ml.html`.

## Folder layout

| Path | Purpose |
|---|---|
| `kindergarten_menu_ml.ipynb` / `.html` | executed notebook and its HTML export (the submission files) |
| `kita_demo.py` | small plain-Python helpers for the demo (no learned models) |
| `scripts/01_download_kaggle_off.py` | download the Kaggle Open Food Facts file (anonymous, `kagglehub`) and extract 22 columns |
| `scripts/02_build_subset.py` | filters, duplicate handling, kJ→kcal and the stratified 12,000-product sample |
| `scripts/03_build_bls_subset.py` | compact BLS 4.0 nutrient table |
| `scripts/04_write_manifest.py` | provenance / checksum manifest |
| `scripts/build_notebook.py`, `scripts/run_notebook.py` | build the notebook from source, execute it from a fresh kernel and export HTML |
| `data/raw/` | immutable downloads (`BLS_4_0_2025_DE.zip`, `extract_meta.json`; the 42 MB `off_kaggle_extract.parquet` is not included because of file-size limits – Option B below rebuilds it in ~2 min) |
| `data/processed/` | `off_foodgroups_sample.csv` (modelling data), `bls40_subset.csv`, `demo_fixtures.json` |
| `data/source_manifest.json` | URLs, versions, licences, checksums |
| `results/` | figures (`fig*.png`), CV/tuning/test tables, classification report, demo summaries |
| `assessment_checklist.md`, `dataset_card.md`, `qa_report.md`, `progress.md` | assessment coverage, data card, QA record, decisions |
| `refs/` | reference documents consulted (DGE standard PDF, OFF taxonomy, BLS documentation text) |

All paths in the notebook are relative to the project root; run everything from that directory.

## 1. Environment (Python 3.10–3.12)

```bash
python -m venv .venv                       # or: uv venv --python 3.11 .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # pinned: scikit-learn 1.6.1, pandas 2.2.3, numpy 2.2.6, matplotlib 3.10.6, kagglehub 1.0.2
python -m ipykernel install --user --name python3
```

The project was executed with Python 3.11.15 on Linux (2 CPU, 8 GB RAM); total notebook run time was
about 10–14 minutes (see `qa_report.md`). No credentials are required (the Kaggle dataset is public and `kagglehub` downloads it anonymously).

## 2. Data acquisition

Option A – **use the shipped sample** (default; nothing to download). `data/processed/off_foodgroups_sample.csv`
is the exact 12,001-row sample the notebook uses (Open Food Facts, ODbL v1.0, see licence note below).

Option B – **rebuild from Kaggle** (≈ 1 GB download, ≈ 2 min processing):

```bash
python scripts/01_download_kaggle_off.py   # kagglehub.dataset_download("openfoodfacts/world-food-facts"), anonymous
#   expected SHA-256 of en.openfoodfacts.org.products.tsv: d1f351a7e2df9f0606126ed82feed45c39b836de5744d792090f1d3dbe7987bf
#   manual alternative: download the file from https://www.kaggle.com/datasets/openfoodfacts/world-food-facts
#   and run  python scripts/01_download_kaggle_off.py /path/to/en.openfoodfacts.org.products.tsv
python scripts/02_build_subset.py          # -> data/processed/off_foodgroups_sample.csv + results/data_funnel.json
```

BLS 4.0 (demo nutrients): `data/raw/BLS_4_0_2025_DE.zip` is included (CC BY 4.0). To re-download, open
https://blsdb.de/download, click "Download BLS-Daten" (a tokenised link to `BLS_4_0_2025_DE.zip`), save it to
`data/raw/` and run `python scripts/03_build_bls_subset.py`.

## 3. Execute the notebook and export HTML

```bash
python scripts/run_notebook.py            # fresh kernel, all cells in order, then HTML export
# equivalent manual commands:
jupyter nbconvert --to notebook --execute --inplace kindergarten_menu_ml.ipynb
jupyter nbconvert --to html kindergarten_menu_ml.ipynb
```

`scripts/build_notebook.py` regenerates the notebook *source* (unexecuted) from the cell texts; run it
only if you edit the cells, then execute again.

## 4. Reproducibility notes

* `RS = 42` is used for the split, cross-validation folds, forest and sampling.
* Cross-validation, tuning and learning curves use the training part only; the test set is evaluated once.
* Results are deterministic for the pinned versions; Logistic Regression scores may differ in the third
  decimal on other BLAS builds.
* Licence: the OFF sample is redistributed under the Open Database License v1.0 with attribution to
  Open Food Facts contributors (https://world.openfoodfacts.org); BLS data under CC BY 4.0 with attribution to
  the Max Rubner-Institut. The DGE standard PDF in `refs/` is a reference copy for the author's use.

## 5. Not included / to be done by the student

The Declaration of Authorship on Canvas and the submission itself (name and ID are filled in the header cell). The notebook does not cover the online assessments or participation components.
