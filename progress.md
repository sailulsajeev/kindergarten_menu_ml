# progress.md – decisions and resume checkpoint

Project: M505 kindergarten menu ML prototype (food-group classification + rule-based Kita demo).
Authoritative notebook: `kindergarten_menu_ml.ipynb`; results directory: `results/`.

## Execution environment (decided 2026-09-16)
- Cowork cloud workspace (Linux x86_64, Python 3.11.15, 2 CPU, 8 GB RAM) is where all experiments were executed (5 full runs; final run 841 s).
  Isolated venv `.venv` created with `uv`, packages pinned in `requirements.txt`.
- Linked computer (Cowork VM on the user's Mac, Python 3.10, 4 CPU, 4 GB RAM) is reachable through
  `device_bash`, but every call is limited to 180 s and background processes are killed when the
  call ends (verified twice with `nohup`/`setsid`). A 15–25 min notebook run therefore cannot be
  executed there from this session; the README procedure was verified there in short steps instead
  (see `qa_report.md`).
- No Kaggle credentials were needed: `kagglehub` downloads public datasets anonymously.

## Stage checkpoints
| Stage | Status | Evidence |
|---|---|---|
| 1 Assessment files read | done | `assessment_checklist.md` |
| 2 Environment | done | `requirements.txt`, `.venv` (cloud) |
| 3 Data research + acquisition | done | `data/source_manifest.json`, `data/raw/extract_meta.json`, `scripts/01_*`, `scripts/02_*` |
| 4 Task locked | done | `dataset_card.md` |
| 5 Data preparation, split | done | notebook §3–4 |
| 6 Pipelines, scaling experiment, tuning | done | notebook §5–6, `results/preprocessing_comparison.csv`, `results/tuning_*.csv` |
| 7 Test evaluation | done | notebook §7, `results/test_results.csv`, `results/classification_report.txt` |
| 8 Kita demo | done | `kita_demo.py`, `data/processed/demo_fixtures.json`, notebook §8 |
| 9 Notebook as report | done | `kindergarten_menu_ml.ipynb` |
| 10 Package | done | `README.md`, `kindergarten_menu_ml.html`, `qa_report.md`; delivered to `SailuMLProject/kindergarten_menu_ml/` on the linked computer |

## Key decisions (with reasons)
1. **Dataset:** Open Food Facts Hugging Face Parquet snapshot pinned to git revision
   `8aea7223b582c312f6fdb82c228cec59755ae9b2` (downloaded 2026-09-16, 7,880,838,030 bytes, SHA-256
   `74eeea47…`). Reasons: it has a `lang` column (Kaggle CSV has none), German ingredient text per
   language, per-100 g nutrients and the documented `food_groups_tags` label. The Kaggle 2017 OFF
   snapshot has only 4,163 German-market products with label+text+nutrients; Food.com has 311
   free-text English categories and no per-100 g basis; the trolukovich table has no labels/text.
2. **Whole-file download instead of remote column reads:** a DuckDB projection over HTTP timed out
   after 10 min; a sequential download took ~3 min. Only the needed columns were then extracted
   locally (`data/raw/off_de_extract.parquet`, 48 MB, 350,618 rows) and the 7.9 GB file was deleted
   after its checksum was recorded.
3. **Target:** OFF level-1 food group; 8 classes kept, composite foods / alcoholic beverages /
   baby foods excluded. Product names, categories, Nutri-Score, NOVA excluded from features.
4. **Duplicates:** identical cleaned ingredient text → one representative (highest completeness);
   153 groups with conflicting labels dropped. Done before the split.
5. **Sample:** stratified random 12,000 (seed 42) from 57,301 eligible products, not first-N rows.
6. **Models:** Dummy, LogisticRegression (lbfgs, balanced), RandomForest (balanced_subsample).
   Scaling experiment = identical folds, only StandardScaler differs.
7. **Tuning:** GridSearchCV macro-F1; LR 5 folds × 5 C values; RF 3 folds × 16 configs.
8. **Demo nutrient source:** BLS 4.0 (MRI, CC BY 4.0, open download found at blsdb.de on
   2026-09-16) – better grounded than OFF packaged products for raw ingredients.
9. **DGE rules:** only Tabelle 3 (5-day lunch) and the 20-day rules (pp. 42–45 of the 2023 edition)
   were implemented; nutrient reference values were deliberately not implemented.
10. **Layout (2026-09-17):** on the user's request the notebook was restructured to the step-by-step report layout of a sample submission (header block, INTRODUCTION / DATASET INFORMATION / FEATURES / Problem Statement, Step 1–8, Final Discussion, References); content, code and results unchanged; name and ID filled from the student card. The previous builder is kept as `scripts/build_notebook_v1_sections.py`.
11. **Character budget:** notebook source (markdown + code) is counted by the last notebook cell;
    HTML byte size is reported separately in `qa_report.md`.

12. **Dataset switch (2026-09-17):** at the student's request ("dataset from Kaggle") the Hugging Face snapshot was replaced by the Kaggle dataset `openfoodfacts/world-food-facts` (version 5, 2017, 356,027 products, SHA-256 `d1f351a7…`). Only ~2,900 products have German text, so the task now uses all languages (mostly French) with a documented language limitation. New scripts `01_download_kaggle_off.py` / `02_build_subset.py`; funnel 356,027 → 47,949 eligible → 12,001 sampled. Results changed from macro-F1 0.831/0.860 (German-only) to 0.887/0.912 (LogReg/RandomForest). Sections added on request: DATA COLLECTION STRATEGY, DATASET OVERVIEW, `df.head()`, `df.info()`, Missing Values and Data Quality Issues, Hyperparameter Tuning and Best Model, Evaluate the Best Model. Previous builders kept as `scripts/build_notebook_v1_sections.py` and `scripts/build_notebook_v2_hf_steps.py`; the German extraction script is no longer in the package.

## Resume instructions
If interrupted: the processed sample, fixtures and scripts are all present; re-run
`python scripts/run_notebook.py` from the project root inside the venv. Re-downloading from Kaggle
(`scripts/01_download_kaggle_off.py`) is only needed if `data/raw/off_kaggle_extract.parquet` is missing.
