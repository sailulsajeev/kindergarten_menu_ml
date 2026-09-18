# QA report – validation outcomes and unresolved issues

Date: 2026-09-17 (v3, Kaggle dataset) · Authoritative artefacts: `kindergarten_menu_ml.ipynb` (executed) and `kindergarten_menu_ml.html`.

## 1. Execution record

| Item | Result |
|---|---|
| Execution environment | Cowork cloud workspace, Linux x86_64, Python 3.11.15, 2 CPU, 8 GB RAM, venv from `requirements.txt` |
| Execution method | `scripts/run_notebook.py` – `nbclient` fresh kernel, all 18 code cells in order, then `nbconvert` HTMLExporter |
| Runs performed | 10 full runs: R1–R5 first version (Hugging Face OFF, German-only), R6–R7 step-by-step layout, R8–R10 after switching to the Kaggle snapshot at the student's request (R8 stopped at the error-analysis cell because of a renamed column, fixed; R9 duplicated one summary print, fixed; R10 final, clean) |
| Final run wall time | ≈ 11 min (R10) – dominated by the RandomForest grid (16 configs × 3 folds) and learning curves |
| Model results (Kaggle data, R9 = R10) | test macro-F1 Dummy 0.046 / LogReg 0.887 / RandomForest 0.912 (accuracy 0.926, balanced accuracy 0.910); CV best LogReg C=10 (0.884), RandomForest 300 trees, depth None, leaf 1, max_features 0.05 (0.893). Earlier German-only version (Hugging Face data): 0.049 / 0.831 / 0.860 – kept in `progress.md` for the record |
| Sample file determinism | `scripts/02_build_subset.py` is seeded (42); the shipped `off_foodgroups_sample.csv` has SHA-256 `40f3ad15…` (recorded in `data/source_manifest.json`) |
| HTML check | rendered in headless Chromium: all headings (intro sections, Steps 1–9 with sub-steps, Final Discussion, References), 6 figure images, `df.info()` and missing-value outputs, all tables and code visible, 0 error outputs; long code lines wrap (a one-line CSS rule is added to the nbconvert output by `run_notebook.py`) |
| Demo focused checks | 10/10 pass in every run (unit conversion, serving division, parser, missing quantity, declared allergen match, unknown allergen info, incomplete week and cycle → *not assessed*, BLS unknown values kept as NaN) |

## 2. Character budget

| Measure | Value |
|---|---|
| Markdown cell source | 9,744 characters |
| Code cell source | 9,991 characters |
| **Combined cell source** | **19,735 characters** (counted by the last notebook cell from the `.ipynb` JSON) |
| Exported HTML file size | ≈ 880 kB (nbconvert embeds CSS/JS and six base64 PNG figures) |

Interpretation: the brief says "keep the size of your notebook below 20,000 characters" without defining the counting method. The combined cell source is below that limit; the execution plan's conservative target of 17,000–19,000 was approached but not fully reached (19,735) because the required sections (seven mandatory figure interpretations, two preprocessing strategies, tuning of two models, error analysis, the kindergarten demonstration and Harvard references) did not fit in fewer characters without removing assessed content. HTML bytes are not comparable to prose characters. If the university counts differently (e.g. rendered text including outputs), the notebook would need further trimming – flagged as an unresolved interpretation.

## 3. Local (linked computer) verification

The linked computer's Cowork VM (Linux aarch64, Python 3.10.12, 4 CPU, 4 GB RAM) cannot run a 14-minute job in this session (180 s per call, background processes are terminated). Verified there in short steps instead:

| Step | Result |
|---|---|
| `uv venv --python 3.10` + `pip install -r requirements.txt` | installed in 14 s; scikit-learn 1.6.1, pandas 2.2.3, numpy 2.2.6, matplotlib 3.10.6 import correctly |
| Data files copied to the project folder (first version) | `off_de_foodgroups_sample.csv` SHA-256 identical |
| `kita_demo.run_checks` | 10/10 pass |
| Frozen configurations refit on the same split (first, German-only version) | LogReg test macro-F1 0.831 identical to the cloud run; RandomForest 0.858 vs 0.860 (floating-point tie-breaking on aarch64 vs x86_64) |
| Kaggle version | the same short-step check was **not repeated** on the linked computer after the dataset switch (see §5) |

The full notebook can be executed on the user's machine with `python scripts/run_notebook.py` (README §3); this was not possible from this session for the reason above.

## 4. Checks on methodology

| Check | Outcome |
|---|---|
| Label leakage | features are ingredient text + 8 nutrients only; categories, `pnns_groups_2`, Nutri-Score, names, brands, codes, countries and `lang_guess` excluded; verified by the `X` column list in Step 5 |
| Train/test independence | duplicate products merged (one representative per identical cleaned ingredient list; 81 conflicting groups dropped) *before* the split; notebook prints 0 duplicate texts in the sample |
| Test set used once | selection uses `val_best` (CV macro-F1) → RandomForest; test evaluation follows and no further tuning cells exist |
| Preprocessing inside CV | all imputation/scaling/TF-IDF steps are inside `Pipeline`/`ColumnTransformer` and fitted per fold by `cross_validate`, `GridSearchCV`, `learning_curve` |
| Scaling claim | LogReg unscaled: lbfgs iteration counts [2000 ×5] (limit), CV macro-F1 0.804 vs 0.872 scaled; RandomForest 0.888 vs 0.886 (within sd) |
| Class imbalance | `class_weight='balanced'` / `'balanced_subsample'`; macro-F1 primary metric; per-class report shows the weakest class is salty-snacks (F1 0.81) – reported, not hidden |
| Figures ↔ text | every figure has a takeaway cell; numbers were re-checked against the final outputs (fibre missing 41 % = 4,982/12,001; sugary snacks 23 %; test macro-F1 0.91 / 0.89 / 0.05; language mix from `lang_guess`) |
| DGE citations | edition, table and pages verified in the downloaded PDF text (`refs/dge.txt`): Tabelle 3 pp. 42–43, 20-day rules p. 43 (footnotes) and p. 45; nutrient reference values deliberately not implemented |
| Allergen taxonomy | 14 categories of Regulation (EU) 1169/2011 Annex II verified on EUR-Lex; statuses declared / traces / unknown; no "safe" label is ever emitted |
| Energy unit | the Kaggle file stores `energy_100g` in kJ; converted to kcal (÷ 4.184) in `scripts/02_build_subset.py` and stated in the notebook |
| Nutrition demo | BLS 4.0 values per 100 g; sodium→salt ×2.5 and mg→g conversion tested; unknown BLS values (empty cells) stay NaN; "<LOD"/"TR" mapped to 0 and documented |

## 5. Unresolved issues and known limitations (honest list)

1. **Character-count interpretation** – see §2.
2. **Full notebook not executed on the user's own machine** – see §3 (cloud execution only; the short-step local check was done for the first version, not repeated after the dataset switch).
3. **Label noise** – OFF categories are crowd-sourced; the notebook shows sampled errors. No manual relabelling was done.
4. **Language scope** – the Kaggle snapshot has no language column; a stop-word heuristic estimates ~69 % French, 6 % English, 5 % German. The German Kita use case is therefore under-represented; the six German demo products are still classified correctly but with lower confidence for two of them. Stated in the notebook as limitation (4) and first item of future work.
5. **OFF food groups ≠ DGE groups** – the demo uses hand-assigned DGE flags in the fixtures; the OFF→DGE mapping is only documented (`dataset_card.md`), not learned or evaluated.
6. **Demo fixtures** – recipes, quantities, allergen labels and children are self-authored; per-serving totals depend on those quantities and BLS raw/cooked choices; they must be verified by a kitchen before any real use.
7. **ROC-AUC** not reported (8-class problem; macro-F1, balanced accuracy and the confusion matrix answer the business question).
8. **Student identity** (name, ID) was taken from the student card the user supplied; the Declaration of Authorship and submission remain the student's responsibility.
9. **Raw Kaggle extract (42 MB)** and the 1.0 GB source TSV are not in the project folder (size limits); the TSV's SHA-256 is recorded in `data/raw/extract_meta.json`, and README Option B re-downloads it anonymously with `kagglehub`.
10. **No production food-safety validation** and no cross-contact modelling – stated in the notebook.

## 6. Items verified against the sample submissions

The three sample submissions were used only to calibrate expectations (structure, HTML export, referencing). Their datasets (Telco churn, credit-card fraud, UCI student success), code and text were not reused. Unlike two of the samples, this project reports validation-based model selection before the single test evaluation, keeps preprocessing inside pipelines, and uses class weights rather than SMOTE.
