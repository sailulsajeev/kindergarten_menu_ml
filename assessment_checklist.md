# Assessment checklist – M505 Individual Project

Sources: `M505 Assessment Brief (7).pdf` (authoritative) and `M505 Assessment Brief-Additional Hints.pdf` (technical companion). Sample submissions (`kranthi_machine_learning.pdf`, `AF_14196_1.html`, `Student_Success_Early-Warning_Analytics (1).html`) were read as examples only – none of their code, wording, data or references was reused.

Status legend: **done** = present in the executed notebook / package with evidence; **partial** = present with a stated limitation; **open** = not done (student action).

| # | Requirement (brief / hints) | Planned notebook section | Status | Evidence |
|---|---|---|---|---|
| 1 | Problem statement: business problem, why it matters, benefit, data collection, ML formulation (brief; hints Stage 1) | Header, INTRODUCTION, Problem Statement | done | header block (Dataset/URL/Business task), INTRODUCTION, DATASET INFORMATION, FEATURES, Problem Statement cells |
| 2 | Public, downloadable dataset with URL, provenance, size, licence; not used in exercises (brief guidelines; hints §2, Stage 2) | Header, DATA COLLECTION STRATEGY, DATASET OVERVIEW, Step 2 | done | Kaggle URL (`openfoodfacts/world-food-facts`, version 5, ODbL), SHA-256 in `data/raw/extract_meta.json`, funnel printed in Step 2; `data/source_manifest.json`, `dataset_card.md` |
| 3 | Dataset loaded in the notebook, manual vs programmatic download explained (hints Stage 2) | DATA COLLECTION STRATEGY, Step 2 | done | programmatic anonymous download via `kagglehub` (`scripts/01_download_kaggle_off.py`), manual alternative in README; `pd.read_csv(...)`, `df.head()`, `df.info()` in Step 2 |
| 4 | EDA: shape, dtypes, missing values, duplicates, target distribution, descriptive statistics, feature visualisations, initial observations (hints Stage 3) | Steps 2–4 | done | `df.info()` (shape, dtypes), `df.isna().sum()`, duplicates and impossible-value counts, Fig 1 (missingness), Fig 2 (target), `describe()` table, Fig 3–4 (features), takeaway cells |
| 5 | Data-quality issues, sampling/balancing discussion, metric choice (brief) | INTRODUCTION, Step 3, Step 5 | done | class imbalance (Fig 1), invalid values, OCR noise, label noise; class weights instead of resampling; macro-F1 justified |
| 6 | Cleaning and feature engineering with scikit-learn: imputation, encoding/transformation, scaling, explanation of purpose (hints Stage 4) | Steps 5–6 | done | `SimpleImputer(add_indicator=True)`, `TfidfVectorizer`, `StandardScaler` inside `ColumnTransformer`/`Pipeline` |
| 7 | At least two preprocessing strategies compared, effect explained (hints Stage 4 "Required Data Engineering Comparison") | Step 6 | done | Strategy A (no scaling) vs B (StandardScaler), identical folds; table + Fig 5 + interpretation |
| 8 | Effect of a scaling technique illustrated (hints Stage 4) | Step 6 | done | LogReg: lbfgs iteration counts (2000 = limit vs ~120), CV macro-F1 0.804 → 0.872; RandomForest unchanged (0.888 vs 0.886) |
| 9 | Stratified `train_test_split`, fixed random state, split rationale (hints §5) | Step 5 | done | 80/20 stratified, `random_state=42`; duplicates merged before the split (leakage prevention) |
| 10 | At least two meaningfully different scikit-learn models, high-level explanation, suitability, limitations (hints §6) | Step 6 | done | Logistic Regression (linear) vs Random Forest (tree ensemble) + Dummy baseline |
| 11 | Pipelines (`Pipeline`, `ColumnTransformer`, CV) and why (hints §7) | Steps 5–7 | done | all fitting through pipelines; leakage explanation in Step 5 |
| 12 | Hyperparameter analysis for both models: parameters explained, multiple values, results, effects, selection by CV (hints §8) | Step 7 ("Hyperparameter Tuning and Best Model") | done | `GridSearchCV`: LogReg C ∈ {0.01…100} (5-fold), RandomForest 16 configs (3-fold); tables in `results/tuning_*.csv`; Fig 7a/7b; best-model summary table (CV macro-F1 + best parameters); interpretation |
| 13 | Final evaluation on an unseen test set: accuracy, precision, recall, F1, confusion matrix; metric importance explained (brief; hints §9) | Step 8 ("Evaluate the Best Model") | done | `results/test_results.csv`, `classification_report.txt`, Fig 9; test set used once, selection by CV |
| 14 | ROC-AUC "if appropriate" (hints §9) | Step 8 | partial | not reported: 8-class problem where macro-F1/balanced accuracy answer the business question; stated by the metric choice in the INTRODUCTION problem statement |
| 15 | Required visualisations 1–7 with interpretation after each (hints §10) | Steps 3–8 | done | Fig 1 missingness, Fig 2 target, Fig 3–4 features, Fig 5 preprocessing/model comparison, Fig 7 hyperparameters, Fig 8 learning curves (training process), Fig 9 confusion matrix – each followed by a takeaway cell |
| 16 | Report-style notebook answering the 11 report questions (hints §11) and final discussion: strengths/limitations, implications, recommendations, informative features, explainability, deployment (brief) | Step 8, Final Discussion | done | Step 8 error analysis + coefficients; Final Discussion: findings, implications, limitations, future work |
| 17 | Required 12-part structure (hints §12) | whole notebook | done | Header (title, name, ID, course, dataset, URL, business task) → INTRODUCTION (with problem statement) → DATA COLLECTION STRATEGY → DATASET OVERVIEW → Step 1 libraries → Step 2 reading the dataset (`df.head()`, `df.info()`) → Step 3 missing values & data quality → Step 4 EDA → Step 5 cleaning & split → Step 6 preprocessing comparison → Step 7 hyperparameter tuning & best model → Step 8 evaluate the best model → Step 9 demonstration → Final Discussion → References |
| 18 | Harvard references incl. dataset and scikit-learn documentation (brief; hints §12) | References | done | 6 verified references (Kaggle dataset, DGE, EU regulation, BLS, scikit-learn paper and user guide); alternatives noted in `dataset_card.md` |
| 19 | Reproducibility: fixed seeds, data acquisition documented, relative paths, runs top-to-bottom, imports, outputs in HTML (hints §14) | all | done | `RS = 42`; relative paths; executed from a fresh kernel by `scripts/run_notebook.py`; `README.md`; `requirements.txt` |
| 20 | Submission format: HTML exported from the notebook, ipynb kept (hints §13) | package | done | `kindergarten_menu_ml.html` generated by nbconvert from the executed notebook |
| 21 | "Keep the size of your notebook below 20,000 characters" (brief) | last cell | done (see note) | markdown 9,744 + code 9,991 = **19,735** characters of cell source, counted by the last notebook cell; HTML byte size is far larger because of embedded figures – the university's counting method is not specified (see `qa_report.md`) |
| 22 | Student name, ID, course on the notebook (hints §12) | header cell | done | Name: Sailu Laly Sajeev, Student_ID: GH1043653 (from the student card supplied by the user), Course: M505 Intro to AI and Machine Learning |
| 23 | Declaration of Authorship on Canvas, submission (brief) | – | **open – student action** | not part of this package; no submission was made |
| 24 | Kindergarten prototype: parsing, group suggestion, nutrition, allergens, DGE criteria, substitutions, weekly/4-week summaries, focused checks (execution brief §8) | Step 9 | done | `kita_demo.py`, `data/processed/demo_fixtures.json`; outputs in Step 9; 10 checks pass |


### Student's additional requirements (17 Sep 2026)

| Requirement | Where |
|---|---|
| Dataset from Kaggle | header URL; DATA COLLECTION STRATEGY; `scripts/01_download_kaggle_off.py` |
| Data Collection Strategy | DATA COLLECTION STRATEGY section |
| Dataset Overview | DATASET OVERVIEW table |
| `df.head()` after reading | Step 2 |
| `df.info()` | Step 2 |
| Missing values and data quality issues | Step 3 (`df.isna().sum()`, Fig 1, impossible values, takeaways) |
| Hyperparameter tuning and best model | Step 7 (grid search, Fig 7, learning curves, best-model table) |
| Evaluate the best model | Step 8 (test metrics, classification report, Fig 9, error analysis) |

Weighting note: the brief allocates 35 % to runnable code and 35 % to report quality/critical evaluation for the primary task; the remaining 30 % (online assessments, participation) is outside this notebook.
