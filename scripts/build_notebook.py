"""Builds kindergarten_menu_ml.ipynb from cell sources (run from the project root).
Layout follows the step-by-step report style (header block, INTRODUCTION / DATASET INFORMATION / FEATURES,
Problem Statement, Step 1..8 with one action per cell and a takeaway after each step, Final Discussion, References)."""
import nbformat as nbf

cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

M("""
# Kindergarten Menu Food-Group Classification — M505

Name: Sailu Laly Sajeev

Student_ID: GH1043653

Course: M505 Intro to AI and Machine Learning

**Dataset**: Open Food Facts (Kaggle), version 5, ODbL v1.0.

**URL**: https://www.kaggle.com/datasets/openfoodfacts/world-food-facts

**Business task**: Suggest the food group of a packaged product from its ingredient list and declared nutrients, so Kita kitchen staff can plan lunches against DGE food-group rules faster.
""")

M("""
## INTRODUCTION
Kitas that follow the *DGE-Qualitätsstandard für die Verpflegung in Kitas* plan lunches by food-group frequencies, so every purchased product must first be assigned to a food group. This project trains and compares scikit-learn classifiers that suggest the food group for staff review; Step 9 shows how confirmed groups feed DGE, nutrition and allergen checks.

**Problem statement:** supervised multiclass classification (8 classes) at product level – not menu level. **Primary metric: macro-F1**, because rare but planning-critical groups (fruits & vegetables, fish/meat/eggs) must weigh as much as the large sugary-snack class.
""")

M("""
## DATA COLLECTION STRATEGY
* **Source:** Open Food Facts on Kaggle (crowd-sourced; 356,027 products, 163 columns, 2017 snapshot) – public, ODbL-licensed, with ingredient text, nutrients per 100 g and a documented food-group label (`pnns_groups_1`); two other Kaggle candidates were rejected (`dataset_card.md`).
* **Download:** programmatic and anonymous with `kagglehub.dataset_download("openfoodfacts/world-food-facts")` (`scripts/01_download_kaggle_off.py`); the 1.0 GB file's SHA-256 is recorded in `data/raw/extract_meta.json`.
* **Subset:** `scripts/02_build_subset.py` applies the filters printed in Step 2, merges duplicate products, converts energy from kJ to kcal and draws a **stratified random sample of 12,000**, shipped with the project.
* **In production:** the same fields from supplier sheets or barcode look-ups; staff corrections become new labels.
""")

M("""
## DATASET OVERVIEW
| Item | Value |
|---|---|
| Rows | 12,001 sampled of 47,949 eligible |
| Inputs | ingredient text (TF-IDF) + 8 nutrients per 100 g/ml: energy (kcal), fat, saturated fat, carbohydrates, sugars, fibre, protein, salt |
| Target | `food_group_l1` – PNNS level-1 food group, 8 classes (composite foods excluded); not identical to DGE groups (`dataset_card.md`) |
| Language | mixed; a stop-word heuristic guesses ~69 % French, 6 % English, 5 % German, 4 % Spanish |
| Excluded (leakage) | category tags, Nutri-Score, product names |

Nutrients are per 100 g (100 ml for liquids).
""")

M("""
## Step 1: Importing necessary libraries
""")

C("""
import json, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, sklearn, sklearn.metrics as skm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore', category=sklearn.exceptions.ConvergenceWarning)
RS, R = 42, 'results/'; pd.set_option('display.max_colwidth', 120)
def fig(name): plt.tight_layout(); plt.savefig(R + name + '.png', dpi=110); plt.show()
""")

M("""
## Step 2: Reading the Dataset
""")

C("""
df = pd.read_csv('data/processed/off_foodgroups_sample.csv'); NUT = [c for c in df.columns if c.endswith('_100g')]  # 8 nutrients
print(pd.Series(json.load(open(R + 'data_funnel.json')), name='products after each filter'))
df.head()
""")

C("""
df.info()
""")

M("""
## Step 3: Missing Values and Data Quality Issues
""")

C("""
print(df.isna().sum().to_dict())
print('duplicate rows:', df.duplicated().sum(), '| duplicate texts:', df.ingredients_text.duplicated().sum())
print('impossible values - negative:', int((df[NUT] < 0).sum().sum()), '| >100 g:', int((df[NUT[1:]] > 100).sum().sum()), '| >900 kcal:', int((df.energy_kcal_100g > 900).sum()))
(df[NUT].isna().mean() * 100).plot.bar(figsize=(6.5, 3), rot=60, title='Fig 1  Missing nutrient values (% of products)', ylabel='%'); fig('fig1_missing')
""")

M("""
**Data-quality takeaways.** Allergen/trace fields are mostly empty, so absence never means "allergen-free" (Step 9). **Fibre is missing for ~41 %**, saturated fat, sugars and salt for ~4 %; missing means *not declared*, not zero, so imputation adds a missing indicator. A few values are physically impossible (> 100 g per 100 g, > 900 kcal). Texts contain OCR noise and several languages; crowd-sourced labels contain noise (nuts under *salty snacks*). No duplicates remain (variants merged before sampling).
""")

M("""
## Step 4: Exploratory Data Analysis (EDA)
#### 1 — Target distribution
""")

C("""
df.food_group_l1.value_counts().plot.barh(figsize=(6.5, 3), title='Fig 2  Target: products per food group', xlabel='products'); fig('fig2_target')
""")

M("""
#### 2 — Features
""")

C("""
print(df[NUT].describe().round(1).T[['mean', 'min', '50%', 'max']])
f, ax = plt.subplots(1, 2, figsize=(11, 3.8))
df.boxplot('energy_kcal_100g', by='food_group_l1', ax=ax[0], rot=60); ax[0].set(title='Fig 3  Energy by food group (axis cut at 1000 kcal)', xlabel='', ylabel='kcal/100 g', ylim=(0, 1000))
df.plot.scatter('fat_100g', 'sugars_100g', c=pd.factorize(df.food_group_l1)[0], cmap='tab10', s=4, alpha=.5, ax=ax[1], colorbar=False, xlim=(0, 100), ylim=(0, 100), title='Fig 4  Fat vs sugars (g/100 g), colour = group')
plt.suptitle(''); fig('fig3_4_features')
""")

M("""
**EDA Takeaways.** *Fig 2*: sugary snacks are ~23 % of products, salty snacks ~5 % – hence macro-F1 and class weights. *Fig 3*: energy separates fats from beverages and vegetables but overlaps between cereals, snacks and dairy – nutrients alone cannot solve the task. *Fig 4*: *fat + sugars ≤ 100 g*; sugary snacks form the upper band, fats the right edge.
""")

M("""
## Step 5: Data Cleaning and Train/Test Split
#### Cleaning
""")

C("""
dfc = df.copy()
dfc.loc[dfc.energy_kcal_100g > 900, 'energy_kcal_100g'] = np.nan          # more than pure fat is impossible
for c in NUT[1:]: dfc.loc[(dfc[c] < 0) | (dfc[c] > 100), c] = np.nan        # unknown, never zero
print('values set to unknown:', int(dfc[NUT].isna().sum().sum() - df[NUT].isna().sum().sum()))
hard = (dfc.fat_100g + dfc.carbohydrates_100g + dfc.proteins_100g) > 101
dfc = dfc[~hard]; print('rows dropped (fat+carbs+protein > 101 g):', int(hard.sum()))
""")

M("""
#### Stratified 80/20 train/test split
""")

C("""
X, y = dfc[['ingredients_text'] + NUT], dfc.food_group_l1
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
print('train', X_tr.shape, '| test', X_te.shape, '| test class shares', y_te.value_counts(normalize=True).round(2).to_dict())
""")

M("""
**Preprocessing & Split Summary.** *Split*: 80/20 stratified, fixed seed, on independent products (duplicates merged *before* splitting); the test set is used **once**, in Step 8. *Text*: TF-IDF uni-/bigrams (`min_df=3`, ≤ 5,000 terms, sublinear tf), fitted **inside each training fold**. *Numeric*: median imputation + missing indicator; B adds `StandardScaler`.
""")

M("""
## Step 6: Preprocessing Comparison — Strategy A (unscaled) vs B (StandardScaler)
Same TF-IDF, same imputation, the **same 5 stratified folds** – only the scaler differs. **Logistic Regression**: one weight per term and nutrient per class – fast, readable, linear only. **Random Forest**: decorrelated trees on bootstrap samples with random feature subsets – non-linear, scale-robust, less interpretable.
""")

C("""
def prep(scale):
    num = [('imp', SimpleImputer(strategy='median', add_indicator=True))] + ([('sc', StandardScaler())] if scale else [])
    return ColumnTransformer([('txt', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=5000, sublinear_tf=True), 'ingredients_text'), ('num', Pipeline(num), NUT)])
models = {'Dummy': DummyClassifier(strategy='most_frequent'), 'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
          'RandomForest': RandomForestClassifier(class_weight='balanced_subsample', random_state=RS, n_jobs=-1)}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
""")

M("""
#### Comparison table
""")

C("""
rows = []
for m, est in models.items():
    for strat, sc in [('A: unscaled', False), ('B: scaled', True)]:
        if m == 'Dummy' and sc: continue
        r = cross_validate(Pipeline([('prep', prep(sc)), ('model', est)]), X_tr, y_tr, cv=cv, scoring='f1_macro', return_estimator=True)
        rows.append(dict(model=m, strategy=strat, macro_f1=r['test_score'].mean(), sd=r['test_score'].std(), fit_s=r['fit_time'].mean()))
        if m == 'LogReg': print(strat, 'lbfgs iterations per fold (limit 2000):', [int(e['model'].n_iter_[0]) for e in r['estimator']])
cmp = pd.DataFrame(rows).round(3); cmp.to_csv(R + 'preprocessing_comparison.csv', index=False); print(cmp.to_string(index=False))
cmp.pivot(index='model', columns='strategy', values='macro_f1').plot.bar(figsize=(6.5, 3.2), rot=0, ylabel='macro-F1', title='Fig 5  5-fold CV macro-F1 by strategy'); fig('fig5_preprocessing')
""")

M("""
**Preprocessing Takeaways (Fig 5).** Scaling changes nothing for the dummy and only noise for the forest: trees split on thresholds, and a monotone rescaling keeps the same splits. Logistic Regression is sensitive: unscaled kcal values in the hundreds next to TF-IDF weights in [0, 1] ill-condition the loss – lbfgs **hits the 2,000-iteration limit in every fold** and macro-F1 drops; scaled, it converges quickly. Strategy B is used from here on.
""")

M("""
## Step 7: Hyperparameter Tuning and Best Model
#### Grid search (training data only)
`GridSearchCV` with stratified folds and macro-F1. **LogReg:** `C`, the inverse regularisation strength (small C = strong shrinkage). **RandomForest:** `n_estimators`, `max_depth` (limits overfitting), `min_samples_leaf`, `max_features`; three folds keep 16 configurations affordable.
""")

C("""
grids = {'LogReg': (LogisticRegression(max_iter=3000, class_weight='balanced'), {'model__C': [0.01, 0.1, 1, 10, 100]}, 5),
         'RandomForest': (RandomForestClassifier(class_weight='balanced_subsample', random_state=RS, n_jobs=-1),
                          {'model__n_estimators': [100, 300], 'model__max_depth': [20, None], 'model__min_samples_leaf': [1, 3], 'model__max_features': ['sqrt', 0.05]}, 3)}
best, cvres, val_best, params = {}, {}, {}, {}
for m, (est, grid, k) in grids.items():
    gs = GridSearchCV(Pipeline([('prep', prep(True)), ('model', est)]), grid, cv=StratifiedKFold(k, shuffle=True, random_state=RS), scoring='f1_macro').fit(X_tr, y_tr)
    best[m], val_best[m], params[m] = gs.best_estimator_, gs.best_score_, gs.best_params_
    cvres[m] = pd.DataFrame(gs.cv_results_).filter(regex='param|mean|std').round(3); cvres[m].to_csv(R + f'tuning_{m}.csv', index=False)
    print(m, f'CV macro-F1 {gs.best_score_:.3f} ± {cvres[m].std_test_score[gs.best_index_]:.3f}')
""")

M("""
#### Hyperparameter effects
""")

C("""
f, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].errorbar(cvres['LogReg'].param_model__C.astype(float), cvres['LogReg'].mean_test_score, yerr=cvres['LogReg'].std_test_score, marker='o')
ax[0].set(xscale='log', title='Fig 7a  LogReg: C vs CV macro-F1 (±1 sd)', xlabel='C', ylabel='macro-F1')
rf = cvres['RandomForest'].astype(str).assign(s=cvres['RandomForest'].mean_test_score)
rf.pivot_table(index='param_model__n_estimators', columns=['param_model__max_features', 'param_model__max_depth'], values='s').plot.bar(ax=ax[1], rot=0, ylim=(0.78, None), title='Fig 7b  RandomForest (mean over leaf sizes)', xlabel='n_estimators')
ax[1].legend(title='max_features, max_depth', fontsize=8); fig('fig7_hyperparameters')
""")

M("""
#### Learning curves and best-model summary
""")

C("""
f, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for a, (m, est) in zip(ax, best.items()):
    n, tr, va = learning_curve(est, X_tr, y_tr, train_sizes=np.linspace(0.1, 1.0, 5), cv=StratifiedKFold(3, shuffle=True, random_state=RS), scoring='f1_macro')
    a.plot(n, tr.mean(1), 'o-', label='training'); a.plot(n, va.mean(1), 's-', label='validation'); a.legend(); a.set(title=f'Fig 8  Learning curve: {m}', xlabel='training products', ylabel='macro-F1')
fig('fig8_learning_curves')
best_model = max(val_best, key=val_best.get)
print(pd.Series(val_best).round(3).to_frame('CV macro-F1').assign(best_parameters=pd.Series(params))); print('Best model:', best_model)
""")

M("""
**Tuning — Summary.** *Fig 7*: Logistic Regression shows the classic bias–variance shape (C = 0.01 underfits; the score peaks at C = 10). For the forest, more trees add little, unlimited depth beats depth 20, and more candidate features per split help on sparse text. *Fig 8*: both models fit the training data almost perfectly while validation scores still rise with more data – the gap is variance, so **more labelled products would help more than a more complex model**. The best model is chosen by CV macro-F1 before the test set is touched.
""")

M("""
## Step 8: Evaluate the Best Model
#### Test-set evaluation (best model, frozen alternative, dummy)
""")

C("""
final = best_model
finals = {'Dummy': Pipeline([('prep', prep(True)), ('model', DummyClassifier(strategy='most_frequent'))]).fit(X_tr, y_tr), **best}
pred = {m: est.predict(X_te) for m, est in finals.items()}
test = pd.DataFrame({m: dict(accuracy=skm.accuracy_score(y_te, p), balanced_accuracy=skm.balanced_accuracy_score(y_te, p), macro_f1=skm.f1_score(y_te, p, average='macro'))
                     for m, p in pred.items()}).T.round(3); test.to_csv(R + 'test_results.csv'); print(test)
rep = skm.classification_report(y_te, pred[final], digits=3); print(rep); open(R + 'classification_report.txt', 'w').write(rep)
f, a = plt.subplots(figsize=(6.5, 5.5)); skm.ConfusionMatrixDisplay.from_predictions(y_te, pred[final], ax=a, xticks_rotation='vertical', cmap='Blues', colorbar=False)
a.set_title(f'Fig 9  Confusion matrix, test set: {final}'); fig('fig9_confusion_matrix')
""")

M("""
#### Error analysis and informative features
""")

C("""
p = pred[final]; err = X_te[p != y_te].assign(true=y_te[p != y_te], pred=p[p != y_te], name=dfc.product_name)
print(len(err), 'test errors; most frequent confusions:'); print(pd.crosstab(err.true, err.pred).stack().nlargest(5))
display(err.sample(6, random_state=RS)[['name', 'true', 'pred', 'ingredients_text']])
coef = pd.DataFrame(best['LogReg']['model'].coef_, index=best['LogReg'].classes_, columns=best['LogReg']['prep'].get_feature_names_out())
print('LogReg strongest terms per class:', {c: [t.split('__')[1] for t in coef.loc[c].nlargest(4).index] for c in coef.index})
""")

M("""
**Test Evaluation — Interpretation.** Both tuned models beat the majority baseline by a wide margin and test scores sit close to the CV estimates. Errors concentrate on neighbouring groups. Coefficients are **associations, not causal explanations**; low-confidence predictions go to staff correction.
""")

M("""
## Step 9: Kindergarten Demonstration (rule-based, human-reviewed)
Nutrients: **Bundeslebensmittelschlüssel 4.0** (Max Rubner-Institut, 2025, CC BY 4.0; per 100 g). Recipes, plans and children are **self-authored fictional fixtures**. DGE rules: food-frequency criteria of Tabelle 3 (lunch, 5 days, ages 1–<7) and the 20-day criteria, pp. 43/45 of the DGE-Qualitätsstandard (6th ed., 2023) – a *selected-criteria check*, not a certification. Allergen matching uses the 14 categories of Regulation (EU) 1169/2011 Annex II, distinguishes declared / traces / unknown, and cannot detect cross-contact.
#### Parser, recipe nutrition, suggestions, allergens
""")

C("""
import kita_demo as kd
bls, fx = kd.load_bls('data/processed/bls40_subset.csv'), kd.load_fixtures('data/processed/demo_fixtures.json')
print([kd.parse_menu_line(l) for l in ['250 g Kartoffeln', '1 Stück Zwiebel']])
r = fx['recipes']['linsenbolognese']; n = kd.recipe_nutrition(r, fx['ingredients'], bls); print(r['name'], '-', n['basis']); display(n['per_serving'].round(2).to_frame('per serving').T)
sug = kd.product_features(['vollkornnudeln', 'joghurt35', 'fischstaebchen', 'apfel', 'rote_linsen', 'gouda30'], fx, bls)
sug['suggested_group'], sug['confidence'] = best[final].predict(sug), best[final].predict_proba(sug).max(1).round(2)
display(sug[['ingredient', 'suggested_group', 'confidence']])
display(kd.allergen_report(fx['recipes']['kaesespaetzle'], fx['ingredients'], fx['children']))
""")

M("""
#### DGE checks, substitutions, summaries, focused checks
""")

C("""
display(kd.dge_week_check(fx['week1'], fx['recipes'])); display(kd.dge_cycle_check(fx['cycle4w'], fx['recipes']))
display(kd.substitution_candidates('milch15', fx['children'][0], fx, bls))
display(kd.period_summary(fx['week1'], fx, bls, 'Week 1 (5 lunch days, per child portion)'))
cyc = kd.period_summary(fx['cycle4w'], fx, bls, '4-week cycle (20 lunch days)'); cyc.to_csv(R + 'demo_cycle_summary.csv', index=False)
print(kd.run_checks(fx, bls).to_string(index=False))
""")

M("""
**Demonstration — Interpretation.** The parser returns *verify* for units without mass. All six German demo products are suggested correctly, with lower confidence where German vocabulary is sparse in the training data (*Fischstäbchen*, *Gouda*). *Käsespätzle* is *contains* for the milk-, gluten- and egg-allergic children but *no declared match* for the lactose-intolerant child – deliberately **not** "safe". Week 1 passes all criteria; the 4-week cycle fails *fettreicher Fisch ≥ 2×/20 days*. Substitutions are screened per child (lactose-free milk is blocked for the milk-protein allergy) ; incomplete periods return *not assessed*.
""")

M("""
## Final Discussion
**Findings.** Ingredient text plus declared nutrients predict the PNNS food group with test macro-F1 0.91 (Random Forest) and 0.89 (Logistic Regression) against 0.05 for the majority baseline; scaling was decisive for the linear model, irrelevant for the forest; learning curves show the models are data-limited.

**Implications.** As a suggestion engine with confidence-based triage the model can cut manual grouping effort and make DGE checks reproducible; it must run behind human review.

**Limitations.** (1) labels are crowd-sourced and *not* DGE groups; (2) product level – no claim about menus or DGE compliance; (3) composite dishes excluded; (4) the snapshot is mostly French-labelled, so the German Kita use case needs German training data; (5) fibre missing for 41 %; (6) demo fixtures need kitchen verification; (7) no production food-safety validation.

**Future work.** German training data; DGE-level groups; calibration.
""")

M("""
## References (Harvard; accessed 16 Sep 2026)
Deutsche Gesellschaft für Ernährung (2023) *DGE-Qualitätsstandard für die Verpflegung in Kitas*, 6. Aufl., 2. korr. Nachdruck. https://www.dge.de/fileadmin/dok/gemeinschaftsgastronomie/dge-qualitaetsstandards/2023/230929-DGE-QST-Kita.pdf

European Parliament and Council (2011) *Regulation (EU) No 1169/2011 on the provision of food information to consumers*, Annex II. https://eur-lex.europa.eu/eli/reg/2011/1169/oj

Max Rubner-Institut (2025) *Bundeslebensmittelschlüssel (BLS), Version 4.0 – Deutsche Nährstoffdatenbank*, CC BY 4.0. DOI: 10.25826/Data20251217-134202-0. https://blsdb.de/download

Open Food Facts (2017) *Open Food Facts* [dataset], version 5, ODbL v1.0. Kaggle. https://www.kaggle.com/datasets/openfoodfacts/world-food-facts

Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

scikit-learn developers (2025) *scikit-learn 1.6 User Guide*. https://scikit-learn.org/1.6/user_guide.html
""")

C("""
n = {t: sum(len(''.join(c['source'])) for c in json.load(open('kindergarten_menu_ml.ipynb', encoding='utf-8'))['cells'] if c['cell_type'] == t) for t in ('markdown', 'code')}
print(f"Notebook source: markdown {n['markdown']:,} + code {n['code']:,} = {sum(n.values()):,} characters (limit 20,000)")
print('Versions: scikit-learn', sklearn.__version__, '| pandas', pd.__version__, '| numpy', np.__version__)
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "kindergarten_menu_ml.ipynb")
md = sum(len(c.source) for c in cells if c.cell_type == "markdown"); code = sum(len(c.source) for c in cells if c.cell_type == "code")
print(f"written: markdown {md:,} code {code:,} total {md + code:,}")
