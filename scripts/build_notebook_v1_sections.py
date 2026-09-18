"""Builds kindergarten_menu_ml.ipynb from cell sources (run from the project root)."""
import nbformat as nbf

cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

M("""
# Food-group classification of German food products for kindergarten menu planning
**M505 Intro to AI and Machine Learning – Individual Project** · Student: `[STUDENT NAME]` (ID `[STUDENT ID]`) · Gisma University of Applied Sciences · September 2026
""")

M("""
## 1. Business problem and prediction task
Kitas following the *DGE-Qualitätsstandard für die Verpflegung in Kitas* plan lunches by food-group frequencies, so every purchased product must first be assigned to a food group. This prototype **suggests the food group of a packaged product from its German ingredient list and declared nutrients** for staff review.

* **Task:** supervised multiclass classification, 8 classes, product level (not menu level); **target:** level-1 food group that Open Food Facts (OFF) derives from category tags; **inputs:** ingredient text (TF-IDF) + 8 nutrients per 100 g/ml.
* **Metric:** **macro-F1**, because rare but planning-critical groups (fruits & vegetables, fish/meat/eggs) must weigh as much as the large sugary-snack class.
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
## 2. Data source, acquisition and target
**Dataset:** Open Food Facts product database, Parquet export on Hugging Face (`openfoodfacts/product-database`, revision `8aea7223`, downloaded 16 Sep 2026, SHA-256 `74eeea47…`), ODbL v1.0. Nutrients are per 100 g (100 ml for beverages), energy in **kcal**.

**Acquisition (scripted, see README):** `scripts/01_extract_off_de.py` extracts the 350,618 German-language products from the 7.9 GB snapshot; `scripts/02_build_subset.py` applies the filters printed below, merges duplicates (identical ingredient list → one representative) and draws a **stratified random sample of 12,000**.

**Target:** OFF `food_groups_tags` level 1 (derived by OFF from crowd-sourced category tags); composite foods, alcoholic beverages and baby foods are excluded. Category tags, Nutri-Score, NOVA and product names are **not features**, so the label cannot leak; OFF groups are not DGE groups (see `dataset_card.md`).
""")

C("""
df = pd.read_csv('data/processed/off_de_foodgroups_sample.csv'); NUT = [c for c in df.columns if c.endswith('_100g')]  # 8 nutrients
print(pd.Series(json.load(open(R + 'data_funnel.json')), name='products after each filter'))
df[['product_name_de', 'ingredients_text', 'food_group_l1', 'energy_kcal_100g', 'sugars_100g', 'salt_100g']].head(3)
""")

M("""
## 3. Exploratory data analysis
""")

C("""
print(df.shape, '| duplicate rows:', df.duplicated().sum(), '| duplicate ingredient texts:', df.ingredients_text.duplicated().sum())
f, ax = plt.subplots(1, 2, figsize=(11, 3.6))
df.food_group_l1.value_counts().plot.barh(ax=ax[0], title='Fig 1  Target: products per food group', xlabel='products')
(df[NUT].isna().mean() * 100).plot.bar(ax=ax[1], rot=60, title='Fig 2  Missing nutrient values (% of products)'); fig('fig1_2_target_missing')
""")

M("""
**Fig 1**: sugary snacks are ~24 % of products, fruits & vegetables ~5 %, so a majority guess reaches ~24 % accuracy – hence macro-F1 and class weights. **Fig 2**: **fibre is missing for ~35 %**; missing means *not declared*, not zero, so imputation adds missing indicators.
""")

C("""
print(df[NUT].describe().round(1).T[['mean', 'min', '50%', 'max']])
f, ax = plt.subplots(1, 2, figsize=(11, 3.8))
df.boxplot('energy_kcal_100g', by='food_group_l1', ax=ax[0], rot=60); ax[0].set(title='Fig 3  Energy by food group (axis cut at 1000 kcal)', xlabel='', ylabel='kcal/100 g', ylim=(0, 1000))
df.plot.scatter('fat_100g', 'sugars_100g', c=pd.factorize(df.food_group_l1)[0], cmap='tab10', s=4, alpha=.5, ax=ax[1], colorbar=False, xlim=(0, 100), ylim=(0, 100), title='Fig 4  Fat vs sugars (g/100 g), colour = group')
plt.suptitle(''); fig('fig3_4_features')
print('impossible values - negative:', int((df[NUT] < 0).sum().sum()), '| > 100 g/100 g:', int((df[NUT[1:]] > 100).sum().sum()), '| > 900 kcal:', int((df.energy_kcal_100g > 900).sum()))
""")

M("""
**Fig 3**: energy separates fats from beverages and vegetables but overlaps between cereals, snacks and dairy – nutrients alone cannot solve the task. **Fig 4**: *fat + sugars ≤ 100 g*; sugary snacks form the upper band, fats the right edge. **Quality issues:** a few impossible values, OCR-garbled texts, label noise from crowd-sourced categories (nuts under *salty snacks*).
""")

M("""
## 4. Cleaning, feature engineering, leakage prevention and split
* **Invalid values** become *unknown* (NaN); records with fat + carbohydrate + protein > 101 g are dropped.
* **Text**: TF-IDF on word uni-/bigrams (`min_df=3`, ≤ 5,000 terms, sublinear tf), fitted **inside each training fold**.
* **Numeric**: median imputation plus a missing indicator per nutrient; Strategy B adds `StandardScaler`.
* **Split**: 80/20 stratified `train_test_split`, fixed seed, on independent products (duplicates merged *before* splitting); the test set is used **once**, in section 7. **Imbalance**: `class_weight='balanced'` instead of resampling.
""")

C("""
dfc = df.copy()
dfc.loc[dfc.energy_kcal_100g > 900, 'energy_kcal_100g'] = np.nan          # more than pure fat is impossible
for c in NUT[1:]: dfc.loc[(dfc[c] < 0) | (dfc[c] > 100), c] = np.nan        # unknown, never zero
print('values set to unknown:', int(dfc[NUT].isna().sum().sum() - df[NUT].isna().sum().sum()))
hard = (dfc.fat_100g + dfc.carbohydrates_100g + dfc.proteins_100g) > 101
dfc = dfc[~hard]; print('rows dropped (fat+carbs+protein > 101 g):', int(hard.sum()))
X, y = dfc[['ingredients_text'] + NUT], dfc.food_group_l1
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
print('train', X_tr.shape, '| test', X_te.shape, '| test class shares', y_te.value_counts(normalize=True).round(2).to_dict())
""")

M("""
## 5. Preprocessing comparison: Strategy A (no scaling) vs B (StandardScaler)
Same TF-IDF, same imputation, the **same 5 stratified folds** – only the scaler differs. **Logistic Regression** learns one weight per term and nutrient per class: fast, readable, linear effects only. **Random Forest** (Breiman, 2001) averages decorrelated trees on bootstrap samples with random feature subsets: non-linear interactions, scale-robust, slower, less interpretable.
""")

C("""
def prep(scale):
    num = [('imp', SimpleImputer(strategy='median', add_indicator=True))] + ([('sc', StandardScaler())] if scale else [])
    return ColumnTransformer([('txt', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=5000, sublinear_tf=True), 'ingredients_text'), ('num', Pipeline(num), NUT)])
models = {'Dummy': DummyClassifier(strategy='most_frequent'), 'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
          'RandomForest': RandomForestClassifier(class_weight='balanced_subsample', random_state=RS, n_jobs=-1)}
cv, rows = StratifiedKFold(5, shuffle=True, random_state=RS), []
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
**Interpretation (Fig 5).** Scaling changes nothing for the dummy and only fold-level noise (0.003, within one sd) for the forest: trees split on thresholds, and a monotone rescaling keeps the same splits. Logistic Regression is sensitive: unscaled kcal values in the hundreds next to TF-IDF weights in [0, 1] ill-condition the loss – lbfgs **hits the 2,000-iteration limit in every fold** and macro-F1 drops by six points; scaled, it converges in ~115 iterations. Strategy B is used from here on.
""")

M("""
## 6. Hyperparameter tuning (training data only)
`GridSearchCV` with stratified folds and macro-F1. **LogReg:** `C`, the inverse regularisation strength (small C = strong shrinkage). **RandomForest:** `n_estimators`, `max_depth` (limits overfitting), `min_samples_leaf`, `max_features` (candidates per split: `sqrt` ≈ 71 of ~5,000, `0.05` ≈ 250); three folds keep 16 configurations affordable.
""")

C("""
grids = {'LogReg': (LogisticRegression(max_iter=3000, class_weight='balanced'), {'model__C': [0.01, 0.1, 1, 10, 100]}, 5),
         'RandomForest': (RandomForestClassifier(class_weight='balanced_subsample', random_state=RS, n_jobs=-1),
                          {'model__n_estimators': [100, 300], 'model__max_depth': [20, None], 'model__min_samples_leaf': [1, 3], 'model__max_features': ['sqrt', 0.05]}, 3)}
best, cvres, val_best = {}, {}, {}
for m, (est, grid, k) in grids.items():
    gs = GridSearchCV(Pipeline([('prep', prep(True)), ('model', est)]), grid, cv=StratifiedKFold(k, shuffle=True, random_state=RS), scoring='f1_macro').fit(X_tr, y_tr)
    best[m], val_best[m] = gs.best_estimator_, gs.best_score_
    cvres[m] = pd.DataFrame(gs.cv_results_).filter(regex='param|mean|std').round(3); cvres[m].to_csv(R + f'tuning_{m}.csv', index=False)
    print(m, gs.best_params_, f'CV macro-F1 {gs.best_score_:.3f} ± {cvres[m].std_test_score[gs.best_index_]:.3f}')
f, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].errorbar(cvres['LogReg'].param_model__C.astype(float), cvres['LogReg'].mean_test_score, yerr=cvres['LogReg'].std_test_score, marker='o')
ax[0].set(xscale='log', title='Fig 7a  LogReg: C vs CV macro-F1 (±1 sd)', xlabel='C', ylabel='macro-F1')
rf = cvres['RandomForest'].astype(str).assign(s=cvres['RandomForest'].mean_test_score)
rf.pivot_table(index='param_model__n_estimators', columns=['param_model__max_features', 'param_model__max_depth'], values='s').plot.bar(ax=ax[1], rot=0, ylim=(0.78, None), title='Fig 7b  RandomForest (mean over leaf sizes)', xlabel='n_estimators')
ax[1].legend(title='max_features, max_depth', fontsize=8); fig('fig7_hyperparameters')
""")

M("""
**Interpretation (Fig 7).** Logistic Regression shows the classic bias–variance shape: C = 0.01 underfits, the score rises steeply to C = 10 and dips at C = 100. For the forest, 300 trees add only 0.003 over 100, unlimited depth beats depth 20, and more candidate features per split (`0.05`) help on sparse text where a random `sqrt` subset often holds only uninformative terms. The best configurations are frozen here.
""")

C("""
f, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for a, (m, est) in zip(ax, best.items()):
    n, tr, va = learning_curve(est, X_tr, y_tr, train_sizes=np.linspace(0.1, 1.0, 5), cv=StratifiedKFold(3, shuffle=True, random_state=RS), scoring='f1_macro')
    a.plot(n, tr.mean(1), 'o-', label='training'); a.plot(n, va.mean(1), 's-', label='validation'); a.legend(); a.set(title=f'Fig 8  Learning curve: {m}', xlabel='training products', ylabel='macro-F1')
fig('fig8_learning_curves')
""")

M("""
**Interpretation (Fig 8).** Learning curves stand in for a loss curve (neither estimator exposes one). Both models fit the training data almost perfectly while validation scores still rise with more data: the gap is variance, so **more labelled products would help more than a more complex model**.
""")

M("""
## 7. Final evaluation on the untouched test set
Selected by **cross-validated** macro-F1; both frozen configurations and the dummy are evaluated once on the 20 % test set, without retuning.
""")

C("""
final = max(val_best, key=val_best.get); print('selected by CV macro-F1:', final, {k: round(float(v), 3) for k, v in val_best.items()})
finals = {'Dummy': Pipeline([('prep', prep(True)), ('model', DummyClassifier(strategy='most_frequent'))]).fit(X_tr, y_tr), **best}
pred = {m: est.predict(X_te) for m, est in finals.items()}
test = pd.DataFrame({m: dict(accuracy=skm.accuracy_score(y_te, p), balanced_accuracy=skm.balanced_accuracy_score(y_te, p), macro_f1=skm.f1_score(y_te, p, average='macro'))
                     for m, p in pred.items()}).T.round(3); test.to_csv(R + 'test_results.csv'); print(test)
rep = skm.classification_report(y_te, pred[final], digits=3); print(rep); open(R + 'classification_report.txt', 'w').write(rep)
f, a = plt.subplots(figsize=(6.5, 5.5)); skm.ConfusionMatrixDisplay.from_predictions(y_te, pred[final], ax=a, xticks_rotation='vertical', cmap='Blues', colorbar=False)
a.set_title(f'Fig 9  Confusion matrix, test set: {final}'); fig('fig9_confusion_matrix')
""")

C("""
p = pred[final]; err = X_te[p != y_te].assign(true=y_te[p != y_te], pred=p[p != y_te], name=dfc.product_name_de)
print(len(err), 'test errors; most frequent confusions:'); print(pd.crosstab(err.true, err.pred).stack().nlargest(5))
display(err.sample(6, random_state=RS)[['name', 'true', 'pred', 'ingredients_text']])
coef = pd.DataFrame(best['LogReg']['model'].coef_, index=best['LogReg'].classes_, columns=best['LogReg']['prep'].get_feature_names_out())
print('LogReg strongest terms per class:', {c: [t.split('__')[1] for t in coef.loc[c].nlargest(4).index] for c in coef.index})
""")

M("""
**Interpretation.** Both tuned models beat the majority baseline by a wide margin, and test scores sit within 0.01 of the CV estimates. The weakest class is the smallest and most planning-relevant one: **fruits & vegetables reach only 0.60 recall** – pickled or oil-preserved vegetables are read as sauces; nut and seed products swing between salty snacks and cereals, drinkable dairy towards beverages. Coefficients (*Joghurt/Milch* → dairy, *Kohlensäure/Apfelsaft* → beverages, but also noise tokens such as *90*) are **associations, not causal explanations**; low-confidence predictions and these pairs should go to staff correction.
""")

M("""
## 8. Compact kindergarten demonstration (rule-based, human-reviewed)
Helpers: `kita_demo.py`. Ingredient nutrients come from the **Bundeslebensmittelschlüssel 4.0** (Max Rubner-Institut, 2025, CC BY 4.0; per 100 g, raw/cooked state as named). Recipes, plans and child records are **self-authored fictional fixtures**. DGE rules are the food-frequency criteria of Tabelle 3 (lunch, 5 days, ages 1–<7) and the 20-day criteria on pp. 43/45 of the DGE-Qualitätsstandard (6th ed., 2023) – a *selected-criteria check*, not a certification. Allergen matching uses the 14 categories of Regulation (EU) 1169/2011 Annex II, distinguishes declared / traces / unknown, and cannot detect cross-contact.
""")

C("""
import kita_demo as kd
bls, fx = kd.load_bls('data/processed/bls40_subset.csv'), kd.load_fixtures('data/processed/demo_fixtures.json')
print([kd.parse_menu_line(l) for l in ['250 g Kartoffeln', '1 Stück Zwiebel']])
r = fx['recipes']['linsenbolognese']; n = kd.recipe_nutrition(r, fx['ingredients'], bls); print(r['name'], '-', n['basis']); display(n['per_serving'].round(2).to_frame('per serving').T)
sug = kd.product_features(['vollkornnudeln', 'joghurt35', 'fischstaebchen', 'apfel', 'rote_linsen', 'gouda30'], fx, bls)
sug['suggested_group'], sug['confidence'] = best[final].predict(sug), best[final].predict_proba(sug).max(1).round(2)
display(sug[['ingredient', 'suggested_group', 'confidence']].assign(review='required'))
display(kd.allergen_report(fx['recipes']['kaesespaetzle'], fx['ingredients'], fx['children']))
""")

C("""
display(kd.dge_week_check(fx['week1'], fx['recipes'])); display(kd.dge_cycle_check(fx['cycle4w'], fx['recipes']))
display(kd.substitution_candidates('milch15', fx['children'][0], fx, bls))
display(kd.period_summary(fx['week1'], fx, bls, 'Week 1 (5 lunch days, per child portion)'))
cyc = kd.period_summary(fx['cycle4w'], fx, bls, '4-week cycle (20 lunch days)'); cyc.to_csv(R + 'demo_cycle_summary.csv', index=False)
print(kd.run_checks(fx, bls).to_string(index=False))
""")

M("""
**Reading the output.** The parser returns *verify* for units without mass instead of inventing a weight; suggestions use OFF vocabulary (lentils → *cereals-and-potatoes*, low confidence). *Käsespätzle* is *contains* for the milk-, gluten- and egg-allergic children but *no declared match* for the lactose-intolerant child – deliberately **not** "safe". Week 1 passes all eleven Tabelle-3 criteria; the 4-week cycle fails *fettreicher Fisch ≥ 2×/20 days*. Substitutions are screened per child (lactose-free milk is blocked for the milk-protein allergy) and remain kitchen-review candidates; incomplete periods return *not assessed*.
""")

M("""
## 9. Discussion, implications, limitations and future work
**Findings.** Ingredient text plus declared nutrients predict the OFF food group with test macro-F1 0.86 (forest) and 0.83 (logistic regression) against 0.05 for the majority baseline; scaling was decisive for the linear model and irrelevant for the forest; tuning gained one to three points; learning curves show the models are data-limited.

**Implications.** As a suggestion engine with confidence-based triage the model can remove most manual grouping effort and make DGE frequency checks reproducible; it should run only behind human review.

**Limitations.** (1) Labels are crowd-sourced OFF categories – noisy and *not* DGE groups; (2) the unit is a packaged product, so test scores say nothing about menus or DGE compliance; (3) composite dishes excluded (optimistic for real shopping lists); (4) German scope only; (5) fibre missing for a third of products; (6) demo fixtures need kitchen verification; (7) no production food-safety validation.

**Future work.** Level-2 groups mapped to DGE groups, calibration and abstention thresholds, a manually verified test set, evaluation on real supplier lists.
""")

M("""
## 10. References (Harvard; URLs accessed 16 September 2026)
Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32.

Deutsche Gesellschaft für Ernährung (2023) *DGE-Qualitätsstandard für die Verpflegung in Kitas*, 6. Aufl., 2. korr. Nachdruck. https://www.dge.de/fileadmin/dok/gemeinschaftsgastronomie/dge-qualitaetsstandards/2023/230929-DGE-QST-Kita.pdf

European Parliament and Council (2011) *Regulation (EU) No 1169/2011 on the provision of food information to consumers*, Annex II. https://eur-lex.europa.eu/eli/reg/2011/1169/oj

Max Rubner-Institut (2025) *Bundeslebensmittelschlüssel (BLS), Version 4.0 – Deutsche Nährstoffdatenbank*, CC BY 4.0. DOI: 10.25826/Data20251217-134202-0. https://blsdb.de/download

Open Food Facts (2026) *Open Food Facts product database*, Parquet export, revision 8aea7223, ODbL v1.0. https://huggingface.co/datasets/openfoodfacts/product-database

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
