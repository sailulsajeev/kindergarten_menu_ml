"""Step 2 - Build the reproducible modelling subset from the Kaggle Open Food Facts extract.

Input : data/raw/off_kaggle_extract.parquet   (output of scripts/01_download_kaggle_off.py)
Output: data/processed/off_foodgroups_sample.csv   (the file the notebook loads)
        results/data_funnel.json                   (row counts after each filter)

Filters (documented in dataset_card.md):
  F1  product has a known PNNS level-1 food group (pnns_groups_1 not empty / 'unknown'); spelling variants unified
  F2  ingredient list present (>= 10 characters after removing HTML/underscore allergen markup)
  F3  energy, fat, carbohydrates and proteins per 100 g all declared
  F4  level-1 group is one of the 8 kindergarten-relevant, non-composite groups (composite foods excluded)
  F5  duplicates: products with an identical cleaned ingredient list form a group; groups with conflicting
      labels are dropped, otherwise ONE representative (most recently modified) is kept
  F6  stratified random sample of SAMPLE_N products (RANDOM_STATE) - not the first N rows
Energy: the Kaggle file stores energy_100g in kJ; it is converted to kcal (/ 4.184) as energy_kcal_100g.
A rough language guess (stop-word heuristic, for reporting only) is stored as lang_guess.
"""
import os, re, json, html
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "off_kaggle_extract.parquet")
OUT = os.path.join(ROOT, "data", "processed", "off_foodgroups_sample.csv")
RANDOM_STATE, SAMPLE_N = 42, 12000
CANON = {"fruits and vegetables": "fruits-and-vegetables", "cereals and potatoes": "cereals-and-potatoes",
         "milk and dairy products": "milk-and-dairy-products", "fish meat eggs": "fish-meat-eggs",
         "fat and sauces": "fats-and-sauces", "sugary snacks": "sugary-snacks", "salty snacks": "salty-snacks",
         "beverages": "beverages", "composite foods": "composite-foods",
         "fruits-and-vegetables": "fruits-and-vegetables", "sugary-snacks": "sugary-snacks", "cereals-and-potatoes": "cereals-and-potatoes"}
KEEP = ["fruits-and-vegetables", "cereals-and-potatoes", "milk-and-dairy-products", "fish-meat-eggs",
        "fats-and-sauces", "sugary-snacks", "salty-snacks", "beverages"]
NUM = {"energy_100g": "energy_kj_100g", "fat_100g": "fat_100g", "saturated-fat_100g": "saturated_fat_100g",
       "carbohydrates_100g": "carbohydrates_100g", "sugars_100g": "sugars_100g", "fiber_100g": "fiber_100g",
       "proteins_100g": "proteins_100g", "salt_100g": "salt_100g"}
LANG = {"fr": r"\b(sucre|sel|eau|farine|huile|lait|arôme|arome|émulsifiant|poudre)\b",
        "en": r"\b(sugar|salt|water|flour|oil|milk|flavou?r|emulsifier|powder)\b",
        "de": r"\b(zucker|salz|wasser|mehl|öl|milch|aroma|emulgator|pulver)\b",
        "es": r"\b(azúcar|azucar|sal|agua|harina|aceite|leche|aroma|emulgente)\b"}

def clean_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = html.unescape(re.sub(r"<[^>]+>", " ", str(s))).replace("_", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

df = pd.read_parquet(RAW)
funnel = {"F0_kaggle_products": len(df)}
df["food_group_l1"] = df.pnns_groups_1.str.strip().str.lower().map(CANON)
df = df[df.food_group_l1.notna()]; funnel["F1_known_food_group"] = len(df)
df["ingredients_text"] = df.ingredients_text.apply(clean_text)
df = df[df.ingredients_text.str.len() >= 10]; funnel["F2_has_ingredients"] = len(df)
for c in NUM: df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df[["energy_100g", "fat_100g", "carbohydrates_100g", "proteins_100g"]].notna().all(axis=1)]
funnel["F3_has_core_nutrients"] = len(df)
df = df[df.food_group_l1.isin(KEEP)]; funnel["F4_kept_classes"] = len(df)

key = df.ingredients_text.str.lower().str.replace(r"[^a-zäöüßéèêàçñ0-9%]+", " ", regex=True).str.strip()
df["dup_group"] = key
grp = df.groupby("dup_group").food_group_l1.nunique()
funnel["F5a_duplicate_groups_total"] = int((df.dup_group.value_counts() > 1).sum())
funnel["F5b_conflicting_label_groups_dropped"] = int((grp > 1).sum())
df = df[~df.dup_group.isin(grp[grp > 1].index)]
df = df.sort_values("last_modified_datetime", ascending=False).drop_duplicates("dup_group", keep="first")
funnel["F5c_after_dedup_one_per_group"] = len(df)

df = (df.groupby("food_group_l1").sample(frac=SAMPLE_N / funnel["F5c_after_dedup_one_per_group"], random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE))
funnel["F6_stratified_sample"] = len(df)

low = df.ingredients_text.str.lower()
scores = pd.DataFrame({k: low.str.count(v) for k, v in LANG.items()})
df["lang_guess"] = np.where(scores.max(axis=1) > 0, scores.idxmax(axis=1), "other")
df = df.rename(columns=NUM)
df["energy_kcal_100g"] = (df.energy_kj_100g / 4.184).round(1)
cols = ["code", "product_name", "brands", "countries_tags", "lang_guess", "ingredients_text", "food_group_l1", "pnns_groups_2",
        "allergens", "traces_tags", "serving_size", "last_modified_datetime", "energy_kcal_100g"] + [v for v in NUM.values() if v != "energy_kj_100g"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df[cols].to_csv(OUT, index=False)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(funnel, open(os.path.join(ROOT, "results", "data_funnel.json"), "w"), indent=2)
print(json.dumps(funnel, indent=2)); print(df.food_group_l1.value_counts()); print(df.lang_guess.value_counts()); print(OUT, os.path.getsize(OUT))
