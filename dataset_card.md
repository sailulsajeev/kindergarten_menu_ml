# Dataset card and task specification

## 1. Sources

| Source | Publisher | Retrieved | Version / snapshot | Licence | Use in project |
|---|---|---|---|---|---|
| Open Food Facts on Kaggle – `openfoodfacts/world-food-facts`, file `en.openfoodfacts.org.products.tsv` | Open Food Facts (openfoodfacts.org), published on Kaggle | 2026-09-16, programmatic and anonymous via `kagglehub.dataset_download` | Kaggle dataset version 5 (2017 snapshot); 1,010,256,825 bytes; SHA-256 `d1f351a7e2df9f0606126ed82feed45c39b836de5744d792090f1d3dbe7987bf`; 356,027 products × 163 columns | Open Database License (ODbL) v1.0 | Supervised experiment (labels, ingredient text, nutrients) |
| OFF food-groups taxonomy `taxonomies/food_groups.txt` | Open Food Facts, openfoodfacts-server repository | 2026-09-16 | copy in `refs/food_groups.txt` | ODbL / AGPL (repo) | Documentation of the PNNS/food-group hierarchy |
| Bundeslebensmittelschlüssel (BLS) 4.0 | Max Rubner-Institut, Karlsruhe | 2026-09-16 | `BLS_4_0_2025_DE.zip` (Dezember 2025), 14,263,306 bytes, SHA-256 `12b7a6ba62807ec9b301eb276f897dc85f99b2292311618dec3749a12d984c91`; 7,140 foods × 138 nutrients | CC BY 4.0 – "Max Rubner-Institut (2025): Bundeslebensmittelschlüssel (BLS), Version 4.0 — Deutsche Nährstoffdatenbank. Karlsruhe. DOI: 10.25826/Data20251217-134202-0" | Nutrients of recipe ingredients in the demo |
| DGE-Qualitätsstandard für die Verpflegung in Kitas, 6. Auflage, 2. korrigierter und aktualisierter Nachdruck 2023 | Deutsche Gesellschaft für Ernährung e. V. | 2026-09-16 | PDF `230929-DGE-QST-Kita.pdf`, 100 pages | © DGE (reference only; not redistributed) | Selected menu-planning criteria (Tabelle 3, pp. 42–43; 20-day rules p. 43 and p. 45) |
| Regulation (EU) No 1169/2011, Annex II | EUR-Lex | 2026-09-16 | consolidated text 02011R1169-20180101 | EU public documents | 14 allergen categories |

**Alternatives inspected.** Kaggle `irkaal/foodcom-recipes-and-reviews` (522,517 English recipes, 311 free-text categories, nutrition per recipe without a per-100 g basis) and Kaggle `trolukovich/nutritional-values-for-common-foods-and-products` (8,789 foods, nutrients only, no ingredient text or label) were downloaded and inspected and rejected. The Hugging Face Parquet export of the same Open Food Facts database (revision `8aea7223`, 2026, 4.7 M products with a `lang` column) was used in the first version of this project (350,618 German-language products); it was replaced by the Kaggle snapshot at the student's request so that the dataset comes from Kaggle. The consequence is a mixed-language corpus (see §6).

## 2. Files produced

| File | Rows | Content |
|---|---|---|
| `data/raw/off_kaggle_extract.parquet` (local, 42 MB) | 356,027 | 22 columns of the Kaggle TSV, values unchanged (`scripts/01_download_kaggle_off.py`) |
| `data/raw/extract_meta.json` | – | source size, SHA-256, Kaggle version, row counts |
| `data/processed/off_foodgroups_sample.csv` | 12,001 | modelling subset (`scripts/02_build_subset.py`), redistributed under ODbL with attribution |
| `data/processed/bls40_subset.csv` | 7,140 | BLS code, names, 12 nutrient columns, CC BY 4.0 |
| `data/processed/demo_fixtures.json` | – | self-authored recipes, 5-day plan, 4-week cycle, fictional children, substitution list |

## 3. Unit of observation and target

* **Unit:** one packaged food product listed in Open Food Facts (product level – not an ingredient, not a menu).
* **Target `food_group_l1`:** the PNNS level-1 food group stored by OFF in `pnns_groups_1`, which OFF derives deterministically from the product's crowd-sourced **category** tags. Spelling variants in the 2017 file ("Fruits and vegetables" / "fruits-and-vegetables") were unified. Eight classes are kept:
  `sugary-snacks`, `beverages`, `milk-and-dairy-products`, `cereals-and-potatoes`, `fish-meat-eggs`, `fats-and-sauces`, `fruits-and-vegetables`, `salty-snacks`.
* **Excluded classes:** `composite-foods` (pizza, one-dish meals, sandwiches – 6,552 labelled products) because a mixed dish has no single food group, and `unknown`. Effect: reported performance describes single-group products only and is optimistic for real shopping lists that contain ready meals.

## 4. Filters (funnel, `results/data_funnel.json`)

| Step | Products |
|---|---|
| F0 products in the Kaggle file | 356,027 |
| F1 with a known PNNS level-1 group | 85,402 |
| F2 with an ingredient list (≥ 10 characters after removing HTML/underscore allergen markup) | 71,415 |
| F3 with energy, fat, carbohydrates and proteins per 100 g declared | 61,872 |
| F4 in one of the 8 kept classes | 55,320 |
| F5 duplicate handling: 2,859 groups share an identical cleaned ingredient list; 81 groups with conflicting labels dropped; one representative per group kept (most recently modified) | 47,949 |
| F6 stratified random sample, seed 42 | 12,001 |

Duplicates are merged **before** the train/test split so that product variants cannot appear on both sides. Energy is stored in kJ in the Kaggle file and converted to kcal (÷ 4.184).

## 5. Features

| Feature | Type | Source field | Note |
|---|---|---|---|
| `ingredients_text` | text | `ingredients_text` | HTML/underscore allergen markup stripped; OCR noise and several languages remain |
| `energy_kcal_100g` | numeric, kcal/100 g (100 ml for liquids) | `energy_100g` (kJ) ÷ 4.184 | |
| `fat_100g`, `saturated_fat_100g`, `carbohydrates_100g`, `sugars_100g`, `fiber_100g`, `proteins_100g`, `salt_100g` | numeric, g/100 g | same-named `*_100g` fields | missing = not declared (kept as NaN + indicator) |

**Excluded on purpose (leakage or proxies):** `categories_tags`, `main_category`, `pnns_groups_2`, Nutri-Score fields, product names, brands, codes, countries. Product names were excluded because they often *are* the category ("Vollmilch", "Jus d'orange"); `lang_guess` is a reporting column only, not a feature.

## 6. Language scope and label limitations

* The Kaggle file has no language column. A stop-word heuristic (`lang_guess`) estimates the sample as ~69 % French, 6 % English, 5 % German, 4 % Spanish and 16 % undetermined. The model therefore learns a predominantly French vocabulary; German ingredient lists (the Kita use case) are under-represented, which is stated in the notebook and is the first item of future work.
* Categories are crowd-sourced: unsalted nuts sit under *salty-snacks*, legumes under *cereals-and-potatoes*, plant-based milk substitutes under *beverages*. The model learns OFF's conventions, not a nutritionist's.
* 81 duplicate groups carried contradictory labels and were removed; further contradictions among non-duplicate products remain.
* The snapshot is from 2017; product formulations and OFF's taxonomy have changed since.

## 7. PNNS food group → DGE Lebensmittelgruppe (used only for interpretation, not for training)

| PNNS level 1 | DGE-QST group (Tabelle 3) | Comment |
|---|---|---|
| cereals-and-potatoes | Getreide, Getreideprodukte, Kartoffeln | legumes are in this group but count as *Hülsenfrüchte* (Gemüse) for DGE |
| fruits-and-vegetables | Gemüse und Salat / Obst | PNNS does not separate fruit from vegetables at level 1 |
| milk-and-dairy-products | Milch und Milchprodukte | includes ice cream and sweetened desserts, which DGE would not count |
| fish-meat-eggs | Fleisch, Wurst, Fisch und Eier | DGE separates fish from meat |
| fats-and-sauces | Öle und Fette | dressings/sauces have no DGE group |
| beverages | Getränke | juices/sweetened drinks are not recommended by DGE |
| sugary-snacks, salty-snacks | – (not a DGE lunch group) | limited by DGE rules on sweets/fried products |

## 8. Intended use and prohibited interpretations

* Intended: suggesting a food group for a packaged product so that a person can confirm or correct it; demonstration of how confirmed groups feed rule-based nutrition, allergen and DGE frequency checks.
* Not to be interpreted as: menu-level DGE compliance, "healthy"/"allergy-safe" labels, detection of undeclared allergens or kitchen cross-contact, nutrition values for complete dishes (the demo totals depend on self-authored quantities that a kitchen must verify), or performance on German-language product data.
