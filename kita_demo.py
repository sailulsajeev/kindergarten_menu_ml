"""Compact, rule-based helpers for the kindergarten menu demonstration (no learned models).

Everything here is deterministic and grounded in two inputs:
  * data/processed/bls40_subset.csv  - nutrient values per 100 g edible portion from the
    Bundeslebensmittelschlüssel 4.0 (Max Rubner-Institut, 2025, CC BY 4.0)
  * data/processed/demo_fixtures.json - SELF-AUTHORED demonstration recipes, a fictional
    5-day plan, a fictional 4-week cycle, fictional child records and a reviewed
    substitution list. They are fixtures, not collected kindergarten records.
DGE criteria implemented are the food-frequency rules of Tabelle 3 (Mittagsverpflegung,
Mischkost, 5 Verpflegungstage, ages 1 to under 7) and the 20-day rules of
DGE-Qualitätsstandard für die Verpflegung in Kitas, 6. Auflage, 2. korr. Nachdruck 2023,
pp. 42-43 and p. 45. Output is a *selected-criteria check*, not a DGE certification.
"""
import json, re
import numpy as np, pandas as pd

BLS_MAP = {"ENERCC": "kcal", "PROT625": "protein_g", "FAT": "fat_g", "FASAT": "sat_fat_g", "CHO": "carbs_g",
           "SUGAR": "sugar_g", "FIBT": "fibre_g", "NA": "sodium_mg", "CA": "calcium_mg"}
NUTS = list(BLS_MAP.values())
LIMIT_STRINGS = ("<LOD", "<LOQ", "TR", "<LOD or <LOQ")   # measured below limit / trace -> 0, NOT unknown


def _num(v):
    if isinstance(v, str):
        v = v.strip()
        return 0.0 if v in LIMIT_STRINGS else pd.to_numeric(v.replace(",", "."), errors="coerce")
    return v


def load_bls(path):
    raw = pd.read_csv(path)
    out = pd.DataFrame({"name_de": raw.iloc[:, 1].values}, index=raw.iloc[:, 0].values)
    for c in raw.columns[3:]:
        code = c.split(" ")[0]
        if code in BLS_MAP:
            out[BLS_MAP[code]] = raw[c].map(_num).astype(float).values
    return out


def load_fixtures(path):
    return json.load(open(path, encoding="utf-8"))


UNIT_TO_G = {"g": 1.0, "kg": 1000.0, "ml": 1.0, "l": 1000.0}   # ml treated as g (density 1) -> flagged


def parse_menu_line(line):
    """Very small regex parser: '<number> <unit> <name>'. Units without a mass (Stück, EL, TL)
    give grams=None so the result asks for verification instead of guessing."""
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(g|kg|ml|l|Stück|Stk\.?|EL|TL)?\s+(.+?)\s*$", line, flags=re.I)
    if not m:
        return {"name": line.strip(), "grams": None, "note": "unparsed - verify"}
    qty, unit, name = float(m.group(1).replace(",", ".")), (m.group(2) or "").lower(), m.group(3)
    if unit in UNIT_TO_G:
        return {"name": name, "grams": qty * UNIT_TO_G[unit], "note": "ml assumed density 1" if unit in ("ml", "l") else ""}
    return {"name": name, "grams": None, "note": f"unit '{unit or 'none'}' has no mass - verify"}


def recipe_nutrition(recipe, ingredients, bls):
    """Totals and per-serving values from ingredient grams x BLS per-100 g values.
    NaN nutrient values stay unknown (never treated as zero); missing grams block the total."""
    rows, problems = [], []
    for it in recipe["ingredients"]:
        ing = ingredients[it["id"]]
        if it.get("grams") is None:
            problems.append(f"{ing['name_de']}: quantity missing - verify with kitchen"); continue
        rows.append(bls.loc[ing["bls_code"], NUTS].astype(float) * it["grams"] / 100.0)
    if problems or recipe.get("servings", 0) <= 0:
        return {"status": "needs_verification", "problems": problems or ["servings missing"]}
    tot = pd.concat(rows, axis=1).sum(axis=1, min_count=len(rows))  # NaN if any ingredient unknown
    per = tot / recipe["servings"]
    per["salt_g"] = per["sodium_mg"] * 2.5 / 1000.0                  # salt = sodium x 2.5, mg -> g
    unknown = [k for k in NUTS if pd.isna(per[k])]
    return {"status": "ok", "per_serving": per, "unknown_nutrients": unknown,
            "basis": "per serving, BLS values per 100 g edible portion in the stated raw/cooked state"}


def allergen_report(recipe, ingredients, children):
    """Match reviewed ingredient allergen information against fictional child restrictions.
    Statuses: 'contains' (declared), 'traces' (may contain), 'unknown' (no reviewed information),
    'no declared match' - which is NOT a safety guarantee (cross-contact is not modelled)."""
    out = []
    for ch in children:
        contains, traces, unknown = [], [], []
        for it in recipe["ingredients"]:
            ing = ingredients[it["id"]]
            if ing.get("allergen_info") == "unknown":
                unknown.append(ing["name_de"]); continue
            for r in ch["restrictions"]:
                key = r["allergen"]
                if key in ing.get("allergens_declared", []):
                    contains.append(f"{ing['name_de']} ({key})")
                elif key in ing.get("allergens_traces", []):
                    traces.append(f"{ing['name_de']} ({key})")
        status = "contains" if contains else "traces" if traces else "unknown" if unknown else "no declared match"
        out.append({"child": ch["name"], "recipe": recipe["name"], "status": status,
                    "detail": "; ".join(contains or traces or unknown) or "-"})
    return pd.DataFrame(out)


# DGE Tabelle 3 (5 lunch days) - frequency rules for Mischkost, ages 1 to under 7.
WEEK_RULES = [("cereals_potatoes_daily", "Getreide/Getreideprodukte/Kartoffeln", "== 5", "getreide_kartoffeln"),
              ("wholegrain_min1", "davon mind. 1x Vollkornprodukte", ">= 1", "vollkorn"),
              ("potato_products_max1", "davon max. 1x Kartoffelerzeugnisse", "<= 1", "kartoffelerzeugnis"),
              ("vegetables_daily", "Gemüse und Salat", "== 5", "gemuese_salat"),
              ("raw_veg_min2", "davon mind. 2x als Rohkost", ">= 2", "rohkost"),
              ("legumes_min1", "davon mind. 1x Hülsenfrüchte", ">= 1", "huelsenfruechte"),
              ("fruit_min2", "Obst", ">= 2", "obst"),
              ("whole_fruit_min1", "davon mind. 1x als Stückobst", ">= 1", "stueckobst"),
              ("dairy_min2", "Milch und Milchprodukte", ">= 2", "milchprodukt"),
              ("meat_max1", "Fleisch/Wurstwaren", "<= 1", "fleisch"),
              ("fish_1", "Fisch", "== 1", "fisch")]
# 20-day rules (Tabelle 3 footnotes p. 43 and criteria p. 45)
CYCLE_RULES = [("lean_meat_min2_20d", "mind. 2x mageres Muskelfleisch in 20 Tagen", ">= 2", "mageres_muskelfleisch"),
               ("fatty_fish_min2_20d", "mind. 2x fettreicher Fisch in 20 Tagen", ">= 2", "fettreicher_fisch"),
               ("fried_max4_20d", "frittierte/panierte Produkte max. 4x in 20 Tagen", "<= 4", "frittiert_paniert"),
               ("meat_alt_max4_20d", "industrielle Fleisch-/Fischalternativen max. 4x in 20 Tagen", "<= 4", "fleischalternative_industriell")]


def _day_flags(day, recipes):
    return set().union(*[set(recipes[r].get("dge_flags", [])) for r in day["recipes"]])


def _eval(rules, days, recipes, expected_days, period):
    rows = []
    for rid, label, cond, flag in rules:
        if len(days) != expected_days:
            rows.append({"criterion": label, "observed": None, "rule": cond, "status": "not assessed",
                         "note": f"{period} needs exactly {expected_days} lunch days, got {len(days)}"}); continue
        n = sum(flag in _day_flags(d, recipes) for d in days)
        op, val = cond.split(); ok = {"==": n == int(val), ">=": n >= int(val), "<=": n <= int(val)}[op]
        rows.append({"criterion": label, "observed": n, "rule": cond, "status": "pass" if ok else "fail", "note": ""})
    return pd.DataFrame(rows)


def dge_week_check(days, recipes):
    """Selected-criteria check for ONE week of 5 lunch days (DGE-QST Kita 2023, Tab. 3, pp. 42-43)."""
    return _eval(WEEK_RULES, days, recipes, 5, "week")


def dge_cycle_check(days, recipes):
    """Selected 20-day criteria (DGE-QST Kita 2023, p. 43 and p. 45). Weeks are checked separately."""
    return _eval(CYCLE_RULES, days, recipes, 20, "4-week cycle")


def substitution_candidates(ingredient_id, child, fixtures, bls):
    """Reviewed substitution candidates for one ingredient, screened against a child's restrictions
    and annotated with the nutrition change per 100 g. Candidates require kitchen review."""
    ings, rows = fixtures["ingredients"], []
    base = bls.loc[ings[ingredient_id]["bls_code"]]
    for s in fixtures["substitutions"]:
        if s["for"] != ingredient_id:
            continue
        cand = ings[s["candidate"]]
        hit = [r["allergen"] for r in child["restrictions"] if r["allergen"] in cand.get("allergens_declared", [])]
        alt = bls.loc[cand["bls_code"]]
        rows.append({"candidate": cand["name_de"], "screen": "blocked: contains " + ",".join(hit) if hit else
                     ("unknown allergen info" if cand.get("allergen_info") == "unknown" else "no declared match"),
                     "kcal/100g": f"{base['kcal']:.0f} -> {alt['kcal']:.0f}",
                     "protein g/100g": f"{base['protein_g']:.1f} -> {alt['protein_g']:.1f}",
                     "calcium mg/100g": f"{base['calcium_mg']:.0f} -> {alt['calcium_mg']:.0f}",
                     "reason / age note": s["reason"]})
    return pd.DataFrame(rows)


def period_summary(days, fixtures, bls, label):
    """Per-day, per-serving nutrition and DGE flags for a stated period (coverage made explicit)."""
    rows = []
    for d in days:
        parts = [recipe_nutrition(fixtures["recipes"][r], fixtures["ingredients"], bls) for r in d["recipes"]]
        ok = [p["per_serving"] for p in parts if p["status"] == "ok"]
        tot = pd.concat(ok, axis=1).sum(axis=1, min_count=len(ok)) if ok else pd.Series(dtype=float)
        rows.append({"period": label, "day": d["day"], "dishes": " + ".join(fixtures["recipes"][r]["name"] for r in d["recipes"]),
                     "kcal": tot.get("kcal", np.nan), "protein_g": tot.get("protein_g", np.nan), "fat_g": tot.get("fat_g", np.nan),
                     "carbs_g": tot.get("carbs_g", np.nan), "fibre_g": tot.get("fibre_g", np.nan),
                     "sugar_g": tot.get("sugar_g", np.nan), "salt_g": tot.get("salt_g", np.nan),
                     "complete": len(ok) == len(parts)})
    df = pd.DataFrame(rows)
    return df.round(1)


NUT_COLS = ["energy_kcal_100g", "fat_100g", "saturated_fat_100g", "carbohydrates_100g", "sugars_100g",
            "fiber_100g", "proteins_100g", "salt_100g"]


def product_features(ids, fixtures, bls):
    """Model-ready rows for demo ingredients that carry a fixture 'label_text' (an OFF-style German
    ingredient list) - nutrients come from the matched BLS entry, salt = sodium x 2.5 / 1000."""
    rows = []
    for i in ids:
        ing, b = fixtures["ingredients"][i], bls.loc[fixtures["ingredients"][i]["bls_code"]]
        vals = [b["kcal"], b["fat_g"], b["sat_fat_g"], b["carbs_g"], b["sugar_g"], b["fibre_g"], b["protein_g"], b["sodium_mg"] * 2.5 / 1000]
        rows.append({"ingredient": ing["name_de"], "ingredients_text": ing["label_text"], **dict(zip(NUT_COLS, vals))})
    return pd.DataFrame(rows)


def run_checks(fixtures, bls):
    """Focused checks for meaningful failure modes. Returns a DataFrame of check -> passed."""
    ings, recs = fixtures["ingredients"], fixtures["recipes"]
    res = []
    # 1 unit conversion: sodium mg -> salt g (400 mg Na = 1.0 g salt)
    r = {"name": "t", "servings": 1, "ingredients": [{"id": "gouda30", "grams": 100}]}
    per = recipe_nutrition(r, ings, bls)["per_serving"]
    res.append(("sodium(mg)->salt(g) conversion", abs(per["salt_g"] - bls.loc[ings["gouda30"]["bls_code"], "sodium_mg"] * 2.5 / 1000) < 1e-9))
    # 2 serving division
    r10 = dict(r, servings=10, ingredients=[{"id": "gouda30", "grams": 1000}])
    res.append(("serving division (1000 g / 10 == 100 g)", abs(recipe_nutrition(r10, ings, bls)["per_serving"]["kcal"] - per["kcal"]) < 1e-9))
    # 3 parser: kg->g and unit without mass
    res.append(("parser 0,5 kg -> 500 g", parse_menu_line("0,5 kg Kartoffeln")["grams"] == 500.0))
    res.append(("parser '1 Stück' -> grams None (verify)", parse_menu_line("1 Stück Zwiebel")["grams"] is None))
    # 4 missing quantity blocks totals
    res.append(("missing quantity -> needs_verification",
                recipe_nutrition({"name": "x", "servings": 4, "ingredients": [{"id": "gouda30", "grams": None}]}, ings, bls)["status"] == "needs_verification"))
    # 5 declared allergen match and unknown status
    kids = fixtures["children"]
    milk_child = next(c for c in kids if any(x["allergen"] == "milk" for x in c["restrictions"]))
    res.append(("declared allergen match (milk)", allergen_report(recs["kaesespaetzle"], ings, [milk_child]).status.iloc[0] == "contains"))
    unk = {"name": "u", "ingredients": [{"id": "gemuesebruehe", "grams": 5}]}
    res.append(("unknown allergen info reported as unknown", allergen_report(unk, ings, [milk_child]).status.iloc[0] == "unknown"))
    # 6 incomplete period -> not assessed
    res.append(("3-day week -> not assessed", (dge_week_check(fixtures["week1"][:3], recs).status == "not assessed").all()))
    res.append(("15-day cycle -> not assessed", (dge_cycle_check(fixtures["cycle4w"][:15], recs).status == "not assessed").all()))
    # 7 unknown nutrient stays unknown (NaN), not zero
    ok_nan = bls[NUTS].isna().any().any()
    res.append(("BLS unknown values kept as NaN (not zero)", bool(ok_nan)))
    return pd.DataFrame(res, columns=["check", "passed"])
