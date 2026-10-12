"""
Analyse exploratoire du catalogue Web Gallery of Art (WGA) et sélection
du sous-ensemble "peinture" pour le projet de GAN génératif.

Entrée : data/art_catalog.xlsx - catalog.csv
Sorties :
  - data/paintings_only.csv : sous-ensemble FORM == "painting"
  - report_data.json        : agrégats utilisés par le rapport (artifact HTML)
"""
import re
import json
import numpy as np
import pandas as pd

SRC = r"data/art_catalog.xlsx - catalog.csv"

df = pd.read_csv(SRC)
form_counts = df["FORM"].value_counts().to_dict()

paint = df[df["FORM"] == "painting"].copy()
paint.to_csv(r"data/paintings_only.csv", index=False)

n_total, n_paint = len(df), len(paint)

type_counts = paint["TYPE"].value_counts().to_dict()

school_vc = paint["SCHOOL"].value_counts()
school_counts = school_vc.head(12).to_dict()
school_counts["Other (14 écoles)"] = int(school_vc.iloc[12:].sum())

tf_vc = paint["TIMEFRAME"].value_counts()
timeframe = [[t, int(tf_vc[t])] for t in sorted(tf_vc.index)]

author_vc = paint["AUTHOR"].value_counts()
top_authors = [[n, int(c)] for n, c in author_vc.head(12).items()]
labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
cuts = pd.cut(author_vc, bins=[0, 1, 2, 4, 9, 19, 49, 100000], labels=labels)
artist_dist = {k: int(v) for k, v in cuts.value_counts().reindex(labels).items()}

medium = paint["TECHNIQUE"].str.split(",").str[0].str.strip()
medium_vc = medium.value_counts()
medium_counts = medium_vc.head(10).to_dict()
medium_counts["Autres"] = int(medium_vc.iloc[10:].sum())

DIM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)


def parse_dims(text):
    m = DIM_RE.search(str(text))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


dims = paint["TECHNIQUE"].apply(parse_dims)
w = dims.apply(lambda t: t[0])
h = dims.apply(lambda t: t[1])
n_with_dims = int(w.notna().sum())

ar = (w / h).dropna()
ar_clean = ar[(ar > 0.1) & (ar < 5)]
counts, edges = np.histogram(ar_clean, bins=20)
ar_hist = [[f"{edges[i]:.2f}", int(counts[i])] for i in range(len(counts))]

h_clean = h[(h > 0) & (h < 600)].dropna()
counts, edges = np.histogram(h_clean, bins=20)
size_hist = [[f"{edges[i]:.0f}", int(counts[i])] for i in range(len(counts))]

date = paint["DATE"].astype(str).str.strip()
date_quality = {
    "Année exacte": int(date.str.match(r"^\d{4}$").sum()),
    "Plage d'années": int(date.str.contains(r"^\d{3,4}-\d{2,4}$").sum()),
    "Date approximative": int(date.str.contains(r"^c\.|ca\.", case=False, regex=True).sum()),
    "Décennie": int(date.str.contains(r"^\d{3,4}0s$").sum()),
    "Inconnue": int((date == "-").sum()),
}
date_quality["Autre format"] = n_paint - sum(date_quality.values())

loc_vc = paint["LOCATION"].value_counts()
top_locations = [[n, int(c)] for n, c in loc_vc.head(10).items()]

summary = {
    "n_total_works": n_total,
    "n_paintings": n_paint,
    "pct_paintings": round(n_paint / n_total * 100, 1),
    "n_authors": int(paint["AUTHOR"].nunique()),
    "n_schools": int(paint["SCHOOL"].nunique()),
    "n_locations": int(paint["LOCATION"].nunique()),
    "n_private_collection": int(loc_vc.get("Private collection", 0)),
    "n_with_dims": n_with_dims,
    "pct_with_dims": round(n_with_dims / n_paint * 100, 1),
    "form_counts": form_counts,
    "type_counts": type_counts,
    "school_counts": school_counts,
    "timeframe": timeframe,
    "top_authors": top_authors,
    "artist_dist": artist_dist,
    "medium_counts": medium_counts,
    "ar_hist": ar_hist,
    "size_hist": size_hist,
    "date_quality": date_quality,
    "top_locations": top_locations,
}

with open("report_data.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"{n_paint:,} peintures sur {n_total:,} œuvres ({summary['pct_paintings']}%)")
print(f"{summary['n_authors']} artistes, {summary['n_schools']} écoles, {summary['n_locations']} lieux")
print(f"Dimensions exploitables : {n_with_dims:,} lignes ({summary['pct_with_dims']}%)")
print("-> data/paintings_only.csv")
print("-> report_data.json")
