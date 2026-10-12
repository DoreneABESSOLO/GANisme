import argparse
import pandas as pd

SEED = 42


def stratified_sample(paint: pd.DataFrame, target_total: int) -> pd.DataFrame:
    types = paint["TYPE"].unique()
    quota_per_type = max(1, target_total // len(types))

    picked = []
    for t in types:
        sub = paint[paint["TYPE"] == t]
        schools = sub["SCHOOL"].unique()
        per_school = max(1, quota_per_type // len(schools))

        chosen = sub.groupby("SCHOOL", group_keys=False).apply(
            lambda g: g.sample(min(len(g), per_school), random_state=SEED)
        )

        
        missing = quota_per_type - len(chosen)
        if missing > 0:
            remaining = sub.drop(chosen.index)
            top_up = remaining.sample(min(missing, len(remaining)), random_state=SEED)
            chosen = pd.concat([chosen, top_up])

        picked.append(chosen)

    return pd.concat(picked).sample(frac=1, random_state=SEED)  # mélange final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4000, help="taille cible du sous-échantillon")
    ap.add_argument("--src", default="data/paintings_only.csv")
    ap.add_argument("--out", default="data/subset_sample.csv")
    args = ap.parse_args()

    paint = pd.read_csv(args.src)
    subset = stratified_sample(paint, args.size)
    subset.to_csv(args.out, index=False)

    def dist(df, col):
        return (df[col].value_counts(normalize=True) * 100).round(1)

    print(f"Sous-échantillon : {len(subset):,} lignes -> {args.out}")
    print("\nTYPE — corpus complet vs sous-échantillon (%) :")
    print(pd.concat([dist(paint, "TYPE"), dist(subset, "TYPE")], axis=1,
                     keys=["complet", "sous-éch."]).fillna(0))
    print("\nSCHOOL (top 8) — corpus complet vs sous-échantillon (%) :")
    top_schools = paint["SCHOOL"].value_counts().head(8).index
    print(pd.concat([dist(paint, "SCHOOL"), dist(subset, "SCHOOL")], axis=1,
                     keys=["complet", "sous-éch."]).reindex(top_schools).fillna(0))


if __name__ == "__main__":
    main()
