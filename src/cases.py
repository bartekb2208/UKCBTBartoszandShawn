"""Pick example cases for the write-up from out-of-fold scores on train/ (run src/cv.py first)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from dd.load import load_split

D = Path("decode-and-detect-training-data/train")
df = load_split(D)
lab = pd.read_csv(D / "labels.csv").rename(columns={"TRANSACTION_HASH": "hash"})
oof = pd.read_csv("cache/oof_submission.csv")
d = df.merge(lab, on="hash").merge(oof[["hash", "oof"]], on="hash")
d["rank"] = d.oof.rank(ascending=False).astype(int)
d["cur"] = d.asset.str.split(":").str[-1]


def show(g, title):
    print(f"\n### {title}\n")
    t0 = g.ts.min(); names = {}
    nm = lambda w: names.setdefault(w, f"W{len(names) + 1}")
    print("| +min | from → to | USD | asset | legs | rank | hash |\n|---|---|---|---|---|---|---|")
    for x in g.sort_values("ts").itertuples():
        print(f"| {(x.ts - t0) / 60:.1f} | {nm(x.src)} → {nm(x.dst)} | {x.usd:,.2f} | {x.cur} | {x.legs} | {x.rank} | `{x.hash[:16]}…` |")


sc = d[d.scenario_id.notna() & d.cohort_id.isna()].groupby("scenario_id").agg(y=("is_suspicious", "max"), fam=("family", "first"),
                                                                            diff=("difficulty", "first"), r=("rank", "median"))
for fam in ("fan_in", "chain", "cycle"):
    s = sc[(sc.y == 1) & (sc.fam == fam)].sort_values("r").index[0]
    show(d[d.scenario_id == s], f"Flagged: planted {fam} (easy), median rank {sc.r[s]:.0f}")
s = sc[(sc.y == 0) & (sc.fam == "fan_in")].sort_values("r", ascending=False).index[0]
show(d[d.scenario_id == s], f"Not flagged: fan-in look-alike (decoy), median rank {sc.r[s]:.0f}")
s = sc[(sc.y == 1) & (sc['diff'] == "medium")].sort_values("r", ascending=False).index[0]
show(d[d.scenario_id == s], f"Missed: planted {sc.fam[s]} (medium), median rank {sc.r[s]:.0f}")
rel = d[d.cohort_id.notna()]
print(f"\nRelay families: {len(rel)} rows, {rel.is_suspicious.sum()} planted; median rank of planted relay rows {rel[rel.is_suspicious == 1]['rank'].median():.0f} of {len(d)}")
top = d.nsmallest(401, "rank")
print(f"Top 1% (401 rows): precision {top.is_suspicious.mean():.3f}; false positives by family: {top[top.is_suspicious == 0].family.value_counts().to_dict()}")
