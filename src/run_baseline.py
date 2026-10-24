"""Build the train table and fold map, then score the starter baseline under our folds."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from dd.load import load_split
from dd.folds import make_folds
from dd.baseline import starter_features, starter_model
from dd.evaluate import report, print_report

DATA = Path("decode-and-detect-training-data")
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)

t0 = time.time()
df = load_split(DATA / "train")
lab = pd.read_csv(DATA / "train" / "labels.csv").rename(columns={"TRANSACTION_HASH": "hash"})
lab = df[["hash", "src"]].merge(lab, on="hash", how="left", validate="1:1")
assert lab.is_suspicious.notna().all()
lab["fold"] = make_folds(lab).to_numpy()
df.to_parquet(CACHE / "train_tx.parquet"); lab.to_parquet(CACHE / "train_labels.parquet")
lab[["hash", "fold"]].to_csv(CACHE / "fold_map.csv", index=False)
print(f"loaded {len(df)} rows in {time.time()-t0:.1f}s")
print(lab.groupby("fold").agg(n=("hash", "size"), pos=("is_suspicious", "sum")).T)
print(pd.crosstab(lab.family[lab.is_suspicious == 1], lab.fold[lab.is_suspicious == 1]))

X = starter_features(df); y = lab.is_suspicious.to_numpy()
oof = np.zeros(len(df))
for k in range(5):
    m = starter_model().fit(X[lab.fold != k], y[lab.fold != k])
    oof[lab.fold == k] = m.predict_proba(X[lab.fold == k])[:, 1]
print_report(report(lab, oof, "starter LR"))
