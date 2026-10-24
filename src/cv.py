"""Out-of-fold AP of the exact submission recipe (features + seed-bagged LightGBM) on train/."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from dd.evaluate import report, print_report
from dd.features import build_features
from dd.folds import make_folds
from dd.load import load_split
from predict import score

D = Path(sys.argv[1] if len(sys.argv) > 1 else "decode-and-detect-training-data/train")
df = load_split(D)
lab = pd.read_csv(D / "labels.csv").rename(columns={"TRANSACTION_HASH": "hash"})
lab = df[["hash", "src"]].merge(lab, on="hash", how="left", validate="1:1")
lab["fold"] = make_folds(lab).to_numpy()
X = build_features(df)
oof = np.zeros(len(df))
for k in range(5):
    tr, te = lab.fold != k, lab.fold == k
    oof[te.to_numpy()] = score(X[tr], lab[tr].reset_index(drop=True), X[te])
res = report(lab, oof, "submission recipe"); print_report(res)
pd.DataFrame({"hash": lab.hash, "fold": lab.fold, "oof": oof}).to_csv("cache/oof_submission.csv", index=False)
