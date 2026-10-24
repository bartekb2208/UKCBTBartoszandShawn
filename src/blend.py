"""Rank-blend our scores with Shawn's: pick the weight on pooled out-of-fold AP, apply to test CSVs.

Usage: python src/blend.py <shawn oof_arrays.npz> <our oof csv> <shawn test csv> <our test csv> <out csv>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from dd.evaluate import report, print_report
from dd.submit import rank_scores, validate

npz, our_oof_csv, shawn_test, our_test, out = sys.argv[1:6]
import load as shawn_load                     # Shawn's loader gives his row order
order = shawn_load.load_split("decode-and-detect-training-data/train")["tx"].to_numpy()
S = np.load(npz)
ours = pd.read_csv(our_oof_csv).set_index("hash")
lab = pd.read_parquet("cache/train_labels.parquet").set_index("hash")
df = pd.DataFrame({"shawn": pd.Series(S["blend"], index=order), "shawn_main": pd.Series(S["stage2"], index=order)}).join(ours[["oof", "fold"]])
df = df.join(lab[["is_suspicious", "family", "difficulty", "scenario_id", "cohort_id"]]).reset_index(drop=True)
r = lambda x: rankdata(x) / len(x)
print_report(report(df, df.oof.to_numpy(), "ours"))
print_report(report(df, df.shawn.to_numpy(), "shawn v3 blend"))
best = (None, -1)
for w in np.arange(0, 1.01, 0.1):
    s = w * r(df.oof) + (1 - w) * r(df.shawn)
    res = report(df, s, f"w_ours={w:.1f}")
    print(f"w_ours={w:.1f}  AP {res['ap']:.2f}  fold mean {res['fold_mean']:.2f}  folds {res['folds']}")
    if res["ap"] > best[1]:
        best = (w, res["ap"])
w = round(best[0], 1)
print_report(report(df, w * r(df.oof) + (1 - w) * r(df.shawn), f"best w_ours={w}"))
a = pd.read_csv(our_test).set_index("TRANSACTION_HASH"); b = pd.read_csv(shawn_test).set_index("TRANSACTION_HASH").reindex(a.index)
blend = w * r(a.score) + (1 - w) * r(b.score)
ranked = rank_scores(blend, a.index.to_numpy())
reason = np.where(ranked >= 0.98, a.reason.fillna("").to_numpy(), "")   # keep evidence notes only for the blended top 2%
pred = pd.DataFrame({"TRANSACTION_HASH": a.index, "score": ranked, "reason": reason})
validate(pred, "decode-and-detect-test-data")
pred.to_csv(out, index=False, float_format="%.8f")
print("wrote", out, "with w_ours =", w)
