"""Out-of-fold check of the blend weight. Not needed to reproduce predictions.csv.

Run from the ZIP root, in this order:
  python ours/src/run_baseline.py                                  -> cache/train_labels.parquet (labels and our fold map)
  python ours/src/cv.py                                            -> cache/oof_submission.csv   (model A out-of-fold scores)
  python analysis/capture_shawn_oof.py <train> <test> <out dir>    -> <out dir>/oof_arrays.npz   (model B out-of-fold scores)
  python analysis/blend_cv.py <out dir>/oof_arrays.npz <train dir>

Each model's scores are out-of-fold for its own training folds, but the two models use different fold
designs (A: cohort/scenario/sender groups; B: scenario groups), so the blended figure is an estimate,
not a clean cross-validation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shawn"))
sys.path.insert(0, str(ROOT / "ours" / "src"))
import load as model_b_loader  # noqa: E402
from dd.evaluate import report  # noqa: E402

npz, train_dir = sys.argv[1], sys.argv[2]
order = model_b_loader.load_split(train_dir)["tx"].to_numpy()  # row order of model B's arrays
b = pd.Series(np.load(npz)["blend"], index=order, name="b")
a = pd.read_csv("cache/oof_submission.csv").set_index("hash")
lab = pd.read_parquet("cache/train_labels.parquet").set_index("hash")
df = (pd.DataFrame(b).join(a[["oof", "fold"]])
      .join(lab[["is_suspicious", "family", "difficulty", "scenario_id", "cohort_id"]]).reset_index(drop=True))
assert df.oof.notna().all() and df.is_suspicious.notna().all(), "inputs do not cover the same transactions"


def rank01(x):
    return rankdata(x) / len(x)


for w in np.arange(0, 1.01, 0.1):
    res = report(df, w * rank01(df.oof) + (1 - w) * rank01(df.b), f"weight of A = {w:.1f}")
    print(f"weight of model A {w:.1f}: pooled AP {res['ap']:.2f}  fold mean {res['fold_mean']:.2f} +/- {res['fold_sd']:.2f}")
