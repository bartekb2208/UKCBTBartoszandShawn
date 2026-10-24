"""Experiment: build features for train (cached) and report OOF AP."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from dd.features import build_features
from dd.model import oof_predict, fit
from dd.evaluate import report, print_report

CACHE = Path("cache")
df = pd.read_parquet(CACHE / "train_tx.parquet"); lab = pd.read_parquet(CACHE / "train_labels.parquet")
fx = CACHE / "train_X.parquet"
if "--rebuild" in sys.argv or not fx.exists():
    t0 = time.time(); X = build_features(df); X.to_parquet(fx); print(f"features {X.shape} in {time.time()-t0:.0f}s")
X = pd.read_parquet(fx)
y = lab.is_suspicious.to_numpy(); folds = lab.fold.to_numpy()
oof = oof_predict(X, y, folds)
res = report(lab, oof, "lgbm all"); print_report(res)
np.save(CACHE / "oof_lgbm.npy", oof)
imp = pd.Series(fit(X, y).booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print((imp / imp.sum()).head(25).round(3).to_string())
