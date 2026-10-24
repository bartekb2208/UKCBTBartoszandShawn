"""Run model B (shawn/) unchanged except that its feature table also gets our relay-motif features.

Usage: python src/run_shawn_with_motifs.py <train dir> <test dir> <out dir>
Model B's files are not edited: its build_features is wrapped at run time.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dd.motifs import motif_features  # noqa: E402

for cand in (HERE.parent.parent / "shawn", HERE.parent):     # zip layout, then repo layout
    if (cand / "scenarios.py").exists():
        sys.path.insert(0, str(cand))
        break
import features as model_b_features  # noqa: E402
import run as model_b_run  # noqa: E402

_original = model_b_features.build_features


def build_with_motifs(e):
    F, ctx = _original(e)
    M = motif_features(e.rename(columns={"t": "ts"}))   # model B's table is time-sorted, one row per payment
    M.index = F.index
    return pd.concat([F, M.add_prefix("mf_")], axis=1).astype(float), ctx


model_b_features.build_features = build_with_motifs
model_b_run.build_features = build_with_motifs              # run.py imported the name directly
sys.argv = ["run.py", "--train", sys.argv[1], "--test", sys.argv[2], "--out", sys.argv[3]]
model_b_run.main()
