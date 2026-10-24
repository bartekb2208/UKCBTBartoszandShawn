"""Run model B (shawn/) with our relay-motif features, and let its self-training take pseudo-labels from an
external teacher ranking (co-training) instead of model B's own blend.

Usage: python src/run_shawn_cotrain.py <train dir> <test dir> <out dir> <teacher csv> [--no-asset]
  --no-asset  also drop model B's asset and conversion columns (the organiser states asset and conversion
              choices do not inspect labels; model A already leaves them out)
Model B's files are not edited: build_features and self_train are wrapped at run time.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_shawn_with_motifs  # noqa: E402,F401  (side effect: model B importable, motif features wired in)

import load as model_b_load  # noqa: E402
import model as model_b_model  # noqa: E402

train_dir, test_dir, out_dir, teacher_csv = sys.argv[1:5]
ASSET = r"log_usd_paid|spread|out_(XRP|GBP|EUR|USD)|in_(XRP|GBP|EUR|USD)|is_conv|n_legs|n_assets|is_path|has_sendmax|has_paths|c\d+_n_assets"
if "--no-asset" in sys.argv[5:]:
    import run as model_b_run_module
    _with_motifs = model_b_run_module.build_features

    def build_without_assets(e):
        F, ctx = _with_motifs(e)
        return F.drop(columns=[c for c in F.columns if re.fullmatch(ASSET, c)]), ctx

    model_b_run_module.build_features = build_without_assets
order = model_b_load.load_split(test_dir)["tx"].to_numpy()          # model B's test row order
teacher = pd.read_csv(teacher_csv).set_index("TRANSACTION_HASH").score.reindex(order).to_numpy()
assert not np.isnan(teacher).any(), "teacher does not cover the test transactions"
_self_train = model_b_model.self_train


def self_train_from_teacher(X, y, Xt, score_te, **kw):
    return _self_train(X, y, Xt, teacher, **kw)


model_b_model.self_train = self_train_from_teacher
try:
    run_shawn_with_motifs.main(train_dir, test_dir, out_dir)
except KeyError as err:
    # without asset columns, model B's optional case-study table (written after every submission file) has a
    # missing column; the predictions are complete at that point
    if "--no-asset" not in sys.argv[5:]:
        raise
    print(f"[cotrain] skipped model B's case-study table ({err}); all submission files were written")
