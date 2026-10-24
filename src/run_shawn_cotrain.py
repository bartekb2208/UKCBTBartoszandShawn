"""Run model B (shawn/) with our relay-motif features, and let its self-training take pseudo-labels from an
external teacher ranking (co-training) instead of model B's own blend.

Usage: python src/run_shawn_cotrain.py <train dir> <test dir> <out dir> <teacher csv>
Model B's files are not edited: build_features and self_train are wrapped at run time.
"""
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
order = model_b_load.load_split(test_dir)["tx"].to_numpy()          # model B's test row order
teacher = pd.read_csv(teacher_csv).set_index("TRANSACTION_HASH").score.reindex(order).to_numpy()
assert not np.isnan(teacher).any(), "teacher does not cover the test transactions"
_self_train = model_b_model.self_train


def self_train_from_teacher(X, y, Xt, score_te, **kw):
    return _self_train(X, y, Xt, teacher, **kw)


model_b_model.self_train = self_train_from_teacher
run_shawn_with_motifs.main(train_dir, test_dir, out_dir)
