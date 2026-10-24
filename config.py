"""Paths and knobs. Run from the project folder.

Expected layout (the organisers' downloads, unzipped as-is):
  data/train/        transactions.parquet, transfers.parquet, blocks.parquet, market_rates.csv, labels.csv
  data/validation/   unlabelled practice split (used only for a dry run)
  data/test/         the scored test download (40,101 transactions)
Override with: python run.py --train data/train --test data/test
"""
import os
from pathlib import Path

SEED = 42
TRAIN_DIR = Path(os.environ.get("AML_TRAIN", "data/train"))
TEST_DIR = Path(os.environ.get("AML_TEST", "data/test"))
OUT_DIR = Path(os.environ.get("AML_OUT", "out"))

XRP_ASSET = "XRP"
DROPS_PER_XRP = 1_000_000

# A pair (sender -> receiver) seen this many times or fewer in the week is "one-off".
# Planted scenarios use one-off pairs; recurring commerce does not (median 1 vs 5).
RARE_PAIR_MAX = 2
# Time windows (seconds) used to stitch one-off payments that share a wallet into clusters.
# Easy scenarios run with ~60 s gaps, medium ~18 min, decoys ~10 min.
CLUSTER_WINDOWS = (120, 300)                  # cluster statistics (wider windows chain into giant clusters)
PROP_WINDOWS = (120, 300, 900, 1800, 3600)    # clusters used only to share scores in stage 2
EGO_WINDOWS = (180, 600, 1800, 3600)          # ego-neighbourhoods: 2 wallet-hops, no transitive chaining
LOCAL_WINDOWS = (60, 180, 600, 1800, 7200)
FLOW_WINDOWS = (3600, 4 * 3600, 12 * 3600)
RELAY_HANDOFF_S = 6 * 3600

N_FOLDS = 5
N_SEEDS = 2
LGB_PARAMS = dict(
    objective="binary",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    verbose=-1,
)
