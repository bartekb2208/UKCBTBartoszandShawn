"""Fixed 5-fold grouped, stratified fold map for train/.

Train has only 700 wallets in one connected component, and each wallet appears
in about 24 scenarios, so wallet-cluster grouping is impossible. Instead:
relay rows group by cohort (test cohorts are unseen, so a fold must hold out
whole cohorts), other planted rows by scenario, background rows by sender.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5
SEED = 20261024


def fold_groups(labels):
    g = labels.scenario_id.copy()
    relay = labels.cohort_id.notna()
    g[relay] = "cohort:" + labels.cohort_id[relay]
    bg = g.isna()
    g[bg] = "sender:" + labels.src[bg]
    return g


def make_folds(labels):
    """labels needs hash, src, is_suspicious, scenario_id, cohort_id. Returns fold per row."""
    groups = fold_groups(labels)
    # Stratify on family so each fold sees every family (and roughly 250 positives).
    strat = labels.family.where(labels.is_suspicious == 1, "neg_" + labels.family)
    folds = np.full(len(labels), -1)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, idx) in enumerate(skf.split(labels, strat, groups)):
        folds[idx] = k
    return pd.Series(folds, index=labels.index, name="fold")
