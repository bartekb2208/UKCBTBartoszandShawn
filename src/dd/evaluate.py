"""Out-of-fold scoring and reporting."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def ap(y, s):
    return 100 * average_precision_score(y, s)


def report(lab, oof, name=""):
    """Overall AP, per-fold AP and spread, per-family AP (family positives vs its own decoys)."""
    y = lab.is_suspicious.to_numpy()
    per_fold = [ap(y[lab.fold == k], oof[lab.fold == k]) for k in sorted(lab.fold.unique())]
    fam = {}
    for f, g in lab.groupby("family"):
        if g.is_suspicious.nunique() == 2:
            fam[f] = ap(g.is_suspicious, oof[g.index])
    # Rank of positives among everything: how many of each family make the top 1%.
    top = np.argsort(-oof)[: int(len(oof) * 0.01)]
    res = dict(name=name, ap=ap(y, oof), fold_mean=np.mean(per_fold), fold_sd=np.std(per_fold),
               folds=np.round(per_fold, 1).tolist(), p_at_1pct=y[top].mean() * 100, **{f"fam_{k}": v for k, v in fam.items()})
    return res


def print_report(res):
    print(f"[{res['name']}] AP {res['ap']:.2f} | fold mean {res['fold_mean']:.2f} ± {res['fold_sd']:.2f} {res['folds']} | P@1% {res['p_at_1pct']:.1f}")
    fams = {k[4:]: round(v, 1) for k, v in res.items() if k.startswith("fam_")}
    print("   family AP:", fams)
