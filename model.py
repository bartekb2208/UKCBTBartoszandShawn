"""v3 model.

1. Auxiliary models (trained on labels.csv annotation, out-of-fold for train rows):
   - aux_motif: is this payment part of a planted shape (laundering OR legitimate look-alike)?
   - aux_relay: is it part of a relay scenario?
2. Main LightGBM on all features + aux scores; stage 2 adds neighbours' out-of-fold scores.
3. Decoy expert: P(suspicious | planted shape), trained on planted-shape payments only.
   Mixture score = aux_motif x expert.
4. Final = rank blend of main and mixture (weight chosen on out-of-fold AP), strict ranking, no ties.
Validation: GroupKFold by scenario_id for everything, so no scenario sits on both sides of a split.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import config as C
from features import propagation_features

MOTIF_FAMILIES = ["bipartite", "chain", "cycle", "fan_in", "fan_out", "fan_out_in", "multi_stage", "random", "stack"]
RELAY_FAMILIES = ["aggregate_relay", "distributed_relay"]
AUX_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.6, verbose=-1)
EXPERT_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.6, verbose=-1)


def ap(y, p):
    return 100 * float(average_precision_score(y, p))


def make_groups(lab):
    rng = np.random.RandomState(C.SEED)
    bucket = pd.Series(rng.randint(0, 2000, len(lab)).astype(str), index=lab.index)
    return lab["scenario_id"].fillna(bucket).values


def _cv(X, y, folds, Xt, params, seeds, train_mask=None):
    """Out-of-fold predictions for X, averaged fold-model predictions for Xt."""
    oof, pt = np.zeros(len(X)), np.zeros(len(Xt))
    imp = pd.Series(0.0, index=X.columns)
    for sd in range(seeds):
        for tr, va in folds:
            fit_idx = tr if train_mask is None else tr[train_mask[tr]]
            m = lgb.LGBMClassifier(**params, random_state=C.SEED + sd, n_jobs=2)
            m.fit(X.iloc[fit_idx], y[fit_idx])
            oof[va] += m.predict_proba(X.iloc[va])[:, 1] / seeds
            pt += m.predict_proba(Xt)[:, 1] / (seeds * len(folds))
            imp += m.booster_.feature_importance("gain")
    return oof, pt, (imp / imp.sum()).sort_values(ascending=False)


def slices(y, p, lab):
    neg = y == 0
    out = {"ap": ap(y, p), "auc": 100 * roc_auc_score(y, p)}
    for dif in ("easy", "medium", "hard"):
        m = neg | (lab["difficulty"].values == dif)
        out[f"ap_{dif}"] = ap(y[m], p[m])
    k = max(1, len(y) // 100)
    out["top1pct_precision"] = float(y[np.argsort(-p)[:k]].mean())
    return out


def show(name, r):
    print(f"[cv] {name}: AP {r['ap']:.2f}  AUC {r['auc']:.1f}  easy {r['ap_easy']:.1f}  medium {r['ap_medium']:.1f}  "
          f"hard {r['ap_hard']:.1f}  top1% precision {r['top1pct_precision']:.2f}")


def per_family(y, p, lab):
    rows, neg = [], y == 0
    for (fam, dif), idx in lab[y == 1].groupby(["family", "difficulty"]).groups.items():
        m = neg.copy(); m[np.asarray(list(idx))] = True
        rows.append((fam, dif, len(idx), round(ap(y[m], p[m]), 1)))
    return pd.DataFrame(rows, columns=["family", "difficulty", "positives", "AP_vs_all_negatives"]).sort_values("AP_vs_all_negatives")


def strict_ranking(*keys):
    """Strict order in (0, 1]: no two transactions share a score (tied scores are grouped by the scorer)."""
    keys = [np.round(np.asarray(k, float), 12) for k in keys]
    order = np.lexsort((np.arange(len(keys[0])),) + tuple(reversed(keys)))
    out = np.empty(len(keys[0]))
    out[order] = (np.arange(len(order)) + 1) / len(order)
    return out


def train_and_predict(Ftr, ctx_tr, lab, Fte, ctx_te):
    y = lab["y"].values.astype(int)
    folds = list(GroupKFold(n_splits=C.N_FOLDS).split(Ftr, y, make_groups(lab)))
    print(f"[cv] {C.N_FOLDS} folds grouped by scenario; {y.sum()} positives of {len(y):,} ({y.mean():.2%})")
    res = {}

    # 1. auxiliary membership models
    motif = lab["family"].isin(MOTIF_FAMILIES).values.astype(int)
    relay = lab["family"].isin(RELAY_FAMILIES).values.astype(int)
    a_m, a_m_te, _ = _cv(Ftr, motif, folds, Fte, AUX_PARAMS, 1)
    a_r, a_r_te, _ = _cv(Ftr, relay, folds, Fte, AUX_PARAMS, 1)
    res["aux_motif_auc"] = 100 * roc_auc_score(motif, a_m)
    res["aux_relay_auc"] = 100 * roc_auc_score(relay, a_r)
    print(f"[aux] planted-shape membership AUC {res['aux_motif_auc']:.1f}, relay membership AUC {res['aux_relay_auc']:.1f}")
    Xtr = Ftr.assign(aux_motif=a_m, aux_relay=a_r)
    Xte = Fte.assign(aux_motif=a_m_te, aux_relay=a_r_te)

    # 2. main model, two stages
    o1, p1, imp = _cv(Xtr, y, folds, Xte, C.LGB_PARAMS, C.N_SEEDS)
    r1 = slices(y, o1, lab); show("main stage1", r1)
    X2 = pd.concat([Xtr, propagation_features(o1, ctx_tr)], axis=1)
    Xt2 = pd.concat([Xte, propagation_features(p1, ctx_te)], axis=1)
    o2, p2, imp2 = _cv(X2, y, folds, Xt2, C.LGB_PARAMS, C.N_SEEDS)
    r2 = slices(y, o2, lab); show("main stage2", r2)
    if r2["ap"] > r1["ap"]:
        main_oof, main_te, imp = o2, p2, imp2
    else:
        main_oof, main_te = o1, p1

    # 3. decoy expert on planted-shape payments only, then the mixture
    q, q_te, _ = _cv(Xtr, y, folds, Xte, EXPERT_PARAMS, C.N_SEEDS, train_mask=motif.astype(bool))
    res["expert_auc_within_shapes"] = 100 * roc_auc_score(y[motif == 1], q[motif == 1])
    mix_oof, mix_te = a_m * q, a_m_te * q_te
    rm = slices(y, mix_oof, lab); show("mixture", rm)
    print(f"[expert] laundering vs look-alike AUC within planted shapes: {res['expert_auc_within_shapes']:.1f}")

    # 4. blend weight on out-of-fold AP
    best_w, best = 1.0, -1
    for w in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        a = ap(y, w * rankdata(main_oof) + (1 - w) * rankdata(mix_oof))
        if a > best + 0.05:
            best_w, best = w, a
    blend_oof = best_w * rankdata(main_oof) / len(y) + (1 - best_w) * rankdata(mix_oof) / len(y)
    blend_te = best_w * rankdata(main_te) / len(main_te) + (1 - best_w) * rankdata(mix_te) / len(mix_te)
    rb = slices(y, blend_oof, lab); show(f"blend (w_main={best_w})", rb)

    res.update({f"main_{k}": v for k, v in (r2 if r2["ap"] > r1["ap"] else r1).items()})
    res.update({f"mixture_{k}": v for k, v in rm.items()})
    res.update({f"final_{k}": v for k, v in rb.items()})
    res["blend_weight_main"] = best_w
    variants = {
        "submission.csv": strict_ranking(blend_te, main_te, mix_te),
        "submission_main_only.csv": strict_ranking(main_te, p1),
        "submission_mixture_only.csv": strict_ranking(mix_te, main_te),
    }
    return variants, blend_oof, imp, res, per_family(y, blend_oof, lab)
