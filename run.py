"""Exact command:  python run.py --train data/train --test data/test
Writes out/submission.csv (TRANSACTION_HASH,score) plus validation results and case-study tables.
Dry run without the test download:  python run.py --test data/validation
CPU only, about 10 minutes on a 2-core machine. Seed 42.
"""
import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import load
from features import build_features
from model import train_and_predict
from scenarios import link_features

warnings.filterwarnings("ignore")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--train", default=str(C.TRAIN_DIR))
    a.add_argument("--test", default=str(C.TEST_DIR))
    a.add_argument("--out", default=str(C.OUT_DIR))
    args = a.parse_args()
    random.seed(C.SEED); np.random.seed(C.SEED)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tr = load.load_split(args.train)
    te = load.load_split(args.test)
    lab = tr[["tx"]].merge(load.load_labels(args.train), on="tx", how="left")
    assert lab["y"].notna().all(), "every training transaction needs a label"

    Ftr, ctx_tr = build_features(tr)
    Fte, ctx_te = build_features(te)
    Ltr, Lte = link_features(tr, ctx_tr, Ftr, lab["scenario_id"].values, te, ctx_te, Fte, seed=C.SEED)
    Ftr, Fte = pd.concat([Ftr, Ltr], axis=1), pd.concat([Fte, Lte], axis=1)
    print(f"[feat] {Ftr.shape[1]} features in {time.time() - t0:.0f}s")

    variants, oof, imp, res, fam = train_and_predict(Ftr, ctx_tr, lab, Fte, ctx_te)
    for name, scores in variants.items():
        sub = pd.DataFrame({"TRANSACTION_HASH": te["tx"].values, "score": scores})
        assert sub["TRANSACTION_HASH"].is_unique and sub["score"].between(0, 1).all() and np.isfinite(sub["score"]).all()
        assert sub["score"].nunique() == len(sub), "ties found"
        sub.to_csv(out / name, index=False, float_format="%.10f")
    final_te = variants["submission.csv"]

    res["seconds"] = round(time.time() - t0)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    fam.to_csv(out / "validation_by_family.csv", index=False)
    imp.head(40).to_csv(out / "feature_importance.csv", header=["gain_share"])
    # case-study material for the report: top flagged test payments with readable evidence
    cols = ["log_usd", "pair_n", "c300_size", "c300_gap_med", "c300_dur", "e600_size", "e600_gap_med", "e600_wal",
            "s_rare_600", "d_rare_600", "all_fwd_dt", "all_fwd_amt_ratio", "is_conv"]
    cs = te[["tx", "src", "dst", "t", "usd", "asset_out"]].join(Fte[cols]).assign(score=final_te)
    cs.sort_values("score", ascending=False).head(50).to_csv(out / "top_flagged_test.csv", index=False)
    cs[(Fte["c300_size"] >= 4) & (cs["score"] < 0.5)].sort_values("score").head(30).to_csv(out / "clusters_not_flagged.csv", index=False)
    print(json.dumps(res, indent=2))
    print(fam.to_string(index=False))
    print(f"[done] {out}: {', '.join(variants)} ({len(te):,} rows each, all scores unique)")


if __name__ == "__main__":
    main()
