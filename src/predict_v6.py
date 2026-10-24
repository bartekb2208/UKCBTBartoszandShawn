"""v6 pipeline: model A + observable-hard relay-motif features, optionally with Shawn's behavioural features.

Usage: python src/predict_v6.py --test <test dir> --out predictions.csv [--union]
  --union   also feed the behavioural features of model B (shawn/features.py; its asset and conversion
            columns are dropped, as for model A)
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dd.conservation import conservation_features  # noqa: E402
from dd.features import build_features, prepare  # noqa: E402
from dd.load import load_split  # noqa: E402
from dd.motifs import motif_features  # noqa: E402
from dd.submit import rank_scores, reasons, validate  # noqa: E402
from predict import score  # noqa: E402

SHAWN_ASSET = r"sh_(log_usd_paid|spread|out_.*|in_.*|is_conv|n_legs|n_assets|is_path|has_sendmax|has_paths|c\d+_n_assets)"


def shawn_dir():
    for cand in (HERE.parent.parent / "shawn", HERE.parent):  # zip layout, then repo layout
        if (cand / "scenarios.py").exists():
            return cand
    raise FileNotFoundError("model B code (shawn/) not found")


def shawn_base_features(directory, hashes):
    """Model B's behavioural features for a split (no model B scores), aligned to our row order."""
    sys.path.insert(0, str(shawn_dir()))
    import features as shawn_features
    import load as shawn_load
    e = shawn_load.load_split(directory)
    F, _ = shawn_features.build_features(e)
    F.index = e.tx.values
    F = F.add_prefix("sh_").reindex(hashes).reset_index(drop=True)
    return F.drop(columns=[c for c in F.columns if re.fullmatch(SHAWN_ASSET, c)])


def assemble(split, directory, union, cons=False):
    X = build_features(split)
    parts = [X, motif_features(split)]
    if cons:
        parts.append(conservation_features(prepare(split)))
    if union:
        parts.append(shawn_base_features(directory, split.hash.to_numpy()))
    U = pd.concat(parts, axis=1)
    return X, U[sorted(U.columns)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="decode-and-detect-training-data/train")
    ap.add_argument("--test", default="decode-and-detect-test-data")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--union", action="store_true")
    ap.add_argument("--cons", action="store_true", help="add value-conservation features of motif clusters")
    a = ap.parse_args()
    t0 = time.time()
    train, test = load_split(a.train), load_split(a.test)
    labels = pd.read_csv(Path(a.train) / "labels.csv").rename(columns={"TRANSACTION_HASH": "hash"})
    labels = train[["hash"]].merge(labels, on="hash", how="left", validate="1:1")
    _, U_train = assemble(train, a.train, a.union, a.cons)
    X_test, U_test = assemble(test, a.test, a.union, a.cons)
    assert list(U_train.columns) == list(U_test.columns), "train/test feature columns differ"
    print(f"features: train {U_train.shape}, test {U_test.shape} ({time.time() - t0:.0f}s)")
    s = score(U_train, labels, U_test)
    ranked = rank_scores(s, test.hash)
    pred = pd.DataFrame({"TRANSACTION_HASH": test.hash, "score": ranked, "reason": reasons(X_test, ranked)})
    validate(pred, a.test)
    pred.to_csv(a.out, index=False, float_format="%.8f")
    print(f"wrote {len(pred)} rows to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
