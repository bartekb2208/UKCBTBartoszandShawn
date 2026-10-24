"""Out-of-fold blend sweep over our variants and Shawn's model B (rank averages).

Reads cache/oof_*.npy (aligned to cache/train_labels.parquet), cache/oof_submission.csv (model A, by hash) and
the oof_arrays.npz written by the model-B capture script. Prints pooled AP and fold mean for each combination.
Usage: python src/blend_sweep.py <oof_arrays.npz>
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from dd.evaluate import report  # noqa: E402
import load as shawn_load  # noqa: E402

lab = pd.read_parquet("cache/train_labels.parquet").reset_index(drop=True)
order = shawn_load.load_split("decode-and-detect-training-data/train")["tx"].to_numpy()
S = np.load(sys.argv[1])
byhash = lambda arr: pd.Series(arr, index=order).reindex(lab.hash.values).to_numpy()
O = {"A": pd.read_csv("cache/oof_submission.csv").set_index("hash").oof.reindex(lab.hash.values).to_numpy(),
     "B": byhash(S["blend"]), "Bmain": byhash(S["stage2"]), "Bmix": byhash(S["mixture"])}
for name in ("A_motif", "U_motif"):
    p = Path(f"cache/oof_{name}.npy")
    if p.exists():
        O[name] = np.load(p)
r = lambda x: rankdata(x) / len(x)
R = {k: r(v) for k, v in O.items()}


def show(name, s):
    res = report(lab, s, name)
    print(f"{name:42s} AP {res['ap']:.2f}  fold mean {res['fold_mean']:.2f} +/- {res['fold_sd']:.2f}  {res['folds']}")
    return res["ap"]


print("single models"); [show(k, v) for k, v in O.items()]
print("\ntwo-way blends (weight on the first)")
for first in [k for k in ("A", "A_motif", "U_motif") if k in R]:
    best = max(((w, show(f"{first} x B  w={w:.1f}", w * R[first] + (1 - w) * R["B"])) for w in np.arange(0.3, 0.91, 0.1)), key=lambda x: x[1])
    print(f"   -> best weight on {first}: {best[0]:.1f} (AP {best[1]:.2f})\n")
if "U_motif" in R:
    print("three-way blends")
    for wa, wu, wb in [(0, .5, .5), (.25, .25, .5), (0, .6, .4), (.2, .4, .4), (.33, .33, .34)]:
        show(f"A_motif {wa:.2f} / U_motif {wu:.2f} / B {wb:.2f}", wa * R["A_motif"] + wu * R["U_motif"] + wb * R["B"])
    show("U_motif .5 / Bmain .25 / Bmix .25", .5 * R["U_motif"] + .25 * R["Bmain"] + .25 * R["Bmix"])
