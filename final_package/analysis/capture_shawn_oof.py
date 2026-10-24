"""Run model B (shawn/) unchanged and also save the out-of-fold prediction arrays it scores,
for the blend-weight check in blend_cv.py. Writes <out dir>/oof_arrays.npz with stage1, stage2,
mixture and blend (row order = shawn/load.py's order for the train split).

Usage (from the ZIP root): python analysis/capture_shawn_oof.py <train dir> <test dir> <out dir>
Not needed to reproduce predictions.csv.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shawn"))
import model  # noqa: E402
import run  # noqa: E402

train_dir, test_dir, out_dir = sys.argv[1:4]
recorded = []
_slices = model.slices


def record(y, p, lab):
    recorded.append(np.asarray(p).copy())
    return _slices(y, p, lab)


model.slices = record
sys.argv = ["run.py", "--train", train_dir, "--test", test_dir, "--out", out_dir]
run.main()
names = ["stage1", "stage2", "mixture", "blend"]
assert len(recorded) == len(names), f"expected {len(names)} scored arrays, got {len(recorded)}"
np.savez(Path(out_dir) / "oof_arrays.npz", **dict(zip(names, recorded)))
print("saved", Path(out_dir) / "oof_arrays.npz")
