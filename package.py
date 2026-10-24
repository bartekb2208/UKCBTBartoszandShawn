"""Build the final code ZIP and check it against the submission rules.
Usage: python package.py [--template path/to/submission_template.csv]

Checks: predictions cover every test transaction exactly once, scores finite in [0,1],
no ties, ZIP under 8 MiB, CSV under 12 MiB, required documents present, no data or caches.
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

REQUIRED_DOCS = ["README.md", "PROMPTS.md", "TEAM.md", "LICENSE", "requirements.txt"]
REPORT_GLOBS = ["report*.pdf", "slides*.pdf", "report*.md", "method*.pdf", "*.pptx"]
CODE = ["config.py", "load.py", "features.py", "scenarios.py", "model.py", "run.py", "ensemble.py", "package.py"]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--template")
    a.add_argument("--test", default=str(C.TEST_DIR))
    args = a.parse_args()
    ok = True
    sub_p = C.OUT_DIR / "submission.csv"
    sub = pd.read_csv(sub_p)
    ids = set(pd.read_parquet(Path(args.test) / "transactions.parquet", columns=["TRANSACTION_HASH"])["TRANSACTION_HASH"].astype(str))
    if args.template:
        ids = set(pd.read_csv(args.template)["TRANSACTION_HASH"].astype(str))
    checks = {
        "columns exactly TRANSACTION_HASH,score": list(sub.columns) == ["TRANSACTION_HASH", "score"],
        f"{len(ids):,} rows": len(sub) == len(ids),
        "no duplicate ids": sub["TRANSACTION_HASH"].is_unique,
        "no missing or extra ids": set(sub["TRANSACTION_HASH"].astype(str)) == ids,
        "scores finite and in [0,1]": np.isfinite(sub["score"]).all() and sub["score"].between(0, 1).all(),
        "no tied scores": sub["score"].nunique() == len(sub),
        "CSV under 12 MiB": sub_p.stat().st_size < 12 * 2**20,
    }
    for doc in REQUIRED_DOCS:
        checks[f"{doc} present"] = Path(doc).exists()
    reports = [p for g in REPORT_GLOBS for p in Path(".").glob(g)]
    checks["method report/slides present (<10 slides or <=4 pages)"] = bool(reports)
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
        ok &= bool(v)

    zp = Path("submission_code.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE + REQUIRED_DOCS + [str(p) for p in reports]:
            if Path(f).exists():
                z.write(f)
        for f in ["submission.csv", "results.json", "feature_importance.csv", "validation_by_family.csv",
                  "top_flagged_test.csv", "clusters_not_flagged.csv"]:
            if (C.OUT_DIR / f).exists():
                z.write(C.OUT_DIR / f, f"results/{f}")
    size = zp.stat().st_size
    print(("PASS " if size < 8 * 2**20 else "FAIL ") + f"ZIP {size / 2**20:.2f} MiB (limit 8)")
    ok &= size < 8 * 2**20
    print("\nREADY: upload out/submission.csv + submission_code.zip" if ok else "\nNOT READY: fix the FAIL lines")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
