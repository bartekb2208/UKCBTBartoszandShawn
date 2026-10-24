"""Build the corrected code ZIP for the Model Phi entry and check it against the submission rules.

Usage: python final_package/build_zip.py <predictions.csv> [--test <test data dir>] [--out <output dir>]

Writes <out>/code.zip and <out>/predictions.csv. Fails (exit 1) on any rule violation; unfilled
[FILL] placeholders in the documents are listed as warnings and must be completed before upload.
The ZIP is built from fixed file times so that identical inputs give an identical archive.
"""
import argparse
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FP = ROOT / "final_package"

OURS_SRC = ["predict.py", "cv.py", "run_baseline.py", "exp.py", "lofo.py", "wallet_split.py", "shift_check.py", "cases.py"]
OURS_DD = ["__init__.py", "baseline.py", "evaluate.py", "features.py", "folds.py", "load.py", "model.py", "submit.py"]
SHAWN_FILES = ["config.py", "load.py", "features.py", "scenarios.py", "model.py", "run.py", "package.py",
               "PROMPTS.md", "README.md", "report.md", "LICENSE"]
ROOT_DOCS = ["README.md", "REPORT.md", "CASES.md", "LICENSE", "requirements.txt", "THIRD_PARTY.md", "AI_USE.md",
             "TEAM.md", "run_v5.sh", "blend_v5.py"]
FORBIDDEN_SUFFIX = {".parquet", ".csv", ".pkl", ".npy", ".npz", ".pyc", ".pptx", ".pdf", ".zip", ".DS_Store"}
SECRET_PATTERN = re.compile(r"api[_-]?key|secret|passw(or)?d|BEGIN [A-Z ]*PRIVATE|seed phrase|@gmail|/Users/|/private/tmp", re.I)
MiB = 2 ** 20


def collect():
    files = {name: FP / name for name in ROOT_DOCS}
    files["analysis/capture_shawn_oof.py"] = FP / "analysis" / "capture_shawn_oof.py"
    files["analysis/blend_cv.py"] = FP / "analysis" / "blend_cv.py"
    files["ours/NOTES.md"] = ROOT / "NOTES.md"
    for n in OURS_SRC:
        files[f"ours/src/{n}"] = ROOT / "src" / n
    for n in OURS_DD:
        files[f"ours/src/dd/{n}"] = ROOT / "src" / "dd" / n
    for n in SHAWN_FILES:
        files[f"shawn/{n}"] = ROOT / n
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--test", default=str(ROOT / "decode-and-detect-test-data"))
    ap.add_argument("--out", default=str(ROOT / "submissions" / "v5_final"))
    args = ap.parse_args()
    problems, warnings = [], []
    files = collect()
    for arc, src in files.items():
        if not src.exists():
            problems.append(f"missing file for {arc}: {src}")
        if Path(arc).suffix in FORBIDDEN_SUFFIX:
            problems.append(f"data or cache file type in ZIP: {arc}")
    if problems:
        sys.exit("\n".join(problems))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "code.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for arc in sorted(files):
            info = zipfile.ZipInfo(arc, date_time=(2026, 10, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if arc.endswith(".sh") else 0o644) << 16
            z.writestr(info, files[arc].read_bytes())

    # --- checks on the finished archive
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        infos = z.infolist()
        total = sum(i.file_size for i in infos)
        if zpath.stat().st_size > 8 * MiB:
            problems.append(f"ZIP is {zpath.stat().st_size / MiB:.2f} MiB (limit 8)")
        if total > 64 * MiB:
            problems.append(f"uncompressed {total / MiB:.1f} MiB (limit 64)")
        if len(infos) > 2000:
            problems.append(f"{len(infos)} files (limit 2000)")
        if any(i.flag_bits & 0x1 for i in infos):
            problems.append("encrypted entries")
        for i in infos:
            data = z.read(i.filename)
            if i.filename.endswith(".py"):
                try:
                    compile(data, i.filename, "exec")
                except SyntaxError as e:
                    problems.append(f"{i.filename} does not compile: {e}")
            if i.filename.endswith((".py", ".md", ".sh", ".txt")) or i.filename == "LICENSE":
                text = data.decode()
                for m in SECRET_PATTERN.finditer(text):
                    line = text[: m.start()].count("\n") + 1
                    problems.append(f"possible secret or personal path in {i.filename}:{line}: {m.group(0)!r}")
                if re.search(r"ouid=\d+", text):
                    warnings.append(f"{i.filename}: a Google Drive link carries an account id (ouid=...); consider removing the parameter")
                if "[FILL" in text:
                    warnings.append(f"{i.filename}: {text.count('[FILL')} placeholder(s) to complete")
        names = {i.filename for i in infos}
        for need in ("README.md", "LICENSE", "requirements.txt", "REPORT.md", "THIRD_PARTY.md", "AI_USE.md", "TEAM.md", "run_v5.sh"):
            if need not in names:
                problems.append(f"{need} missing at the ZIP root")

    # --- checks on the predictions CSV
    pred = pd.read_csv(args.predictions)
    ids = pd.read_parquet(Path(args.test) / "transactions.parquet", columns=["TRANSACTION_HASH"]).TRANSACTION_HASH
    checks = {
        "columns start with TRANSACTION_HASH,score": list(pred.columns[:2]) == ["TRANSACTION_HASH", "score"],
        f"{len(ids):,} rows": len(pred) == len(ids),
        "no duplicate ids": pred.TRANSACTION_HASH.is_unique,
        "same ids as the test transactions": set(pred.TRANSACTION_HASH) == set(ids),
        "scores finite and within [0, 1]": bool(pred.score.notna().all() and pred.score.between(0, 1).all()),
        "no tied scores": pred.score.is_unique,
        "CSV under 12 MiB": Path(args.predictions).stat().st_size < 12 * MiB,
    }
    for k, v in checks.items():
        if not v:
            problems.append(f"CSV check failed: {k}")
    shutil.copy(args.predictions, out / "predictions.csv")

    md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
    print(f"ZIP  {zpath}  {zpath.stat().st_size / 1024:.0f} KiB, {len(infos)} files, {total / 1024:.0f} KiB uncompressed, md5 {md5(zpath)}")
    print(f"CSV  {out / 'predictions.csv'}  md5 {md5(out / 'predictions.csv')}")
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    for w in warnings:
        print("WARN", w)
    if problems:
        print("\nPROBLEMS:\n- " + "\n- ".join(problems))
        sys.exit(1)
    print("\nAll automated checks passed" + (" (placeholders remain)" if warnings else "") + ".")


if __name__ == "__main__":
    main()
