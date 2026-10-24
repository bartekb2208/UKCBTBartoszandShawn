"""Package one leaderboard-test variant: submissions/<name>/{code.zip, predictions.csv, WHAT_IS_THIS.md}.

Usage: python final_package/build_variant.py <name> <predictions.csv> <mode> "<one-paragraph note>" [--test <dir>]
Modes: a        model A + relay-motif features                         (ours/ only)
       u        model A + motif features + model B's behavioural features (ours/ and shawn/)
       blend_a  50/50 rank blend of mode a and model B's full model     (ours/ and shawn/)
       blend_u  50/50 rank blend of mode u and model B's full model     (ours/ and shawn/)
       blend3   equal rank blend of modes a and u and model B's full model
The ZIP holds the code that produced the CSV plus README, LICENSE and pinned requirements. The compliance
documents (report, AI use, team) are built separately for the final entry (final_package/build_zip.py).
"""
import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FP = ROOT / "final_package"
OURS = ["predict.py", "predict_v6.py", "cv.py", "run_baseline.py", "exp.py", "run_shawn_with_motifs.py", "self_train_v12.py",
        "run_shawn_cotrain.py"]
OURS_DD = ["__init__.py", "baseline.py", "evaluate.py", "features.py", "folds.py", "load.py", "model.py", "submit.py", "motifs.py",
           "conservation.py"]
SHAWN = ["config.py", "load.py", "features.py", "scenarios.py", "model.py", "run.py", "LICENSE"]

RUN = {
    "a": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out ../predictions.csv)',
    "u": '(cd ours && "$PY" src/predict_v6.py --union --train "$TRAIN" --test "$TEST" --out ../predictions.csv)',
    "blend_a": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
               '(cd shawn && "$PY" run.py --train "$TRAIN" --test "$TEST" --out "$WORK/shawn_out")\n'
               '"$PY" blend_v5.py "$WORK/a.csv" "$WORK/shawn_out/submission.csv" 0.5 predictions.csv "$TEST"',
    "blend_u": '(cd ours && "$PY" src/predict_v6.py --union --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
               '(cd shawn && "$PY" run.py --train "$TRAIN" --test "$TEST" --out "$WORK/shawn_out")\n'
               '"$PY" blend_v5.py "$WORK/a.csv" "$WORK/shawn_out/submission.csv" 0.5 predictions.csv "$TEST"',
    "blend_motif_both": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
                        '"$PY" ours/src/run_shawn_with_motifs.py "$TRAIN" "$TEST" "$WORK/shawn_out"\n'
                        '"$PY" blend_v5.py "$WORK/a.csv" "$WORK/shawn_out/submission.csv" 0.5 predictions.csv "$TEST"',
    "st_blend": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
                '"$PY" ours/src/run_shawn_with_motifs.py "$TRAIN" "$TEST" "$WORK/shawn_out"\n'
                '"$PY" blend_v5.py "$WORK/a.csv" "$WORK/shawn_out/submission.csv" 0.5 "$WORK/teacher.csv" "$TEST"\n'
                '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher.csv" --out "$WORK/a_st.csv")\n'
                '"$PY" blend_v5.py "$WORK/a_st.csv" "$WORK/shawn_out/SHAWN_FILE" 0.5 predictions.csv "$TEST"',
    "cotrain": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
               '"$PY" ours/src/run_shawn_with_motifs.py "$TRAIN" "$TEST" "$WORK/shawn_out"\n'
               '"$PY" blend_v5.py "$WORK/a.csv" "$WORK/shawn_out/submission.csv" 0.5 "$WORK/teacher1.csv" "$TEST"\n'
               '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher1.csv" --out "$WORK/a_st.csv")\n'
               '"$PY" blend_v5.py "$WORK/a_st.csv" "$WORK/shawn_out/submission_selftrain_top900.csv" 0.5 "$WORK/teacher2.csv" "$TEST"\n'
               '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher2.csv" --top 900 --out "$WORK/a_st2.csv")\n'
               '"$PY" ours/src/run_shawn_cotrain.py "$TRAIN" "$TEST" "$WORK/shawn_co" "$WORK/teacher2.csv"\n'
               '"$PY" blend_v5.py "$WORK/a_st2.csv" "$WORK/shawn_co/SHAWN_FILE" 0.5 predictions.csv "$TEST"',
    "ensemble4": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
                 '"$PY" ours/src/run_shawn_with_motifs.py "$TRAIN" "$TEST" "$WORK/shawn_out"\n'
                 '"$PY" blend_v5.py "$WORK/a.csv" "$WORK/shawn_out/submission.csv" 0.5 "$WORK/teacher1.csv" "$TEST"\n'
                 '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher1.csv" --out "$WORK/a_st.csv")\n'
                 '"$PY" blend_v5.py "$WORK/a_st.csv" "$WORK/shawn_out/submission_selftrain_top900.csv" 0.5 "$WORK/v13.csv" "$TEST"\n'
                 '"$PY" blend_v5.py "$WORK/a_st.csv" "$WORK/shawn_out/submission_selftrain_2rounds.csv" 0.5 "$WORK/v14.csv" "$TEST"\n'
                 '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/v13.csv" --top 900 --out "$WORK/a_st2.csv")\n'
                 '"$PY" ours/src/run_shawn_cotrain.py "$TRAIN" "$TEST" "$WORK/shawn_co" "$WORK/v13.csv"\n'
                 '"$PY" blend_v5.py "$WORK/a_st2.csv" "$WORK/shawn_co/submission_selftrain_top900.csv" 0.5 "$WORK/v15.csv" "$TEST"\n'
                 '"$PY" blend_v5.py "$WORK/a_st2.csv" "$WORK/shawn_co/submission_selftrain_2rounds.csv" 0.5 "$WORK/v16.csv" "$TEST"\n'
                 '"$PY" blend_n.py "$TEST" predictions.csv "$WORK/v13.csv:1" "$WORK/v14.csv:1" "$WORK/v15.csv:1" "$WORK/v16.csv:1"',
    "cotrain2": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
                '"$PY" ours/src/run_shawn_with_motifs.py "$TRAIN" "$TEST" "$WORK/shawn_out"\n'
                '"$PY" blend_v5.py "$WORK/a.csv" "$WORK/shawn_out/submission.csv" 0.5 "$WORK/teacher1.csv" "$TEST"\n'
                '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher1.csv" --out "$WORK/a_st.csv")\n'
                '"$PY" blend_v5.py "$WORK/a_st.csv" "$WORK/shawn_out/submission_selftrain_top900.csv" 0.5 "$WORK/teacher2.csv" "$TEST"\n'
                '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher2.csv" --top 900 --out "$WORK/a_st2.csv")\n'
                '"$PY" ours/src/run_shawn_cotrain.py "$TRAIN" "$TEST" "$WORK/shawn_co" "$WORK/teacher2.csv"\n'
                '"$PY" blend_v5.py "$WORK/a_st2.csv" "$WORK/shawn_co/submission_selftrain_2rounds.csv" 0.5 "$WORK/teacher3.csv" "$TEST"\n'
                '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher3.csv" --top 1000 --out "$WORK/a_st3.csv")\n'
                '"$PY" ours/src/run_shawn_cotrain.py "$TRAIN" "$TEST" "$WORK/shawn_co2" "$WORK/teacher3.csv"\n'
                '"$PY" blend_v5.py "$WORK/a_st3.csv" "$WORK/shawn_co2/submission_selftrain_2rounds.csv" 0.5 predictions.csv "$TEST"',
    "cotrain_cons": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
                    '"$PY" ours/src/run_shawn_with_motifs.py "$TRAIN" "$TEST" "$WORK/shawn_out"\n'
                    '"$PY" blend_v5.py "$WORK/a.csv" "$WORK/shawn_out/submission.csv" 0.5 "$WORK/teacher1.csv" "$TEST"\n'
                    '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher1.csv" --out "$WORK/a_st.csv")\n'
                    '"$PY" blend_v5.py "$WORK/a_st.csv" "$WORK/shawn_out/submission_selftrain_top900.csv" 0.5 "$WORK/teacher2.csv" "$TEST"\n'
                    '"$PY" ours/src/run_shawn_cotrain.py "$TRAIN" "$TEST" "$WORK/shawn_co" "$WORK/teacher2.csv"\n'
                    '(cd ours && "$PY" src/self_train_v12.py --cons --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher1.csv" --out "$WORK/a_cons_st.csv")\n'
                    '(cd ours && "$PY" src/self_train_v12.py --cons --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher2.csv" --top 900 --out "$WORK/a_cons_st2.csv")\n'
                    '"$PY" blend_v5.py "$WORK/a_cons_st2.csv" "$WORK/shawn_co/submission_selftrain_2rounds.csv" 0.5 predictions.csv "$TEST"',
    "cotrain3": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
                '"$PY" ours/src/run_shawn_with_motifs.py "$TRAIN" "$TEST" "$WORK/shawn_out"\n'
                '"$PY" blend_v5.py "$WORK/a.csv" "$WORK/shawn_out/submission.csv" 0.5 "$WORK/teacher1.csv" "$TEST"\n'
                '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher1.csv" --out "$WORK/a_st.csv")\n'
                '"$PY" blend_v5.py "$WORK/a_st.csv" "$WORK/shawn_out/submission_selftrain_top900.csv" 0.5 "$WORK/teacher2.csv" "$TEST"\n'
                '(cd ours && "$PY" src/self_train_v12.py --train "$TRAIN" --test "$TEST" --teacher "$WORK/teacher2.csv" --top 900 --out "$WORK/a_st2.csv")\n'
                '"$PY" ours/src/run_shawn_cotrain.py "$TRAIN" "$TEST" "$WORK/shawn_co" "$WORK/teacher2.csv"\n'
                '"$PY" blend_v5.py "$WORK/a_st2.csv" "$WORK/shawn_co/submission_selftrain_2rounds.csv" 0.3 "$WORK/teacher3.csv" "$TEST"\n'
                '"$PY" ours/src/run_shawn_cotrain.py "$TRAIN" "$TEST" "$WORK/shawn_co3" "$WORK/teacher3.csv"\n'
                '"$PY" blend_v5.py "$WORK/a_st2.csv" "$WORK/shawn_co3/submission_selftrain_2rounds.csv" 0.5 predictions.csv "$TEST"',
    "blend3": '(cd ours && "$PY" src/predict_v6.py --train "$TRAIN" --test "$TEST" --out "$WORK/a.csv")\n'
              '(cd ours && "$PY" src/predict_v6.py --union --train "$TRAIN" --test "$TEST" --out "$WORK/u.csv")\n'
              '(cd shawn && "$PY" run.py --train "$TRAIN" --test "$TEST" --out "$WORK/shawn_out")\n'
              '"$PY" blend_n.py "$TEST" predictions.csv "$WORK/a.csv:1" "$WORK/u.csv:1" "$WORK/shawn_out/submission.csv:1"',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"), ap.add_argument("csv"), ap.add_argument("mode", choices=RUN), ap.add_argument("note")
    ap.add_argument("--test", default=str(ROOT / "decode-and-detect-test-data"))
    ap.add_argument("--boost", choices=["dist", "dist+agg"], help="lift ultra-regular relay motifs (motif_boost.py)")
    ap.add_argument("--floor", default="0.97")
    ap.add_argument("--shawn-file", default="submission.csv", help="which model B output the st_blend mode blends with")
    ap.add_argument("--weight", default="0.5", help="weight of our model in the final 50/50-style blend")
    ap.add_argument("--final", action="store_true", help="add the compliance documents and run the rule checks")
    ap.add_argument("--fusion", choices=["rank", "rrf"], default="rank", help="final combination: rank average or reciprocal-rank fusion (k=60)")
    ap.add_argument("--smooth", help="lambda for the scenario-consensus smoothing step (cluster_smooth.py)")
    ap.add_argument("--b-no-asset", action="store_true", help="last model B co-training step without asset/conversion columns")
    ap.add_argument("--time", default="about 25 minutes", help="measured run time, for the final README")
    a = ap.parse_args()
    need_shawn = a.mode != "a"
    out = ROOT / "submissions" / a.name
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(a.csv)
    ids = pd.read_parquet(Path(a.test) / "transactions.parquet", columns=["TRANSACTION_HASH"]).TRANSACTION_HASH
    assert list(pred.columns[:2]) == ["TRANSACTION_HASH", "score"] and len(pred) == len(ids) and pred.TRANSACTION_HASH.is_unique
    assert set(pred.TRANSACTION_HASH) == set(ids) and pred.score.between(0, 1).all() and pred.score.is_unique, "invalid predictions file"

    steps = RUN[a.mode].replace("SHAWN_FILE", a.shawn_file).replace(" 0.5 predictions.csv", f" {a.weight} predictions.csv")
    if a.b_no_asset:
        steps = steps.replace('"$WORK/shawn_co2" "$WORK/teacher3.csv"', '"$WORK/shawn_co2" "$WORK/teacher3.csv" --no-asset')
    if a.fusion == "rrf":
        last = steps.split("\n")[-1]
        args = last.split('"$PY" blend_v5.py ')[1].split(" ")
        steps = "\n".join(steps.split("\n")[:-1] + [f'"$PY" blend_rrf.py "$TEST" predictions.csv 60 {args[0]} {args[1]}'])
    run_sh = ('#!/usr/bin/env bash\n# Reproduce predictions.csv (python 3.14.6, requirements.txt). Data next to this file.\n'
              'set -euo pipefail\ncd "$(dirname "$0")"\nROOT="$(pwd)"\n'
              'TRAIN="$(cd "${TRAIN:-decode-and-detect-training-data/train}" && pwd)"\nTEST="$(cd "${TEST:-decode-and-detect-test-data}" && pwd)"\n'
              'PY="${PY:-python3}"\nWORK="$ROOT/work"\nmkdir -p "$WORK"\n' + steps + '\n'
              + (f'mv predictions.csv "$WORK/unsmoothed.csv"\n"$PY" cluster_smooth.py "$WORK/unsmoothed.csv" "$TEST" predictions.csv {a.smooth}\n' if a.smooth else '')
              + (f'mv predictions.csv "$WORK/unboosted.csv"\n"$PY" motif_boost.py "$WORK/unboosted.csv" "$TEST" predictions.csv {a.floor} {a.boost}\n' if a.boost else '')
              + 'echo "done: $ROOT/predictions.csv"\n')
    readme = (f"# {a.name}\n\n{a.note}\n\nReproduce: `PY=python3 ./run.sh` (Python 3.14.6, `pip install -r requirements.txt`; organiser data in "
              "`decode-and-detect-training-data/train/` and `decode-and-detect-test-data/`). Licence: MIT. "
              "No external data, API, LLM or pre-trained model is used; see the final entry's THIRD_PARTY.md and AI_USE.md.\n")
    files = {"README.md": readme, "run.sh": run_sh, "LICENSE": (FP / "LICENSE").read_text(), "requirements.txt": (FP / "requirements.txt").read_text()}
    for n in OURS:
        files[f"ours/src/{n}"] = (ROOT / "src" / n).read_text()
    for n in OURS_DD:
        files[f"ours/src/dd/{n}"] = (ROOT / "src" / "dd" / n).read_text()
    if a.smooth:
        files["cluster_smooth.py"] = (FP / "cluster_smooth.py").read_text()
    if a.boost:
        files["motif_boost.py"] = (FP / "motif_boost.py").read_text()
    if need_shawn:
        files["blend_v5.py"] = (FP / "blend_v5.py").read_text()
        files["blend_n.py"] = (FP / "blend_n.py").read_text()
        files["blend_rrf.py"] = (FP / "blend_rrf.py").read_text()
        for n in SHAWN:
            files[f"shawn/{n}"] = (ROOT / n).read_text()
    if a.final:
        files["README.md"] = (FP / "README_final.md").read_text().replace("__NOTE__", a.note).replace("__TIME__", a.time)
        for n in ("REPORT.md", "AI_USE.md", "TEAM.md", "THIRD_PARTY.md", "CASES.md"):
            files[n] = (FP / n).read_text()
        files["ours/NOTES.md"] = (ROOT / "NOTES.md").read_text()
        for n in ("PROMPTS.md", "README.md", "report.md"):
            files[f"shawn/{n}"] = (ROOT / n).read_text()
        import re
        secret = re.compile(r"api[_-]?key|secret|passw(or)?d|BEGIN [A-Z ]*PRIVATE|seed phrase|@gmail|/Users/|/private/tmp", re.I)
        problems = [f"{k}: {m.group(0)!r}" for k, v in files.items() for m in secret.finditer(v)]
        for k, v in files.items():
            if k.endswith(".py"):
                compile(v, k, "exec")
            if "[FILL" in v:
                print(f"WARN {k}: {v.count('[FILL')} placeholder(s) to complete")
            if "[FINAL" in v:
                print(f"WARN {k}: {v.count('[FINAL')} final-version note(s) to complete")
        if problems:
            sys.exit("possible secrets or personal paths: " + "; ".join(problems))
    with zipfile.ZipFile(out / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for arc in sorted(files):
            info = zipfile.ZipInfo(arc, date_time=(2026, 10, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if arc.endswith(".sh") else 0o644) << 16
            z.writestr(info, files[arc])
    (out / "predictions.csv").write_bytes(Path(a.csv).read_bytes())
    (out / "WHAT_IS_THIS.md").write_text(f"# {a.name}\n\n{a.note}\n")
    print(f"{a.name}: {len(files)} files, zip {(out / 'code.zip').stat().st_size // 1024} KiB, csv {len(pred)} rows")


if __name__ == "__main__":
    sys.exit(main())
