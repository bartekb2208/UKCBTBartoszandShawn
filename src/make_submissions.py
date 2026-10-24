"""Assemble submissions/vN/{code.zip, predictions.csv, WHAT_IS_THIS.md} from the prediction files in the repo."""
import re, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submissions"
OURS_DOCS = {"README.ours.md": "README.md", "LICENSE.ours": "LICENSE", "requirements.ours.txt": "requirements.txt",
             "run.sh": "run.sh", "AI_USE.md": "AI_USE.md", "THIRD_PARTY.md": "THIRD_PARTY.md", "NOTES.md": "NOTES.md", "CASES.md": "CASES.md"}
SHAWN_CODE = ["config.py", "load.py", "features.py", "scenarios.py", "model.py", "run.py", "package.py"]
SHAWN_DOCS = ["README.md", "PROMPTS.md", "TEAM.md", "LICENSE", "requirements.txt", "report.md"]


def add_ours(z, prefix="", keep_assets=False):
    for src, dst in OURS_DOCS.items():
        z.write(ROOT / src, prefix + dst)
    for p in sorted((ROOT / "src").rglob("*.py")):
        text = p.read_text()
        if keep_assets and p.name == "features.py":
            text, n = re.subn(r'^ASSET_FEATURES = r".*"$', 'ASSET_FEATURES = r"(?!)"  # v1: asset and conversion features kept', text, flags=re.M)
            assert n == 1
        z.writestr(prefix + str(p.relative_to(ROOT)), text)


def add_shawn(z, prefix=""):
    for f in SHAWN_CODE + SHAWN_DOCS:
        z.write(ROOT / f, prefix + f)


def build(name, csv, note, ours=False, shawn=False, keep_assets=False, extra=None):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / csv, d / "predictions.csv")
    with zipfile.ZipFile(d / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        if ours:
            add_ours(z, "ours/" if shawn else "", keep_assets)
        if shawn:
            add_shawn(z, "shawn/" if ours else "")
        z.writestr("SUBMISSION.md", note)
    (d / "WHAT_IS_THIS.md").write_text(note)
    print(name, csv, (d / "code.zip").stat().st_size // 1024, "KiB zip")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build("v1", "upload_v1_baseline.csv", "# v1: our model\nLightGBM x3 seeds + ExtraTrees rank blend, 243 features including asset and conversion features. Train CV 62.0.\nReproduce: `./run.sh` (ours/ files; Python 3.14.6, requirements.txt).\n", ours=True, keep_assets=True)
    build("v2", "upload_v2_noasset.csv", "# v2: our model without asset and conversion features\nSame as v1 but 205 features (organiser README: asset and conversion choices do not inspect labels; test converts more than train). Train CV 62.3.\nReproduce: `./run.sh`.\n", ours=True)
    build("v3", "upload_v3_selftrain.csv", "# v3: our model v2 + one round of self-training on the test split\nTop 500 test payments pseudo-labelled suspicious, bottom 60% normal, refit, rescore. No train-CV equivalent.\nReproduce: `python src/self_train.py --out predictions.csv`.\n", ours=True)
    build("v4", "v3_submission.csv", "# v4: Shawn's v3 model, unchanged\nScenario-linking model, auxiliary shape/relay models, decoy expert, two-stage LightGBM. Train CV 61.6 (his grouping).\nReproduce: `python run.py --train <train dir> --test <test dir>`; out/submission.csv equals this file (checked).\n", shawn=True)
    build("v5", "upload_v5_blend50.csv", "# v5: 50/50 rank blend of v2 (ours) and v4 (Shawn)\nReproduce: run both pipelines, then `python ours/src/blend_test.py <v2 csv> <v4 csv> 0.5 out.csv`.\n", ours=True, shawn=True)
