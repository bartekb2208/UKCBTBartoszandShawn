"""Refresh only the documents inside an already-verified code ZIP; every code file is copied byte-for-byte.

Usage: python final_package/update_docs.py <verified code.zip> <out code.zip> "<README note>" "<time text>" [<variant code.zip>]

With a variant ZIP, its run.sh replaces the verified one (the chosen version differs only in its last steps,
for example the blend weight); every other code file still comes byte-for-byte from the verified ZIP.

Documents come from final_package/ (README_final.md becomes README.md) and from the repository (ours/NOTES.md,
shawn/PROMPTS.md, shawn/README.md, shawn/report.md). The script prints the SHA-256 of every code file in both
archives so it is visible that the verified code did not change, and checks for secrets and placeholders.
"""
import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FP = ROOT / "final_package"
DOCS = {"README.md": None, "REPORT.md": FP / "REPORT.md", "AI_USE.md": FP / "AI_USE.md", "TEAM.md": FP / "TEAM.md",
        "THIRD_PARTY.md": FP / "THIRD_PARTY.md", "CASES.md": FP / "CASES.md", "ours/NOTES.md": ROOT / "NOTES.md",
        "shawn/PROMPTS.md": ROOT / "PROMPTS.md", "shawn/README.md": ROOT / "README.md", "shawn/report.md": ROOT / "report.md"}
SECRET = re.compile(r"api[_-]?key|secret|passw(or)?d|BEGIN [A-Z ]*PRIVATE|seed phrase|@gmail|/Users/|/private/tmp", re.I)

src, dst, note, time_text = sys.argv[1:5]
variant = sys.argv[5] if len(sys.argv) > 5 else None
with zipfile.ZipFile(src) as z:
    entries = {i.filename: (i, z.read(i.filename)) for i in z.infolist()}
if variant:
    with zipfile.ZipFile(variant) as z:
        entries["run.sh"] = (entries["run.sh"][0], z.read("run.sh"))
code = {k: hashlib.sha256(v).hexdigest() for k, (i, v) in entries.items() if k not in DOCS}
if variant:
    print("run.sh taken from", variant, "- its last lines:")
    print("\n".join(entries["run.sh"][1].decode().strip().splitlines()[-3:]))
new = {}
for name in entries:
    if name in DOCS:
        text = ((FP / "README_final.md").read_text().replace("__NOTE__", note).replace("__TIME__", time_text)
                if name == "README.md" else DOCS[name].read_text())
        new[name] = text.encode()
    else:
        new[name] = entries[name][1]
problems = [f"{k}: {m.group(0)!r}" for k, v in new.items() if k.endswith((".md", ".py", ".sh", ".txt")) or k == "LICENSE"
            for m in SECRET.finditer(v.decode())]
if problems:
    sys.exit("possible secrets or personal paths: " + "; ".join(problems))
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(new):
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 24, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = entries[name][0].external_attr
        z.writestr(info, new[name])
with zipfile.ZipFile(dst) as z:
    after = {k: hashlib.sha256(z.read(k)).hexdigest() for k in z.namelist() if k not in DOCS}
assert after == code, "a code file changed"
print(f"{len(code)} code files identical to the verified ZIP; {len(DOCS)} documents refreshed -> {dst}")
for k in sorted(new):
    if k in DOCS:
        t = new[k].decode()
        if "[FILL" in t:
            print(f"WARN {k}: {t.count('[FILL')} placeholder(s) to complete")
