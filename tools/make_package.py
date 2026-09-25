"""Build <team>_submission.zip in the exact structure required by the official problem statement:

    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/src/            (+ configs/, needed to reproduce the run)
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    Documentation_template.md                        (filled in)

    python tools/make_package.py --team <team_name> --doc <path to filled Documentation_template.md>

Refuses to build if the output files fail the local rule checks, or if base.yaml is still in
synthetic mode (a synthetic output must never be packaged as a competition submission).
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.getcwd())
from src.data.adapters import make_adapter  # noqa: E402
from src.pipeline.core import load_cfg  # noqa: E402
from src.pipeline.submit import local_checks  # noqa: E402

CODE = "code/business_entity_resolution"


def build(team, doc, out_dir, zip_path, base, allow_synthetic=False):
    if base["mode"] != "real" and not allow_synthetic:
        raise SystemExit("base.yaml is in synthetic mode: refusing to package a synthetic output")
    if not os.path.exists(doc):
        raise SystemExit(f"Documentation_template.md not found at {doc}")
    ents, recs, _ = make_adapter(base).load("test")
    issues = local_checks(out_dir, ents["entity_id"].tolist(), set(recs["record_id"]))
    if issues:
        raise SystemExit("output files fail local checks:\n" + "\n".join(issues))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(os.path.join(out_dir, f), f"output/{f}")
        for top in ("src", "configs"):
            for dp, dn, fn in os.walk(top):
                dn[:] = [d for d in dn if d != "__pycache__"]
                for f in fn:
                    if not f.endswith(".pyc"):
                        z.write(os.path.join(dp, f), f"{CODE}/{os.path.join(dp, f)}".replace("\\", "/"))
        z.write("README.md", f"{CODE}/README.md")
        z.write("requirements.txt", f"{CODE}/requirements.txt")
        z.write(doc, "Documentation_template.md")
    return zip_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--base", default="base")
    a = ap.parse_args()
    base = load_cfg(a.base)
    print(build(a.team, a.doc, base["output"]["dir"], f"{a.team}_submission.zip", base))


if __name__ == "__main__":
    main()
