import os
import sys

import pandas as pd
import pytest

from src.data.adapters import CompetitionDataAdapter, SchemaViolation, read_tsv
from src.pipeline.submit import (OutputContractError, build_lists, local_checks, official_gate, write_outputs)

HDR = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _tree(root, s1, s2, s3, gt=None, split="train"):
    d = os.path.join(root, "dataset", split)
    os.makedirs(d, exist_ok=True)
    for name, rows in (("source1", s1), ("source2", s2), ("source3", s3)):
        with open(os.path.join(d, f"{split}_{name}.tsv"), "w", encoding="utf-8", newline="") as f:
            f.write(HDR + "".join("\t".join(r) + "\n" for r in rows))
    if gt is not None:
        with open(os.path.join(d, "train_ground_truth.tsv"), "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n" + "".join(f"{a}\t{b}\n" for a, b in gt))


def test_official_layout_is_parsed(tmp_path):
    _tree(tmp_path,
          [("S1-1", 'Joe\'s "Best" Cafe, LLC', "12 MG Rd, Near SBI ATM, Bengaluru - 560001", "India"),
           ("S1-2", "Acme Corp", "", "Atlantis")],                        # open-set country, empty address
          [("S2-1", "Joes Best Cafe", "12 M.G. Road, Bengaluru", "India"),
           ("S2-2", '"Quoted Start Traders', '"7 Park St, Kolkata', "India"),   # leading quote: default parser
           ("S2-3", "Plain Name", "9 Lake Rd", "India")],                      # would swallow rows -> fallback
          [("S3-1", "ACME CORPORATION", "1 Main St, Springfield, IL 62701", "US")],
          gt=[("S1-1", "S2-1"), ("S1-2", "")])
    ents, recs, labels = CompetitionDataAdapter(str(tmp_path)).load("train")
    assert ents.loc[0, "name"] == 'Joe\'s "Best" Cafe, LLC'             # quote chars survive
    assert len(recs) == 4 and recs.loc[1, "name"] == '"Quoted Start Traders' and recs.loc[2, "name"] == "Plain Name"
    assert ents.loc[1, "country"] == "Atlantis"
    assert ents.loc[1, "full_address"] is None
    assert set(recs["source"]) == {"source2", "source3"}
    assert labels.values.tolist() == [["S1-1", "S2-1"]]


def test_ground_truth_row_without_trailing_tab_is_empty_match(tmp_path):
    _tree(tmp_path, [("S1-1", "a", "b", "US"), ("S1-2", "c", "d", "US")], [("S2-1", "a", "b", "US")],
          [("S3-1", "a", "b", "US")])
    (tmp_path / "dataset" / "train" / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\n", encoding="utf-8")
    _, _, labels = CompetitionDataAdapter(str(tmp_path)).load("train")
    assert sorted(labels.values.tolist()) == [["S1-1", "S2-1"], ["S1-1", "S3-1"]]


def test_extra_field_fails_loudly(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text(HDR + "S1-1\ta\tb\tUS\textra\n", encoding="utf-8")
    with pytest.raises(SchemaViolation):
        read_tsv(str(p), ["entity_id", "business_name", "business_address", "country"])


def test_prefix_violation_fails_loudly(tmp_path):
    _tree(tmp_path, [("S2-9", "x", "y", "US")], [("S2-1", "a", "b", "US")], [("S3-1", "a", "b", "US")])
    with pytest.raises(SchemaViolation):
        CompetitionDataAdapter(str(tmp_path)).load("train")


def test_missing_column_fails_loudly(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("entity_id\tbusiness_name\tcountry\nS1-1\ta\tUS\n", encoding="utf-8")
    with pytest.raises(SchemaViolation):
        read_tsv(str(p), ["entity_id", "business_name", "business_address", "country"])


def test_matches_must_be_subset_of_candidates():
    cands = pd.DataFrame({"entity_id": ["S1-1"], "record_id": ["S2-1"]})
    bad = pd.DataFrame({"entity_id": ["S1-1"], "record_id": ["S3-7"], "score": [0.9]})
    with pytest.raises(OutputContractError):
        build_lists(["S1-1"], cands, bad)


def test_outputs_roundtrip_and_local_checks(tmp_path):
    cands = pd.DataFrame({"entity_id": ["S1-1", "S1-1", "S1-2"], "record_id": ["S2-1", "S3-1", "S2-2"]})
    matches = pd.DataFrame({"entity_id": ["S1-1", "S1-1"], "record_id": ["S3-1", "S2-1"], "score": [0.7, 0.9]})
    cl, ml = build_lists(["S1-1", "S1-2", "S1-3"], cands, matches)
    write_outputs(str(tmp_path), ["S1-1", "S1-2", "S1-3"], cl, ml)
    txt = (tmp_path / "matching_results.tsv").read_text(encoding="utf-8")
    assert txt == "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\nS1-3\t\n"
    assert local_checks(str(tmp_path), ["S1-1", "S1-2", "S1-3"], {"S2-1", "S3-1", "S2-2"}) == []
    # a self-match / unknown id / missing entity must be caught
    (tmp_path / "matching_results.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS1-2,S2-1\nS1-2\t\n", encoding="utf-8")
    issues = local_checks(str(tmp_path), ["S1-1", "S1-2", "S1-3"], {"S2-1", "S3-1", "S2-2"})
    assert any("non S2-/S3-" in i for i in issues) and any("missing" in i for i in issues)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_official_validator_is_a_hard_gate(tmp_path, exit_code):
    root = tmp_path / "student_resource"
    (root / "utils").mkdir(parents=True)
    (root / "utils" / "validate_submission.py").write_text(
        f"import sys\nprint('PASS' if {exit_code} == 0 else '1. issue')\nsys.exit({exit_code})\n", encoding="utf-8")
    base = {"mode": "real", "data": {"root": str(root), "official_validator": "utils/validate_submission.py"}}
    if exit_code == 0:
        assert official_gate(base, str(tmp_path))["passed"] is True
    else:
        with pytest.raises(OutputContractError):
            official_gate(base, str(tmp_path))


def test_real_mode_refuses_without_official_validator(tmp_path):
    base = {"mode": "real", "data": {"root": str(tmp_path), "official_validator": "utils/validate_submission.py"}}
    with pytest.raises(OutputContractError):
        official_gate(base, str(tmp_path))
    assert official_gate(base, str(tmp_path), allow_missing=True)["ran"] is False


def test_validator_invoked_with_current_interpreter(tmp_path):
    root = tmp_path / "sr"
    (root / "utils").mkdir(parents=True)
    (root / "utils" / "validate_submission.py").write_text(
        "import sys\nprint(sys.argv[1:])\n", encoding="utf-8")
    g = official_gate({"mode": "real", "data": {"root": str(root)}}, str(tmp_path))
    assert "--matching" in g["stdout"] and "--test-dir" in g["stdout"] and sys.executable in g["cmd"]


def test_package_structure(small_cfg, tmp_path):
    import zipfile
    from src.data.adapters import make_adapter
    from tools.make_package import build
    ents, recs, _ = make_adapter(small_cfg).load("test")
    cl = {e: [] for e in ents.entity_id}
    write_outputs(small_cfg["output"]["dir"], list(ents.entity_id), cl, cl)
    doc = tmp_path / "Documentation_template.md"
    doc.write_text("# doc", encoding="utf-8")
    with pytest.raises(SystemExit):   # synthetic mode must refuse
        build("team", str(doc), small_cfg["output"]["dir"], str(tmp_path / "x.zip"), small_cfg)
    z = build("team", str(doc), small_cfg["output"]["dir"], str(tmp_path / "t.zip"), small_cfg, allow_synthetic=True)
    names = set(zipfile.ZipFile(z).namelist())
    for req in ("output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
                "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
                "code/business_entity_resolution/src/pipeline/predict.py"):
        assert req in names, req
