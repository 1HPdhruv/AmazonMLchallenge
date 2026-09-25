import copy

import pytest

from src.pipeline.core import load_cfg
from tests.synthetic.generate_data import generate


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    """Tiny synthetic tree in the OFFICIAL layout + a base config pointing at it (outputs to temp dirs)."""
    d = tmp_path_factory.mktemp("synth")
    generate(400, 11, str(d), n_test=150)
    base = copy.deepcopy(load_cfg("base"))
    base["mode"] = "synthetic"
    base["data"]["root"] = str(d)
    base["paths"] = {k: str(tmp_path_factory.mktemp(k)) for k in base["paths"]}
    base["output"] = {"dir": str(tmp_path_factory.mktemp("output"))}
    return base
