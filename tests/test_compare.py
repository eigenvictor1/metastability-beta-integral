"""Native betas must agree with PyR@TE on benchmark models (skipped without PyR@TE)."""
import pytest

from higgsflow.pyrate.bridge import run_pyrate
from higgsflow.pyrate.compare import compare
from higgsflow.pyrate.install import PyRATEUnavailable, find_pyrate
from higgsflow.rge.engine import compute_betas
from higgsflow.theory.fields import WeylFermion as W
from higgsflow.theory.model import Model

try:
    find_pyrate()
    HAVE = True
except PyRATEUnavailable:
    HAVE = False

BENCH = {
    "SM": [],
    "VLL_doublets_singlet": [W("L4", "1", 2, "-1/2"), W("L4c", "1", 2, "1/2"), W("N", "1", 1, 0)],
    "real_triplet": [W("S", "1", 3, 0), W("L4", "1", 2, "-1/2"), W("L4c", "1", 2, "1/2")],
    "triplet_Y1": [W("T", "1", 3, 1), W("Tc", "1", 3, -1), W("L4", "1", 2, "-1/2"), W("L4c", "1", 2, "1/2")],
    "VLQ": [W("Qp", "3", 2, "1/6"), W("Qpc", "3b", 2, "-1/6"), W("Up", "3", 1, "2/3"), W("Upc", "3b", 1, "-2/3")],
    "no_yukawas": [W("X", "8", 1, 0), W("D", "1", 4, "1/2"), W("Dc", "1", 4, "-1/2")],
}


@pytest.mark.pyrate
@pytest.mark.skipif(not HAVE, reason="PyR@TE not installed (higgsflow install-pyrate)")
@pytest.mark.parametrize("name", list(BENCH))
def test_native_vs_pyrate(name, tmp_path):
    m = Model()
    m.add(*BENCH[name])
    rep = compare(compute_betas(m), run_pyrate(m, workdir=tmp_path))
    assert rep.agree, rep.to_markdown()


@pytest.mark.pyrate
@pytest.mark.skipif(not HAVE, reason="PyR@TE not installed (higgsflow install-pyrate)")
def test_copies_vs_pyrate(tmp_path):
    m = Model()
    m.add(W("L4", "1", 2, "-1/2"), W("L4c", "1", 2, "1/2"), W("N", "1", 1, 0, copies=2))
    assert "yL4N2" in m.new_yukawa_names()
    rep = compare(compute_betas(m), run_pyrate(m, workdir=tmp_path))
    assert rep.agree, rep.to_markdown()
