import numpy as np
import pytest

from higgsflow.rge.engine import compute_betas
from higgsflow.running.solver import Runner
from higgsflow.running.threshold import LOOP, MassError, ThresholdMatching
from higgsflow.theory.model import Model

SEESAW = {"include_sm_mixing": True, "exclude": ["yNL1", "yNL2"], "fields": ["N:(1,1,0)"]}
VLL = {"fields": ["L4:(1,2,-1/2)", "L4c:(1,2,1/2)", "N:(1,1,0)"]}


def _V_full(tm, h, Y, a=2):
    m = np.linalg.svd(tm.M0 + h * Y[a], compute_uv=False)
    x = m[m > 1e-12] ** 2
    return float(-2 * np.sum(x ** 2 / 4 * (np.log(x) - 1.5)) / LOOP)


@pytest.mark.parametrize("spec,vals", [(VLL, {"yL4N": 0.6, "yL4cNc": 0.3}),
                                       ({"fields": ["S:(1,3,0)", "L4:(1,2,-1/2)", "L4c:(1,2,1/2)"]},
                                        {"ySL4": 0.5, "ySL4cc": 0.8})])
def test_self_energy_matches_effective_potential(spec, vals):
    """Π(0) from the loop functions must equal ∂²V_CW/∂h²: fixes all normalisations."""
    tm = ThresholdMatching(Model.from_dict(spec))
    Y = tm.yukawa(vals)
    for a in (0, 2):                       # any Higgs direction gives the same answer
        e = 1e-3
        vpp = (_V_full(tm, e, Y, a) - 2 * _V_full(tm, 0, Y, a) + _V_full(tm, -e, Y, a)) / e ** 2
        assert np.isclose(tm.self_energy(Y, a, a, derivative=False, heavy_only=False), vpp, rtol=1e-5)


def test_seesaw_analytic():
    """One heavy Majorana N: δλ = -y⁴/16π², Π' = -y²/(2·16π²)."""
    tm = ThresholdMatching(Model.from_dict(SEESAW))
    y = 0.7
    Y = tm.yukawa({"yNL3": y})
    assert np.isclose(4 * tm.c4(Y), -y ** 4 / LOOP, rtol=1e-6)
    assert np.isclose(tm.self_energy(Y), -y ** 2 / (2 * LOOP), rtol=1e-9)


def test_matching_in_running():
    m = Model.from_dict(SEESAW)
    full, sm = compute_betas(m), compute_betas(Model())
    tree = Runner(full, sm).run({"yNL3": 0.8}, 1e6)
    loop = Runner(full, sm, threshold=ThresholdMatching(m)).run({"yNL3": 0.8}, 1e6)
    mt = loop.matching
    assert mt is not None and mt.delta_lambda < 0
    assert np.isclose(mt.lam_SM, mt.lam_full * (1 + 2 * mt.dZ_H) + mt.delta_lambda)
    assert loop.mu_I != tree.mu_I
    assert tree.matching is None


def test_missing_partner():
    with pytest.raises(MassError):
        ThresholdMatching(Model.from_dict({"fields": ["(1,2,-1/2)", "(1,1,0)"]}))
