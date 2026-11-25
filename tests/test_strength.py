import numpy as np

from higgsflow.rge.engine import compute_betas
from higgsflow.running.solver import Runner
from higgsflow.running.strength import YukawaStrength
from higgsflow.theory.model import Model


def _x(model, value):
    return np.array([value if c in model.new_yukawa_names() else 0.5 for c in model.coupling_names()])


def test_effective_yukawa():
    one = Model.from_dict({"fields": ["N:(1,1,0)"], "include_sm_mixing": True, "exclude": ["yNL1", "yNL2"]})
    assert np.isclose(YukawaStrength(one, one.coupling_names())(_x(one, 0.7)), 0.7)
    two = Model.from_dict({"fields": ["N:2x(1,1,0)"], "include_sm_mixing": True})   # one combination, √6 y
    assert np.isclose(YukawaStrength(two, two.coupling_names())(_x(two, 0.7)), np.sqrt(6) * 0.7)


def test_nonperturbative_uses_strength():
    two = Model.from_dict({"fields": ["N:2x(1,1,0)"], "include_sm_mixing": True})
    full, sm = compute_betas(two), compute_betas(Model())
    vals = {c: 1.6 for c in two.new_yukawa_names()}             # each < √(4π), but √6·1.6 > √(4π)
    assert Runner(full, sm).run(vals, 1e3).status != "nonperturbative"
    assert Runner(full, sm, model=two).run(vals, 1e3).status == "nonperturbative"
