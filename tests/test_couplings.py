import numpy as np
import pytest

from higgsflow.theory import couplings as cp
from higgsflow.theory.model import Model


def test_sm_registry():
    assert cp.SM_NAMES == ("g1", "g2", "g3", "yt", "lam")
    assert cp.GAUGE == ("g1", "g2", "g3")
    assert cp.get("lam").limit == pytest.approx(4 * np.pi)
    assert cp.get("yt").limit == pytest.approx(np.sqrt(4 * np.pi))
    assert cp.get("lam").symbol == "lambda"
    with pytest.raises(KeyError):
        cp.get("foo")


def test_new_yukawa_and_order():
    name = cp.new_yukawa_name("L", "N", True)
    assert name == "yLNc"
    assert cp.new_yukawa_name("N_1", "L_3", False) == "yN1L3"
    c = cp.get(name)
    assert c.kind == "yukawa" and not c.sm and cp.latex(name) == r"$y_{\mathrm{LNc}}$"
    assert not cp.is_new_yukawa("y") and not cp.is_new_yukawa("yt")
    assert cp.ordered(["yt", name]) == ["g1", "g2", "g3", "yt", name, "lam"]


def test_model_uses_registry():
    m = Model.from_dict({"name": "t", "fields": ["N:(1,1,0)"]})
    names = m.coupling_names() if callable(getattr(m, "coupling_names", None)) else m.coupling_names
    assert names[:3] == list(cp.GAUGE) and names[-1] == cp.QUARTIC
    for n in names:
        cp.get(n)
