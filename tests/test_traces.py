import numpy as np
import pytest

from higgsflow.config import TraceSpec
from higgsflow.rge.engine import compute_betas
from higgsflow.running.scan import even_subset, run_samples, run_scan, sample_couplings
from higgsflow.running.solver import Runner
from higgsflow.running.traces import TR4, YukawaTraces, with_trace_axis
from higgsflow.theory.model import Model

SEESAW = {"fields": ["N:(1,1,0)"], "include_sm_mixing": True, "exclude": ["yNL1", "yNL2"]}


def test_seesaw_traces():
    tr = YukawaTraces(Model.from_dict(SEESAW))
    t2, t4 = tr({"yNL3": 0.8, "yt": 0.9})            # y_t is not included
    assert np.isclose(t2, 0.8 ** 2) and np.isclose(t4, 0.8 ** 4 / 2)


def test_samples_cover_the_trace_range():
    rng = np.random.default_rng(0)
    v = sample_couplings(["a", "b", "c"], 500, (0.0, 1.5), rng)
    s2 = (v ** 2).sum(axis=1)
    assert v.min() >= 0 and np.allclose(s2.max(), 1.5 ** 2, rtol=0.05) and s2.min() < 0.05
    x, y = rng.uniform(size=20000) ** 3, rng.uniform(size=20000)    # crowded at small x
    k = even_subset(x, y, 400, rng)
    assert 400 <= len(k) < 480
    counts, _ = np.histogram(x[k], bins=4, range=(0, 1))
    assert counts.min() > 0.6 * counts.max()                          # roughly even in x


def test_trace_spec():
    assert TraceSpec.from_dict(None) is None
    assert TraceSpec.from_dict({"M": 1e3}).range == (0.0, 1.5)
    with pytest.raises(ValueError):
        TraceSpec.from_dict({"M": 1e3, "foo": 1})
    with pytest.raises(ValueError):
        TraceSpec.from_dict({"samples": 10})


def test_trace_axis_and_samples():
    m = Model.from_dict(SEESAW)
    r = Runner(compute_betas(m), compute_betas(Model()))
    links = {"y": ["yNL3"]}
    sc = run_scan(r, {"y": np.linspace(0, 1, 5), "M": np.array([1e3, 1e5])}, links=links)
    t = with_trace_axis(sc, YukawaTraces(m), links, 1e3)
    assert list(t.axes) == [TR4, "M"] and np.allclose(t.axes[TR4], np.linspace(0, 1, 5)[1:] ** 4 / 2)
    assert t.mu_I.shape == (4, 2)
    res = run_samples(r, YukawaTraces(m), ["yNL3"], 1e3, 6, (0.0, 1.0))
    assert np.allclose(res.tr4, res.tr2 ** 2 / 2) and 6 <= len(res.status) < 12
