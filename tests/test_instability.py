import numpy as np
import pytest

from higgsflow.rge.engine import compute_betas
from higgsflow.running.boundary import SMInputs
from higgsflow.running.solver import Runner
from higgsflow.theory.fields import WeylFermion as W
from higgsflow.theory.model import Model


def test_sm_instability_scale():
    sm = compute_betas(Model())
    r = Runner(sm, sm).run({}, 1e30, SMInputs())
    assert r.status == "unstable"
    # two-loop running from NNLO boundary conditions: μ_I ~ 10^10 GeV
    assert 9.0 < np.log10(r.mu_I) < 11.5


def test_new_yukawas_lower_mu_I():
    m = Model()
    m.add(W("L4", "1", 2, "-1/2"), W("L4c", "1", 2, "1/2"), W("N", "1", 1, 0))
    run = Runner(compute_betas(m), compute_betas(Model()))
    r0 = run.run({}, 1e3)
    r1 = run.run({"yL4N": 0.5, "yL4cNc": 0.5}, 1e3)
    assert r1.mu_I < r0.mu_I


def test_divergence_is_reported_not_raised():
    """Huge Yukawas hit the perturbativity limit; a broken beta gives 'failed', never 'stable'."""
    m = Model()
    m.add(W("L4", "1", 2, "-1/2"), W("L4c", "1", 2, "1/2"), W("N", "1", 1, 0))
    run = Runner(compute_betas(m), compute_betas(Model()))
    r = run.run({"yL4N": 3.0, "yL4cNc": 3.0}, 1e3)   # λ is driven negative within ~1 e-fold
    assert r.status in ("unstable", "nonperturbative") and r.mu_stop < 1e5
    r = run.run({"yL4N": 5.0}, 1e3)                  # above sqrt(4 pi) already at M
    assert r.status == "nonperturbative"

    def blow_up(t, x):                                 # betas that produce NaN after 1 e-fold
        return np.full_like(x, np.nan) if t > np.log(1e4) else np.zeros_like(x)
    run.f_full = blow_up
    r = run.run({"yL4N": 0.5}, 1e3)
    assert r.status == "failed" and r.message


def test_scan_boundaries_are_refined(tmp_path):
    from higgsflow.output.plots import plot_instability
    from higgsflow.running.scan import run_scan
    m = Model.from_dict({"fields": ["N:(1,1,0)"], "include_sm_mixing": True, "exclude": ["yNL1", "yNL2"]})
    r = Runner(compute_betas(m), compute_betas(Model()))
    ys = np.linspace(2.5, 4.0, 4)
    sc = run_scan(r, {"y": ys, "M": np.array([1e3, 1e5])}, links={"y": ["yNL3"]})
    rows = list(sc.boundaries())
    assert len(rows) == 2 and sc.edge_axis == "y"
    for other, e, below, above in rows:
        assert (below, above) == ("unstable", "nonperturbative")
        assert r.run({"yNL3": e - 1e-2}, other["M"]).status == below
        assert r.run({"yNL3": e + 1e-2}, other["M"]).status == above
    plot_instability(tmp_path / "map.pdf", sc)
    sc1 = run_scan(r, {"y": ys}, {"M": 1e3}, links={"y": ["yNL3"]})       # 1D: same refinement
    assert len(list(sc1.boundaries())) == 1
    plot_instability(tmp_path / "line.pdf", sc1)


def test_g2_landau_pole_is_labelled():
    """Many SU(2) triplets: g_2 reaches its limit with λ > 0 -> 'g2_nonpert', not 'nonperturbative'."""
    m = Model.from_dict({"fields": ["Sigma:6x(1,3,0)"]})       # no Yukawas: gauge running only
    r = Runner(compute_betas(m), compute_betas(Model())).run({}, 2e2)
    assert r.status == "g2_nonpert" and r.mu_I is None
    assert r.coupling("lam")[np.isfinite(r.coupling("lam"))][-1] > 0


def test_running_continues_past_lambda_zero_and_finds_the_minimum():
    """The SM runs on after λ = 0 up to M_Pl; λ has its minimum (≈ -0.01 near 10^17 GeV)
    inside the range. A run that stops while λ still decreases has no minimum."""
    sm = compute_betas(Model())
    r = Runner(sm, sm).run({}, 1e30, SMInputs())
    assert r.status == "unstable" and r.mu_stop > 1e18
    assert -0.03 < r.lam_min < 0 and 1e15 < r.mu_min < 1e19
    lam = r.coupling("lam")
    assert np.nanmin(lam) == pytest.approx(r.lam_min, abs=1e-5)
    m = Model.from_dict({"fields": ["N:(1,1,0)"], "include_sm_mixing": True, "exclude": ["yNL1", "yNL2"]})
    run = Runner(compute_betas(m), sm, model=m)
    big = run.run({"yNL3": 1.0}, 1e3)                 # λ still falling at M_Pl: no minimum
    assert big.status == "unstable" and big.lam_min is None and big.mu_min is None
    lam = big.coupling("lam")
    assert lam[np.isfinite(lam)][-1] < lam[np.isfinite(lam)][-2]


def test_scan_keeps_the_minimum_and_its_boundary(tmp_path):
    from higgsflow.output.plots import plot_instability
    from higgsflow.running.scan import run_scan
    m = Model.from_dict({"fields": ["N:(1,1,0)"], "include_sm_mixing": True, "exclude": ["yNL1", "yNL2"]})
    r = Runner(compute_betas(m), compute_betas(Model()), model=m)
    sc = run_scan(r, {"y": np.linspace(0.05, 1.0, 5), "M": np.array([1e3, 1e5])}, links={"y": ["yNL3"]})
    assert np.isfinite(sc.lam_min[0]).all() and not np.isfinite(sc.lam_min[-1]).any()
    assert sc.sm_lam_min is not None
    e = sc.min_edges[np.isfinite(sc.min_edges)]
    assert len(e) == 2
    for y, M in zip(e, (1e3, 1e5)):                     # bisected: minimum just below, none above
        assert r.run({"yNL3": y * 0.97}, M).lam_min is not None
        assert r.run({"yNL3": y * 1.03}, M).lam_min is None
    plot_instability(tmp_path / "min.pdf", sc, minima=True)


def test_fixed_mass_field():
    """A field with its own mass: at mass = M it is the old common threshold; far above μ_max
    it is absent; the SM reference run never includes it."""
    from higgsflow.config import spec_from_dict
    from higgsflow.pipeline import build_runners, compute_steps, compute_systems

    def runner(*x5):
        spec = spec_from_dict({"model": {"name": "t", "fields": ["(1,1,0)", *x5], "include_sm_mixing": True},
                               "scan": {"y": {"linspace": [0.1, 0.5, 2]}}, "run": {"plots": False}}).prepared()
        spec.model.check_fixed()
        sy, sm, _ = compute_systems(spec.model, "native", 2)
        return build_runners(spec, sy, sm, compute_steps(spec.model, sy, 2))[0]["native"]

    common, at_M, absent = runner("X5:(1,5,0)"), runner("X5:(1,5,0)@1e4"), runner("X5:(1,5,0)@1e40")
    base = runner()
    p = {"yF1L3": 0.4}
    assert common.run(p, 1e4).mu_min == pytest.approx(at_M.run(p, 1e4).mu_min, rel=1e-6)
    assert absent.run(p, 1e3).mu_I == pytest.approx(base.run(p, 1e3).mu_I, rel=1e-6)
    assert at_M.run({}, 1e30, sm_only=True).mu_I == pytest.approx(base.run({}, 1e30).mu_I, rel=1e-6)



def test_singlet_betas_and_threshold():
    """Complex singlet S: one-loop betas of the scalar sector, λ jump κ²/4λ_S at m_S, and
    κ = 0 reproduces the run without S (y_S is tiny for M ≪ m_S)."""
    import sympy as sp

    from higgsflow.config import spec_from_dict
    from higgsflow.pipeline import build_runners, compute_steps, compute_systems
    from higgsflow.theory.model import Singlet

    m = Model()
    m.add(W("D", "3", 1, "-1/3"), W("Dc", "3b", 1, "1/3"))
    m.singlet = Singlet(1e8, 0.2, 0.2)
    bs = compute_betas(m, 1, use_cache=False)
    lam, lamS, kap, yS, yt, g1, g2 = sp.symbols("lambda lamS kap yS yt g1 g2", real=True)
    assert sp.expand(bs.expr("lamS", 1) - (20 * lamS**2 + 2 * kap**2 + 12 * lamS * yS**2 - 6 * yS**4)) == 0
    assert sp.expand(bs.expr("kap", 1) - kap * (12 * lam + 8 * lamS + 4 * kap + 6 * yt**2 + 6 * yS**2
                                                 - sp.Rational(9, 2) * g2**2 - sp.Rational(3, 2) * g1**2)) == 0
    assert sp.expand(bs.expr("lam", 1)).coeff(kap, 2) == 1

    def runner(singlet):
        mdl = {"name": "t", "fields": ["(1,1,0)"], "include_sm_mixing": True}
        if singlet:
            mdl["singlet"] = singlet
        spec = spec_from_dict({"model": mdl, "scan": {"y": {"linspace": [0.1, 0.5, 2]}},
                               "run": {"plots": False}}).prepared()
        sy, sm, _ = compute_systems(spec.model, "native", 2)
        return build_runners(spec, sy, sm, compute_steps(spec.model, sy, 2))[0]["native"]

    p = {"yF1L3": 0.3}
    base, k0 = runner(None), runner({"m_S": 1e8, "lamS": 0.2, "kappa": 0.0})
    k2 = runner({"m_S": 1e8, "lamS": 0.2, "kappa": 0.2})
    assert k0.run(p, 1e4).mu_I == pytest.approx(base.run(p, 1e4).mu_I, rel=1e-3)
    r = k2.run(p, 1e4)
    lamv, t = r.coupling("lam"), r.t
    at = np.flatnonzero(np.isclose(t, np.log(1e8)))       # the two sides of the threshold
    below, above = lamv[at[0]], lamv[at[-1]]
    assert above - below == pytest.approx(0.05, abs=2e-3)
    assert r.status == "stable"
