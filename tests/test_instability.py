import numpy as np

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
    r1 = run.run({"y_L4_N": 0.5, "y_L4c_N_c": 0.5}, 1e3)
    assert r1.mu_I < r0.mu_I


def test_divergence_is_reported_not_raised():
    """Huge Yukawas hit the perturbativity limit; a broken beta gives 'failed', never 'stable'."""
    m = Model()
    m.add(W("L4", "1", 2, "-1/2"), W("L4c", "1", 2, "1/2"), W("N", "1", 1, 0))
    run = Runner(compute_betas(m), compute_betas(Model()))
    r = run.run({"y_L4_N": 3.0, "y_L4c_N_c": 3.0}, 1e3)   # λ is driven negative within ~1 e-fold
    assert r.status in ("unstable", "nonperturbative") and r.mu_stop < 1e5
    r = run.run({"y_L4_N": 5.0}, 1e3)                  # above sqrt(4 pi) already at M
    assert r.status == "nonperturbative"

    def blow_up(t, x):                                 # betas that produce NaN after 1 e-fold
        return np.full_like(x, np.nan) if t > np.log(1e4) else np.zeros_like(x)
    run.f_full = blow_up
    r = run.run({"y_L4_N": 0.5}, 1e3)
    assert r.status == "failed" and r.message
