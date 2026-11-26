"""Curve tracing (running/curves.py) on analytic stand-ins for the runner."""
from types import SimpleNamespace

import numpy as np

from higgsflow.running.curves import TraceTask, join_ends, scaled_grid, seeds, trace


class FakeRunner:
    """λ_low = (log10 y + 0.5)^2 + (log10 M - 4)^2 / 4 - 0.25 (an ellipse in the log plane:
    unstable inside); μ_I = 10^(6 + log10 M) inside, μ_min = 10^8 everywhere."""

    def run(self, vals, M, inputs=None):
        ly, lm = np.log10(vals["y"]), np.log10(M)
        lam = (ly + 0.5) ** 2 + (lm - 4) ** 2 / 4 - 0.25
        unstable = lam < 0
        return SimpleNamespace(status="unstable" if unstable else "stable",
                               mu_I=10 ** (6 + lm) if unstable else None,
                               lam_low=lam, lam_min=lam, mu_min=1e8, mu_stop=1e19)


def _task(kind, level, seed, lo, hi):
    return TraceTask(kind, level, seed, "y", lo, hi, {}, None, 1e3, None)


def test_traced_boundary_is_on_the_curve_and_oriented():
    names, grids = ["y", "M"], [np.logspace(-2, 1, 25), np.logspace(2, 6, 25)]
    yname, u, v, lo, hi = scaled_grid(names, grids)
    r = FakeRunner()
    LL = np.array([[r.run({"y": y}, M).lam_low for y in grids[0]] for M in grids[1]])     # [M, y]
    sd = seeds(u, v, LL, 0.0)
    assert len(sd) == 1
    c = trace(r, _task("lam_low", 0.0, sd[0], lo, hi))
    assert c is not None and c.ends == ("closed", "closed")
    lam = (np.log10(c.y) + 0.5) ** 2 + (np.log10(c.M) - 4) ** 2 / 4 - 0.25
    assert np.max(np.abs(lam)) < 1e-4
    # the unstable side (λ < 0, inside the ellipse) is on the left of the direction of travel
    P = np.column_stack([(np.log10(c.M) - lo[0]) / (hi[0] - lo[0]), (np.log10(c.y) - lo[1]) / (hi[1] - lo[1])])
    area = 0.5 * np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1])
    assert area > 0                       # counter-clockwise around the inside


def test_mu_contour_ends_on_the_boundary():
    names, grids = ["y", "M"], [np.logspace(-2, 1, 25), np.logspace(2, 6, 25)]
    yname, u, v, lo, hi = scaled_grid(names, grids)
    r = FakeRunner()
    MU = np.array([[np.log10(r.run({"y": y}, M).mu_I) if r.run({"y": y}, M).mu_I else np.nan
                    for y in grids[0]] for M in grids[1]])
    curves = [trace(r, _task("lam_low", 0.0, seeds(u, v, np.array(
        [[r.run({"y": y}, M).lam_low for y in grids[0]] for M in grids[1]]), 0.0)[0], lo, hi))]
    sd = seeds(u, v, MU, 10.0)                    # log10 μ_I = 10 at M = 10^4: a vertical chord
    assert sd
    c = trace(r, _task("mu_I", 10.0, sd[0], lo, hi))
    assert c is not None and set(c.ends) <= {"stable", "lost"}
    assert np.max(np.abs(np.log10(c.M) - 4)) < 1e-3
    # the ends lie on the boundary λ = 0 after joining (μ_min = 10^8 there, not the level):
    # no joint, since log10 μ_min never equals the level on this boundary
    joined = join_ends(curves + [c])
    assert len(joined[-1].y) == len(c.y)
