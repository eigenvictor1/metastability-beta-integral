"""Curves of a 2D scan (M and one Yukawa parameter) traced to high precision.

Instead of interpolating between grid points, each curve is followed point by point with
runs placed on it:

* the boundary of the unstable region, λ_low = 0 (λ_low: the lowest λ along the running;
  it is < 0 exactly when λ crosses zero, so this is where the stable region ends);
* the metastability line λ_low = META_LAMBDA inside the unstable region;
* every integer contour log10(μ_I/GeV) = c, and the contour μ_I = μ_I^SM (kind 'mu_I_sm').

Tracing (predictor-corrector continuation) happens in the plane of the plot, u = log10 M and
v = log10 y, both scaled to [0, 1] over the scanned range:

1. seeds: the contour of the grid values (contourpy) gives each connected piece of a curve
   roughly; its middle point is moved onto the curve by a 1-D root search along the gradient;
2. predictor: a step of length h along the tangent (from the last two points);
3. corrector: root search for the curve along the normal through the predicted point, started
   from a secant step with the slope found at the previous point and finished with Brent's
   method once the root is bracketed (tolerance XTOL);
4. step control: the step is halved if the corrector fails, moves the point by more than
   0.3 h, the curve turns by more than MAX_TURN, or the new point has a status where the curve
   does not exist (for μ_I contours: not unstable; for all: non-perturbative, failed); it grows
   by half where the curve is straight; it stays between H_MIN and H_MAX;
5. the curve is followed in both directions from the seed until it leaves the scanned range
   (plus a margin), closes on itself, or can no longer be continued (step below H_MIN).

Near the stable region μ_I has a square-root edge (λ just dips below zero around its minimum:
log10 μ_I ≈ log10 μ_min − sqrt(−λ_min / a)), so a μ_I contour meets the boundary tangentially,
at the point where log10 μ_min = c. There it ends with status 'stable'; that end point is found
afterwards on the traced boundary (interpolating log10 μ_min along it) and appended.

On the stable side the contour function is continued with the limit of μ_I at the boundary
(μ_min, or μ_stop where λ has no minimum), so that it is continuous for the root search.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .boundary import SMInputs
from .solver import GAUGE_NONPERT, Runner

log = logging.getLogger(__name__)

META_LAMBDA = -0.06          # metastable if the lowest λ stays above this (see output/plots.py)
CUT_SHORT = -1.0             # meta_value of a run cut short by non-perturbativity (counts as unstable)


def meta_value(lam_low, lam_min, mu_stop, mu_max):
    """The λ compared with META_LAMBDA (metastable above it): λ at its minimum; without a
    minimum, λ where the running ends if it reached μ_max (still decreasing all the way).
    A run cut short by non-perturbativity with λ still decreasing tells nothing about how deep
    λ goes: CUT_SHORT (unstable), so that no metastability line runs along the edge of the
    non-perturbative region. Works on numbers and arrays (NaN lam_min: no minimum)."""
    lam_low, lam_min, mu_stop = (np.asarray(v, dtype=float) for v in (lam_low, lam_min, mu_stop))
    full = mu_stop >= 0.99 * mu_max
    out = np.where(np.isfinite(lam_min), lam_min, np.where(full, lam_low, CUT_SHORT))
    return out if out.ndim else float(out)

XTOL = 2e-5                  # root tolerance, in units of the scanned range (plot width/height)
H_START = 0.01               # first step
H_MIN = 2e-4                 # steps below this end the curve
H_MAX = 0.03                 # longest step
MAX_TURN = 0.25              # largest turn of the curve per step (rad)
MARGIN = 0.02                # the curve is followed this far past the scanned range
MAX_POINTS = 3000            # per direction
VALID = ("unstable", "stable", *GAUGE_NONPERT)    # statuses where λ_low is meaningful


@dataclass
class Curve:
    """A traced curve: kind 'mu_I' (log10 μ_I = level), 'mu_I_sm' (the same at the SM value)
    or 'lam_low' (λ_low = level).

    Points in scan parameters (y: the Yukawa parameter, M in GeV), ordered so that the side
    where the function is below the level lies on the left of the direction of travel (for
    'lam_low' at 0: the unstable side). aux: log10 of the limit of μ_I at each point (μ_min, or
    μ_stop without minimum), used to end μ_I contours on the boundary. ends: why the curve ends
    at its start and at its end ('range', 'closed', 'stable', 'invalid', 'lost').
    """
    kind: str
    level: float
    y: np.ndarray
    M: np.ndarray
    aux: np.ndarray
    ends: tuple[str, str] = ("range", "range")
    runs: int = 0


@dataclass
class TraceTask:
    """Everything a worker needs to trace one curve from one seed."""
    kind: str
    level: float
    seed: tuple[float, float]          # (u, v), roughly on the curve
    yname: str
    lo: tuple[float, float]            # log10 of the scanned range: (M, y) lower ends
    hi: tuple[float, float]            # upper ends
    fixed: dict[str, float]
    links: dict | None
    M_default: float
    inputs: SMInputs | None
    extra: dict = field(default_factory=dict)


class _Field:
    """The function whose zero is the curve, at scaled coordinates p = (u, v); runs cached."""

    def __init__(self, runner: Runner, task: TraceTask):
        self.runner, self.task = runner, task
        self.cache: dict[tuple[float, float], tuple[float, str, float]] = {}
        self.runs = 0

    def params(self, p: np.ndarray) -> tuple[float, float]:
        """(y, M) at the scaled point p."""
        lo, hi = self.task.lo, self.task.hi
        M = 10 ** (lo[0] + p[0] * (hi[0] - lo[0]))
        y = 10 ** (lo[1] + p[1] * (hi[1] - lo[1]))
        return float(y), float(M)

    def __call__(self, p: np.ndarray) -> tuple[float, str, float]:
        """(value, status, aux) at p: value = f(p) - level; NaN where the curve cannot pass."""
        key = (round(float(p[0]), 12), round(float(p[1]), 12))
        if key in self.cache:
            return self.cache[key]
        from .scan import resolve_params
        t = self.task
        y, M = self.params(p)
        vals, M = resolve_params({**t.fixed, t.yname: y, "M": M}, t.links, t.M_default)
        r = self.runner.run(vals, M, t.inputs)
        self.runs += 1
        lim = r.mu_min if r.mu_min is not None else r.mu_stop
        aux = float(np.log10(lim)) if lim and lim > 0 else np.nan
        if r.status not in VALID or r.lam_low is None:
            out = (np.nan, r.status, aux)
        elif t.kind == "lam_low" and t.level == 0:
            out = (float(r.lam_low) - t.level, r.status, aux)
        elif t.kind == "lam_low":             # the metastability line (see meta_value)
            lm = np.nan if r.lam_min is None else r.lam_min
            out = (meta_value(r.lam_low, lm, r.mu_stop, self.runner.numerics.mu_max) - t.level,
                   r.status, aux)
        elif r.status == "unstable" and r.mu_I is not None:
            out = (float(np.log10(r.mu_I)) - t.level, r.status, aux)
        else:                          # stable side: continued with the boundary limit of μ_I
            out = (aux - t.level, r.status, aux)
        self.cache[key] = out
        return out

    def ok(self, status: str) -> bool:
        """Whether the curve exists at a point with this status."""
        return status == "unstable" if self.task.kind != "lam_low" else status in VALID


def _root_along(F: _Field, p: np.ndarray, d: np.ndarray, slope: float | None, smax: float
                ) -> tuple[np.ndarray, float] | None:
    """Point p + s d with F = 0, |s| <= smax (d a unit vector); also returns the slope dF/ds.
    Secant steps from s = 0 (first one with the given slope) until the root is bracketed,
    then Brent's method."""
    f0 = F(p)[0]
    if not np.isfinite(f0):
        return None
    if f0 == 0:
        return p, slope or 1.0
    s_prev, f_prev = 0.0, f0
    if slope is None or not np.isfinite(slope) or slope == 0:
        s1 = 0.1 * smax
        f1 = F(p + s1 * d)[0]
        if not np.isfinite(f1):
            s1, f1 = -s1, F(p - s1 * d)[0]
        if not np.isfinite(f1) or f1 == f0:
            return None
        slope = (f1 - f0) / s1
        if f0 * f1 < 0:
            s = brentq(lambda s: _finite(F(p + s * d)[0], f0), min(0, s1), max(0, s1), xtol=XTOL)
            return p + s * d, slope
        s_prev, f_prev = s1, f1
    s = -f0 / slope
    for _ in range(8):
        s = float(np.clip(s, -smax, smax))
        f = F(p + s * d)[0]
        if not np.isfinite(f):
            s *= 0.5
            continue
        if f == 0:
            return p + s * d, slope
        if f * f0 < 0:                 # bracketed between 0 and s
            a, b = (0.0, s) if s > 0 else (s, 0.0)
            r = brentq(lambda x: _finite(F(p + x * d)[0], f0), a, b, xtol=XTOL)
            return p + r * d, (f - f0) / s
        if f * f_prev < 0 and s != s_prev:        # bracketed between the last two
            a, b = sorted((s, s_prev))
            r = brentq(lambda x: _finite(F(p + x * d)[0], f0), a, b, xtol=XTOL)
            return p + r * d, (f - f_prev) / (s - s_prev)
        if s != s_prev and f != f_prev:            # secant through the last two
            slope_new = (f - f_prev) / (s - s_prev)
            s_prev, f_prev = s, f
            s = s - f / slope_new if slope_new != 0 else 2 * s
        else:
            s_prev, f_prev, s = s, f, 2 * s
        if abs(s_prev) >= smax and abs(s) >= smax:
            return None
    return None


def _finite(v: float, ref: float) -> float:
    """A NaN inside a bracket (curve not defined there) is taken on the side of ref."""
    return v if np.isfinite(v) else ref


def _gradient(F: _Field, p: np.ndarray, h: float = 1e-3) -> np.ndarray | None:
    """Finite-difference gradient of F at p (scaled coordinates)."""
    f0 = F(p)[0]
    parts: list[float] = []
    for e in (np.array([h, 0.0]), np.array([0.0, h])):
        f1 = F(p + e)[0]
        if not np.isfinite(f1):
            f1 = F(p - e)[0]
            if not np.isfinite(f1):
                return None
            parts.append((f0 - f1) / h)
        else:
            parts.append((f1 - f0) / h)
    g = np.array(parts)
    return g if np.all(np.isfinite(g)) and np.hypot(*g) > 0 else None


def _inside(p: np.ndarray) -> bool:
    return bool(np.all(p > -MARGIN) and np.all(p < 1 + MARGIN))


def _follow(F: _Field, p0: np.ndarray, t0: np.ndarray, slope: float) -> tuple[list[np.ndarray], str]:
    """Points of the curve from p0 (on it) in the direction t0 (unit tangent), until it ends."""
    pts, tan, h = [p0], t0, H_START
    while len(pts) < MAX_POINTS:
        if not _inside(pts[-1]):
            return pts, "range"
        if len(pts) > 3 and np.hypot(*(pts[-1] - pts[0])) < 0.75 * h:
            return pts + [pts[0]], "closed"
        p = pts[-1]
        q = p + h * tan
        n = np.array([-tan[1], tan[0]])
        res = _root_along(F, q, n, slope, 0.5 * h)
        reason = "lost"
        if res is not None:
            new, sl = res
            f, status, _ = F(new)
            step = new - p
            dist = float(np.hypot(*step))
            turn = float(np.arccos(np.clip(np.dot(step / dist, tan), -1, 1))) if dist > 0 else np.pi
            if not F.ok(status):
                reason = "stable" if status in ("stable", *GAUGE_NONPERT) else "invalid"
            elif dist > 0 and turn <= MAX_TURN and np.hypot(*(new - q)) <= 0.3 * h:
                pts.append(new)
                slope = sl if np.isfinite(sl) and sl != 0 else slope
                tan = step / dist
                if turn < 0.3 * MAX_TURN:
                    h = min(1.5 * h, H_MAX)
                continue
        h *= 0.5
        if h < H_MIN:
            return pts, reason
    return pts, "lost"


def trace(runner: Runner, task: TraceTask) -> Curve | None:
    """Trace one curve from the task's seed (see the module docstring); None if the seed
    cannot be moved onto it."""
    F = _Field(runner, task)
    p = np.array(task.seed, dtype=float)
    g = _gradient(F, p)
    if g is None:
        return None
    gn = g / np.hypot(*g)
    res = _root_along(F, p, gn, float(np.hypot(*g)), 0.05)
    if res is None:
        return None
    p0, slope = res
    if not F.ok(F(p0)[1]):
        return None
    g = _gradient(F, p0)
    if g is None:
        return None
    # tangent with the side where F < 0 on its left: left normal (-t_y, t_x) = -∇F/|∇F|
    t0 = np.array([-g[1], g[0]]) / np.hypot(*g)
    slope = float(np.hypot(*g))       # first secant guess only (refined from the first step)
    fwd, end_f = _follow(F, p0, t0, slope)
    if end_f == "closed":
        pts, ends = fwd, ("closed", "closed")
    else:
        bwd, end_b = _follow(F, p0, -t0, slope)
        pts, ends = bwd[::-1] + fwd[1:], (end_b, end_f)
    P = np.array(pts)
    yM = np.array([F.params(q) for q in P])
    aux = np.array([F(q)[2] for q in P])
    return Curve(task.kind, task.level, yM[:, 0], yM[:, 1], aux, ends, F.runs)


# --------------------------------------------------------------------------
# Seeds from the grid, and the curves of a scan
# --------------------------------------------------------------------------

def scaled_grid(names: list[str], grids: list[np.ndarray]) -> tuple[str, np.ndarray, np.ndarray,
                                                                    tuple[float, float], tuple[float, float]] | None:
    """(Yukawa parameter, scaled u per M grid value, scaled v per y value (NaN for y <= 0),
    lo, hi) for a 2D scan over M and one positive Yukawa parameter, else None."""
    if len(names) != 2 or "M" not in names:
        return None
    yname = next(n for n in names if n != "M")
    M, y = grids[names.index("M")], grids[names.index(yname)]
    if np.sum(y > 0) < 2 or np.any(M <= 0):
        return None
    lo = (float(np.log10(M.min())), float(np.log10(y[y > 0].min())))
    hi = (float(np.log10(M.max())), float(np.log10(y.max())))
    u = (np.log10(M) - lo[0]) / (hi[0] - lo[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(y > 0, (np.log10(np.where(y > 0, y, 1)) - lo[1]) / (hi[1] - lo[1]), np.nan)
    return yname, u, v, lo, hi


def seeds(u: np.ndarray, v: np.ndarray, Z: np.ndarray, level: float) -> list[tuple[float, float]]:
    """One seed (u, v) per connected piece of the grid contour Z = level (Z indexed [M, y],
    NaN where the curve does not exist): the middle point of the piece."""
    import contourpy
    keep = np.isfinite(v)
    if keep.sum() < 2:
        return []
    z = Z[:, keep].T                                      # contourpy: z[y, x]
    if np.isfinite(z).sum() < 4:
        return []
    gen = contourpy.contour_generator(u, v[keep], np.ma.masked_invalid(z))
    out = []
    for line in gen.lines(level):
        line = np.asarray(line)
        if line.ndim == 2 and len(line) >= 2:
            arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
            k = int(np.searchsorted(arc, 0.5 * arc[-1]))
            out.append((float(line[min(k, len(line) - 1), 0]), float(line[min(k, len(line) - 1), 1])))
    return out


def mu_levels(mu_I: np.ndarray) -> np.ndarray:
    """The integer levels of log10 μ_I drawn in the plots (see output/plots.py)."""
    Z = np.log10(mu_I[np.isfinite(mu_I)])
    if Z.size == 0 or np.ptp(Z) <= 0.2:
        return np.array([])
    return np.arange(np.floor(Z.min()), np.ceil(Z.max()) + 1, 1)


def same_curve(a: Curve, b: Curve, span: tuple[float, float], tol: float = 0.01) -> bool:
    """Whether b runs along a: same kind and level, and most sampled points of b within tol
    (fraction of the scanned range, span = its size in decades of M and y) of a's polyline
    (not all: the two may end slightly differently, e.g. past the edge of the range)."""
    if a.kind != b.kind or a.level != b.level:
        return False
    sc = np.array(span, dtype=float)
    A = np.column_stack([np.log10(a.M), np.log10(a.y)]) / sc
    B = np.column_stack([np.log10(b.M), np.log10(b.y)]) / sc
    if len(A) < 2:
        return False
    P, AB = A[:-1], np.diff(A, axis=0)
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-30)
    close = []
    for q in B[np.linspace(0, len(B) - 1, min(len(B), 15)).astype(int)]:
        t = np.clip(((q - P) * AB).sum(axis=1) / L2, 0, 1)
        close.append(np.min(np.hypot(*(q - P - t[:, None] * AB).T)) <= tol)
    return float(np.mean(close)) >= 0.75


JOIN_LOST = 0.3      # decades: a contour that got lost this close to its end point on the boundary
                     # (squeezed against it) is joined to it too


def join_ends(curves: list[Curve]) -> list[Curve]:
    """μ_I contours ending on the stable boundary ('stable', or 'lost' close to it): append the
    point of the traced boundary (λ_low = 0) where log10 μ_min equals the level, nearest to that
    end."""
    bounds = [c for c in curves if c.kind == "lam_low" and c.level == 0 and len(c.y) > 1]
    if not bounds:
        return curves
    out = []
    for c in curves:
        if c.kind == "lam_low" or not ({"stable", "lost"} & set(c.ends)):
            out.append(c)
            continue
        y, M, aux = list(c.y), list(c.M), list(c.aux)
        for at_end in (False, True):
            why = c.ends[int(at_end)]
            if why not in ("stable", "lost"):
                continue
            q = np.array([np.log10(M[-1] if at_end else M[0]), np.log10(y[-1] if at_end else y[0])])
            best = None
            for b in bounds:
                la = b.aux - c.level
                for i in np.flatnonzero(np.isfinite(la[:-1]) & np.isfinite(la[1:]) & (la[:-1] * la[1:] <= 0)):
                    w = la[i] / (la[i] - la[i + 1]) if la[i] != la[i + 1] else 0.5
                    lm = (1 - w) * np.log10(b.M[i]) + w * np.log10(b.M[i + 1])
                    ly = (1 - w) * np.log10(b.y[i]) + w * np.log10(b.y[i + 1])
                    d = float(np.hypot(lm - q[0], ly - q[1]))
                    if best is None or d < best[0]:
                        best = (d, 10 ** ly, 10 ** lm)
            if best is not None and best[0] < (0.5 if why == "stable" else JOIN_LOST):
                k = len(y) if at_end else 0
                y.insert(k, best[1])
                M.insert(k, best[2])
                aux.insert(k, c.level)
        out.append(Curve(c.kind, c.level, np.array(y), np.array(M), np.array(aux), c.ends, c.runs))
    return out
