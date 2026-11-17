"""RG running with a single new-physics threshold and the search for μ_I.

A run has three steps:

1. SM running from M_t to M (new fermions decoupled), from the boundary values
   in ``boundary.py``;
2. matching at μ = M: the new Yukawas take the values given by the user (defined
   at M); the SM couplings are continuous (tree level) or get the one-loop
   threshold correction of ``threshold.py``;
3. running of the full theory from M to μ_max (default M_Planck).

The running stops at the first of

* λ(μ) = 0        -> status 'unstable', μ_I = μ
* perturbativity  -> status 'nonperturbative' (limits in theory/couplings.py)
* μ = μ_max       -> status 'stable'

If the solver itself gives up (step size underflow, non-finite couplings from
a divergence faster than the perturbativity event can catch, overflow in the
betas), the point gets status 'failed' with a message, instead of an exception
or a wrong 'stable'.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from ..rge.system import BetaSystem
from ..settings import Numerics
from ..theory import couplings as cp
from ..theory.couplings import QUARTIC
from .boundary import SMInputs
from .threshold import Matching, ThresholdMatching

SM_COUPLINGS = list(cp.SM_NAMES)
STATUSES = ("unstable", "stable", "nonperturbative", "failed")
log = logging.getLogger(__name__)

BetaFunction = Callable[[float, np.ndarray], np.ndarray]


class IntegrationError(RuntimeError):
    """Raised inside the solver to stop a run that cannot be integrated."""


@dataclass
class RunResult:
    couplings: list[str]
    t: np.ndarray                 # ln(μ/GeV)
    x: np.ndarray                 # shape (len(couplings), len(t)); NaN for new couplings below M
    status: str                   # one of STATUSES
    mu_I: float | None
    mu_stop: float                # where the running stopped (GeV)
    M: float
    params: dict[str, float] = field(default_factory=dict)
    message: str = ""             # why it failed, if status == 'failed'
    matching: Matching | None = None   # one-loop threshold correction applied at M, if any

    @property
    def mu(self) -> np.ndarray:
        return np.exp(self.t)

    def coupling(self, name: str) -> np.ndarray:
        return self.x[self.couplings.index(name)]


@dataclass
class Segment:
    """Result of integrating one segment (SM or full theory)."""
    t: np.ndarray
    x: np.ndarray                 # rows in the order of the segment's couplings
    status: str                   # 'running' (reached the end) or a final status
    t_stop: float
    message: str = ""


@dataclass(frozen=True)
class RunnerSpec:
    """Picklable description of a Runner, used to rebuild it in worker processes."""
    full: dict
    sm: dict
    loops: int
    model: dict | None            # Model.to_dict() if one-loop matching is used
    numerics: Numerics


class Runner:
    """Holds compiled beta functions for the SM and the full model."""

    def __init__(self, full: BetaSystem, sm: BetaSystem, loops: int = 2,
                 threshold: ThresholdMatching | None = None, numerics: Numerics | None = None):
        """threshold: one-loop matching at M, or None (tree level)."""
        self.full = full
        self.sm = sm.restrict(SM_COUPLINGS)
        self.loops = loops
        self.threshold = threshold
        self.numerics = numerics or Numerics()
        self.f_full = full.numeric(loops)
        self.f_sm = self.sm.numeric(loops)
        self.names = list(full.couplings)
        self.new = [c for c in self.names if c not in SM_COUPLINGS]

    # -- worker processes -------------------------------------------------------
    def spec(self) -> RunnerSpec:
        model = self.threshold.model.to_dict() if self.threshold is not None else None
        return RunnerSpec(self.full.to_json(), self.sm.to_json(), self.loops, model, self.numerics)

    @classmethod
    def from_spec(cls, spec: RunnerSpec) -> Runner:
        threshold = None
        if spec.model is not None:
            from ..theory.model import Model
            threshold = ThresholdMatching(Model.from_dict(spec.model))
        return cls(BetaSystem.from_json(spec.full), BetaSystem.from_json(spec.sm), spec.loops,
                   threshold, spec.numerics)

    # -- public -----------------------------------------------------------------
    def run(self, params: dict[str, float], M: float, inputs: SMInputs | None = None) -> RunResult:
        """Run from M_t to μ_max; never raises for numerical trouble (status 'failed').

        params: values at μ = M of the new couplings (missing ones are 0).
        """
        inputs = inputs or SMInputs()
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                return self._run(params, M, inputs)
        except (IntegrationError, ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            log.debug("integration failed for %s, M=%g: %s", params, M, exc)
            t0 = np.log(inputs.Mt)
            return RunResult(self.names, np.array([t0]), np.full((len(self.names), 1), np.nan),
                             "failed", None, float(inputs.Mt), M, dict(params),
                             message=f"{type(exc).__name__}: {exc}")

    # -- the three steps ----------------------------------------------------------
    def _run(self, params: dict[str, float], M: float, inputs: SMInputs) -> RunResult:
        t0, t_end = np.log(inputs.Mt), np.log(self.numerics.mu_max)
        tM = max(np.log(M), t0)
        out = _Trajectory(self.names, M, params, span=t_end - t0, n_points=self.numerics.n_points)
        x0 = inputs.couplings_at_mt()
        x_sm = np.array([x0[c] for c in SM_COUPLINGS])

        if tM > t0:                                            # 1. SM: M_t -> M
            seg = self._run_sm(x_sm, t0, min(tM, t_end), out.points(t0, min(tM, t_end)))
            out.add(seg, SM_COUPLINGS)
            if seg.status != "running":
                return out.result(seg.status, seg.t_stop, seg.message)
            if tM >= t_end:
                return out.result("stable", t_end)
            x_sm = seg.x[:, -1]

        new_vals = {c: float(params.get(c, 0.0)) for c in self.new}   # 2. matching at M
        x_full, out.matching = self._match(dict(zip(SM_COUPLINGS, map(float, x_sm))), new_vals)
        verdict = self._check_start(x_full)
        if verdict:
            out.add_point(tM, x_full)
            return out.result(verdict, tM)

        seg = self._segment(self.f_full, (tM, t_end), x_full, self.names, out.points(tM, t_end))
        out.add(seg, self.names)                               # 3. full theory: M -> μ_max
        return out.result("stable" if seg.status == "running" else seg.status, seg.t_stop, seg.message)

    def _run_sm(self, x: np.ndarray, t0: float, t1: float, n: int) -> Segment:
        return self._segment(self.f_sm, (t0, t1), x, SM_COUPLINGS, n)

    def _match(self, sm_vals: dict[str, float],
               new_vals: dict[str, float]) -> tuple[np.ndarray, Matching | None]:
        """Full-theory couplings at M (in self.names order) and the correction applied."""
        values, matching = {**sm_vals, **new_vals}, None
        perturbative = all(abs(v) < cp.get(c).limit for c, v in new_vals.items())
        if self.threshold is not None and perturbative:
            values, matching = self.threshold.match(sm_vals, new_vals)
        return np.array([values[c] for c in self.names]), matching

    def _check_start(self, x: np.ndarray) -> str | None:
        """Status if the full theory is already outside its domain at M, else None."""
        if np.any(np.abs(x) >= cp.limits(self.names)):
            return "nonperturbative"
        if x[self.names.index(QUARTIC)] <= 0:      # the matching itself pushes λ negative at M
            return "unstable"
        return None

    # -- integration of one segment -------------------------------------------------
    def _segment(self, f: BetaFunction, t_span: tuple[float, float], x0: np.ndarray,
                 names: list[str], n: int) -> Segment:
        """Integrate the couplings ``names`` (state order) over t_span, with events."""
        idx_lam = names.index(QUARTIC)
        lim = cp.limits(names)
        num = self.numerics

        def lam_zero(t, x):
            return x[idx_lam]
        lam_zero.terminal, lam_zero.direction = True, -1     # type: ignore[attr-defined]

        def nonpert(t, x):
            return float(np.min(lim - np.abs(x)))
        nonpert.terminal, nonpert.direction = True, -1       # type: ignore[attr-defined]

        calls, bad_t = [0], []

        def rhs(t, x):
            calls[0] += 1
            if calls[0] > num.max_rhs_calls:
                raise IntegrationError(f"no convergence after {num.max_rhs_calls} steps near "
                                       f"mu = {np.exp(t):.3g} GeV")
            dx = f(t, x)
            if not np.all(np.isfinite(dx)):
                # remember where it happened and freeze the system, so that the solver
                # finishes quickly; the trajectory is cut there below
                bad_t.append(t)
                return np.zeros_like(x)
            return dx

        sol = solve_ivp(rhs, t_span, x0, method="DOP853", events=[lam_zero, nonpert],
                        rtol=num.rtol, atol=num.atol, dense_output=True)
        npts = max(50, n)

        def grid(t_last: float) -> np.ndarray:
            return np.linspace(t_span[0], t_last, npts) if t_last > t_span[0] else np.array([t_span[0]])

        if bad_t:
            t_bad = min(bad_t)
            good = sol.t[sol.t < t_bad]
            t_cut = float(good[-1]) if good.size else t_span[0]
            for k, te in enumerate(sol.t_events):
                if te.size and te[0] < t_cut:      # a genuine event happened before the failure
                    tt = np.linspace(t_span[0], te[0], npts)
                    return Segment(tt, sol.sol(tt), ("unstable", "nonperturbative")[k], float(te[0]))
            tt = grid(t_cut)
            return Segment(tt, sol.sol(tt), "failed", t_cut,
                           f"non-finite beta functions at mu = {np.exp(t_bad):.3g} GeV")
        t_last = float(sol.t[-1])
        tt = grid(t_last)
        xx = sol.sol(tt) if sol.sol is not None else sol.y[:, -1:].repeat(len(tt), axis=1)
        if sol.t_events[0].size:
            return Segment(tt, xx, "unstable", float(sol.t_events[0][0]))
        if sol.t_events[1].size:
            return Segment(tt, xx, "nonperturbative", float(sol.t_events[1][0]))
        if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
            # solver gave up (step size underflow, NaN/inf in the betas, ...) before reaching
            # the end: report it instead of mistaking the stopping point for stability
            return Segment(tt, xx, "failed", t_last, sol.message or "non-finite couplings")
        return Segment(tt, xx, "running", t_last)


class _Trajectory:
    """Collects the segments of one run and builds the RunResult."""

    def __init__(self, names: list[str], M: float, params: dict[str, float], span: float, n_points: int):
        self.names, self.M, self.params = names, M, dict(params)
        self.span, self.n_points = span, n_points
        self.ts: list[np.ndarray] = []
        self.xs: list[np.ndarray] = []
        self.matching: Matching | None = None

    def points(self, a: float, b: float) -> int:
        """Grid points for [a, b], proportional to its share of the whole range."""
        return int(self.n_points * (b - a) / self.span)

    def add(self, seg: Segment, seg_names: list[str]) -> None:
        x = np.full((len(self.names), len(seg.t)), np.nan)   # couplings not in the segment: NaN
        for k, c in enumerate(seg_names):
            x[self.names.index(c)] = seg.x[k]
        self.ts.append(seg.t)
        self.xs.append(x)

    def add_point(self, t: float, x: np.ndarray) -> None:
        self.ts.append(np.array([t]))
        self.xs.append(x[:, None])

    def result(self, status: str, t_stop: float, message: str = "") -> RunResult:
        mu_stop = float(np.exp(t_stop))
        return RunResult(self.names, np.concatenate(self.ts), np.concatenate(self.xs, axis=1), status,
                         mu_stop if status == "unstable" else None, mu_stop, self.M, self.params,
                         message=message, matching=self.matching)
