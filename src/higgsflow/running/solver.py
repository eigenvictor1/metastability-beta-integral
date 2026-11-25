"""RG running with the new-physics threshold M (plus optional fixed-mass and singlet
thresholds) and the search for μ_I.

A run has three steps:

1. SM running from M_t to M (new fermions decoupled), from the boundary values
   in ``boundary.py``;
2. matching at μ = M: the new Yukawas take the values given by the user (defined
   at M); the SM couplings are continuous (tree level) or get the one-loop
   threshold correction of ``threshold.py``;
3. running of the full theory from M to μ_max (default M_Planck).

Fields with a fixed mass (``X5:(1,5,0)@9.6e3``, no Yukawas) add further thresholds at their
mass, wherever it lies relative to M: below it they are decoupled, above it they enter the
gauge running (continuous matching, as for fermions at μ = their mass in MS-bar).

With a scalar singlet S (theory/model.py: Singlet) the new fermions get their mass from
⟨S⟩: M = y_S v_S/√2. S is integrated out at m_S with the tree-level matching
λ_SM = λ − κ²/(4λ_S); above m_S, λ_S, κ (given at m_S) and y_S (above M too) run.

The running continues past λ = 0 and stops at the first of

* perturbativity  -> a gauge coupling, y_t or λ reaches its limit (theory/couplings.py), or the
                     overall strength y_eff of the new Yukawas reaches √(4π) (running/strength.py;
                     without a model: each new Yukawa), or the loop expansion of the gauge
                     sector in g₂ breaks down: in β_λ, the two-loop g₂⁶ term outgrows the
                     pure-gauge one-loop terms (g⁴, always positive). With many SU(2) charged
                     fermions the g₂⁶ term is large and negative, and beyond that point it would
                     drive λ down artificially (two loops beating one loop)
* μ = μ_max

The status of the run is

* 'unstable'         λ crossed zero before the running stopped; μ_I = the first crossing
* 'nonperturbative'  the perturbativity limit was reached with λ > 0 all the way
                     ('g2_nonpert' instead if what was reached is the limit of g_2 or the loop
                     breakdown driven by g_2: no instability while g_2 is perturbative)
* 'stable'           μ_max was reached with λ > 0 all the way

Minimum of λ: a point where β_λ changes sign from negative to positive inside the range of
the running (also at μ = M, where β_λ jumps at the threshold). The end of the running is never
taken as a minimum: if λ is still decreasing there, the run has no minimum (λ_min = None).
With several minima the deepest is kept.

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
from ..theory.model import Model, Singlet
from .boundary import SMInputs
from .strength import YukawaStrength
from .threshold import Matching, ThresholdMatching

SM_COUPLINGS = list(cp.SM_NAMES)
LOOP = 16 * np.pi ** 2
STATUSES = ("unstable", "stable", "g2_nonpert", "nonperturbative", "failed")
log = logging.getLogger(__name__)

BetaFunction = Callable[[float, np.ndarray], np.ndarray]


class IntegrationError(RuntimeError):
    """Raised inside the solver to stop a run that cannot be integrated."""


@dataclass
class RunResult:
    """The outcome of one run: the couplings along μ, why the running stopped, and μ_I."""
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
    lam_min: float | None = None  # λ at its (deepest) minimum inside the running range, if any
    mu_min: float | None = None   # where that minimum is (GeV)
    lam_low: float | None = None  # lowest λ along the running: λ_min, or λ where the running
                                  # stops if it is still decreasing there

    @property
    def mu(self) -> np.ndarray:
        """μ in GeV along the run."""
        return np.exp(self.t)

    def coupling(self, name: str) -> np.ndarray:
        """Values of one coupling along the run."""
        return self.x[self.couplings.index(name)]


@dataclass
class Segment:
    """Result of integrating one segment (SM or full theory)."""
    t: np.ndarray
    x: np.ndarray                 # rows in the order of the segment's couplings
    status: str                   # 'running' (reached the end) or a final status
    t_stop: float
    message: str = ""
    t_zero: float | None = None   # first point where λ crosses zero going down
    minima: list[tuple[float, float]] = field(default_factory=list)   # (t, λ) of λ's minima


@dataclass(frozen=True)
class RunnerSpec:
    """Picklable description of a Runner, used to rebuild it in worker processes."""
    full: dict
    sm: dict
    loops: int
    model: dict | None            # Model.to_dict() (perturbativity of the new Yukawas, matching)
    numerics: Numerics
    one_loop: bool = False        # one-loop threshold matching
    states: tuple = ()            # ((n_fixed, above_M, above_S), system json) of the other states
    masses: tuple = ()            # masses of the fixed-mass fields, lightest first
    singlet: dict | None = None   # Singlet.as_dict()


State = tuple[int, bool, bool]    # (fixed-mass fields active, above M, above m_S)


class Runner:
    """Holds compiled beta functions for the SM, the full model and the theories in between."""

    def __init__(self, full: BetaSystem, sm: BetaSystem, loops: int = 2,
                 threshold: ThresholdMatching | None = None, numerics: Numerics | None = None,
                 model: Model | None = None, states: dict[State, BetaSystem] | None = None,
                 masses: list[float] | None = None, singlet: Singlet | None = None):
        """threshold: one-loop matching at M, or None (tree level).
        model: the model of the fields at M, without fixed-mass fields or singlet (taken from
        threshold if not given), for the perturbativity limit on the overall strength of the
        new Yukawas; without it each is tested alone.
        masses: masses of the fixed-mass fields; singlet: the scalar singlet S, if any.
        states: beta functions of the theories between thresholds, keyed by (number of
        fixed-mass fields active, above M, above m_S); the full theory and the SM are
        full and sm. Every combination met in a run must be there."""
        self.full = full
        self.sm = sm.restrict(SM_COUPLINGS)
        self.loops = loops
        self.threshold = threshold
        self.numerics = numerics or Numerics()
        self.f_full = full.numeric(loops)
        self.f_sm = self.sm.numeric(loops)
        self.names = list(full.couplings)
        self.masses = sorted(float(m) for m in (masses or []))
        self.singlet = singlet
        self.full_state: State = (len(self.masses), True, singlet is not None)
        self.states = {k: v for k, v in (states or {}).items()
                       if k not in (self.full_state, (0, False, False))}
        self._f: dict[State, BetaFunction] = {}
        self._names: dict[State, list[str]] = {self.full_state: self.names, (0, False, False): SM_COUPLINGS}
        for k, bs in self.states.items():
            names = list(bs.couplings)
            self._f[k] = bs.numeric(loops)
            self._names[k] = names
        for k in self._needed():
            if k not in self._names:
                raise ValueError(f"no beta functions for the state {k} (fixed fields, above M, above m_S)")
        self.new = [c for c in self.names if c not in SM_COUPLINGS and c not in cp.SINGLET_NAMES]
        self.model = model or (threshold.model if threshold is not None else None)
        self._lim: dict[int, tuple[np.ndarray, YukawaStrength | None]] = {}
        # beta functions per state (by the identity of its coupling list), for the loop breakdown
        self._systems: dict[int, BetaSystem] = {id(self.names): full, id(SM_COUPLINGS): self.sm}
        for k, bs in self.states.items():
            self._systems[id(self._names[k])] = bs
        self._gauge: dict[int, list | None] = {}

    def _needed(self) -> list[State]:
        """Every state a run can go through."""
        s_opts = (False, True) if self.singlet is not None else (False,)
        return [(n, m, s) for n in range(len(self.masses) + 1) for m in (False, True) for s in s_opts]

    # -- worker processes -------------------------------------------------------
    def spec(self) -> RunnerSpec:
        """Picklable description of this runner (sent to worker processes)."""
        model = self.model.to_dict() if self.model is not None else None
        return RunnerSpec(self.full.to_json(), self.sm.to_json(), self.loops, model, self.numerics,
                          self.threshold is not None,
                          tuple((k, bs.to_json()) for k, bs in self.states.items()),
                          tuple(self.masses), self.singlet.as_dict() if self.singlet else None)

    @classmethod
    def from_spec(cls, spec: RunnerSpec) -> Runner:
        """Rebuild a Runner from spec() in another process (recompiles betas and matching)."""
        model = Model.from_dict(spec.model) if spec.model is not None else None
        threshold = ThresholdMatching(model) if spec.one_loop and model is not None else None
        states = {tuple(k): BetaSystem.from_json(js) for k, js in spec.states}
        singlet = Singlet.from_dict(spec.singlet) if spec.singlet else None
        return cls(BetaSystem.from_json(spec.full), BetaSystem.from_json(spec.sm), spec.loops,
                   threshold, spec.numerics, model, states, list(spec.masses), singlet)

    # -- public -----------------------------------------------------------------
    def run(self, params: dict[str, float], M: float, inputs: SMInputs | None = None,
            sm_only: bool = False) -> RunResult:
        """Run from M_t to μ_max; never raises for numerical trouble (status 'failed').

        params: values at μ = M of the new couplings (missing ones are 0).
        sm_only: leave out the fixed-mass fields and the singlet too (pure SM reference run,
        with M huge).
        """
        inputs = inputs or SMInputs()
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                return self._run(params, M, inputs, sm_only)
        except (IntegrationError, ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            log.debug("integration failed for %s, M=%g: %s", params, M, exc)
            t0 = np.log(inputs.Mt)
            return RunResult(self.names, np.array([t0]), np.full((len(self.names), 1), np.nan),
                             "failed", None, float(inputs.Mt), M, dict(params),
                             message=f"{type(exc).__name__}: {exc}")

    # -- the running through the thresholds ---------------------------------------------
    def _run(self, params: dict[str, float], M: float, inputs: SMInputs, sm_only: bool) -> RunResult:
        """SM from M_t up, matching at M, full theory up to μ_max; fixed-mass fields switch on
        at their masses on the way (continuous couplings), and the singlet at m_S (λ jumps
        up by κ²/4λ_S); β_λ may jump at each threshold: a kink minimum.

        Returns early if a segment ends the run (non-perturbative, solver failure, or M above
        μ_max); λ < 0 does not stop it (see the module docstring).
        """
        t0, t_end = np.log(inputs.Mt), np.log(self.numerics.mu_max)
        tM = max(np.log(M), t0)
        out = _Trajectory(self.names, M, params, span=t_end - t0, n_points=self.numerics.n_points)
        x0 = inputs.couplings_at_mt()
        x, names = np.array([x0[c] for c in SM_COUPLINGS]), SM_COUPLINGS
        new_vals = {c: float(params.get(c, 0.0)) for c in self.new}
        sing = None if sm_only else self.singlet
        # thresholds in order (0: fixed-mass field, 1: singlet, 2: M); fixed fields first at equal scales
        marks = [(max(np.log(m), t0), 0) for m in ([] if sm_only else self.masses)] + [(tM, 2)]
        if sing is not None:
            marks.append((max(np.log(sing.m_S), t0), 1))
        n_fixed, above_M, above_S, t = 0, False, False, t0
        for t_b, kind in sorted(marks) + [(t_end, 3)]:
            state = (n_fixed, above_M, above_S)
            f = self._betas(state)
            t_seg = min(t_b, t_end)
            if t_seg > t:
                seg = self._segment(f, (t, t_seg), x, names, out.points(t, t_seg))
                out.add(seg, names)
                if seg.status != "running":
                    return out.result(seg.status, seg.t_stop, seg.message)
                x, t = seg.x[:, -1], t_seg
            if t_b >= t_end:
                return out.result("stable", t_end)
            f_below, names_below, x_below = f, names, x
            vals = dict(zip(names, map(float, x)))
            if kind == 0:
                n_fixed += 1
            elif kind == 1:                                    # the singlet S enters
                assert sing is not None
                above_S = True
                vals[QUARTIC] += sing.shift
                vals.update({cp.LAMS.name: sing.lamS, cp.KAP.name: sing.kappa})
                if above_M:
                    vals[cp.YS.name] = sing.y_at(M)
            else:                                              # the fermions enter: matching at M
                above_M = True
                nv = dict(new_vals)
                if above_S and sing is not None:
                    nv[cp.YS.name] = sing.y_at(M)
                vals, out.matching = self._match(vals, nv, self._names[(n_fixed, True, above_S)])
            names = self._names[(n_fixed, above_M, above_S)]
            x = np.array([vals[c] for c in names])
            if kind and self._margin(x, names) <= 0:
                out.add_point(t, x)
                return out.result("nonperturbative", t)
            if t > t0:
                lam_below = float(x_below[names_below.index(QUARTIC)])
                out.kink_minimum(t, self._beta_lambda(f_below, names_below, t, x_below),
                                 self._beta_lambda(self._betas((n_fixed, above_M, above_S)), names, t, x),
                                 min(lam_below, float(x[names.index(QUARTIC)])))
        raise AssertionError("unreachable")

    def _betas(self, state: State) -> BetaFunction:
        """Beta functions of the theory in this state."""
        if state == self.full_state:
            return self.f_full
        if state == (0, False, False):
            return self.f_sm
        return self._f[state]

    @staticmethod
    def _beta_lambda(f: BetaFunction, names: list[str], t: float, x: np.ndarray) -> float:
        """β_λ at (t, x) with the beta functions f (NaN if they are not finite there)."""
        v = float(f(t, x)[names.index(QUARTIC)])
        return v if np.isfinite(v) else np.nan

    def _match(self, vals: dict[str, float], new_vals: dict[str, float],
               names: list[str]) -> tuple[dict[str, float], Matching | None]:
        """Couplings just above M (all values, keyed by name) and the correction applied."""
        values, matching = {**vals, **new_vals}, None
        x = np.array([values[c] for c in names])
        if self.threshold is not None and self._margin(x, names) > 0:
            values, matching = self.threshold.match(vals, new_vals)
        return values, matching

    def _limits(self, names: list[str]) -> tuple[np.ndarray, YukawaStrength | None]:
        """Per-coupling limits for a state (new Yukawas: tested together through y_eff when the
        model is known) and the y_eff function, cached per state."""
        key = id(names)
        if key not in self._lim:
            lim = cp.limits(names)
            strength = None
            idx = [k for k, c in enumerate(names) if c in self.new]
            if idx and self.model is not None:
                strength = YukawaStrength(self.model, names)
                lim[idx] = np.inf
            self._lim[key] = (lim, strength)
        return self._lim[key]

    def _margin(self, x: np.ndarray, names: list[str]) -> float:
        """Distance to the perturbativity limit (< 0 beyond it): per coupling, and for the new
        Yukawas through y_eff."""
        lim, strength = self._limits(names)
        m = float(np.min(lim - np.abs(x)))
        if strength is not None:
            m = min(m, cp.PERT_GAUGE_YUKAWA - strength(x))
        loop = self._loop_margin(x, names)
        return m if loop is None else min(m, loop[0])

    def _gauge_parts(self, names: list[str]):
        """Pure-gauge monomials of β_λ at one and two loops for a state, as (coefficients,
        exponents of (g1, g2, g3), g₂-only flags) per loop; None below two loops."""
        key = id(names)
        if key not in self._gauge:
            bs = self._systems.get(key)
            parts = None
            if bs is not None and self.loops >= 2:
                gi = [names.index(g) for g in cp.GAUGE]
                others = [k for k in range(len(names)) if k not in gi]
                parts = []
                for L in (1, 2):
                    terms = [(c, [m[k] for k in gi]) for m, c in bs.coeffs[QUARTIC].get(L, {}).items()
                             if not any(m[k] for k in others)]
                    coef = np.array([c for c, _ in terms])
                    exps = np.array([e for _, e in terms], dtype=float).reshape(-1, 3)
                    only_g2 = (exps[:, 0] == 0) & (exps[:, 2] == 0)
                    parts.append((coef, exps, only_g2, gi))
            self._gauge[key] = parts
        return self._gauge[key]

    def _loop_margin(self, x: np.ndarray, names: list[str]) -> tuple[float, bool] | None:
        """1 − |g₂⁶ term of β_λ (two loops)| / (one-loop pure-gauge part of β_λ, > 0): < 0 once
        the loop expansion in g₂ has broken down. (Only g₂ is tested: a large g₁ can give the
        same breakdown, but it is left to the plain limit on |g₁|.)"""
        parts = self._gauge_parts(names)
        if parts is None:
            return None
        (c1, e1, _, gi), (c2, e2, g2only, _) = parts
        g = np.abs(x[gi])
        one = float(np.sum(c1 * np.prod(g ** e1, axis=1))) / LOOP
        two = float(np.sum((c2 * np.prod(g ** e2, axis=1))[g2only])) / LOOP ** 2
        if one <= 0:
            return None
        return 1 - abs(two) / one, True

    def _perturbative(self, x: np.ndarray) -> bool:
        return self._margin(x, self.names) > 0

    def _nonpert_status(self, x: np.ndarray, names: list[str]) -> str:
        """Status when the perturbativity limit is reached at x: 'g2_nonpert' if the limit
        reached is that of g_2 (λ is then still positive), 'nonperturbative' otherwise."""
        if cp.G2.name not in names:
            return "nonperturbative"
        lim, strength = self._limits(names)
        margins = lim - np.abs(x)
        positive = x[names.index(QUARTIC)] > 0
        loop = self._loop_margin(x, names)
        if loop is not None and loop[0] < min(float(np.min(margins)),
                                              cp.PERT_GAUGE_YUKAWA - strength(x) if strength else np.inf):
            return "g2_nonpert" if loop[1] and positive else "nonperturbative"   # loop breakdown
        if strength is not None and cp.PERT_GAUGE_YUKAWA - strength(x) < float(np.min(margins)):
            return "nonperturbative"                         # the new Yukawas got there first
        k = int(np.argmin(margins))
        return "g2_nonpert" if names[k] == cp.G2.name and x[names.index(QUARTIC)] > 0 else "nonperturbative"

    # -- integration of one segment -------------------------------------------------
    def _segment(self, f: BetaFunction, t_span: tuple[float, float], x0: np.ndarray,
                 names: list[str], n: int) -> Segment:
        """Integrate the couplings ``names`` (state order) over t_span, with events."""
        idx_lam = names.index(QUARTIC)
        num = self.numerics
        t_start = t_span[0]

        def lam_zero(t, x):
            """Event: λ crosses zero going down (not terminal) -> μ_I."""
            return x[idx_lam]
        lam_zero.terminal, lam_zero.direction = False, -1    # type: ignore[attr-defined]

        def nonpert(t, x):
            """Event: the perturbativity limit is reached (terminal) -> 'nonperturbative'."""
            return self._margin(x, names)
        nonpert.terminal, nonpert.direction = True, -1       # type: ignore[attr-defined]

        def lam_min(t, x):
            """Event: β_λ crosses zero going up (not terminal) -> a minimum of λ."""
            v = f(t, x)[idx_lam]
            return float(v) if np.isfinite(v) else -1.0
        lam_min.terminal, lam_min.direction = False, 1       # type: ignore[attr-defined]

        calls, bad_t = [0], []

        def rhs(t, x):
            """Beta functions with guards: stop after max_rhs_calls; on NaN/inf note where and freeze."""
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

        sol = solve_ivp(rhs, t_span, x0, method="DOP853", events=[lam_zero, nonpert, lam_min],
                        rtol=num.rtol, atol=num.atol, dense_output=True)
        npts = max(50, n)

        def grid(t_last: float) -> np.ndarray:
            """Output grid from the segment start to t_last (a single point if nothing was integrated)."""
            return np.linspace(t_start, t_last, npts) if t_last > t_start else np.array([t_start])

        def found(t_cut: float) -> dict:
            """λ zero crossing and minima strictly inside (t_start, t_cut)."""
            zero = t_start if x0[idx_lam] <= 0 else None
            te = sol.t_events[0]
            if zero is None and te.size and te[0] <= t_cut:
                zero = float(te[0])
            minima = [(float(t), float(y[idx_lam])) for t, y in zip(sol.t_events[2], sol.y_events[2])
                      if t_start < t < t_cut]
            return {"t_zero": zero, "minima": minima}

        if bad_t:
            t_bad = min(bad_t)
            good = sol.t[sol.t < t_bad]
            t_cut = float(good[-1]) if good.size else t_start
            te = sol.t_events[1]
            if te.size and te[0] < t_cut:          # the limit was reached before the failure
                tt = grid(float(te[0]))
                return Segment(tt, sol.sol(tt), self._nonpert_status(sol.y_events[1][0], names),
                               float(te[0]), **found(float(te[0])))
            tt = grid(t_cut)
            return Segment(tt, sol.sol(tt) if sol.sol is not None else np.repeat(x0[:, None], len(tt), 1),
                           "failed", t_cut, f"non-finite beta functions at mu = {np.exp(t_bad):.3g} GeV",
                           **found(t_cut))
        t_last = float(sol.t[-1])
        tt = grid(t_last)
        xx = sol.sol(tt) if sol.sol is not None else sol.y[:, -1:].repeat(len(tt), axis=1)
        ev = found(t_last)
        if sol.t_events[1].size:
            return Segment(tt, xx, self._nonpert_status(sol.y_events[1][0], names),
                           float(sol.t_events[1][0]), **ev)
        if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
            # solver gave up (step size underflow, NaN/inf in the betas, ...) before reaching
            # the end: report it instead of mistaking the stopping point for stability
            return Segment(tt, xx, "failed", t_last, sol.message or "non-finite couplings", **ev)
        return Segment(tt, xx, "running", t_last, **ev)


class _Trajectory:
    """Collects the segments of one run and builds the RunResult."""

    def __init__(self, names: list[str], M: float, params: dict[str, float], span: float, n_points: int):
        """Empty trajectory of a run with couplings names; span and n_points set the grid density."""
        self.names, self.M, self.params = names, M, dict(params)
        self.span, self.n_points = span, n_points
        self.ts: list[np.ndarray] = []
        self.xs: list[np.ndarray] = []
        self.matching: Matching | None = None
        self.t_zero: float | None = None
        self.minima: list[tuple[float, float]] = []

    def points(self, a: float, b: float) -> int:
        """Grid points for [a, b], proportional to its share of the whole range."""
        return int(self.n_points * (b - a) / self.span)

    def add(self, seg: Segment, seg_names: list[str]) -> None:
        """Append a segment, placing its couplings in the full coupling order (others NaN)."""
        x = np.full((len(self.names), len(seg.t)), np.nan)   # couplings not in the segment: NaN
        for k, c in enumerate(seg_names):
            x[self.names.index(c)] = seg.x[k]
        self.ts.append(seg.t)
        self.xs.append(x)
        if self.t_zero is None:
            self.t_zero = seg.t_zero
        self.minima += seg.minima

    def kink_minimum(self, t: float, beta_below: float, beta_above: float, lam: float) -> None:
        """Record a minimum of λ at the threshold if β_λ jumps there from negative to positive."""
        if beta_below < 0 < beta_above:
            self.minima.append((t, lam))

    def add_point(self, t: float, x: np.ndarray) -> None:
        """Append a single point (used when the run stops right at M)."""
        self.ts.append(np.array([t]))
        self.xs.append(x[:, None])

    def result(self, status: str, t_stop: float, message: str = "") -> RunResult:
        """Join the segments into a RunResult.

        status is how the running ended; it becomes 'unstable' (with μ_I) if λ crossed zero
        on the way. The deepest minimum of λ found inside the running range is kept.
        """
        mu_stop = float(np.exp(t_stop))
        mu_I = float(np.exp(self.t_zero)) if self.t_zero is not None else None
        if mu_I is not None:
            status = "unstable"
        lam_min = mu_min = None
        inside = [(t, lam) for t, lam in self.minima if t < t_stop]
        if inside:
            t_m, lam_min = min(inside, key=lambda m: m[1])
            mu_min = float(np.exp(t_m))
        x = np.concatenate(self.xs, axis=1)
        lam = x[self.names.index(QUARTIC)]
        lam = lam[np.isfinite(lam)]
        return RunResult(self.names, np.concatenate(self.ts), x, status,
                         mu_I, mu_stop, self.M, self.params, message=message, matching=self.matching,
                         lam_min=lam_min, mu_min=mu_min, lam_low=float(lam.min()) if lam.size else None)
