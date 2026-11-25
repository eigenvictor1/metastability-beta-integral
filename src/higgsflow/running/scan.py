"""Parameter scans of the instability scale.

Parameters are the new-physics threshold ``M`` (GeV) and the new Yukawa
couplings (values at μ = M). ``links`` lets one scan parameter drive several
couplings at once, e.g. links={"y": ["yL4N", "yL4cNc"]}.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from .boundary import SMInputs
from .solver import Runner, RunnerSpec

log = logging.getLogger(__name__)

NO_NEW_PHYSICS = 1e30            # threshold mass used for the pure-SM reference run
Task = tuple[tuple[int, ...], dict[str, float], float, SMInputs | None]


@dataclass
class ScanResult:
    """Result of a 1D or 2D scan; arrays have one axis per scanned parameter."""
    axes: dict[str, np.ndarray]          # 1 or 2 scanned parameters
    fixed: dict[str, float]
    mu_I: np.ndarray                     # NaN where not unstable
    status: np.ndarray                   # str array
    mu_stop: np.ndarray
    sm_mu_I: float | None = None
    # Region boundaries located by bisection: edges[idx] is where the status changes between
    # idx and the next point along edge_axis (NaN where it doesn't change, or not refined)
    edges: np.ndarray | None = None
    edge_axis: str | None = None
    # minimum of λ inside the running range (NaN where λ has none): its value and scale (GeV)
    lam_min: np.ndarray | None = None
    mu_min: np.ndarray | None = None
    # where λ starts or stops having a minimum, located by bisection along edge_axis like edges
    min_edges: np.ndarray | None = None
    sm_lam_min: float | None = None      # λ at the minimum of the SM running
    sm_mu_min: float | None = None       # and its scale (GeV)
    # 2D scans: the same boundaries located along the other axis (M), between idx and the next
    # point along it; they resolve boundaries that run nearly parallel to edge_axis
    cross_edges: np.ndarray | None = None
    min_cross_edges: np.ndarray | None = None
    # extra boundary points between grid lines, where a boundary bends (adaptive refinement):
    # (axis along which it was bisected, value of the other axis, status below, above, position)
    refined: list[tuple[str, float, str, str, float]] | None = None
    # lowest λ along the running (λ_min, or λ where the running stops if still decreasing)
    lam_low: np.ndarray | None = None

    POINT_ARRAYS = ("mu_I", "status", "mu_stop", "edges", "lam_min", "mu_min", "min_edges",
                    "cross_edges", "min_cross_edges", "lam_low")

    @property
    def cross_axis(self) -> str | None:
        """The scanned axis other than edge_axis (2D scans), along which cross_edges lie."""
        others = [n for n in self.axes if n != self.edge_axis]
        return others[0] if len(self.axes) == 2 and self.edge_axis in self.axes else None

    def minimum_labels(self) -> np.ndarray:
        """Per point: 'min' if λ has a minimum, 'nomin' if not, or the status if the point is
        non-perturbative, failed or unscanned (no minimum shown there)."""
        lam = self.lam_min if self.lam_min is not None else np.full(self.status.shape, np.nan)
        lab = np.where(np.isfinite(lam), "min", "nomin").astype(object)
        return np.where(np.isin(self.status, MINIMUM_SHOWN), lab, self.status).astype(object)

    def map_points(self, fn: Callable[[np.ndarray], np.ndarray], **changes) -> ScanResult:
        """Copy with fn applied to every per-point array (transpose, slice, ...), then the
        given fields replaced (e.g. the new axes)."""
        new = {k: fn(getattr(self, k)) for k in self.POINT_ARRAYS if getattr(self, k) is not None}
        return replace(self, **{**new, **changes})

    def boundaries(self) -> Iterator[tuple[dict[str, float], float, str, str]]:
        """Refined region boundaries: ({other parameter: value}, boundary value, status below, above)."""
        if self.edges is None or self.edge_axis is None:
            return
        names = list(self.axes)
        a = names.index(self.edge_axis)
        for idx in product(*[range(len(self.axes[n])) for n in names]):
            e = self.edges[idx]
            if np.isfinite(e):
                nxt = idx[:a] + (idx[a] + 1,) + idx[a + 1:]
                other = {n: float(self.axes[n][i]) for n, i in zip(names, idx) if n != self.edge_axis}
                yield other, float(e), str(self.status[idx]), str(self.status[nxt])

    def rows(self) -> Iterator[tuple[dict[str, float], float, str, float, float, float, float]]:
        """Iterate over grid points: ({param: value}, μ_I, status, μ_stop, λ_min, μ_min, lowest λ)."""
        names = list(self.axes)
        grids = [self.axes[n] for n in names]
        nan = np.full(self.status.shape, np.nan)
        lam_min = self.lam_min if self.lam_min is not None else nan
        mu_min = self.mu_min if self.mu_min is not None else nan
        lam_low = self.lam_low if self.lam_low is not None else nan
        for idx in product(*[range(len(g)) for g in grids]):
            vals = [g[i] for g, i in zip(grids, idx)]
            yield (dict(zip(names, vals)), self.mu_I[idx], self.status[idx], self.mu_stop[idx],
                   lam_min[idx], mu_min[idx], lam_low[idx])


Links = dict[str, dict[str, float]]   # {alias: {coupling: factor}}: the alias drives couplings


def expand_links(links: dict | None, yukawas: list[str]) -> Links:
    """Normalise links: "all" -> every new Yukawa, a list -> factor 1 each, a dict -> the given
    factors (coupling = factor x alias); check the coupling names."""
    out: Links = {}
    for alias, targets in (links or {}).items():
        if targets == "all":
            weights = {c: 1.0 for c in yukawas}
        elif isinstance(targets, dict):
            weights = {c: float(f) for c, f in targets.items()}
        else:
            weights = {c: 1.0 for c in targets}
        unknown = [t for t in weights if t not in yukawas]
        if unknown:
            raise ValueError(f"link {alias!r}: unknown couplings {unknown}; model has {yukawas}")
        out[alias] = weights
    return out


def resolve_params(point: dict[str, float], links: dict | None,
                   default_M: float) -> tuple[dict[str, float], float]:
    """Map scan/user parameters to (coupling values, M). A linked alias sets each of its
    couplings to factor x value (factor 1 for links given as a list)."""
    links = links or {}
    vals, M = {}, default_M
    for k, v in point.items():
        if k == "M":
            M = float(v)
        elif k in links:
            t = links[k]
            for c, f in (t.items() if isinstance(t, dict) else ((c, 1.0) for c in t)):
                vals[c] = f * float(v)
        else:
            vals[k] = float(v)
    return vals, M


_WORKER_RUNNER: Runner | None = None


def _init_worker(spec: RunnerSpec) -> None:
    """Runs once in each worker process: rebuild the Runner from its picklable spec."""
    global _WORKER_RUNNER
    _WORKER_RUNNER = Runner.from_spec(spec)


REFINABLE = ("unstable", "nonperturbative", "stable", "g2_nonpert")   # region boundaries located by bisection
MINIMUM_SHOWN = ("unstable", "stable", "g2_nonpert")    # statuses where the minimum of λ is shown
EdgeTask = tuple[tuple[int, ...], dict[str, float], str, float, float, bool, str, str,
                 dict | None, float, SMInputs | None, int, str]


def _label(r, what: str) -> str:
    """What a bisection follows: the status of a run ('status'), or whether λ has a minimum
    ('minimum': 'min' / 'nomin')."""
    if what == "status":
        return r.status
    return "min" if r.lam_min is not None else "nomin"


def _bisect(runner: Runner, task: EdgeTask) -> tuple[tuple[int, ...], float]:
    """Locate where the status (or the presence of a minimum of λ) changes from s_lo to s_hi
    between lo and hi along one parameter.

    Each step runs the midpoint (geometric for M) and keeps the half that still contains the
    change; steps halvings give a precision of (hi - lo) / 2^steps. Stops early if the midpoint
    has a third status (e.g. 'failed').
    """
    idx, base, name, lo, hi, log_axis, s_lo, s_hi, links, M_default, inputs, steps, what = task
    for _ in range(steps):
        mid = float(np.sqrt(lo * hi)) if log_axis else 0.5 * (lo + hi)
        vals, M = resolve_params({**base, name: mid}, links, M_default)
        st = _label(runner.run(vals, M, inputs), what)
        if st == s_lo:
            lo = mid
        elif st == s_hi:
            hi = mid
        else:
            break
    return idx, (float(np.sqrt(lo * hi)) if log_axis else 0.5 * (lo + hi))


def _bisect_worker(task: EdgeTask) -> tuple[tuple[int, ...], float]:
    """_bisect in a worker process."""
    assert _WORKER_RUNNER is not None
    return _bisect(_WORKER_RUNNER, task)


def _edge_tasks(names: list[str], grids: list[np.ndarray], status: np.ndarray, fixed: dict[str, float],
                links: dict | None, M_default: float, inputs: SMInputs | None,
                steps: int, what: str = "status",
                refinable: tuple[str, ...] = REFINABLE, along: str | None = None) -> tuple[str, list[EdgeTask]]:
    """One bisection task per pair of neighbouring points (along the Yukawa axis, or the axis
    ``along``) whose labels (status, or 'min'/'nomin' with what='minimum') differ, both in
    refinable."""
    edge_axis = along or next((n for n in reversed(names) if n != "M"), names[-1])
    a = names.index(edge_axis)
    tasks: list[EdgeTask] = []
    for idx in product(*[range(len(g)) for g in grids]):
        if idx[a] + 1 >= len(grids[a]):
            continue
        nxt = idx[:a] + (idx[a] + 1,) + idx[a + 1:]
        s1, s2 = status[idx], status[nxt]
        if s1 == s2 or s1 not in refinable or s2 not in refinable:
            continue
        base = dict(fixed)
        base.update({n: float(g[i]) for n, g, i in zip(names, grids, idx) if n != edge_axis})
        tasks.append((idx, base, edge_axis, float(grids[a][idx[a]]), float(grids[a][idx[a] + 1]),
                      edge_axis == "M", str(s1), str(s2), links, M_default, inputs, steps, what))
    return edge_axis, tasks


def _summary(r) -> tuple[str, float, float | None, float | None, float | None, float | None]:
    """(status, μ_stop, μ_I, λ_min, μ_min, lowest λ) of a RunResult: what a scan keeps of each point."""
    return r.status, r.mu_stop, r.mu_I, r.lam_min, r.mu_min, r.lam_low


BEND_DEPTH = 3          # adaptive refinement: at most 2^3 - 1 extra lines between two grid lines
BEND_TOL = 0.15         # refine while the boundary deviates from a straight line by more than this
                        # fraction of a grid cell (along the bisection axis)


def _refine_bends(runner: Runner, pool: ProcessPoolExecutor | None, nproc: int, progress: _Progress,
                  names: list[str], grids: list[np.ndarray], along: str, labelled: list,
                  fixed: dict[str, float], links: dict | None, M_default: float,
                  inputs: SMInputs | None, steps: int) -> list[tuple[str, float, str, str, float]]:
    """Adaptive refinement of the region boundaries between neighbouring grid lines.

    For every boundary (a change of label along ``along``, e.g. stable -> unstable along the
    Yukawa axis) found on two neighbouring lines of the other axis (M), the boundary is also
    bisected on the line half-way between them (geometric mean for M). If that point lies off
    the straight line between the two neighbours by more than BEND_TOL of a grid cell, i.e. the
    boundary bends there, both halves are refined again, down to BEND_DEPTH levels. Straight
    stretches cost one extra bisection per grid cell; bends get more.

    labelled: (labels on the grid, their bisected edges along ``along``, what) per kind of
    boundary ('status' or 'minimum'). Returns the extra points (along, other value, label
    below, label above, position along ``along``).
    """
    a = names.index(along)
    other = names[1 - a]
    ya, yo = grids[a], grids[1 - a]
    log_a, log_o = along == "M", other == "M"
    ta = np.log10(ya) if (log_a or np.all(ya > 0)) else ya       # where "straight" is judged
    cell = float(np.median(np.abs(np.diff(ta)))) if len(ta) > 1 else 1.0

    def t(v: float) -> float:
        return float(np.log10(v)) if (log_a or np.all(ya > 0)) else float(v)

    def mid(u: float, w: float, log: bool) -> float:
        return float(np.sqrt(u * w)) if log else 0.5 * (u + w)

    def transitions(lab: np.ndarray, ed: np.ndarray, j: int) -> list[tuple[str, str, int, float]]:
        """(label below, above, index below, bisected position or NaN) along line j of the other axis."""
        line = lab[(slice(None), j)] if a == 0 else lab[j]
        eline = ed[(slice(None), j)] if a == 0 else ed[j]
        return [(str(line[k]), str(line[k + 1]), k, float(eline[k]))
                for k in range(len(line) - 1) if line[k] != line[k + 1]]

    # intervals to refine: (other lo, other hi, pos lo, pos hi, label below, above, bracket lo, hi, what)
    work = []
    for lab, ed, what in labelled:
        for j in range(len(yo) - 1):
            t1, t2 = transitions(lab, ed, j), transitions(lab, ed, j + 1)
            if [(x[0], x[1]) for x in t1] != [(x[0], x[1]) for x in t2]:
                continue
            for (s_lo, s_hi, k1, e1), (_, _, k2, e2) in zip(t1, t2):
                allowed = REFINABLE if what == "status" else ("min", "nomin")
                if s_lo not in allowed or s_hi not in allowed:
                    continue
                e1 = e1 if np.isfinite(e1) else mid(ya[k1], ya[k1 + 1], log_a)
                e2 = e2 if np.isfinite(e2) else mid(ya[k2], ya[k2 + 1], log_a)
                work.append((float(yo[j]), float(yo[j + 1]), e1, e2, s_lo, s_hi,
                             min(k1, k2), max(k1, k2) + 1, what))
    found: list[tuple[str, float, str, str, float]] = []
    for _ in range(BEND_DEPTH):
        if not work:
            break
        tasks = []
        for n, (o1, o2, _, _, s_lo, s_hi, k1, k2, what) in enumerate(work):
            base = dict(fixed)
            base[other] = mid(o1, o2, log_o)
            tasks.append((n, ((n,), base, along, float(ya[k1]), float(ya[k2]), log_a, s_lo, s_hi,
                              links, M_default, inputs, steps, what)))
        progress.set_total(progress.total + steps * len(tasks))
        it = (_unordered(pool, "edge", tasks, max(1, min(20, len(tasks) // (8 * nproc))))
              if pool is not None else ((n, *_bisect(runner, tk)) for n, tk in tasks))
        nxt = []
        for n, _, em in it:
            progress.step(steps)
            o1, o2, e1, e2, s_lo, s_hi, k1, k2, what = work[n]
            om = mid(o1, o2, log_o)
            found.append((along, om, s_lo, s_hi, em))
            if abs(t(em) - 0.5 * (t(e1) + t(e2))) > BEND_TOL * cell:      # bends: refine both halves
                nxt += [(o1, om, e1, em, s_lo, s_hi, k1, k2, what), (om, o2, em, e2, s_lo, s_hi, k1, k2, what)]
        work = nxt
    return found


def _chunk_worker(kind: str, items: list) -> list:
    """Run a chunk of grid points (kind 'point') or of (job, bisection task) pairs ('edge')."""
    if kind == "point":
        return [_run_point(t) for t in items]
    return [(j, *_bisect_worker(t)) for j, t in items]


def _unordered(pool: ProcessPoolExecutor, kind: str, items: list, chunk: int) -> Iterator:
    """Results of _chunk_worker over items, yielded as the chunks finish (not in order), so that
    the progress bar advances steadily instead of waiting for the slowest early chunk."""
    futs = [pool.submit(_chunk_worker, kind, items[i:i + chunk]) for i in range(0, len(items), chunk)]
    for f in as_completed(futs):
        yield from f.result()


def _run_point(args: Task) -> tuple:
    """Run one grid point in a worker; returns (grid index, status, μ_stop, μ_I, λ_min, μ_min,
    lowest λ)."""
    idx, vals, M, inputs = args
    assert _WORKER_RUNNER is not None
    return (idx, *_summary(_WORKER_RUNNER.run(vals, M, inputs)))


class _Progress:
    """A progress bar that slides along one terminal line: label, bar, percentage, time left.

    Drawn only on an interactive terminal and in the main process (when models run in parallel
    processes their bars would mix); otherwise silent. The line is redrawn in place ("\\r") at
    most every 0.1 s, always shorter than the terminal is wide (a longer line would wrap, and
    each redraw would then start a new line), and ends with a newline when the work is done.
    """
    WIDTH = 30                     # bar width (characters) on a wide terminal

    def __init__(self, total: int, label: str):
        self.total, self.label, self.done = max(total, 1), label, 0
        self.t0 = self.last = time.time()
        self.show = total > 0 and sys.stderr.isatty() and multiprocessing.parent_process() is None
        self._draw()

    def step(self, weight: int = 1) -> None:
        """Advance by weight units of work; finishes the line when all the work is done."""
        self.done += weight
        now = time.time()
        if now - self.last >= 0.1 or self.done >= self.total:
            self.last = now
            self._draw()
            if self.done >= self.total:
                self.close()

    def set_total(self, total: int) -> None:
        """Replace the (estimated) amount of work by the actual one."""
        self.total = max(total, self.done, 1)
        if self.done >= self.total:
            self._draw()
            self.close()

    def close(self) -> None:
        if self.show:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.show = False

    def _draw(self) -> None:
        if not self.show:
            return
        frac = self.done / self.total
        elapsed = time.time() - self.t0
        if frac >= 1:
            left = f"{elapsed:5.0f} s     "
        elif frac < 0.02 or elapsed < 2:                # too early for a sensible estimate
            left = "    ? s left"
        else:
            left = f"{elapsed * (1 - frac) / frac:5.0f} s left"
        tail = f" {100 * frac:3.0f}% {left}"
        cols = shutil.get_terminal_size((80, 20)).columns - 1       # never touch the last column
        label = self.label[:max(10, cols // 3)]
        width = max(5, min(self.WIDTH, cols - len(label) - len(tail) - 3))
        n = int(frac * width)
        bar = "=" * n + (">" if n < width else "") + " " * (width - n - 1)
        line = f"{label} [{bar}]{tail}"[:cols]
        sys.stderr.write("\r" + line.ljust(cols))
        sys.stderr.flush()


def run_scan(runner: Runner, scan: dict[str, np.ndarray], fixed: dict[str, float] | None = None,
             links: dict | None = None, inputs: SMInputs | None = None,
             M_default: float = 1e3,
             workers: int | None = 1, label: str = "Scanning") -> ScanResult:
    """Run the grid. ``workers`` > 1 (or None = all cores) runs points in parallel processes.
    label: text of the progress bar of the grid (the boundary search has no bar)."""
    if not 1 <= len(scan) <= 2:
        raise ValueError("Scan over one or two parameters (fix the rest with `fixed`).")
    fixed = dict(fixed or {})
    names = list(scan)
    grids = [np.asarray(scan[n], dtype=float) for n in names]
    shape = tuple(len(g) for g in grids)
    mu_I = np.full(shape, np.nan)
    mu_stop = np.full(shape, np.nan)
    lam_min = np.full(shape, np.nan)
    mu_min = np.full(shape, np.nan)
    lam_low = np.full(shape, np.nan)
    status = np.empty(shape, dtype=object)

    tasks: list[Task] = []
    for idx in product(*[range(n) for n in shape]):
        point = dict(fixed)
        point.update({n: g[i] for n, g, i in zip(names, grids, idx)})
        vals, M = resolve_params(point, links, M_default)
        tasks.append((idx, vals, M, inputs))

    nproc: int = (os.cpu_count() or 1) if workers is None else workers
    parallel = nproc > 1 and len(tasks) > 1
    pool = (ProcessPoolExecutor(max_workers=nproc, initializer=_init_worker, initargs=(runner.spec(),))
            if parallel else None)
    # The progress bar counts runs: one per grid point, plus the bisections of the boundary
    # search (boundary_steps runs each), first estimated as one boundary per grid line and
    # direction (what the seesaw scans need), then replaced by the actual number. Points are
    # run in random order, so that the cost per point (cheap where the running stops early,
    # dearer up to M_Pl) is the same throughout and the time left is estimated well.
    steps = runner.numerics.boundary_steps
    lines = sum(int(np.prod(shape)) // n for n in shape) if len(shape) == 2 else 1
    progress = _Progress(len(tasks) + steps * lines, label)
    order = np.random.default_rng(0).permutation(len(tasks))
    tasks = [tasks[k] for k in order]
    try:
        # 1. the grid
        if pool is not None:
            chunk = max(1, min(50, len(tasks) // (8 * nproc)))     # small chunks: steady progress
            it = _unordered(pool, "point", tasks, chunk)
        else:
            it = ((idx, *_summary(runner.run(vals, M, inp))) for idx, vals, M, inp in tasks)
        for idx, st, ms, mi, lm, mm, ll in it:
            status[idx] = st
            mu_stop[idx] = ms
            if mi is not None:
                mu_I[idx] = mi
            if lm is not None:
                lam_min[idx], mu_min[idx] = lm, mm
            if ll is not None:
                lam_low[idx] = ll
            progress.step()

        # 2. region boundaries (unstable / non-perturbative / stable), and where λ starts having
        #    a minimum, by bisection: along the Yukawa axis and, in 2D, along the other axis too
        edges, min_edges = np.full(shape, np.nan), np.full(shape, np.nan)
        cross, min_cross = (np.full(shape, np.nan), np.full(shape, np.nan)) if len(names) == 2 else (None, None)
        labels = ScanResult({}, fixed, mu_I, status, mu_stop, lam_min=lam_min).minimum_labels()
        edge_axis, _ = _edge_tasks(names, grids, status, fixed, links, M_default, inputs, 0)
        directions = [(edge_axis, edges, min_edges)]
        if cross is not None and min_cross is not None:
            directions.append((next(n for n in names if n != edge_axis), cross, min_cross))
        jobs: list[tuple[np.ndarray, list[EdgeTask]]] = []
        for along, out_s, out_m in directions:
            jobs.append((out_s, _edge_tasks(names, grids, status, fixed, links, M_default, inputs, steps,
                                            along=along)[1]))
            jobs.append((out_m, _edge_tasks(names, grids, labels, fixed, links, M_default, inputs, steps,
                                            "minimum", ("min", "nomin"), along=along)[1]))
        tagged = [(j, t) for j, (_, ts) in enumerate(jobs) for t in ts] if steps > 0 else []
        progress.set_total(len(tasks) + steps * len(tagged))
        if tagged:
            eit = (_unordered(pool, "edge", tagged, max(1, min(20, len(tagged) // (8 * nproc))))
                   if pool is not None else ((j, *_bisect(runner, t)) for j, t in tagged))
            for j, idx, e in eit:
                jobs[j][0][idx] = e
                progress.step(steps)

        # 3. where a boundary bends between two grid lines, extra bisections at intermediate
        #    values of the other axis (adaptive), so that its curve is followed closely
        refined: list[tuple[str, float, str, str, float]] = []
        if len(names) == 2 and steps > 0:
            refined = _refine_bends(runner, pool, nproc, progress, names, grids, edge_axis,
                                    [(status, edges, "status"), (labels, min_edges, "minimum")],
                                    fixed, links, M_default, inputs, steps)
    finally:
        progress.close()
        if pool is not None:
            pool.shutdown()
    n_failed = int(np.sum(status == "failed"))
    if n_failed:
        log.warning("scan: %d of %d points failed to integrate (status 'failed' in scan.csv)",
                    n_failed, status.size)
    sm = runner.run({}, NO_NEW_PHYSICS, inputs, sm_only=True)
    return ScanResult(dict(zip(names, grids)), fixed, mu_I, status, mu_stop, sm.mu_I, edges, edge_axis,
                      lam_min, mu_min, min_edges, sm.lam_min, sm.mu_min, cross, min_cross, refined,
                      lam_low=lam_low)


# --------------------------------------------------------------------------
# Random samples of the Yukawa couplings at fixed M (trace plots)
# --------------------------------------------------------------------------

@dataclass
class SampleResult:
    """Runs at random values of the new Yukawas, all at the same M."""
    M: float
    couplings: list[str]
    values: np.ndarray                   # (samples, couplings)
    tr2: np.ndarray                      # Tr(YY†) of each sample
    tr4: np.ndarray                      # Tr((YY†)²)
    mu_I: np.ndarray                     # NaN where not unstable
    status: np.ndarray
    sm_mu_I: float | None = None


CANDIDATES_PER_SAMPLE = 40        # candidate points per kept sample (only their traces are computed)


def sample_couplings(couplings: list[str], n: int, size_range: tuple[float, float],
                     rng: np.random.Generator) -> np.ndarray:
    """n random candidate points (n, len(couplings)) with all couplings >= 0.

    The overall size s = sqrt(Σ y²) has s² uniform in [lo², hi²]. The direction mixes
    sparse points (one coupling dominates) and democratic ones (all similar): the squared
    components are Dirichlet-distributed with a concentration drawn log-uniformly in
    [0.05, 5], so that the candidates reach the edges of the trace region.
    """
    lo, hi = size_range
    s = np.sqrt(rng.uniform(lo ** 2, hi ** 2, n))
    alpha = np.exp(rng.uniform(np.log(0.05), np.log(5.0), n))
    g = rng.gamma(np.repeat(alpha[:, None], len(couplings), axis=1))
    w = g / np.maximum(g.sum(axis=1, keepdims=True), 1e-300)
    return s[:, None] * np.sqrt(w)


def even_subset(x: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of about n of the points (x, y), spread evenly over the area they cover.

    The bounding box is divided into k x k cells, with k the smallest size for which at
    least n cells are occupied, and one random point is kept per occupied cell.
    """
    if len(x) <= n:
        return np.arange(len(x))
    u = (x - x.min()) / max(np.ptp(x), 1e-300)
    v = (y - y.min()) / max(np.ptp(y), 1e-300)
    order = rng.permutation(len(x))
    lo_k, hi_k = 1, 4 * n
    while lo_k < hi_k:                                  # smallest k with >= n occupied cells
        k = (lo_k + hi_k) // 2
        cells = np.minimum((u * k).astype(int), k - 1) * k + np.minimum((v * k).astype(int), k - 1)
        if len(np.unique(cells)) >= n:
            hi_k = k
        else:
            lo_k = k + 1
    k = lo_k
    cells = np.minimum((u * k).astype(int), k - 1) * k + np.minimum((v * k).astype(int), k - 1)
    _, first = np.unique(cells[order], return_index=True)      # a random point of each cell
    return np.sort(order[first])


def run_samples(runner: Runner, traces, couplings: list[str], M: float, n: int,
                size_range: tuple[float, float], inputs: SMInputs | None = None,
                workers: int | None = 1, label: str = "Sampling") -> SampleResult:
    """Run about n coupling points at fixed M, spread evenly in the plane of the trace plot,
    (Tr(YY†), Tr((YY†)²)^(1/4)).

    traces maps {coupling: value} -> (Tr2, Tr4). Many candidates are drawn (cheap: only their
    traces are computed) and an even subset of them is run. Fixed seed: the same file gives
    the same points.
    """
    rng = np.random.default_rng(0)
    cand = sample_couplings(couplings, CANDIDATES_PER_SAMPLE * n, size_range, rng)
    ctr = np.array([traces(dict(zip(couplings, map(float, v)))) for v in cand])
    # even in the plotted coordinates: Tr(YY†) and the fourth root of Tr((YY†)²)
    keep = even_subset(ctr[:, 0], np.abs(ctr[:, 1]) ** 0.25, n, rng)
    vals, tr = cand[keep], ctr[keep]
    n = len(vals)
    tasks: list[Task] = [((k,), dict(zip(couplings, map(float, v))), M, inputs) for k, v in enumerate(vals)]
    status = np.empty(n, dtype=object)
    mu_I = np.full(n, np.nan)
    nproc: int = (os.cpu_count() or 1) if workers is None else workers
    progress = _Progress(n, label)
    if nproc > 1 and n > 1:
        with ProcessPoolExecutor(max_workers=nproc, initializer=_init_worker, initargs=(runner.spec(),)) as ex:
            it = list_progress(_unordered(ex, "point", tasks, max(1, min(50, n // (8 * nproc)))), progress)
    else:
        it = list_progress(((idx, *_summary(runner.run(v, m, inp))) for idx, v, m, inp in tasks), progress)
    for (k,), st, _, mi, _, _, _ in it:
        status[k] = st
        if mi is not None:
            mu_I[k] = mi
    sm = runner.run({}, NO_NEW_PHYSICS, inputs, sm_only=True)
    return SampleResult(M, list(couplings), vals, tr[:, 0], tr[:, 1], mu_I, status, sm.mu_I)


def list_progress(it, progress: _Progress) -> list:
    """Collect an iterator of results, advancing the progress bar."""
    out = []
    for x in it:
        out.append(x)
        progress.step()
    return out
