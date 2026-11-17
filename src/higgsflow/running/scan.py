"""Parameter scans of the instability scale.

Parameters are the new-physics threshold ``M`` (GeV) and the new Yukawa
couplings (values at μ = M). ``links`` lets one scan parameter drive several
couplings at once, e.g. links={"y": ["y_L4_N", "y_L4c_N_c"]}.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .boundary import SMInputs
from .solver import Runner, RunnerSpec

log = logging.getLogger(__name__)

NO_NEW_PHYSICS = 1e30            # threshold mass used for the pure-SM reference run
Task = tuple[tuple[int, ...], dict[str, float], float, SMInputs | None]


@dataclass
class ScanResult:
    axes: dict[str, np.ndarray]          # 1 or 2 scanned parameters
    fixed: dict[str, float]
    mu_I: np.ndarray                     # NaN where not unstable
    status: np.ndarray                   # str array
    mu_stop: np.ndarray
    sm_mu_I: float | None = None
    extra: dict = field(default_factory=dict)

    def rows(self) -> Iterator[tuple[dict[str, float], float, str, float]]:
        names = list(self.axes)
        grids = [self.axes[n] for n in names]
        for idx in product(*[range(len(g)) for g in grids]):
            vals = [g[i] for g, i in zip(grids, idx)]
            yield dict(zip(names, vals)), self.mu_I[idx], self.status[idx], self.mu_stop[idx]


def expand_links(links: dict | None, yukawas: list[str]) -> dict[str, list[str]]:
    """Replace the shorthand "all" by the list of all new Yukawa couplings and check names."""
    out = {}
    for alias, targets in (links or {}).items():
        targets = list(yukawas) if targets == "all" else list(targets)
        unknown = [t for t in targets if t not in yukawas]
        if unknown:
            raise ValueError(f"link {alias!r}: unknown couplings {unknown}; model has {yukawas}")
        out[alias] = targets
    return out


def resolve_params(point: dict[str, float], links: dict[str, list[str]] | None,
                   default_M: float) -> tuple[dict[str, float], float]:
    """Map scan/user parameters to (coupling values, M)."""
    links = links or {}
    vals, M = {}, default_M
    for k, v in point.items():
        if k == "M":
            M = float(v)
        elif k in links:
            for c in links[k]:
                vals[c] = float(v)
        else:
            vals[k] = float(v)
    return vals, M


_WORKER_RUNNER: Runner | None = None


def _init_worker(spec: RunnerSpec) -> None:
    global _WORKER_RUNNER
    _WORKER_RUNNER = Runner.from_spec(spec)


def _run_point(args: Task) -> tuple:
    idx, vals, M, inputs = args
    assert _WORKER_RUNNER is not None
    r = _WORKER_RUNNER.run(vals, M, inputs)
    return idx, r.status, r.mu_stop, r.mu_I


def run_scan(runner: Runner, scan: dict[str, np.ndarray], fixed: dict[str, float] | None = None,
             links: dict[str, list[str]] | None = None, inputs: SMInputs | None = None,
             M_default: float = 1e3,
             workers: int | None = 1) -> ScanResult:
    """Run the grid. ``workers`` > 1 (or None = all cores) runs points in parallel processes."""
    if not 1 <= len(scan) <= 2:
        raise ValueError("Scan over one or two parameters (fix the rest with `fixed`).")
    fixed = dict(fixed or {})
    names = list(scan)
    grids = [np.asarray(scan[n], dtype=float) for n in names]
    shape = tuple(len(g) for g in grids)
    mu_I = np.full(shape, np.nan)
    mu_stop = np.full(shape, np.nan)
    status = np.empty(shape, dtype=object)

    tasks: list[Task] = []
    for idx in product(*[range(n) for n in shape]):
        point = dict(fixed)
        point.update({n: g[i] for n, g, i in zip(names, grids, idx)})
        vals, M = resolve_params(point, links, M_default)
        tasks.append((idx, vals, M, inputs))

    workers = os.cpu_count() if workers is None else workers
    if workers and workers > 1 and len(tasks) > 1:
        log.info("scan: %d points on %d processes", len(tasks), workers)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(runner.spec(),)) as ex:
            results = list(ex.map(_run_point, tasks, chunksize=max(1, len(tasks) // (4 * workers))))
    else:
        log.info("scan: %d points", len(tasks))
        results = []
        for idx, vals, M, inp in tasks:
            r = runner.run(vals, M, inp)
            results.append((idx, r.status, r.mu_stop, r.mu_I))
    for idx, st, ms, mi in results:
        status[idx] = st
        mu_stop[idx] = ms
        if mi is not None:
            mu_I[idx] = mi
    n_failed = int(np.sum(status == "failed"))
    if n_failed:
        log.warning("scan: %d of %d points failed to integrate (status 'failed' in scan.csv)",
                    n_failed, status.size)
    sm = runner.run({}, NO_NEW_PHYSICS, inputs)
    return ScanResult(dict(zip(names, grids)), fixed, mu_I, status, mu_stop, sm.mu_I)
