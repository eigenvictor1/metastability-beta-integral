"""Native beta functions: evaluate the general formulas, project onto the couplings.

The projection residual (closure of the RG flow on the generated couplings) and
the gauge invariance of the Yukawa tensors are checked and stored in
BetaSystem.meta. Results are cached on disk (see higgsflow.paths).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from ..paths import cache_dir as default_cache_dir
from ..theory.couplings import GAUGE
from ..theory.model import Model
from .formulas import TwoLoopFormulas
from .system import BetaSystem
from .tensors import GAUGE_INDEX, NGI, check_gauge_invariance

log = logging.getLogger(__name__)


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file so parallel workers never see half-written caches."""
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


Coeffs = dict[tuple[int, ...], float]          # monomial exponents -> coefficient


def compute_betas(model: Model, loops: int = 2, cache_dir: str | os.PathLike | None = None,
                  use_cache: bool = True) -> BetaSystem:
    """Native two-loop beta functions for ``model`` (cached on disk).

    For each loop order, the general formulas give the beta tensors; projecting
    them onto the coupling tensors gives the beta functions of the couplings.
    The projection residuals measure whether the flow closes on these couplings.
    """
    cache = _cache_path(model, cache_dir) if use_cache else None
    if cache and cache.exists():
        bs = BetaSystem.from_json(json.loads(cache.read_text()))
        if bs.meta.get("loops", 0) >= loops:
            return bs

    eng = TwoLoopFormulas(model)
    gi_err = check_gauge_invariance(eng.tens)
    if gi_err > 1e-10:
        raise RuntimeError(f"Yukawa tensors are not gauge invariant (violation {gi_err:.2e}).")

    coeffs: dict[str, dict[int, Coeffs]] = {c: {} for c in eng.names}
    closure = {}
    for L in range(1, loops + 1):
        log.debug("native betas: %d-loop", L)
        for gname, d in _project_gauge(eng.gauge(L)).items():
            coeffs[gname][L] = d
        yuk, ry = _project_yukawas(eng.yukawa(L), eng.tens["ybasis"])
        for name, d in yuk.items():
            coeffs[name][L] = d
        quart, rl = _project_quartic(eng.quartic(L), eng.tens["quartics"])
        for name, d in quart.items():
            coeffs[name][L] = d
        closure[L] = {"yukawa_residual": ry, "quartic_residual": rl}

    bs = BetaSystem(eng.names, coeffs, "native",
                    {"loops": loops, "closure": closure, "gauge_invariance": gi_err,
                     "model_hash": model.hash()})
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(cache, json.dumps(bs.to_json()))
    return bs


def _project_gauge(Bg) -> dict[str, Coeffs]:
    """β of each gauge coupling from the gauge beta tensor (g^3 times the diagonal entry / 2)."""
    out = {}
    for alpha, gname in enumerate(GAUGE):
        Aset = [A for A in range(NGI) if GAUGE_INDEX[A] == alpha]
        d: Coeffs = {}
        for m, arr in Bg.terms.items():
            val = np.mean([arr[A, A].real for A in Aset])
            if abs(val) < 1e-12:
                continue
            mm = list(m)
            mm[alpha] += 3
            d[tuple(mm)] = d.get(tuple(mm), 0.0) + 0.5 * val
        out[gname] = d
    return out


def _project_yukawas(By, ybasis: dict[str, np.ndarray]) -> tuple[dict[str, Coeffs], float]:
    """Projection of the Yukawa beta tensor onto each coupling tensor; max residual."""
    res = {m: arr.copy() for m, arr in By.terms.items()}
    out = {}
    for name, basis in ybasis.items():
        nb = np.vdot(basis, basis).real
        d: Coeffs = {}
        for m, arr in By.terms.items():
            v = (np.vdot(basis, arr) / nb).real
            if abs(v) > 1e-11:
                d[m] = v
                res[m] = res[m] - v * basis
        out[name] = d
    return out, max([float(np.abs(a).max()) for a in res.values()] + [0.0])


def _project_quartic(Bl, bases: dict[str, np.ndarray]) -> tuple[dict[str, Coeffs], float]:
    """Projection of the quartic beta tensor onto each quartic coupling tensor (λ, and λ_S, κ
    with a singlet: disjoint supports, hence orthogonal); max residual."""
    res = {m: arr.copy() for m, arr in Bl.terms.items()}
    out: dict[str, Coeffs] = {}
    for name, Lb in bases.items():
        nb = np.vdot(Lb, Lb).real
        d: Coeffs = {}
        for m, arr in Bl.terms.items():
            v = (np.vdot(Lb, arr) / nb).real
            if abs(v) > 1e-11:
                d[m] = v
                res[m] = res[m] - v * Lb
        out[name] = d
    return out, max([float(np.abs(a).max()) for a in res.values()] + [0.0])


def is_cached(model: Model, cache_dir: str | os.PathLike | None = None) -> bool:
    """True if the native betas of model are already in the disk cache."""
    return _cache_path(model, cache_dir).exists()


def _cache_path(model: Model, cache_dir: str | os.PathLike | None) -> Path:
    """Cache file of the native betas of a model (keyed by the model hash)."""
    return Path(cache_dir or default_cache_dir()) / f"native_{model.hash()}.json"
