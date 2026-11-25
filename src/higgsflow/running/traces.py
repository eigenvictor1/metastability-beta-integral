"""Yukawa traces of the new fermions: Tr(YY†) and Tr((YY†)²).

Y is the Yukawa matrix along a neutral real Higgs direction, in the convention
-L ⊃ ½ Y^a_IJ φ_a ψ_I ψ_J (two-component fermions, H = (φ1 + iφ2, φ3 + iφ4)/√2),
i.e. Y ≡ Y^3 here, the matrix called Y in the paper's β functions. Only the new
Yukawas enter (y_t is set to zero), so the traces describe the new fermions.

For the type-I seesaw with one N and coupling y (−L ⊃ y N ℓ·H): Tr(YY†) = y²,
Tr((YY†)²) = y⁴/2.
"""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from ..rge.tensors import build_tensors
from ..theory.model import Model
from .scan import ScanResult, resolve_params

log = logging.getLogger(__name__)

H_NEUTRAL = 2                                   # φ3
TR2, TR4 = "Tr(YY+)", "Tr((YY+)^2)"             # axis names of the two traces in plots
TR4R = "Tr((YY+)^2)^(1/4)"                      # its fourth root (the axis drawn in the maps)


class YukawaTraces:
    """Computes the two traces for given values of the model's new Yukawas."""

    def __init__(self, model: Model):
        """Precompute, for each new Yukawa, its Weyl-block tensor along the neutral direction."""
        tens = build_tensors(model)
        n = tens["N"]
        names = set(model.new_yukawa_names())
        self.basis = {c: arr[H_NEUTRAL, :n, :n] for c, arr in tens["ybasis"].items() if c in names}
        self.n = n

    def matrix(self, values: dict[str, float]) -> np.ndarray:
        """Y for the given coupling values (missing ones are 0)."""
        Y = np.zeros((self.n, self.n), dtype=complex)
        for c, b in self.basis.items():
            Y += float(values.get(c, 0.0)) * b
        return Y

    def __call__(self, values: dict[str, float]) -> tuple[float, float]:
        """(Tr(YY†), Tr((YY†)²))."""
        Y = self.matrix(values)
        P = Y @ Y.conj().T
        return float(np.trace(P).real), float(np.trace(P @ P).real)


def with_trace_axis(scan: ScanResult, traces: YukawaTraces, links: dict | None,
                    M_default: float) -> ScanResult:
    """The same scan with its Yukawa parameter replaced by Tr((YY†)²) (for plotting).

    Needs exactly one scanned Yukawa parameter, with the trace growing along it; points
    where the trace is 0 (y = 0) are dropped, since the trace axis is logarithmic. The
    refined region boundaries are mapped too. Otherwise the scan is returned unchanged.
    """
    names = list(scan.axes)
    yuk = [n for n in names if n != "M"]
    if len(yuk) != 1:
        log.info("trace axis needs one scanned Yukawa parameter (have %s): plot keeps the couplings", yuk)
        return scan
    p = yuk[0]
    a = names.index(p)

    def t4(v: float) -> float:
        vals, _ = resolve_params({**scan.fixed, p: v}, links, M_default)
        return traces(vals)[1]

    T = np.array([t4(v) for v in scan.axes[p]])
    keep = T > 0
    if keep.sum() < 2 or np.any(np.diff(T[keep]) <= 0):
        log.warning("Tr((YY+)^2) does not grow along %s: plot keeps the couplings", p)
        return scan
    sl = tuple(np.flatnonzero(keep) if i == a else slice(None) for i in range(len(names)))
    to_trace = np.vectorize(lambda e: t4(e) if np.isfinite(e) else np.nan, otypes=[float])
    moved = {}                     # bisected boundaries, mapped to the trace if they lie along p
    for k in ("edges", "min_edges"):
        arr = getattr(scan, k)
        if arr is not None:
            moved[k] = (to_trace(arr) if scan.edge_axis == p else arr)[sl]
    axes = {(TR4 if n == p else n): (T[keep] if n == p else scan.axes[n]) for n in names}
    refined = None
    if scan.refined is not None:               # adaptive boundary points along p, mapped too
        refined = [((TR4, o, s1, s2, t4(e)) if ax == p else (ax, o, s1, s2, e))
                   for ax, o, s1, s2, e in scan.refined]
    return scan.map_points(lambda arr: arr[sl], axes=axes, **moved, refined=refined,
                           edge_axis=TR4 if scan.edge_axis == p else scan.edge_axis)


def with_root_axis(scan: ScanResult) -> ScanResult:
    """The same scan with its Tr((YY†)²) axis replaced by the fourth root, [Tr((YY†)²)]^(1/4),
    which has the dimension of a Yukawa coupling (for plotting). Unchanged without that axis."""
    if TR4 not in scan.axes:
        return scan
    axes = {(TR4R if n == TR4 else n): (v ** 0.25 if n == TR4 else v) for n, v in scan.axes.items()}
    def root(arr: np.ndarray | None) -> np.ndarray | None:
        """A bisected boundary along the trace axis, mapped to the fourth root."""
        if arr is None or scan.edge_axis != TR4:
            return arr
        return np.where(np.isfinite(arr), np.abs(arr) ** 0.25, np.nan)
    refined = None
    if scan.refined is not None:
        refined = [((TR4R, o, s1, s2, abs(e) ** 0.25) if ax == TR4 else (ax, o, s1, s2, e))
                   for ax, o, s1, s2, e in scan.refined]
    return replace(scan, axes=axes, edges=root(scan.edges), min_edges=root(scan.min_edges),
                   refined=refined, edge_axis=TR4R if scan.edge_axis == TR4 else scan.edge_axis)
