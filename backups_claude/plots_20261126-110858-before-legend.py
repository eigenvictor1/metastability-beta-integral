"""The result plots (PDF).

running-<theory>.pdf       λ(μ) for example parameter points (+ SM), and the other couplings.
inscale-<theory>.pdf       μ_I as a function of one or two new-physics parameters.
inscale-min-<theory>.pdf   the same, with the scale of the minimum of λ overlaid where λ has one.
inscale-trs-<theory>.pdf   μ_I over the (Tr(YY†), Tr((YY†)²)) plane at fixed M.
"""
from __future__ import annotations

import functools
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.tri as mtri
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import PathPatch

from ..running.boundary import M_PLANCK
from ..running.curves import META_LAMBDA, Curve, meta_value
from ..running.scan import SampleResult, ScanResult
from ..running.solver import GAUGE_NONPERT, RunResult
from ..running.traces import TR2, TR4, TR4R
from ..settings import MAP_GRID, PLOT_DPI
from ..theory import couplings as cp

# TeX look: Computer Modern for text (cmr10, glyphs it lacks from DejaVu Serif) and for math
matplotlib.rcParams.update({"font.family": ["cmr10", "DejaVu Serif"],
                            "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})
BOLD = {"fontfamily": ["cmb10", "DejaVu Serif"]}     # bold Computer Modern (for the names in the maps)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#b9b8b2"
HATCH = "#8a8984"                                                        # hatch lines of the regions
REGION = tuple(0.45 * np.array(matplotlib.colors.to_rgb(NEUTRAL)) + 0.55)   # NEUTRAL at 45% on white, opaque
SEQ = matplotlib.colormaps["magma"]           # colour map of the instability-scale maps (same as seaborn's "magma")
STATUS_TEXT = {"unstable": "unstable", "stable": "stable up to $M_{Pl}$",
               **{f"{g}_nonpert": f"{cp.latex(g)} non-perturbative" for g in cp.GAUGE},
               "nonperturbative": "non-perturbative", "failed": "integration failed"}
STATUS_MARKER = {"g1_nonpert": "v", "g2_nonpert": "s", "g3_nonpert": "D", "nonperturbative": "^", "failed": "x"}
LAMBDA_FLOOR = -0.5                           # lowest λ shown in the running plot
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 1))]


TRACE_LABELS = {TR2: r"$\mathrm{Tr}(YY^\dagger)$", TR4: r"$\mathrm{Tr}[(YY^\dagger)^2]$",
                TR4R: r"$\mathrm{Tr}[(YY^\dagger)^2]^{1/4}$"}
LOG_AXES = ("M", TR2, TR4, TR4R)                      # parameters drawn (and interpolated) on log scales
Limits = dict[str, tuple[float, float]]          # {axis name: (lowest, highest) value shown}


def pretty(name: str) -> str:
    """Axis / legend label of a coupling or scan parameter."""
    if name == "M":
        return r"$M$ [GeV]"
    return TRACE_LABELS.get(name) or cp.latex(name)


def _style(ax) -> None:
    """Common axis style: light grid, no top/right spines, grey ticks."""
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)


def _example_label(params: dict, M: float) -> str:
    """Legend text of an example run: M and the parameter values."""
    bits = [rf"$M$={M:.3g} GeV"] + [f"{pretty(k)}={v:.3g}" for k, v in params.items() if k != "M"]
    return ", ".join(bits)


def _refine(xa: np.ndarray, ya: np.ndarray, Z: np.ndarray, logx: bool, logy: bool, n: int = MAP_GRID
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear interpolation of Z (on the scan grid) onto an n x n grid.

    NaN cells (no instability) stay NaN, so the shaded regions keep their shape.
    """
    from scipy.interpolate import RegularGridInterpolator
    tx = np.log10(xa) if logx else np.asarray(xa, float)
    ty = np.log10(ya) if logy else np.asarray(ya, float)
    fx, fy = np.linspace(tx[0], tx[-1], n), np.linspace(ty[0], ty[-1], n)
    interp = RegularGridInterpolator((tx, ty), Z, bounds_error=False, fill_value=np.nan)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    Zf = interp(np.stack([FX, FY], axis=-1))
    return (10 ** FX if logx else FX), (10 ** FY if logy else FY), Zf


def _mu_I_field(scan: ScanResult, xa: np.ndarray, ya: np.ndarray, logx: bool, logy: bool,
                green: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """log10 μ_I on the fine grid, with the right shape near the stable boundary.

    Near the boundary λ touches zero at its minimum: λ(t) ≈ λ_min + a (t - t_min)² with
    t = log10 μ, so t_I = t_min - sqrt(-λ_min / a). μ_I therefore has a square-root edge,
    which bilinear interpolation of log10 μ_I cannot follow (contours bunch up and wiggle
    along the boundary). Instead λ_min, t_min and the curvature a = -λ_min / (t_min - t_I)²
    are interpolated (all three vary slowly, also across the boundary) and t_I is rebuilt
    from them; at the scan points this is exact. Cells next to a point without a minimum of
    λ use plain interpolation. Also returns where the model is used near the boundary,
    so that contour lines reach the stable boundary instead of stopping a cell short of it.
    """
    Z = np.log10(scan.mu_I)
    Xf, Yf, plain = _refine(xa, ya, _fill_nan(Z), logx, logy)
    none = np.zeros(plain.shape, bool)
    if scan.lam_min is None or scan.mu_min is None:
        return Xf, Yf, plain, none, None
    lam = np.asarray(scan.lam_min, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tm = np.log10(np.asarray(scan.mu_min, float))
        ok = np.isfinite(lam) & np.isfinite(tm)
        gap = tm - Z
        a = np.where(ok & np.isfinite(Z) & (lam < 0) & (gap > 0), -lam / gap ** 2, np.nan)
    if np.isfinite(a).sum() < 4 or ok.sum() < 4:
        return Xf, Yf, plain, none, None
    _, _, lamf = _refine(xa, ya, _fill_nan(np.where(ok, lam, np.nan)), logx, logy)
    _, _, tmf = _refine(xa, ya, _fill_nan(np.where(ok, tm, np.nan)), logx, logy)
    _, _, af = _refine(xa, ya, _fill_nan(a), logx, logy)
    _, _, okf = _refine(xa, ya, ok.astype(float), logx, logy)
    with np.errstate(invalid="ignore"):
        model = tmf - np.sqrt(np.clip(-lamf, 0, None) / af)
    use = (okf > 1 - 1e-9) & np.isfinite(model)
    # only near the stable boundary (where the square-root edge matters), blended smoothly
    # into plain interpolation from MODEL_CELLS to 2 MODEL_CELLS scan cells away from it
    # (distance to the green region from its field, see _region_fields; the λ_min field is no
    # good for this far from the boundary, where λ_min is large and changes fast)
    if green is None or green.shape != plain.shape:
        return Xf, Yf, plain, none, None
    c = 1.0 / max(min(len(xa), len(ya)) - 1, 1)
    w = np.clip(2 - np.abs(green) / (MODEL_CELLS * c), 0, 1)
    w = np.where(use & np.isfinite(w), w, 0.0)
    Zf = np.where(w > 0, w * np.nan_to_num(model) + (1 - w) * np.nan_to_num(plain, nan=0.0), plain)
    Zf = np.where((w > 0) & ~np.isfinite(plain), model, Zf)
    near = (w > 0) & use
    # past the boundary only by a couple of fine cells: further in, the continued field (μ_min)
    # would have contours (and labels) of its own inside the stable region
    d = _signed_distance(np.where(near, lamf, np.nan))
    return Xf, Yf, Zf, near & (d < 2.0 / plain.shape[0]), np.where(near, lamf, np.nan)


MODEL_CELLS = 2.0     # _mu_I_field: square-root model within this many scan cells of the boundary


def _signed_distance(lamf: np.ndarray) -> np.ndarray:
    """λ_min / |∇λ_min| on the fine grid: approximately the signed distance to λ_min = 0, in
    units of the axis lengths (the units of the region fields), > 0 on the stable side."""
    gx, gy = np.gradient(lamf, 1.0 / (lamf.shape[0] - 1), 1.0 / (lamf.shape[1] - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return lamf / np.hypot(gx, gy)


def _lg(r: RunResult) -> np.ndarray:
    """log10 of the run's μ grid (the x axis of the running plot)."""
    return np.log10(r.mu)


# --------------------------------------------------------------------------
# Running plot
# --------------------------------------------------------------------------

def plot_running(path: Path, examples: list[RunResult], sm_run: RunResult,
                 overlay: list[RunResult] | None = None, title: str | None = None) -> None:
    """examples: list of RunResult (native); overlay: optional list of RunResult (PyR@TE), same order."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})
    for ax in (ax1, ax2):
        _style(ax)
    _lambda_panel(ax1, examples, sm_run, overlay, title)
    _yukawa_panel(ax2, examples, sm_run)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _lambda_panel(ax, examples: list[RunResult], sm_run: RunResult, overlay: list[RunResult] | None,
                  title: str | None) -> None:
    """λ(μ) for the SM and each example, with μ_I, minimum and status markers."""
    ax.plot(_lg(sm_run), sm_run.coupling(cp.QUARTIC), color=INK2, lw=2, label="SM")
    if sm_run.mu_I:
        ax.plot(np.log10(sm_run.mu_I), 0, "o", ms=8, mfc="white", mec=INK2, mew=2)
    if sm_run.mu_min:
        ax.plot(np.log10(sm_run.mu_min), sm_run.lam_min, "D", ms=6, mfc="white", mec=INK2, mew=1.5)
    ax.axhline(0, color=INK2, lw=0.8)
    seen = set()
    for k, r in enumerate(examples[: len(SERIES)]):
        col = SERIES[k]
        label = _example_label(r.params, r.M) + ("" if r.status == "unstable" else f"  [{STATUS_TEXT[r.status]}]")
        ax.plot(_lg(r), r.coupling(cp.QUARTIC), color=col, lw=2, label=label)
        ax.axvline(np.log10(r.M), color=col, lw=1, ls=(0, (2, 3)))
        seen.add(r.status)
        _end_marker(ax, r, col)
        if overlay is not None:
            o = overlay[k]
            ax.plot(_lg(o), o.coupling(cp.QUARTIC), color=INK, lw=1, ls="--")
    if overlay is not None:
        ax.plot([], [], color=INK, lw=1, ls="--", label="PyR@TE betas (dashed)")
    ax.plot([], [], "o", ms=7, color=INK2, mec="white", mew=1.5, label=r"$\lambda=0$: instability scale $\mu_I$")
    if sm_run.mu_min or any(r.mu_min for r in examples):
        ax.plot([], [], "D", ms=6, color=INK2, mec="white", mew=1.2, label=r"minimum of $\lambda$")
    for st in (*GAUGE_NONPERT, "nonperturbative", "failed"):
        if st in seen:
            ax.plot([], [], STATUS_MARKER[st], ms=8, color=INK2, mew=2, label=STATUS_TEXT[st])
    ax.set_ylabel(r"$\lambda(\mu)$", color=INK)
    # λ keeps running after μ_I and can fall far below zero before the running stops: show
    # down to LAMBDA_FLOOR at most, so that the region around λ = 0 stays readable
    lam = np.concatenate([r.coupling(cp.QUARTIC) for r in [sm_run, *examples]])
    lam = lam[np.isfinite(lam)]
    if lam.size and lam.min() < LAMBDA_FLOOR:
        top = float(lam.max())
        ax.set_ylim(LAMBDA_FLOOR, top + 0.05 * (top - LAMBDA_FLOOR))
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    ax.set_title(title or "Running of the Higgs quartic", color=INK, fontsize=11, loc="left")


def _end_marker(ax, r: RunResult, col: str) -> None:
    """Circle at μ_I (or the status marker where a non-perturbative/failed run stops) and a
    diamond at the minimum of λ, if it has one."""
    if r.mu_min:
        ax.plot(np.log10(r.mu_min), r.lam_min, "D", ms=6, color=col, mec="white", mew=1.2)
    if r.mu_I:
        ax.plot(np.log10(r.mu_I), 0, "o", ms=8, color=col, mec="white", mew=2)
    elif r.status in STATUS_MARKER:
        lam = r.coupling(cp.QUARTIC)
        ok = np.isfinite(lam)
        if ok.any():
            i = np.flatnonzero(ok)[-1]
            ax.plot(_lg(r)[i], lam[i], STATUS_MARKER[r.status], ms=9, color=col, mew=2.2)


def _yukawa_panel(ax, examples: list[RunResult], sm_run: RunResult) -> None:
    """The other couplings: colour = example, line style = coupling."""
    others = [c for c in examples[0].couplings if c not in (cp.QUARTIC, *cp.GAUGE)] if examples else []
    for k, r in enumerate(examples[: len(SERIES)]):
        for j, c in enumerate(others):
            ax.plot(_lg(r), r.coupling(c), color=SERIES[k], lw=1.6, ls=LINESTYLES[j % len(LINESTYLES)])
    for j, c in enumerate(others):
        ax.plot([], [], color=INK2, lw=1.6, ls=LINESTYLES[j % len(LINESTYLES)], label=pretty(c))
    ax.plot(_lg(sm_run), sm_run.coupling(cp.TOP_YUKAWA), color=INK2, lw=1.2, alpha=0.7)
    ax.set_ylabel("couplings" if any(c in cp.SINGLET_NAMES for c in others) else "Yukawa couplings",
                  color=INK)
    ax.set_xlabel(r"$\log_{10}(\mu/\mathrm{GeV})$", color=INK)
    if others:
        ax.legend(fontsize=7.5, frameon=False, ncol=min(4, len(others)), loc="best")
    ax.set_xlim(np.log10(sm_run.mu[0]), np.log10(M_PLANCK))


# --------------------------------------------------------------------------
# Instability-scale plot
# --------------------------------------------------------------------------

# status shading: (status, hatch, legend label)
# status regions: (status, hatch, legend label, colour)
STABLE_GREEN = "#9fd39a"
STABLE_KEYS = ("stable", *GAUGE_NONPERT)       # statuses with λ > 0 all the way (green regions)
STABLE_ANY = "stable or gauge_nonpert"         # field of the stable regions together
SHADED = (("stable", None, "stable", STABLE_GREEN),
          ("g1_nonpert", "||", STATUS_TEXT["g1_nonpert"], STABLE_GREEN),
          ("g2_nonpert", "\\\\", STATUS_TEXT["g2_nonpert"], STABLE_GREEN),
          ("g3_nonpert", "--", STATUS_TEXT["g3_nonpert"], STABLE_GREEN),
          ("nonperturbative", "///", "non-perturbative", None),
          ("failed", "xx", "integration failed", None),
          ("unscanned", "..", "outside the scan", None))
MU_I_LABEL = r"$\log_{10}(\mu_I/\mathrm{GeV})$"
MU_MIN_LABEL = r"$\log_{10}(\mu_{\min}/\mathrm{GeV})$"   # scale of the minimum of λ
CONTOUR_LW = 0.6                              # width of the contour lines of the maps
MIN_LINE = "#6495ed"                          # μ_min contours, their labels and the minimum line: cornflower blue
MIN_DASH = (0, (8, 6))                        # μ_min contours: long dashes, wide gaps (points)
MIN_SHADE = (0.10, 0.05, 0.95)                # cornflower shading between them: opacity of the band of
                                              # highest μ_min, step per contour, cap (never opaque)
LAM_MIN_LABEL = r"$\lambda_{\min}$"            # value of λ at its minimum
LAM_MIN_KEEP = -0.06                          # λ_min contour always drawn (when in range)
META_YELLOW = (1.00, 0.85, 0.10, 0.22)        # translucent yellow: metastable
UNSTABLE_RED = (0.80, 0.15, 0.12, 0.22)       # translucent red: unstable
META_LABEL_BOX = (1.00, 0.85, 0.10, 0.55)      # semi-transparent yellow behind the label of that line
META_LINE = "#f2c200"                         # yellow: the line λ_min = META_LAMBDA
LAM_MIN_LINE = "#2e8b57"                      # λ_min contours, labels and minimum line: seaborn "seagreen"
EXCLUDED = ("nonperturbative", "failed", "unscanned")   # regions where λ_min is not drawn
LEGEND_BOX = {"frameon": True, "facecolor": "white", "edgecolor": "none", "framealpha": 0.9}


def _swapped(scan: ScanResult | None) -> ScanResult | None:
    """Same scan with the two axes exchanged (arrays transposed)."""
    if scan is None:
        return None
    names = list(scan.axes)[::-1]
    return scan.map_points(np.transpose, axes={n: scan.axes[n] for n in names})


def _cropped(scan: ScanResult, limits: Limits) -> ScanResult:
    """Drop the grid lines outside the limits of each axis, keeping the nearest one outside
    so that the map still reaches the edges of the plot."""
    for a, n in enumerate(scan.axes):
        v = scan.axes[n]
        lo, hi = limits.get(n, (v[0], v[-1]))
        start = max(int(np.searchsorted(v, lo)) - 1, 0)
        stop = min(int(np.searchsorted(v, hi, side="right")) + 1, len(v))
        if start == 0 and stop == len(v):
            continue
        sl = tuple(slice(start, stop) if i == a else slice(None) for i in range(len(scan.axes)))
        scan = scan.map_points(_slicer(sl),
                               axes={k: (w[start:stop] if k == n else w) for k, w in scan.axes.items()})
    return scan


def _slicer(sl: tuple) -> Callable[[np.ndarray], np.ndarray]:
    """Function taking the slice sl of an array."""
    return lambda arr: arr[sl]


def _extended(scan: ScanResult, limits: Limits) -> ScanResult:
    """Extend each limited axis of a 2D scan to its limits, so that all plots with the same
    limits have the same axes.

    The added grid line at the upper end (larger couplings than scanned) is shown as
    non-perturbative, unless the outermost scanned line is stable; there it is not computed.
    At the lower end it takes the status of the outermost line if that is non-perturbative or
    stable, and 'unscanned' otherwise. Added lines have no μ_I; stable ones keep the minimum of
    λ of the outermost line.
    """
    if len(scan.axes) != 2:
        return scan
    for a, n in enumerate(scan.axes):
        if n not in limits:
            continue
        lo, hi = limits[n]
        for at_end, value in ((False, lo), (True, hi)):
            v = scan.axes[n]
            if (value <= v[-1]) if at_end else (value >= v[0]):
                continue
            edge = np.take(scan.status, -1 if at_end else 0, axis=a)
            if at_end:
                st = np.where(np.isin(edge, STABLE_KEYS), edge, "nonperturbative").astype(object)
            else:
                st = np.where(np.isin(edge, ("nonperturbative", *STABLE_KEYS)), edge,
                              "unscanned").astype(object)

            def add(arr, new, a=a, at_end=at_end):
                new = np.expand_dims(new, a)
                return np.concatenate([arr, new] if at_end else [new, arr], axis=a)

            nan = np.full(st.shape, np.nan)
            edges = scan.edges if scan.edges is not None else np.full(scan.status.shape, np.nan)
            if scan.edge_axis == n:
                # a region that is not in the scan starts right at the outermost scanned line
                outer = v[-1] if at_end else v[0]
                k = len(v) - 1 if at_end else 0
                new_e = np.where(st != edge, outer, np.nan)       # new region starts at the scan's end
                if at_end:
                    edges = np.copy(edges)
                    idx = tuple(k if i == a else slice(None) for i in range(2))
                    edges[idx] = np.where(np.isfinite(new_e), new_e, edges[idx])
                    edges = add(edges, nan)
                else:
                    edges = add(edges, new_e)
            else:
                edges = add(edges, nan)
            ax_vals = np.concatenate([v, [value]] if at_end else [[value], v])
            # the minimum of λ carries over from the outermost line where it is shown
            same = {k2: add(getattr(scan, k2), np.where(np.isin(st, STABLE_KEYS),
                                                       np.take(getattr(scan, k2), -1 if at_end else 0, axis=a),
                                                       np.nan))
                    for k2 in ("lam_min", "mu_min", "lam_low") if getattr(scan, k2) is not None}
            scan = scan.map_points(functools.partial(add, new=nan),
                                   axes={k2: (ax_vals if k2 == n else w) for k2, w in scan.axes.items()},
                                   status=add(scan.status, st), edges=edges, **same)
    return scan


# -- status regions with refined boundaries -----------------------------------

def _t(v, log: bool) -> np.ndarray:
    """Plot coordinate used for interpolation: log10 for M, the value itself otherwise."""
    return np.log10(v) if log else np.asarray(v, dtype=float)


def _segments(t: np.ndarray, statuses, edges_t: np.ndarray) -> list[tuple[float, float, str]]:
    """Status segments along one line of the grid, as (start, end, status) in plot coordinates.

    A segment ends where the status changes: at the boundary found by bisection if known
    (edges_t[k], between points k and k+1), else halfway. The first and last segments are
    open-ended (±inf), so the frame is never mistaken for a boundary.
    """
    out, lo = [], -np.inf
    for k in range(len(t) - 1):
        if statuses[k] != statuses[k + 1]:
            e = edges_t[k] if np.isfinite(edges_t[k]) else 0.5 * (t[k] + t[k + 1])
            out.append((lo, float(e), str(statuses[k])))
            lo = float(e)
    out.append((lo, np.inf, str(statuses[-1])))
    return out


def _inside(t: np.ndarray, segs: list[tuple[float, float, str]], key: str | frozenset, far: float) -> np.ndarray:
    """Signed distance (along the line) to the region with status ``key`` (or any of a set of
    statuses): > 0 inside, < 0 outside."""
    keys = key if isinstance(key, frozenset) else frozenset((key,))
    with np.errstate(invalid="ignore"):
        d = np.full(len(t), -far)
        for lo, hi, st in segs:
            if st in keys:
                d = np.maximum(d, np.minimum(t - lo, hi - t))
    return np.clip(d, -far, far)


def _line_fields(scan: ScanResult, along: str, edges: np.ndarray | None, keys: dict[str, str | frozenset],
                 n: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fields of _region_fields computed along one axis: for each region, (F, reliable).

    On each grid line along ``along`` the field is the signed distance to the region (bisected
    boundaries from ``edges``, else halfway), in units of the axis length; between grid lines
    it is interpolated linearly, including the extra lines of the adaptive refinement
    (scan.refined, see _refined_lines). reliable is False where one of the two neighbouring
    lines does not cross the region boundary at all: there the interpolated zero is only a
    guess halfway between the lines.
    """
    names = list(scan.axes)
    a = names.index(along)
    ed = edges if edges is not None else np.full(scan.status.shape, np.nan)
    st = scan.status
    if len(names) == 1:
        st, ed = st[None, :], ed[None, :]
    elif a == 0:                                   # put the bisection axis last
        st, ed = st.T, ed.T
    ta = _t(scan.axes[along], along in LOG_AXES)
    with np.errstate(invalid="ignore"):
        eda = _t(ed, along in LOG_AXES)
    fa = np.linspace(ta[0], ta[-1], n)
    span = abs(ta[-1] - ta[0]) or 1.0
    far = 10 * span + 1
    out = {}
    segs = [_segments(ta, srow, erow) for srow, erow in zip(st, eda)]
    to = np.zeros(1)
    if len(names) == 2:
        other = names[1 - a]
        to = _t(scan.axes[other], other in LOG_AXES)
        extra = _refined_lines(scan, along, other, to, segs)
        if extra:                                     # grid lines and refined lines, in order
            to = np.concatenate([to, [v for v, _ in extra]])
            segs = segs + [sg for _, sg in extra]
            order = np.argsort(to, kind="stable")
            to, segs = to[order], [segs[k] for k in order]
    for name, key in keys.items():
        cols = np.array([_inside(fa, sg, key, far) for sg in segs]) / span
        clipped = (np.abs(cols) >= far / span * (1 - 1e-9)).all(axis=1).astype(float)   # no boundary on the line
        if len(names) == 1:
            out[name] = (cols[0], np.full(n, clipped[0] == 0))
            continue
        fo = np.linspace(to.min(), to.max(), n)
        F = np.array([np.interp(fo, to, cols[:, i]) for i in range(n)]).T     # (other, along)
        ok = np.interp(fo, to, clipped) == 0                                   # both neighbours cross it
        R = np.repeat(ok[:, None], n, axis=1)
        out[name] = (F.T, R.T) if a == 0 else (F, R)
    return out


def _refined_lines(scan: ScanResult, along: str, other: str, to: np.ndarray,
                   segs: list[list[tuple[float, float, str]]]) -> list[tuple[float, list]]:
    """Extra lines between grid lines from the adaptive refinement of the boundaries.

    Each refined point (scan.refined) lies at a value of the other axis between two grid
    lines; there a line is built with the same sequence of regions as the two neighbours (it
    is used only if they have the same one), its boundaries placed half-way between theirs,
    except those that were bisected there. Returns (other-axis coordinate, segments).
    """
    if not scan.refined:
        return []
    log_a, log_o = along in LOG_AXES, other in LOG_AXES
    at: dict[float, list[tuple[str, str, float]]] = {}
    for ax, o, s1, s2, e in scan.refined:
        if ax == along and np.isfinite(e) and (not log_o or o > 0) and (not log_a or e > 0):
            at.setdefault(float(_t(o, log_o)), []).append((s1, s2, float(_t(e, log_a))))
    lines = []
    for v, points in at.items():
        j = int(np.searchsorted(to, v)) - 1
        if not 0 <= j < len(to) - 1:
            continue
        la, lb = segs[j], segs[j + 1]
        if [x[2] for x in la] != [x[2] for x in lb]:
            continue
        w = (v - to[j]) / (to[j + 1] - to[j])
        cuts = [(1 - w) * p[1] + w * q[1] for p, q in zip(la[:-1], lb[:-1])]    # interpolated
        free = list(points)
        for k in range(len(cuts)):
            pair = (la[k][2], la[k + 1][2])
            hit = next((pt for pt in free if (pt[0], pt[1]) == pair), None)
            if hit is not None:
                cuts[k] = hit[2]
                free.remove(hit)
        bounds = [-np.inf] + cuts + [np.inf]
        lines.append((v, [(bounds[k], bounds[k + 1], la[k][2]) for k in range(len(la))]))
    return lines


def _region_fields(scan: ScanResult, n: int = MAP_GRID) -> dict[str, np.ndarray]:
    """For each status, a field on the fine n x n grid (same grid as _refine) that is > 0 inside
    the region and crosses zero at its boundary (in units of the axis length).

    The boundary is located by bisection along the Yukawa axis (edges) and, in 2D scans, also
    along the other axis (cross_edges); the fields are interpolated linearly between grid
    lines, so the boundary is a smooth curve instead of the staircase of the grid cells. Where
    both directions cross the boundary their fields are blended, weighted towards the smaller
    distance (the direction that crosses the boundary more steeply), so boundaries running
    nearly parallel to one axis are still resolved by the other, without a corner where one
    direction takes over.
    """
    present = {str(x) for x in scan.status.flat}
    keys: dict[str, str | frozenset] = {k: k for k in present}
    stable_present = present & set(STABLE_KEYS)
    if len(stable_present) > 1:                  # the stable regions together (one green area)
        keys[STABLE_ANY] = frozenset(stable_present)
    names = list(scan.axes)
    along = scan.edge_axis if scan.edge_axis in names else names[-1]
    main = _line_fields(scan, along, scan.edges, keys, n)
    cross_axis = scan.cross_axis
    if cross_axis is None or scan.cross_edges is None:
        return {k: F for k, (F, _) in main.items()}
    other = _line_fields(scan, cross_axis, scan.cross_edges, keys, n)
    fields = {}
    for k, (F1, R1) in main.items():
        F2, R2 = other[k]
        # where both directions cross the boundary, a smooth blend weighted towards the smaller
        # distance (|F1|·F2 + |F2|·F1)/(|F1| + |F2|): no corner where one takes over from the other
        a1, a2 = np.abs(F1), np.abs(F2)
        with np.errstate(invalid="ignore", divide="ignore"):
            blend = np.where(a1 + a2 > 0, (a2 * F1 + a1 * F2) / (a1 + a2), 0.0)
        fields[k] = np.where(R1 & R2, blend, np.where(R2, F2, F1))
    return fields


STABLE_SMOOTH = 6                  # smoothing of the stable (green) regions, in fine-grid cells


def _smooth_stable(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The fields of the stable regions smoothed with a Gaussian of width STABLE_SMOOTH cells,
    so that the green area has a smooth outline instead of the corners left by the grid (the
    boundary moves by much less than a scan cell). Other regions are unchanged."""
    from scipy.ndimage import gaussian_filter
    out = dict(fields)
    for key in (*STABLE_KEYS, STABLE_ANY):
        if key in out and out[key].ndim == 2:
            out[key] = gaussian_filter(out[key], STABLE_SMOOTH, mode="nearest")
    return out


def _lambda_boundary(fields: dict[str, np.ndarray], lamf: np.ndarray | None, cells: list[int],
                     traced: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Fields with the boundary between the unstable and the stable (green) regions placed
    where the interpolated λ_min crosses zero (see _mu_I_field), so that it is the same smooth
    curve the μ_I contours run into. λ_min varies smoothly across that boundary, unlike the
    status, whose boundary is only known at bisected points and interpolated between them.

    Only within NEAR_CELLS scan cells of that boundary; elsewhere (e.g. stable next to
    non-perturbative) the fields are unchanged. λ_min / |∇λ_min| is used as the field, a
    signed distance in the same units (axis lengths) as the other fields.

    traced: the signed distance to the traced boundary (see _curve_distance; NaN away from it),
    used instead where given: the boundary is then the traced curve itself.
    """
    if (lamf is None and traced is None) or "unstable" not in fields or fields["unstable"].ndim != 2:
        return fields
    stable = [k for k in (STABLE_ANY, *STABLE_KEYS) if k in fields]
    if not stable:
        return fields
    d = _signed_distance(lamf) if lamf is not None else np.full(fields["unstable"].shape, np.nan)
    if traced is not None:                        # > 0 on the stable side
        d = np.where(np.isfinite(traced), traced, d)
    c = NEAR_CELLS / max(min(cells) - 1, 1)
    U = fields["unstable"]
    out = dict(fields)
    green = fields.get(STABLE_ANY, fields[stable[-1]])
    near = np.isfinite(d) & (np.abs(U) < c) & (green > -c)
    out["unstable"] = np.where(near, -d, U)
    for k in stable:
        F = fields[k]
        if k == STABLE_ANY or len(stable) == 1:
            out[k] = np.where(near, d, F)
        else:   # one of several green regions: its boundary with the others stays as it was
            out[k] = np.where(near, np.minimum(d, F), F)
    return out


NEAR_CELLS = 1.5                   # band around the stable boundary drawn from λ_min (scan cells)


def _curves(scan: ScanResult, kind: str, level: float | None = None) -> list[Curve]:
    """The traced curves of a scan of one kind (and level), if any."""
    return [c for c in (scan.curves or []) if c.kind == kind and (level is None or c.level == level)
            and len(c.y) > 1]


def _curve_xy(scan: ScanResult, c: Curve) -> np.ndarray:
    """Points of a traced curve in the data coordinates of the map (x, y)."""
    nx, _ = scan.axes
    return np.column_stack([c.M, c.y] if nx == "M" else [c.y, c.M])


def _curve_distance(scan: ScanResult, curves: list[Curve], Xf: np.ndarray, Yf: np.ndarray,
                    width: float) -> np.ndarray:
    """Signed distance of the fine-grid points to the traced curves, in axis lengths (log
    axes in log space, as the region fields), positive on the side where the curve's function
    is above its level (Curve: on the right); NaN farther than width from every curve."""
    nx, ny = scan.axes
    xa, ya = scan.axes[nx], scan.axes[ny]

    def norm(vals: np.ndarray, ax: np.ndarray, name: str) -> np.ndarray:
        if name in LOG_AXES:
            lo, hi = np.log10(ax[0]), np.log10(ax[-1])
            with np.errstate(divide="ignore", invalid="ignore"):
                return (np.log10(vals) - lo) / (hi - lo)
        return (vals - ax[0]) / (ax[-1] - ax[0])

    P = np.column_stack([norm(Xf.ravel(), xa, nx), norm(Yf.ravel(), ya, ny)])
    best = np.full(len(P), np.inf)
    sign = np.zeros(len(P))
    # orientation: curves run with the side below their level on the left in the (M, y) plane
    flip = -1.0 if nx == "M" else 1.0
    for c in curves:
        xy = _curve_xy(scan, c)
        Q = np.column_stack([norm(xy[:, 0], xa, nx), norm(xy[:, 1], ya, ny)])
        Q = Q[np.all(np.isfinite(Q), axis=1)]
        if len(Q) < 2:
            continue
        lo_b, hi_b = Q.min(axis=0) - width, Q.max(axis=0) + width
        cand = np.flatnonzero(np.all((P > lo_b) & (P < hi_b), axis=1))
        if cand.size == 0:
            continue
        A, B = Q[:-1], Q[1:]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(axis=1), 1e-30)
        for chunk in np.array_split(cand, max(1, cand.size // 4000)):
            X = P[chunk][:, None, :]
            t = np.clip(((X - A) * AB).sum(axis=2) / L2, 0, 1)
            proj = A + t[..., None] * AB
            dist = np.hypot(*(X - proj).transpose(2, 0, 1))
            k = np.argmin(dist, axis=1)
            dk = dist[np.arange(len(chunk)), k]
            rel = P[chunk] - proj[np.arange(len(chunk)), k]
            cross = AB[k, 0] * rel[:, 1] - AB[k, 1] * rel[:, 0]       # > 0: left of the curve
            better = dk < best[chunk]
            best[chunk[better]] = dk[better]
            sign[chunk[better]] = np.where(cross[better] > 0, 1.0, -1.0) * flip
    d = np.where(best <= width, sign * np.where(np.isfinite(best), best, 0.0), np.nan)
    return d.reshape(Xf.shape)


# font sizes of the maps (pt at MAP_SIZE; a map is usually shown at about half that width)
TITLE_SIZE = 18                  # title (bold)
MAP_TICK = 14                    # tick numbers
MAP_AXIS_LABEL = 15              # axis labels
NAME_SIZE = 16                   # region names (bold)
NAME_SMALL = 12                  # ... where the region is too small for NAME_SIZE
LINE_LABEL = 12                  # contour numbers and the labels of the SM and metastability lines
CBAR_TICK, CBAR_LABEL = 14, 15   # the separate colour bar
TITLE_ALPHA = 0.7                # opacity of the white box behind the title of a map
MAP_SIZE = 6.0                   # side of the (square) figure of the 2D maps, inches
MAP_AXES = (0.15, 0.12, 0.82, 0.82)   # its plot area (figure fraction): left, bottom, width, height


def map_colorbar(path: Path, color_range: tuple[float, float]) -> None:
    """Horizontal colour bar of log10 μ_I for the maps of a run (all drawn with color_range),
    as a figure of its own to set above them in a LaTeX figure: same width as a map figure
    (MAP_SIZE), with the bar exactly as wide and placed as their plot area (MAP_AXES), so that
    with equal widths in LaTeX (e.g. both at \\linewidth) it lines up with the map below.
    Ticks at every integer and the label above the bar."""
    lo, hi = color_range
    fig = plt.figure(figsize=(MAP_SIZE, CBAR_HEIGHT))
    left, _, width, _ = MAP_AXES
    cax = fig.add_axes((left, 0.12, width, 0.28))
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(lo, hi), cmap=SEQ)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.locator = matplotlib.ticker.MultipleLocator(1)
    cb.update_ticks()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.set_label(MU_I_LABEL, color=INK, fontsize=CBAR_LABEL)
    cax.tick_params(colors=INK, labelsize=CBAR_TICK)
    cb.outline.set_linewidth(0.6)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


CBAR_HEIGHT = 1.1                # height of the colour-bar figure, inches
SM_DASH = (0, (6, 3))     # the contour μ_I = μ_I^SM
SM_LINE_LW = 1.4


SM_LINE_TEXT = r"$\log_{{10}} \mu_I^{{\mathrm{{SM}}}} = {sm:.1f}$"     # label of the SM contour


def _sm_contour(ax, scan: ScanResult, Xf: np.ndarray, Yf: np.ndarray, Zf: np.ndarray,
                sm: float | None, placed: list[np.ndarray]) -> bool:
    """Dashed contour μ_I = μ_I^SM (sm: log10 of it), labelled on the line; traced if the scan
    has it, else from the interpolated map (Zf, masked to where μ_I exists). True if drawn."""
    from matplotlib.contour import ContourSet
    if sm is None:
        return False
    traced = _curves(scan, "mu_I_sm")
    if traced:
        cs = ContourSet(ax, [sm], [[_curve_xy(scan, c) for c in traced]], colors=INK,
                        linewidths=SM_LINE_LW, linestyles=[SM_DASH], zorder=2.2)  # type: ignore[list-item]
    elif np.isfinite(Zf).any() and np.nanmin(Zf) < sm < np.nanmax(Zf):
        # not traced (results from before it was): from the interpolated map
        cs = ax.contour(Xf, Yf, Zf, levels=[sm], colors=INK, linewidths=SM_LINE_LW, linestyles=[SM_DASH])
    else:
        return False                   # the scan never reaches μ_I^SM
    if not any(len(sg) > 1 for sg in cs.allsegs[0]):
        return False
    cs.set_path_effects([patheffects.withStroke(linewidth=SM_LINE_LW + 1.6, foreground="white", alpha=0.7)])
    _label_line(ax, cs, SM_LINE_TEXT.format(sm=sm), placed)
    return True


def _traced_contours(ax, scan: ScanResult):
    """The traced μ_I contours as a ContourSet (so that they are labelled like grid contours),
    or None without them."""
    from matplotlib.contour import ContourSet
    curves = _curves(scan, "mu_I")
    if not curves:
        return None
    levels = sorted({c.level for c in curves})
    segs = [[_curve_xy(scan, c) for c in curves if c.level == lvl] for lvl in levels]
    return ContourSet(ax, levels, segs, colors=INK, linewidths=CONTOUR_LW)


def _fill_nan(Z: np.ndarray) -> np.ndarray:
    """Copy of Z with NaN cells filled by the nearest finite cell of the grid (drawing only:
    the colour map then continues under the opaque status regions up to their boundaries).

    Nearest in both grid directions, so that a boundary running along either axis is filled
    from the cells right next to it (near the stable region μ_I tends to the scale where λ
    has its minimum, which is what the neighbouring unstable cells show).
    """
    from scipy.ndimage import distance_transform_edt
    Z = np.array(Z, dtype=float)
    bad = ~np.isfinite(Z)
    if not bad.any() or bad.all():
        return Z
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return Z[tuple(idx)]


def plot_instability(path: Path, scan: ScanResult, overlay: ScanResult | None = None,
                     title: str | None = None, limits: Limits | None = None,
                     minima: bool | str = False, y_label: str | None = None,
                     color_range: tuple[float, float] | None = None) -> None:
    """μ_I against one parameter (curve) or two (colour map, M always on the x axis).

    limits: {axis name: (lowest, highest)} to show, e.g. {TR4: (1e-3, 1e3)}; other axes show
    the whole scan.
    minima: where λ has a minimum inside the running range, also show "mu" (or True)
    log10(μ_min/GeV), the scale of the minimum (cornflower blue): dashed contours over the map
    (2D), or a second panel (1D); or "lam": in 2D the metastable / unstable split of the
    unstable region at λ_min = META_LAMBDA (see _metastability_2d), in 1D λ_min itself.
    y_label: label of the Tr[(YY†)²]^(1/4) axis of a map (model setting "y_label"; None: that name).
    color_range: (lowest, highest) log10 μ_I of the colour map (None: the scan's own range).
    """
    what = "mu" if minima is True else (minima or None)
    limits = limits or {}
    names = list(scan.axes)
    if len(names) == 2 and names[1] == "M":
        scan, overlay = _swapped(scan) or scan, _swapped(overlay)
    scan = _extended(_cropped(scan, limits), limits)
    overlay = _extended(_cropped(overlay, limits), limits) if overlay is not None else None
    if scan.lam_min is None:
        what = None
    if what and len(scan.axes) == 1:
        fig, (ax, ax_min) = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True,
                                         gridspec_kw={"height_ratios": [1.6, 1]})
        _style(ax_min)
    elif len(scan.axes) == 2:            # maps: square figure, square plot area at a fixed place
        fig = plt.figure(figsize=(MAP_SIZE, MAP_SIZE))
        ax = fig.add_axes(MAP_AXES)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _style(ax)
    sm = np.log10(scan.sm_mu_I) if scan.sm_mu_I else None
    if len(scan.axes) == 1:
        _instability_1d(ax, scan, overlay, sm, limits)
        if what:
            _minima_1d(ax_min, scan, what)
    fixed = ", ".join((f"$M$ = {v:.3g} GeV" if k == "M" else f"{pretty(k)} = {v:.3g}")
                      for k, v in scan.fixed.items())
    full_title = (title or "Higgs instability scale") + (f"\n{fixed}" if fixed else "")
    if len(scan.axes) == 2:              # maps: the title in a box in the top left corner
        _instability_2d(fig, ax, scan, overlay, sm, limits, what, y_label, color_range, full_title)
    else:
        ax.set_title(full_title, color=INK, fontsize=11, loc="left")
    if len(scan.axes) != 2:              # maps keep MAP_AXES (lined up with map_colorbar)
        fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _instability_1d(ax, scan: ScanResult, overlay: ScanResult | None, sm: float | None,
                    limits: Limits) -> None:
    """One scanned parameter: log10 μ_I against it, the SM value as a dotted line, and
    shaded bands where the points are stable, non-perturbative or failed.
    """
    (n,) = scan.axes
    x = scan.axes[n]
    ax.plot(x, np.log10(scan.mu_I), color=SERIES[0], lw=2, marker="o", ms=3, label=r"$\mu_I$ (native betas)")
    if overlay is not None:
        ax.plot(x, np.log10(overlay.mu_I), color=INK, lw=1, ls="--", label=r"$\mu_I$ (PyR@TE betas)")
    if sm is not None:
        ax.axhline(sm, color=INK2, lw=1, ls=":", label=rf"SM: $\mu_I=10^{{{sm:.2f}}}$ GeV")
    logx = n in LOG_AXES
    tx = _t(x, logx)
    edges = scan.edges if scan.edges is not None else np.full(len(x), np.nan)
    with np.errstate(invalid="ignore"):
        segs = _segments(tx, scan.status, _t(edges, logx))
    for key, hatch, lab, colour in SHADED:
        first = True
        for lo, hi, st in segs:
            if st == key:
                lo, hi = max(lo, tx[0]), min(hi, tx[-1])
                ax.axvspan(10 ** lo if logx else lo, 10 ** hi if logx else hi, color=colour or NEUTRAL, alpha=0.35,
                           hatch=hatch, lw=0, label=lab if first else None)
                first = False
    ax.set_xlabel(pretty(n), color=INK)
    ax.set_ylabel(MU_I_LABEL, color=INK)
    if logx:
        ax.set_xscale("log")
    if n in limits:
        ax.set_xlim(max(x[0], limits[n][0]), min(x[-1], limits[n][1]))
    ax.legend(fontsize=8, **LEGEND_BOX)


def _minima_1d(ax, scan: ScanResult, what: str = "mu") -> None:
    """Panel under the 1D μ_I curve: log10 of μ_min, the scale where λ has its minimum ("mu"),
    or λ_min, its value there ("lam"), against the scanned parameter (gaps where λ has no
    minimum inside the running range)."""
    (n,) = scan.axes
    assert scan.mu_min is not None and scan.lam_min is not None
    y = np.log10(scan.mu_min) if what == "mu" else np.asarray(scan.lam_min, dtype=float)
    colour = MIN_LINE if what == "mu" else LAM_MIN_LINE
    ax.plot(scan.axes[n], y, color=colour, lw=1.5, ls=MIN_DASH, marker="o", ms=3)
    ax.set_ylabel(MU_MIN_LABEL if what == "mu" else LAM_MIN_LABEL, color=INK)
    ax.set_xlabel(pretty(n), color=INK)


def _instability_2d(fig, ax, scan: ScanResult, overlay: ScanResult | None, sm: float | None,
                    limits: Limits, minima: str | None = None, y_label: str | None = None,
                    color_range: tuple[float, float] | None = None, title: str | None = None) -> None:
    """Two scanned parameters: colour map of log10 μ_I (interpolated, with integer contours),
    PyR@TE contours if given, and hatched regions for stable / non-perturbative / failed points.
    With minima ("mu" or "lam"), dashed contours of the scale of the minimum of λ, or of its
    value there, are drawn on top (see _minima_2d).
    """
    nx, ny = scan.axes
    xa, ya = scan.axes[nx], scan.axes[ny]
    logx, logy = nx in LOG_AXES, ny in LOG_AXES
    # set the scales first: contour labels are placed in screen coordinates
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
        if ny == TR4R:                          # narrow log range: plain numbers at 1, 2, 3, 5 x 10^k
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 3, 5)))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(*limits.get(nx, (xa[0], xa[-1])))
    ax.set_ylim(*limits.get(ny, (ya[0], ya[-1])))
    Z = np.log10(scan.mu_I)
    fields = _smooth_stable(_region_fields(scan))
    unstable = fields.get("unstable", np.full((MAP_GRID, MAP_GRID), -1.0))
    # smooth map: interpolate log10 μ_I on a fine grid (in log space for log axes), up to the
    # refined region boundaries
    green = next((fields[k] for k in (STABLE_ANY, *STABLE_KEYS) if k in fields), None)
    Xf, Yf, Zf, below, lamf = _mu_I_field(scan, xa, ya, logx, logy, green)
    bounds = _curves(scan, "lam_low", 0.0)
    cell = NEAR_CELLS / max(min(len(xa), len(ya)) - 1, 1)
    traced = _curve_distance(scan, bounds, Xf, Yf, 2 * cell) if bounds else None
    fields = _lambda_boundary(fields, lamf, [len(xa), len(ya)], traced)
    unstable = fields.get("unstable", unstable)
    # the colour map continues under the (opaque) status regions, so its edge is their smooth
    # boundary; contour lines stop at the boundary
    # colours on a fixed range when given (shared by all maps of a run, see map_colorbar)
    vmin, vmax = color_range if color_range is not None else (None, None)
    ax.pcolormesh(Xf, Yf, Zf, cmap=SEQ, shading="gouraud", rasterized=True, vmin=vmin, vmax=vmax)
    # lines where the point is unstable, and near the stable boundary up to it (the opaque
    # status regions drawn on top hide what lies under them)
    Zf = np.where((unstable > 0) | below, Zf, np.nan)
    labels_at: list[np.ndarray] = []           # label positions, shared with the μ_min contours
    finite = Z[np.isfinite(Z)]
    levels = (np.arange(np.floor(finite.min()), np.ceil(finite.max()) + 1, 1)
              if finite.size and np.ptp(finite) > 0.2 else None)
    traced_cs = _traced_contours(ax, scan)
    if traced_cs is not None:          # contours traced point by point (running/curves.py)
        cs = traced_cs
        levels = np.asarray(cs.levels)
        cs.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="white", alpha=0.5)])
        _label_contours(ax, cs, placed=labels_at)
    elif levels is not None:
        cs = ax.contour(Xf, Yf, Zf, levels=levels, colors=INK, linewidths=CONTOUR_LW)
        # light halo around the lines, so they are visible on the dark end of the map
        cs.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="white", alpha=0.5)])
        _label_contours(ax, cs, placed=labels_at)
    names: list = []                           # region names: written last, clear of all lines
    handles = _status_regions(ax, fields, Xf, Yf, overlap=0.5 / MAP_GRID, placed=labels_at, names=names)
    # the opaque regions: what they hide does not block labels, and labels set apart with a
    # pointer are not put on them (that room is for their own names)
    shaded = [fields[k] > 0 for k, *_ in SHADED if k in fields] + \
        ([fields[STABLE_ANY] > 0] if STABLE_ANY in fields else [])
    ax._hf_cover = _raster(ax, Xf, Yf, np.logical_or.reduce(shaded)) if shaded else None
    sm_line = _sm_contour(ax, scan, Xf, Yf, Zf, sm, labels_at)
    if overlay is not None:
        Zo = np.log10(overlay.mu_I)
        if np.isfinite(Zo).sum() > 3:
            _, _, Zof, below_o, _ = _mu_I_field(overlay, xa, ya, logx, logy, green)
            Zof = np.where((unstable > 0) | below_o, Zof, np.nan)
            ax.contour(Xf, Yf, Zof, levels=levels if levels is not None else 5,
                       colors=SERIES[1], linewidths=0.8, linestyles="--")
    if minima:
        if levels is not None and minima != "lam":
            handles.append(plt.Line2D([], [], color=INK, lw=CONTOUR_LW,
                                      label=MU_I_LABEL + ": instability scale"))
        if minima == "lam":
            handles += _metastability_2d(ax, scan, Xf, Yf, logx, logy, fields, labels_at, names)
        else:
            handles += _minima_2d(ax, scan, Xf, Yf, logx, logy, labels_at, fields, minima)
    if overlay is not None:
        handles.append(plt.Line2D([], [], color=SERIES[1], ls="--", label="PyR@TE betas"))
    if sm is not None:
        # with the dashed μ_I = μ_I^SM contour drawn (labelled in the plot), only the other SM
        # numbers are listed
        parts = [] if sm_line else [rf"$\mu_I=10^{{{sm:.2f}}}$ GeV"]
        if minima == "mu" and scan.sm_mu_min is not None:
            parts.append(rf"$\mu_{{\min}}=10^{{{np.log10(scan.sm_mu_min):.2f}}}$ GeV")
        if minima == "lam" and scan.sm_lam_min is not None:
            parts.append(rf"$\lambda_{{\min}}={scan.sm_lam_min:.4f}$")
        if parts:
            handles.append(plt.Line2D([], [], color="none", label="SM: " + ", ".join(parts)))
    if title:                                  # framed, semi-transparent, in the top left corner; bold
        bold = re.sub(r"\$(.+?)\$", lambda m: r"$\mathbf{" + m.group(1) + "}$", title)
        ax.text(0.02, 0.98, bold, transform=ax.transAxes, ha="left", va="top", fontsize=TITLE_SIZE,
                color=INK, zorder=8, **BOLD,
                bbox={"boxstyle": "square,pad=0.35", "facecolor": (1, 1, 1, TITLE_ALPHA),
                      "edgecolor": INK, "linewidth": 0.6})
    # last: they keep clear of every line and label; first every name that fits in its region,
    # then the others with a pointer (so that a pointer's text does not take a region's room)
    later = [n for n in names
             if not _region_text(ax, Xf, Yf, n[0], n[1], left=n[2], bottom=n[3], placed=labels_at, pointer=False)]
    for mask, text, left, bottom in later:
        _region_text(ax, Xf, Yf, mask, text, left=left, bottom=bottom, placed=labels_at)
    # no legend on the maps: lines and regions are named in the plot itself
    ax.set_xlabel(pretty(nx), color=INK, fontsize=MAP_AXIS_LABEL)
    ax.set_ylabel(y_label if (ny == TR4R and y_label) else pretty(ny), color=INK, fontsize=MAP_AXIS_LABEL)
    ax.tick_params(labelsize=MAP_TICK)
    ax.grid(visible=False, which="both")   # no grid on top of a colour map


def _metastability_2d(ax, scan: ScanResult, Xf: np.ndarray, Yf: np.ndarray, logx: bool, logy: bool,
                      fields: dict[str, np.ndarray], placed: list[np.ndarray] | None = None,
                      names: list | None = None) -> list:
    """Metastability over the map: inside the unstable region, the contour of the lowest λ
    along the running (λ_min, or λ where the running stops if it is still decreasing) at
    META_LAMBDA; above it (shallower) the region is shaded yellow and reads "metastable", below
    it (deeper) red and "unstable". The stable regions (no instability scale) read "stable".
    Returns the legend handles."""
    nx, ny = scan.axes
    unst = fields.get("unstable")
    if scan.lam_low is None or unst is None or not (unst > 0).any():
        return []
    # the λ compared with META_LAMBDA: runs cut short with λ still falling count as unstable
    # (meta_value); μ_max is where the stable runs end
    lam_min = scan.lam_min if scan.lam_min is not None else np.full(scan.status.shape, np.nan)
    with np.errstate(invalid="ignore"):
        mu_max = float(np.nanmax(np.where(scan.status == "stable", scan.mu_stop, np.nan))) \
            if (scan.status == "stable").any() else float(np.nanmax(scan.mu_stop))
    V = np.where(scan.status == "unstable", meta_value(scan.lam_low, lam_min, scan.mu_stop, mu_max),
                 np.nan).astype(float)
    if not np.isfinite(V).any():
        return []
    _, _, Vf = _refine(scan.axes[nx], scan.axes[ny], _fill_nan(V), logx, logy)
    # everything clipped to the smooth boundary of the unstable region
    region = ax.contourf(Xf, Yf, unst, levels=[0.0, float(unst.max()) + 1.0], colors="none")
    clip = PathPatch(region.get_paths()[0], transform=ax.transData)
    region.remove()
    metas = _curves(scan, "lam_low", META_LAMBDA)
    if metas:       # only pieces along which the map really changes there (older results also
                    # traced one along the edge of the non-perturbative region)
        fV = _fine_lookup(scan, Xf, Yf, Vf)
        metas = [c for c in metas if np.nanmedian(np.abs(fV(_curve_xy(scan, c)) - META_LAMBDA)) < 0.03]
    if metas:       # near the traced line, the shading follows it exactly
        cell = NEAR_CELLS / max(min(len(scan.axes[nx]), len(scan.axes[ny])) - 1, 1)
        dm = _curve_distance(scan, metas, Xf, Yf, cell)
        gx, gy = np.gradient(Vf, 1.0 / (Vf.shape[0] - 1), 1.0 / (Vf.shape[1] - 1))
        slope = np.nan_to_num(np.hypot(gx, gy), nan=1.0) + 1e-12
        Vf = np.where(np.isfinite(dm), META_LAMBDA + dm * slope, Vf)
    lo, hi = float(np.nanmin(Vf)), float(np.nanmax(Vf))
    shade = ax.contourf(Xf, Yf, Vf, levels=[min(lo, META_LAMBDA) - 1, META_LAMBDA, max(hi, META_LAMBDA) + 1],
                        colors=[UNSTABLE_RED, META_YELLOW], zorder=2.4, antialiased=True)
    shade.set_clip_path(clip)
    if metas:
        from matplotlib.contour import ContourSet
        line = ContourSet(ax, [META_LAMBDA], [[_curve_xy(scan, c) for c in metas]], colors=META_LINE,
                          linewidths=2 * CONTOUR_LW, linestyles=[MIN_DASH], zorder=3)  # type: ignore[list-item]
    else:
        line = ax.contour(Xf, Yf, np.where(unst > -3.0 / MAP_GRID, Vf, np.nan), levels=[META_LAMBDA],
                          colors=META_LINE, linewidths=2 * CONTOUR_LW, linestyles=[MIN_DASH], zorder=3)
    line.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground="white", alpha=0.6)])
    line.set_clip_path(clip)
    inside = unst > 0
    placed = placed if placed is not None else []
    if any(len(sg) > 1 for sg in line.allsegs[0]):      # the line names its level
        _label_line(ax, line, rf"$\lambda(\mu_*) < {META_LAMBDA:g}$", placed, box=META_LABEL_BOX)
    _region_text(ax, Xf, Yf, inside & (Vf >= META_LAMBDA), "metastable", placed=placed, names=names)
    _region_text(ax, Xf, Yf, inside & (Vf < META_LAMBDA), "unstable", placed=placed, names=names)
    return []


def _fine_lookup(scan: ScanResult, Xf: np.ndarray, Yf: np.ndarray, V: np.ndarray):
    """Function giving V (on the fine grid) at data points (x, y), from the nearest grid point."""
    nx, ny = scan.axes
    lx, ly = nx in LOG_AXES, ny in LOG_AXES
    gx = np.log10(Xf) if lx else Xf
    gy = np.log10(Yf) if ly else Yf
    xs, ys = np.unique(gx), np.unique(gy)
    x_along_1 = Xf.shape[1] > 1 and Xf[0, 1] != Xf[0, 0]

    def look(xy: np.ndarray) -> np.ndarray:
        px = np.log10(xy[:, 0]) if lx else xy[:, 0]
        py = np.log10(xy[:, 1]) if ly else xy[:, 1]
        i = np.clip(np.searchsorted(xs, px), 0, len(xs) - 1)
        j = np.clip(np.searchsorted(ys, py), 0, len(ys) - 1)
        return V[j, i] if x_along_1 else V[i, j]
    return look


def _stable_mask(fields: dict[str, np.ndarray]) -> np.ndarray | None:
    """Where the map is stable (either stable region): no instability scale."""
    masks = [fields[k] > 0 for k in (*STABLE_KEYS, STABLE_ANY) if k in fields]
    return np.logical_or.reduce(masks) if masks else None


RASTER = 320                  # name placement: the frame as a RASTER x RASTER grid (axes fraction)


def _raster(ax, Xf: np.ndarray, Yf: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A fine-grid mask on the placement raster [row = y, column = x] (nearest grid point)."""
    pos = (ax.transData + ax.transAxes.inverted()).transform(np.column_stack([Xf.ravel(), Yf.ravel()]))
    pos = pos.reshape(*Xf.shape, 2)
    x_along_1 = Xf.shape[1] > 1 and Xf[0, 1] != Xf[0, 0]
    M = mask if x_along_1 else mask.T
    P = pos if x_along_1 else pos.transpose(1, 0, 2)
    ux, vy = P[0, :, 0], P[:, 0, 1]
    c = (np.arange(RASTER) + 0.5) / RASTER
    ix = np.clip(np.searchsorted(ux, c), 0, len(ux) - 1)
    iy = np.clip(np.searchsorted(vy, c), 0, len(vy) - 1)
    return M[np.ix_(iy, ix)]


def _busy(ax) -> np.ndarray:
    """Placement raster of what labels must not cover: the drawn lines (contours, the SM and
    metastability lines) and the texts already written."""
    from matplotlib.collections import QuadMesh
    from matplotlib.contour import ContourSet
    from scipy.ndimage import binary_dilation
    busy = np.zeros((RASTER, RASTER), bool)
    to_axes = ax.transAxes.inverted().transform

    def mark(pts: np.ndarray) -> None:
        if len(pts) < 2:
            return
        seg = np.diff(pts, axis=0)
        n = np.maximum(1, (np.hypot(*seg.T) * RASTER * 2).astype(int))
        dense = np.concatenate([pts[k] + np.outer(np.linspace(0, 1, n[k], endpoint=False), seg[k])
                                for k in range(len(seg))] + [pts[-1:]])
        ok = np.all((dense >= 0) & (dense < 1), axis=1)
        idx = (dense[ok] * RASTER).astype(int)
        busy[idx[:, 1], idx[:, 0]] = True

    for art in list(ax.collections) + list(ax.lines):
        if isinstance(art, QuadMesh) or (isinstance(art, ContourSet) and art.filled) or not art.get_visible():
            continue
        tr = art.get_transform()
        paths = art.get_paths() if hasattr(art, "get_paths") else [art.get_path()]
        for path in paths:
            for poly in path.to_polygons(closed_only=False):
                if len(poly):
                    mark(to_axes(tr.transform(poly)))
    busy = binary_dilation(busy, iterations=2)
    cover = getattr(ax, "_hf_cover", None)       # lines under the opaque regions are not seen
    if cover is not None:
        busy &= ~cover
    rend = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(rend)
    for t in ax.texts:
        if not t.get_visible() or not t.get_text():
            continue
        bb = t.get_window_extent(rend)
        x0, x1 = (bb.x0 - frame.x0) / frame.width, (bb.x1 - frame.x0) / frame.width
        y0, y1 = (bb.y0 - frame.y0) / frame.height, (bb.y1 - frame.y0) / frame.height
        c0, c1 = max(int(x0 * RASTER) - 2, 0), min(int(x1 * RASTER) + 3, RASTER)
        r0, r1 = max(int(y0 * RASTER) - 2, 0), min(int(y1 * RASTER) + 3, RASTER)
        busy[r0:r1, c0:c1] = True
    return busy


def _text_cells(ax, text: str, fontsize: float, bold: bool) -> tuple[int, int]:
    """Size (rows, columns) of a text on the placement raster, with a small margin."""
    probe = ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=fontsize, **(BOLD if bold else {}))
    rend = ax.figure.canvas.get_renderer()
    bb, frame = probe.get_window_extent(rend), ax.get_window_extent(rend)
    probe.remove()
    return (int(np.ceil(bb.height / frame.height * RASTER)) + 4,
            int(np.ceil(bb.width / frame.width * RASTER)) + 6)


def _free_boxes(ok: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Share of each raster cell's text box (centred on it) where ok holds; 0 where the box
    would leave the frame."""
    from scipy.ndimage import uniform_filter
    share = uniform_filter(ok.astype(float), size=(rows, cols), mode="constant", cval=0.0)
    edge = np.zeros_like(share, dtype=bool)
    edge[rows // 2 + 1: RASTER - rows // 2 - 1, cols // 2 + 1: RASTER - cols // 2 - 1] = True
    return np.where(edge, share, 0.0)


NAME_INSIDE = 0.8             # share of a region name's box that must lie inside its region


def _write_name(ax, x: float, y: float, text: str, ha: str = "center", size: float = NAME_SIZE) -> None:
    t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="center", fontsize=size, color=INK, zorder=6,
                **BOLD)
    t.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white", alpha=0.8)])


def _region_text(ax, Xf: np.ndarray, Yf: np.ndarray, mask: np.ndarray | None, text: str,
                 min_share: float = 0.002, left: bool = False,
                 placed: list[np.ndarray] | None = None, bottom: bool = False,
                 names: list | None = None, pointer: bool = True) -> bool:
    """Write a region's name inside it, in a spot where the whole name lies in the region and
    covers no line or other text: the deepest such spot (left: the rightmost, right-aligned and
    as low as possible, i.e. just above the region's lower boundary (the non-perturbative strip);
    bottom: the lowest, just above the lower edge, centred on the region's stretch there).
    If there is no such spot, the name goes nearby with a pointer (_annotate_near).
    With names (a list), the request is only recorded there, to be written once all lines are
    drawn (see _instability_2d)."""
    if mask is None or not mask.any() or mask.mean() < min_share:
        return True
    if names is not None:
        names.append((mask, text, left, bottom))
        return True
    from scipy.ndimage import distance_transform_edt
    region = _raster(ax, Xf, Yf, mask)
    if not region.any():
        return True
    depth = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    free = ~_busy(ax)
    for size in (NAME_SIZE, NAME_SMALL):       # a smaller font before resorting to a pointer
        rows, cols = _text_cells(ax, text, size, True)
        # room: nothing drawn under the name, and most of it (NAME_INSIDE) inside the region
        fits = (_free_boxes(free, rows, cols) >= 0.999) & (_free_boxes(region, rows, cols) >= NAME_INSIDE)
        if not fits.any():
            continue
        r, c = np.argwhere(fits).T
        if bottom:                       # lowest spots, then the middle of the region there
            low = r <= r.min() + 2
            cols_low = np.flatnonzero(region[r.min()])
            mid = 0.5 * (cols_low.min() + cols_low.max()) if cols_low.size else c[low].mean()
            k = np.flatnonzero(low)[np.argmin(np.abs(c[low] - mid))]
        elif left:                       # "edge" mode: rightmost spots, the lowest of them, i.e.
            rgt = c >= c.max() - 2           # right-aligned just above the region's lower boundary
            k = np.flatnonzero(rgt)[np.argmin(r[rgt])]
            x = (c[k] + cols // 2 - 3 + 0.5) / RASTER
            _write_name(ax, x, (r[k] + 0.5) / RASTER, text, ha="right", size=size)
            return True
        else:
            k = int(np.argmax(depth[r, c]))
        _write_name(ax, (c[k] + 0.5) / RASTER, (r[k] + 0.5) / RASTER, text, size=size)
        return True
    if not pointer:                      # no room inside: the caller retries with a pointer later
        return False
    i, j = np.unravel_index(int(np.argmax(depth)), depth.shape)
    _annotate_near(ax, np.array([(j + 0.5) / RASTER, (i + 0.5) / RASTER]), text, placed,
                   fontsize=NAME_SIZE, bold=True)
    return True




def _annotate_near(ax, target: np.ndarray, text: str, placed: list[np.ndarray] | None,
                   fontsize: float = LINE_LABEL, bold: bool = False, box: tuple | None = None) -> None:
    """Label with a pointer: the text in the nearest spot around target (axes fraction) where
    it covers no line or other text (rings of growing radius), with a thin line to target."""
    rows, cols = _text_cells(ax, text, fontsize, bold)
    cover = getattr(ax, "_hf_cover", None)
    share = _free_boxes(~_busy(ax) & (~cover if cover is not None else True), rows, cols)
    best, best_share = None, -1.0
    for rad in (0.08, 0.11, 0.15, 0.2, 0.26, 0.33, 0.42):
        for ang in np.radians(np.arange(0, 360, 15)):
            c = target + rad * np.array([np.cos(ang), np.sin(ang)])
            if not (0 < c[0] < 1 and 0 < c[1] < 1):
                continue
            sh = share[int(c[1] * RASTER), int(c[0] * RASTER)]
            if sh > best_share + 1e-9:
                best, best_share = c, sh
        if best_share >= 0.999:
            break
    if best is None:
        return
    a = ax.annotate(text, xy=tuple(target), xycoords="axes fraction", xytext=tuple(best),
                    textcoords="axes fraction", ha="center", va="center", fontsize=fontsize,
                    color=INK, zorder=7, **(BOLD if bold else {}),
                    arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7, "shrinkA": 3, "shrinkB": 0})
    a.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white", alpha=0.8)])
    if box is not None:
        a.set_bbox({"boxstyle": "square,pad=0.25", "facecolor": box, "edgecolor": "none"})
    if placed is not None:
        placed.append(np.asarray(best))


def _lam_levels(lo: float, hi: float) -> np.ndarray:
    """Contour levels for λ_min in (lo, hi): ±1, 2, 5 × 10^k, so about three per decade of |λ|
    whatever the scale, over the two decades below the largest |λ_min| (1, 2, …, 9 × 10^k if
    that gives fewer than three; evenly spaced round values if the range is narrower still),
    plus LAM_MIN_KEEP, whose neighbours closer than a factor 1.3 are dropped (e.g. −0.05 next
    to −0.06)."""
    floor = max(abs(lo), abs(hi)) / 100

    def pick(mantissas):
        mags = [m * 10.0 ** k for k in range(-6, 3) for m in mantissas]
        return [v for v in sorted({*mags, *(-m for m in mags)}) if lo < v < hi and abs(v) >= floor]
    levels = pick((1, 2, 5))
    if len(levels) < 3:
        levels = pick(range(1, 10))
    if len(levels) < 3:
        ticks = matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(lo, hi)
        levels = [float(v) for v in ticks if lo < v < hi]
    keep = LAM_MIN_KEEP
    if lo < keep < hi:
        levels = [v for v in levels if not (v * keep > 0 and 1 / 1.3 < v / keep < 1.3)] + [keep]
    return np.round(np.array(sorted(levels)), 8)


def _minima_2d(ax, scan: ScanResult, Xf: np.ndarray, Yf: np.ndarray, logx: bool, logy: bool,
               labels_at: list[np.ndarray] | None = None,
               status_fields: dict[str, np.ndarray] | None = None, what: str = "mu") -> list:
    """Minimum of λ over the map. what="lam": the same drawing for λ_min, the value of λ at the
    minimum, in seagreen (levels: see _lam_levels; more opaque for deeper minima).
    what="mu": scale of the minimum: cornflower-blue dashed labelled contours of log10(μ_min/GeV)
    where λ has a minimum inside the running range (integer values, or half-integers if no
    integer is in the range) with cornflower shading between them (more opaque for lower μ_min),
    and a thicker dashed line where the minimum appears
    (beyond it λ still decreases where the running stops). That line is the boundary located by
    bisection (scan.min_edges), drawn like the status regions. Not drawn in the
    non-perturbative / failed / unscanned regions.
    labels_at: positions of labels already drawn (the μ_I contours), which these labels avoid.
    status_fields: the status regions of the map (_region_fields); the μ_min contours are not
    drawn inside the stable ones.
    Returns the legend handles.
    """
    nx, ny = scan.axes
    if scan.lam_min is None or not np.isfinite(np.asarray(scan.lam_min, dtype=float)).any():
        return []
    labels = scan.minimum_labels()
    edges = scan.edges if scan.edges is not None else np.full(labels.shape, np.nan)
    if scan.min_edges is not None:            # min / nomin from min_edges, the rest from the status edges
        edges = np.where(np.isfinite(scan.min_edges), scan.min_edges, edges)
    cross = scan.cross_edges
    if cross is not None and scan.min_cross_edges is not None:
        cross = np.where(np.isfinite(scan.min_cross_edges), scan.min_cross_edges, cross)
    fields = _region_fields(replace(scan, status=labels, edges=edges, cross_edges=cross))
    F = fields["min"]
    shown = np.ones_like(F, dtype=bool)
    for key in EXCLUDED:
        if key in fields:
            shown &= fields[key] <= 0
    handles: list = []
    colour = MIN_LINE if what == "mu" else LAM_MIN_LINE
    if "nomin" in fields:
        edge = ax.contour(Xf, Yf, np.where(shown, F, np.nan), levels=[0.0], colors=colour,
                          linewidths=2 * CONTOUR_LW, linestyles=[MIN_DASH], zorder=3)
        edge.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground="white", alpha=0.5)])
    xa, ya = scan.axes[nx], scan.axes[ny]
    assert scan.mu_min is not None and scan.lam_min is not None
    value = np.log10(np.asarray(scan.mu_min, dtype=float)) if what == "mu" else np.asarray(scan.lam_min, dtype=float)
    L = np.where(labels == "min", value, np.nan).astype(float)
    _, _, Lf = _refine(xa, ya, _fill_nan(L), logx, logy)
    # shading and contours are clipped to the region where λ has a minimum (its smooth boundary
    # is the wide dashed line), instead of being cut along the cells of the fine grid
    # not in the stable (green) regions either: there λ has a positive minimum, not shown
    drawn = shown.copy()
    for key in (*STABLE_KEYS, STABLE_ANY):
        if status_fields is not None and key in status_fields:
            drawn &= status_fields[key] <= 0
    inside = np.where(drawn, F, -1.0)
    if not (inside > 0).any():
        return handles
    region = ax.contourf(Xf, Yf, inside, levels=[0.0, float(inside.max()) + 1.0], colors="none")
    clip = PathPatch(region.get_paths()[0], transform=ax.transData)
    region.remove()
    vals = L[np.isfinite(L)]
    if vals.size and np.ptp(vals) > 0 and what == "lam":
        levels = _lam_levels(float(vals.min()), float(vals.max()))
        fmt = "%g"
    elif vals.size and np.ptp(vals) > 0:
        levels = np.arange(np.ceil(vals.min()), np.floor(vals.max()) + 1)       # integers
        fmt = "%d"
        if not levels.size:                                    # no integer in the range: half-integers
            levels = np.arange(np.ceil(2 * vals.min()), np.floor(2 * vals.max()) + 1) / 2
            fmt = "%g"
        # no contour right at the top of the range: it would run along the wide dashed line
        # (where the minimum reaches the end of the running) and wiggle around it
        near_top = levels > vals.max() - 0.25
        if near_top.any() and not near_top.all():
            levels = levels[~near_top]
    if vals.size and np.ptp(vals) > 0:
        if levels.size:
            # cornflower shading between the contours, 0.05 more opaque after each contour towards
            # lower μ_min (never opaque);
            # under the status regions, so the stable region stays plain green
            bounds = np.unique(np.concatenate([[vals.min()], levels, [vals.max()]]))
            if bounds.size > 1:
                start, step, cap = MIN_SHADE
                alphas = np.minimum(start + step * np.arange(bounds.size - 1)[::-1], cap)
                rgb = matplotlib.colors.to_rgb(colour)
                bounds[0], bounds[-1] = bounds[0] - 1e-6, bounds[-1] + 1e-6
                shade = ax.contourf(Xf, Yf, Lf, levels=bounds, colors=[(*rgb, float(al)) for al in alphas],
                                    zorder=2.4, antialiased=True)
                shade.set_clip_path(clip)
            # lines only inside the region (a few fine cells beyond it, cut by the clip), so that
            # their labels are inside too
            Lc = np.where(inside > -3.0 / MAP_GRID, Lf, np.nan)
            cs = ax.contour(Xf, Yf, Lc, levels=levels, colors=colour, linewidths=CONTOUR_LW,
                            linestyles=[MIN_DASH], zorder=3)
            cs.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="white", alpha=0.5)])
            cs.set_clip_path(clip)
            _label_contours(ax, cs, fmt=fmt, colour=colour, placed=labels_at)
    label = (MU_MIN_LABEL + r": scale of the minimum of $\lambda$" if what == "mu"
             else LAM_MIN_LABEL + r": value of $\lambda$ at its minimum")
    handles.append(plt.Line2D([], [], color=colour, lw=CONTOUR_LW, ls=MIN_DASH, label=label))
    return handles


LABEL_BOX = (0.1, 0.053)        # half-free zone around a contour label (axes fraction): width, height


def _label_line(ax, cs, text: str, placed: list[np.ndarray], box: tuple | None = None) -> None:
    """Label a single-level line (the SM contour, the metastability line) with a longer text:
    where it crosses x = 0.8, 0.7, 0.9, 0.6 or 0.5 of the frame (right of the column of the μ_I
    labels) clear of the labels already placed, else at the best spot along the line. box: face
    colour (RGBA) of a rectangle behind the text (which keeps its white halo)."""
    to_axes = (ax.transData + ax.transAxes.inverted()).transform
    to_data = (ax.transAxes + ax.transData.inverted()).transform
    segs = [to_axes(sg) for sg in cs.allsegs[0] if len(sg) > 1]
    if not segs:
        return
    spot = None
    for xc in (0.8, 0.7, 0.9, 0.6, 0.5):
        for pts in segs:
            dx = pts[:, 0] - xc
            for i in np.flatnonzero(dx[:-1] * dx[1:] <= 0):
                if dx[i] == dx[i + 1]:
                    continue
                p = pts[i] + dx[i] / (dx[i] - dx[i + 1]) * (pts[i + 1] - pts[i])
                if 0.06 < p[1] < 0.94 and _gap(p, placed) > LONG_LABEL:
                    spot = p
                    break
            if spot is not None:
                break
        if spot is not None:
            break
    if spot is None:                     # no clear spot on the line: label nearby, with a pointer
        target = _label_on_line(segs, [], 0.06)
        if target is not None:
            _annotate_near(ax, target, text, placed, box=box)
        return
    placed.append(spot)
    texts = ax.clabel(cs, levels=list(cs.levels), manual=[tuple(to_data(spot[None, :])[0])],
                      fmt={cs.levels[0]: text}, fontsize=LINE_LABEL, inline=True,
                      inline_spacing=6)
    for t in texts:
        t.set_color(INK)
        t.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")])
        if box is not None:
            t.set_bbox({"boxstyle": "square,pad=0.25", "facecolor": box, "edgecolor": "none"})


LONG_LABEL = 2.2        # _label_line: clearance from other labels, in units of LABEL_BOX


def _gap(p: np.ndarray, placed: list[np.ndarray]) -> float:
    """Distance from p to the nearest placed label in units of LABEL_BOX (> 1: no overlap)."""
    w, h = LABEL_BOX
    return min((max(abs(p[0] - q[0]) / w, abs(p[1] - q[1]) / h) for q in placed), default=np.inf)


def _label_contours(ax, cs, margin: float = 0.04, fmt: str | dict = "%d",
                    colour: str = INK, placed: list[np.ndarray] | None = None) -> None:
    """Readable contour labels: one for every level, lined up and without overlaps.

    The labels of a set of contours are lined up in one column: the vertical line (axes
    fraction 0.1 ... 0.9) that crosses the most levels at spots inside the frame and clear of
    every other label (LABEL_BOX). Levels that this column does not reach get their label
    along their own line instead (see _label_on_line). ``placed`` collects the label positions
    (axes coordinates) and can be shared between calls, so that the labels of two sets of
    contours avoid each other too. Labels are drawn with a white halo, so they can be read on
    dark parts of the colour map.
    """
    to_axes = (ax.transData + ax.transAxes.inverted()).transform
    to_data = (ax.transAxes + ax.transData.inverted()).transform
    placed = placed if placed is not None else []
    lines = [(lvl, [to_axes(sg) for sg in segs if len(sg) > 1]) for lvl, segs in zip(cs.levels, cs.allsegs)]
    lines = [(lvl, segs) for lvl, segs in lines if segs]
    if not lines:
        return

    def column(xc: float) -> list[tuple[float, np.ndarray]]:
        """Label spots of the levels on the vertical line x = xc (greedy, no overlaps)."""
        taken = list(placed)
        spots = []
        for lvl, segs in lines:
            best = None
            for pts in segs:
                dx = pts[:, 0] - xc
                for i in np.flatnonzero(dx[:-1] * dx[1:] <= 0):
                    if dx[i] == dx[i + 1]:
                        continue
                    t = dx[i] / (dx[i] - dx[i + 1])
                    p = pts[i] + t * (pts[i + 1] - pts[i])
                    if not margin < p[1] < 1 - margin:
                        continue
                    gap = _gap(p, taken)
                    if gap > 1 and (best is None or gap > best[0]):
                        best = (gap, p)
            if best is not None:
                taken.append(best[1])
                spots.append((lvl, best[1]))
        return spots

    xcs = np.linspace(0.1, 0.9, 33)
    columns = [column(xc) for xc in xcs]
    # most levels; among equals the column nearest the middle
    best_col = max(range(len(columns)), key=lambda k: (len(columns[k]), -abs(k - len(columns) // 2)))
    spots = dict((lvl, p) for lvl, p in columns[best_col])
    taken = list(placed) + list(spots.values())

    def near_column(segs: list[np.ndarray]) -> np.ndarray | None:
        """A spot on the line as close as possible to the chosen column (left or right of it,
        up to 0.2 of the frame), clear of the other labels: the labels stay lined up."""
        for off in (0.03, -0.03, 0.06, -0.06, 0.1, -0.1, 0.15, -0.15, 0.2, -0.2):
            xc = xcs[best_col] + off
            if not 0.05 < xc < 0.95:
                continue
            for pts in segs:
                dx = pts[:, 0] - xc
                for i in np.flatnonzero(dx[:-1] * dx[1:] <= 0):
                    if dx[i] == dx[i + 1]:
                        continue
                    p = pts[i] + dx[i] / (dx[i] - dx[i + 1]) * (pts[i + 1] - pts[i])
                    if margin < p[1] < 1 - margin and _gap(p, taken) > 1:
                        return p
        return None

    positions, levels = [], []
    pointers: list[tuple[float, np.ndarray]] = []
    for lvl, segs in lines:
        p = spots.get(lvl)
        if p is None:
            p = near_column(segs)
            if p is not None:
                taken.append(p)
        if p is None:
            p = _label_on_line(segs, placed, margin)
            if p is not None and _gap(p, placed) <= 1:      # no room on the line: with a pointer
                pointers.append((lvl, p))
                continue
        if p is not None:
            placed.append(p)
            positions.append(tuple(to_data(p[None, :])[0]))
            levels.append(lvl)
    for lvl, p in pointers:
        txt = fmt.get(lvl, f"{lvl:g}") if isinstance(fmt, dict) else fmt % lvl
        _annotate_near(ax, p, txt, placed)
    if not positions:
        return
    texts = ax.clabel(cs, levels=levels, manual=positions, fmt=fmt, fontsize=LINE_LABEL, inline=True,
                      inline_spacing=6)
    for t in texts:
        t.set_color(colour)
        t.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")])


def _label_on_line(segs: list[np.ndarray], placed: list[np.ndarray], margin: float) -> np.ndarray | None:
    """Label spot (axes coordinates) along a level's own lines, longest line first, from the
    middle outwards: the first spot inside the frame and clear of the other labels
    (LABEL_BOX), else the spot farthest from them."""
    fracs = 0.5 + np.array([0] + [sg * d for d in np.arange(0.05, 0.5, 0.05) for sg in (-1, 1)])
    best: tuple[float, np.ndarray] | None = None
    for pts in sorted(segs, key=lambda p: -np.hypot(*np.diff(p, axis=0).T).sum()):
        arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        for frac in fracs:
            p = pts[min(int(np.searchsorted(arc, frac * arc[-1])), len(pts) - 1)]
            if not np.all((p > margin) & (p < 1 - margin)):
                continue
            gap = _gap(p, placed)
            if best is None or gap > best[0]:
                best = (gap, p)
            if gap > 1:
                return p
    return best[1] if best is not None else None


def _status_regions(ax, fields: dict[str, np.ndarray], Xf: np.ndarray, Yf: np.ndarray,
                    overlap: float = 0.0, placed: list[np.ndarray] | None = None,
                    names: list | None = None) -> list:
    """Regions where the scan is stable (green) or non-perturbative / failed / unscanned
    (grey, hatched); returns their legend handles.

    Each region is the area where its field (see _region_fields) is positive. It is opaque and
    drawn on top of the colour map (which continues underneath), from -overlap, so that its
    smooth boundary is the edge of the map: no gaps or steps from the fine grid.
    """
    handles: list = []                         # regions are named in the plot, not in the legend
    for key, hatch, lab, colour in SHADED:
        F = fields.get(key)
        if F is None or not (F > 0).any():
            continue
        own = F > 0                            # the region itself (where its name is written)
        if key == "stable":                    # not under the gauge non-perturbative regions on top
            for g in GAUGE_NONPERT:
                if g in fields:
                    own &= ~(fields[g] > 0)
        if key == "stable" and STABLE_ANY in fields:
            # one green area for all stable regions (the gauge non-perturbative ones are hatched on top),
            # so that no strip of the colour map shows between them
            F = fields[STABLE_ANY]
        top = float(F.max()) + 1.0
        face = colour or REGION
        with matplotlib.rc_context({"hatch.color": HATCH, "hatch.linewidth": 0.8}):
            ax.contourf(Xf, Yf, F, levels=[-overlap, top], colors=[face], hatches=[hatch], zorder=2.5,
                        antialiased=True)
        # the non-perturbative strip along the top: its name at the right, just above its lower
        # boundary; stable: along the bottom
        _region_text(ax, Xf, Yf, own, lab, left=key == "nonperturbative", placed=placed,
                     bottom=key == "stable", names=names)
    return handles


# --------------------------------------------------------------------------
# Trace plot
# --------------------------------------------------------------------------

def plot_traces(path: Path, res: SampleResult, title: str | None = None) -> None:
    """μ_I at fixed M over the (Tr(YY†), Tr((YY†)²)) plane, from the sampled Yukawa points.

    The samples cover the reachable part of the plane evenly; log10 μ_I is interpolated
    linearly on their triangulation (filled colours plus integer contour lines), drawn only
    inside the sampled region, and each sample is kept as a faint dot. Samples that are stable
    or non-perturbative before μ_I are grey markers. The traces of a model only fill part of
    the plane, so the blank area is not reachable with that field content. If all samples
    lie on a curve (one Yukawa direction, e.g. the 1N seesaw) they are drawn as coloured dots.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _style(ax)
    ax.set_axisbelow(True)                      # grid lines behind the map
    ok = np.isfinite(res.tr2) & np.isfinite(res.tr4)
    uns = ok & (res.status == "unstable")
    x, y4, z = res.tr2[uns], res.tr4[uns], np.log10(res.mu_I[uns])
    y = y4 ** 0.25                              # drawn as [Tr((YY†)²)]^(1/4)
    ratio = y4 / np.maximum(x, 1e-300) ** 2
    on_curve = len(x) < 10 or np.ptp(ratio) <= 1e-6 * np.median(ratio)   # Tr² ∝ Tr²: one direction
    tri = None if on_curve else _sample_triangulation(x, y)
    if tri is not None:
        lo, hi = float(z.min()), float(z.max())
        fill = ax.tricontourf(tri, z, levels=np.linspace(lo, hi, 64), cmap=SEQ)
        for c in getattr(fill, "collections", [fill]):
            c.set_rasterized(True)
        fig.colorbar(fill, ax=ax, ticks=np.arange(np.ceil(lo), np.floor(hi) + 1)).set_label(MU_I_LABEL, color=INK)
        levels = np.arange(np.ceil(lo), np.floor(hi) + 1)
        if levels.size:
            cs = ax.tricontour(tri, z, levels=levels, colors=INK, linewidths=0.6)
            cs.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="white", alpha=0.5)])
            _label_contours(ax, cs)
        ax.scatter(x, y, s=3, color=NEUTRAL, alpha=0.5, lw=0, rasterized=True)   # the samples
    else:
        order = np.argsort(-z)                                              # low μ_I on top
        sc = ax.scatter(x[order], y[order], c=z[order], cmap=SEQ, s=10, lw=0, rasterized=True)
        fig.colorbar(sc, ax=ax).set_label(MU_I_LABEL, color=INK)
    handles = []
    for key, marker, lab in (("nonperturbative", "^", "non-perturbative before $\\mu_I$"),
                             ("stable", "o", "stable up to $M_{Pl}$"),
                             *((k, STATUS_MARKER[k], STATUS_TEXT[k]) for k in GAUGE_NONPERT),
                             ("failed", "x", "integration failed")):
        sel = ok & (res.status == key)
        if sel.any():
            ax.scatter(res.tr2[sel], res.tr4[sel] ** 0.25, marker=marker, s=12, color=NEUTRAL,
                       lw=0.6 if key == "failed" else 0)
            handles.append(plt.Line2D([], [], ls="", marker=marker, color=NEUTRAL, label=lab))
    if res.sm_mu_I:
        handles.append(plt.Line2D([], [], color="none", label=rf"SM: $\mu_I=10^{{{np.log10(res.sm_mu_I):.2f}}}$ GeV"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(TRACE_LABELS[TR2], color=INK)
    ax.set_ylabel(TRACE_LABELS[TR4R], color=INK)
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="upper left", **LEGEND_BOX)
    ax.set_title((title or "Higgs instability scale") + f"\n$M$ = {res.M:.3g} GeV, {len(res.status)} samples",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _sample_triangulation(x: np.ndarray, y: np.ndarray, max_edge: float = 4.0):
    """Delaunay triangulation of the samples, without the triangles that bridge empty space.

    Coordinates are scaled to the unit box; a triangle is dropped if one of its edges is
    longer than max_edge times the median edge, so the map follows the (non-convex) shape
    of the sampled region. None if the triangulation fails.
    """
    u = (x - x.min()) / max(np.ptp(x), 1e-300)
    v = (y - y.min()) / max(np.ptp(y), 1e-300)
    try:
        tri = mtri.Triangulation(u, v)
    except (ValueError, RuntimeError):
        return None
    t = tri.triangles
    e = np.stack([np.hypot(u[t[:, a]] - u[t[:, b]], v[t[:, a]] - v[t[:, b]]) for a, b in ((0, 1), (1, 2), (2, 0))])
    longest = e.max(axis=0)
    mask = longest > max_edge * np.median(e)
    return mtri.Triangulation(x, y, triangles=t, mask=mask)
