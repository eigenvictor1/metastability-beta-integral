"""The two result plots (PDF).

running-<theory>.pdf   λ(μ) for example parameter points (+ SM), and the other couplings.
inscale-<theory>.pdf   μ_I as a function of one or two new-physics parameters.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.tri as mtri
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Patch

from ..running.boundary import M_PLANCK
from ..running.scan import SampleResult, ScanResult
from ..running.solver import RunResult
from ..running.traces import TR2, TR4, TR4R
from ..settings import MAP_GRID, PLOT_DPI
from ..theory import couplings as cp

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#b9b8b2"
HATCH = "#8a8984"                                                        # hatch lines of the regions
REGION = tuple(0.45 * np.array(matplotlib.colors.to_rgb(NEUTRAL)) + 0.55)   # NEUTRAL at 45% on white, opaque
SEQ = matplotlib.colormaps["magma"]           # colour map of the instability-scale maps (same as seaborn's "magma")
STATUS_TEXT = {"unstable": "unstable", "stable": "stable up to $M_{Pl}$",
               "landau_g2": "stable, $g_2$ Landau pole",
               "nonperturbative": "non-perturbative", "failed": "integration failed"}
STATUS_MARKER = {"landau_g2": "s", "nonperturbative": "^", "failed": "x"}
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
    """λ(μ) for the SM and each example, with μ_I and status markers."""
    ax.plot(_lg(sm_run), sm_run.coupling(cp.QUARTIC), color=INK2, lw=2, label="SM")
    if sm_run.mu_I:
        ax.plot(np.log10(sm_run.mu_I), 0, "o", ms=8, mfc="white", mec=INK2, mew=2)
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
    for st in ("landau_g2", "nonperturbative", "failed"):
        if st in seen:
            ax.plot([], [], STATUS_MARKER[st], ms=8, color=INK2, mew=2, label=STATUS_TEXT[st])
    ax.set_ylabel(r"$\lambda(\mu)$", color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    ax.set_title(title or "Running of the Higgs quartic", color=INK, fontsize=11, loc="left")


def _end_marker(ax, r: RunResult, col: str) -> None:
    """Circle at μ_I, or the status marker where a non-perturbative/failed run stops."""
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
    ax.set_ylabel("Yukawa couplings", color=INK)
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
SHADED = (("stable", None, "Stable", STABLE_GREEN),
          ("landau_g2", "\\\\", "stable, $g_2$ Landau pole", STABLE_GREEN),
          ("nonperturbative", "///", "non-perturbative", None),
          ("failed", "xx", "integration failed", None),
          ("unscanned", "..", "outside the scan", None))
MU_I_LABEL = r"$\log_{10}(\mu_I/\mathrm{GeV})$"
LEGEND_BOX = {"frameon": True, "facecolor": "white", "edgecolor": "none", "framealpha": 0.9}


def _swapped(scan: ScanResult | None) -> ScanResult | None:
    """Same scan with the two axes exchanged (arrays transposed)."""
    if scan is None:
        return None
    names = list(scan.axes)[::-1]
    return replace(scan, axes={n: scan.axes[n] for n in names}, mu_I=scan.mu_I.T,
                   status=scan.status.T, mu_stop=scan.mu_stop.T,
                   edges=None if scan.edges is None else scan.edges.T)


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
        scan = replace(scan, axes={k: (w[start:stop] if k == n else w) for k, w in scan.axes.items()},
                       mu_I=scan.mu_I[sl], status=scan.status[sl], mu_stop=scan.mu_stop[sl],
                       edges=None if scan.edges is None else scan.edges[sl])
    return scan


def _extended(scan: ScanResult, limits: Limits) -> ScanResult:
    """Extend each limited axis of a 2D scan to its limits, so that all plots with the same
    limits have the same axes.

    The added grid line at the upper end (larger couplings than scanned) is shown as
    non-perturbative, unless the outermost scanned line is stable; there it is not computed.
    At the lower end it takes the status of the outermost line if that is non-perturbative or
    stable, and 'unscanned' otherwise. Added lines have no μ_I.
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
                st = np.where(np.isin(edge, ("stable", "landau_g2")), edge, "nonperturbative").astype(object)
            else:
                st = np.where(np.isin(edge, ("nonperturbative", "stable", "landau_g2")), edge,
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
            scan = replace(scan, axes={k2: (ax_vals if k2 == n else w) for k2, w in scan.axes.items()},
                           mu_I=add(scan.mu_I, nan), status=add(scan.status, st), mu_stop=add(scan.mu_stop, nan),
                           edges=edges)
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


def _inside(t: np.ndarray, segs: list[tuple[float, float, str]], key: str, far: float) -> np.ndarray:
    """Signed distance (along the line) to the region with status ``key``: > 0 inside, < 0 outside."""
    with np.errstate(invalid="ignore"):
        d = np.full(len(t), -far)
        for lo, hi, st in segs:
            if st == key:
                d = np.maximum(d, np.minimum(t - lo, hi - t))
    return np.clip(d, -far, far)


def _region_fields(scan: ScanResult, n: int = MAP_GRID) -> dict[str, np.ndarray]:
    """For each status, a field on the fine n x n grid (same grid as _refine) that is > 0 inside
    the region and crosses zero at its boundary.

    Along the bisection axis the boundary is the refined one; between grid lines of the other
    axis the fields are interpolated linearly, so the boundary is a smooth curve instead of
    the staircase of the grid cells.
    """
    names = list(scan.axes)
    a = names.index(scan.edge_axis) if scan.edge_axis in names else len(names) - 1
    edges = scan.edges if scan.edges is not None else np.full(scan.status.shape, np.nan)
    st, ed = scan.status, edges
    if len(names) == 1:
        st, ed = st[None, :], ed[None, :]
    elif a == 0:                                   # put the bisection axis last
        st, ed = st.T, ed.T
    along = names[a]
    ta = _t(scan.axes[along], along in LOG_AXES)
    with np.errstate(invalid="ignore"):
        eda = _t(ed, along in LOG_AXES)
    fa = np.linspace(ta[0], ta[-1], n)
    far = 10 * abs(ta[-1] - ta[0]) + 1
    fields = {}
    for key in {str(x) for x in scan.status.flat}:
        cols = np.array([_inside(fa, _segments(ta, srow, erow), key, far) for srow, erow in zip(st, eda)])
        if len(names) == 1:
            fields[key] = cols[0]
            continue
        other = names[1 - a]
        to = _t(scan.axes[other], other in LOG_AXES)
        fo = np.linspace(to[0], to[-1], n)
        F = np.array([np.interp(fo, to, cols[:, i]) for i in range(n)]).T     # (other, along)
        fields[key] = F.T if a == 0 else F
    return fields


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
                     title: str | None = None, limits: Limits | None = None) -> None:
    """μ_I against one parameter (curve) or two (colour map, M always on the x axis).

    limits: {axis name: (lowest, highest)} to show, e.g. {TR4: (1e-3, 1e3)}; other axes show
    the whole scan.
    """
    limits = limits or {}
    names = list(scan.axes)
    if len(names) == 2 and names[1] == "M":
        scan, overlay = _swapped(scan) or scan, _swapped(overlay)
    scan = _extended(_cropped(scan, limits), limits)
    overlay = _extended(_cropped(overlay, limits), limits) if overlay is not None else None
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _style(ax)
    sm = np.log10(scan.sm_mu_I) if scan.sm_mu_I else None
    if len(scan.axes) == 1:
        _instability_1d(ax, scan, overlay, sm, limits)
    else:
        _instability_2d(fig, ax, scan, overlay, sm, limits)
    fixed = ", ".join((f"$M$ = {v:.3g} GeV" if k == "M" else f"{pretty(k)} = {v:.3g}")
                      for k, v in scan.fixed.items())
    ax.set_title((title or "Higgs instability scale") + (f"\n{fixed}" if fixed else ""),
                 color=INK, fontsize=11, loc="left")
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


def _instability_2d(fig, ax, scan: ScanResult, overlay: ScanResult | None, sm: float | None,
                    limits: Limits) -> None:
    """Two scanned parameters: colour map of log10 μ_I (interpolated, with integer contours),
    PyR@TE contours if given, and hatched regions for stable / non-perturbative / failed points.
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
    fields = _region_fields(scan)
    unstable = fields.get("unstable", np.full((MAP_GRID, MAP_GRID), -1.0))
    # smooth map: interpolate log10 μ_I on a fine grid (in log space for log axes), up to the
    # refined region boundaries
    Xf, Yf, Zf = _refine(xa, ya, _fill_nan(Z), logx, logy)
    # the colour map continues under the (opaque) status regions, so its edge is their smooth
    # boundary; contour lines stop at the boundary
    pc = ax.pcolormesh(Xf, Yf, Zf, cmap=SEQ, shading="gouraud", rasterized=True)
    Zf = np.where(unstable > 0, Zf, np.nan)
    fig.colorbar(pc, ax=ax).set_label(MU_I_LABEL, color=INK)
    finite = Z[np.isfinite(Z)]
    levels = (np.arange(np.floor(finite.min()), np.ceil(finite.max()) + 1, 1)
              if finite.size and np.ptp(finite) > 0.2 else None)
    if levels is not None:
        cs = ax.contour(Xf, Yf, Zf, levels=levels, colors=INK, linewidths=0.6)
        # light halo around the lines, so they are visible on the dark end of the map
        cs.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="white", alpha=0.5)])
        _label_contours(ax, cs)
    if overlay is not None:
        Zo = np.log10(overlay.mu_I)
        if np.isfinite(Zo).sum() > 3:
            _, _, Zof = _refine(xa, ya, _fill_nan(Zo), logx, logy)
            Zof = np.where(unstable > 0, Zof, np.nan)
            ax.contour(Xf, Yf, Zof, levels=levels if levels is not None else 5,
                       colors=SERIES[1], linewidths=0.8, linestyles="--")
    ta = _t(scan.axes[scan.edge_axis or ny], (scan.edge_axis or ny) in LOG_AXES)
    handles = _status_regions(ax, fields, Xf, Yf, overlap=0.5 * abs(ta[-1] - ta[0]) / MAP_GRID)
    if overlay is not None:
        handles.append(plt.Line2D([], [], color=SERIES[1], ls="--", label="PyR@TE betas"))
    if sm is not None:
        handles.append(plt.Line2D([], [], color="none", label=rf"SM: $\mu_I=10^{{{sm:.2f}}}$ GeV"))
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="best", **LEGEND_BOX)
    ax.set_xlabel(pretty(nx), color=INK)
    ax.set_ylabel(pretty(ny), color=INK)
    ax.grid(visible=False, which="both")   # no grid on top of a colour map


def _label_contours(ax, cs, margin: float = 0.06, min_gap: float = 0.09) -> None:
    """Readable contour labels: one per level, placed along its longest line.

    Each label goes at the middle of the line (or at 1/3, 2/3, ... if that spot is too
    close to the frame or to another label), so that labels don't pile up where the
    contours bunch together. Labels are drawn larger, with a white halo, so they can
    be read on dark parts of the colour map.
    """
    fig_to_axes = ax.transData + ax.transAxes.inverted()
    placed: list[np.ndarray] = []
    positions, levels = [], []
    for level, segs in zip(cs.levels, cs.allsegs):
        segs = [sg for sg in segs if len(sg) > 1]
        if not segs:
            continue
        seg = max(segs, key=lambda sg: np.ptp(fig_to_axes.transform(sg), axis=0).sum())
        pts = fig_to_axes.transform(seg)                      # line in axes coordinates (0..1)
        arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        for frac in (0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9):
            i = int(np.searchsorted(arc, frac * arc[-1]))
            p = pts[min(i, len(pts) - 1)]
            inside = np.all((p > margin) & (p < 1 - margin))
            if inside and all(np.hypot(*(p - q)) > min_gap for q in placed):
                placed.append(p)
                positions.append(tuple(seg[min(i, len(seg) - 1)]))
                levels.append(level)
                break
    if not positions:
        return
    texts = ax.clabel(cs, levels=levels, manual=positions, fmt="%d", fontsize=9, inline=True, inline_spacing=6)
    for t in texts:
        t.set_color(INK)
        t.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")])


def _status_regions(ax, fields: dict[str, np.ndarray], Xf: np.ndarray, Yf: np.ndarray,
                    overlap: float = 0.0) -> list:
    """Regions where the scan is stable (green) or non-perturbative / failed / unscanned
    (grey, hatched); returns their legend handles.

    Each region is the area where its field (see _region_fields) is positive. It is opaque and
    drawn on top of the colour map (which continues underneath), from -overlap, so that its
    smooth boundary is the edge of the map: no gaps or steps from the fine grid.
    """
    handles = []
    for key, hatch, lab, colour in SHADED:
        F = fields.get(key)
        if F is None or not (F > 0).any():
            continue
        if key == "stable" and "landau_g2" in fields:
            # one green area for both stable regions (the Landau-pole one is hatched on top),
            # so that no strip of the colour map shows between them
            F = np.maximum(F, fields["landau_g2"])
        top = float(F.max()) + 1.0
        face = colour or REGION
        with matplotlib.rc_context({"hatch.color": HATCH, "hatch.linewidth": 0.8}):
            ax.contourf(Xf, Yf, F, levels=[-overlap, top], colors=[face], hatches=[hatch], zorder=2.5,
                        antialiased=True)
        handles.append(Patch(facecolor=face, edgecolor=HATCH, lw=0, hatch=hatch, label=lab))
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
                             ("landau_g2", "s", "stable, $g_2$ Landau pole"), ("failed", "x", "integration failed")):
        sel = ok & (res.status == key)
        if sel.any():
            ax.scatter(res.tr2[sel], res.tr4[sel] ** 0.25, marker=marker, s=12, color=NEUTRAL, lw=0.6 if key == "failed" else 0)
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
