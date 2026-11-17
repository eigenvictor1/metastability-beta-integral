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
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from ..running.boundary import M_PLANCK
from ..running.scan import ScanResult
from ..running.solver import RunResult
from ..settings import MAP_GRID, PLOT_DPI
from ..theory import couplings as cp

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#b9b8b2"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
STATUS_TEXT = {"unstable": "unstable", "stable": "stable up to $M_{Pl}$",
               "nonperturbative": "non-perturbative", "failed": "integration failed"}
STATUS_MARKER = {"nonperturbative": "^", "failed": "x"}
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 1))]


def pretty(name: str) -> str:
    """Axis / legend label of a coupling or scan parameter."""
    return r"$M$ [GeV]" if name == "M" else cp.latex(name)


def _style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)


def _example_label(params: dict, M: float) -> str:
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
    for st in ("nonperturbative", "failed"):
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
SHADED = (("stable", None, "stable up to $M_{Pl}$"),
          ("nonperturbative", "///", "non-perturbative before $\\mu_I$"),
          ("failed", "xx", "integration failed"))
MU_I_LABEL = r"$\log_{10}(\mu_I/\mathrm{GeV})$"
LEGEND_BOX = {"frameon": True, "facecolor": "white", "edgecolor": "none", "framealpha": 0.9}


def _swapped(scan: ScanResult | None) -> ScanResult | None:
    """Same scan with the two axes exchanged (arrays transposed)."""
    if scan is None:
        return None
    names = list(scan.axes)[::-1]
    return replace(scan, axes={n: scan.axes[n] for n in names}, mu_I=scan.mu_I.T,
                   status=scan.status.T, mu_stop=scan.mu_stop.T)


def plot_instability(path: Path, scan: ScanResult, overlay: ScanResult | None = None,
                     title: str | None = None) -> None:
    """μ_I against one parameter (curve) or two (colour map, M always on the x axis)."""
    names = list(scan.axes)
    if len(names) == 2 and names[1] == "M":
        scan, overlay = _swapped(scan) or scan, _swapped(overlay)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _style(ax)
    sm = np.log10(scan.sm_mu_I) if scan.sm_mu_I else None
    if len(scan.axes) == 1:
        _instability_1d(ax, scan, overlay, sm)
    else:
        _instability_2d(fig, ax, scan, overlay, sm)
    fixed = ", ".join((f"$M$ = {v:.3g} GeV" if k == "M" else f"{pretty(k)} = {v:.3g}")
                      for k, v in scan.fixed.items())
    ax.set_title((title or "Higgs instability scale") + (f"\n{fixed}" if fixed else ""),
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _instability_1d(ax, scan: ScanResult, overlay: ScanResult | None, sm: float | None) -> None:
    (n,) = scan.axes
    x = scan.axes[n]
    ax.plot(x, np.log10(scan.mu_I), color=SERIES[0], lw=2, marker="o", ms=3, label=r"$\mu_I$ (native betas)")
    if overlay is not None:
        ax.plot(x, np.log10(overlay.mu_I), color=INK, lw=1, ls="--", label=r"$\mu_I$ (PyR@TE betas)")
    if sm is not None:
        ax.axhline(sm, color=INK2, lw=1, ls=":", label=rf"SM: $\mu_I=10^{{{sm:.2f}}}$ GeV")
    dx = np.diff(x).mean() / 2 if len(x) > 1 else 0.5
    for key, hatch, lab in SHADED:
        first = True
        for xi, s in zip(x, scan.status):
            if s == key:
                ax.axvspan(xi - dx, xi + dx, color=NEUTRAL, alpha=0.35, hatch=hatch, lw=0,
                           label=lab if first else None)
                first = False
    ax.set_xlabel(pretty(n), color=INK)
    ax.set_ylabel(MU_I_LABEL, color=INK)
    if n == "M":
        ax.set_xscale("log")
    ax.legend(fontsize=8, **LEGEND_BOX)


def _instability_2d(fig, ax, scan: ScanResult, overlay: ScanResult | None, sm: float | None) -> None:
    nx, ny = scan.axes
    xa, ya = scan.axes[nx], scan.axes[ny]
    logx, logy = nx == "M", ny == "M"
    Z = np.log10(scan.mu_I)
    # smooth map: interpolate log10 μ_I on a fine grid (in log space for log axes)
    Xf, Yf, Zf = _refine(xa, ya, Z, logx, logy)
    pc = ax.pcolormesh(Xf, Yf, Zf, cmap=SEQ, shading="gouraud", rasterized=True)
    fig.colorbar(pc, ax=ax).set_label(MU_I_LABEL, color=INK)
    finite = Z[np.isfinite(Z)]
    levels = (np.arange(np.floor(finite.min()), np.ceil(finite.max()) + 1, 1)
              if finite.size and np.ptp(finite) > 0.2 else None)
    if levels is not None:
        cs = ax.contour(Xf, Yf, Zf, levels=levels, colors=INK, linewidths=0.6)
        ax.clabel(cs, fmt="%d", fontsize=7)
    if overlay is not None:
        Zo = np.log10(overlay.mu_I)
        if np.isfinite(Zo).sum() > 3:
            _, _, Zof = _refine(xa, ya, Zo, logx, logy)
            ax.contour(Xf, Yf, Zof, levels=levels if levels is not None else 5,
                       colors=SERIES[1], linewidths=0.8, linestyles="--")
    handles = _status_regions(ax, scan, *np.meshgrid(xa, ya, indexing="ij"))
    if overlay is not None:
        handles.append(plt.Line2D([], [], color=SERIES[1], ls="--", label="PyR@TE betas"))
    if sm is not None:
        handles.append(plt.Line2D([], [], color="none", label=rf"SM: $\mu_I=10^{{{sm:.2f}}}$ GeV"))
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="best", **LEGEND_BOX)
    ax.set_xlabel(pretty(nx), color=INK)
    ax.set_ylabel(pretty(ny), color=INK)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(visible=False, which="both")   # no grid on top of a colour map


def _status_regions(ax, scan: ScanResult, X: np.ndarray, Y: np.ndarray) -> list:
    """Hatched regions where the scan is stable / non-perturbative / failed; legend handles."""
    handles = []
    for key, hatch, lab in SHADED:
        mask = (scan.status == key).astype(float)
        if mask.any():
            ax.contourf(X, Y, mask, levels=[0.5, 1.5], colors=[NEUTRAL], alpha=0.45, hatches=[hatch])
            handles.append(Patch(facecolor=NEUTRAL, alpha=0.45, hatch=hatch, label=lab))
    return handles
