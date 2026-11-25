"""Old vs new bound for the type-I seesaw of arXiv:2503.03825, using the old code itself.

"old file":  sample-nu-space-BOUND.m, written by "NEUTRINO LIMS + MET BOUND.nb" ({M_N, Y_nu, Sqrt[mb2]}).
"old code":  the same notebook calculation (its SM + N beta functions, threshold at M_N, lambda_eff =
             lambda + lambda_1, M_S = 100 M_N as in its scan), through the Python port in
             ../metastability-lifetime (metastability_lifetime.running.Running), with mu_I the first
             zero of lambda_eff above M_N, searched only where the running is known (up to M_S, or to
             where t2 or lambda reach 4 pi). Points without such a zero have no instability scale here.
"new":       this code (two-loop running, one-loop matching at M_N, MS-bar lambda = 0), Eq. (1) bound.

Run:  python3 examples/compare_2503.03825_oldcode.py path/to/sample-nu-space-BOUND.m
"""
from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "metastability-lifetime" / "src"))
from metastability_lifetime.running import Running  # noqa: E402

_spec = importlib.util.spec_from_file_location("cmp", HERE / "compare_2503.03825.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)
_spec = importlib.util.spec_from_file_location("nb", HERE / "compare_2503.03825_notebook.py")
nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb)

MS_OVER_MN = 100.0          # the notebook's scan: Ms2 = (100 M_N)^2


def old_code(MN: float, y: float) -> tuple[str, float, float]:
    """(status, mu_I, bound) of the notebook calculation, with mu_I the first zero of lambda_eff
    above M_N, searched only where the running is known: up to M_S, or up to where t2 or lambda
    reach 4 pi (the notebook's WhenEvent)."""
    from scipy.optimize import brentq
    r = Running(MS_OVER_MN * MN, MN, y)
    x = np.linspace(r.xM, r.x_max, 400)
    f = np.asarray(r.lam_eff(x), dtype=float)
    neg = np.flatnonzero(f < 0)
    if not neg.size:
        return ("nonperturbative" if not r.perturbative else "no zero below M_S"), math.nan, math.nan
    k = int(neg[0])
    if k == 0:
        return "unstable at M_N", math.nan, math.nan
    xin = brentq(lambda t: float(r.lam_eff(t)), x[k - 1], x[k], xtol=1e-12)
    mu = math.exp(xin)
    return "unstable", mu, math.sqrt(abs(r.beta_lam(xin)) * math.exp(-1.5)) * mu


def main(src: str) -> None:
    a = nb.read_m(Path(src))
    r_new = cmp.runner(True)
    rows = []
    for M, y, old_file in a:
        st_o, mu_o, mb_o = old_code(M, y)
        st_n, mu_n, mb_n = cmp.point(r_new, M, y)
        rows.append((M, y, old_file, st_o, mu_o, mb_o, st_n, mu_n, mb_n))
    out = cmp.OUT
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "compare_oldcode.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M_N_GeV", "Y_nu", "old_file_m_bound_GeV", "old_code_status", "old_code_mu_I_GeV",
                    "old_code_m_bound_GeV", "new_status", "new_mu_I_GeV", "new_m_bound_GeV"])
        for row in rows:
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])

    Ms, Ys = np.unique(a[:, 0]), np.unique(a[:, 1])
    shape = (len(Ms), len(Ys))                              # file order: M outer, Y inner
    col = lambda k: np.array([r[k] for r in rows], dtype=float).reshape(shape).T   # noqa: E731
    old_file, old_code_mb, new_mb = col(2), col(5), col(8)
    X, Y = np.meshgrid(Ms, Ys)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), sharey=True)
    lo, hi = 2, 9
    panels = ((old_file, "old: notebook file"),
              (old_code_mb, "old: notebook calculation (fixed root)"),
              (new_mb, "new: this code"))
    for ax, (Z, title) in zip(axes, panels):
        L = np.log10(Z)
        pc = ax.pcolormesh(X, Y, L, cmap="magma", vmin=lo, vmax=hi, shading="nearest", rasterized=True)
        cs = ax.contour(X, Y, L, levels=np.arange(3, 9), colors="white", linewidths=0.5)
        ax.clabel(cs, fmt="%d", fontsize=7)
        ax.set_title(title, loc="left", fontsize=9)
    fig.colorbar(pc, ax=axes[:3], pad=0.01).set_label(r"$\log_{10}(m_h^{\max}/\mathrm{GeV})$")
    ratio = np.log10(old_code_mb / new_mb)
    pc = axes[3].pcolormesh(X, Y, ratio, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
    fig.colorbar(pc, ax=axes[3], pad=0.02).set_label(r"$\log_{10}(\mathrm{old\ calculation}/\mathrm{new})$")
    axes[3].set_title("old calculation / new", loc="left", fontsize=9)
    for M, y, mu_p, m_p in cmp.BENCHMARKS:
        for ax in axes:
            ax.plot(M, y, "o", ms=5, mfc="none", mec="#2a78d6", mew=1.3)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$M_N$ [GeV]")
        ax.set_facecolor("#d6d5d0")
    axes[0].set_ylabel(r"$|Y_\nu|$")
    fig.text(0.01, -0.04, "Grey: no instability scale (old calculation: no zero of $\\lambda_{\\rm eff}$ below "
             "$M_S=100M_N$; new: non-perturbative). Circles: Table 3 benchmarks.", fontsize=8)
    fig.savefig(out / "compare_oldcode.pdf", dpi=200, bbox_inches="tight")

    # benchmarks
    with open(out / "benchmarks_oldcode.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M_N_GeV", "Y_nu", "table3_mu_I_GeV", "table3_m_crit_GeV", "old_code_mu_I_GeV",
                    "old_code_m_bound_GeV", "new_mu_I_GeV", "new_m_bound_GeV"])
        for M, y, mu_p, m_p in cmp.BENCHMARKS:
            _, mu_o, mb_o = old_code(M, y)
            _, mu_n, mb_n = cmp.point(r_new, M, y)
            w.writerow([M, y, mu_p, m_p, f"{mu_o:.4g}", f"{mb_o:.4g}", f"{mu_n:.4g}", f"{mb_n:.4g}"])
            print(f"M_N={M:.3g}, Y={y}: Table 3 mu_I={mu_p:.3g}, m={m_p:.3g} | old code mu_I={mu_o:.3g}, "
                  f"m={mb_o:.3g} | new mu_I={mu_n:.3g}, m={mb_n:.3g}")
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1])
