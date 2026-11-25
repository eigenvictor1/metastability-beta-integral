"""Type-I seesaw compared with Enguita, Gavela and Steingasser, arXiv:2503.03825 (Fig. 7, Table 3).

Their type-I set-up: two right-handed neutrinos with degenerate mass M_N, only one of which
couples to the leptons (one column of Y_nu), with strength |Y_nu|^2 = sum_a |Y_a2|^2. The
uncoupled one neither runs nor enters the one-loop matching, and only y_t is kept among the
SM Yukawas, so here this is the 1N model with y = |Y_nu| (same normalisation: the new term in
beta_lambda is (4 lambda |Y|^2 - 2 |Y|^4)/(16 pi^2) in both). Not reproduced: their mass
structure (degenerate pair), their three-loop SM running and their threshold corrections.

For each (M_N, |Y_nu|) it computes mu_I (lambda = 0, MS-bar, two-loop running, one-loop
matching at M_N) and their Eq. (1) bound  m_h^2 < |beta_lambda(mu_I)| e^{-3/2} mu_I^2,
with beta_lambda = d lambda / d ln mu.

Run:  python examples/compare_2503.03825.py      (writes results/compare_2503.03825/)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from higgsflow.rge.engine import compute_betas
from higgsflow.running.solver import Runner
from higgsflow.running.threshold import ThresholdMatching
from higgsflow.theory.model import Model

OUT = Path(__file__).resolve().parent.parent / "results" / "compare_2503.03825"
COUPLING = "yNL3"
# Table 3 of arXiv:2503.03825: M_N [GeV], |Y_nu|, mu_I [GeV], m_crit [GeV]
BENCHMARKS = [(5.1e3, 1.8, 25.5e3, 5.2e3), (6.15e3, 2.2, 19e3, 5.6e3), (8.0e3, 1.8, 39.8e3, 7.8e3)]
M_GRID = np.logspace(3, 7, 61)
Y_GRID = np.linspace(0.2, 3.5, 67)


def runner(one_loop: bool = True) -> Runner:
    m = Model.from_dict({"fields": ["N:(1,1,0)"], "include_sm_mixing": True, "exclude": ["yNL1", "yNL2"]})
    return Runner(compute_betas(m), compute_betas(Model()), 2, ThresholdMatching(m) if one_loop else None, model=m)


def point(r: Runner, M: float, y: float) -> tuple[str, float, float]:
    """(status, mu_I, Eq. (1) bound on m_h); NaN where there is no instability scale."""
    res = r.run({COUPLING: y}, M)
    if res.status != "unstable":
        return res.status, np.nan, np.nan
    x = res.x[:, -1]
    beta = float(r.f_full(np.log(res.mu_I), x)[res.couplings.index("lam")])
    return res.status, res.mu_I, float(np.sqrt(abs(beta) * np.exp(-1.5)) * res.mu_I)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r1, r0 = runner(True), runner(False)

    # benchmarks: theirs against ours (one-loop and tree-level matching)
    with open(OUT / "benchmarks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M_N_GeV", "Y_nu", "old_mu_I_GeV", "old_m_crit_GeV",
                    "new_mu_I_GeV", "new_m_bound_GeV", "new_mu_I_tree_matching_GeV", "new_m_bound_tree_matching_GeV"])
        for M, y, mu_p, m_p in BENCHMARKS:
            _, mu1, mb1 = point(r1, M, y)
            _, mu0, mb0 = point(r0, M, y)
            w.writerow([M, y, mu_p, m_p, f"{mu1:.4g}", f"{mb1:.4g}", f"{mu0:.4g}", f"{mb0:.4g}"])
            print(f"M_N={M:.3g}, Y={y}: mu_I {mu_p:.3g} (old) vs {mu1:.3g} (new; tree matching {mu0:.3g});"
                  f"  m_bound {m_p:.3g} vs {mb1:.3g} GeV")

    # scan of the (M_N, |Y_nu|) plane
    shape = (len(Y_GRID), len(M_GRID))
    st, mu, mb = np.empty(shape, dtype=object), np.full(shape, np.nan), np.full(shape, np.nan)
    for i, y in enumerate(Y_GRID):
        for j, M in enumerate(M_GRID):
            st[i, j], mu[i, j], mb[i, j] = point(r1, M, y)
    with open(OUT / "scan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M_N_GeV", "Y_nu", "status", "mu_I_GeV", "m_bound_GeV"])
        for i, y in enumerate(Y_GRID):
            for j, M in enumerate(M_GRID):
                w.writerow([f"{M:.6g}", f"{y:.6g}", st[i, j], f"{mu[i, j]:.6g}", f"{mb[i, j]:.6g}"])

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    X, Yg = np.meshgrid(M_GRID, Y_GRID)
    for ax, Z, lab, title in ((axes[0], mu, r"$\log_{10}(\mu_I/\mathrm{GeV})$", r"instability scale $\mu_I$"),
                              (axes[1], mb, r"$\log_{10}(m_h^{\max}/\mathrm{GeV})$",
                               r"Eq. (1) bound: $m_h^2<|\beta_\lambda|e^{-3/2}\mu_I^2$")):
        L = np.log10(Z)
        pc = ax.pcolormesh(X, Yg, L, cmap="magma", shading="gouraud", rasterized=True)
        fig.colorbar(pc, ax=ax, pad=0.02).set_label(lab)
        cs = ax.contour(X, Yg, L, levels=np.arange(3, 12), colors="black", linewidths=0.6)
        cs.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="white", alpha=0.5)])
        ax.clabel(cs, fmt="%d", fontsize=7)
        for key, face, hatch in (("stable", "#9fd39a", None), ("nonperturbative", "#d6d5d0", "///")):
            region = np.where(st == key, 1.0, np.nan)
            if np.isfinite(region).any():
                ax.contourf(X, Yg, region, levels=[0, 2], colors=[face], hatches=[hatch])
        for M, y, mu_p, m_p in BENCHMARKS:
            ax.plot(M, y, "o", ms=6, mfc="white", mec="#2a78d6", mew=1.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$M_N$ [GeV]")
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].set_ylabel(r"$|Y_\nu|$")
    axes[0].plot([], [], "o", ms=6, mfc="white", mec="#2a78d6", mew=1.5, label="old: benchmarks of 2503.03825")
    axes[0].fill_between([], [], color="#9fd39a", label="no instability below $M_{Pl}$")
    axes[0].fill_between([], [], color="#d6d5d0", hatch="///", label=r"non-perturbative before $\mu_I$")
    axes[0].legend(fontsize=7, loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    fig.suptitle("new: type-I seesaw, one coupled singlet $N(1,1,0)$ (as in arXiv:2503.03825): "
                 "two-loop running, one-loop matching at $M_N$", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "compare_2503.03825.pdf", dpi=200)
    print("written to", OUT)


if __name__ == "__main__":
    main()
