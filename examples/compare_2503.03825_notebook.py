"""Compare our Eq. (1) bound with the notebook data of arXiv:2503.03825 (sample-nu-space-BOUND.m).

The .m file holds {M_N [GeV], |Y_nu|, Sqrt[mb2]}: the notebook ("NEUTRINO LIMS + MET BOUND.nb",
function Everything) stores the Eq. (1) bound on m_h itself, in GeV.
Run:  python examples/compare_2503.03825_notebook.py path/to/sample-nu-space-BOUND.m
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cmp", HERE / "compare_2503.03825.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)


def read_m(path: Path) -> np.ndarray:
    """Rows (M_N, Y_nu, value) of a Mathematica list exported with Put/Export."""
    s = path.read_text()
    s = s[s.index("{"):].replace("*^", "e")
    return np.array([float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", s)]).reshape(-1, 3)


def main(src: str) -> None:
    a = read_m(Path(src))
    r = cmp.runner(True)
    ours = np.array([cmp.point(r, M, y)[2] for M, y, _ in a])
    Ms, Ys = np.unique(a[:, 0]), np.unique(a[:, 1])
    shape = (len(Ms), len(Ys))                     # file order: M outer, Y inner
    theirs = a[:, 2].reshape(shape).T
    ours = ours.reshape(shape).T
    X, Y = np.meshgrid(Ms, Ys)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    lo, hi = 2, 18.5
    for ax, Z, title in ((axes[0], theirs, "old: notebook of 2503.03825"),
                         (axes[1], ours, "new: this code, Eq. (1) bound")):
        pc = ax.pcolormesh(X, Y, np.log10(Z), cmap="magma", vmin=lo, vmax=hi, shading="nearest", rasterized=True)
        cs = ax.contour(X, Y, np.log10(Z), levels=np.arange(3, 19, 2), colors="white", linewidths=0.5)
        ax.clabel(cs, fmt="%d", fontsize=7)
        ax.set_title(title, loc="left", fontsize=9)
    fig.colorbar(pc, ax=axes[:2], pad=0.01).set_label(r"$\log_{10}(m_h^{\max}/\mathrm{GeV})$")
    ratio = np.log10(theirs / ours)
    pc = axes[2].pcolormesh(X, Y, ratio, cmap="RdBu_r", vmin=-10, vmax=10, shading="nearest", rasterized=True)
    fig.colorbar(pc, ax=axes[2], pad=0.02).set_label(r"$\log_{10}(\mathrm{old}/\mathrm{new})$")
    axes[2].set_title("old / new (grey: no $\\mu_I$ in new)", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$M_N$ [GeV]")
        ax.set_facecolor("#d6d5d0")
    axes[0].set_ylabel(r"$|Y_\nu|$")
    out = cmp.OUT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "compare_notebook.pdf", dpi=200, bbox_inches="tight")
    np.savetxt(out / "compare_notebook.csv", np.column_stack([a[:, 0], a[:, 1], a[:, 2],
               np.array([cmp.point(r, M, y)[2] for M, y, _ in a])]),
               delimiter=",", header="M_N_GeV,Y_nu,old_m_bound_GeV,new_m_bound_GeV", comments="")
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1])
