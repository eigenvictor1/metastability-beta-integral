"""Results folder layout.

results/<run>/
    summary.csv, summary.json          one row per theory of the run
    running-<theory>.pdf               plots of each theory
    inscale-<theory>.pdf
    inscale-min-<theory>.pdf
    inscale-lambdamin-<theory>.pdf
    inscale-trs-<theory>.pdf
    backup/                            older plots/summaries, renamed <stem>_<YYYY-MM-DD_HHMMSS><ext>
    <theory>/
        model.json
        betas/   betas_native.tex/.json, betas_pyrate.tex/.json, comparison.md
        data/    running_<k>.csv, running_SM.csv, scan.csv

<run> is the "name" of the JSON file (default: its file name); <theory> is the
model's name, or a label built from its representations.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


def plot_names(theory: str) -> tuple[str, str, str, str, str]:
    """File names of the plots of a theory: running-, inscale-, inscale-trs-, inscale-min- and
    inscale-lambdamin-<theory>.pdf."""
    return (f"running-{theory}.pdf", f"inscale-{theory}.pdf", f"inscale-trs-{theory}.pdf",
            f"inscale-min-{theory}.pdf", f"inscale-lambdamin-{theory}.pdf")


def prepare_folder(label: str, root) -> Path:
    """Create results/<run>/<label>/ with its betas/ and data/ subfolders; return it."""
    folder = Path(root) / label
    for sub in ("", "betas", "data"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    return folder


def backup_files(folder: Path, names) -> list[Path]:
    """Move existing files to backup/<stem>_<YYYY-MM-DD_HHMMSS><suffix> (file mtime)."""
    moved = []
    for name in names:
        p = folder / name
        if p.exists():
            stamp = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d_%H%M%S")
            dest_dir = folder / "backup"
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / f"{p.stem}_{stamp}{p.suffix}"
            k = 1
            while dest.exists():
                dest = dest_dir / f"{p.stem}_{stamp}_{k}{p.suffix}"
                k += 1
            shutil.move(str(p), str(dest))
            moved.append(dest)
    return moved


def write_json(path: Path, obj):
    """Write obj as indented JSON; numpy values and other objects are converted by _default."""
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    """JSON fallback: numpy scalars/arrays -> Python values, anything else -> str."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_run_csv(path: Path, run):
    """One run as CSV.

    First line: parameters, M, status, μ_I, failure message and threshold correction;
    then μ (GeV) and every coupling per row (NaN for new couplings below M).
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# params"] + [f"{k}={v}" for k, v in run.params.items()] + [f"M={run.M}",
                   f"status={run.status}", f"mu_I={run.mu_I}",
                   f"lam_min={run.lam_min}", f"mu_min={run.mu_min}"]
                   + ([f"message={run.message}"] if run.message else [])
                   + ([f"{k}={v}" for k, v in run.matching.as_dict().items()] if run.matching else []))
        w.writerow(["mu_GeV"] + run.couplings)
        for i in range(len(run.t)):
            w.writerow([f"{run.mu[i]:.8e}"] + [f"{run.x[k, i]:.10e}" for k in range(len(run.couplings))])


def write_scan_csv(path: Path, scan, overlay=None):
    """A scan as CSV: one row per grid point with μ_I, status, μ_stop and the minimum of λ
    (λ_min and its scale; empty where λ has no minimum), the lowest λ along the running
    (lam_low: λ_min, or λ where the running stops if still decreasing) (+ the PyR@TE columns
    if an overlay is given); the first line lists the fixed parameters and the SM μ_I.
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        names = list(scan.axes)
        head = names + ["mu_I_GeV", "status", "mu_stop_GeV", "lam_min", "mu_min_GeV", "lam_low"]
        if overlay is not None:
            head += ["mu_I_pyrate_GeV", "status_pyrate"]
        w.writerow(["# fixed"] + [f"{k}={v}" for k, v in scan.fixed.items()] + [f"SM_mu_I={scan.sm_mu_I}",
                                                                              f"SM_lam_min={scan.sm_lam_min}",
                                                                              f"SM_mu_min={scan.sm_mu_min}"])
        w.writerow(head)
        rows_o = list(overlay.rows()) if overlay is not None else None
        for k, (pt, mu, st, ms, lm, mm, ll) in enumerate(scan.rows()):
            row = [pt[n] for n in names] + [mu, st, ms, lm, mm, ll]
            if rows_o is not None:
                row += [rows_o[k][1], rows_o[k][2]]
            w.writerow(row)


def write_boundaries_csv(path: Path, scan) -> None:
    """Region boundaries of a scan found by bisection (one row per grid line of the other
    parameter): where the status changes along the Yukawa axis, and from what to what."""
    rows = list(scan.boundaries())
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        others = list(rows[0][0])
        w.writerow(others + [f"{scan.edge_axis}_boundary", "status_below", "status_above"])
        for other, e, s1, s2 in rows:
            w.writerow([other[n] for n in others] + [e, s1, s2])


def write_curves_csv(path: Path, scan) -> None:
    """The curves traced point by point (running/curves.py): one row per point, with the curve
    number, its kind ('lam_low': λ_low = level, i.e. the stable boundary at 0 and the
    metastability line; 'mu_I': log10(μ_I/GeV) = level) and why it ends at each side."""
    curves = getattr(scan, "curves", None)
    if not curves:
        return
    yname = next((n for n in scan.axes if n != "M"), "y")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["curve", "kind", "level", yname, "M", "log10_mu_limit", "ends"])
        for k, c in enumerate(curves):
            for y, M, a in zip(c.y, c.M, c.aux):
                w.writerow([k, c.kind, c.level, y, M, a, "/".join(c.ends)])


def write_samples_csv(path: Path, res) -> None:
    """The trace-plot samples: coupling values, Tr(YY†), Tr((YY†)²), μ_I and status."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# M", res.M, f"SM_mu_I={res.sm_mu_I}"])
        w.writerow(list(res.couplings) + ["Tr_YYdag", "Tr_YYdag2", "mu_I_GeV", "status"])
        for v, t2, t4, mu, st in zip(res.values, res.tr2, res.tr4, res.mu_I, res.status):
            w.writerow(list(v) + [t2, t4, mu, st])
