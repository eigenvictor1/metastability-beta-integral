"""Results folder layout.

results/<run>/
    summary.csv, summary.json          one row per theory of the run
    running-<theory>.pdf               plots, one pair per theory
    inscale-<theory>.pdf
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


def plot_names(theory: str) -> tuple[str, str]:
    return f"running-{theory}.pdf", f"inscale-{theory}.pdf"


def prepare_folder(label: str, root) -> Path:
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
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_run_csv(path: Path, run):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# params"] + [f"{k}={v}" for k, v in run.params.items()] + [f"M={run.M}",
                   f"status={run.status}", f"mu_I={run.mu_I}"]
                   + ([f"message={run.message}"] if run.message else [])
                   + ([f"{k}={v}" for k, v in run.matching.as_dict().items()] if run.matching else []))
        w.writerow(["mu_GeV"] + run.couplings)
        for i in range(len(run.t)):
            w.writerow([f"{run.mu[i]:.8e}"] + [f"{run.x[k, i]:.10e}" for k in range(len(run.couplings))])


def write_scan_csv(path: Path, scan, overlay=None):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        names = list(scan.axes)
        head = names + ["mu_I_GeV", "status", "mu_stop_GeV"]
        if overlay is not None:
            head += ["mu_I_pyrate_GeV", "status_pyrate"]
        w.writerow(["# fixed"] + [f"{k}={v}" for k, v in scan.fixed.items()] + [f"SM_mu_I={scan.sm_mu_I}"])
        w.writerow(head)
        rows_o = list(overlay.rows()) if overlay is not None else None
        for k, (pt, mu, st, ms) in enumerate(scan.rows()):
            row = [pt[n] for n in names] + [mu, st, ms]
            if rows_o is not None:
                row += [rows_o[k][1], rows_o[k][2]]
            w.writerow(row)
