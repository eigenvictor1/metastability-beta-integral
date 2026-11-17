"""Run every model of a JSON file (one or many) and write a summary table.

Everything goes to results/<run>/ (<run> = the file's "name", default its file
name): summary.csv (+ .json) with one row per theory, the plots
running-<theory>.pdf and inscale-<theory>.pdf, and one data folder per theory.
<theory> is the model's name or a label from its representations; entries with
identical field content get "_2", "_3", ... appended.

Parallelism: with one model, the workers share its scan points; with several
models, each worker takes whole models. A model that fails is
reported in the summary ("error" column) and does not stop the others.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import Study, merge, spec_from_dict
from .output.layout import backup_files, write_json
from .paths import results_root as default_results_root
from .running.solver import STATUSES, IntegrationError
from .running.threshold import MassError

if TYPE_CHECKING:
    from .pipeline import Analysis
    from .theory.model import Model

log = logging.getLogger(__name__)


def compact_fields(model: Model) -> str:
    return " + ".join((f"{f.copies}x" if f.copies > 1 else "") + f"{f.name}{f.label()}"
                      for f in model.new_fields)


# Errors that mean "this model cannot be analysed as specified" (reported in the summary
# without a traceback); anything else is a bug and is logged with its traceback.
EXPECTED_ERRORS = (ValueError, KeyError, TypeError, MassError, IntegrationError, RuntimeError, OSError)


def process_model(entry: dict, index: int, results_root: str | None = None, workers: int | None = 1,
                  run_name: str | None = None) -> dict:
    """Analyse one model (runs in a worker process). Returns its summary row."""
    from .pipeline import run_analysis

    t0 = time.time()
    row: dict = {"index": index}
    try:
        spec = replace(spec_from_dict(entry), workers=workers)
        row.update(_model_columns(spec.model))
        a = run_analysis(spec, run_name=run_name, results_root=results_root)
        row.update(_result_columns(a))
    except EXPECTED_ERRORS as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        log.debug("model %d", index, exc_info=True)
    except Exception as exc:  # noqa: BLE001 - a bug in one model must not stop the study
        row["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("unexpected error in model %d (%s)", index, row.get("label", "?"))
    row["time_s"] = round(time.time() - t0, 2)
    return row


def _model_columns(m: Model) -> dict:
    return dict(name=m.name or "", label=m.label(), fields=compact_fields(m),
                yukawas=";".join(m.new_yukawa_names()), n_yukawas=len(m.new_yukawa_names()),
                anomaly_free=all(v == 0 for v in m.anomalies().values()),
                warnings=" | ".join(m.check()))


def _result_columns(a: Analysis) -> dict:
    row: dict = {}
    for k, r in enumerate(a.benchmarks, 1):
        row[f"bench{k}_mu_I_GeV"] = r.mu_I
        row[f"bench{k}_status"] = r.status
        if r.matching is not None:
            row[f"bench{k}_delta_lambda"] = r.matching.delta_lambda
    mu = a.scan.mu_I[np.isfinite(a.scan.mu_I)]
    st = np.asarray(a.scan.status).ravel()
    row.update(sm_mu_I_GeV=a.scan.sm_mu_I,
               scan_log10_mu_I_min=float(np.log10(mu.min())) if mu.size else None,
               scan_log10_mu_I_max=float(np.log10(mu.max())) if mu.size else None,
               **{f"scan_frac_{s}": float(np.mean(st == s)) for s in STATUSES},
               pyrate_agree=(a.report.agree if a.report is not None else None),
               folder=str(a.folder), error="")
    return row


def run_study(study: Study, results_root: str | Path | None = None, workers: int | None = None) -> Path:
    """Run every model; return the path of the summary CSV."""
    root = Path(results_root or default_results_root())
    workers = study.workers if workers is None else workers
    ncpu = workers or os.cpu_count() or 1
    entries = _unique_labels(study.entries)
    n = len(entries)
    rows = []
    if n == 1:
        log.info("%s: 1 model, scan on %d processes", study.name, ncpu)
        rows.append(_report(process_model(entries[0], 1, str(root), ncpu, study.name), n))
    elif ncpu > 1:
        nproc = min(n, ncpu)
        log.info("%s: %d models on %d processes", study.name, n, nproc)
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futs = [ex.submit(process_model, e, i, str(root), 1, study.name) for i, e in enumerate(entries, 1)]
            for f in as_completed(futs):
                rows.append(_report(f.result(), n))
    else:
        log.info("%s: %d models", study.name, n)
        for i, e in enumerate(entries, 1):
            rows.append(_report(process_model(e, i, str(root), 1, study.name), n))
    rows.sort(key=lambda r: r["index"])

    out = root / study.name
    out.mkdir(parents=True, exist_ok=True)
    for p in backup_files(out, ("summary.csv", "summary.json")):
        log.info("old summary moved to %s", p)
    path = out / "summary.csv"
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_json(out / "summary.json", {"name": study.name,
                                            "created": datetime.now().isoformat(timespec="seconds"),
                                            "models": study.entries, "summary": rows})
    failed = [r for r in rows if r.get("error")]
    log.info("done: %d ok, %d failed -> %s", n - len(failed), len(failed), path)
    return path


def _unique_labels(entries: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for e in entries:
        try:
            label = spec_from_dict(e).model.label()
        except EXPECTED_ERRORS:    # invalid entry: process_model will report it
            out.append(e)
            continue
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            e = {**e, "model": {**e["model"], "name": f"{label}_{seen[label]}"}}
        out.append(e)
    return out


def _report(row: dict, n: int) -> dict:
    if row.get("error"):
        log.warning("[%d/%d] %s failed: %s", row["index"], n, row.get("label", "?"), row["error"])
    else:
        log.info("[%d/%d] %s done (%.1f s)", row["index"], n, row["label"], row["time_s"])
    return row


def study_from_models(name: str, models: Iterable[Model], defaults: dict | None = None) -> Study:
    """Build a Study from Model objects generated in Python (e.g. by a classification rule)."""
    return Study(name, [merge(defaults or {}, m.to_dict()) for m in models], None)
