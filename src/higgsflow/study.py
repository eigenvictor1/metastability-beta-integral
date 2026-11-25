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
from .paths import short
from .running.solver import STATUSES, IntegrationError
from .running.threshold import MassError

if TYPE_CHECKING:
    from .pipeline import Analysis
    from .theory.model import Model

log = logging.getLogger(__name__)


def compact_fields(model: Model) -> str:
    """Short text of the new fields, e.g. 'L4(1,2,-1_2) + 3xN(1,1,0)' (summary column)."""
    return " + ".join((f"{f.copies}x" if f.copies > 1 else "") + f"{f.name}{f.label()}"
                      for f in model.new_fields)


# Errors that mean "this model cannot be analysed as specified" (reported in the summary
# without a traceback); anything else is a bug and is logged with its traceback.
HEADER_WIDTH = 64                # width of the separator line before each model
EXPECTED_ERRORS = (ValueError, KeyError, TypeError, MassError, IntegrationError, RuntimeError, OSError)


def process_model(entry: dict, index: int, results_root: str | None = None, workers: int | None = 1,
                  run_name: str | None = None, total: int | None = None, header: bool = True) -> dict:
    """Analyse one model (runs in a worker process). Returns its summary row.
    header: start with a separator line and the model's Yukawas (models run one after the other)."""
    from .pipeline import run_analysis

    t0 = time.time()
    row: dict = {"index": index}
    try:
        spec = replace(spec_from_dict(entry), workers=workers)
        row.update(_model_columns(spec.model))
        if header:
            _header(index, total, spec.model.label())
        a = run_analysis(spec, run_name=run_name, results_root=results_root, announce=False, details=header)
        row.update(_result_columns(a))
    except EXPECTED_ERRORS as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        log.debug("model %d", index, exc_info=True)
    except Exception as exc:  # noqa: BLE001 - a bug in one model must not stop the study
        row["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("unexpected error in model %d (%s)", index, row.get("label", "?"))
    row["time_s"] = round(time.time() - t0, 2)
    return row


def _announce(name: str, entries: list[dict], root: Path) -> None:
    """Log once, before the first model: where the results go and which beta functions are used."""
    from .pipeline import betas_text, fixed_text

    log.info("results folder: %s", short(root / name))
    lines: list[str] = []
    for e in entries:
        try:
            spec = spec_from_dict(e)
        except EXPECTED_ERRORS:
            continue
        lines += [x for x in [betas_text(spec), *fixed_text(spec.model)] if x not in lines]
    for line in lines:
        log.info("%s", line)


def _header(index: int, total: int | None, label: str) -> None:
    """A separator line in the log where a new model starts: ── [2/13] Sigma_X5 ───…"""
    head = f"── [{index}/{total or '?'}] {label} "
    log.info("\n%s", head + "─" * max(3, HEADER_WIDTH - len(head)), extra={"plain": True})


def _model_columns(m: Model) -> dict:
    """Summary columns that depend only on the model: names, fields, Yukawas, anomalies, warnings."""
    return dict(name=m.name or "", label=m.label(), fields=compact_fields(m),
                yukawas=";".join(m.new_yukawa_names()), n_yukawas=len(m.new_yukawa_names()),
                anomaly_free=all(v == 0 for v in m.anomalies().values()),
                warnings=" | ".join(m.check()))


def _result_columns(a: Analysis) -> dict:
    """Summary columns from an analysis: benchmark μ_I/status/δλ, scan range of μ_I,
    fraction of points per status, PyR@TE agreement and the data folder.
    """
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
    rows: list[dict] = []
    _announce(study.name, entries, root)
    if ncpu > 1:
        precompute_betas(entries, ncpu)
    # one model after the other, each scan on all cores: the beta functions are already
    # cached, so every model starts at once and shows its progress
    for i, e in enumerate(entries, 1):
        rows.append(_report(process_model(e, i, str(root), ncpu, study.name, n), n))
    rows.sort(key=lambda r: r["index"])

    out = root / study.name
    out.mkdir(parents=True, exist_ok=True)
    if backup_files(out, ("summary.csv", "summary.json")):
        log.info("old summary moved to backup/")
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
    log.info("done: %d ok, %d failed -> %s", n - len(failed), len(failed), short(path))
    return path


BETA_JOB_GB = 3.0     # memory budget per parallel beta computation (the largest models need ~3 GB)


def _beta_job(model: Model, loops: int) -> float:
    """Compute (and cache) the native betas of one theory; returns the time taken."""
    from .rge.engine import compute_betas

    t0 = time.time()
    compute_betas(model, loops)
    return time.time() - t0


def _available_gb() -> float | None:
    """Free memory in GB (None if unknown)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _Mem()
        m.dwLength = ctypes.sizeof(_Mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):  # type: ignore[attr-defined]
            return m.ullAvailPhys / 1e9
        return None
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def precompute_betas(entries: list[dict], ncpu: int) -> None:
    """Compute the native betas of every theory of a study that is not cached yet, in parallel
    (largest first, as many at once as cores and memory allow), with one line per theory."""
    from .pipeline import native_models
    from .rge.engine import is_cached

    jobs: dict[str, tuple[Model, int]] = {}
    for e in entries:
        try:
            spec = spec_from_dict(e)
            for m in native_models(spec):
                if m.hash() not in jobs and not is_cached(m):
                    jobs[m.hash()] = (m, spec.loops)
        except EXPECTED_ERRORS:      # invalid entry: process_model will report it
            continue
    if not jobs:
        return
    todo = sorted(jobs.values(), key=lambda j: -len(j[0].components()))
    gb = _available_gb()
    nproc = min(len(todo), ncpu, max(1, int(gb / BETA_JOB_GB)) if gb else max(1, ncpu // 2))
    log.info("computing the beta functions of %d theories on %d processes (cached afterwards)",
             len(todo), nproc)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futs = {ex.submit(_beta_job, m, loops): m for m, loops in todo}
        for k, f in enumerate(as_completed(futs), 1):
            m = futs[f]
            label = m.label() if m.new_fields else "SM"
            try:
                log.info("  betas [%d/%d] %s: %.0f s", k, len(todo), label, f.result())
            except Exception as exc:  # noqa: BLE001 - the model reports it again when it runs
                log.warning("  betas [%d/%d] %s failed: %s", k, len(todo), label, exc)
    log.info("beta functions done in %.0f s", time.time() - t0)


def _unique_labels(entries: list[dict]) -> list[dict]:
    """Give entries with identical labels distinct names (<label>_2, _3, ...) so folders don't clash."""
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
    """Log one finished model (with its error, if any); returns the row unchanged."""
    if row.get("error"):
        log.warning("[%d/%d] %s failed: %s", row["index"], n, row.get("label", "?"), row["error"])
    else:
        log.info("[%d/%d] %s done (%.1f s)", row["index"], n, row["label"], row["time_s"])
    return row


def study_from_models(name: str, models: Iterable[Model], defaults: dict | None = None) -> Study:
    """Build a Study from Model objects generated in Python (e.g. by a classification rule)."""
    return Study(name, [merge(defaults or {}, m.to_dict()) for m in models], None)
