"""End-to-end analysis of one model: betas -> runs -> results/<run>/.

    run_analysis(spec)
      ├─ compute_systems   native and/or PyR@TE beta functions (+ comparison)
      ├─ build_runners     one Runner per beta source, with the threshold matching
      ├─ run_points        examples, SM reference, scan, benchmarks
      └─ write_results     plots, data, betas, model.json
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__
from .config import RunSpec
from .output.layout import (
    backup_files,
    plot_names,
    prepare_folder,
    write_json,
    write_run_csv,
    write_scan_csv,
)
from .output.plots import plot_instability, plot_running
from .paths import results_root as default_results_root
from .pyrate.compare import ComparisonReport, compare, map_conventions
from .rge.engine import compute_betas
from .rge.system import BetaSystem
from .running.scan import NO_NEW_PHYSICS, ScanResult, resolve_params, run_scan
from .running.solver import Runner, RunResult
from .running.threshold import MassError, ThresholdMatching
from .theory import couplings as cp
from .theory.model import Model

log = logging.getLogger(__name__)


@dataclass
class Analysis:
    """Everything an analysis produces (also returned to the caller)."""
    spec: RunSpec                         # the prepared spec (links expanded, defaults filled in)
    folder: Path                          # results/<run>/<theory>/  (data, betas, model.json)
    run_folder: Path                      # results/<run>/  (plots, summary)
    systems: dict[str, BetaSystem]
    runner: Runner                        # main runner, reusable for further points
    threshold_used: str                   # "one-loop", "tree" or "tree (<reason>)"
    runs: list[RunResult]                 # the examples
    sm_run: RunResult
    scan: ScanResult
    benchmarks: list[RunResult] = field(default_factory=list)
    overlay_runs: list[RunResult] = field(default_factory=list)   # PyR@TE betas (betas="both")
    overlay_scan: ScanResult | None = None
    report: ComparisonReport | None = None


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_analysis(spec: RunSpec, *, run_name: str | None = None, results_root: str | Path | None = None,
                 pyrate_dir: str | Path | None = None) -> Analysis:
    """Compute betas, run examples, scan and benchmarks, write plots + data.

    run_name: results go to results/<run_name>/ (default: the model's label); the plots are
    running-<theory>.pdf and inscale-<theory>.pdf there, the data in <theory>/.
    """
    spec = spec.prepared()
    model = spec.model
    for w in model.check():
        log.warning(w)
    run_folder = Path(results_root or default_results_root()) / (run_name or model.label())
    folder = prepare_folder(model.label(), run_folder)
    log.info("results folder: %s", run_folder)
    log.info("Yukawa couplings: %s", "; ".join(model.yukawa_invariants()))

    systems, sm, report = compute_systems(model, spec.betas, spec.loops, pyrate_dir)
    runners, threshold_used = build_runners(spec, systems, sm)
    main = "native" if "native" in runners else "pyrate"
    a = Analysis(spec, folder, run_folder, systems, runners[main], threshold_used,
                 runs=run_points(runners[main], spec.examples or [], spec),
                 sm_run=runners[main].run({}, NO_NEW_PHYSICS, spec.inputs),
                 scan=_scan(runners[main], spec),
                 benchmarks=run_points(runners[main], spec.benchmarks, spec),
                 report=report)
    if spec.betas == "both":
        a.overlay_runs = run_points(runners["pyrate"], spec.examples or [], spec)
        a.overlay_scan = _scan(runners["pyrate"], spec)

    write_results(a)
    log.info("done: %s", folder)
    return a


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

def compute_systems(model: Model, betas: str, loops: int, pyrate_dir: str | Path | None = None
                    ) -> tuple[dict[str, BetaSystem], dict[str, BetaSystem], ComparisonReport | None]:
    """Return ({source: BetaSystem}, {source: SM BetaSystem}, report or None).

    With betas='both' the PyR@TE systems are re-expressed in the native Yukawa
    normalisation, so that both run from identical boundary conditions.
    """
    systems, sm = {}, {}
    if betas in ("native", "both"):
        log.info("computing native two-loop betas")
        systems["native"] = compute_betas(model, loops)
        sm["native"] = compute_betas(Model(), loops)
    if betas in ("pyrate", "both"):
        from .pyrate.bridge import run_pyrate
        log.info("computing betas with PyR@TE")
        systems["pyrate"] = run_pyrate(model, loops, pyrate_dir=pyrate_dir)
        sm["pyrate"] = run_pyrate(Model(), loops, pyrate_dir=pyrate_dir)
    if not systems:
        raise ValueError(f"unknown beta source {betas!r}")
    report = None
    if betas == "both":
        report = compare(systems["native"], systems["pyrate"])
        _, _, systems["pyrate"] = map_conventions(systems["native"], systems["pyrate"])
        _, _, sm["pyrate"] = map_conventions(sm["native"], sm["pyrate"])
        log.info("native vs PyR@TE agreement: %s (Yukawa map %s)",
                 "YES" if report.agree else "NO", report.calibration)
        if not report.agree:
            log.warning("beta functions disagree: see betas/comparison.md")
    return systems, sm, report


def build_runners(spec: RunSpec, systems: dict[str, BetaSystem], sm: dict[str, BetaSystem]
                  ) -> tuple[dict[str, Runner], str]:
    """One Runner per beta source; returns them and the matching actually used."""
    matching, used = None, "tree"
    if spec.threshold == "one-loop" and spec.model.new_fields:
        try:
            matching, used = ThresholdMatching(spec.model), "one-loop"
        except MassError as exc:
            log.warning("no one-loop threshold matching, using tree level: %s", exc)
            used = f"tree ({exc})"
    runners = {s: Runner(systems[s], sm[s], spec.loops, matching, spec.numerics) for s in systems}
    return runners, used


def run_points(runner: Runner, points: list[dict[str, float]], spec: RunSpec) -> list[RunResult]:
    """Run parameter points (examples or benchmarks); unset parameters take spec.fixed."""
    runs = []
    for p in points:
        pt = {**spec.fixed, **p}
        vals, M = resolve_params(pt, spec.links, spec.M)
        r = runner.run(vals, M, spec.inputs)
        r.params = {k: v for k, v in pt.items() if k != "M"}
        runs.append(r)
    return runs


def _scan(runner: Runner, spec: RunSpec) -> ScanResult:
    return run_scan(runner, spec.scan, spec.fixed, spec.links, spec.inputs, spec.M, workers=spec.workers)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_results(a: Analysis) -> None:
    if a.spec.plots:
        _write_plots(a)
    _write_data(a)
    _write_betas(a)
    _write_model_json(a)


def _write_plots(a: Analysis) -> None:
    theory = a.spec.model.label()
    running, inscale = plot_names(theory)
    for p in backup_files(a.run_folder, (running, inscale)):
        log.info("old plot moved to %s", p)
    plot_running(a.run_folder / running, a.runs, a.sm_run, a.overlay_runs or None,
                 title=f"{theory}: running of λ")
    plot_instability(a.run_folder / inscale, a.scan, a.overlay_scan,
                     title=f"{theory}: instability scale")


def _write_data(a: Analysis) -> None:
    data = a.folder / "data"
    for k, r in enumerate(a.runs, 1):
        write_run_csv(data / f"running_{k}.csv", r)
    write_run_csv(data / "running_SM.csv", a.sm_run)
    write_scan_csv(data / "scan.csv", a.scan, a.overlay_scan)


def _write_betas(a: Analysis) -> None:
    betas = a.folder / "betas"
    for src, bs in a.systems.items():
        (betas / f"betas_{src}.tex").write_text(bs.latex(), encoding="utf-8")
        write_json(betas / f"betas_{src}.json", bs.to_json())
    if a.report is None or a.overlay_scan is None:
        return
    rep = a.report
    rep.mu_I = {f"example {k}": f"native {r.mu_I}, PyR@TE {o.mu_I}"
                for k, (r, o) in enumerate(zip(a.runs, a.overlay_runs), 1)}
    ok = np.isfinite(a.scan.mu_I) & np.isfinite(a.overlay_scan.mu_I)
    if ok.any():
        d = np.abs(np.log10(a.scan.mu_I[ok]) - np.log10(a.overlay_scan.mu_I[ok])).max()
        rep.mu_I["scan: max |Δ log10 μ_I|"] = f"{d:.2e}"
    (betas / "comparison.md").write_text(rep.to_markdown(), encoding="utf-8")
    write_json(betas / "comparison.json",
               {"agree": rep.agree, "calibration": rep.calibration, "numeric_max_rel": rep.numeric_max_rel,
                "mismatches": [m.__dict__ for m in rep.mismatches]})


def _write_model_json(a: Analysis) -> None:
    main = a.systems.get("native") or a.systems["pyrate"]
    write_json(a.folder / "model.json", {
        "created": datetime.now().isoformat(timespec="seconds"),
        "higgsflow_version": __version__, "python": platform.python_version(),
        "model": a.spec.model.as_dict(), **a.spec.record(), "threshold_matching": a.threshold_used,
        "closure": main.meta.get("closure"), "warnings": a.spec.model.check(),
        "conventions": cp.conventions(), "sm_mu_I_GeV": a.scan.sm_mu_I,
    })
