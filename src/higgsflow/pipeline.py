"""End-to-end analysis of one model: betas -> runs -> results/<run>/.

    run_analysis(spec)
      ├─ compute_systems   native and/or PyR@TE beta functions (+ comparison)
      ├─ build_runners     one Runner per beta source, with the threshold matching
      ├─ run_points        examples, SM reference, scan, benchmarks
      └─ write_results     plots, data, betas, model.json
"""
from __future__ import annotations

import dataclasses
import logging
import pickle
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
    write_boundaries_csv,
    write_curves_csv,
    write_json,
    write_run_csv,
    write_samples_csv,
    write_scan_csv,
)
from .paths import results_root as default_results_root
from .paths import short
from .pyrate.compare import ComparisonReport, compare, map_conventions
from .rge.engine import compute_betas
from .rge.system import BetaSystem
from .running.curves import Curve
from .running.scan import NO_NEW_PHYSICS, SampleResult, ScanResult, resolve_params, run_samples, run_scan
from .running.solver import Runner, RunResult
from .running.threshold import MassError, ThresholdMatching
from .running.traces import TR4R, YukawaTraces, with_root_axis, with_trace_axis
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
    samples: SampleResult | None = None                           # trace plot (spec.trs)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_analysis(spec: RunSpec, *, run_name: str | None = None, results_root: str | Path | None = None,
                 pyrate_dir: str | Path | None = None, announce: bool = True,
                 details: bool = True) -> Analysis:
    """Compute betas, run examples, scan and benchmarks, write plots + data.

    run_name: results go to results/<run_name>/ (default: the model's label); the plots are
    running-, inscale-, inscale-min-, inscale-lambdamin- and inscale-trs-<theory>.pdf there, the data in <theory>/.
    announce: log the results folder and the beta functions used (a study does it once itself).
    details: log the model's Yukawa couplings.
    """
    spec = spec.prepared()
    model = spec.model
    for w in model.check():
        log.warning(w)
    run_folder = Path(results_root or default_results_root()) / (run_name or model.label())
    folder = prepare_folder(model.label(), run_folder)
    if announce:
        log.info("results folder: %s", short(run_folder))
    if details:
        log.info("Yukawa couplings: %s", "; ".join(model.yukawa_invariants()))

    model.check_fixed()
    if announce:
        log.info("%s", betas_text(spec))
        for line in fixed_text(model):
            log.info("%s", line)
    systems, sm, report = compute_systems(model, spec.betas, spec.loops, pyrate_dir)
    steps = compute_steps(model, systems, spec.loops, pyrate_dir)
    runners, threshold_used = build_runners(spec, systems, sm, steps)
    main = "native" if "native" in runners else "pyrate"
    a = Analysis(spec, folder, run_folder, systems, runners[main], threshold_used,
                 runs=run_points(runners[main], spec.examples or [], spec),
                 sm_run=runners[main].run({}, NO_NEW_PHYSICS, spec.inputs, sm_only=True),
                 scan=_scan(runners[main], spec, f"{model.label()}: scanning"),
                 benchmarks=run_points(runners[main], spec.benchmarks, spec),
                 report=report)
    if spec.betas == "both":
        a.overlay_runs = run_points(runners["pyrate"], spec.examples or [], spec)
        a.overlay_scan = _scan(runners["pyrate"], spec, f"{model.label()}: scanning (PyR@TE)")
    if spec.trs is not None and model.new_yukawa_names():
        a.samples = run_samples(runners[main], YukawaTraces(model), model.new_yukawa_names(), spec.trs.M,
                                spec.trs.samples, spec.trs.range, spec.inputs, spec.workers,
                                f"{model.label()}: sampling traces at fixed M")

    write_results(a)
    log.debug("done: %s", short(folder))
    return a


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

def compute_systems(model: Model, betas: str, loops: int, pyrate_dir: str | Path | None = None
                    ) -> tuple[dict[str, BetaSystem], dict[str, BetaSystem], ComparisonReport | None]:
    """Return ({source: BetaSystem}, {source: SM BetaSystem}, report or None).

    PyR@TE betas are always checked against the native ones (also with betas='pyrate',
    where the native betas are only used for the check): a warning is logged if they do not
    coincide. The PyR@TE systems are re-expressed in the native Yukawa normalisation, which
    the boundary conditions, threshold matching and traces use.
    """
    systems, sm = {}, {}
    native = sm_native = None
    if betas in ("native", "both", "pyrate"):
        log.debug("computing native betas")
        native, sm_native = compute_betas(model, loops), compute_betas(Model(), loops)
        if betas != "pyrate":
            systems["native"], sm["native"] = native, sm_native
    if betas in ("pyrate", "both"):
        from .pyrate.bridge import run_pyrate
        log.debug("computing betas with PyR@TE")
        systems["pyrate"] = run_pyrate(model, loops, pyrate_dir=pyrate_dir)
        sm["pyrate"] = run_pyrate(Model(), loops, pyrate_dir=pyrate_dir)
    if not systems:
        raise ValueError(f"unknown beta source {betas!r}")
    report = None
    if native is not None and sm_native is not None and "pyrate" in systems:
        report = compare(native, systems["pyrate"])
        _, _, systems["pyrate"] = map_conventions(native, systems["pyrate"])
        _, _, sm["pyrate"] = map_conventions(sm_native, sm["pyrate"])
        log.debug("native vs PyR@TE agreement: %s (Yukawa map %s)",
                  "YES" if report.agree else "NO", report.calibration)
        if not report.agree:
            _warn_disagree(model.label(), report)
    return systems, sm, report


def _warn_disagree(label: str, report: ComparisonReport, where: str = "") -> None:
    """Terminal warning: PyR@TE and native beta functions differ."""
    n = len(report.mismatches)
    log.warning("%s%s: PyR@TE and native beta functions do NOT coincide (%d coefficient(s) differ, "
                "max relative difference at random points %.2g); see betas/comparison.md",
                label, where, n, report.numeric_max_rel)


def betas_text(spec: RunSpec) -> str:
    """E.g. 'beta functions: native, 2-loop, one-loop matching at M'."""
    src = {"both": "native and PyR@TE"}.get(spec.betas, spec.betas)
    return f"beta functions: {src}, {spec.loops}-loop, {spec.threshold} matching at M"


def fixed_text(model: Model) -> list[str]:
    """One line per field with its own mass, and one for the singlet."""
    out = [f"{f.name}{f.rep()} decouples at its own mass, {f.mass:.3g} GeV" for f in model.fixed_fields]
    if model.singlet is not None:
        sg = model.singlet
        ys = "2√λ_S M/m_S" if sg.yS is None else f"{sg.yS:g}"
        out.append(f"singlet S: m_S = {sg.m_S:.3g} GeV, λ_S = {sg.lamS:g}, κ = {sg.kappa:g}, "
                   f"y_S = {ys} (λ jumps by κ²/4λ_S = {sg.shift:.3g} at m_S)")
    return out


def step_models(model: Model):
    """(state, theory) for every theory between thresholds (fixed-mass fields, singlet S),
    keyed by (number of fixed-mass fields active, above M, above m_S); the SM and the full
    theory are not included."""
    k = len(model.fixed_fields)
    has_S = model.singlet is not None
    for n in range(k + 1):
        for above_M in (False, True):
            for above_S in ((False, True) if has_S else (False,)):
                state = (n, above_M, above_S)
                if state in ((0, False, False), (k, True, has_S)):
                    continue
                yield state, model.with_fixed(n, scanned=above_M, singlet=above_S)


def native_models(spec: RunSpec) -> list[Model]:
    """Every theory whose native beta functions run_analysis(spec) needs (SM included)."""
    if spec.betas not in ("native", "both", "pyrate"):     # PyR@TE is checked against them
        return []
    model = spec.prepared().model
    return [Model(), model, *(m for _, m in step_models(model))]


def compute_steps(model: Model, systems: dict[str, BetaSystem], loops: int,
                  pyrate_dir: str | Path | None = None) -> dict[str, dict]:
    """Per beta source, the beta functions of the theories between thresholds (see step_models)."""
    out: dict[str, dict] = {s: {} for s in systems}
    if model.singlet is not None and "pyrate" in systems:
        raise ValueError("the scalar singlet is only implemented with the native betas")
    for state, m in step_models(model):
        if "native" in systems:
            out["native"][state] = compute_betas(m, loops)
        if "pyrate" in systems:
            from .pyrate.bridge import run_pyrate
            bs = run_pyrate(m, loops, pyrate_dir=pyrate_dir)
            native = out["native"][state] if "native" in systems else compute_betas(m, loops)
            rep = compare(native, bs)
            if not rep.agree:
                _warn_disagree(model.label(), rep, f" (theory below threshold {state})")
            _, _, bs = map_conventions(native, bs)
            out["pyrate"][state] = bs
    return out


def build_runners(spec: RunSpec, systems: dict[str, BetaSystem], sm: dict[str, BetaSystem],
                  steps: dict[str, dict] | None = None) -> tuple[dict[str, Runner], str]:
    """One Runner per beta source; returns them and the matching actually used."""
    matching, used = None, "tree"
    model = spec.model
    if model.fixed_fields or model.singlet is not None:
        model = model.with_fixed(0)                        # the fields at M only
    if spec.threshold == "one-loop" and model.new_fields:
        try:
            matching, used = ThresholdMatching(model), "one-loop"
        except MassError as exc:
            log.warning("no one-loop threshold matching, using tree level: %s", exc)
            used = f"tree ({exc})"
    masses = [f.mass for f in spec.model.fixed_fields if f.mass is not None]
    runners = {s: Runner(systems[s], sm[s], spec.loops, matching, spec.numerics, model,
                         (steps or {}).get(s), masses, spec.model.singlet) for s in systems}
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


def _scan(runner: Runner, spec: RunSpec, label: str = "Scanning") -> ScanResult:
    """Run the spec's scan with the given runner (1D or 2D grid, possibly in parallel)."""
    return run_scan(runner, spec.scan, spec.fixed, spec.links, spec.inputs, spec.M, workers=spec.workers,
                    label=label)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_results(a: Analysis) -> None:
    """Write everything of an analysis: plots (if spec.plots), data CSVs, betas and model.json."""
    if a.spec.plots:
        _write_plots(a)
    _write_data(a)
    _write_betas(a)
    _write_model_json(a)


def _write_plots(a: Analysis) -> None:
    """Back up old plots of this theory and draw running-, inscale-, inscale-min-,
    inscale-lambdamin- and inscale-trs-<theory>.pdf."""
    draw_plots(a.run_folder, plot_data(a))


PLOT_DATA = "plotdata.pkl"        # <theory>/data/: everything the plots show (see replot)
PLOT_DATA_VERSION = 1


def plot_data(a: Analysis) -> dict:
    """What the plots of an analysis show, ready to draw: titles, the example runs, and the
    scan with the Yukawa axis already shown as Tr((YY†)²)^(1/4) (if the scan has one)."""
    theory = a.spec.model.label()
    traces = YukawaTraces(a.spec.model)
    scan = with_root_axis(with_trace_axis(a.scan, traces, a.spec.links, a.spec.M))
    overlay = (with_root_axis(with_trace_axis(a.overlay_scan, traces, a.spec.links, a.spec.M))
               if a.overlay_scan is not None else None)
    return {"version": PLOT_DATA_VERSION, "theory": theory,
            "name": f"{theory}, {a.spec.model.fields_text()}",       # running and trace plot titles
            "title": a.spec.model.latex_title(),                      # map titles (LaTeX)
            "runs": a.runs, "sm_run": a.sm_run, "overlay_runs": a.overlay_runs,
            "scan": scan, "overlay": overlay, "samples": a.samples,
            "limits": {TR4R: tuple(v ** 0.25 for v in a.spec.trace_range)},
            "y_label": a.spec.y_label}


def draw_plots(run_folder: Path, d: dict, backup: bool = True) -> None:
    """Draw the plots of one theory from plot_data() (or stored plot data) into run_folder;
    with backup, older plots of the theory are moved to backup/ first."""
    from .output.plots import plot_instability, plot_running, plot_traces  # matplotlib only where plotting
    theory, name = d["theory"], d["name"]
    running, inscale, trs, inscale_min, inscale_lammin = plot_names(theory)
    if backup and backup_files(run_folder, (running, inscale, trs, inscale_min, inscale_lammin)):
        log.info("%s: old plots moved to backup/", theory)
    plot_running(run_folder / running, d["runs"], d["sm_run"], d["overlay_runs"] or None,
                 title=f"{name}: running of λ")
    scan, limits = d["scan"], d["limits"]
    title = d.get("title") or name            # the maps: the particle and its representation
    style = {"limits": limits, "y_label": d.get("y_label"), "color_range": d.get("color_range")}
    plot_instability(run_folder / inscale, scan, d["overlay"], title=title, **style)
    plot_instability(run_folder / inscale_min, scan, title=title, minima="mu", **style)
    plot_instability(run_folder / inscale_lammin, scan, title=title, minima="lam", **style)
    if d["samples"] is not None:
        plot_traces(run_folder / trs, d["samples"], title=f"{name}: instability scale")


_STORED = {"ScanResult": ScanResult, "SampleResult": SampleResult, "RunResult": RunResult, "Curve": Curve}


def _pack(x):
    """Plot data as plain containers: the result dataclasses become {"__class__", fields}, so
    that data stored by an older version still loads when fields are added (see _unpack)."""
    if type(x).__name__ in _STORED and dataclasses.is_dataclass(x):
        return {"__class__": type(x).__name__,
                **{f.name: _pack(getattr(x, f.name)) for f in dataclasses.fields(x)}}
    if isinstance(x, dict):
        return {k: _pack(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_pack(v) for v in x]
    return x


def _unpack(x):
    """Inverse of _pack: fields missing in old data take their defaults, unknown ones are dropped."""
    if isinstance(x, dict) and x.get("__class__") in _STORED:
        cls = _STORED[x["__class__"]]
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: _unpack(v) for k, v in x.items() if k in names})
    if isinstance(x, dict):
        return {k: _unpack(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_unpack(v) for v in x]
    return x


def save_plot_data(path: Path, d: dict) -> None:
    """Store plot_data() (pickle of plain containers and numpy arrays)."""
    with open(path, "wb") as fh:
        pickle.dump(_pack(d), fh)


def load_plot_data(path: Path) -> dict:
    """Read data stored by save_plot_data."""
    with open(path, "rb") as fh:
        return _unpack(pickle.load(fh))


def replot(run_folder: Path, only: list[str] | None = None,
           settings: dict[str, dict] | None = None) -> list[str]:
    """Redraw the plots of every theory in a results folder from their stored plot data,
    without running anything (e.g. after changing the plot style). only: theory labels to
    redraw. settings: {theory: plot settings} replacing the stored ones (e.g. "y_label", from
    the model file). Returns the theories drawn; theories without stored data are skipped with a
    warning (they were run before plot data was stored: run them once more).

    All maps are drawn on one colour range of log10 μ_I (exactly the values shown, over every
    theory of the folder, also those not in only), and its colour bar is written as COLORBAR in the folder,
    to be set above the maps in a LaTeX figure (the maps have none of their own)."""
    color_range = shared_color_range(run_folder)
    drawn = []
    for data in sorted(run_folder.glob(f"*/data/{PLOT_DATA}")):
        theory = data.parent.parent.name
        if only and theory not in only:
            continue
        d = load_plot_data(data)
        d.update((settings or {}).get(theory, {}))
        if color_range is not None:
            d["color_range"] = color_range
        draw_plots(run_folder, d, backup=False)
        log.info("%s: plots redrawn", theory)
        drawn.append(theory)
    if color_range is not None and drawn:
        from .output.plots import map_colorbar
        map_colorbar(run_folder / COLORBAR, color_range)
        log.info("colour bar of the maps: %s", short(run_folder / COLORBAR))
        from .output.plots import map_legend
        map_legend(run_folder / LEGEND)
        log.info("legend of the map regions: %s", short(run_folder / LEGEND))
    for folder in sorted(p for p in run_folder.iterdir() if p.is_dir() and p.name != "backup"):
        if not (folder / "data" / PLOT_DATA).exists() and (not only or folder.name in only):
            log.warning("%s: no stored plot data (run it once with this version)", folder.name)
    return drawn


COLORBAR = "inscale-colorbar.pdf"     # in a run folder: the colour bar shared by its maps
LEGEND = "inscale-legend.pdf"         # ... and the legend of their stable/metastable/unstable regions


def shared_color_range(run_folder: Path) -> tuple[float, float] | None:
    """Range of log10 μ_I shown in the maps of every theory in a run folder: exactly from the
    lowest to the highest value inside the plotted axis limits (not rounded)."""
    lo, hi = np.inf, -np.inf
    for data in run_folder.glob(f"*/data/{PLOT_DATA}"):
        d = load_plot_data(data)
        scan = d.get("scan")
        if scan is None or len(scan.axes) != 2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.log10(np.asarray(scan.mu_I, dtype=float))
        shown = np.ones(z.shape, bool)
        for a, (n, v) in enumerate(scan.axes.items()):          # inside the plotted limits only
            if n in (d.get("limits") or {}):
                vlo, vhi = d["limits"][n]
                inside = (np.asarray(v) >= vlo) & (np.asarray(v) <= vhi)
                shown &= np.expand_dims(inside, 1 - a)
        z = z[shown & np.isfinite(z)]
        if z.size:
            lo, hi = min(lo, float(z.min())), max(hi, float(z.max()))
    return (lo, hi) if np.isfinite(lo) and hi > lo else None


def _write_data(a: Analysis) -> None:
    """data/: running_<k>.csv per example, running_SM.csv, scan.csv, boundaries.csv and
    curves.csv (traced curves)."""
    data = a.folder / "data"
    for k, r in enumerate(a.runs, 1):
        write_run_csv(data / f"running_{k}.csv", r)
    write_run_csv(data / "running_SM.csv", a.sm_run)
    write_scan_csv(data / "scan.csv", a.scan, a.overlay_scan)
    save_plot_data(data / PLOT_DATA, plot_data(a))
    write_boundaries_csv(data / "boundaries.csv", a.scan)
    write_curves_csv(data / "curves.csv", a.scan)
    if a.samples is not None:
        write_samples_csv(data / "trs_samples.csv", a.samples)


def _write_betas(a: Analysis) -> None:
    """betas/: LaTeX and JSON per beta source; with both sources also comparison.md/.json,
    including the μ_I differences between native and PyR@TE running.
    """
    betas = a.folder / "betas"
    for src, bs in a.systems.items():
        (betas / f"betas_{src}.tex").write_text(bs.latex(), encoding="utf-8")
        write_json(betas / f"betas_{src}.json", bs.to_json())
    if a.report is None:
        return
    rep = a.report
    if a.overlay_scan is not None:          # betas='both': the μ_I differences too
        rep.mu_I = {f"example {k}": f"native {r.mu_I}, PyR@TE {o.mu_I}"
                    for k, (r, o) in enumerate(zip(a.runs, a.overlay_runs), 1)}
    ok = (np.isfinite(a.scan.mu_I) & np.isfinite(a.overlay_scan.mu_I)) if a.overlay_scan is not None else None
    if ok is not None and ok.any():
        assert a.overlay_scan is not None
        d = np.abs(np.log10(a.scan.mu_I[ok]) - np.log10(a.overlay_scan.mu_I[ok])).max()
        rep.mu_I["scan: max |Δ log10 μ_I|"] = f"{d:.2e}"
    (betas / "comparison.md").write_text(rep.to_markdown(), encoding="utf-8")
    write_json(betas / "comparison.json",
               {"agree": rep.agree, "calibration": rep.calibration, "numeric_max_rel": rep.numeric_max_rel,
                "mismatches": [m.__dict__ for m in rep.mismatches]})


def _write_model_json(a: Analysis) -> None:
    """model.json: model, all settings, matching used, closure checks, warnings, conventions
    and the SM μ_I, i.e. everything needed to reproduce the analysis.
    """
    main = a.systems.get("native") or a.systems["pyrate"]
    write_json(a.folder / "model.json", {
        "created": datetime.now().isoformat(timespec="seconds"),
        "higgsflow_version": __version__, "python": platform.python_version(),
        "model": a.spec.model.as_dict(), **a.spec.record(), "threshold_matching": a.threshold_used,
        "closure": main.meta.get("closure"), "warnings": a.spec.model.check(),
        "conventions": cp.conventions(), "sm_mu_I_GeV": a.scan.sm_mu_I,
        "sm_lam_min": a.scan.sm_lam_min, "sm_mu_min_GeV": a.scan.sm_mu_min,
    })
