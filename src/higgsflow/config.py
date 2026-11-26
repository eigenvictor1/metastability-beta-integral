"""JSON input files: one structure for one or many models.

    {
      "name":     "leptons",                       optional (default: file name)
      "defaults": {                                required: the scan, shared by all models,
        "scan": {"y": {"linspace": [0, 1, 26]}, "M": {"logspace": [3, 8, 21]}},
        ...any other per-model key...              and defaults for the keys below
      },
      "models": [                                  required, one entry per model
        {
          "name": "VLL",                           optional (default: label from the reps)
          "fields": ["L4:(1,2,-1/2)", "L4c:(1,2,1/2)", "N:(1,1,0)"],
          "include_sm_mixing": false,
          "exclude": [],
          "links":      {"y": "all"},
          "fixed":      {},
          "examples":   [{"y": 0.5, "M": 1e3}],
          "benchmarks": [{"y": 0.5}],
          "run":        {"betas": "pyrate", "loops": 2, "M": 1e3, "plots": true,
                         "threshold": "one-loop", "trace_range": [1e-3, 1e3], "y_label": "$y$"},
          "sm_inputs":  {"Mt": 172.57},
          "numerics":   {"rtol": 1e-9}             optional, see settings.Numerics
          "trs":        {"M": 1e3, "samples": 2000, "range": [0, 1.5]}
                                                   optional: trace plot, see TraceSpec
        }
      ],
      "workers": 8                                 optional; processes (default 3/4 of the cores), 0 = all
    }

The scan is set in "defaults" only, so that every model of a file is scanned over the
same grid; a model entry with its own "scan" is an error. Its parameter names must exist
in every model, e.g. through "links" ({"y": "all"} or a list of that model's Yukawas).
Any other key of a model entry overrides ``defaults`` key by key (one level deep). Fields:
compact strings "(su3,su2,Y)" with optional "name:" and "<n>x" prefixes, or
dicts {"name", "su3", "su2", "Y", "copies"}; unnamed fields are F1, F2, ...
Axes: a list, {"linspace": [a, b, n]} or {"logspace": [a, b, n]} (exponents).
A link "all" drives every new Yukawa.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .running.boundary import SMInputs
from .running.scan import expand_links
from .settings import Numerics
from .theory.model import Model


def axis(v: list | dict) -> np.ndarray:
    """Scan axis from JSON: a list of values, {"linspace": [a, b, n]} or {"logspace": [a, b, n]}."""
    if isinstance(v, dict):
        ((kind, args),) = v.items()
        if kind == "linspace":
            return np.linspace(args[0], args[1], int(args[2]))
        if kind == "logspace":
            return np.logspace(args[0], args[1], int(args[2]))
        raise ValueError(f"unknown axis spec {kind!r} (use linspace, logspace or a list)")
    return np.asarray(v, dtype=float)


WORKER_SHARE = 0.75      # processes when a file does not say: this share of the cores


def default_workers() -> int:
    """Default number of processes: WORKER_SHARE of the machine's cores (at least 1), so that
    some are left free for other work."""
    return max(1, int((os.cpu_count() or 1) * WORKER_SHARE))


def _workers(v: int | None) -> int | None:
    """JSON 'workers' value -> processes: 0 or None means all cores (None)."""
    return None if v in (0, None) else int(v)


@dataclass(frozen=True)
class TraceSpec:
    """The trace plot inscale-trs-<theory>.pdf: μ_I against Tr(YY†) and Tr((YY†)²) at fixed M.

    about ``samples`` points of all the model's new Yukawas, with overall size sqrt(Σ y²) in
    range, chosen to cover the reachable part of the trace plane evenly (running.scan.run_samples).
    It is optional: made only if "trs" is given and "enabled" is not false ("trs": false also
    switches it off); `higgsflow run --no-trs` skips it for every model, `--trs` makes it for
    every model that has its settings.
    """
    M: float
    samples: int = 2000
    range: tuple[float, float] = (0.0, 1.5)

    @classmethod
    def from_dict(cls, d: dict | None) -> TraceSpec | None:
        """From the JSON key "trs" (None if absent, false, or with "enabled": false); unknown keys
        are an error."""
        if not d or not isinstance(d, dict) or not d.get("enabled", True):
            return None
        unknown = set(d) - {"M", "samples", "range", "enabled"}
        if unknown:
            raise ValueError(f"unknown trs keys {sorted(unknown)}; allowed: M, samples, range, enabled")
        if "M" not in d:
            raise ValueError('"trs" needs a fixed mass "M"')
        lo, hi = (float(x) for x in d.get("range", cls.range))
        if not 0 <= lo < hi:
            raise ValueError('"trs" range must be [lo, hi] with 0 <= lo < hi')
        return cls(float(d["M"]), int(d.get("samples", cls.samples)), (lo, hi))


BETA_SOURCES = ("native", "pyrate", "both")
THRESHOLDS = ("one-loop", "tree")


@dataclass
class RunSpec:
    """Everything run_analysis needs for one model.

    scan       {param: values} for 1 or 2 parameters: 'M' (GeV), new Yukawa names or keys of links
    links      {alias: [couplings], {coupling: factor} or "all"}: one parameter drives several
               couplings (each = factor x parameter; factor 1 for a list or "all")
    fixed      values of the parameters that are not scanned
    examples   parameter points for the running plot (default: 3 points from the scan)
    benchmarks parameter points whose μ_I goes into the summary table
    betas      'native', 'pyrate' or 'both' (compare, and overlay in the plots)
    M          threshold mass if 'M' is neither scanned nor fixed
    workers    processes for the scan (None = all cores)
    plots      write the two PDFs (data files are always written)
    threshold  matching at M: 'one-loop' (running/threshold.py) or 'tree'
    trs        settings of the trace plot (None: not made)
    trace_range  range of the Tr((YY†)²) axis (log scale) of the instability maps
    y_label    label of that axis in the maps (LaTeX); None: Tr[(YY†)²]^(1/4)
    """
    model: Model
    scan: dict[str, np.ndarray]
    links: dict = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)
    examples: list[dict[str, float]] | None = None
    benchmarks: list[dict[str, float]] = field(default_factory=list)
    betas: str = "pyrate"
    loops: int = 2
    M: float = 1e3
    workers: int | None = 1
    plots: bool = True
    threshold: str = "one-loop"
    inputs: SMInputs = field(default_factory=SMInputs)
    numerics: Numerics = field(default_factory=Numerics)
    trs: TraceSpec | None = None
    trace_range: tuple[float, float] = (1e-3, 1e3)
    y_label: str | None = None

    def prepared(self) -> RunSpec:
        """Checked copy: scanned keys removed from ``fixed`` (scanned values win), links
        expanded to coupling names, default examples filled in."""
        if self.betas not in BETA_SOURCES:
            raise ValueError(f"betas must be one of {BETA_SOURCES}, not {self.betas!r}")
        if self.threshold not in THRESHOLDS:
            raise ValueError(f"threshold must be one of {THRESHOLDS}, not {self.threshold!r}")
        if not 1 <= len(self.scan) <= 2:
            raise ValueError("scan one or two parameters (fix the rest with 'fixed')")
        return replace(
            self,
            fixed={k: v for k, v in self.fixed.items() if k not in self.scan},
            links=expand_links(self.links, self.model.new_yukawa_names()),
            examples=self.examples or default_examples(self.scan),
        )

    def record(self) -> dict:
        """The settings as written to model.json."""
        return {"sm_inputs": self.inputs.as_dict(), "sm_couplings_at_Mt": self.inputs.couplings_at_mt(),
                "loops": self.loops, "betas": self.betas, "threshold_M_default": self.M,
                "numerics": self.numerics.as_dict(),
                "trs": None if self.trs is None else {"M": self.trs.M, "samples": self.trs.samples,
                                                      "range": list(self.trs.range)},
                "scan": {k: list(map(float, v)) for k, v in self.scan.items()},
                "trace_range": list(self.trace_range), "y_label": self.y_label,
                "fixed": self.fixed, "links": self.links or {}, "examples": self.examples,
                "benchmarks": self.benchmarks}


def default_examples(scan: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """Three points along the first scan axis, at the middle of the second."""
    names = list(scan)
    vals = np.asarray(scan[names[0]], dtype=float)
    out = []
    for v in (vals[len(vals) // 4], vals[len(vals) // 2], vals[(3 * len(vals)) // 4]):
        d = {names[0]: float(v)}
        for m in names[1:]:
            mv = np.asarray(scan[m], dtype=float)
            d[m] = float(mv[len(mv) // 2])
        out.append(d)
    return out


def _trace_range(v) -> tuple[float, float]:
    """JSON "trace_range": [lo, hi] with 0 < lo < hi (log axis)."""
    lo, hi = (float(x) for x in v)
    if not 0 < lo < hi:
        raise ValueError('"trace_range" must be [lo, hi] with 0 < lo < hi')
    return lo, hi


def spec_from_dict(d: dict) -> RunSpec:
    """One merged model entry (defaults applied, see merge) -> RunSpec. Missing keys take defaults."""
    run = d.get("run", {})
    return RunSpec(
        model=Model.from_dict(d.get("model", {})),
        scan={k: axis(v) for k, v in d.get("scan", {}).items()},
        links=d.get("links", {}),
        fixed={k: float(v) for k, v in d.get("fixed", {}).items()},
        examples=d.get("examples"),
        benchmarks=d.get("benchmarks", []),
        betas=run.get("betas", "pyrate"),
        loops=int(run.get("loops", 2)),
        M=float(run.get("M", 1e3)),
        workers=_workers(run.get("workers", 1)),
        plots=bool(run.get("plots", True)),
        threshold=run.get("threshold", "one-loop"),
        trace_range=_trace_range(run.get("trace_range", (1e-3, 1e3))),
        y_label=run.get("y_label"),
        inputs=SMInputs(**d.get("sm_inputs", {})),
        numerics=Numerics.from_dict(d.get("numerics")),
        trs=TraceSpec.from_dict(d.get("trs")),
    )


@dataclass
class Study:
    """A JSON file: a named list of model entries (defaults already merged in)."""
    name: str
    entries: list[dict]
    workers: int | None = field(default_factory=default_workers)

    def specs(self) -> list[RunSpec]:
        """The RunSpec of every entry (raises on the first invalid one)."""
        return [spec_from_dict(e) for e in self.entries]


_MODEL_KEYS = ("name", "fields", "include_sm_mixing", "exclude", "singlet")


def merge(defaults: dict, entry: dict) -> dict:
    """Model entry (flat) + defaults -> the dict spec_from_dict expects."""
    out = copy.deepcopy(defaults)
    out["model"] = {k: entry[k] for k in _MODEL_KEYS if k in entry}
    singlet = out.pop("singlet", None)          # a singlet in "defaults" applies to every model
    if singlet is not None and "singlet" not in out["model"]:
        out["model"]["singlet"] = singlet
    for k, v in entry.items():
        if k in _MODEL_KEYS:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def study_from_dict(d: dict, default_name: str = "study") -> Study:
    """Parsed JSON file -> Study; each model entry is merged with 'defaults'."""
    if "models" not in d or not d["models"]:
        raise ValueError('the file needs a non-empty "models" list')
    defaults = d.get("defaults", {})
    if not defaults.get("scan"):
        raise ValueError('the scan must be given in "defaults": {"scan": {...}}')
    with_scan = [e.get("name", f"model {i}") for i, e in enumerate(d["models"], 1) if "scan" in e]
    if with_scan:
        raise ValueError(f'"scan" is only allowed in "defaults", not in a model entry ({", ".join(with_scan)})')
    return Study(d.get("name", default_name), [merge(defaults, e) for e in d["models"]],
                 _workers(d["workers"]) if "workers" in d else default_workers())


def load(path: str | Path) -> Study:
    """Read a JSON input file; the run name defaults to the file name."""
    path = Path(path)
    return study_from_dict(json.loads(path.read_text(encoding="utf-8")), path.stem)
