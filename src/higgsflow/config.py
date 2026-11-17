"""JSON input files: one structure for one or many models.

    {
      "name":     "leptons",                       optional (default: file name)
      "defaults": { ...any per-model key... },     optional, shared by all models
      "models": [                                  required, one entry per model
        {
          "name": "VLL",                           optional (default: label from the reps)
          "fields": ["L4:(1,2,-1/2)", "L4c:(1,2,1/2)", "N:(1,1,0)"],
          "include_sm_mixing": false,
          "exclude": [],
          "scan":       {"y": {"linspace": [0, 1, 26]}, "M": {"logspace": [3, 8, 21]}},
          "links":      {"y": "all"},
          "fixed":      {},
          "examples":   [{"y": 0.5, "M": 1e3}],
          "benchmarks": [{"y": 0.5}],
          "run":        {"betas": "native", "loops": 2, "M": 1e3, "plots": true,
                         "threshold": "one-loop"},
          "sm_inputs":  {"Mt": 172.57},
          "numerics":   {"rtol": 1e-9}             optional, see settings.Numerics
        }
      ],
      "workers": 0                                 optional; processes, 0 = all cores
    }

A model entry overrides ``defaults`` key by key (one level deep). Fields:
compact strings "(su3,su2,Y)" with optional "name:" and "<n>x" prefixes, or
dicts {"name", "su3", "su2", "Y", "copies"}; unnamed fields are F1, F2, ...
Axes: a list, {"linspace": [a, b, n]} or {"logspace": [a, b, n]} (exponents).
A link "all" drives every new Yukawa.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .running.boundary import SMInputs
from .running.scan import expand_links
from .settings import Numerics
from .theory.model import Model


def axis(v: list | dict) -> np.ndarray:
    if isinstance(v, dict):
        ((kind, args),) = v.items()
        if kind == "linspace":
            return np.linspace(args[0], args[1], int(args[2]))
        if kind == "logspace":
            return np.logspace(args[0], args[1], int(args[2]))
        raise ValueError(f"unknown axis spec {kind!r} (use linspace, logspace or a list)")
    return np.asarray(v, dtype=float)


def _workers(v: int | None) -> int | None:
    return None if v in (0, None) else int(v)


BETA_SOURCES = ("native", "pyrate", "both")
THRESHOLDS = ("one-loop", "tree")


@dataclass
class RunSpec:
    """Everything run_analysis needs for one model.

    scan       {param: values} for 1 or 2 parameters: 'M' (GeV), new Yukawa names or keys of links
    links      {alias: [couplings] or "all"}: one parameter drives several couplings
    fixed      values of the parameters that are not scanned
    examples   parameter points for the running plot (default: 3 points from the scan)
    benchmarks parameter points whose μ_I goes into the summary table
    betas      'native', 'pyrate' or 'both' (compare, and overlay in the plots)
    M          threshold mass if 'M' is neither scanned nor fixed
    workers    processes for the scan (None = all cores)
    plots      write the two PDFs (data files are always written)
    threshold  matching at M: 'one-loop' (running/threshold.py) or 'tree'
    """
    model: Model
    scan: dict[str, np.ndarray]
    links: dict = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)
    examples: list[dict[str, float]] | None = None
    benchmarks: list[dict[str, float]] = field(default_factory=list)
    betas: str = "native"
    loops: int = 2
    M: float = 1e3
    workers: int | None = 1
    plots: bool = True
    threshold: str = "one-loop"
    inputs: SMInputs = field(default_factory=SMInputs)
    numerics: Numerics = field(default_factory=Numerics)

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
                "scan": {k: list(map(float, v)) for k, v in self.scan.items()},
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


def spec_from_dict(d: dict) -> RunSpec:
    run = d.get("run", {})
    return RunSpec(
        model=Model.from_dict(d.get("model", {})),
        scan={k: axis(v) for k, v in d.get("scan", {}).items()},
        links=d.get("links", {}),
        fixed={k: float(v) for k, v in d.get("fixed", {}).items()},
        examples=d.get("examples"),
        benchmarks=d.get("benchmarks", []),
        betas=run.get("betas", "native"),
        loops=int(run.get("loops", 2)),
        M=float(run.get("M", 1e3)),
        workers=_workers(run.get("workers", 1)),
        plots=bool(run.get("plots", True)),
        threshold=run.get("threshold", "one-loop"),
        inputs=SMInputs(**d.get("sm_inputs", {})),
        numerics=Numerics.from_dict(d.get("numerics")),
    )


@dataclass
class Study:
    """A JSON file: a named list of model entries (defaults already merged in)."""
    name: str
    entries: list[dict]
    workers: int | None = None

    def specs(self) -> list[RunSpec]:
        return [spec_from_dict(e) for e in self.entries]


_MODEL_KEYS = ("name", "fields", "include_sm_mixing", "exclude")


def merge(defaults: dict, entry: dict) -> dict:
    """Model entry (flat) + defaults -> the dict spec_from_dict expects."""
    out = copy.deepcopy(defaults)
    out["model"] = {k: entry[k] for k in _MODEL_KEYS if k in entry}
    for k, v in entry.items():
        if k in _MODEL_KEYS:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def study_from_dict(d: dict, default_name: str = "study") -> Study:
    if "models" not in d or not d["models"]:
        raise ValueError('the file needs a non-empty "models" list')
    defaults = d.get("defaults", {})
    return Study(d.get("name", default_name), [merge(defaults, e) for e in d["models"]],
                 _workers(d.get("workers", 0)))


def load(path: str | Path) -> Study:
    path = Path(path)
    return study_from_dict(json.loads(path.read_text(encoding="utf-8")), path.stem)
