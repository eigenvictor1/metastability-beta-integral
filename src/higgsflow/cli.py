"""Command line interface. Every command takes the same JSON file (one or many models).

    higgsflow run models/seesaw_typeI.json [--betas both] [--workers 0] [--trs | --no-trs]
    higgsflow yukawas models/seesaw_typeI.json       list the gauge-invariant Yukawas
    higgsflow betas models/leptons.json [--source both] [--latex out.tex]
    higgsflow install-pyrate
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config
from .paths import short


class _Formatter(logging.Formatter):
    """Time + message; the level only for warnings and errors; records with plain=True
    (the header of each model) are written as they are."""

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "plain", False):
            return record.getMessage()
        self._style._fmt = ("%(asctime)s %(levelname)s: %(message)s" if record.levelno >= logging.WARNING
                            else "%(asctime)s %(message)s")
        return super().format(record)


def _setup_logging(verbose: bool):
    """Log to the terminal: INFO by default, DEBUG with -v."""
    handler = logging.StreamHandler()
    handler.setFormatter(_Formatter(datefmt="%H:%M:%S"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[handler])


def _study(a):
    """Load the JSON file of the command; --betas and --trs/--no-trs (if given) override every
    model's choice (--trs only where the model has trace-plot settings)."""
    study = config.load(a.file)
    if getattr(a, "betas", None):
        for e in study.entries:
            e.setdefault("run", {})["betas"] = a.betas
    if getattr(a, "trs", None) is not None:
        for e in study.entries:
            if isinstance(e.get("trs"), dict):
                e["trs"] = {**e["trs"], "enabled": a.trs}
            elif not a.trs:
                e["trs"] = False
    return study


def cmd_run(a):
    """`higgsflow run`: analyse every model of the file; prints the path of summary.csv."""
    from .study import run_study
    print(short(run_study(_study(a), workers=a.workers)))
    return 0


def cmd_replot(a):
    """`higgsflow replot`: redraw the plots from the data stored by a previous run (no running)."""
    from .pipeline import replot
    target = Path(a.target)
    if target.suffix == ".json":                 # a model file: its results folder
        from .paths import results_root
        target = results_root() / config.load(target).name
    if not target.is_dir():
        print(f"no results folder {short(target)}")
        return 1
    drawn = replot(target, a.only)
    print(f"{len(drawn)} model(s) redrawn in {short(target)}")
    return 0 if drawn else 1


def cmd_yukawas(a):
    """`higgsflow yukawas`: list the gauge-invariant Yukawa operators of each model, and warnings."""
    for spec in _study(a).specs():
        m = spec.model
        print(m.label())
        for line in m.yukawa_invariants():
            print("  " + line)
        for w in m.check():
            print("  WARNING:", w)
    return 0


def cmd_betas(a):
    """`higgsflow betas`: compute (and with --source both, compare) the beta functions only.

    Prints them, or writes a LaTeX file per model with --latex. Exit code 1 if native and
    PyR@TE disagree.
    """
    from .pipeline import compute_systems
    ok = True
    for k, spec in enumerate(_study(a).specs()):
        m = spec.model
        print(f"# {m.label()}")
        systems, _, report = compute_systems(m, a.source, spec.loops)
        if report is not None:
            print(report.to_markdown())
            ok &= report.agree
        if a.latex:
            src = "native" if "native" in systems else "pyrate"
            out = Path(a.latex)
            if k:
                out = out.with_name(f"{out.stem}_{k + 1}{out.suffix}")
            out.write_text(systems[src].latex(), encoding="utf-8")
            print(f"LaTeX written to {short(out)}")
        elif report is None:
            bs = next(iter(systems.values()))
            for c in bs.couplings:
                for L in sorted(bs.coeffs[c]):
                    print(f"beta^({L})_{c} = {bs.expr(c, L)}")
    return 0 if ok else 1


def cmd_install_pyrate(a):
    """`higgsflow install-pyrate`: download the pinned PyR@TE version to ~/.higgsflow/pyrate."""
    from .pyrate.install import install_pyrate
    install_pyrate(force=a.force)
    return 0


def main(argv=None) -> int:
    """Parse the command line and dispatch to the cmd_* function; returns the exit code."""
    p = argparse.ArgumentParser(prog="higgsflow", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse every model in the file")
    r.add_argument("file")
    r.add_argument("--betas", choices=["native", "pyrate", "both"], help="override for all models")
    r.add_argument("--workers", type=int, help="processes (0 = all cores)")
    r.add_argument("--trs", action=argparse.BooleanOptionalAction, default=None,
                   help="make (--trs) or skip (--no-trs) the trace plots, overriding the file")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("replot", help="redraw the plots from stored data, without running the scans")
    rp.add_argument("target", help="the model file (.json) or its results folder")
    rp.add_argument("--only", nargs="+", metavar="MODEL", help="only these models (folder names)")
    rp.set_defaults(fn=cmd_replot)

    y = sub.add_parser("yukawas", help="list the gauge-invariant Yukawa couplings of each model")
    y.add_argument("file")
    y.set_defaults(fn=cmd_yukawas)

    b = sub.add_parser("betas", help="compute (and compare) the beta functions only")
    b.add_argument("file")
    b.add_argument("--source", choices=["native", "pyrate", "both"], default="native")
    b.add_argument("--latex")
    b.set_defaults(fn=cmd_betas)

    i = sub.add_parser("install-pyrate", help="download the pinned PyR@TE 3")
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_install_pyrate)

    a = p.parse_args(argv)
    _setup_logging(a.verbose)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
