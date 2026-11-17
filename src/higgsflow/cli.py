"""Command line interface. Every command takes the same JSON file (one or many models).

    higgsflow run models/leptons.json [--betas both] [--workers 0]
    higgsflow yukawas models/leptons.json            list the gauge-invariant Yukawas
    higgsflow betas models/leptons.json [--source both] [--latex out.tex]
    higgsflow install-pyrate
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config


def _setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def _study(a):
    study = config.load(a.file)
    if getattr(a, "betas", None):
        for e in study.entries:
            e.setdefault("run", {})["betas"] = a.betas
    return study


def cmd_run(a):
    from .study import run_study
    print(run_study(_study(a), workers=a.workers))
    return 0


def cmd_yukawas(a):
    for spec in _study(a).specs():
        m = spec.model
        print(m.label())
        for line in m.yukawa_invariants():
            print("  " + line)
        for w in m.check():
            print("  WARNING:", w)
    return 0


def cmd_betas(a):
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
            print(f"LaTeX written to {out}")
        elif report is None:
            bs = next(iter(systems.values()))
            for c in bs.couplings:
                for L in sorted(bs.coeffs[c]):
                    print(f"beta^({L})_{c} = {bs.expr(c, L)}")
    return 0 if ok else 1


def cmd_install_pyrate(a):
    from .pyrate.install import install_pyrate
    install_pyrate(force=a.force)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="higgsflow", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse every model in the file")
    r.add_argument("file")
    r.add_argument("--betas", choices=["native", "pyrate", "both"], help="override for all models")
    r.add_argument("--workers", type=int, help="processes (0 = all cores)")
    r.set_defaults(fn=cmd_run)

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
