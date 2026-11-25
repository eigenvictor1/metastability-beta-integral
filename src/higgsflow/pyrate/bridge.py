"""Bridge to PyR@TE 3 (Sartore & Schienbein, arXiv:2007.12700).

Model -> PyR@TE model file -> run PyR@TE -> parse its Python export -> BetaSystem.

Every Weyl component (each copy of each field) becomes a separate PyR@TE
fermion with Gen: 1, so that all couplings are scalars. Yukawa operators are
written with PyR@TE's own Clebsch-Gordan tensors (``cgc``); their normalisation
may differ from ours by a constant per coupling, which compare.py calibrates.

PyR@TE is run with ``--RealBasis None``: with its default (rotating real
representations such as the SU(2) triplet to a real basis) we found that the
two-loop g1^2 g2^2 y^2 term of beta_lambda becomes basis dependent when the
real-representation fermion carries hypercharge (e.g. a (1,3,1) triplet), which
cannot be physical. Without the rotation PyR@TE agrees with the native engine.

PyR@TE is not shipped with higgsflow: ``higgsflow install-pyrate`` downloads a
pinned version (see pyrate/install.py).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import sympy as sp

from ..paths import cache_dir as default_cache_dir
from ..rge.system import BetaSystem
from ..theory import couplings as cp
from ..theory.model import Model
from .install import find_pyrate

log = logging.getLogger(__name__)


def _atomic_write(path: Path, text: str):
    """Write via a temporary file so parallel workers never see half-written caches."""
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)

# --------------------------------------------------------------------------
# Model file
# --------------------------------------------------------------------------

def _qnb(f) -> str:
    """PyR@TE quantum-number string of a field, e.g. '{U1Y: 1/6, SU2L: [1], SU3c: [1,0]}'."""
    parts = []
    if f.Y != 0:
        parts.append(f"U1Y: {f.Y}")
    if f.su2 > 1:
        parts.append(f"SU2L: [{f.su2 - 1}]")
    if f.su3 != (0, 0):
        parts.append(f"SU3c: [{f.su3[0]},{f.su3[1]}]")
    return "{" + ", ".join(parts) + "}"


def pyrate_names(model: Model) -> dict[str, str]:
    """our coupling name -> PyR@TE-safe name."""
    out = {c.name: c.pyrate for c in cp.SM if c.name != cp.TOP_YUKAWA}
    k = 0
    for y in model.yukawas():
        if y.name == cp.TOP_YUKAWA:
            out[y.name] = cp.YT.pyrate
        else:
            k += 1
            out[y.name] = f"yk{k}"
    return out


def write_model_file(model: Model, path: Path) -> dict[str, str]:
    """Write the PyR@TE model file (YAML) for model; returns {our coupling name: PyR@TE name}.

    Each Weyl component is a separate fermion (Gen: 1). Each Yukawa is written with PyR@TE's
    Clebsch-Gordan tensors (cgc) for SU(2) and SU(3), contracted with the two fermions and
    H or Htilde (for H^*).
    """
    comps = model.components()
    names = pyrate_names(model)
    lines = ["Author: higgsflow", "Date: 01.01.2026", f"Name: hf{model.hash()}",
             "Groups: {U1Y: U1, SU2L: SU2, SU3c: SU3}", "", "Fermions: {"]
    for k, c in enumerate(comps, 1):
        lines.append(f"    F{k} : {{Gen: 1, Qnb: {_qnb(c.field)}}},   # {c.name}")
    lines += ["}", "", "RealScalars: {}", "",
              "ComplexScalars: {",
              "    H : {RealFields: [Pi, Sigma], Norm: 1/sqrt(2), Qnb: {U1Y: 1/2, SU2L: [1]}},",
              "}", "", "Potential: {", "    Definitions: {",
              "        Htilde[i] : Eps[i,j]*Hbar[j],"]
    defs, terms = [], []
    cmap = {c.name: c for c in comps}
    safe = {c.name: f"F{k}" for k, c in enumerate(comps, 1)}   # PyR@TE-safe field names
    for n, yk in enumerate(model.yukawas()):
        a, b = cmap[yk.f1].field, cmap[yk.f2].field
        # H^* is written through Htilde = eps.H^*, which is an SU(2) doublet like H
        h = "Htilde" if yk.conj_higgs else "H"
        idx_a, idx_b = [], []
        factors = []
        # SU(2): H is always a doublet
        su2_fields, su2_idx = [], []
        if a.su2 > 1:
            su2_fields.append(safe[yk.f1])
            su2_idx.append(f"i{n}")
            idx_a.append(f"i{n}")
        if b.su2 > 1:
            su2_fields.append(safe[yk.f2])
            su2_idx.append(f"j{n}")
            idx_b.append(f"j{n}")
        su2_fields.append("H")
        su2_idx.append(f"k{n}")
        defs.append(f"        C2x{n} : cgc(SU2L, {', '.join(su2_fields)}),")
        factors.append(f"C2x{n}[{','.join(su2_idx)}]")
        if a.su3 != (0, 0):
            defs.append(f"        C3x{n} : cgc(SU3c, {safe[yk.f1]}, {safe[yk.f2]}),")
            factors.append(f"C3x{n}[a{n},b{n}]")
            idx_a.append(f"a{n}")
            idx_b.append(f"b{n}")
        fa = safe[yk.f1] + (f"[{','.join(idx_a)}]" if idx_a else "")
        fb = safe[yk.f2] + (f"[{','.join(idx_b)}]" if idx_b else "")
        terms.append(f"        {names[yk.name]} : {' '.join(factors)} {fa} {fb} {h}[k{n}],")
    lines += defs + ["    },", "    Yukawas: {"] + terms + ["    },",
              "    QuarticTerms: { lam : (Hbar[i] H[i])**2 },", "}", "",
              "Substitutions: { g_U1Y : g1, g_SU2L : g2, g_SU3c : g3 }", ""]
    path.write_text("\n".join(lines))
    return names


# --------------------------------------------------------------------------
# Run + parse
# --------------------------------------------------------------------------

def run_pyrate(model: Model, loops: int = 2, pyrate_dir=None, timeout: int = 1800,
               workdir=None, cache_dir=None, use_cache: bool = True) -> BetaSystem:
    """Two-loop beta functions of model computed by PyR@TE, as a BetaSystem (cached on disk).

    Runs PyR@TE in a subprocess on the model file, with only the Python output enabled,
    and parses its RGEs.py. Raises RuntimeError (with PyR@TE's output) if that file is missing.
    """
    cache = Path(cache_dir or default_cache_dir()) / f"pyrate_{model.hash()}.json"
    if use_cache and cache.exists():
        bs = BetaSystem.from_json(json.loads(cache.read_text()))
        if bs.meta.get("loops", 0) >= loops:
            return bs

    exe = find_pyrate(pyrate_dir) / "pyR@TE.py"
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="hf_pyrate_"))
    tmp.mkdir(parents=True, exist_ok=True)
    mfile = (tmp / f"hf{model.hash()}.model").resolve()
    names = write_model_file(model, mfile)
    out = (tmp / "out").resolve()
    cmd = [sys.executable, str(exe), "-m", str(mfile), "-l", str(loops), "-res", str(out),
           "-no-tex", "-no-math", "-no-cpp", "-py", "--RealBasis", "None"]
    log.info("running PyR@TE on %s", mfile.name)
    proc = subprocess.run(cmd, cwd=str(exe.parent), stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout, check=False)
    rges = list(out.glob("*/PythonOutput/RGEs.py"))
    if not rges:
        tail = (proc.stdout + proc.stderr)[-3000:]
        raise RuntimeError(f"PyR@TE did not produce RGEs.py. Output tail:\n{tail}")
    bs = parse_rges(rges[0].read_text(), names, model.coupling_names(), loops)
    bs.meta.update({"loops": loops, "model_file": mfile.read_text(), "pyrate_names": names})
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(cache, json.dumps(bs.to_json()))
    return bs


_DEF = re.compile(r"def beta_(\w+)\(nLoop,([^)]*)\):(.*?)(?=\ndef |\Z)", re.DOTALL)
_LOOP = re.compile(r"if nLoop == (\d+):\s*\n\s*return (.*)")


def parse_rges(text: str, names: dict[str, str], couplings: list[str], loops: int) -> BetaSystem:
    """Parse PyR@TE's RGEs.py into a BetaSystem in our coupling names and order.

    Each beta_<name>(nLoop, ...) function has one return expression per loop order; it is
    expanded with sympy and split into monomials.
    """
    inv = {v: k for k, v in names.items()}
    syms = {pn: sp.Symbol(pn, real=True) for pn in names.values()}
    ordered = [syms[names[c]] for c in couplings]
    loc = dict(syms)
    loc["abs"] = lambda x: x          # all couplings are real
    loc["lambda_"] = syms.get(cp.LAM.pyrate)
    coeffs: dict[str, dict] = {c: {} for c in couplings}
    for m in _DEF.finditer(text):
        pname, body = m.group(1), m.group(3)
        pname = pname.rstrip("_") if pname.rstrip("_") in inv else pname
        if pname not in inv:
            continue
        c = inv[pname]
        for lm in _LOOP.finditer(body):
            L = int(lm.group(1))
            if L > loops:
                continue
            expr = sp.expand(sp.sympify(lm.group(2).strip(), locals=loc))
            d = {}
            if expr != 0:
                for mono, v in sp.Poly(expr, *ordered).terms():
                    d[tuple(mono)] = float(v)
            coeffs[c][L] = d
    for c in couplings:
        for L in range(1, loops + 1):
            coeffs[c].setdefault(L, {})
    return BetaSystem(list(couplings), coeffs, "pyrate", {})
