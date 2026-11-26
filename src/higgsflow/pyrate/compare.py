"""Compare two BetaSystems (native vs PyR@TE).

1. Convention mapping: a Yukawa may be normalised differently by the two codes
   (different Clebsch-Gordan normalisation): y_native = k_c y_pyrate.
   |k_c| is fixed from the one-loop coefficient of y_c^3 in beta_{y_c}
   (that single coefficient is then a calibration, not a check); the signs are
   chosen to minimise the mismatch. Gauge couplings and lambda are defined
   identically in both codes, so they are never rescaled.
2. Symbolic comparison: monomial-by-monomial, per coupling and loop order.
3. Numeric comparison: both beta vectors at random coupling points.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ..rge.system import LOOP, BetaSystem
from ..theory.couplings import GAUGE, QUARTIC


@dataclass
class Mismatch:
    """One coefficient where native and (mapped) PyR@TE betas differ."""
    coupling: str
    loop: int
    monomial: str
    native: float
    pyrate: float


@dataclass
class ComparisonReport:
    """Result of compare(): normalisation map, per-coupling table, mismatches, numeric check."""
    couplings: list[str]
    calibration: dict[str, float]
    calibrated_terms: list[str]
    table: dict[tuple[str, int], dict]
    mismatches: list[Mismatch]
    numeric_max_rel: float
    mu_I: dict = field(default_factory=dict)
    tol: float = 1e-8

    @property
    def agree(self) -> bool:
        """True if no coefficient differs and the numeric check is within 1e-6."""
        return not self.mismatches and self.numeric_max_rel < 1e-6

    def to_markdown(self) -> str:
        """The report as Markdown (written to betas/comparison.md)."""
        L = ["# Native vs PyR@TE beta functions", "",
             f"**Overall agreement:** {'YES' if self.agree else 'NO'}", ""]
        L.append("## Yukawa normalisation map (y_native = k * y_pyrate)")
        L.append("")
        if self.calibration:
            L += ["| coupling | k |", "|---|---|"]
            L += [f"| {c} | {k:.10g} |" for c, k in self.calibration.items()]
            L.append("")
            L.append("Calibrated (hence not independently checked) coefficients: "
                     + ", ".join(self.calibrated_terms))
        else:
            L.append("No Yukawa couplings besides those with identical normalisation.")
        L += ["", "## Symbolic comparison", "",
              "| coupling | loop | # monomials | max rel. diff | # mismatches |",
              "|---|---|---|---|---|"]
        for (c, lp), row in self.table.items():
            L.append(f"| {c} | {lp} | {row['n']} | {row['maxrel']:.2e} | {row['bad']} |")
        L += ["", f"Numeric check at random points: max relative difference = {self.numeric_max_rel:.2e}", ""]
        if self.mismatches:
            L += ["## Mismatching coefficients", "",
                  "| coupling | loop | monomial | native | PyR@TE (mapped) |", "|---|---|---|---|---|"]
            for m in self.mismatches:
                L.append(f"| {m.coupling} | {m.loop} | {m.monomial} | {m.native:.10g} | {m.pyrate:.10g} |")
        if self.mu_I:
            L += ["", "## Instability scale", ""]
            for k, v in self.mu_I.items():
                L.append(f"- {k}: {v}")
        return "\n".join(L) + "\n"


def _mono_str(m, names):
    """Readable form of a monomial, e.g. 'g1^2*yt'."""
    s = "*".join(f"{n}^{k}" if k > 1 else n for n, k in zip(names, m) if k)
    return s or "1"


def map_conventions(native: BetaSystem, pyrate: BetaSystem, loops: tuple[int, ...] | None = None):
    """Return (k dict, calibrated term names, pyrate system rescaled to native normalisation).

    loops: the loop orders used to choose the Yukawa signs (default: all); with a native
    system computed at one loop only, pass (1,).
    """
    names = native.couplings
    pyrate = pyrate.restrict(names)
    yuks = [c for c in names if c not in GAUGE and c != QUARTIC]
    absk, calibrated = {}, []
    for c in yuks:
        i = names.index(c)
        m = tuple(3 if k == i else 0 for k in range(len(names)))
        an = native.coeffs[c].get(1, {}).get(m)
        ap = pyrate.coeffs[c].get(1, {}).get(m)
        if an and ap and an * ap > 0:
            absk[c] = float(np.sqrt(ap / an))
            calibrated.append(f"beta^(1)_{c}[{c}^3]")
        else:
            absk[c] = 1.0

    def rescale(signs):
        """PyR@TE system in native normalisation for one choice of Yukawa signs.

        With y_native = k y_pyrate, β_c scales as k_c and each monomial by Π k^(-power).
        """
        k = {c: absk[c] * s for c, s in zip(yuks, signs)}
        kv = np.array([k.get(c, 1.0) for c in names])
        new: dict[str, dict] = {}
        for c in names:
            new[c] = {}
            kc = k.get(c, 1.0)
            for L, d in pyrate.coeffs[c].items():
                new[c][L] = {m: v * kc / float(np.prod(kv ** np.array(m))) for m, v in d.items()}
        return k, BetaSystem(names, new, "pyrate (mapped)", pyrate.meta)

    best = None
    sign_sets = product([1, -1], repeat=len(yuks)) if len(yuks) <= 10 else [tuple([1] * len(yuks))]
    for signs in sign_sets:
        k, mapped = rescale(signs)
        score = _diff_score(native, mapped, loops)
        if best is None or score < best[0] - 1e-12:
            best = (score, k, mapped)
    if best is None:
        k, mapped = rescale(())
        return k, calibrated, mapped
    return best[1], calibrated, best[2]


def _diff_score(a: BetaSystem, b: BetaSystem, loops: tuple[int, ...] | None = None) -> float:
    """Sum of |coefficient differences| between two systems (used to choose the Yukawa signs)."""
    s = 0.0
    for c in a.couplings:
        for L in set(a.coeffs[c]) | set(b.coeffs[c]):
            if loops is not None and L not in loops:
                continue
            da, db = a.coeffs[c].get(L, {}), b.coeffs[c].get(L, {})
            for m in set(da) | set(db):
                s += abs(da.get(m, 0.0) - db.get(m, 0.0))
    return s


def compare(native: BetaSystem, pyrate: BetaSystem, tol: float = 1e-8, seed: int = 0) -> ComparisonReport:
    """Compare native and PyR@TE betas after the normalisation map.

    Symbolic: every coefficient, per coupling and loop (relative tolerance tol).
    Numeric: both beta vectors at 50 random coupling points.
    """
    names = native.couplings
    k, calibrated, mapped = map_conventions(native, pyrate)
    table, mism = {}, []
    for c in names:
        for L in sorted(set(native.coeffs[c]) | set(mapped.coeffs[c])):
            da, db = native.coeffs[c].get(L, {}), mapped.coeffs[c].get(L, {})
            monos = set(da) | set(db)
            maxrel, bad = 0.0, 0
            for m in monos:
                a, b = da.get(m, 0.0), db.get(m, 0.0)
                rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
                maxrel = max(maxrel, rel)
                if rel > tol and abs(a - b) > 1e-10:
                    bad += 1
                    mism.append(Mismatch(c, L, _mono_str(m, names), a, b))
            table[(c, L)] = {"n": len(monos), "maxrel": maxrel, "bad": bad}
    # numeric check
    rng = np.random.default_rng(seed)
    fn, fp = native.numeric(2), mapped.numeric(2)
    worst = 0.0
    for _ in range(50):
        x = rng.uniform(0.1, 1.2, len(names)) * rng.choice([-1, 1], len(names))
        a, b = fn(0, x), fp(0, x)
        rel = np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12 / LOOP))
        worst = max(worst, float(rel))
    return ComparisonReport(names, {c: v for c, v in k.items()}, calibrated, table, mism, worst, tol=tol)
