"""BetaSystem: beta functions as exact polynomials, per coupling and loop order.

Common container for the native engine and PyR@TE; everything downstream
(comparison, running, LaTeX export) works on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import sympy as sp

from ..theory.couplings import get as get_coupling

LOOP = 16 * np.pi ** 2


@dataclass
class BetaSystem:
    """Beta functions as polynomials, per coupling and loop order.

    ``coeffs[c][L]`` maps a monomial (tuple of exponents over ``couplings``)
    to its coefficient, *without* the 1/(16π²)^L loop factor.
    """
    couplings: list[str]
    coeffs: dict[str, dict[int, dict[tuple, float]]]
    source: str
    meta: dict = field(default_factory=dict)

    # -- sympy views --------------------------------------------------------
    def symbols(self):
        return [sp.Symbol(_sym_name(c), real=True) for c in self.couplings]

    def expr(self, c: str, loop: int, exact=True) -> sp.Expr:
        syms = self.symbols()
        e = 0
        for mono, v in self.coeffs[c].get(loop, {}).items():
            term = to_rational(v) if exact else sp.Float(v)
            for s, k in zip(syms, mono):
                if k:
                    term = term * s ** k
            e += term
        return sp.expand(e)

    # -- numerics ------------------------------------------------------------
    def numeric(self, loops: int = 2):
        """Return f(t, x) = dx/d ln μ for the coupling vector x (order = couplings)."""
        idx = []
        for c in self.couplings:
            monos, vals = [], []
            for L in range(1, loops + 1):
                for m, v in self.coeffs[c].get(L, {}).items():
                    monos.append(m)
                    vals.append(v / LOOP ** L)
            idx.append((np.array(monos, dtype=float).reshape(-1, len(self.couplings)),
                        np.array(vals)))

        def f(t, x):
            x = np.asarray(x, dtype=float)
            out = np.empty(len(idx))
            for k, (M, V) in enumerate(idx):
                if len(V) == 0:
                    out[k] = 0.0
                else:
                    out[k] = np.dot(V, np.prod(x[None, :] ** M, axis=1))
            return out
        return f

    def restrict(self, couplings: list[str]) -> BetaSystem:
        """Re-express on a subset/permutation of the couplings (missing ones set to 0)."""
        pos = [self.couplings.index(c) if c in self.couplings else None for c in couplings]
        drop = [k for k, c in enumerate(self.couplings) if c not in couplings]
        new: dict[str, dict[int, dict]] = {}
        for c in couplings:
            new[c] = {}
            for L, d in self.coeffs.get(c, {}).items():
                nd: dict = {}
                for m, v in d.items():
                    if any(m[k] for k in drop):
                        continue
                    nm = tuple(m[p] if p is not None else 0 for p in pos)
                    nd[nm] = nd.get(nm, 0.0) + v
                new[c][L] = nd
        return BetaSystem(list(couplings), new, self.source, dict(self.meta))

    def to_json(self) -> dict:
        return {"couplings": self.couplings, "source": self.source, "meta": self.meta,
                "coeffs": {c: {str(L): [[list(m), v] for m, v in d.items()]
                               for L, d in Ls.items()} for c, Ls in self.coeffs.items()}}

    @classmethod
    def from_json(cls, d: dict) -> BetaSystem:
        co = {c: {int(L): {tuple(m): v for m, v in lst} for L, lst in Ls.items()}
              for c, Ls in d["coeffs"].items()}
        return cls(d["couplings"], co, d["source"], d.get("meta", {}))

    def latex(self) -> str:
        syms = self.symbols()
        lines = [r"\documentclass{article}", r"\usepackage{amsmath,breqn}",
                 r"\begin{document}",
                 rf"\section*{{Two-loop beta functions ({self.source})}}",
                 r"$\beta_x \equiv \frac{dx}{d\ln\mu} = \sum_L \frac{\beta^{(L)}_x}{(16\pi^2)^L}$."]
        for c, s in zip(self.couplings, syms):
            for L in sorted(self.coeffs[c]):
                e = self.expr(c, L)
                lines.append(r"\begin{dmath*}")
                lines.append(rf"\beta^{{({L})}}_{{{sp.latex(s)}}} = {sp.latex(e)}")
                lines.append(r"\end{dmath*}")
        lines.append(r"\end{document}")
        return "\n".join(lines)


def _sym_name(c: str) -> str:
    try:
        return get_coupling(c).symbol
    except KeyError:
        return c


def to_rational(v: float, max_den=10 ** 6):
    fr = Fraction(v).limit_denominator(max_den)
    if abs(float(fr) - v) <= 1e-9 * max(1.0, abs(v)):
        return sp.Rational(fr.numerator, fr.denominator)
    return sp.Float(v)
