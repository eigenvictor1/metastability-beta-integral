"""Field content: left-handed Weyl fermions and the SM.

Conventions: all fermions are left-handed Weyl spinors (SM right-handed fields
enter as u^c, d^c, e^c); Q = T3 + Y, Higgs Y_H = +1/2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from . import group_theory as gt

_COMPACT = re.compile(r"(?:([A-Za-z][A-Za-z0-9]*):)?(?:(\d+)x)?\(([^,]+),(\d+),([-+]?[\d/.]+)\)"
                      r"(?:@([0-9.]+(?:[eE][-+]?\d+)?))?")


def _frac(x: Fraction | float | int | str) -> Fraction:
    """Hypercharge as an exact Fraction (floats are rounded to denominator <= 1000)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        return Fraction(x).limit_denominator(1000)
    return Fraction(x)


@dataclass
class WeylFermion:
    """A left-handed Weyl fermion multiplet (su3 Dynkin labels, su2 dimension, hypercharge).

    For convenience su3, su2 and Y also accept strings and numbers ("3b", "2", "1/6", 0.5, ...);
    they are normalised to the annotated types on creation.
    """
    name: str
    su3: tuple[int, int] = (0, 0)
    su2: int = 1
    Y: Fraction = Fraction(0)
    copies: int = 1
    sm: bool = False
    mass: float | None = None     # fixed decoupling mass (GeV); None: decouples at the scanned M

    def __post_init__(self) -> None:
        """Normalise su3/su2/Y given as strings or numbers, and check the name."""
        self.su3 = gt.parse_su3(self.su3)
        self.su2 = gt.parse_su2(self.su2)
        self.Y = _frac(self.Y)
        if self.mass is not None:
            self.mass = float(self.mass)
            if not self.mass > 0:
                raise ValueError(f"Field {self.name}: the mass must be positive, not {self.mass}")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", self.name):
            raise ValueError(f"Field name {self.name!r} must be alphanumeric, starting with a letter.")

    @property
    def dim(self) -> int:
        """Dimension of the multiplet (colour x isospin), per copy."""
        return gt.su3_dim(self.su3) * self.su2

    def label(self) -> str:
        """File-system-safe label, e.g. '(3,2,1_6)'."""
        y = self.Y
        ys = str(y.numerator) if y.denominator == 1 else f"{y.numerator}_{y.denominator}"
        return f"({gt.su3_label(self.su3)},{self.su2},{ys})"

    def rep(self) -> str:
        """Representation for display, e.g. '(3,2,1/6)'."""
        return f"({gt.su3_label(self.su3)},{self.su2},{self.Y})"

    @classmethod
    def from_spec(cls, spec: str | dict, default_name: str) -> WeylFermion:
        """Build from a dict {name, su3, su2, Y, copies, mass} or a compact string.

        Compact strings: "(3,2,1/6)", "2x(1,1,0)", "N:(1,1,0)", "Q4:2x(3b,1,-2/3)",
        "X5:(1,5,0)@9.6e3" (fixed mass in GeV: the field decouples there, not at the scanned M).
        SU(3) may be a name ("1", "3", "3b", "8", "6b", ...) or Dynkin labels "p-q".
        """
        if isinstance(spec, dict):
            d = dict(spec)
            d.setdefault("name", default_name)
            if isinstance(d.get("su3"), list):
                d["su3"] = tuple(d["su3"])
            return cls(**{k: d[k] for k in ("name", "su3", "su2", "Y", "copies", "mass") if k in d})
        m = _COMPACT.fullmatch(str(spec).replace(" ", ""))
        if not m:
            raise ValueError(f"Cannot parse field {spec!r}; expected e.g. '(3,2,1/6)' or 'N:2x(1,1,0)'")
        name, copies, su3, su2, y, mass = m.groups()
        return cls(name or default_name, gt.parse_su3(su3), int(su2), _frac(y), int(copies or 1),
                   mass=float(mass) if mass else None)

    def as_dict(self) -> dict:
        """JSON form (inverse of from_spec with a dict)."""
        d = {"name": self.name, "su3": list(self.su3), "su2": self.su2,
             "Y": str(self.Y), "copies": self.copies}
        if self.mass is not None:
            d["mass"] = self.mass
        return d


@dataclass
class Component:
    """One copy of a Weyl field (multiplicity expanded)."""
    name: str           # e.g. 'N', or 'N_2' for the second copy
    field: WeylFermion
    offset: int = 0     # first index in the flattened Weyl component space


SM_FIELDS = [
    WeylFermion("Q", (1, 0), 2, Fraction(1, 6), copies=3, sm=True),     # (3, 2, 1/6)
    WeylFermion("uc", (0, 1), 1, Fraction(-2, 3), copies=3, sm=True),   # (3b, 1, -2/3)
    WeylFermion("dc", (0, 1), 1, Fraction(1, 3), copies=3, sm=True),    # (3b, 1, 1/3)
    WeylFermion("L", (0, 0), 2, Fraction(-1, 2), copies=3, sm=True),    # (1, 2, -1/2)
    WeylFermion("ec", (0, 0), 1, Fraction(1), copies=3, sm=True),       # (1, 1, 1)
]
