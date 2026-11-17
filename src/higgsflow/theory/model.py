"""Model = SM + new Weyl fermions, with automatic search of gauge-invariant Yukawas.

Yukawa operators: -L ⊃ y * C_{αβk} ψ1_α ψ2_β Φ_k + h.c., Φ = H or H^*, C the
gauge-invariant tensor (colour x isospin) normalised to max|entry| = 1 per
gauge factor. The SM top Yukawa is -L ⊃ y_t ε_{ij} Q^{iα} u^c_α H^j + h.c.
Only Yukawas among the new fields (plus y_t) are generated unless
``include_sm_mixing=True``.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np

from . import couplings as cp
from . import group_theory as gt
from .fields import SM_FIELDS, Component, WeylFermion


@dataclass
class YukawaCoupling:
    name: str
    f1: str            # component names
    f2: str
    conj_higgs: bool   # True -> operator with H^*
    C: np.ndarray      # shape (d1, d2, 2): colour x isospin x Higgs-isospin, flattened fields

    def describe(self) -> str:
        h = "H^*" if self.conj_higgs else "H"
        return f"{self.name}: {self.f1} {self.f2} {h}"


class Model:
    def __init__(self, name: str | None = None, include_sm_mixing: bool = False):
        self.name = name
        self.include_sm_mixing = include_sm_mixing
        self.fields: list[WeylFermion] = [WeylFermion(**{k: getattr(f, k) for k in
                                           ("name", "su3", "su2", "Y", "copies", "sm")})
                                          for f in SM_FIELDS]
        self._yukawas: list[YukawaCoupling] | None = None
        self.excluded_yukawas: set[str] = set()

    # -- construction -------------------------------------------------------
    @classmethod
    def standard_model(cls, name: str | None = None, **kw) -> Model:
        return cls(name=name, **kw)

    def add(self, *fermions: WeylFermion) -> Model:
        for f in fermions:
            names = {c.name for c in self.components()}
            if f.name in {g.name for g in self.fields} or f.name in names:
                raise ValueError(f"Duplicate field name {f.name}")
            self.fields.append(f)
        self._yukawas = None
        return self

    def exclude(self, *coupling_names: str) -> Model:
        """Switch off some of the automatically derived Yukawa couplings."""
        self.excluded_yukawas.update(coupling_names)
        self._yukawas = None
        return self

    @property
    def new_fields(self) -> list[WeylFermion]:
        return [f for f in self.fields if not f.sm]

    @classmethod
    def from_dict(cls, d: dict) -> Model:
        """Inverse of the 'model' part of a JSON spec:
        {"name": ..., "include_sm_mixing": ..., "exclude": [...], "fields": [specs]}."""
        m = cls(name=d.get("name"), include_sm_mixing=d.get("include_sm_mixing", False))
        for i, spec in enumerate(d.get("fields", []), 1):
            m.add(WeylFermion.from_spec(spec, f"F{i}"))
        if d.get("exclude"):
            m.exclude(*d["exclude"])
        return m

    def to_dict(self) -> dict:
        return {"name": self.name, "include_sm_mixing": self.include_sm_mixing,
                "exclude": sorted(self.excluded_yukawas),
                "fields": [f.as_dict() for f in self.new_fields]}

    def new_yukawa_names(self) -> list[str]:
        return [y.name for y in self.yukawas() if y.name != cp.TOP_YUKAWA]

    # -- components ---------------------------------------------------------
    def components(self) -> list[Component]:
        comps, off = [], 0
        for f in self.fields:
            for c in range(f.copies):
                nm = f.name + (f"_{c + 1}" if f.copies > 1 else "")
                comps.append(Component(nm, f, off))
                off += f.dim
        return comps

    def component(self, name) -> Component:
        for c in self.components():
            if c.name == name:
                return c
        raise KeyError(name)

    # -- Yukawa search -----------------------------------------------------
    def yukawas(self) -> list[YukawaCoupling]:
        if self._yukawas is None:
            self._yukawas = [y for y in self._find_yukawas() if y.name not in self.excluded_yukawas]
        return self._yukawas

    def yukawa_invariants(self) -> list[str]:
        return [y.describe() for y in self.yukawas()]

    def _find_yukawas(self) -> list[YukawaCoupling]:
        comps = self.components()
        out = []
        # SM top Yukawa: third-generation Q and u^c with H^* ... (Q u^c H with Y sum -1/2+... )
        q3, u3 = self.component("Q_3"), self.component("uc_3")
        out += self._invariants_for(q3, u3, forced_name=cp.TOP_YUKAWA)
        new = [c for c in comps if not c.field.sm]
        cands = list(combinations_with_replacement(new, 2))
        if self.include_sm_mixing:
            sm = [c for c in comps if c.field.sm]
            cands += [(a, b) for a in new for b in sm]
        for a, b in cands:
            out += self._invariants_for(a, b)
        return out

    def _invariants_for(self, a: Component, b: Component, forced_name=None):
        res = []
        for conj in (False, True):
            yh = Fraction(-1, 2) if conj else Fraction(1, 2)
            if a.field.Y + b.field.Y + yh != 0:
                continue
            C = yukawa_tensor(a.field, b.field, conj)
            if C is None:
                continue
            if a.name == b.name:
                # ψψΦ: only the part symmetric in the two fermion slots survives
                C = 0.5 * (C + C.transpose(1, 0, 2))
                if np.abs(C).max() < 1e-10:
                    continue
                C = C / np.abs(C).max()
            nm = forced_name or cp.new_yukawa_name(a.name, b.name, conj)
            res.append(YukawaCoupling(nm, a.name, b.name, conj, C))
        return res

    # -- bookkeeping ----------------------------------------------------------
    def coupling_names(self) -> list[str]:
        return cp.ordered(y.name for y in self.yukawas())

    def label(self) -> str:
        if self.name:
            return re.sub(r'[<>:"/\\|?*\s]+', "_", self.name)
        parts = []
        for f in sorted(self.new_fields, key=lambda f: (f.su3, f.su2, f.Y, f.name)):
            parts.append((f"{f.copies}x" if f.copies > 1 else "") + f.label())
        lab = "+".join(parts) if parts else "SM"
        if len(lab) > 120:
            lab = lab[:100] + "_" + hashlib.sha1(lab.encode()).hexdigest()[:8]
        return lab

    def as_dict(self) -> dict:
        return {"name": self.name, "label": self.label(),
                "include_sm_mixing": self.include_sm_mixing,
                "new_fields": [f.as_dict() for f in self.new_fields],
                "yukawas": [{"name": y.name, "fermions": [y.f1, y.f2],
                             "higgs": "H^*" if y.conj_higgs else "H"} for y in self.yukawas()],
                "excluded_yukawas": sorted(self.excluded_yukawas)}

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def anomalies(self):
        flat = []
        for f in self.fields:
            flat += [f] * f.copies
        return gt.anomaly_coefficients(flat)

    def check(self) -> list[str]:
        """Warnings about anomalies and fields that cannot get a mass."""
        warns = []
        an = self.anomalies()
        bad = {k: v for k, v in an.items() if v != 0}
        if bad:
            warns.append("Gauge anomalies do not cancel: " + ", ".join(f"{k}={v}" for k, v in bad.items()))
        # vector-like mass partner check
        new = self.new_fields
        for f in new:
            real = (f.su3 == f.su3[::-1]) and f.Y == 0
            partner = any(g.su3 == f.su3[::-1] and g.su2 == f.su2 and g.Y == -f.Y for g in new if g is not f)
            if not (real or partner or f.copies > 1 and real):
                warns.append(f"Field {f.name} {f.label()} has no vector-like partner: "
                             "it can only get a mass from electroweak symmetry breaking.")
        return warns


def yukawa_tensor(f1: WeylFermion, f2: WeylFermion, conj_higgs: bool):
    """Invariant tensor C of shape (d1, d2, 2) for ψ1 ψ2 Φ, or None.

    Flattened field index = colour ⊗ isospin (colour outer).
    """
    g3_1, g3_2 = gt.su3_generators(f1.su3), gt.su3_generators(f2.su3)
    inv3 = gt.invariant_tensors([g3_1, g3_2])
    if len(inv3) != 1:
        return None
    g2_1, g2_2 = gt.su2_generators(f1.su2), gt.su2_generators(f2.su2)
    gH = gt.su2_generators(2)
    if conj_higgs:
        gH = -gH.conj()
    inv2 = gt.invariant_tensors([g2_1, g2_2, gH])
    if len(inv2) != 1:
        return None
    C3, C2 = inv3[0], inv2[0]
    # C[(α,i),(β,j),k] = C3[α,β] C2[i,j,k]
    d31, d32 = C3.shape
    d21, d22, _ = C2.shape
    C = np.einsum("ab,ijk->aibjk", C3, C2).reshape(d31 * d21, d32 * d22, 2)
    return C
