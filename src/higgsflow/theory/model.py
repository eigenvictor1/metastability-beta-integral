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
    """One Yukawa operator y C ψ1 ψ2 Φ (Φ = H or H^*), with its invariant tensor C."""
    name: str
    f1: str            # component names
    f2: str
    conj_higgs: bool   # True -> operator with H^*
    C: np.ndarray      # shape (d1, d2, 2): colour x isospin x Higgs-isospin, flattened fields

    def describe(self) -> str:
        """Readable form, e.g. 'yL4N: L4 N H'."""
        h = "H^*" if self.conj_higgs else "H"
        return f"{self.name}: {self.f1} {self.f2} {h}"


@dataclass(frozen=True)
class Singlet:
    """A complex scalar singlet S that gives the new fermions (except fixed-mass ones) their
    mass: M = y_S v_S/√2. It is integrated out at m_S (tree-level matching λ_SM = λ − κ²/4λ_S).

    m_S, lamS, kappa: mass of the radial mode (GeV), λ_S and κ at μ = m_S.
    yS: y_S, or None for the value fixed by M and m_S: y_S = 2√λ_S M/m_S (from
    m_S² = 2λ_S v_S² and M = y_S v_S/√2).
    """
    m_S: float
    lamS: float
    kappa: float
    yS: float | None = None

    def __post_init__(self) -> None:
        if not (self.m_S > 0 and self.lamS > 0):
            raise ValueError("singlet: m_S and lamS must be positive")

    def y_at(self, M: float) -> float:
        """y_S for the fermion mass M."""
        return self.yS if self.yS is not None else 2 * np.sqrt(self.lamS) * M / self.m_S

    @property
    def shift(self) -> float:
        """λ_full − λ_SM at m_S (tree level): κ²/(4λ_S)."""
        return self.kappa ** 2 / (4 * self.lamS)

    @classmethod
    def from_dict(cls, d: dict) -> Singlet:
        """{"m_S": GeV, "lamS": .., "kappa": .., "yS": number or "derived" (default)}."""
        unknown = set(d) - {"m_S", "lamS", "kappa", "yS"}
        if unknown:
            raise ValueError(f"singlet: unknown keys {sorted(unknown)}")
        ys = d.get("yS", "derived")
        return cls(float(d["m_S"]), float(d["lamS"]), float(d["kappa"]),
                   None if ys in (None, "derived") else float(ys))

    def as_dict(self) -> dict:
        return {"m_S": self.m_S, "lamS": self.lamS, "kappa": self.kappa,
                "yS": "derived" if self.yS is None else self.yS}


class Model:
    """The SM fermions plus new Weyl fermions, with their gauge-invariant Yukawa couplings.

    The Yukawas are found automatically (and cached) the first time they are needed;
    adding fields or excluding couplings resets the cache.
    """

    def __init__(self, name: str | None = None, include_sm_mixing: bool = False):
        """SM-only model. include_sm_mixing: also generate Yukawas between new and SM fields."""
        self.name = name
        self.include_sm_mixing = include_sm_mixing
        self.fields: list[WeylFermion] = [WeylFermion(**{k: getattr(f, k) for k in
                                           ("name", "su3", "su2", "Y", "copies", "sm")})
                                          for f in SM_FIELDS]
        self._yukawas: list[YukawaCoupling] | None = None
        self.excluded_yukawas: set[str] = set()
        self.singlet: Singlet | None = None

    # -- construction -------------------------------------------------------
    @classmethod
    def standard_model(cls, name: str | None = None, **kw) -> Model:
        """Same as Model(): the SM fermion content, no new fields."""
        return cls(name=name, **kw)

    def add(self, *fermions: WeylFermion) -> Model:
        """Add new fields (names must be unique); returns self for chaining."""
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
        """The fields that are not part of the SM."""
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
        if d.get("singlet"):
            m.singlet = Singlet.from_dict(d["singlet"])
        return m

    @property
    def fixed_fields(self) -> list[WeylFermion]:
        """New fields with a fixed mass, lightest first (they decouple there, not at M)."""
        return sorted((f for f in self.new_fields if f.mass is not None), key=lambda f: f.mass or 0.0)

    def with_fixed(self, n: int, scanned: bool = True, singlet: bool = False) -> Model:
        """The theory at a scale between thresholds: the n lightest fixed-mass fields, plus the
        fields decoupling at M if scanned (otherwise the SM fermions only, no mixing), plus
        the singlet S if singlet (and the model has one)."""
        m = Model(name=self.name, include_sm_mixing=self.include_sm_mixing and scanned)
        if scanned:
            m.add(*(f for f in self.new_fields if f.mass is None))
            m.excluded_yukawas = set(self.excluded_yukawas)
        m.add(*self.fixed_fields[:n])
        if singlet:
            m.singlet = self.singlet
        return m

    @property
    def has_yS(self) -> bool:
        """The singlet couples to new fermions (some field gets its mass from S)."""
        return self.singlet is not None and any(f.mass is None for f in self.new_fields)

    def check_fixed(self) -> None:
        """Fields with a fixed mass must not have Yukawa couplings (they only change the gauge
        running between their mass and M)."""
        fixed = self.fixed_fields
        if fixed and set(self.coupling_names()) != set(self.with_fixed(0, singlet=True).coupling_names()):
            raise ValueError("fields with a fixed mass (" + ", ".join(f.name for f in fixed)
                             + ") must not have Yukawa couplings")

    def to_dict(self) -> dict:
        """JSON form of the model part of an input entry (inverse of from_dict)."""
        d: dict = {"name": self.name, "include_sm_mixing": self.include_sm_mixing,
                   "exclude": sorted(self.excluded_yukawas),
                   "fields": [f.as_dict() for f in self.new_fields]}
        if self.singlet is not None:
            d["singlet"] = self.singlet.as_dict()
        return d

    def new_yukawa_names(self) -> list[str]:
        """Names of the Yukawas other than y_t (the parameters of the new physics)."""
        return [y.name for y in self.yukawas() if y.name != cp.TOP_YUKAWA]

    # -- components ---------------------------------------------------------
    def components(self) -> list[Component]:
        """Every copy of every field, with its offset in the flattened fermion index.

        Copies of a field with copies > 1 are named <name>_1, <name>_2, ...
        """
        comps, off = [], 0
        for f in self.fields:
            for c in range(f.copies):
                nm = f.name + (f"_{c + 1}" if f.copies > 1 else "")
                comps.append(Component(nm, f, off))
                off += f.dim
        return comps

    def component(self, name) -> Component:
        """The component with this name (KeyError if absent)."""
        for c in self.components():
            if c.name == name:
                return c
        raise KeyError(name)

    # -- Yukawa search -----------------------------------------------------
    def yukawas(self) -> list[YukawaCoupling]:
        """All Yukawa couplings of the model (y_t first), minus the excluded ones; cached."""
        if self._yukawas is None:
            self._yukawas = [y for y in self._find_yukawas() if y.name not in self.excluded_yukawas]
        return self._yukawas

    def yukawa_invariants(self) -> list[str]:
        """Readable list of the Yukawa operators."""
        return [y.describe() for y in self.yukawas()]

    def _find_yukawas(self) -> list[YukawaCoupling]:
        """Search all pairs: y_t (Q_3 uc_3), new x new, and new x SM if include_sm_mixing."""
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
        names = [y.name for y in out]
        clash = sorted({n for n in names if names.count(n) > 1})
        if clash:      # names drop the underscores, so e.g. fields 'A1' and copy 'A_1' could collide
            raise ValueError(f"Yukawa names {clash} are ambiguous; rename the fields")
        return out

    def _invariants_for(self, a: Component, b: Component, forced_name=None):
        """Yukawas of the pair (a, b) with H and with H^*: hypercharge must sum to zero and
        the gauge-invariant tensor must be unique. For a = b only the symmetric part survives.
        """
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
        """All couplings of the model in state order: g1, g2, g3, y_t, new Yukawas, [y_S], λ,
        [λ_S, κ]."""
        yuk = [y.name for y in self.yukawas()] + ([cp.YS.name] if self.has_yS else [])
        return cp.ordered(yuk, (cp.LAMS.name, cp.KAP.name) if self.singlet is not None else ())

    def label(self) -> str:
        """Folder/plot name: the model's name, or the representations, e.g. '(1,1,0)+(1,2,-1_2)'."""
        if self.name:
            return re.sub(r'[<>:"/\\|?*\s]+', "_", self.name)
        parts = []
        for f in sorted(self.new_fields, key=lambda f: (f.su3, f.su2, f.Y, f.name)):
            parts.append((f"{f.copies}x" if f.copies > 1 else "") + f.label())
        lab = "+".join(parts) if parts else "SM"
        if len(lab) > 120:
            lab = lab[:100] + "_" + hashlib.sha1(lab.encode()).hexdigest()[:8]
        return lab

    def fields_text(self) -> str:
        """New fields with their representations, for plot titles, e.g. 'L4(1,2,-1/2) + 3×N(1,1,0)'."""
        out = " + ".join((f"{f.copies}×" if f.copies > 1 else "") + f"{f.name}{f.rep()}"
                         + (f" @ {f.mass:.3g} GeV" if f.mass is not None else "")
                         for f in self.new_fields) or "SM"
        if self.singlet is not None:
            out += f" + S @ {self.singlet.m_S:.3g} GeV"
        return out

    def as_dict(self) -> dict:
        """Full description written to model.json (fields, Yukawas, exclusions)."""
        return {"name": self.name, "label": self.label(),
                "include_sm_mixing": self.include_sm_mixing,
                **({"singlet": self.singlet.as_dict()} if self.singlet is not None else {}),
                "new_fields": [f.as_dict() for f in self.new_fields],
                "yukawas": [{"name": y.name, "fermions": [y.f1, y.f2],
                             "higgs": "H^*" if y.conj_higgs else "H"} for y in self.yukawas()],
                "excluded_yukawas": sorted(self.excluded_yukawas)}

    def hash(self) -> str:
        """Short hash of as_dict(): the cache key of the beta functions."""
        d = self.as_dict()
        if self.singlet is not None:
            d["singlet"] = True             # the betas do not depend on its parameters
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def anomalies(self):
        """Gauge and mixed anomaly coefficients of all fields (SM + new); all zero if anomaly free."""
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
