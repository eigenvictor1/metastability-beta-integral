"""One-loop matching of the SM couplings at the new-physics threshold μ = M.

All new fermions get the common mass M: vector-like pairs (ψ, ψ^c) a Dirac mass,
fields in a real representation (Y = 0) a Majorana mass. At μ = M (MS-bar):

* gauge couplings: no finite threshold from heavy fermions at μ = M;
* Higgs field renormalisation from heavy loops, Π' = dΠ_hh/dp² at p² = 0, so that
  H_SM = (1 - Π'/2) H_full + ...;
* effective potential of the heavy states (Coleman-Weinberg) gives
  V ⊃ c4 h⁴  with  H = (0, h/√2)  ->  δλ = 4 c4.

Hence  λ_SM = λ_full (1 + 2Π') + 4 c4,   y_t,SM = y_t,full (1 + Π'/2).

References
  S. Coleman, E. Weinberg, Phys. Rev. D 7 (1973) 1888        one-loop effective potential
  S.P. Martin, Phys. Rev. D 70 (2004) 016005, hep-ph/0312092  scalar self-energies, general theory
  The B, A functions at s = 0 are the standard one-loop integrals in closed form.

Conventions as in Martin, hep-ph/0312092 (Weyl fermions, -L ⊃ ½ y^{aIJ} φ_a ψ_I ψ_J
+ ½ M^{IJ} ψ_I ψ_J + c.c.):
  Π_ab(s) = [Re(y^{aIJ} y*_{bIJ}) B_FF(I,J) + Re(y^{aIJ} y^{bIJ}) m_I m_J B_F̄F̄(I,J)] / 16π²,
  B_FF = (x+y-s) B(x,y) - A(x) - A(y),  B_F̄F̄ = 2 B(x,y),
  V_1 = -2 Σ_I f(m_I²) / 16π²,  f(x) = x²/4 (ln x/Q² - 3/2).
The normalisation is checked in the tests against Π(0) = ∂²V/∂φ².
Only loops with at least one heavy fermion contribute to the matching.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import null_space, sqrtm

from ..rge.tensors import build_tensors
from ..settings import C4_FIT_H, C4_FIT_ORDER, MATCH_ITERATIONS
from ..theory import group_theory as gt
from ..theory.couplings import QUARTIC, TOP_YUKAWA
from ..theory.model import Model

LOOP = 16 * np.pi ** 2
H_DIRECTION = 2          # φ3: H = (0, (φ3 + i φ4)/√2) -> the neutral, vev direction


class MassError(ValueError):
    """The new fields cannot all get a common, gauge-invariant mass M."""


def _xlogx(x):
    """x ln x, continued to 0 at x = 0."""
    return x * np.log(x) if x > 0 else 0.0


def _B0(x, y):
    """B(x, y) at s = 0, Q² = M² = 1 (closed form)."""
    if abs(x - y) < 1e-12 * max(x, y, 1.0):
        return -np.log(x)
    return 1.0 - (_xlogx(x) - _xlogx(y)) / (x - y)


def _B0p(x, y):
    """dB/ds at s = 0 (closed form)."""
    if abs(x - y) < 1e-6 * max(x, y, 1.0):
        return 1.0 / (6.0 * x)
    xy_log = 0.0 if x == 0 or y == 0 else 2 * x * y * np.log(x / y)
    return (x ** 2 - y ** 2 - xy_log) / (2.0 * (x - y) ** 3)


def _A(x):
    """One-loop tadpole function A(x) = x (ln x - 1) at Q = M (0 for massless states)."""
    return x * (np.log(x) - 1) if x > 0 else 0.0


# --------------------------------------------------------------------------
# Mass matrix of the new fields
# --------------------------------------------------------------------------

def _gauge_inv(f1, f2):
    """Invariant tensor of R1 x R2 (no Higgs), normalised to C C^dagger = 1, or None."""
    if f1.Y + f2.Y != 0:
        return None
    c3 = gt.invariant_tensors([gt.su3_generators(f1.su3), gt.su3_generators(f2.su3)])
    c2 = gt.invariant_tensors([gt.su2_generators(f1.su2), gt.su2_generators(f2.su2)])
    if len(c3) != 1 or len(c2) != 1:
        return None
    C = np.kron(c3[0], c2[0])
    s = np.linalg.svd(C, compute_uv=False)
    return C / s[0]


def mass_matrix(model: Model) -> np.ndarray:
    """Weyl mass matrix M0/M (N x N, symmetric) with all new states at mass 1.

    Pairs each new field with itself (Majorana: real representation, symmetric
    invariant) or with an unpaired new field in the conjugate representation (Dirac).
    """
    comps = model.components()
    N = sum(c.field.dim for c in comps)
    M0 = np.zeros((N, N), dtype=complex)
    new = [c for c in comps if not c.field.sm]
    paired = set()
    for a in new:
        if a.name in paired:
            continue
        sl_a = slice(a.offset, a.offset + a.field.dim)
        C = _gauge_inv(a.field, a.field)
        if C is not None and np.allclose(C, C.T):
            M0[sl_a, sl_a] = C
            paired.add(a.name)
            continue
        partner = next((b for b in new if b.name not in paired and b.name != a.name
                        and _gauge_inv(a.field, b.field) is not None), None)
        if partner is None:
            raise MassError(f"{a.name} {a.field.label()} has no gauge-invariant mass term "
                            "with the other new fields (no vector-like partner, not a real rep)")
        C = _gauge_inv(a.field, partner.field)
        sl_b = slice(partner.offset, partner.offset + partner.field.dim)
        M0[sl_a, sl_b] = C
        M0[sl_b, sl_a] = C.T
        paired |= {a.name, partner.name}
    return M0


def heavy_basis(M0: np.ndarray):
    """Unitary V with V^T M0 V = diag(1,...,1, 0,...,0) (heavy states first)."""
    P = M0.conj().T @ M0
    w = np.linalg.eigvalsh(P)
    if not np.all((np.abs(w) < 1e-9) | (np.abs(w - 1) < 1e-9)):
        raise MassError("new-field masses are not all equal to M")
    EH = null_space(P - np.eye(len(P)))
    EL = null_space(P)
    S = EH.T @ M0 @ EH                     # symmetric unitary
    U = sqrtm(S)                           # symmetric, S = U U^T
    VH = EH @ U.conj()
    V = np.hstack([VH, EL])
    D = V.T @ M0 @ V
    assert np.allclose(D, np.diag([1.0] * EH.shape[1] + [0.0] * EL.shape[1]), atol=1e-9)
    return V, EH.shape[1]


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Matching:
    """The one-loop threshold correction applied at μ = M (stored with each run)."""
    delta_lambda: float          # 4 c4, from the heavy-fermion effective potential
    dZ_H: float                  # Π' (Higgs field renormalisation, includes 1/16π²)
    lam_SM: float                # λ just below M (SM side)
    lam_full: float              # λ just above M (full theory)
    yt_SM: float | None = None
    yt_full: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        """The fields as a dict (written to the run CSV header)."""
        return asdict(self)


class ThresholdMatching:
    """Precomputes the mass basis; ``match()`` maps SM values at M to full-theory values."""

    def __init__(self, model: Model):
        """Precompute the Weyl-block Yukawa tensors, the mass matrix and the heavy mass basis."""
        self.model = model
        tens = build_tensors(model)
        N = tens["N"]
        self.N = N
        self.yuk = {name: arr[:, :N, :N] for name, arr in tens["ybasis"].items()}   # Weyl blocks
        self.M0 = mass_matrix(model)
        self.V, self.nH = heavy_basis(self.M0)

    # -- coupling tensors -----------------------------------------------------
    def yukawa(self, values: dict) -> np.ndarray:
        """Yukawa tensor Y^a_IJ (4 real Higgs directions, Weyl indices) for the given coupling values."""
        Y = np.zeros((4, self.N, self.N), dtype=complex)
        for name, arr in self.yuk.items():
            Y += float(values.get(name, 0.0)) * arr
        return Y

    # -- effective potential --------------------------------------------------
    def V_heavy(self, h: float, Y: np.ndarray, a: int = H_DIRECTION) -> float:
        """One-loop effective potential of the heavy fermions at Higgs value h (units of M).

        The heavy masses are the largest singular values of M0 + h Y^a; V = -2 Σ f(m²)/16π².
        """
        m = np.linalg.svd(self.M0 + h * Y[a], compute_uv=False)
        mh = np.sort(m)[::-1][: self.nH]               # the heavy states
        x = mh ** 2
        return float(-2 * np.sum(x ** 2 / 4 * (np.log(x) - 1.5)) / LOOP)

    def c4(self, Y: np.ndarray, a: int = H_DIRECTION) -> float:
        """Coefficient of h⁴ in V_heavy (h in units of M), by a polynomial fit in h²."""
        h = np.linspace(*C4_FIT_H)
        f = np.array([self.V_heavy(x, Y, a) for x in h]) - self.V_heavy(0.0, Y, a)
        u = h ** 2
        A = np.vstack([u ** k for k in range(1, C4_FIT_ORDER + 1)]).T
        coef, *_ = np.linalg.lstsq(A, f, rcond=None)
        return float(coef[1])

    # -- self-energy ----------------------------------------------------------
    def self_energy(self, Y: np.ndarray, a: int = H_DIRECTION, b: int = H_DIRECTION,
                    derivative: bool = True, heavy_only: bool = True) -> float:
        """Π'_ab(0) (derivative=True) or Π_ab(0), from fermion loops, including 1/16π²."""
        ya = self.V.T @ Y[a] @ self.V
        yb = self.V.T @ Y[b] @ self.V
        m = np.array([1.0] * self.nH + [0.0] * (self.N - self.nH))
        x = m ** 2
        tot = 0.0
        idx = np.argwhere((np.abs(ya) > 1e-14) | (np.abs(yb) > 1e-14))
        for i, j in idx:
            if heavy_only and m[i] == 0 and m[j] == 0:
                continue
            if x[i] == 0 and x[j] == 0:
                continue
            t1 = np.real(ya[i, j] * np.conj(yb[i, j]))
            t2 = np.real(ya[i, j] * yb[i, j]) * m[i] * m[j]
            B, Bp = _B0(x[i], x[j]), _B0p(x[i], x[j])
            if derivative:
                tot += t1 * (-B + (x[i] + x[j]) * Bp) + t2 * 2 * Bp
            else:
                tot += t1 * ((x[i] + x[j]) * B - _A(x[i]) - _A(x[j])) + t2 * 2 * B
        return tot / LOOP

    # -- matching -------------------------------------------------------------
    def match(self, sm_values: dict[str, float], new_values: dict[str, float],
              iterations: int = MATCH_ITERATIONS) -> tuple[dict[str, float], Matching]:
        """Full-theory couplings at μ = M from SM couplings and new Yukawas (both at M).

        Returns (all full-theory couplings at M, the correction applied).
        """
        full = {**sm_values, **new_values}
        dlam = dZ = 0.0
        for _ in range(iterations):      # y_t enters the heavy spectrum only if the top mixes
            Y = self.yukawa(full)
            dZ = self.self_energy(Y)
            dlam = 4 * self.c4(Y)
            full[QUARTIC] = (sm_values[QUARTIC] - dlam) / (1 + 2 * dZ)
            if TOP_YUKAWA in sm_values:
                full[TOP_YUKAWA] = sm_values[TOP_YUKAWA] / (1 + dZ / 2)
        record = Matching(dlam, dZ, sm_values[QUARTIC], full[QUARTIC],
                          sm_values.get(TOP_YUKAWA), full.get(TOP_YUKAWA))
        return full, record
