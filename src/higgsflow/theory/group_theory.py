"""Group theory for G_SM = SU(3)_c x SU(2)_L x U(1)_Y.

Closed-form invariants (dimension, quadratic Casimir, Dynkin index) are exact
``Fraction``s. Explicit, Hermitian generator matrices are built numerically:

* SU(2): standard spin-j matrices.
* SU(3): irrep (p, q) realised on symmetric, traceless tensors with p upper and
  q lower indices. For p < q the generators are obtained as -t* of the (q, p)
  irrep, so that R and its conjugate always live in conjugate bases and the
  invariant contraction R x Rbar is the identity.

Normalisation: Tr(t^a t^b) = 1/2 delta^{ab} in the fundamental of SU(N).
"""
from __future__ import annotations

from fractions import Fraction
from functools import cache
from itertools import permutations, product

import numpy as np

# --------------------------------------------------------------------------
# Irrep labels
# --------------------------------------------------------------------------

_SU3_NAMES = {
    "1": (0, 0), "3": (1, 0), "3b": (0, 1), "3bar": (0, 1),
    "6": (2, 0), "6b": (0, 2), "6bar": (0, 2), "8": (1, 1),
    "10": (3, 0), "10b": (0, 3), "10bar": (0, 3),
    "15": (2, 1), "15b": (1, 2), "15bar": (1, 2),
    "15p": (4, 0), "15'": (4, 0), "15pb": (0, 4),
    "27": (2, 2),
}


def parse_su3(label) -> tuple[int, int]:
    """Accept '3', '3b'/'3bar', '8', ..., an int dimension, or Dynkin labels (p, q)."""
    if isinstance(label, (tuple, list)):
        p, q = (int(x) for x in label)
        return (p, q)
    s = str(label).strip().lower().replace(" ", "")
    if s in _SU3_NAMES:
        return _SU3_NAMES[s]
    if "-" in s:                       # Dynkin labels written as "p-q", e.g. "2-1"
        ps, qs = s.split("-")
        return (int(ps), int(qs))
    raise ValueError(f"Unknown SU(3) irrep label {label!r}; use Dynkin labels (p, q).")


def parse_su2(label) -> int:
    """Return the dimension of the SU(2) irrep. Accepts 1, 2, '3', 'j=1', 'j=3/2'."""
    if isinstance(label, int):
        dim = label
    else:
        s = str(label).strip().lower().replace(" ", "")
        if s.startswith("j="):
            dim = int(2 * Fraction(s[2:]) + 1)
        else:
            dim = int(s)
    if dim < 1:
        raise ValueError(f"Invalid SU(2) irrep {label!r}")
    return dim


def su3_label(pq: tuple[int, int]) -> str:
    """Short, file-system-safe label for an SU(3) irrep."""
    for name, val in _SU3_NAMES.items():
        if val == pq and name in ("1", "3", "3b", "6", "6b", "8", "10", "10b", "15", "15b", "27"):
            return name
    return f"{pq[0]}-{pq[1]}"


# --------------------------------------------------------------------------
# Closed-form invariants
# --------------------------------------------------------------------------

def su3_dim(pq) -> int:
    p, q = pq
    return (p + 1) * (q + 1) * (p + q + 2) // 2


def su3_casimir(pq) -> Fraction:
    p, q = pq
    return Fraction(p * p + q * q + p * q + 3 * p + 3 * q, 3)


def su3_index(pq) -> Fraction:
    """Dynkin index T(R), with T(3) = 1/2."""
    return Fraction(su3_dim(pq)) * su3_casimir(pq) / 8


def su3_anomaly(pq) -> Fraction:
    """Cubic anomaly coefficient A(R), A(3) = 1."""
    p, q = pq
    return Fraction((p - q) * (p + 2 * q + 3) * (2 * p + q + 3) * su3_dim(pq), 1) / (
        Fraction((1) * (1 + 3) * (2 + 3) * 3)
    ) * 1


def su2_casimir(dim) -> Fraction:
    j = Fraction(dim - 1, 2)
    return j * (j + 1)


def su2_index(dim) -> Fraction:
    return Fraction(dim) * su2_casimir(dim) / 3


# --------------------------------------------------------------------------
# Explicit generators
# --------------------------------------------------------------------------

@cache
def su2_generators(dim: int) -> np.ndarray:
    """Spin-j matrices (Jx, Jy, Jz), shape (3, dim, dim)."""
    if dim == 1:
        return np.zeros((3, 1, 1), dtype=complex)
    j = (dim - 1) / 2
    m = np.array([j - k for k in range(dim)])
    jp = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):
        jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    jm = jp.conj().T
    jx = (jp + jm) / 2
    jy = (jp - jm) / (2j)
    jz = np.diag(m).astype(complex)
    return np.array([jx, jy, jz])


def gell_mann() -> np.ndarray:
    l = np.zeros((8, 3, 3), dtype=complex)
    l[0][0, 1] = l[0][1, 0] = 1
    l[1][0, 1], l[1][1, 0] = -1j, 1j
    l[2][0, 0], l[2][1, 1] = 1, -1
    l[3][0, 2] = l[3][2, 0] = 1
    l[4][0, 2], l[4][2, 0] = -1j, 1j
    l[5][1, 2] = l[5][2, 1] = 1
    l[6][1, 2], l[6][2, 1] = -1j, 1j
    l[7][0, 0] = l[7][1, 1] = 1 / np.sqrt(3)
    l[7][2, 2] = -2 / np.sqrt(3)
    return l


def structure_constants(gens: np.ndarray) -> np.ndarray:
    """f^{abc} from [t^a, t^b] = i f^{abc} t^c with Tr(t^a t^b) = delta/2."""
    n = len(gens)
    f = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            comm = gens[a] @ gens[b] - gens[b] @ gens[a]
            for c in range(n):
                f[a, b, c] = np.real(np.trace(comm @ gens[c]) * 2 / 1j)
    return f


def _orth_basis(M: np.ndarray, tol=1e-9) -> np.ndarray:
    """Orthonormal basis of the column space of a Hermitian projector-like matrix."""
    u, s, _ = np.linalg.svd(M)
    return u[:, s > tol]


def _null_space(M: np.ndarray, tol=1e-9) -> np.ndarray:
    if M.shape[0] == 0:
        return np.eye(M.shape[1], dtype=complex)
    _, s, vh = np.linalg.svd(M)
    rank = int((s > tol * max(1.0, s.max() if len(s) else 1.0)).sum())
    return vh[rank:].conj().T


@cache
def _su3_generators_direct(p: int, q: int) -> np.ndarray:
    """Generators of (p, q) built on symmetric traceless tensors, shape (8, d, d)."""
    if (p, q) == (0, 0):
        return np.zeros((8, 1, 1), dtype=complex)
    t3 = gell_mann() / 2
    t3b = -t3.conj()
    n = p + q
    D = 3 ** n
    slots = [t3] * p + [t3b] * q
    eye3 = np.eye(3)

    def kron_list(mats):
        out = np.array([[1.0 + 0j]])
        for m in mats:
            out = np.kron(out, m)
        return out

    gens_V = np.zeros((8, D, D), dtype=complex)
    for s in range(n):
        for a in range(8):
            gens_V[a] += kron_list([slots[s][a] if k == s else eye3 for k in range(n)])

    # Symmetriser within the p upper and within the q lower slots
    def perm_matrix(perm):
        P = np.zeros((D, D))
        for idx in product(range(3), repeat=n):
            new = tuple(idx[perm[k]] for k in range(n))
            i = np.ravel_multi_index(idx, (3,) * n)
            j = np.ravel_multi_index(new, (3,) * n)
            P[j, i] = 1
        return P

    perms = []
    for pu in permutations(range(p)):
        for pl in permutations(range(q)):
            perms.append(list(pu) + [p + x for x in pl])
    Psym = sum(perm_matrix(pr) for pr in perms) / len(perms)
    S = _orth_basis(Psym)

    # Trace map: contract first upper with first lower index
    if p > 0 and q > 0:
        Dout = 3 ** (n - 2)
        Tr = np.zeros((Dout, D))
        for idx in product(range(3), repeat=n):
            if idx[0] == idx[p]:
                rest = tuple(idx[k] for k in range(n) if k not in (0, p))
                i = np.ravel_multi_index(idx, (3,) * n)
                j = np.ravel_multi_index(rest, (3,) * (n - 2)) if n > 2 else 0
                Tr[j, i] += 1
        N = _null_space(Tr @ S)
        B = S @ N
    else:
        B = S
    if B.shape[1] != su3_dim((p, q)):
        raise RuntimeError(f"SU(3) irrep construction failed for {(p, q)}")
    return np.array([B.conj().T @ g @ B for g in gens_V])


@cache
def su3_generators(pq) -> np.ndarray:
    p, q = pq
    if p >= q:
        return _su3_generators_direct(p, q)
    return -_su3_generators_direct(q, p).conj()


# --------------------------------------------------------------------------
# Invariant tensors (Clebsch-Gordan coefficients)
# --------------------------------------------------------------------------

def invariant_tensors(gen_list: list[np.ndarray], tol=1e-8) -> list[np.ndarray]:
    """All invariant tensors of R_1 x ... x R_n (each given by its generators).

    Returns a list of arrays of shape (d_1, ..., d_n), orthogonalised, each
    normalised so that its largest-modulus entry equals +1.
    """
    dims = [g.shape[1] for g in gen_list]
    nA = gen_list[0].shape[0]
    D = int(np.prod(dims))
    rows = []
    for a in range(nA):
        tot = np.zeros((D, D), dtype=complex)
        for s, g in enumerate(gen_list):
            mats = [np.eye(d) for d in dims]
            mats[s] = g[a]
            m = np.array([[1.0 + 0j]])
            for x in mats:
                m = np.kron(m, x)
            tot += m
        rows.append(tot)
    N = _null_space(np.vstack(rows), tol)
    out = []
    for k in range(N.shape[1]):
        v = N[:, k]
        i = np.argmax(np.abs(v))
        v = v / v[i]
        v[np.abs(v) < 1e-12] = 0
        out.append(v.reshape(dims))
    return out


def anomaly_coefficients(fields) -> dict[str, Fraction]:
    """Gauge and mixed anomalies for a list of left-handed Weyl fields.

    ``fields`` items need .su3 (p, q), .su2 (dim), .Y (Fraction).
    Returns a dict of anomaly name -> value (all must vanish), plus the number
    of SU(2) doublets-equivalent for Witten's anomaly.
    """
    res = {"SU3^3": Fraction(0), "SU3^2 Y": Fraction(0), "SU2^2 Y": Fraction(0),
           "Y^3": Fraction(0), "grav^2 Y": Fraction(0)}
    witten = Fraction(0)
    for f in fields:
        d3, d2 = su3_dim(f.su3), f.su2
        res["SU3^3"] += su3_anomaly(f.su3) * d2
        res["SU3^2 Y"] += su3_index(f.su3) * d2 * f.Y
        res["SU2^2 Y"] += su2_index(d2) * d3 * f.Y
        res["Y^3"] += Fraction(d3 * d2) * f.Y ** 3
        res["grav^2 Y"] += Fraction(d3 * d2) * f.Y
        witten += su2_index(d2) * 2 * d3  # in units where a doublet gives 1
    res["Witten SU2 (mod 2)"] = witten % 2
    return res
