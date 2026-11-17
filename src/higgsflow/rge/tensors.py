"""Explicit coupling tensors in the doubled Weyl formalism.

Fermions: N left-handed Weyl components + their conjugates (2N indices);
y = diag(Y, Y*), T^A = diag(t^A, -t^A*). Scalars: the four real components of
H = (φ1 + i φ2, φ3 + i φ4)/√2. Gauge index A: 0 = U(1), 1-3 = SU(2), 4-11 = SU(3).
"""
from __future__ import annotations

import numpy as np

from ..theory import group_theory as gt
from ..theory.model import Model

GAUGE_INDEX = [0] + [1] * 3 + [2] * 8        # A -> group (0: U1, 1: SU2, 2: SU3)
NGI = len(GAUGE_INDEX)


def higgs_embedding():
    """U (2x4): H_k = U_{ka} φ_a, and real-basis generators θ^A (12, 4, 4)."""
    s = 1 / np.sqrt(2)
    U = np.array([[s, 1j * s, 0, 0], [0, 0, s, 1j * s]])
    t_list = [np.eye(2) * 0.5] + list(gt.su2_generators(2)) + [np.zeros((2, 2))] * 8
    theta = np.zeros((NGI, 4, 4), dtype=complex)
    order = [0, 2, 1, 3]  # (x0, x1, y0, y1) -> (x0, y0, x1, y1)
    for A, t in enumerate(t_list):
        R = np.block([[-t.imag, -t.real], [t.real, -t.imag]])
        R = R[np.ix_(order, order)]
        theta[A] = -1j * R
        assert np.allclose(U @ (1j * theta[A]), 1j * t @ U)
    return U, theta


def build_tensors(model: Model):
    comps = model.components()
    N = sum(c.field.dim for c in comps)
    t = np.zeros((NGI, 2 * N, 2 * N), dtype=complex)
    for c in comps:
        fld = c.field
        d3, d2 = gt.su3_dim(fld.su3), fld.su2
        g3, g2 = gt.su3_generators(fld.su3), gt.su2_generators(d2)
        blocks = [float(fld.Y) * np.eye(d3 * d2)]
        blocks += [np.kron(np.eye(d3), g2[a]) for a in range(3)]
        blocks += [np.kron(g3[a], np.eye(d2)) for a in range(8)]
        sl = slice(c.offset, c.offset + d3 * d2)
        slc = slice(N + c.offset, N + c.offset + d3 * d2)
        for A, b in enumerate(blocks):
            b = np.asarray(b, dtype=complex)
            t[A][sl, sl] = b
            t[A][slc, slc] = -b.conj()
    # structure constants
    f = np.zeros((NGI, NGI, NGI))
    f[1:4, 1:4, 1:4] = gt.structure_constants(gt.su2_generators(2))
    f[4:, 4:, 4:] = gt.structure_constants(gt.gell_mann() / 2)

    U, theta = higgs_embedding()

    ybasis = {}
    cmap = {c.name: c for c in comps}
    for yk in model.yukawas():
        ca, cb = cmap[yk.f1], cmap[yk.f2]
        Phi = U.conj() if yk.conj_higgs else U          # dΦ_k/dφ_s
        blk = np.einsum("ijk,ks->sij", yk.C, Phi)
        Y = np.zeros((4, N, N), dtype=complex)
        ia = slice(ca.offset, ca.offset + ca.field.dim)
        ib = slice(cb.offset, cb.offset + cb.field.dim)
        if ca.name == cb.name:
            Y[:, ia, ia] += 2 * blk
        else:
            Y[:, ia, ib] += blk
            Y[:, ib, ia] += blk.transpose(0, 2, 1)
        yd = np.zeros((4, 2 * N, 2 * N), dtype=complex)
        yd[:, :N, :N] = Y
        yd[:, N:, N:] = Y.conj()
        ybasis[yk.name] = yd

    d = np.eye(4)
    L = 2 * (np.einsum("ab,cd->abcd", d, d) + np.einsum("ac,bd->abcd", d, d)
             + np.einsum("ad,bc->abcd", d, d))
    return dict(N=N, T=t, f=f, theta=theta, ybasis=ybasis, L=L)


def check_gauge_invariance(tens) -> float:
    """max |violation| of  -T̃^A y^a + y^a T^A + Σ_b y^b θ^A_{ba}  over all couplings."""
    N, T, th = tens["N"], tens["T"], tens["theta"]
    perm = np.r_[N:2 * N, 0:N]
    Tt = T[:, perm][:, :, perm]
    worst = 0.0
    for y in tens["ybasis"].values():
        v = (-np.einsum("Aij,ajk->Aaik", Tt, y) + np.einsum("aij,Ajk->Aaik", y, T)
             + np.einsum("bik,Aba->Aaik", y, th))
        worst = max(worst, float(np.abs(v).max()))
    return worst
