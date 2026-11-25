"""Explicit coupling tensors in the doubled Weyl formalism.

Fermions: N left-handed Weyl components + their conjugates (2N indices);
y = diag(Y, Y*), T^A = diag(t^A, -t^A*). Scalars: the four real components of
H = (φ1 + i φ2, φ3 + i φ4)/√2, then (with a singlet) S = (s1 + i s2)/√2 as φ5, φ6.
Gauge index A: 0 = U(1), 1-3 = SU(2), 4-11 = SU(3).
"""
from __future__ import annotations

import numpy as np

from ..theory import couplings as cp
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
    """All numerical tensors of model in the doubled Weyl basis.

    Returns dict: N (number of Weyl components), NS (number of real scalars: 4, or 6 with
    the singlet), T (12, 2N, 2N) fermion generators, f (12,12,12) structure constants,
    theta (12, NS, NS) scalar generators, ybasis {coupling: (NS, 2N, 2N)} Yukawa tensor per
    unit coupling, quartics {coupling: (NS,)*4} quartic tensor per unit coupling
    (V = L_abcd φ_a φ_b φ_c φ_d / 4!; λ (H†H)², λ_S |S|⁴, κ (H†H)|S|²), L = quartics["lam"].
    """
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

    U, theta4 = higgs_embedding()
    singlet = model.singlet is not None
    NS = 6 if singlet else 4
    theta = np.zeros((NGI, NS, NS), dtype=complex)
    theta[:, :4, :4] = theta4                        # S is a gauge singlet

    ybasis = {}
    cmap = {c.name: c for c in comps}
    for yk in model.yukawas():
        ca, cb = cmap[yk.f1], cmap[yk.f2]
        Phi = U.conj() if yk.conj_higgs else U          # dΦ_k/dφ_s
        blk = np.einsum("ijk,ks->sij", yk.C, Phi)
        Y = np.zeros((NS, N, N), dtype=complex)
        blk = np.concatenate([blk, np.zeros((NS - 4, *blk.shape[1:]))]) if NS > 4 else blk
        ia = slice(ca.offset, ca.offset + ca.field.dim)
        ib = slice(cb.offset, cb.offset + cb.field.dim)
        if ca.name == cb.name:
            Y[:, ia, ia] += 2 * blk
        else:
            Y[:, ia, ib] += blk
            Y[:, ib, ia] += blk.transpose(0, 2, 1)
        ybasis[yk.name] = _doubled(Y)
    if model.has_yS:
        # -L ⊃ ½ y_S S ψ M0 ψ + h.c. with S = (φ5 + i φ6)/√2; fixed-mass fields keep a bare mass
        from ..running.threshold import mass_matrix
        at_M = model.with_fixed(0)
        M0 = mass_matrix(at_M)
        full = {c.name: c.offset for c in comps}
        idx = np.concatenate([full[c.name] + np.arange(c.field.dim) for c in at_M.components()])
        Y = np.zeros((NS, N, N), dtype=complex)
        Y[4][np.ix_(idx, idx)] = M0 / np.sqrt(2)
        Y[5][np.ix_(idx, idx)] = 1j * M0 / np.sqrt(2)
        ybasis[cp.YS.name] = _doubled(Y)

    H, S = np.zeros(NS), np.zeros(NS)
    H[:4], S[4:] = 1, 1
    dH, dS = np.diag(H), np.diag(S)
    quartics = {cp.QUARTIC: _pairings(dH, dH)}
    if singlet:
        quartics[cp.LAMS.name] = _pairings(dS, dS)
        quartics[cp.KAP.name] = 0.5 * (_pairings(dH, dS) + _pairings(dS, dH))
    return dict(N=N, NS=NS, T=t, f=f, theta=theta, ybasis=ybasis, quartics=quartics,
                L=quartics[cp.QUARTIC])


def _doubled(Y: np.ndarray) -> np.ndarray:
    """diag(Y, Y*) in the doubled fermion basis."""
    ns, N = Y.shape[0], Y.shape[1]
    yd = np.zeros((ns, 2 * N, 2 * N), dtype=complex)
    yd[:, :N, :N] = Y
    yd[:, N:, N:] = Y.conj()
    return yd


def _pairings(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """2 Σ over the three pairings of (d1)_xy (d2)_zw, symmetrised: for d1 = d2 = δ on n real
    fields this is the tensor of (φ·φ)²/4; for (δ_H, δ_S) + (δ_S, δ_H), times ½, that of
    (φ_H·φ_H)(φ_S·φ_S)/4 = (H†H)|S|²."""
    e = np.einsum
    out = (e("ab,cd->abcd", d1, d2) + e("ac,bd->abcd", d1, d2) + e("ad,bc->abcd", d1, d2))
    out = out + out.transpose(2, 3, 0, 1)
    return out


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
