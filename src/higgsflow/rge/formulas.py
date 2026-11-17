"""General two-loop beta functions of a renormalisable 4D gauge theory.

References
  M.E. Machacek, M.T. Vaughn, Nucl. Phys. B222 (1983) 83; B236 (1984) 221; B249 (1985) 70
  M. Luo, H. Wang, Y. Xiao, Phys. Rev. D 67 (2003) 065019, hep-ph/0211440
  I. Schienbein, F. Staub, T. Steudtner, K. Svirina, Nucl. Phys. B 939 (2019) 1,
      arXiv:1809.06797 (corrections to the above)
  C. Poole, A.E. Thomsen, JHEP 09 (2019) 055, arXiv:1906.04625 (tensor notation used here)
  L. Sartore, I. Schienbein, Comput. Phys. Commun. 261 (2021) 107819, arXiv:2007.12700
      (PyR@TE 3, from which the coefficients are transcribed)

Contractions are written for our index convention (see rge/tensors.py), which is
checked by the closure test and by the SM two-loop RGEs.

Each method returns the *unprojected* beta tensor as a Poly; the engine
projects it onto the couplings.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from ..theory.couplings import QUARTIC
from ..theory.model import Model
from .polytensor import Poly, contract
from .tensors import GAUGE_INDEX, NGI, build_tensors


class TwoLoopFormulas:
    def __init__(self, model: Model):
        self.model = model
        self.names = model.coupling_names()
        nv = self.nv = len(self.names)
        tens = self.tens = build_tensors(model)
        N = self.N = tens["N"]
        self.perm = np.r_[N:2 * N, 0:N]

        gmask = []
        for alpha in range(3):
            m = np.diag([1.0 if GAUGE_INDEX[A] == alpha else 0.0 for A in range(NGI)])
            gmask.append(m)
        G = Poly.zero((NGI, NGI), nv)
        for alpha in range(3):
            G = G + Poly.var(alpha, gmask[alpha], nv, power=2)
        self.G = G
        self.T = Poly.const(tens["T"], nv)
        self.Tt = self.tilde(self.T, (1, 2))
        self.Ts = Poly.const(tens["theta"], nv)
        f = tens["f"]
        self.C2G = Poly.const(np.einsum("ACD,BCD->AB", f, f), nv)

        y = Poly.zero((4, 2 * N, 2 * N), nv)
        for name, arr in tens["ybasis"].items():
            y = y + Poly.var(self.names.index(name), arr, nv)
        self.y = y
        self.yt = self.tilde(y, (1, 2))
        self.l = Poly.var(self.names.index(QUARTIC), tens["L"], nv)
        self._helpers()

    def tilde(self, p: Poly, axes):
        return p.take(self.perm, axes)

    def _helpers(self):
        c = contract
        G, T, Ts, y, yt = self.G, self.T, self.Ts, self.y, self.yt
        self.S2F = c("Aij,Bji->AB", T, T)
        self.S2S = c("Aab,Bba->AB", Ts, Ts)
        self.C2F = c("AB,Aik,Bkj->ij", G, T, T)
        self.C2Ft = self.tilde(self.C2F, (0, 1))
        self.Y2F = c("aik,akj->ij", y, yt)
        self.Y2Ft = self.tilde(self.Y2F, (0, 1))
        self.C2S = c("AB,Aac,Bcb->ab", G, Ts, Ts)
        self.Y2S = c("aij,bji->ab", y, yt)
        # 2-loop gauge
        self.S2FCF = c("ij,Ajk,Bki->AB", self.C2F, T, T)
        self.S2SCS = c("ab,Abc,Bca->AB", self.C2S, Ts, Ts)
        self.S2FYF = c("ij,Ajk,Bki->AB", self.Y2F, T, T)
        # 2-loop fermion
        GCG = c("AC,CD,DB->AB", G, self.C2G, G)
        GSS = c("AC,CD,DB->AB", G, self.S2S, G)
        GSF = c("AC,CD,DB->AB", G, self.S2F, G)
        self.C2FG = c("AB,Aij,Bjk->ik", GCG, T, T)
        self.C2FS = c("AB,Aij,Bjk->ik", GSS, T, T)
        self.C2FF = c("AB,Aij,Bjk->ik", GSF, T, T)
        self.Y2FCF = c("aij,jk,akl->il", y, self.C2F, yt)
        self.Y2FCS = c("aij,bjk,ab->ik", y, yt, self.C2S)
        self.Y2FYF = c("aij,jk,akl->il", y, self.Y2Ft, yt)
        self.Y2FYFt = self.tilde(self.Y2FYF, (0, 1))
        self.Y2FYS = c("ab,aij,bjk->ik", self.Y2S, y, yt)
        self.Y4F = c("aij,bjk,akl,blm->im", y, yt, y, yt)
        # 2-loop scalar
        self.C2SG = c("AB,Aab,Bbc->ac", GCG, Ts, Ts)
        self.C2SS = c("AB,Aab,Bbc->ac", GSS, Ts, Ts)
        self.C2SF = c("AB,Aab,Bbc->ac", GSF, Ts, Ts)
        self.Y2SCF = c("aij,jk,bki->ab", y, self.C2F, yt)
        self.Y2SYF = c("aij,jk,bki->ab", y, self.Y2Ft, yt)
        self.Y4S = c("aij,cjk,bkl,cli->ab", y, yt, y, yt)

    # -- gauge ------------------------------------------------------------
    def gauge(self, loop):
        c, G = contract, self.G
        if loop == 1:
            return self.C2G * (-22 / 3) + self.S2F * (2 / 3) + self.S2S * (1 / 3)
        terms: list[tuple[float, Poly]] = [
            (2, self.S2FCF), (4, self.S2SCS),
            (-68 / 3, c("AC,CD,DB->AB", self.C2G, G, self.C2G)),
            (10 / 3, c("AC,CD,DB->AB", self.C2G, G, self.S2F)),
            (2 / 3, c("AC,CD,DB->AB", self.C2G, G, self.S2S)),
            (-1, self.S2FYF)
        ]
        return _lin(terms)

    # -- Yukawa -----------------------------------------------------------
    def yukawa(self, loop):
        c = contract
        y, yt, G, T = self.y, self.yt, self.G, self.T
        C2S, C2F = self.C2S, self.C2F
        if loop == 1:
            terms: list[tuple[float, Poly]] = [
                (-6, c("aik,kj->aij", y, C2F)),
                (2, c("bik,akl,blj->aij", y, yt, y)),
                (1, c("aik,kj->aij", y, self.Y2Ft)),
                (0.5, c("ba,bij->aij", self.Y2S, y))
            ]
        else:
            terms = [
                (-21 / 2, c("ba,cb,cij->aij", C2S, C2S, y)),
                (12, c("ba,bik,kj->aij", C2S, y, C2F)),
                (-3, c("aik,kl,lj->aij", y, C2F, C2F)),
                (49 / 4, c("ba,bij->aij", self.C2SG, y)),
                (-1 / 4, c("ba,bij->aij", self.C2SS, y)),
                (-1 / 2, c("ba,bij->aij", self.C2SF, y)),
                (-97 / 3, c("aik,kj->aij", y, self.C2FG)),
                (11 / 6, c("aik,kj->aij", y, self.C2FS)),
                (5 / 3, c("aik,kj->aij", y, self.C2FF)),
                (1 / 12, c("bij,cdea,bcde->aij", y, self.l, self.l)),
                (12, c("bik,Akl,AB,blm,Bnj,amn->aij", y, T, G, yt, T, y)),
                (6, c("bik,akl,bc,clj->aij", y, yt, C2S, y)),
                (-12, c("ca,bik,ckl,blj->aij", C2S, y, yt, y)),
                (10, c("bik,kl,alm,bmj->aij", y, C2F, yt, y)),
                (6, c("bik,akl,blm,mj->aij", y, yt, y, C2F)),
                (5 / 2, c("ba,bij->aij", self.Y2SCF, y)),
                (9, c("ik,akj->aij", self.Y2FCS, y)),
                (-1 / 2, c("ik,akj->aij", self.Y2FCF, y)),
                (-7 / 2, c("aik,kl,lj->aij", y, self.Y2Ft, C2F)),
                (-2, c("bcda,bik,ckl,dlj->aij", self.l, y, yt, y)),
                (2, c("bik,ckl,alm,bmn,cnj->aij", y, yt, y, yt, y)),
                (-2, c("bik,ckl,alm,cmn,bnj->aij", y, yt, y, yt, y)),
                (-1 / 2, c("ba,bij->aij", self.Y4S, y)),
                (-2, c("bik,akl,lm,bmj->aij", y, yt, self.Y2F, y)),
                (-1 / 4, c("aik,kj->aij", y, self.Y2FYFt)),
                (-3 / 4, c("ba,bij->aij", self.Y2SYF, y)),
                (-1, c("bik,bc,akl,clj->aij", y, self.Y2S, yt, y)),
                (-3 / 4, c("ik,akj->aij", self.Y2FYS, y)),
            ]
        B = _lin(terms)
        return (B + B.transpose(0, 2, 1)) * 0.5

    # -- quartic ----------------------------------------------------------
    def quartic(self, loop):
        c = contract
        G, Ts, l, y, yt = self.G, self.Ts, self.l, self.y, self.yt
        C2S = self.C2S
        if loop == 1:
            terms: list[tuple[float, Poly]] = [
                (36, c("AB,CD,Aae,Ceb,Bcf,Dfd->abcd", G, G, Ts, Ts, Ts, Ts)),
                (-12, c("ae,ebcd->abcd", C2S, l)),
                (3, c("abef,efcd->abcd", l, l)),
                (2, c("ae,ebcd->abcd", self.Y2S, l)),
                (-12, c("aij,bjk,ckl,dli->abcd", y, yt, y, yt))
            ]
        else:
            GGG = c("AB,CD,EF->ABCDEF", G, G, G)
            th2 = c("Aai,Cib->ACab", Ts, Ts)  # (θ^A θ^C)_ab
            th4 = c("Aai,Cij,Ejk,Bkb->ACEBab", Ts, Ts, Ts, Ts)
            GG = c("AB,CD->ABCD", G, G)

            def Gx(X):
                return c("CE,EF,FD->CD", G, X, G)

            terms = [
                (324, c("ABCDEF,ACEBab,DFcd->abcd", GGG, th4, th2)),
                (-684, c("ai,ACib,ABCD,BDcd->abcd", C2S, th2, GG, th2)),
                (646, c("ACab,AB,CD,BDcd->abcd", th2, G, Gx(self.C2G), th2)),
                (-28, c("ACab,AB,CD,BDcd->abcd", th2, G, Gx(self.S2S), th2)),
                (-32, c("ACab,AB,CD,BDcd->abcd", th2, G, Gx(self.S2F), th2)),
                (12, c("ACae,ABCD,BDbf,efcd->abcd", th2, GG, th2, l)),
                (60, c("ACab,ABCD,BDef,efcd->abcd", th2, GG, th2, l)),
                (6, c("ae,ef,fbcd->abcd", C2S, C2S, l)),
                (-143 / 3, c("ae,ebcd->abcd", self.C2SG, l)),
                (11 / 3, c("ae,ebcd->abcd", self.C2SS, l)),
                (10 / 3, c("ae,ebcd->abcd", self.C2SF, l)),
                (-18, c("Aae,Bbf,AB,efgh,ghcd->abcd", Ts, Ts, G, l, l)),
                (24, c("abef,fg,egcd->abcd", l, C2S, l)),
                (-18, c("ae,ebfg,fgcd->abcd", C2S, l, l)),
                (1 / 3, c("aefg,efgh,hbcd->abcd", l, l, l)),
                (-6, c("abef,eghc,fghd->abcd", l, l, l)),
                # PyR@TE writes T T here; with our index convention the covariant
                # (gauge-invariant) contraction needs the tilded generators.
                (-144, c("ACab,ABCD,Dij,Bjk,ckl,dli->abcd", th2, GG, self.Tt, self.Tt, y, yt)),
                (60, c("ae,ACeb,ABCD,BDcd->abcd", self.Y2S, th2, GG, th2)),
                (10, c("ae,ebcd->abcd", self.Y2SCF, l)),
                (-3, c("abef,fg,egcd->abcd", l, self.Y2S, l)),
                (24, c("ae,eij,bjk,ckl,dli->abcd", C2S, y, yt, y, yt)),
                (-48, c("aij,bjk,ckl,dlm,mi->abcd", y, yt, y, yt, self.C2F)),
                (12, c("aij,ejk,bkl,fli,efcd->abcd", y, yt, y, yt, l)),
                (-2, c("ae,ebcd->abcd", self.Y4S, l)),
                (-3, c("ae,ebcd->abcd", self.Y2SYF, l)),
                (48, c("aij,bjk,ckl,elm,dmn,eni->abcd", y, yt, y, yt, y, yt)),
                (24, c("aij,bjk,ekl,clm,dmn,eni->abcd", y, yt, y, yt, y, yt)),
                (24, c("aij,bjk,ckl,dlm,mi->abcd", y, yt, y, yt, self.Y2F)),
            ]
        B = _lin(terms)
        perms = list(permutations(range(4)))
        out = B.transpose(*perms[0])
        for p in perms[1:]:
            out = out + B.transpose(*p)
        return out * (1 / 24)


def _lin(terms: list[tuple[float, Poly]]) -> Poly:
    """Linear combination sum(coef * term) of polynomial tensors."""
    out: Poly | None = None
    for coef, p in terms:
        if coef == 0:
            continue
        q = p * coef
        out = q if out is None else out + q
    assert out is not None, "empty linear combination"
    return out
