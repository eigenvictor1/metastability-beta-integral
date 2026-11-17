"""Polynomial-valued tensors.

A ``Poly`` is a dict  monomial -> numpy array,  where a monomial is a tuple of
non-negative integer exponents over a fixed list of coupling names. All
coupling tensors in the theory are linear in the couplings (G is linear in
g_alpha^2), so every beta-function tensor is an exact polynomial whose
coefficients are obtained by contracting the numerical basis tensors.
"""
from __future__ import annotations

import string
from collections import defaultdict

import numpy as np

_TOL = 1e-13


class Poly:
    __slots__ = ("nv", "shape", "terms")

    def __init__(self, terms: dict, nv: int, shape: tuple):
        self.terms = terms
        self.nv = nv
        self.shape = tuple(shape)

    # -- constructors -------------------------------------------------------
    @classmethod
    def const(cls, arr, nv):
        arr = np.asarray(arr, dtype=complex)
        return cls({(0,) * nv: arr}, nv, arr.shape)

    @classmethod
    def var(cls, k: int, arr, nv, power=1):
        arr = np.asarray(arr, dtype=complex)
        m = [0] * nv
        m[k] = power
        return cls({tuple(m): arr}, nv, arr.shape)

    @classmethod
    def zero(cls, shape, nv):
        return cls({}, nv, shape)

    # -- arithmetic ---------------------------------------------------------
    def copy(self):
        return Poly({k: v.copy() for k, v in self.terms.items()}, self.nv, self.shape)

    def __add__(self, other):
        if not isinstance(other, Poly):
            raise TypeError
        out = {k: v.copy() for k, v in self.terms.items()}
        for k, v in other.terms.items():
            out[k] = out[k] + v if k in out else v.copy()
        return Poly(out, self.nv, self.shape or other.shape).prune()

    def __neg__(self):
        return Poly({k: -v for k, v in self.terms.items()}, self.nv, self.shape)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, c):
        if isinstance(c, Poly):
            raise TypeError("use contract() for tensor products")
        return Poly({k: v * complex(c) for k, v in self.terms.items()}, self.nv, self.shape)

    __rmul__ = __mul__

    def map(self, fn):
        terms = {k: fn(v) for k, v in self.terms.items()}
        shape = next(iter(terms.values())).shape if terms else self.shape
        return Poly(terms, self.nv, shape)

    def transpose(self, *axes):
        return self.map(lambda a: np.transpose(a, axes))

    def take(self, perm, axes):
        def f(a):
            for ax in axes:
                a = np.take(a, perm, axis=ax)
            return a
        return self.map(f)

    def prune(self, tol=_TOL):
        dead = [k for k, v in self.terms.items() if not np.any(np.abs(v) > tol)]
        for k in dead:
            del self.terms[k]
        return self

    def __repr__(self):
        return f"Poly(shape={self.shape}, {len(self.terms)} monomials)"


def _pair(p1: Poly, s1: str, p2: Poly, s2: str, so: str) -> Poly:
    out: dict[tuple[int, ...], np.ndarray] = {}
    sub = f"{s1},{s2}->{so}"
    for m1, a1 in p1.terms.items():
        for m2, a2 in p2.terms.items():
            m = tuple(x + y for x, y in zip(m1, m2))
            r = np.einsum(sub, a1, a2, optimize=True)
            if m in out:
                out[m] = out[m] + r
            else:
                out[m] = r
    shape = next(iter(out.values())).shape if out else ()
    return Poly(out, p1.nv, shape).prune()


def contract(subscripts: str, *ops: Poly) -> Poly:
    """einsum for Polys, done pairwise along numpy's greedy contraction path."""
    lhs, out = subscripts.replace(" ", "").split("->")
    subs = lhs.split(",")
    assert len(subs) == len(ops)
    if len(ops) == 1:
        return ops[0].map(lambda a: np.einsum(f"{subs[0]}->{out}", a))
    if any(len(o.terms) == 0 for o in ops):
        return Poly.zero((), ops[0].nv)
    sizes = {}
    for o, sb in zip(ops, subs):
        for ch, n in zip(sb, o.shape):
            sizes[ch] = n
    items = list(zip(ops, subs))
    while len(items) > 1:
        best = None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                s1, s2 = items[i][1], items[j][1]
                rest = "".join(s for k, (_, s) in enumerate(items) if k not in (i, j)) + out
                so = "".join(dict.fromkeys(ch for ch in s1 + s2 if ch in rest))
                shared = set(s1) & set(s2)
                cost = (0 if shared else 1, int(np.prod([sizes[c] for c in so])) if so else 1,
                        int(np.prod([sizes[c] for c in set(s1 + s2)])))
                if best is None or cost < best[0]:
                    best = (cost, i, j, so)
        assert best is not None
        _, i, j, so = best
        (p2, s2), (p1, s1) = items.pop(j), items.pop(i)
        items.append((_pair(p1, s1, p2, s2, so), so))
    (p, s), = items
    if s != out:
        p = p.map(lambda a: np.einsum(f"{s}->{out}", a))
    return p


def trace_last(p: Poly, i: int, j: int) -> Poly:
    return p.map(lambda a: np.trace(a, axis1=i, axis2=j))


__all__ = ["Poly", "contract"]
_ = string, defaultdict
