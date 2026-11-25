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
    """Tensor whose entries are polynomials in the couplings: {monomial exponents: array}.

    nv is the number of couplings, shape the tensor shape. Supports +, -, scalar *,
    and contraction with contract().
    """
    __slots__ = ("nv", "shape", "terms")

    def __init__(self, terms: dict, nv: int, shape: tuple):
        """terms: {exponent tuple (length nv): complex array of the given shape}."""
        self.terms = terms
        self.nv = nv
        self.shape = tuple(shape)

    # -- constructors -------------------------------------------------------
    @classmethod
    def const(cls, arr, nv):
        """A coupling-independent tensor."""
        arr = np.asarray(arr, dtype=complex)
        return cls({(0,) * nv: arr}, nv, arr.shape)

    @classmethod
    def var(cls, k: int, arr, nv, power=1):
        """arr times (coupling k)^power."""
        arr = np.asarray(arr, dtype=complex)
        m = [0] * nv
        m[k] = power
        return cls({tuple(m): arr}, nv, arr.shape)

    @classmethod
    def zero(cls, shape, nv):
        """The zero tensor of the given shape."""
        return cls({}, nv, shape)

    # -- arithmetic ---------------------------------------------------------
    def copy(self):
        """Deep copy (arrays copied)."""
        return Poly({k: v.copy() for k, v in self.terms.items()}, self.nv, self.shape)

    def __add__(self, other):
        """Sum of two Polys (terms with the same monomial are added; zeros dropped)."""
        if not isinstance(other, Poly):
            raise TypeError
        out = {k: v.copy() for k, v in self.terms.items()}
        for k, v in other.terms.items():
            out[k] = out[k] + v if k in out else v.copy()
        return Poly(out, self.nv, self.shape or other.shape).prune()

    def iadd(self, other: Poly) -> Poly:
        """Add other into self in place (other's arrays are taken over, not copied; zeros
        are not dropped: call prune() at the end). Returns self."""
        for k, v in other.terms.items():
            if k in self.terms:
                self.terms[k] += v
            else:
                self.terms[k] = v
        self.shape = self.shape or other.shape
        return self

    def __neg__(self):
        """Negative of every term."""
        return Poly({k: -v for k, v in self.terms.items()}, self.nv, self.shape)

    def __sub__(self, other):
        """Difference of two Polys."""
        return self + (-other)

    def __mul__(self, c):
        """Multiplication by a number (tensor products go through contract())."""
        if isinstance(c, Poly):
            raise TypeError("use contract() for tensor products")
        return Poly({k: v * complex(c) for k, v in self.terms.items()}, self.nv, self.shape)

    __rmul__ = __mul__

    def map(self, fn):
        """Apply fn to every coefficient array (e.g. a trace or reshuffle)."""
        terms = {k: fn(v) for k, v in self.terms.items()}
        shape = next(iter(terms.values())).shape if terms else self.shape
        return Poly(terms, self.nv, shape)

    def transpose(self, *axes):
        """Permute the tensor axes of every term."""
        return self.map(lambda a: np.transpose(a, axes))

    def take(self, perm, axes):
        """Reorder the entries along the given axes with the index permutation perm."""
        def f(a):
            """Apply np.take with perm along each of the axes."""
            for ax in axes:
                a = np.take(a, perm, axis=ax)
            return a
        return self.map(f)

    def prune(self, tol=_TOL):
        """Drop terms whose array is zero within tol (in place); returns self."""
        dead = [k for k, v in self.terms.items() if not np.any(np.abs(v) > tol)]
        for k in dead:
            del self.terms[k]
        return self

    def __repr__(self):
        """Short description: shape and number of monomials."""
        return f"Poly(shape={self.shape}, {len(self.terms)} monomials)"


def _pair(p1: Poly, s1: str, p2: Poly, s2: str, so: str) -> Poly:
    """Contract two Polys (einsum s1,s2->so): every pair of monomials multiplies, exponents add."""
    out: dict[tuple[int, ...], np.ndarray] = {}
    sub = f"{s1},{s2}->{so}"
    for m1, a1 in p1.terms.items():
        for m2, a2 in p2.terms.items():
            m = tuple(x + y for x, y in zip(m1, m2))
            r = np.einsum(sub, a1, a2, optimize=True)
            if m in out:
                out[m] += r
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
                # ties broken by the number of monomial pairs (memory and work scale with it)
                pairs = len(items[i][0].terms) * len(items[j][0].terms)
                cost = (0 if shared else 1, int(np.prod([sizes[c] for c in so])) if so else 1,
                        pairs, int(np.prod([sizes[c] for c in set(s1 + s2)])))
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
    """Trace of every term over axes i and j."""
    return p.map(lambda a: np.trace(a, axis1=i, axis2=j))


__all__ = ["Poly", "contract"]
_ = string, defaultdict
