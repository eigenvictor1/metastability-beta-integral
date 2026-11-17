from fractions import Fraction

import numpy as np
import pytest

from higgsflow.theory import group_theory as gt
from higgsflow.theory.model import Model


@pytest.mark.parametrize("pq", [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (3, 0), (2, 1), (2, 2)])
def test_su3_irreps(pq):
    t = gt.su3_generators(pq)
    d = gt.su3_dim(pq)
    assert t.shape == (8, d, d)
    for a in range(8):
        assert np.allclose(t[a], t[a].conj().T)
    cas = sum(t[a] @ t[a] for a in range(8))
    assert np.allclose(cas, float(gt.su3_casimir(pq)) * np.eye(d))
    tr = np.einsum("aij,bji->ab", t, t)
    assert np.allclose(tr, float(gt.su3_index(pq)) * np.eye(8))
    # commutation relations with the fundamental structure constants
    f = gt.structure_constants(gt.gell_mann() / 2)
    for a in range(8):
        for b in range(8):
            assert np.allclose(t[a] @ t[b] - t[b] @ t[a], 1j * np.einsum("c,cij->ij", f[a, b], t))


@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
def test_su2_irreps(dim):
    t = gt.su2_generators(dim)
    cas = sum(t[a] @ t[a] for a in range(3))
    assert np.allclose(cas, float(gt.su2_casimir(dim)) * np.eye(dim))
    assert np.isclose(np.trace(t[2] @ t[2]).real, float(gt.su2_index(dim)))


def test_known_values():
    assert gt.su3_index((1, 0)) == Fraction(1, 2)
    assert gt.su3_index((1, 1)) == 3
    assert gt.su3_casimir((1, 1)) == 3
    assert gt.su3_anomaly((2, 0)) == 7
    assert gt.su3_anomaly((1, 1)) == 0


def test_conjugate_contraction_is_identity():
    for pq in [(1, 0), (2, 0), (2, 1)]:
        inv = gt.invariant_tensors([gt.su3_generators(pq), gt.su3_generators(pq[::-1])])
        assert len(inv) == 1
        assert np.allclose(inv[0], np.eye(gt.su3_dim(pq)))


def test_sm_anomaly_free():
    assert all(v == 0 for v in Model().anomalies().values())
