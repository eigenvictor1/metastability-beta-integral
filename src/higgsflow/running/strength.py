"""Overall strength of the new Yukawa couplings, used for the perturbativity limit.

    y_eff = sqrt( λ_max( Σ_a Y^a Y^a† ) / 2 )

with Y^a the Yukawa matrices of the new couplings along the four real Higgs components
(−L ⊃ ½ Y^a_IJ φ_a ψ_I ψ_J). It is gauge and basis independent, equals y for a single N
coupled as y N ℓ·H, and is √6 y for two N coupled with the same y to all three leptons
(one heavy combination coupled with √6 y). A run is non-perturbative once y_eff reaches
√(4π), instead of testing each new coupling on its own.
"""
from __future__ import annotations

import numpy as np

from ..rge.tensors import build_tensors
from ..theory.model import Model


class YukawaStrength:
    """y_eff as a function of the coupling vector of a run."""

    def __init__(self, model: Model, names: list[str]):
        """names: the couplings of the running state, in order (new Yukawas among them)."""
        tens = build_tensors(model)
        n = tens["N"]
        new = set(model.new_yukawa_names())
        self.idx = [k for k, c in enumerate(names) if c in new]
        blocks = [tens["ybasis"][names[k]][:, :n, :n] for k in self.idx]
        # keep only the fermion components that the new couplings touch: small matrices
        touched = np.zeros(n, dtype=bool)
        for b in blocks:
            touched |= np.any(b != 0, axis=(0, 2)) | np.any(b != 0, axis=(0, 1))
        sel = np.flatnonzero(touched)
        self.basis = np.array([b[:, sel][:, :, sel] for b in blocks]) if blocks else np.zeros((0, 4, 0, 0))

    def __call__(self, x: np.ndarray) -> float:
        """y_eff for the coupling vector x (all couplings of the run, in the order given)."""
        if not self.idx:
            return 0.0
        Y = np.tensordot(np.asarray(x)[self.idx], self.basis, axes=1)       # (4, k, k)
        P = np.einsum("aij,akj->ik", Y, Y.conj())
        return float(np.sqrt(max(np.linalg.eigvalsh(P)[-1], 0.0) / 2))
