"""SM MS-bar couplings at μ = M_t.

NNLO fits of Buttazzo et al., "Investigating the near-criticality of the Higgs
boson", JHEP 12 (2013) 089, arXiv:1307.3536 (v4), eqs. (55)-(60).
Their potential convention V = -m²/2 |H|² + λ |H|⁴ matches ours, and g_Y is
the SM (non-GUT) hypercharge coupling g1.

Default inputs are PDG 2024 central values; the fits are linear expansions
around M_t = 173.34, M_h = 125.15, M_W = 80.384 GeV, α_s(M_Z) = 0.1184.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..theory import couplings as cp

M_PLANCK = 1.22e19  # GeV


@dataclass
class SMInputs:
    """Measured SM inputs (pole masses, α_s); they fix the couplings at μ = M_t."""
    Mt: float = 172.57       # GeV, top pole mass
    Mh: float = 125.20       # GeV, Higgs pole mass
    alpha_s: float = 0.1180  # α_s(M_Z)
    MW: float = 80.3692      # GeV

    def couplings_at_mt(self) -> dict[str, float]:
        """MS-bar g1, g2, g3, y_t, λ at μ = M_t from the Buttazzo et al. NNLO fits."""
        dt = self.Mt - 173.34
        dh = self.Mh - 125.15
        das = (self.alpha_s - 0.1184) / 0.0007
        dw = (self.MW - 80.384) / 0.014
        return {
            cp.G1.name: 0.35830 + 0.00011 * dt - 0.00020 * dw,
            cp.G2.name: 0.64779 + 0.00004 * dt + 0.00011 * dw,
            cp.G3.name: 1.1666 + 0.00314 * das - 0.00046 * dt,
            cp.YT.name: 0.93690 + 0.00556 * dt - 0.00042 * das,
            cp.LAM.name: 0.12604 + 0.00206 * dh - 0.00004 * dt,
        }

    def as_dict(self):
        """The inputs as a dict (for model.json)."""
        return asdict(self)
