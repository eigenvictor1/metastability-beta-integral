"""The native engine must reproduce the textbook two-loop SM RGEs (top Yukawa only)."""
import sympy as sp

from higgsflow.rge.engine import compute_betas
from higgsflow.theory.model import Model

g1, g2, g3, yt, lam = sp.symbols("g1 g2 g3 yt lambda", real=True)
R = sp.Rational

EXPECTED = {
    ("g1", 1): R(41, 6) * g1**3,
    ("g1", 2): R(199, 18) * g1**5 + R(9, 2) * g1**3 * g2**2 + R(44, 3) * g1**3 * g3**2 - R(17, 6) * g1**3 * yt**2,
    ("g2", 1): -R(19, 6) * g2**3,
    ("g2", 2): R(3, 2) * g1**2 * g2**3 + R(35, 6) * g2**5 + 12 * g2**3 * g3**2 - R(3, 2) * g2**3 * yt**2,
    ("g3", 1): -7 * g3**3,
    ("g3", 2): R(11, 6) * g1**2 * g3**3 + R(9, 2) * g2**2 * g3**3 - 26 * g3**5 - 2 * g3**3 * yt**2,
    ("yt", 1): yt * (R(9, 2) * yt**2 - 8 * g3**2 - R(9, 4) * g2**2 - R(17, 12) * g1**2),
    ("yt", 2): yt * (-12 * yt**4 + yt**2 * (36 * g3**2 + R(225, 16) * g2**2 + R(131, 16) * g1**2 - 12 * lam)
                     + 6 * lam**2 - 108 * g3**4 - R(23, 4) * g2**4 + R(1187, 216) * g1**4
                     + 9 * g2**2 * g3**2 + R(19, 9) * g1**2 * g3**2 - R(3, 4) * g1**2 * g2**2),
    ("lam", 1): 24 * lam**2 + 12 * lam * yt**2 - 6 * yt**4 - 3 * lam * (3 * g2**2 + g1**2)
                + R(3, 8) * (2 * g2**4 + (g2**2 + g1**2) ** 2),
    ("lam", 2): (-312 * lam**3 - 144 * lam**2 * yt**2 + 36 * lam**2 * (3 * g2**2 + g1**2)
                 - 3 * lam * yt**4 + 80 * g3**2 * lam * yt**2 + R(45, 2) * g2**2 * lam * yt**2
                 + R(85, 6) * g1**2 * lam * yt**2 - R(73, 8) * g2**4 * lam + R(39, 4) * g1**2 * g2**2 * lam
                 + R(629, 24) * g1**4 * lam + 30 * yt**6 - 32 * g3**2 * yt**4 - R(8, 3) * g1**2 * yt**4
                 - R(9, 4) * g2**4 * yt**2 + R(21, 2) * g1**2 * g2**2 * yt**2 - R(19, 4) * g1**4 * yt**2
                 + R(305, 16) * g2**6 - R(289, 48) * g1**2 * g2**4 - R(559, 48) * g1**4 * g2**2
                 - R(379, 48) * g1**6),
}


def test_sm_two_loop():
    bs = compute_betas(Model())
    for (c, L), expected in EXPECTED.items():
        assert sp.expand(bs.expr(c, L) - expected) == 0, (c, L, bs.expr(c, L))


def test_closure_and_invariance():
    bs = compute_betas(Model())
    assert bs.meta["gauge_invariance"] < 1e-10
    for d in bs.meta["closure"].values():
        assert d["yukawa_residual"] < 1e-9 and d["quartic_residual"] < 1e-9
