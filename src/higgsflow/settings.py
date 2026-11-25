"""Numerical settings, in one place.

``Numerics`` controls the integration and can be set per model in the JSON file
(key "numerics", e.g. {"rtol": 1e-10}); the plot constants are fixed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from .running.boundary import M_PLANCK


@dataclass(frozen=True)
class Numerics:
    """Integration settings of the RG running (per model, JSON key 'numerics')."""
    rtol: float = 1e-9                 # solve_ivp (DOP853) relative tolerance
    atol: float = 1e-11                # solve_ivp absolute tolerance
    n_points: int = 2000               # output grid points between M_t and mu_max
    mu_max: float = M_PLANCK           # upper end of the running (GeV)
    max_rhs_calls: int = 200_000       # a run needing more beta evaluations is 'failed'
    boundary_steps: int = 8            # bisection steps locating region boundaries in scans (0 = off)

    @classmethod
    def from_dict(cls, d: dict | None) -> Numerics:
        """Numerics from a JSON dict; unknown keys are an error, missing keys take the defaults."""
        d = d or {}
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown numerics keys {sorted(unknown)}; allowed: {sorted(known)}")
        return cls(**{k: type(getattr(cls, k))(v) for k, v in d.items()})

    def as_dict(self) -> dict[str, float]:
        """The settings as a dict (for model.json)."""
        return asdict(self)


# Plots
PLOT_DPI = 300                         # resolution of rasterised parts (colour maps)
MAP_GRID = 400                         # 2D maps are interpolated on MAP_GRID x MAP_GRID points

# Threshold matching: c4 is the h^4 coefficient of a polynomial fit of V_heavy(h)
C4_FIT_H = (0.004, 0.06, 40)           # h/M range and number of points
C4_FIT_ORDER = 5                       # fit in h^2, h^4, ..., h^(2*order)
MATCH_ITERATIONS = 3                   # fixed-point iterations of the matching (top mixing)
