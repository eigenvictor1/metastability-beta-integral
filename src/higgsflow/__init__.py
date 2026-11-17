"""higgsflow: two-loop RG running of the SM + new Weyl fermions and the Higgs instability scale.

Subpackages
-----------
theory   group theory, fields, models and the Yukawa-invariant search
rge      native two-loop beta functions (tensors, general formulas, engine)
pyrate   PyR@TE 3 bridge, installer and native-vs-PyR@TE comparison
running  boundary conditions, RG solver with threshold, parameter scans
output   results folder layout and plots
"""
__version__ = "0.2.0"

from .config import RunSpec
from .pipeline import Analysis, run_analysis
from .rge.engine import compute_betas
from .rge.system import BetaSystem
from .running.boundary import M_PLANCK, SMInputs
from .running.solver import Runner
from .study import run_study, study_from_models
from .theory.fields import WeylFermion
from .theory.model import Model

__all__ = [
    "M_PLANCK",
    "Analysis",
    "BetaSystem",
    "Model",
    "RunSpec",
    "Runner",
    "SMInputs",
    "WeylFermion",
    "compute_betas",
    "run_analysis",
    "run_study",
    "study_from_models",
]
