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

from importlib import import_module

# Lazy exports: importing a submodule (e.g. in a scan worker process) does not load
# the whole package (pipeline, plotting, sympy), which keeps worker start-up fast.
_EXPORTS = {
    "RunSpec": ".config",
    "Analysis": ".pipeline",
    "run_analysis": ".pipeline",
    "compute_betas": ".rge.engine",
    "BetaSystem": ".rge.system",
    "M_PLANCK": ".running.boundary",
    "SMInputs": ".running.boundary",
    "Runner": ".running.solver",
    "run_study": ".study",
    "study_from_models": ".study",
    "WeylFermion": ".theory.fields",
    "Model": ".theory.model",
}


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
