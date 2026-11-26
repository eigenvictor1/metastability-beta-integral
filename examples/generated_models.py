"""Generate models with a rule in Python and run them like a JSON file with many models.

Here: a vector-like lepton doublet pair plus one extra multiplet (1, n, Y) for
n = 1, 2, 3 and a few hypercharges; models without new Yukawas are skipped.
Output: results/generated/ (summary.csv, plots, one data folder per model).
"""
from fractions import Fraction

from higgsflow import Model, WeylFermion, run_study, study_from_models

DEFAULTS = {
    "scan": {"y": {"linspace": [0.0, 1.0, 21]}},
    "links": {"y": "all"},
    "fixed": {"M": 1e3},
    "benchmarks": [{"y": 0.5}],
    "run": {"betas": "pyrate", "plots": False},
}


def models():
    for n in (1, 2, 3):
        for Y in (Fraction(0), Fraction(1, 2), Fraction(1)):
            m = Model()
            m.add(WeylFermion("L4", su2=2, Y="-1/2"), WeylFermion("L4c", su2=2, Y="1/2"),
                  WeylFermion("X", su2=n, Y=Y))
            if Y != 0:
                m.add(WeylFermion("Xc", su2=n, Y=-Y))
            if m.new_yukawa_names():
                yield m


if __name__ == "__main__":        # required on Windows for the process pool
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(run_study(study_from_models("generated", models(), DEFAULTS)))
