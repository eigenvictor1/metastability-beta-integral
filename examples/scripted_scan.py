"""Using higgsflow from Python instead of a JSON file.

Vector-like quarks Qp(3,2,1/6), Up(3,1,2/3) + partners; 1D scan of a common Yukawa at M = 5 TeV.
"""
import numpy as np

from higgsflow import Model, RunSpec, WeylFermion, run_analysis

m = Model(name="VLQ_doublet_singlet")
m.add(WeylFermion("Qp", "3", 2, "1/6"), WeylFermion("Qpc", "3b", 2, "-1/6"),
      WeylFermion("Up", "3", 1, "2/3"), WeylFermion("Upc", "3b", 1, "-2/3"))
print(m.yukawa_invariants())

if __name__ == "__main__":      # needed on Windows for the parallel scan
    spec = RunSpec(m, scan={"y": np.linspace(0, 1, 41)}, fixed={"M": 5e3}, links={"y": "all"},
                   workers=None)
    run_analysis(spec, run_name="VLQ_scripted")   # -> results/VLQ_scripted/
