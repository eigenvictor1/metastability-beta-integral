# higgsflow

Two-loop RG running of the Standard Model extended by new left-handed Weyl fermions,
given by their SU(3)c × SU(2)L × U(1)Y representations, and the Higgs instability
scale μ_I defined by λ(μ_I) = 0.

* The native engine evaluates the general two-loop β-functions exactly, as polynomials
  built from explicit group-theory tensors.
* The same β-functions can also be computed with [PyR@TE 3](https://github.com/LSartore/pyrate)
  and compared coefficient by coefficient.
* Each analysis writes to `results/<model name or representation label>/`.

## Install (Windows, PowerShell)

```powershell
winget install Python.Python.3.12
cd C:\Users\victo\dev\metastability-beta-integral
python3 -m pip install -e .[pyrate,dev]
python3 -m higgsflow install-pyrate   # optional: downloads PyR@TE to ~/.higgsflow/pyrate
python3 -m pytest
```

`higgsflow ...` and `python3 -m higgsflow ...` are equivalent. Use the `python3 -m` form if
Windows says `higgsflow` is not recognized (its Scripts folder is not on PATH).

## Usage

Every input is a JSON file with the same structure, whether it holds one model or many
(see `models/`):

```powershell
higgsflow run models\vll_doublet_singlet.json     # one model
higgsflow run models\leptons.json                 # two models
higgsflow run models\five_models.json --workers 0 # 0 = all cores

higgsflow yukawas models\leptons.json             # Yukawas found for each model
higgsflow betas models\vll_doublet_singlet.json --source both   # betas + native/PyR@TE comparison
```

Results go to `results\<name of the file>\` (see Output). With one model, the workers share its scan points. With several, each worker takes
whole models.

From Python: `examples/scripted_scan.py` analyses one model, and
`examples/generated_models.py` generates models from a rule and runs them like a file.

### File structure

```json
{
  "name": "leptons",
  "defaults": {
    "scan": {"y": {"linspace": [0, 1, 21]}},
    "links": {"y": "all"},
    "fixed": {"M": 1e3},
    "benchmarks": [{"y": 0.5}],
    "run": {"betas": "native", "plots": false}
  },
  "models": [
    {"name": "VLL", "fields": ["(1,2,-1/2)", "(1,2,1/2)", "(1,1,0)"]},
    {"name": "triplet_VLL", "fields": ["S:(1,3,0)", "(1,2,-1/2)", "(1,2,1/2)"],
     "fixed": {"M": 2e3}, "run": {"plots": true}}
  ],
  "workers": 0
}
```

| key | where | meaning |
|---|---|---|
| `name` | top | name of the run, i.e. its folder in `results/` (default: the file name) |
| `defaults` | top | any per-model key below, shared by all models |
| `models` | top | **required** list, one entry per model |
| `workers` | top | number of processes; 0 = all cores (default) |
| `name` | model | name of the theory in plot and folder names (default: a label from the representations) |
| `fields` | model | **required** list of new left-handed Weyl fields |
| `include_sm_mixing`, `exclude` | model | also generate new–SM Yukawas / switch off some Yukawas |
| `scan` | model | 1 or 2 parameters: `M` (GeV) and/or Yukawas at μ = M |
| `links` | model | `{alias: [Yukawas]}`, or `"all"`: one parameter drives several Yukawas |
| `fixed` | model | values of parameters that aren't scanned |
| `examples` | model | points drawn in `running-<theory>.pdf` |
| `benchmarks` | model | points whose μ_I goes into the summary |
| `run` | model | `betas` (`native`/`pyrate`/`both`), `loops`, `M` (default threshold), `plots`, `threshold` (`one-loop`/`tree`) |
| `sm_inputs` | model | `Mt`, `Mh`, `alpha_s`, `MW` |
| `numerics` | model | optional: `rtol`, `atol`, `n_points`, `mu_max`, `max_rhs_calls` (see `settings.py`) |

A model entry overrides `defaults` key by key, one level deep. For example,
`"run": {"plots": true}` changes only `plots`.

* Fields are written `"(su3,su2,Y)"`, with optional prefixes: `"N:2x(1,1,0)"` gives two
  copies of a singlet named N.
  - SU(3) is `1, 3, 3b, 6, 6b, 8, 10, 15, 27` or Dynkin labels `"p-q"`.
  - Unnamed fields are called F1, F2, …; copies are suffixed `_1`, `_2`.
  - A field can also be a dict `{"name", "su3", "su2", "Y", "copies"}`.
* An axis is a list of values, `{"linspace": [a, b, n]}` or `{"logspace": [a, b, n]}`
  (exponents).

## Output

Each run (one JSON file) gets its own folder, named after the file's `"name"`
(default: the file name):

```
results/<run>/
  summary.csv, summary.json    one row per theory in the run
  running-<theory>.pdf         λ(μ) for the example points and the SM, plus the Yukawas
  inscale-<theory>.pdf         μ_I against 1 or 2 parameters (M, if scanned, is always on the x axis)
  backup/                      older plots and summaries, renamed <name>_<YYYY-MM-DD_HHMMSS>
  <theory>/
    model.json                 fields, couplings, conventions, inputs and settings
    betas/                     betas_*.tex/.json and comparison.md (native vs PyR@TE)
    data/                      running_*.csv and scan.csv (plots can be remade from these)
```

`<theory>` is the model's `name`, or a label built from its representations, e.g.
`(1,1,0)+(1,2,-1_2)+(1,2,1_2)`. Plots are written only for models with
`"plots": true`.

## Code layout

```
src/higgsflow/
  theory/    group_theory.py   SU(3)/SU(2) irreps, Casimirs, invariant tensors, anomalies
             couplings.py      all couplings: names, order, labels, perturbativity limits, conventions
             fields.py         WeylFermion, SM content
             model.py          Model, Yukawa-invariant search, folder label
  rge/       polytensor.py     exact polynomial-valued tensors
             tensors.py        explicit gauge/Yukawa/quartic tensors (doubled Weyl formalism)
             formulas.py       general two-loop formulas (Poole–Thomsen)
             engine.py         projection onto the couplings, closure check, cache
             system.py         BetaSystem: common container, LaTeX, numerics
  pyrate/    install.py        pinned PyR@TE download
             bridge.py         Model -> PyR@TE model file -> parsed BetaSystem
             compare.py        normalization map and coefficient comparison
  running/   boundary.py       SM MS-bar couplings at M_t (Buttazzo et al.)
             threshold.py      one-loop matching at M (heavy-fermion masses, δλ, Higgs Z factor)
             solver.py         running with threshold M, stopping events
             scan.py           1D/2D scans, optionally in parallel
  output/    layout.py, plots.py
  config.py    JSON -> Study (list of RunSpec)   cli.py  `higgsflow` command
  settings.py  numerical settings (tolerances, grids, plot resolution)
  pipeline.py  run_analysis(): betas -> examples + scan -> results (one model)
  study.py     run_study(): every model of a file, in parallel -> results/<run>/summary.csv
```

Extending it:
- A new kind of output goes in `output/` and is called from `pipeline.write_results`.
- A new summary column goes in `study._result_columns`.
- A new run setting goes in `config.RunSpec` (and `spec_from_dict`); it then reaches every step.
- New physics in the running (several thresholds, matching corrections) goes in `running/solver.py`.
- Extra scalars would need `rge/tensors.py` and `theory/`; `formulas.py` is already general.

## Development

```powershell
python3 -m pytest                 # fast tests (PyR@TE comparison skipped)
python3 -m pytest -m pyrate       # PyR@TE comparison (needs install-pyrate)
python3 -m ruff check .           # style
python3 -m mypy                   # types
python3 -m pre_commit install     # once: run ruff, mypy and the fast tests on every git commit
```

## Physics conventions

* V = λ (H†H)², H = (φ1 + iφ2, φ3 + iφ4)/√2, Y_H = 1/2, and g1 = g′ (no GUT normalization).
* Yukawas: −L ⊃ y C ψ1 ψ2 Φ + h.c., with Φ = H or H*. C is the gauge-invariant tensor,
  normalized to max|entry| = 1 per gauge factor. In the SM, only y_t is kept.
* β_x = Σ_L β_x^(L)/(16π²)^L, with L = 1, 2.
* Boundary conditions: NNLO MS-bar fits at μ = M_t from Buttazzo et al., arXiv:1307.3536.
  Defaults: M_t = 172.57, M_h = 125.20, M_W = 80.3692 GeV, α_s(M_Z) = 0.1180.
* One common threshold M: the SM runs below M; above it, the new fields run. The new
  Yukawas are set at μ = M. All new fermions get mass M: vector-like pairs a Dirac
  mass, real representations a Majorana mass.
* Matching at μ = M is one loop by default (`"run": {"threshold": "one-loop"}`, or `"tree"`):
  λ_SM = λ_full (1 + 2Π') + δλ and y_t,SM = y_t,full (1 + Π'/2).
  - δλ = 4c₄ comes from the h⁴ term of the heavy-fermion effective potential.
  - Π' = dΠ_hh/dp² at p² = 0 is the Higgs field renormalization from loops with a heavy
    fermion.
  - Gauge couplings get no correction at μ = M in MS-bar.
  - For the type-I seesaw this gives δλ = −y⁴/16π² and Π' = −y²/(2·16π²).
  - `model.json` records the matching used. `running_<k>.csv` gives δλ, Π', and λ and
    y_t on both sides of the threshold; `summary.csv` gives δλ at each benchmark.
* The running stops at λ = 0 (unstable, μ_I), at a perturbativity limit
  (|g|, |y| > √(4π) or |λ| > 4π), or at M_Pl = 1.22·10¹⁹ GeV (stable).

## Notes on the PyR@TE comparison

* PyR@TE is run with `--RealBasis None`. With its default rotation of real
  representations, the two-loop g1²g2²y² term in β_λ depends on the basis when a
  real-representation fermion carries hypercharge, e.g. a (1,3,1) triplet.
* PyR@TE's Clebsch–Gordan normalization can differ from ours. The ratio k is fixed by
  the one-loop y³ coefficient: that single coefficient is a calibration, not a check.
* Both codes use the same published general formulas, so agreement checks the group
  theory, the tensors and the conventions. The independent check of the formulas is
  `tests/test_sm_betas.py`, which compares against the textbook SM two-loop RGEs.

## Where things are kept

`~/.higgsflow/` holds the β-function cache and the PyR@TE install; override it with
`HIGGSFLOW_HOME`. Results go to `<project>/results`; override with `HIGGSFLOW_RESULTS`.
