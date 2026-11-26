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
higgsflow run models\seesaw_typeI.json            # three models
higgsflow run models\mixed.json --workers 0         # 0 = all cores (default: 3/4 of them)
higgsflow run models\seesaw_typeI.json --no-trs   # without the trace plots
higgsflow replot models\mixed.json                 # redraw the plots from stored data (no running)
higgsflow replot models\mixed.json --only Q5 T2    # only some models

higgsflow yukawas models\seesaw_typeI.json        # Yukawas found for each model
higgsflow betas models\vll_doublet_singlet.json --source both   # betas + native/PyR@TE comparison
```

Results go to `results\<name of the file>\` (see Output). With fewer models than cores, the models run one after the other and all cores share each
scan; with more, each core takes whole models. A long scan logs its progress every 10%.

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
    "run": {"betas": "pyrate", "plots": false}
  },
  "models": [
    {"name": "VLL", "fields": ["(1,2,-1/2)", "(1,2,1/2)", "(1,1,0)"]},
    {"name": "triplet_VLL", "fields": ["S:(1,3,0)", "(1,2,-1/2)", "(1,2,1/2)"],
     "fixed": {"M": 2e3}, "run": {"plots": true}}
  ],
  "workers": 8
}
```

| key | where | meaning |
|---|---|---|
| `name` | top | name of the run, i.e. its folder in `results/` (default: the file name) |
| `defaults` | top | **required**: the `scan`, plus defaults for any per-model key below |
| `scan` | defaults | 1 or 2 parameters: `M` (GeV) and/or Yukawas (or link aliases) at μ = M; the same grid for every model |
| `models` | top | **required** list, one entry per model |
| `workers` | top | number of processes (default 3/4 of the machine's cores; at most the number of cores); 0 = all cores. A study first computes the missing beta functions in parallel (as many at once as memory allows, ~3 GB each), then runs the models one after the other, each scan on all processes |
| `name` | model | name of the theory in plot and folder names (default: a label from the representations) |
| `fields` | model | **required** list of new left-handed Weyl fields |
| `include_sm_mixing`, `exclude` | model | also generate new–SM Yukawas / switch off some Yukawas |
| `links` | model | `{alias: [Yukawas]}`, `{alias: {Yukawa: factor}}` or `"all"`: one parameter drives several Yukawas (each = factor × parameter) |
| `fixed` | model | values of parameters that aren't scanned |
| `examples` | model | points drawn in `running-<theory>.pdf` |
| `benchmarks` | model | points whose μ_I goes into the summary |
| `run` | model | `betas` (`pyrate`, the default, needs `higgsflow install-pyrate`; `native`; `both`; models with the scalar singlet need `native`), `loops`, `M` (default threshold), `plots`, `threshold` (`one-loop`/`tree`), `trace_range` (range of Tr[(YY†)²] shown in `inscale-` plots, default `[1e-3, 1e3]`; the axis is drawn as Tr[(YY†)²]^(1/4)), `y_label` (label of that axis, LaTeX, e.g. `"$y$"`; default Tr[(YY†)²]^(1/4); `replot` with the model file uses the file's value) |
| `sm_inputs` | model | `Mt`, `Mh`, `alpha_s`, `MW` |
| `numerics` | model | optional: `rtol`, `atol`, `n_points`, `mu_max`, `max_rhs_calls`, `boundary_steps`, `trace_curves` (default `true`: trace the stable boundary, the metastability line and the μ_I contours of 2D scans point by point; see below and `settings.py`) |
| `trs` | model | optional trace plot: `{"M": 1e3, "samples": 2000, "range": [0, 1.5]}` — fixed M, number of Yukawa points (spread evenly over the trace plane), range of their size √(Σy²); switch it off with `"enabled": false` (or `"trs": false`), or for the whole run with `higgsflow run ... --no-trs` (`--trs` turns it on where it has settings) |

`scan` is allowed only in `defaults`, so all models of a file share one grid; a model
with its own `scan` is an error. Its parameter names must exist in every model, normally
through `links` (e.g. `"y"` as an alias). Any other key of a model entry overrides
`defaults` key by key, one level deep. For example,
`"run": {"plots": true}` changes only `plots`.

* Fields are written `"(su3,su2,Y)"`, with optional prefixes: `"N:2x(1,1,0)"` gives two
  copies of a singlet named N.
  - SU(3) is `1, 3, 3b, 6, 6b, 8, 10, 15, 27` or Dynkin labels `"p-q"`.
  - Unnamed fields are called F1, F2, …; copies are suffixed `_1`, `_2`.
  - A field can also be a dict `{"name", "su3", "su2", "Y", "copies", "mass"}`.
  - A suffix `@mass` (GeV) gives the field its own fixed mass, e.g. `"X5:(1,5,0)@9.6e3"` (the
    Minimal Dark Matter quintuplet). It then decouples at that mass, not at the scanned M:
    below it, it is absent; above it, it enters the gauge running. Such fields must not have
    Yukawa couplings. The SM reference run leaves them out.
* `"singlet": {"m_S": 1e8, "lamS": 0.2, "kappa": 0.2, "yS": "derived"}` (in a model entry
  or in `defaults`) adds a complex scalar singlet S, with V ⊃ λ_S|S|⁴ + κ(H†H)|S|², that gives
  the new fermions their mass: −L ⊃ ½ y_S S ψM₀ψ + h.c., so M = y_S v_S/√2 (fixed-mass fields
  keep a bare mass).
  - m_S is the mass of the radial mode; λ_S and κ are given at μ = m_S.
  - `"yS": "derived"` sets y_S = 2√λ_S M/m_S at each point (from m_S² = 2λ_S v_S²); a
    number fixes it.
  - S is integrated out at m_S with the tree-level matching λ_SM = λ − κ²/(4λ_S): running up,
    λ jumps by +κ²/(4λ_S). Above m_S, y_S, λ_S and κ run with the two-loop betas of the
    native engine (not available with PyR@TE).
  - Example: `models/mixed_singlet.json`, the 13 models of `mixed.json` with S at 10⁸ GeV.
* Yukawa names have no underscores: `y` + the two component names (copy suffix included,
  e.g. `N1`, `L3`) + `c` if the operator uses H\*. Examples: `yNL3` (N L₃ H), `yL4cNc`
  (L4c N H\*). Run `python3 -m higgsflow yukawas <file>` to list them for a model.
* An axis is a list of values, `{"linspace": [a, b, n]}` or `{"logspace": [a, b, n]}`
  (exponents).

## Output

Each run (one JSON file) gets its own folder, named after the file's `"name"`
(default: the file name):

### Traced curves

In 2D scans over M and one Yukawa parameter, the lines of the maps are not interpolated
between grid points but traced with runs placed on them (`running/curves.py`):

* the stable boundary, λ_low = 0 (λ_low: the lowest λ along the running, < 0 exactly when λ
  crosses zero), the metastability line λ_low = −0.06, and every integer contour of log10 μ_I;
* each connected piece of a curve is seeded from the grid, moved onto the curve by a root
  search and followed in both directions by predictor–corrector continuation: a step along
  the tangent, then Brent's method along the normal (tolerance 2·10⁻⁵ of the plot), with the
  step adapted to the curvature (0.0002–0.03 of the plot);
* a μ_I contour meets the stable boundary tangentially (μ_I has a square-root edge there); it
  ends at the point of the traced boundary where log10 μ_min equals its level.

The pieces are traced in parallel; it costs roughly 150–250 runs per curve. The colour map is
still interpolated from the grid. `"numerics": {"trace_curves": false}` switches it off (grid
contours as before); the points are in `data/curves.csv`.

```
results/<run>/
  summary.csv, summary.json    one row per theory in the run
  running-<theory>.pdf         λ(μ) for the example points and the SM, plus the Yukawas
  inscale-<theory>.pdf         μ_I against 1 or 2 parameters (M, if scanned, is always on the x axis);
                               the Yukawa parameter is shown as Tr[(YY†)²]^(1/4) (log scale), and region
                               boundaries (non-perturbative, stable) are located by bisection; the
                               stable boundary and the μ_I contours are traced (see "Traced curves")
  inscale-min-<theory>.pdf     the same map with log10 of μ_min (scale of the minimum of λ) as
                               cornflower-blue dashed contours and shading (0.1, then 0.05 more opaque per
                               contour towards lower μ_min), and a thicker dashed line where a
                               minimum of λ appears
  inscale-lambdamin-<theory>.pdf  the μ_I map split by the lowest λ of the running (λ_min, or λ where the
                               running stops if still decreasing; column lam_low of scan.csv): a yellow
                               dashed line at λ_min = −0.06, "metastable" (yellow) above it, "unstable" (red)
                               below; "stable" where there is no instability scale
  inscale-trs-<theory>.pdf     (with "trs") μ_I at fixed M against Tr(YY†) and Tr[(YY†)²]^(1/4), for
                               points of all new Yukawas spread evenly over the reachable region
  backup/                      older plots and summaries, renamed <name>_<YYYY-MM-DD_HHMMSS>
  <theory>/
    model.json                 fields, couplings, conventions, inputs and settings
    betas/                     betas_*.tex/.json and comparison.md (native vs PyR@TE)
    data/                      running_*.csv, scan.csv, boundaries.csv, curves.csv (traced curves),
                               trs_samples.csv, and
                               plotdata.pkl: everything the plots show, so that `higgsflow replot`
                               can redraw them (e.g. after a style change) without running again
```

The traces use the Yukawa matrix of the new fermions along the neutral Higgs direction,
Y, in the convention −L ⊃ ½ Y_IJ φ ψ_I ψ_J (two-component fermions; y_t excluded). For the
seesaw with one N, Tr(YY†) = y² and Tr[(YY†)²] = y⁴/2. With n equal couplings the two traces
are tied, so the trace plot fills a region only for models with several independent Yukawas.

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
             scan.py           1D/2D scans and random samples, optionally in parallel
             traces.py         Tr(YY†), Tr[(YY†)²] and the trace axis of the plots
             strength.py       y_eff of the new Yukawas (perturbativity limit)
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
* The running continues past λ = 0 and stops at a perturbativity limit or at
  M_Pl = 1.22·10¹⁹ GeV. A point is unstable if λ crossed zero on the way (μ_I = first
  crossing), otherwise non-perturbative or stable. A minimum of λ is a point where β_λ turns
  from negative to positive inside that range (or at μ = M, if β_λ jumps there from negative
  to positive); the end of the running is never taken as a minimum. The deepest minimum is
  kept: `lam_min` and `mu_min_GeV` in scan.csv (NaN if λ has none), diamonds in the running
  plot. The limits are |g|, |y_t| > √(4π), |λ| > 4π, and for the
  new Yukawas together y_eff > √(4π), with y_eff = √(λ_max(Σ_a Y^a Y^a†)/2) the strength of
  the whole Yukawa matrix (basis independent; y_eff = y for one N, √6·y for two N coupled
  with the same y to all three leptons).

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
