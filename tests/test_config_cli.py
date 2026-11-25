import csv
import json
from pathlib import Path

import numpy as np
import pytest

from higgsflow import config
from higgsflow.cli import main
from higgsflow.theory.fields import WeylFermion

DATA = Path(__file__).resolve().parent / "data"          # input files used only by the tests
LABEL = "(1,1,0)+(1,2,-1_2)+(1,2,1_2)"


def test_compact_field_specs():
    f = WeylFermion.from_spec("Q4:2x(3b,1,-2/3)", "F1")
    assert (f.name, f.su3, f.su2, str(f.Y), f.copies) == ("Q4", (0, 1), 1, "-2/3", 2)
    g = WeylFermion.from_spec("(2-1,3,1/2)", "F7")
    assert (g.name, g.su3, g.su2) == ("F7", (2, 1), 3)


def test_single_model_file():
    study = config.load(DATA / "vll_doublet_singlet.json")
    assert study.name == "vll_doublet_singlet" and study.workers is None
    (spec,) = study.specs()
    assert spec.model.label() == "VLL_doublet_singlet"
    assert spec.model.new_yukawa_names() == ["yL4N", "yL4cNc"]
    assert np.isclose(spec.scan["M"][-1], 1e8) and len(spec.scan["y"]) == 26
    assert spec.betas == "both"


def test_unnamed_label():
    (spec,) = config.load(DATA / "real_triplet.json").specs()
    assert spec.model.label() == "(1,2,-1_2)+(1,2,1_2)+(1,3,0)"
    assert spec.fixed == {"M": 2e3}


def test_defaults_are_merged():
    study = config.load(DATA / "five_models.json")
    specs = study.specs()
    assert len(specs) == 5 and study.name == "five_models"
    assert specs[0].fixed == {"M": 1e3} and specs[3].fixed == {"M": 2e3}
    assert specs[4].plots is True and specs[0].plots is False
    assert specs[2].model.label() == "(1,2,-1_2)+(1,2,1_2)+(1,3,-1)+(1,3,1)"


def test_models_list_required():
    with pytest.raises(ValueError):
        config.study_from_dict({"defaults": SCAN, "fields": ["(1,1,0)"]})


def test_cli_yukawas(capsys):
    assert main(["yukawas", str(DATA / "real_triplet.json")]) == 0
    out = capsys.readouterr().out
    assert "ySL4: S L4 H" in out


def _write(tmp_path, models, **top):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"models": models, **top}))
    return f


SMALL = {"fields": ["N:(1,1,0)", "L4:(1,2,-1/2)", "L4c:(1,2,1/2)"],
         "links": {"y": "all"}, "benchmarks": [{"y": 0.5}]}
SCAN = {"scan": {"y": {"linspace": [0, 1, 3]}}}


def test_scan_only_in_defaults():
    with pytest.raises(ValueError, match="only allowed in"):
        config.study_from_dict({"defaults": SCAN, "models": [{**SMALL, **SCAN}]})
    with pytest.raises(ValueError, match="must be given"):
        config.study_from_dict({"models": [SMALL]})


def test_run_one_model_and_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HIGGSFLOW_RESULTS", str(tmp_path))
    f = _write(tmp_path, [SMALL], workers=2, defaults=SCAN)
    assert main(["run", str(f)]) == 0
    run = tmp_path / "s"
    assert (run / f"running-{LABEL}.pdf").exists() and (run / f"inscale-{LABEL}.pdf").exists()
    assert (run / "summary.csv").exists() and (run / LABEL / "data" / "scan.csv").exists()
    assert main(["run", str(f)]) == 0          # second run backs up plots and summary
    assert len(list((run / "backup").glob("*.pdf"))) == 4
    assert len(list((run / "backup").glob("summary_*.csv"))) == 1
    # the plots are redrawn from the stored data, without running anything (and no backup)
    pdf = run / f"inscale-{LABEL}.pdf"
    pdf.unlink()
    assert (run / LABEL / "data" / "plotdata.pkl").exists()
    assert main(["replot", str(f)]) == 0
    assert pdf.exists() and len(list((run / "backup").glob("*.pdf"))) == 4


def test_run_many_models(tmp_path, monkeypatch):
    monkeypatch.setenv("HIGGSFLOW_RESULTS", str(tmp_path))
    f = _write(tmp_path, [
        {"fields": ["(1,2,-1/2)", "(1,2,1/2)", "(1,1,0)"]},
        {"fields": ["(1,3,0)", "(1,2,-1/2)", "(1,2,1/2)"]},
        {"fields": ["(1,2,-1/2)", "(1,2,1/2)", "(1,1,0)"], "links": {"y": ["nonexistent"]}},
    ], name="t", workers=2,
        defaults={"scan": {"y": {"linspace": [0, 1, 3]}}, "links": {"y": "all"},
                  "benchmarks": [{"y": 0.5}], "run": {"plots": False}})
    assert main(["run", str(f)]) == 0
    rows = list(csv.DictReader(open(tmp_path / "t" / "summary.csv", encoding="utf-8")))
    assert [r["index"] for r in rows] == ["1", "2", "3"]
    assert rows[0]["error"] == "" and float(rows[0]["bench1_mu_I_GeV"]) > 1e3
    assert "unknown couplings" in rows[2]["error"]
    assert (tmp_path / "t" / LABEL / "data" / "scan.csv").exists()
    assert not list((tmp_path / "t").glob("*.pdf"))


def test_parallel_scan_matches_serial():
    from higgsflow import Runner, compute_betas
    from higgsflow.running.scan import run_scan
    (spec,) = config.load(DATA / "vll_doublet_singlet.json").specs()
    m = spec.model
    r = Runner(compute_betas(m), compute_betas(type(m)()))
    scan = {"y": np.linspace(0, 1, 4)}
    links = {"y": ["yL4N", "yL4cNc"]}
    a = run_scan(r, scan, {"M": 1e3}, links, workers=1)
    b = run_scan(r, scan, {"M": 1e3}, links, workers=2)
    assert np.allclose(a.mu_I, b.mu_I, equal_nan=True)


def test_weighted_links():
    from higgsflow.running.scan import expand_links, resolve_params
    links = expand_links({"y": {"a": 2.0, "b": 1.0}, "z": ["c"]}, ["a", "b", "c"])
    vals, M = resolve_params({"y": 0.5, "z": 0.3, "M": 1e4}, links, 1e3)
    assert vals == {"a": 1.0, "b": 0.5, "c": 0.3} and M == 1e4


def test_trace_plot_is_optional():
    from higgsflow.config import TraceSpec
    assert TraceSpec.from_dict({"M": 1e3}) is not None
    assert TraceSpec.from_dict({"M": 1e3, "enabled": False}) is None
    assert TraceSpec.from_dict(False) is None
    assert TraceSpec.from_dict(None) is None
