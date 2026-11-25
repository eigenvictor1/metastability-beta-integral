"""Where higgsflow keeps things.

* results:  <project>/results (the project containing pyproject.toml), else ./results;
            override with HIGGSFLOW_RESULTS.
* cache, PyR@TE install:  ~/.higgsflow  (override with HIGGSFLOW_HOME).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def home() -> Path:
    """higgsflow's own folder: $HIGGSFLOW_HOME or ~/.higgsflow."""
    return Path(os.environ.get("HIGGSFLOW_HOME", Path.home() / ".higgsflow"))


def cache_dir() -> Path:
    """Cache of computed beta functions (native_<hash>.json, pyrate_<hash>.json)."""
    return home() / "cache"


def pyrate_dir() -> Path:
    """Where install-pyrate puts PyR@TE."""
    return home() / "pyrate"


def short(path: str | Path) -> str:
    """A path for messages, without the full location: relative to the current folder or to
    the project (e.g. results/seesaw_typeI/summary.csv), ~/... under the home folder, else
    just its last parts."""
    p = Path(path).resolve()
    for base in (Path.cwd().resolve(), PROJECT_ROOT):
        try:
            return p.relative_to(base).as_posix() or "."
        except ValueError:
            pass
    try:
        return "~/" + p.relative_to(Path.home().resolve()).as_posix()
    except ValueError:
        return Path(*p.parts[-2:]).as_posix()


def results_root() -> Path:
    """The results/ folder: $HIGGSFLOW_RESULTS, else <project>/results, else ./results."""
    if "HIGGSFLOW_RESULTS" in os.environ:
        return Path(os.environ["HIGGSFLOW_RESULTS"])
    if (PROJECT_ROOT / "pyproject.toml").exists():
        return PROJECT_ROOT / "results"
    return Path.cwd() / "results"
