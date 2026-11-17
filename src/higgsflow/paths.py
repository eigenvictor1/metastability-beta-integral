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
    return Path(os.environ.get("HIGGSFLOW_HOME", Path.home() / ".higgsflow"))


def cache_dir() -> Path:
    return home() / "cache"


def pyrate_dir() -> Path:
    return home() / "pyrate"


def results_root() -> Path:
    if "HIGGSFLOW_RESULTS" in os.environ:
        return Path(os.environ["HIGGSFLOW_RESULTS"])
    if (PROJECT_ROOT / "pyproject.toml").exists():
        return PROJECT_ROOT / "results"
    return Path.cwd() / "results"
