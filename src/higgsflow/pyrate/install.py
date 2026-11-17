"""Download and locate PyR@TE 3 (Sartore & Schienbein, arXiv:2007.12700, Apache-2.0).

PyR@TE is not bundled. ``higgsflow install-pyrate`` fetches the pinned commit below
from GitHub (git if available, otherwise a zip download) into ~/.higgsflow/pyrate.
Its own dependencies (PyYAML, h5py) come with ``pip install -e .[pyrate]``.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from .. import paths

PINNED_COMMIT = "04b219c2016f3fc4f2371d72607edc26a7e06364"
REPO = "https://github.com/LSartore/pyrate.git"
URL = f"https://github.com/LSartore/pyrate/archive/{PINNED_COMMIT}.zip"

log = logging.getLogger(__name__)


class PyRATEUnavailable(RuntimeError):
    pass


def find_pyrate(pyrate_dir=None) -> Path:
    """Explicit argument > $HIGGSFLOW_PYRATE > ~/.higgsflow/pyrate."""
    for c in (pyrate_dir, os.environ.get("HIGGSFLOW_PYRATE"), paths.pyrate_dir()):
        if c and (Path(c) / "pyR@TE.py").exists():
            return Path(c).resolve()
    raise PyRATEUnavailable("PyR@TE 3 not found: run `higgsflow install-pyrate` "
                            "(or set HIGGSFLOW_PYRATE to an existing checkout).")


def _git_install(target: Path):
    log.info("cloning PyR@TE at %s", PINNED_COMMIT[:8])
    def run(*args):
        subprocess.run(["git", *args], cwd=target, check=True, capture_output=True, text=True)

    target.mkdir()
    run("init", "-q")
    run("remote", "add", "origin", REPO)
    run("fetch", "-q", "--depth", "1", "origin", PINNED_COMMIT)
    run("checkout", "-q", "FETCH_HEAD")


def _zip_install(target: Path):
    log.info("downloading PyR@TE %s", PINNED_COMMIT[:8])
    with urllib.request.urlopen(URL, timeout=120) as r:
        data = r.read()
    tmp = target.parent / "_pyrate_download"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(tmp)
    (inner,) = list(tmp.iterdir())
    shutil.move(str(inner), str(target))
    shutil.rmtree(tmp)


def install_pyrate(target: Path | None = None, force: bool = False) -> Path:
    target = Path(target or paths.pyrate_dir())
    if (target / "pyR@TE.py").exists() and not force:
        log.info("PyR@TE already installed at %s", target)
        return target
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        _git_install(target)
    else:
        _zip_install(target)
    # do not let PyR@TE try to run pdflatex after each run
    settings = target / "default.settings"
    lines = settings.read_text().splitlines()
    settings.write_text("\n".join('EndCommands : ""' if l.startswith("EndCommands") else l
                                  for l in lines) + "\n")
    (target / "INSTALLED_COMMIT").write_text(PINNED_COMMIT + "\n")
    log.info("PyR@TE installed at %s", target)
    return target
