import os
import time

from higgsflow.output.layout import backup_files, plot_names, prepare_folder
from higgsflow.theory.fields import WeylFermion as W
from higgsflow.theory.model import Model


def test_label_from_reps():
    m = Model()
    m.add(W("L4", "1", 2, "-1/2"), W("N", "1", 1, 0, copies=2))
    assert m.label() == "2x(1,1,0)+(1,2,-1_2)"
    assert Model(name="my model").label() == "my_model"


def test_backup(tmp_path):
    prepare_folder("VLL", tmp_path)
    names = plot_names("VLL")
    assert names == ("running-VLL.pdf", "inscale-VLL.pdf")
    p = tmp_path / names[0]
    p.write_text("old")
    t = time.mktime((2026, 1, 2, 3, 4, 5, 0, 0, -1))
    os.utime(p, (t, t))
    moved = backup_files(tmp_path, names)
    assert not p.exists()
    assert moved[0] == tmp_path / "backup" / "running-VLL_2026-01-02_030405.pdf"
