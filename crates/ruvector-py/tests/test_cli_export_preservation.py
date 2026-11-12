import os
import numpy as np
import pytest
from click.testing import CliRunner
from ruvector import Collection
from ruvector.cli import main

@pytest.mark.parametrize("kind", ["same_prefix", "index_output", "hardlink_meta", "symlink_meta"])
def test_export_rejects_source_overlap_without_mutation(tmp_path, kind):
    path = tmp_path / ("source.npy" if kind == "index_output" else "source.rbpx")
    coll = Collection.create(2)
    coll.insert(np.array([1., 0.], dtype=np.float32), metadata={"keep": True})
    coll.save(path)
    meta = Collection.meta_path(path)
    prefix = tmp_path / "exported"
    if kind in ("same_prefix", "index_output"): prefix = path.with_suffix("") if kind == "index_output" else path
    elif kind == "hardlink_meta": os.link(meta, str(prefix) + ".meta.json")
    else: os.symlink(meta, str(prefix) + ".meta.json")
    before = (path.read_bytes(), meta.read_bytes())
    result = CliRunner().invoke(main, ["export", "--path", str(path), "--out", str(prefix)])
    assert result.exit_code != 0, result.output
    assert (path.read_bytes(), meta.read_bytes()) == before
    assert len(Collection.load(path)) == 1

def test_export_separate_outputs_control(tmp_path):
    path = tmp_path / "source.rbpx"
    coll = Collection.create(2)
    coll.insert(np.array([1., 0.], dtype=np.float32), metadata={"keep": True})
    coll.save(path)
    result = CliRunner().invoke(main, ["export", "--path", str(path), "--out", str(tmp_path / "exported")])
    assert result.exit_code == 0, result.output
    assert np.load(tmp_path / "exported.npy").shape == (1, 2)
    assert len(Collection.load(path)) == 1
