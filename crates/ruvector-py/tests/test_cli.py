"""End-to-end tests for the ``ruvector`` CLI (ADR-352 M1.5) via click's
CliRunner — real subprocess-free invocations against a tmp_path collection.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from ruvector.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_help_lists_subcommands(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["create", "search", "delete", "export", "import", "benchmark", "serve", "info"]:
        assert cmd in result.output


def test_create_insert_search_delete_info(runner, tmp_path):
    db = tmp_path / "coll.rbpx"
    r = runner.invoke(main, ["create", "--path", str(db), "--dim", "4"])
    assert r.exit_code == 0, r.output
    assert "created empty collection" in r.output

    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((10, 4)).astype(np.float32)
    vecs_path = tmp_path / "vecs.npy"
    np.save(vecs_path, vecs)
    metas = [{"i": i} for i in range(10)]
    meta_path = tmp_path / "metas.json"
    meta_path.write_text(json.dumps(metas))

    r = runner.invoke(
        main,
        ["insert-batch", "--path", str(db), "--vectors", str(vecs_path), "--metadata", str(meta_path)],
    )
    assert r.exit_code == 0, r.output
    assert "inserted 10 vectors" in r.output

    query_path = tmp_path / "q.npy"
    np.save(query_path, vecs[3])
    r = runner.invoke(main, ["search", "--path", str(db), "--query", str(query_path), "-k", "3", "--json"])
    assert r.exit_code == 0, r.output
    hits = json.loads(r.output)
    assert hits[0]["id"] == 3
    assert hits[0]["metadata"] == {"i": 3}

    r = runner.invoke(main, ["info", "--path", str(db), "--json"])
    assert r.exit_code == 0, r.output
    info = json.loads(r.output)
    assert info["count"] == 10
    assert info["dim"] == 4

    r = runner.invoke(main, ["delete", "--path", str(db), "--id", "3", "--vacuum"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["info", "--path", str(db), "--json"])
    assert json.loads(r.output)["count"] == 9


def test_import_and_export_roundtrip(runner, tmp_path):
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((5, 3)).astype(np.float32)
    vecs_path = tmp_path / "in.npy"
    np.save(vecs_path, vecs)

    db = tmp_path / "imported.rbpx"
    r = runner.invoke(main, ["import", "--path", str(db), "--vectors", str(vecs_path)])
    assert r.exit_code == 0, r.output
    assert "built collection with 5 vectors" in r.output

    out_prefix = str(tmp_path / "out")
    r = runner.invoke(main, ["export", "--path", str(db), "--out", out_prefix])
    assert r.exit_code == 0, r.output
    exported = np.load(out_prefix + ".npy")
    assert exported.shape == (5, 3)


def test_create_twice_fails(runner, tmp_path):
    db = tmp_path / "dup.rbpx"
    r1 = runner.invoke(main, ["create", "--path", str(db), "--dim", "2"])
    assert r1.exit_code == 0
    r2 = runner.invoke(main, ["create", "--path", str(db), "--dim", "2"])
    assert r2.exit_code != 0
    assert "already exists" in r2.output


def test_search_missing_collection_fails(runner, tmp_path):
    q = tmp_path / "q.npy"
    np.save(q, np.zeros(4, dtype=np.float32))
    r = runner.invoke(main, ["search", "--path", str(tmp_path / "nope.rbpx"), "--query", str(q)])
    assert r.exit_code != 0


def test_benchmark_runs(runner):
    r = runner.invoke(main, ["benchmark", "-n", "500", "--dim", "16", "--queries", "20", "--json"])
    assert r.exit_code == 0, r.output
    result = json.loads(r.output)
    assert result["n"] == 500
    assert result["p50_ms"] >= 0
    assert result["qps"] > 0
