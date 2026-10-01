"""End-to-end tests for the ``ruvector`` CLI (ADR-352 M1.5) via click's
CliRunner — real subprocess-free invocations against a tmp_path collection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from ruvector.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_help_lists_subcommands(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["create", "search", "delete", "export", "import", "benchmark", "serve", "info"]:
        assert cmd in result.output


def test_create_insert_search_delete_info(runner: CliRunner, tmp_path: Path) -> None:
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
    # --json must stay exactly plain, parseable JSON: no stray ANSI/color
    # escape sequences, under CliRunner (non-TTY) or anywhere else.
    assert "\x1b" not in r.output
    hits = json.loads(r.output)
    assert hits[0]["id"] == 3
    assert hits[0]["metadata"] == {"i": 3}

    # Human-readable table: assert on the underlying data (ids, score,
    # metadata) being present, not on exact table/box-drawing formatting,
    # which is fragile and not the point of this test.
    r = runner.invoke(main, ["search", "--path", str(db), "--query", str(query_path), "-k", "3"])
    assert r.exit_code == 0, r.output
    assert "\x1b" not in r.output  # CliRunner isn't a real TTY -> no color
    assert "3" in r.output
    assert '"i": 3' in r.output

    r = runner.invoke(main, ["info", "--path", str(db), "--json"])
    assert r.exit_code == 0, r.output
    assert "\x1b" not in r.output
    info = json.loads(r.output)
    assert info["count"] == 10
    assert info["dim"] == 4
    assert info["backend"] == "hnsw"

    # Human-readable info panel: same data, non-TTY so no escape codes.
    r = runner.invoke(main, ["info", "--path", str(db)])
    assert r.exit_code == 0, r.output
    assert "\x1b" not in r.output
    assert "10" in r.output
    assert "hnsw" in r.output

    r = runner.invoke(main, ["delete", "--path", str(db), "--id", "3", "--vacuum"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["info", "--path", str(db), "--json"])
    assert json.loads(r.output)["count"] == 9


def test_insert_batch_shows_progress_and_inserts_correct_count(runner: CliRunner, tmp_path: Path) -> None:
    """The insert-batch spinner is cosmetic (see cli.py's _run_with_spinner
    docstring for why it's not a chunked/granular progress bar) — the
    thing that actually matters is that wrapping the call in a Progress
    context doesn't change the result."""
    db = tmp_path / "coll.rbpx"
    r = runner.invoke(main, ["create", "--path", str(db), "--dim", "4"])
    assert r.exit_code == 0, r.output

    rng = np.random.default_rng(2)
    vecs = rng.standard_normal((25, 4)).astype(np.float32)
    vecs_path = tmp_path / "vecs.npy"
    np.save(vecs_path, vecs)

    r = runner.invoke(main, ["insert-batch", "--path", str(db), "--vectors", str(vecs_path)])
    assert r.exit_code == 0, r.output
    assert "inserted 25 vectors" in r.output

    r = runner.invoke(main, ["info", "--path", str(db), "--json"])
    assert json.loads(r.output)["count"] == 25


def test_import_and_export_roundtrip(runner: CliRunner, tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((5, 3)).astype(np.float32)
    vecs_path = tmp_path / "in.npy"
    np.save(vecs_path, vecs)

    db = tmp_path / "imported.rbpx"
    r = runner.invoke(main, ["import", "--path", str(db), "--vectors", str(vecs_path)])
    assert r.exit_code == 0, r.output
    assert "built collection with 5 vectors" in r.output
    # import also runs behind the spinner (_run_with_spinner) — same
    # no-escape-under-non-TTY expectation as everything else.
    assert "\x1b" not in r.output

    out_prefix = str(tmp_path / "out")
    r = runner.invoke(main, ["export", "--path", str(db), "--out", out_prefix])
    assert r.exit_code == 0, r.output
    exported = np.load(out_prefix + ".npy")
    assert exported.shape == (5, 3)


def test_create_twice_fails(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "dup.rbpx"
    r1 = runner.invoke(main, ["create", "--path", str(db), "--dim", "2"])
    assert r1.exit_code == 0
    r2 = runner.invoke(main, ["create", "--path", str(db), "--dim", "2"])
    assert r2.exit_code != 0
    assert "already exists" in r2.output


def test_search_missing_collection_fails(runner: CliRunner, tmp_path: Path) -> None:
    q = tmp_path / "q.npy"
    np.save(q, np.zeros(4, dtype=np.float32))
    r = runner.invoke(main, ["search", "--path", str(tmp_path / "nope.rbpx"), "--query", str(q)])
    assert r.exit_code != 0


def test_benchmark_runs(runner: CliRunner) -> None:
    r = runner.invoke(main, ["benchmark", "-n", "500", "--dim", "16", "--queries", "20", "--json"])
    assert r.exit_code == 0, r.output
    assert "\x1b" not in r.output
    result = json.loads(r.output)
    assert result["n"] == 500
    assert result["p50_ms"] >= 0
    assert result["qps"] > 0


def test_benchmark_human_readable_has_data(runner: CliRunner) -> None:
    r = runner.invoke(main, ["benchmark", "-n", "500", "--dim", "16", "--queries", "20"])
    assert r.exit_code == 0, r.output
    assert "\x1b" not in r.output  # non-TTY under CliRunner -> no color
    assert "n=500" in r.output
    assert "dim=16" in r.output


def test_search_no_color_flag_forces_plain(runner: CliRunner, tmp_path: Path) -> None:
    """--no-color is a no-op in terms of data under CliRunner (already
    non-TTY => already plain), but exercise the flag itself end to end so
    a future regression in its wiring (e.g. an exception from a bad
    Console() call) would fail a test, not just a manual check."""
    db = tmp_path / "coll.rbpx"
    runner.invoke(main, ["create", "--path", str(db), "--dim", "4"])
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((5, 4)).astype(np.float32)
    vecs_path = tmp_path / "vecs.npy"
    np.save(vecs_path, vecs)
    runner.invoke(main, ["insert-batch", "--path", str(db), "--vectors", str(vecs_path)])

    query_path = tmp_path / "q.npy"
    np.save(query_path, vecs[0])
    r = runner.invoke(main, ["search", "--path", str(db), "--query", str(query_path), "-k", "2", "--no-color"])
    assert r.exit_code == 0, r.output
    assert "\x1b" not in r.output
    assert "0" in r.output
