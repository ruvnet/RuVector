from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from ruvector import (
    UPSTREAM_COMMIT,
    ClosedError,
    DimensionError,
    DistanceMetric,
    DuplicateIDError,
    HNSWConfig,
    InvalidVectorError,
    SearchResult,
    StorageError,
    VectorDB,
    VectorRecord,
    _native,
)


@pytest.fixture
def db():
    with VectorDB(3, hnsw=HNSWConfig(max_elements=128)) as value:
        yield value


def test_real_native_extension():
    assert any(str(_native.__file__).endswith(s) for s in (".so", ".pyd"))
    assert UPSTREAM_COMMIT == "5a93328f2fceb0307c25929ed38cd7a0911fdf00"


def test_crud_batch_and_distance(db):
    assert db.insert_batch(
        [
            VectorRecord("a", [1, 0, 0], {"tenant": "a", "nested": {"x": [1, None]}}),
            VectorRecord("b", [0, 1, 0], {"tenant": "b"}),
            VectorRecord("c", [0, 0, 1]),
        ]
    ) == ["a", "b", "c"]
    assert len(db) == 3
    assert set(db.keys()) == {"a", "b", "c"}
    assert db["a"].metadata == {"tenant": "a", "nested": {"x": [1, None]}}
    result = db.search([1, 0, 0], k=3)
    assert isinstance(result[0], SearchResult)
    assert result[0].id == "a"
    assert result[0].score == pytest.approx(0, abs=1e-5)
    assert result[0].vector == [1, 0, 0]
    assert all(a.score <= b.score for a, b in zip(result, result[1:]))
    assert db.get("missing") is None
    with pytest.raises(KeyError):
        _ = db["missing"]
    assert db.delete_batch(["a", "a", "missing"]) == [True, False, False]
    assert len(db) == 2
    assert all(r.id != "a" for r in db.search([1, 0, 0], k=3))


def test_filters_follow_native_post_top_k(db):
    db.insert([1, 0, 0], id="nearest", metadata={"tenant": "other", "kind": 1})
    db.insert([0.8, 0.2, 0], id="match", metadata={"tenant": "a", "kind": 2})
    db.insert([0, 1, 0], id="bare")
    assert db.search([1, 0, 0], k=1, filter={"tenant": "a"}) == []
    assert [r.id for r in db.search([1, 0, 0], k=3, filter={"tenant": "a"})] == ["match"]
    assert db.search([1, 0, 0], k=3, filter={"tenant": "a", "kind": 1}) == []
    assert {r.id for r in db.search([1, 0, 0], k=3, filter={})} == {"nearest", "match"}


def test_nested_json_equality_and_null(db):
    db.insert([1, 0, 0], id="json", metadata={"nested": {"items": [1, None]}, "null": None})
    assert db.search([1, 0, 0], k=3, filter={"nested": {"items": [1, None]}, "null": None})
    assert not db.search([1, 0, 0], k=3, filter={"nested": {"items": [2, None]}})


def test_batch_search_preserves_query_order(db):
    db.insert_batch(
        [
            VectorRecord("x", [1, 0, 0], {"kind": "axis"}),
            VectorRecord("y", [0, 1, 0], {"kind": "axis"}),
        ]
    )
    results = db.search_batch(([1, 0, 0], [0, 1, 0]), k=2, filter={"kind": "axis"})
    assert [rows[0].id for rows in results] == ["x", "y"]


def test_duplicate_rejected_without_overwrite(db):
    db.insert([1, 0, 0], id="x")
    with pytest.raises(DuplicateIDError):
        db.insert([0, 1, 0], id="x")
    assert list(db["x"].vector) == [1, 0, 0]
    with pytest.raises(DuplicateIDError):
        db.insert_batch([VectorRecord("new", [0, 1, 0]), VectorRecord("x", [1, 0, 0])])
    assert db.get("new") is None
    with pytest.raises(DuplicateIDError):
        db.insert_batch([VectorRecord("same", [1, 0, 0]), VectorRecord("same", [0, 1, 0])])
    assert db.get("same") is None


def test_all_entries_validated_before_batch_writes(db):
    with pytest.raises(DimensionError):
        db.insert_batch([VectorRecord("valid", [1, 0, 0]), VectorRecord("bad", [1, 0])])
    assert len(db) == 0
    # Also prove preflight at the native boundary, not just the Python facade.
    with pytest.raises(DimensionError):
        db._native.execute(
            "insert_batch",
            json.dumps(
                [
                    {"id": "valid", "vector": [1, 0, 0], "metadata": None},
                    {"id": "bad", "vector": [1], "metadata": None},
                ]
            ),
        )
    assert len(db) == 0


@pytest.mark.parametrize(
    "vector",
    [
        [1, 2],
        [float("nan"), 0, 0],
        [float("inf"), 0, 0],
        [1e40, 0, 0],
        [True, 0, 0],
        ["1", 0, 0],
        None,
    ],
)
def test_invalid_vectors(db, vector):
    with pytest.raises(InvalidVectorError):
        db.insert(vector)
    with pytest.raises(InvalidVectorError):
        db.search(vector)
    assert len(db) == 0


@pytest.mark.parametrize(
    "metadata",
    [
        {"bad": float("nan")},
        {1: "bad"},
        {"nested": {1: "bad"}},
        {"bad": object()},
        ["bad"],
    ],
)
def test_invalid_metadata(db, metadata):
    with pytest.raises(InvalidVectorError):
        db.insert([1, 0, 0], metadata=metadata)
    assert len(db) == 0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(db, k):
    with pytest.raises(InvalidVectorError):
        db.search([1, 0, 0], k=k)


def test_empty_batches_and_generated_ids(db):
    assert db.insert_batch([]) == []
    assert db.search_batch([]) == []
    assert db.delete_batch([]) == []
    assert db.search([1, 0, 0], k=3) == []
    generated = db.insert([1, 0, 0], metadata={})
    assert isinstance(generated, str) and generated
    assert db.get(generated) is not None


def test_close_and_context_exception():
    db = VectorDB(3, hnsw=HNSWConfig(max_elements=128))
    with pytest.raises(RuntimeError), db:
        raise RuntimeError("user error")
    db.close()
    for operation in (
        lambda: len(db),
        db.keys,
        lambda: db.get("x"),
        lambda: db.search([1, 0, 0]),
        lambda: db.insert([1, 0, 0]),
        lambda: db.delete("x"),
        lambda: db.__enter__(),
    ):
        with pytest.raises(ClosedError):
            operation()


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_metrics(metric):
    with VectorDB(2, distance_metric=metric, hnsw=HNSWConfig(max_elements=128)) as db:
        db.insert_batch([VectorRecord("x", [1, 0]), VectorRecord("y", [0, 1])])
        result = db.search([1, 0], k=2)
        assert result[0].id == "x"
        assert result[0].score < result[1].score


@pytest.mark.parametrize("directory", ["nested", "nested folder 記憶"])
def test_persistence_in_fresh_process(tmp_path, directory):
    path = tmp_path / directory / "data.redb"
    hnsw = HNSWConfig(m=8, ef_search=64, ef_construction=64, max_elements=128)
    with VectorDB(3, path=path, distance_metric="euclidean", hnsw=hnsw) as db:
        db.insert_batch(
            [
                VectorRecord("keep", [1, 0, 0], {"tenant": "acme", "nested": [1, None]}),
                VectorRecord("delete", [0, 1, 0]),
            ]
        )
        assert db.delete("delete")
    script = """
import json, sys
from ruvector import VectorDB
with VectorDB(99, path=sys.argv[1], distance_metric="manhattan") as db:
    assert db.dimensions == 3
    assert db.options.distance_metric.value == "euclidean"
    assert db.options.hnsw.m == 8
    assert len(db) == 1
    assert db.get("delete") is None
    r = db.search([1, 0, 0], k=3, filter={"tenant": "acme"})[0]
    assert r.id == "keep" and r.metadata["nested"] == [1, None]
    db.insert([0, 0, 1], id="second")
    print(json.dumps({"dimension": db.dimensions, "count": len(db), "id": r.id}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {"dimension": 3, "count": 2, "id": "keep"}
    with VectorDB(path=path) as db:
        assert set(db.keys()) == {"keep", "second"}


def test_storage_error(tmp_path):
    path = tmp_path / "directory"
    path.mkdir()
    with pytest.raises(StorageError):
        VectorDB(3, path=path)


@pytest.mark.parametrize("dimensions", [0, -1, True, 2.5])
def test_invalid_dimensions(dimensions):
    with pytest.raises(InvalidVectorError):
        VectorDB(dimensions)


def test_invalid_options():
    with pytest.raises(InvalidVectorError):
        VectorDB(3, distance_metric="unknown")
    with pytest.raises(InvalidVectorError):
        VectorDB(3, distance_metric="dot_product")
    with pytest.raises(InvalidVectorError):
        HNSWConfig(m=1)
    with pytest.raises(InvalidVectorError):
        HNSWConfig(ef_search=0)
    with pytest.raises(InvalidVectorError):
        VectorDB(3, path="")


def test_threads_on_one_handle(db):
    def insert(i):
        return db.insert([1, i / 100, 0], id=f"thread-{i}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert len(list(executor.map(insert, range(24)))) == 24
        rows = list(executor.map(lambda _: db.search([1, 0, 0], k=5), range(12)))
    assert len(db) == 24
    assert all(row[0].id == "thread-0" for row in rows)


def test_in_memory_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with VectorDB(3, hnsw=HNSWConfig(max_elements=128)) as db:
        db.insert([1, 0, 0])
    assert list(Path(".").iterdir()) == []


def test_large_native_batch_parallel_path():
    with VectorDB(8, distance_metric="euclidean", hnsw=HNSWConfig(max_elements=256)) as db:
        entries = [
            VectorRecord(f"v-{i}", [i / 100, 1, 0, 0, 0, 0, 0, 0], {"index": i}) for i in range(96)
        ]
        assert len(db.insert_batch(entries)) == 96
        for i in (0, 31, 64, 95):
            rows = db.search(entries[i].vector, k=5)
            assert rows[0].id == f"v-{i}"
            assert rows[0].score == pytest.approx(0, abs=1e-5)


def test_reinsert_deleted_id(db):
    db.insert([1, 0, 0], id="reuse")
    assert db.delete("reuse")
    db.insert([0, 1, 0], id="reuse", metadata={"new": True})
    assert db.get("reuse").metadata == {"new": True}
    assert db.search([0, 1, 0], k=2)[0].id == "reuse"


def test_native_errors_and_batch_search_preflight(db):
    with pytest.raises(InvalidVectorError):
        db._native.execute(
            "search", json.dumps({"vector": [1, 0, 0], "k": 0, "filter": None, "ef_search": None})
        )
    with pytest.raises(InvalidVectorError):
        db._native.execute(
            "insert", json.dumps({"id": "huge", "vector": [1e100, 0, 0], "metadata": None})
        )
    with pytest.raises(DimensionError):
        db.search_batch([[1, 0, 0], [1, 0]], k=2)
    assert len(db) == 0


def test_unicode_and_zero_vectors(db):
    db.insert([0, 0, 0], id="記憶-🦀", metadata={"text": "café"})
    assert db.get("記憶-🦀").metadata == {"text": "café"}
    rows = db.search([0, 0, 0], k=1)
    assert rows[0].id == "記憶-🦀"
    assert isinstance(rows[0].score, float)
