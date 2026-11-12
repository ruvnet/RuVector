"""Exercise native persistence, retrieval, mutation and input rejection."""
import pytest
from ruvector_rvf import RvfStore


def test_persistence_query_delete(tmp_path):
    path = str(tmp_path / 'test.rvf')
    store = RvfStore.create(path, 3)
    try:
        assert store.dimension == 3
        assert store.ingest([[1, 0, 0], [0, 1, 0]], [10, 20]) == 2
        assert store.query([1, 0, 0], 1)[0] == (10, 0.0)
    finally:
        store.close()
    store = RvfStore.open(path, read_only=False)
    try:
        assert store.status()['total_vectors'] == 2
        assert store.delete([10]) == 1
        assert [hit[0] for hit in store.query([1, 0, 0], 10)] == [20]
    finally:
        store.close()
    store = RvfStore.open(path)
    try:
        assert store.status()['read_only'] == 1
        assert store.query([0, 1, 0], 1)[0][0] == 20
        with pytest.raises(RuntimeError):
            store.ingest([[0, 0, 1]], [30])
    finally:
        store.close()


def test_no_overwrite_and_closed(tmp_path):
    path = str(tmp_path / 'keep.rvf')
    store = RvfStore.create(path, 2)
    store.close()
    before = (tmp_path / 'keep.rvf').read_bytes()
    with pytest.raises(FileExistsError):
        RvfStore.create(path, 2)
    assert (tmp_path / 'keep.rvf').read_bytes() == before
    store.close()
    with pytest.raises(RuntimeError, match='closed'):
        store.status()


def test_input_validation(tmp_path):
    with pytest.raises(ValueError):
        RvfStore.create(str(tmp_path / 'bad.rvf'), 0)
    store = RvfStore.create(str(tmp_path / 'valid.rvf'), 2)
    try:
        for vectors, ids in [([], []), ([[1, 2]], []), ([[float('nan'), 1]], [1]), ([[1e30, 1]], [1]), ([[1]], [1]), ([[1, 2], [2, 3]], [1, 1])]:
            with pytest.raises(ValueError):
                store.ingest(vectors, ids)
        for vector, k in [([1], 1), ([float('inf'), 1], 1), ([1, 2], 0), ([1, 2], 10001)]:
            with pytest.raises(ValueError):
                store.query(vector, k)
        assert store.status()['total_vectors'] == 0
    finally:
        store.close()
