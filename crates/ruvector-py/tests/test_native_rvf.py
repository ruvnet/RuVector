"""Exercise native persistence, retrieval, mutation and input rejection."""
import pytest
from ruvector import RvfStore


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


def test_checkpoint_metadata_and_export(tmp_path):
    path = str(tmp_path / 'checkpoint.rvf')
    store = RvfStore.create(path, 2)
    try:
        assert store.get_metadata('app.checkpoint') is None
        store.ingest([[1, 0], [0, 1], [-1, 0]], [10, 20, 30])
        store.delete([20])
        store.set_metadata('app.checkpoint', '{"revision":1,"text":"café"}')
        store.set_metadata('app.checkpoint', '{"revision":2}')
        assert sorted(store.export_vectors()) == [(10, [1.0, 0.0]), (30, [-1.0, 0.0])]
    finally:
        store.close()
    store = RvfStore.open(path)
    try:
        assert store.metadata_recovery()['dropped_generations'] == 0
        assert store.metadata_recovery()['dropped_records'] == 0
        assert store.get_metadata('app.checkpoint') == '{"revision":2}'
        assert sorted(store.export_vectors()) == [(10, [1.0, 0.0]), (30, [-1.0, 0.0])]
        with pytest.raises(RuntimeError):
            store.set_metadata('app.checkpoint', 'unauthorized')
        assert store.metadata_recovery()['dropped_generations'] == 0
        assert store.metadata_recovery()['dropped_records'] == 0
        assert store.get_metadata('app.checkpoint') == '{"revision":2}'
        with pytest.raises(ValueError, match='exceed export limits'):
            store.export_vectors(max_vectors=1)
        with pytest.raises(ValueError, match='exceed export limits'):
            store.export_vectors(max_bytes=15)
        assert len(store.export_vectors(max_vectors=2, max_bytes=16)) == 2
    finally:
        store.close()
    for operation in [lambda: store.export_vectors(), lambda: store.get_metadata('app.x'), lambda: store.set_metadata('app.x', 'value')]:
        with pytest.raises(RuntimeError, match='closed'):
            operation()


@pytest.mark.parametrize('key', ['', 'x' * 257, 'é' * 129, 'rvf.private'])
def test_metadata_key_bounds(tmp_path, key):
    store = RvfStore.create(str(tmp_path / 'bounds.rvf'), 2)
    try:
        with pytest.raises(ValueError):
            store.set_metadata(key, 'test')
        with pytest.raises(ValueError):
            store.get_metadata(key)
    finally:
        store.close()


@pytest.mark.parametrize('limits', [{'max_vectors': 0}, {'max_vectors': 1000001}, {'max_bytes': 0}, {'max_bytes': 1073741825}])
def test_export_bounds(tmp_path, limits):
    store = RvfStore.create(str(tmp_path / 'bounds.rvf'), 2)
    try:
        with pytest.raises(ValueError):
            store.export_vectors(**limits)
        assert store.export_vectors() == []
    finally:
        store.close()


def test_metadata_value_bound(tmp_path):
    store = RvfStore.create(str(tmp_path / 'bounds.rvf'), 2)
    try:
        with pytest.raises(ValueError, match='16 MiB'):
            store.set_metadata('app.oversize', 'x' * (16 * 1024 * 1024 + 1))
        assert store.get_metadata('app.oversize') is None
    finally:
        store.close()


def test_metadata_recovery_reports_corrupt_generation(tmp_path):
    path = tmp_path / 'corrupt.rvf'
    store = RvfStore.create(str(path), 2)
    try:
        store.ingest([[1, 0]], [1])
        store.set_metadata('app.state', 'previous')
        store.set_metadata('app.state', 'latest')
    finally:
        store.close()
    data = bytearray(path.read_bytes())
    offset = 0
    metadata = []
    while offset + 64 <= len(data):
        length = int.from_bytes(data[offset + 16:offset + 24], 'little')
        end = offset + 64 + length
        assert end <= len(data)
        if data[offset + 5] == 7:
            metadata.append((offset + 64, length))
        offset = end
    start, length = metadata[-1]
    data[start + length // 2] ^= 255
    path.write_bytes(data)
    store = RvfStore.open(str(path))
    try:
        assert store.metadata_recovery()['dropped_generations'] > 0
        assert store.get_metadata('app.state') == 'previous'
    finally:
        store.close()


def test_metadata_exact_utf8_bounds_reopen(tmp_path):
    path = str(tmp_path / 'exact-bounds.rvf')
    key = 'é' * 128  # 256 UTF-8 bytes
    value = 'é' * (8 * 1024 * 1024)  # 16 MiB UTF-8, native codec maximum
    store = RvfStore.create(path, 2)
    try:
        store.set_metadata(key, value)
        with pytest.raises(ValueError, match='16 MiB'):
            store.set_metadata(key, value + 'x')
        assert store.get_metadata(key) == value
    finally:
        store.close()
    store = RvfStore.open(path)
    try:
        assert store.get_metadata(key) == value
        assert store.metadata_recovery()['dropped_generations'] == 0
    finally:
        store.close()
