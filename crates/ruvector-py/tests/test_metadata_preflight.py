"""A rejected metadata conversion must not leave native rows behind."""

import numpy as np
import pytest
from ruvector import Collection


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_rejected_metadata_preserves_rows_and_next_insert(existing, batch):
    coll = Collection.create(dim=2, backend="rabitq")
    if existing:
        coll.insert(np.array([1.0, 0.0], dtype=np.float32), metadata={"name": "old"})
    before = len(coll)
    if batch:
        with pytest.raises((ValueError, TypeError)):
            coll.insert_batch(
                np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
                metadatas=[{"name": "valid"}, "invalid"],
            )
    else:
        with pytest.raises((ValueError, TypeError)):
            coll.insert(np.array([0.0, 1.0], dtype=np.float32), metadata="invalid")
    assert len(coll) == before
    assert (
        coll.insert(np.array([0.0, 1.0], dtype=np.float32), metadata={"name": "new"})
        == before
    )
    assert len(coll) == before + 1
    assert coll.get_metadata(before) == {"name": "new"}


@pytest.mark.parametrize("backend", ["hnsw", "rabitq"])
def test_valid_metadata_and_none_control(backend):
    coll = Collection.create(dim=2, backend=backend)
    assert (
        coll.insert(np.array([1.0, 0.0], dtype=np.float32), metadata={"name": "one"})
        == 0
    )
    assert coll.insert_batch(
        np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        metadatas=[None, {"name": "three"}],
    ) == [1, 2]
    assert len(coll) == 3
    assert coll.get_metadata(0) == {"name": "one"}
    assert coll.get_metadata(2) == {"name": "three"}


@pytest.mark.parametrize("backend", ["hnsw", "rabitq"])
def test_metadata_is_materialized_once_and_is_a_snapshot(backend):
    from collections.abc import Mapping

    class OnceMapping(Mapping):
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise RuntimeError("metadata was converted twice")
            return iter(["name"])

        def __len__(self):
            return 1

        def __getitem__(self, key):
            if key == "name":
                return "once"
            raise KeyError(key)

    coll = Collection.create(dim=2, backend=backend)
    single = OnceMapping()
    batch = OnceMapping()
    assert coll.insert(np.array([1.0, 0.0], dtype=np.float32), metadata=single) == 0
    assert coll.insert_batch(
        np.array([[0.0, 1.0]], dtype=np.float32), metadatas=[batch]
    ) == [1]
    assert single.iterations == batch.iterations == 1
    assert coll.get_metadata(0) == coll.get_metadata(1) == {"name": "once"}
    original = {"name": "snapshot"}
    row = coll.insert(np.array([1.0, 1.0], dtype=np.float32), metadata=original)
    original["name"] = "changed"
    assert coll.get_metadata(row) == {"name": "snapshot"}


@pytest.mark.parametrize("backend", ["hnsw", "rabitq"])
def test_none_and_empty_metadata_control(backend):
    coll = Collection.create(dim=2, backend=backend)
    assert coll.insert(np.array([1.0, 0.0], dtype=np.float32), metadata={}) == 0
    assert coll.insert_batch(
        np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32), metadatas=[{}, None]
    ) == [1, 2]
    assert len(coll) == 3
    assert coll.get_metadata(0) in (None, {})
    assert coll.get_metadata(1) in (None, {})
    assert coll.get_metadata(2) in (None, {})
