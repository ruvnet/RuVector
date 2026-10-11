"""Incremental native batches preserve serial encoding and validation semantics."""
import numpy as np
import pytest
import ruvector as rv


@pytest.mark.parametrize("dim", [16, 65])
def test_batch_matches_build_and_serial_with_external_ids(dim):
    data = np.random.default_rng(1141).standard_normal((80, dim), dtype=np.float32)
    ids = np.arange(80, dtype=np.uint64) * 7 + 1000
    bulk = rv.RabitqIndex.build(data, ids=ids, seed=42)
    batch = rv.RabitqIndex.build(data[:3], ids=ids[:3], seed=42)
    serial = rv.RabitqIndex.build(data[:3], ids=ids[:3], seed=42)
    batch.add_batch(ids[3:3], data[3:3])
    batch.add_batch(ids[3:4], data[3:4])
    batch.add_batch(ids[4:], data[4:])
    for ident, row in zip(ids[3:], data[3:]):
        serial.add(int(ident), row)
    assert len(batch) == len(bulk) == len(serial) == 80
    assert batch.memory_bytes == bulk.memory_bytes == serial.memory_bytes
    for query in data[:8]:
        expected = bulk.search(query, 10)
        assert batch.search(query, 10) == expected
        assert serial.search(query, 10) == expected
    # Export rows are also retained in caller order; external-ID export is
    # covered by its separate identity fix, so this test compares vectors.
    np.testing.assert_array_equal(np.stack([v for _, v in batch.export_items()]), data)


@pytest.mark.parametrize("invalid", ["dimension", "nonfinite", "duplicate", "overflow"])
def test_rejected_batch_does_not_append_or_reserve_ids(invalid):
    data = np.random.default_rng(1141).standard_normal((8, 16), dtype=np.float32)
    idx = rv.RabitqIndex.build(data[:3], seed=42)
    before = idx.search(data[0], 3)
    ids = np.array([100, 101], dtype=np.uint64)
    rows = data[3:5].copy()
    if invalid == "dimension":
        rows = rows[:, :-1].copy()
    elif invalid == "nonfinite":
        rows[1, 3] = np.nan
    elif invalid == "duplicate":
        ids[1] = 1
    else:
        ids[1] = 2**32
    with pytest.raises(ValueError):
        idx.add_batch(ids, rows)
    assert len(idx) == 3
    assert idx.search(data[0], 3) == before
    idx.add_batch(np.array([100, 101], dtype=np.uint64), data[3:5])
    assert len(idx) == 5
    assert idx.search(data[3], 1)[0][0] == 100
