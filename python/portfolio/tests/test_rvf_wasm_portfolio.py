import pytest
from ruvector_rvf_wasm import VectorStore

@pytest.mark.parametrize('metric,expected', [('l2', 25), ('inner_product', -11), ('cosine', 0)])
def test_real_wasm_distances(metric, expected):
    with VectorStore(2, metric) as db:
        if metric == 'l2':
            db.add(2**63 + 7, [3, 4]); query = [0, 0]
        elif metric == 'inner_product':
            db.add(2**63 + 7, [1, 2]); query = [3, 4]
        else:
            db.add(2**63 + 7, [3, 4]); query = [3, 4]
        hit = db.query(query, 1)[0]
        assert hit.id == 2**63 + 7
        assert hit.distance == pytest.approx(expected, abs=1e-6)

def test_ranking_and_isolation():
    with VectorStore(2) as first, VectorStore(2) as second:
        first.add(5, [10, 10]); first.add(8, [1, 1])
        assert [x.id for x in first.query([0, 0])] == [8, 5]
        assert second.query([0, 0]) == []
        with pytest.raises(ValueError): first.add(5, [1, 2])
        with pytest.raises(ValueError): first.add(7, [float('nan'), 0])
        with pytest.raises(ValueError): first.query([0])
        with pytest.raises(ValueError): first.query([0, 0], 0)
    with pytest.raises(RuntimeError): first.query([0, 0])

def test_fuel_limit_poisoning():
    db = VectorStore(2)
    db._fuel = 1
    with pytest.raises(RuntimeError, match='execution failed'):
        db.add(1, [1, 2])
    with pytest.raises(RuntimeError, match='closed'):
        db.query([0, 0])
    db.close()

def test_memory_limit_and_no_imports():
    db = VectorStore(2, memory_bytes=2**21)
    with pytest.raises(Exception):
        db._memory.grow(db._store, 100)
    db.close()

@pytest.mark.parametrize('scale', [1e-20, 1.0, 1e15])
def test_cosine_scale_invariance(scale):
    with VectorStore(2, 'cosine') as db:
        db.add(1, [scale, scale])
        db.add(2, [-scale, -scale])
        hits = db.query([scale, scale])
        assert [hit.id for hit in hits] == [1, 2]
        assert [hit.distance for hit in hits] == pytest.approx([0, 2], abs=1e-6)
        with pytest.raises(ValueError, match='nonzero norm'):
            db.add(3, [0, 0])
        with pytest.raises(ValueError, match='nonzero norm'):
            db.query([0, 0])

def test_input_iterator_is_bounded():
    consumed = []
    def stream():
        for i in range(100):
            consumed.append(i)
            yield float(i)
    with VectorStore(2) as db:
        with pytest.raises(ValueError, match='dimension mismatch'):
            db.add(1, stream())
    assert consumed == [0, 1, 2]
