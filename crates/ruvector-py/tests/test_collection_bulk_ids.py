import numpy as np
import pytest
from ruvector import Collection
from ruvector.collection import CollectionError

@pytest.mark.parametrize('backend',['hnsw','rabitq'])
@pytest.mark.parametrize('bad_id',[1.5,True])
def test_bulk_constructor_rejects_noninteger_ids_before_build(backend,bad_id):
    with pytest.raises(CollectionError):
        Collection.from_vectors(np.array([[1.,0.]],dtype=np.float32),ids=[bad_id],backend=backend)

@pytest.mark.parametrize('backend',['hnsw','rabitq'])
def test_numpy_integer_ids_roundtrip_control(tmp_path,backend):
    coll=Collection.from_vectors(np.array([[1.,0.]],dtype=np.float32),ids=[np.int64(7)],backend=backend)
    assert coll.search(np.array([1.,0.],dtype=np.float32),1)[0].id==7
    path=tmp_path/'ids.rbpx'
    coll.save(path)
    loaded=Collection.load(path)
    assert len(loaded)==1
    hits=loaded.search(np.array([1.,0.],dtype=np.float32),1)
    assert hits[0].score==0
    if backend=='hnsw':
        assert hits[0].id==7
    # RaBitQ saved-ID preservation is separately repaired in PR1156.
