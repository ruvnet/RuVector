import pytest
from ruvector_router import SemanticRouter

def test_native_router_replace_delete_threshold():
    router = SemanticRouter(2, threshold=.8)
    router.add('support', [1,0]); router.add('sales', [0,1])
    assert router.route([2,0])[0].name == 'support'
    router.add('support', [-1,0])
    assert len(router) == 2
    assert router.route([1,0]) == []
    assert router.remove('support')
    assert not router.remove('support')
    assert router.route([0,1])[0].name == 'sales'

def test_bad_replace_preserves_route():
    router = SemanticRouter(2)
    router.add('ok', [1,0])
    with pytest.raises(ValueError): router.add('ok', [float('nan'),0])
    assert router.route([1,0])[0].name == 'ok'

@pytest.mark.parametrize('vector', [[0,0], [1], [float('inf'),1], [[1,0]]])
def test_bad_embeddings(vector):
    with pytest.raises(ValueError): SemanticRouter(2).route(vector)

@pytest.mark.parametrize('threshold', [True, float('nan'), 1.1, -1.1])
def test_bad_threshold(threshold):
    with pytest.raises(ValueError): SemanticRouter(2, threshold=threshold)

def test_empty_and_zero_threshold():
    router = SemanticRouter(2, threshold=0)
    assert router.route([1,0]) == []
    router.add('orthogonal',[0,1])
    assert router.route([1,0])[0].score == 0
