import numpy as np
import pytest
from ruvector_graph import GraphDB, reachable
from ruvector_cluster import cluster
from ruvector_rerank import rank


def test_cycle_and_depth():
    graph = GraphDB()
    for key in ('a', 'b', 'c'):
        graph.create_node(id=key)
    graph.create_edge('a', 'b', 'LINK')
    graph.create_edge('b', 'c', 'LINK')
    graph.create_edge('c', 'a', 'LINK')
    assert reachable(graph, 'a', max_depth=0) == ['a']
    assert reachable(graph, 'a', max_depth=1) == ['a', 'b']
    assert reachable(graph, 'a') == ['a', 'b', 'c']
    with pytest.raises(ValueError, match='budget'):
        reachable(graph, 'a', max_nodes=2)
    with pytest.raises(KeyError):
        reachable(graph, 'missing')


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_invalid_depth(value):
    with pytest.raises(ValueError):
        reachable(GraphDB(), 'a', max_depth=value)


@pytest.mark.parametrize('vectors,k', [([],1), ([[1]],2), ([[float('nan')]],1), ([[1]],0), ([[1]],True)])
def test_cluster_validation(vectors,k):
    with pytest.raises(ValueError):
        cluster(vectors,k)


def test_cluster_conservation():
    data = np.array([[1,0],[1.1,0],[10,10],[10.1,10]])
    result = cluster(data,2)
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]
    assert result.sizes.sum() == len(data)
    for i in range(2):
        np.testing.assert_allclose(result.centroids[i],data[result.labels == i].mean(axis=0),rtol=1e-5)


def test_attention_matches_independent_numpy_reference():
    q = np.array([1,2,3],dtype=np.float32)
    c = np.array([[2,0,1],[1,1,1],[-2,0,-1],[1,1,1]],dtype=np.float32)
    logits = c @ q / np.sqrt(3)
    expected = np.exp(logits-logits.max())
    expected /= expected.sum()
    hits = rank(q,c)
    assert [h.index for h in hits] == np.argsort(-expected,kind='stable').tolist()
    np.testing.assert_allclose([h.weight for h in hits],expected[[h.index for h in hits]],rtol=1e-5)
    assert len(rank(q,c,top_k=2)) == 2


@pytest.mark.parametrize('query,candidates', [([],[[1]]),([1],[]),([1],[[1,2]]),([float('inf')],[[1]])])
def test_rank_validation(query,candidates):
    with pytest.raises(ValueError):
        rank(query,candidates)


@pytest.mark.parametrize('top_k', [0, -1, True, 0.5])
def test_rank_top_k_validation(top_k):
    with pytest.raises(ValueError):
        rank([1],[[1]],top_k=top_k)
