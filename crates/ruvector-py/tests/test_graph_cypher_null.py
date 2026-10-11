"""Missing Cypher properties stay unknown through negation and logic."""
import pytest
import ruvector

@pytest.fixture
def graph():
    graph = ruvector.GraphDB()
    graph.create_node(['Thing'], {}, id='missing')
    graph.create_node(['Thing'], {'x': None}, id='null')
    graph.create_node(['Thing'], {'x': 1}, id='one')
    graph.create_node(['Thing'], {'x': 2}, id='two')
    return graph

@pytest.mark.parametrize('predicate,expected', [
    ('n.x <> 1', {'two'}),
    ('NOT (n.x = 1)', {'two'}),
    ('NOT (n.x < 2)', {'two'}),
    ('NOT (n.x = 1 OR n.x = 2)', set()),
    ('NOT (n.x = 1 AND true)', {'two'}),
    ('NOT (n.x = 1 XOR false)', {'two'}),
    ('NOT null', set()),
    ('null <> 1', set()),
    ('NOT (null = null)', set()),
    ('NOT (n.x = 1 AND false)', {'missing','null','one','two'}),
    ('n.x = 1 OR true', {'missing','null','one','two'}),
    ('n.x = 1 AND false', set()),
    ('n.x = 1 OR false', {'one'}),
    ('n.x = 1 AND true', {'one'}),
    ('NOT (n.x = 1 OR true)', set()),
    ('n.x = 1', {'one'}),
    ('n.x < 2', {'one'}),
])
def test_null_predicate_truth_tables(graph, predicate, expected):
    result = graph.query_cypher('MATCH (n:Thing) WHERE '+predicate+' RETURN n')
    assert {n['id'] for n in result['nodes']} == expected
