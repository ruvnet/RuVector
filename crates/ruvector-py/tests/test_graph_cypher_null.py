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


@pytest.mark.parametrize('predicate,expected', [
    ('n.x = 9007199254740992', {'low'}),
    ('n.x <> 9007199254740992', {'high'}),
    ('n.x > 9007199254740992', {'high'}),
    ('n.x < 9007199254740993', {'low'}),
    ('n.x = 9007199254740992.0', {'low'}),
    ('n.x > 9007199254740992.0', {'high'}),
    ('9007199254740992.0 < n.x', {'high'}),
    ('n.x >= 9007199254740993', {'high'}),
    ('n.x <= 9007199254740992', {'low'}),
])
def test_large_integer_predicates(predicate, expected):
    graph = ruvector.GraphDB()
    graph.create_node(['Thing'], {'x': 9007199254740992}, id='low')
    graph.create_node(['Thing'], {'x': 9007199254740993}, id='high')
    result = graph.query_cypher('MATCH (n:Thing) WHERE '+predicate+' RETURN n')
    assert {n['id'] for n in result['nodes']} == expected


@pytest.mark.parametrize('value,predicate,expected', [
    (9223372036854775807, 'n.x < 9223372036854775808.0', True),
    (9223372036854775807, 'n.x = 9223372036854775808.0', False),
    (-9223372036854775808, 'n.x = -9223372036854775808.0', True),
    (-9007199254740993, 'n.x < -9007199254740992.0', True),
    (-9007199254740993, 'n.x = -9007199254740992.0', False),
    (2, 'n.x > 1.5', True),
    (-2, 'n.x < -1.5', True),
    (2, 'n.x = 2.0', True),
    (2.0, 'n.x = 2', True),
    (2.5, 'n.x > 2', True),
])
def test_mixed_numeric_predicate_boundaries(value,predicate,expected):
    graph = ruvector.GraphDB()
    graph.create_node(['Thing'], {'x': value}, id='value')
    result = graph.query_cypher('MATCH (n:Thing) WHERE '+predicate+' RETURN n')
    assert bool(result['nodes']) == expected


@pytest.mark.parametrize('stored,query,expected', [
    ([9007199254740993], '[9007199254740992]', False),
    ([9007199254740993], '[9007199254740992.0]', False),
    ([2], '[2.0]', True),
])
def test_numeric_list_equality_control(stored, query, expected):
    graph = ruvector.GraphDB()
    graph.create_node(['Thing'], {'xs': stored}, id='value')
    result = graph.query_cypher('MATCH (n:Thing) WHERE n.xs = '+query+' RETURN n')
    assert bool(result['nodes']) == expected
