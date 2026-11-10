from ruvector_graph import GraphDB, reachable
graph = GraphDB()
a = graph.create_node(["Document"], {"title": "Rust"}, id="a")
b = graph.create_node(["Document"], {"title": "Python"}, id="b")
graph.create_edge(a, b, "RELATED")
assert reachable(graph, a) == [a, b]
