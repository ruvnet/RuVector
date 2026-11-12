# ruvector-graph: Property graph traversal in Python

![Property graph traversal: Python API to PyO3 to Rust](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/graph.svg)

## What is ruvector-graph?

ruvector-graph provides property graph traversal through `ruvector-graph` in the RuVector Rust library family. It adds a focused Python API over the existing `ruvector` PyO3 runtime. It does not ship a separate native engine or reduce the shared runtime's installation size.

## Install

```bash
pip install ruvector-graph==0.2.0
```

Release candidate: this command works only after publication. From a checkout, install `ruvector==0.1.4` and then `pip install ./python/portfolio/ruvector-graph`. Python 3.10 or newer; native platform support follows the shared runtime wheels.

## Quick start

```python
from ruvector_graph import GraphDB, reachable
graph = GraphDB()
a = graph.create_node(["Document"], {"title": "Rust"}, id="a")
b = graph.create_node(["Document"], {"title": "Python"}, id="b")
graph.create_edge(a, b, "RELATED")
assert reachable(graph, a) == [a, b]
```

## API and limitations

Public exports: `GraphDB, reachable`. In memory property graphs and bounded breadth first traversal. Not a persistent graph server. Node caps do not cap the temporary outgoing edge list; isolate untrusted workloads with process memory limits.

Input conversion can copy arrays. Reuse contiguous float32 arrays for repeated operations. No service, telemetry or credentials are added by this package. Apply application memory and request limits for untrusted data.

## How does it relate to RuVector?

The algorithm executes through the existing Rust PyO3 binding, pinned to `ruvector==0.1.4`. Python implements convenience conversion and result formatting. For vector search use `pip install ruvector`; for other capabilities see the [Python portfolio](https://github.com/ruvnet/RuVector/tree/main/python/portfolio).

## Verification and security

CI tests built wheels against a native runtime compiled from this repository, checks sdists and README rendering, audits resolved dependencies and publishes through PyPI Trusted Publishing. Native vulnerability scanning is a separate required gate. See [security scope](https://github.com/ruvnet/RuVector/blob/main/python/portfolio/SECURITY.md).
