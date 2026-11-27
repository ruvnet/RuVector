# RuVector for LionAGI

Native Rust vector retrieval for LionAGI nodes, branch memory and agent tools.

![LionAGI and RuVector architecture](https://raw.githubusercontent.com/ruvnet/RuVector/main/python/integrations/ruvector-lionagi/docs/architecture.svg)

The adapter uses the real RuVector Python SDK HNSW index. Your application supplies embeddings; retrieval stays in memory inside your Python process. No embedding model, API credential, or database server is supplied implicitly. Search quality depends on the embeddings you choose.

## Status and installation

**Development preview, version 0.1.0. Publication to PyPI is not yet verified.** From this repository:

```bash
cd python/integrations/ruvector-lionagi
uv pip install .
python examples/quickstart.py
```

Install into an activated virtual environment. Python 3.10 or later and a compatible native RuVector wheel are required. Declared dependency ranges are `ruvector>=0.1.2,<0.2`, `lionagi>=0.35.3,<0.36`, and `numpy>=1.24,<3`. The validation report identifies the versions and platforms actually tested; the dependency ranges are not a claim that every combination has been tested.

## Getting started without API keys

The executable [quickstart](examples/quickstart.py) exercises native retrieval, a real `Branch.act()` tool call, the actual `MemoryStore` protocol and filtering. Its deterministic three-dimensional embedding function is a toy fixture, not a semantic language model.

For existing embeddings:

```python
from lionagi import Node
from ruvector_lionagi import RuVectorStore

store = RuVectorStore(dimensions=3, capacity=100)
node = Node(
    content="RuVector provides native vector search.",
    embedding=[1.0, 0.0, 0.0],
    metadata={"source": "example"},
)
store.add_nodes([node])
hit = store.search([1.0, 0.0, 0.0], k=1)[0]
assert hit.node_id == str(node.id)
print(hit.content, hit.distance)
```

Distances are cosine distances, with lower values indicating closer vectors. All vectors must have the configured dimensions, finite coordinates and nonzero norm. The adapter normalizes vectors before native indexing.

### LionAGI retrieval tool

Bind the store and an async embedding callback in application code. The tool exposes only `query` and `k` to the agent:

```python
import asyncio
from lionagi import Branch, Node
from ruvector_lionagi import RuVectorStore

async def embed(text: str) -> list[float]:
    # Demonstration only. Replace with your application's embedding service.
    return [1.0, 0.0, 0.0]

async def main():
    store = RuVectorStore(3, capacity=100)
    await store.aadd_nodes([
        Node(content="Local knowledge", embedding=await embed("Local knowledge"))
    ])
    branch = Branch(tools=[store.as_tool(embed)])
    responses = await branch.act(
        {"function": "ruvector_search", "arguments": {"query": "knowledge", "k": 1}},
        suppress_errors=False,
    )
    assert responses[0].output["untrusted_data"] is True
    print(responses[0].output["hits"])

asyncio.run(main())
```

`Branch.act()` executes the requested tool directly; this example does not call an LLM. LionAGI may write its normal local logs. Production embedding callbacks own network access, credentials, timeouts and cost control.

### LionAGI branch memory

Use a dedicated store for memory records. UUIDs, creation time, tags and metadata travel with the stored vector, without a separate mutable lookup table:

```python
import asyncio
from lionagi import Branch, MemoryItem, MemoryQuery
from ruvector_lionagi import RuVectorMemoryStore, RuVectorStore

async def embed(text: str) -> list[float]:
    # Constant fixture proves the protocol, not semantic relevance.
    return [1.0, 0.0, 0.0]

async def main():
    backend = RuVectorMemoryStore(RuVectorStore(3, capacity=100), embed)
    branch = Branch(memory=backend)
    item = MemoryItem(content="Deployment notes", tags=["ops"], metadata={"project": "demo"})
    key = await branch.memory.store(item)
    assert (await branch.memory.retrieve(key)).content == "Deployment notes"
    results = await branch.memory.search(
        MemoryQuery(text="deployment", tags=["ops"], filters={"project": "demo"}, limit=1)
    )
    assert results[0].id == key

asyncio.run(main())
```

Tags match **any** requested tag; metadata filters must **all** match. Filtered semantic queries evaluate every eligible stored record before ranking, using an exact cosine scan bounded by store capacity. They do not discard nonmatching results after approximate top-k retrieval. Unfiltered text queries use approximate HNSW. Empty text queries enumerate matching memories in insertion order without calling the embedder.

## Architecture and performance

The [animated architecture](docs/architecture.svg) shows the two retrieval paths. Async wrappers move store operations to worker threads; operations within a store are serialized. This is an async integration contract, not a claim that native computation releases the Python GIL.

| Operation | Behavior and cost |
| --- | --- |
| Unfiltered vector search | Native approximate HNSW; recall depends on data and configuration |
| Filtered search | Exact cosine ranking over eligible records; linear scan of bounded store |
| Read by UUID | Detached payload lookup without embedding |
| Delete | Rebuilds the remaining native index to reclaim removed entries |
| Insert | Validates the batch before native mutation; rejects duplicate IDs |

See the [benchmark report](benchmarks/README.md) for measured latency and recall, and rerun its commands on your workload. Hardware, dataset size, dimensions and query distribution affect results. Do not extrapolate synthetic measurements into production guarantees.

## Security boundaries and limits

Use separate stores for separate authorization scopes. Metadata filters are not authentication or tenant isolation. The tool binds its store and embedder in trusted host code; callers cannot select a path, endpoint, model or tenant. Returned content is explicitly marked as untrusted data and must not become agent instructions.

The adapter checks dimensions, coordinate values, content bytes, finite JSON metadata, metadata depth, capacity and output limits. It preserves retrieval payloads as detached snapshots and does not dynamically reconstruct classes from stored metadata. There is no pickle loading, filesystem persistence or arbitrary query expression execution.

Memory content is text only. Memory tags are limited to 128 strings of at most 256 bytes each. Query limits must be positive and within the configured store maximum. Duplicate IDs are rejected rather than updated. After cancellation, a worker that has already started may complete insertion; retrieve the known UUID before retrying. A native insertion failure disables that store instance to avoid using partially mutated state.

## Validation and scope

Run the focused tests and offline example:

```bash
python -m pytest tests -q
python examples/quickstart.py
```

Install test dependencies with `python -m pip install pytest==9.1.1 pytest-asyncio==1.4.0 pytest-cov==7.1.0` before running pytest. Tests use real LionAGI and the native RuVector SDK. Consult the [validation](docs/VALIDATION.md) and [security](docs/SECURITY.md) reports for the tested boundaries and any remaining gaps.

This preview supplies in-memory cosine retrieval, LionAGI Nodes, a read-only retrieval Tool and branch memory. Embedding generation, persistence, distributed access, authorization middleware, prompt-injection defenses and LLM inference remain application responsibilities. It does not claim full LionAGI, RuVector SDK or npm feature parity.

## Release automation

The repository workflow `python-lionagi.yml` builds wheel and source distributions, validates metadata, runs the contract suite and quickstart across Linux, macOS and Windows, and audits dependencies. A `python-lionagi-v0.1.0` tag pointing to an ancestor of main triggers publishing after the test matrix succeeds. Maintainers can also explicitly dispatch publishing from main. The protected `pypi` environment supplies its configured credential or Trusted Publishing identity; no credential belongs in source code. A final matrix installs the exact version from PyPI and reruns acceptance tests.

Publication is complete only when those post-publication checks succeed. Version numbers must advance for subsequent releases; an existing artifact is never overwritten.
