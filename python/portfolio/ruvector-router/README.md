# ruvector-router

![RuVector semantic routing](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/router.svg)

Route embedding vectors to named intents with native RuVector HNSW search and an explicit cosine similarity threshold. Supply your own embeddings; the router makes no network requests.

## Getting started

```sh
python -m pip install ruvector-router==0.2.0
```

```python
from ruvector_router import SemanticRouter
router = SemanticRouter(2, threshold=0.8)
router.add('support', [1, 0])
router.add('sales', [0, 1])
assert router.route([1, 0])[0].name == 'support'
router.add('support', [-1, 0])  # Replace an existing route
assert router.route([1, 0]) == []
assert router.remove('support')
assert not router.remove('support')
```

## Architecture and scope

Caller embeddings → finite nonzero vector validation → normalization → native HNSW nearest routes → cosine threshold → typed `RouteMatch` values.

One representative embedding per route. HNSW is approximate; thresholds are inclusive and do not guarantee intent classification quality. This is a Python semantic policy over the native core index, not a binding to the separate neural router or an embedding model. Mutation rebuilds the index before committing, ensuring replacements and deletions cannot leave stale routes, at a cost proportional to the number of routes. Designed for small intent catalogs rather than high frequency writes to large indexes. Ties among retrieved results sort by name.

Python 3.10 or later; runtime `ruvector==0.1.4`. [Portfolio documentation](https://github.com/ruvnet/RuVector/tree/main/python/portfolio).
