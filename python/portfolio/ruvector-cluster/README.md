# ruvector-cluster: Embedding clustering in Python

![Embedding clustering: Python API to PyO3 to Rust](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/cluster.svg)

## What is ruvector-cluster?

ruvector-cluster provides embedding clustering through `ruvector-cluster-rag` in the RuVector Rust library family. It adds a focused Python API over the existing `ruvector` PyO3 runtime. It does not ship a separate native engine or reduce the shared runtime's installation size.

## Install

```bash
pip install ruvector-cluster==0.2.0
```

Release candidate: this command works only after publication. From a checkout, install `ruvector==0.1.4` and then `pip install ./python/portfolio/ruvector-cluster`. Python 3.10 or newer; native platform support follows the shared runtime wheels.

## Quick start

```python
from ruvector_cluster import cluster
result = cluster([[1, 0], [1.1, 0], [10, 10], [10.1, 10]], 2)
assert result.labels[0] == result.labels[1]
assert result.labels[2] == result.labels[3]
assert result.labels[0] != result.labels[2]
assert result.sizes.sum() == 4
```

## API and limitations

Public exports: `ClusterResult, cluster`. Lloyd clustering for numeric embeddings. No embedding model download or network service. Quality and latency depend on your data; no universal speedup or recall claim.

Input conversion can copy arrays. Reuse contiguous float32 arrays for repeated operations. No service, telemetry or credentials are added by this package. Apply application memory and request limits for untrusted data.

## How does it relate to RuVector?

The algorithm executes through the existing Rust PyO3 binding, pinned to `ruvector==0.1.4`. Python implements convenience conversion and result formatting. For vector search use `pip install ruvector`; for other capabilities see the [Python portfolio](https://github.com/ruvnet/RuVector/tree/main/python/portfolio).

## Verification and security

CI tests built wheels against a native runtime compiled from this repository, checks sdists and README rendering, audits resolved dependencies and publishes through PyPI Trusted Publishing. Native vulnerability scanning is a separate required gate. See [security scope](https://github.com/ruvnet/RuVector/blob/main/python/portfolio/SECURITY.md).
