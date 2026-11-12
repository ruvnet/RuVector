# ruvector-rerank: Attention ranking for RAG in Python

![Attention ranking for RAG: Python API to PyO3 to Rust](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/rerank.svg)

## What is ruvector-rerank?

ruvector-rerank provides attention ranking for rag through `ruvector-attention` in the RuVector Rust library family. It adds a focused Python API over the existing `ruvector` PyO3 runtime. It does not ship a separate native engine or reduce the shared runtime's installation size.

## Install

```bash
pip install ruvector-rerank==0.2.0
```

Release candidate: this command works only after publication. From a checkout, install `ruvector==0.1.4` and then `pip install ./python/portfolio/ruvector-rerank`. Python 3.10 or newer; native platform support follows the shared runtime wheels.

## Quick start

```python
from ruvector_rerank import rank
hits = rank([1, 0], [[0, 1], [1, 0], [-1, 0]], top_k=2)
assert [hit.index for hit in hits] == [1, 0]
```

## API and limitations

Public exports: `RankedCandidate, rank`. Scaled dot product attention ranking, not a trained semantic cross encoder. It cannot improve ordering over the same dot product scores. Useful for attention weights, deterministic candidate ordering and RAG plumbing.

Input conversion can copy arrays. Reuse contiguous float32 arrays for repeated operations. No service, telemetry or credentials are added by this package. Apply application memory and request limits for untrusted data.

## How does it relate to RuVector?

The algorithm executes through the existing Rust PyO3 binding, pinned to `ruvector==0.1.4`. Python implements convenience conversion and result formatting. For vector search use `pip install ruvector`; for other capabilities see the [Python portfolio](https://github.com/ruvnet/RuVector/tree/main/python/portfolio).

## Verification and security

CI tests built wheels against a native runtime compiled from this repository, checks sdists and README rendering, audits resolved dependencies and publishes through PyPI Trusted Publishing. Native vulnerability scanning is a separate required gate. See [security scope](https://github.com/ruvnet/RuVector/blob/main/python/portfolio/SECURITY.md).
