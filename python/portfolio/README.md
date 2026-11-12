# RuVector Python portfolio

![RuVector Python capability tracks](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/overview.svg)

Python access to the ten most downloaded RuVector npm packages. Eight new companion distributions join the existing `ruvector` SDK and `ruvector-graph`. Previously released clustering and reranking remain available. Every companion has its own introduction, animated diagram, runnable getting started example and explicit capability limits.

## Release status, October 12, 2026

The 0.2.0 portfolio is **partially published**. Attention and cluster wheels are
available; both source archives and nine target releases are still missing.
The original files pin native `ruvector==0.1.3`. Current source targets 0.1.4
and must not replace immutable 0.2.0 artifacts. See the
[verified inventory and recovery evidence](../../docs/python-portfolio/evidence/2026-10-12/README.md)
and [recovery design](../../docs/python-portfolio/resume-release-adr.md).

The published attention package has a
[tested synthetic-data tutorial](tutorials/README.md). Other implemented tracks
below describe source capabilities and do not establish PyPI availability.

## Which packages are most popular?

Measured npm downloads for September 11 through October 10, 2026. These are package retrievals, not unique users. Platform binary variants are excluded. RVF WASM remains a separate distribution because it provides a distinct portable execution backend. [Raw evidence and all source URLs](npm-popularity-2026-10-11.json).

| Rank | npm package | Monthly downloads | Python package | Implemented scope |
| --- | --- | ---: | --- | --- |
| 1 | `@ruvector/ruvllm` | 519,297 | [`ruvector-ruvllm`](https://pypi.org/project/ruvector-ruvllm/) | Rust KV cache and attention; no text generation |
| 2 | `ruvector` | 506,811 | [`ruvector`](https://pypi.org/project/ruvector/) | Shared native SDK, CLI and integrations |
| 3 | `@ruvector/rvf` | 494,402 | [`ruvector-rvf`](https://pypi.org/project/ruvector-rvf/) | Persistent native RVF vector stores |
| 4 | `@ruvector/graph-node` | 415,506 | [`ruvector-graph`](https://pypi.org/project/ruvector-graph/) | Property graph CRUD and bounded traversal |
| 5 | `@ruvector/sona` | 395,416 | [`ruvector-sona`](https://pypi.org/project/ruvector-sona/) | Adapter inference; fresh engine is identity |
| 6 | `@ruvector/attention` | 364,417 | [`ruvector-attention`](https://pypi.org/project/ruvector-attention/) | Scaled dot product attention |
| 7 | `@ruvector/router` | 362,362 | [`ruvector-router`](https://pypi.org/project/ruvector-router/) | Semantic intent routing over Rust HNSW |
| 8 | `@ruvector/gnn` | 344,076 | [`ruvector-gnn`](https://pypi.org/project/ruvector-gnn/) | GNN forward inference; no training API |
| 9 | `@ruvector/core` | 341,954 | [`ruvector-core`](https://pypi.org/project/ruvector-core/) | Validated vector indexing and search |
| 10 | `@ruvector/rvf-wasm` | 328,808 | [`ruvector-rvf-wasm`](https://pypi.org/project/ruvector-rvf-wasm/) | Actual RVF WASM in a bounded Wasmtime host |

## Getting started

```sh
python -m venv .venv
# Activate the environment using your platform's normal command.
python -m pip install ruvector-attention==0.2.0 ruvector==0.1.3 numpy==2.2.6
# Run the published-package example in tutorials/attention_quickstart.py.
```

Choose a task and open its package guide:

* [ruvector-ruvllm](ruvector-ruvllm/README.md): Rust KV cache and attention; no text generation.
* [ruvector-rvf](ruvector-rvf/README.md): Persistent native RVF vector stores.
* [ruvector-graph](ruvector-graph/README.md): Property graph CRUD and bounded traversal.
* [ruvector-sona](ruvector-sona/README.md): Adapter inference; fresh engine is identity.
* [ruvector-attention](ruvector-attention/README.md): Scaled dot product attention.
* [ruvector-router](ruvector-router/README.md): Semantic intent routing over Rust HNSW.
* [ruvector-gnn](ruvector-gnn/README.md): GNN forward inference; no training API.
* [ruvector-core](ruvector-core/README.md): Validated vector indexing and search.
* [ruvector-rvf-wasm](ruvector-rvf-wasm/README.md): Actual RVF WASM in a bounded Wasmtime host.

The base SDK guide is [here](../../crates/ruvector-py/README.md). All companions pin the same `ruvector==0.1.4` native runtime, avoiding duplicate Rust extensions. RVF WASM additionally pins Wasmtime and bundles an integrity verified upstream module. It requires no Node, browser, WASI, network downloads or npm install at runtime.

## What is supported?

These are deliberately scoped Python APIs, not promises of complete npm parity. SONA has inference only and cannot train through this binding. Fresh GNN parameters are untrained. The ruvllm package implements full precision KV caching, not language generation. The RVF store opens trusted local files; it is not an untrusted file sandbox. The WASM package runs actual upstream bytecode with fuel and memory caps, and exposes exhaustive vector queries rather than HNSW.

## Release and acceptance

Native version changes merged into `main` build and test wheels on Linux x86/ARM, macOS x86/ARM and Windows x86 before publishing. Companion version changes in `manifest.json` trigger build, security audit, source installation and native contract tests, then publish through the protected `pypi` environment. The native runtime release must complete before merging a companion version that requires it. Ordinary documentation edits do not publish.

Acceptance is a successful upload followed by clean PyPI installation and tests on Linux, macOS and Windows. The token stays in the GitHub environment; it is never included in artifacts. Trusted Publishing remains available when configured. See [security](SECURITY.md), [validation](VALIDATION.md), [architecture decision](../../docs/adr/ADR-python-top-ten-portfolio.md), and [changelog](CHANGELOG.md).
