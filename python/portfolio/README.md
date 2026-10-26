# RuVector Python portfolio

![RuVector Python capability tracks](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/overview.svg)

Python access to the ten most downloaded RuVector npm packages. Eight new companion distributions join the existing `ruvector` SDK and `ruvector-graph`. Previously released clustering and reranking remain available. Every companion has its own introduction, animated diagram, runnable getting started example and explicit capability limits.

## Release status, October 26, 2026

The delivery target is native **`ruvector==0.1.5`** plus all eleven companion
**`0.2.1`** distributions. These are **unpublished successor candidates**;
candidate wheels/sdists and tests do not establish public PyPI availability.
The [release disposition](../../docs/python-portfolio/resume-release-adr.md)
records the selected successor and the remaining review/publication prerequisites.

The original `0.2.0` release remains incomplete and security-blocked: public
registry readback found eight wheels, no sdists, and no rvf/rvf-wasm/sona releases.
Those original artifacts pin native `0.1.3`, whose frozen dependency audit now
rejects `tokio 1.52.3` under RUSTSEC-2026-0386. Missing original uploads are no
longer the delivery plan. Preserve public files and the immutable inventory;
do not rebuild old identities, replay the supervisor, or waive the frozen audit.

The [October 12 inventory](../../docs/python-portfolio/evidence/2026-10-12/README.md)
and [attention tutorial](tutorials/README.md) retain historical evidence for the
old release. The tutorial's pinned native `0.1.3` environment is not the current
successor recommendation. Package guide commands below may also describe the
historical release; consult the dated disposition before using registry versions.

Actual successor release requires independent review, explicit release approval,
native-first publication, exact public-artifact reconciliation and fresh consumer
tests. The current two main-triggered publishers do not enforce native-first
ordering, so an ordered landing plan or qualified dependency is still needed.
No publication is authorized by this README.

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

## Candidate qualification before publication

Use the approved native wheel and companion wheels from the same reviewed CI
candidate. Download only the native wheel matching your host platform, plus
the eleven pure-Python companion wheels. For example, after downloading those
artifacts locally:

```sh
python -m venv .venv
# Activate the environment using your platform's normal command.
python -m pip install /path/to/native/ruvector-0.1.5-*.whl /path/to/companions/*-0.2.1-*.whl
python -m pip check
```

These paths are placeholders for reviewed build artifacts, not public download
URLs. Do not request unpublished versions from PyPI or substitute older native
versions to make installation succeed. Public installation examples must be
verified against the actual authorized release after publication.

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

The base SDK guide is [here](../../crates/ruvector-py/README.md). The successor companions pin the same `ruvector==0.1.5` native runtime, avoiding duplicate Rust extensions; historical `0.2.0` artifacts retain their `ruvector==0.1.3` pin. RVF WASM additionally pins Wasmtime and bundles an integrity verified upstream module. It requires no Node, browser, WASI, network downloads or npm install at runtime.

## What is supported?

These are deliberately scoped Python APIs, not promises of complete npm parity. SONA has inference only and cannot train through this binding. Fresh GNN parameters are untrained. The ruvllm package implements full precision KV caching, not language generation. The RVF store opens trusted local files; it is not an untrusted file sandbox. The WASM package runs actual upstream bytecode with fuel and memory caps, and exposes exhaustive vector queries rather than HNSW.

## Release and acceptance

Native version changes merged into `main` build and test wheels on Linux x86/ARM, macOS x86/ARM and Windows x86 before publishing. Companion version changes in `manifest.json` trigger build, security audit, source installation and native contract tests, then publish through the protected `pypi` environment. The native runtime release must complete before merging a companion version that requires it. Ordinary documentation edits do not publish.

Acceptance is a successful upload followed by clean PyPI installation and tests on Linux, macOS and Windows. The token stays in the GitHub environment; it is never included in artifacts. Trusted Publishing remains available when configured. See [security](SECURITY.md), [validation](VALIDATION.md), [architecture decision](../../docs/adr/ADR-python-top-ten-portfolio.md), and [changelog](CHANGELOG.md).
