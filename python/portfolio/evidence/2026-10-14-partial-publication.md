# Companion 0.2.0 partial-publication receipt — 2026-10-14

Checked at 06:37 UTC. Owner: RuVector Python Factory /root. This is evidence, not a release authorization or a claim of complete publication.

## Release reconciliation

[Recovery PR #1205](https://github.com/ruvnet/RuVector/pull/1205) merged on October 13 at 13:04:05 UTC as `25ed48e9e31823dd8bf56533881ecebc56d131ba`, after its exact-head gates passed. [Recovery run 38402061555](https://github.com/ruvnet/RuVector/actions/runs/38402061555) passed its contract, frozen native audit, and Linux/macOS/Windows pre-upload consumers. Publication then stopped at 13:15:58 UTC with HTTP 429, “Too many new projects created,” while uploading ruvector-rvf. The supervisor stopped without retry. The post-publication matrix was skipped; it did not pass.

Fresh unauthenticated PyPI JSON queries found eight matching, non-yanked files out of 22 expected. Filenames, SHA-256 and byte sizes match the byte-pinned original inventory. The new consumer's pip installation report independently matches those eight wheel hashes.

| Companion version 0.2.0 | Wheel | Source distribution |
| --- | --- | --- |
| attention, cluster, core, gnn, graph, rerank, router, ruvllm | Published, hash verified | Missing |
| rvf, rvf-wasm, sona | Missing | Missing |

All names above use the `ruvector-` prefix. Fourteen files remain: three wheels and eleven sdists. Each original companion pins `ruvector==0.1.3`; rebuilding current source is not an acceptable substitute.

Frozen source: `fccf36e9a76d35bce6750dfd3b25802cae3d683b`.
Original artifact: 11713787141, run 38169271248.
Inventory SHA-256: `ca4c4b47df73a26619c048ed1eb02013bfd96484aeb88e20fe16464ef1e0a32d`.

## Fresh consumer acceptance

A new isolated environment installed only from https://pypi.org/simple, without a wheel cache or source-checkout imports. CPython 3.12.3, Linux x86_64, glibc 2.39; NumPy 2.2.6, pytest 9.1.1, pip 26.2.1, pip-audit 2.10.1.

- Eight published companion wheels installed at 0.2.0 with native ruvector 0.1.3; pip check passed.
- 38 tests passed from six unchanged test modules fetched from the frozen source.
- All eight corresponding README Python quickstarts passed.
- The exact [existing gist](https://gist.github.com/ruvnet/2491c7631bf3e186e815196f52fd8954) at revision `7c09fe56274683c0b838ed0a2b176abdd86ff079` passed, including an independent float64 reference and invalid-input rejection. Its script SHA-256 is `0d9961c2c93a1cec8debcd744aeacc141c662042042e7646e1f8a1a14a61ad0f`.
- pip-audit queried 43 installed dependencies and reported no known vulnerabilities. No ignored advisory IDs or unpublished-package skips were used. This is a timestamped advisory result, not proof that vulnerabilities cannot exist.

The new-run receipt and report digests are in the adjacent JSON file. The initial system `venv` attempt lacked ensurepip; the existing uv executable created the isolated environment instead. No system package or permission change was needed.

Reproduction after obtaining the frozen source:

```bash
uv venv --python 3.12 --seed consumer
consumer/bin/python -m pip install --index-url https://pypi.org/simple --only-binary=:all: --no-cache-dir \
  pip==26.2.1 numpy==2.2.6 pytest==9.1.1 pip-audit==2.10.1 ruvector==0.1.3 \
  ruvector-attention==0.2.0 ruvector-cluster==0.2.0 ruvector-core==0.2.0 \
  ruvector-gnn==0.2.0 ruvector-graph==0.2.0 ruvector-rerank==0.2.0 \
  ruvector-router==0.2.0 ruvector-ruvllm==0.2.0
consumer/bin/python -m pip check
consumer/bin/python -m pytest --import-mode=importlib \
  python/portfolio/tests/test_contracts.py \
  python/portfolio/tests/test_router_portfolio.py \
  python/portfolio/tests/test_ruvllm_portfolio.py \
  python/portfolio/tests/test_cluster_example.py \
  python/portfolio/tests/test_graph_example.py \
  python/portfolio/tests/test_rerank_example.py -q
consumer/bin/python -m pip_audit --progress-spinner off
```

Execute each Python fenced block in the eight package READMEs separately, and the gist's exact `attention_quickstart.py`, using that environment outside the source tree. The full eleven-package example runner is intentionally not used for a partial release.

## Limits and next acceptance

This new post-publication run covers Linux only. RVF, RVF-WASM and SONA are not installed. Their tests and the mixed `test_top_native.py` module were not collected; core and GNN have README smoke coverage here, not a claim of repeating their full adversarial suites. Earlier pre-upload CI is separate evidence and does not replace the missing post-publication matrix. No performance, inference-quality or full npm-parity claim is made.

No publisher is active, and no upload or workflow retry was performed on October 14. The error is a project-creation throttle, not a failed test or invalid token. The [Trusted Publisher registration documentation](https://docs.pypi.org/trusted-publishers/troubleshooting/#ratelimiting) describes a different limit; its 24-hour guidance must not be asserted as this account's project-creation reset. No applicable fixed cooldown was verified. Preserve any actual Retry-After/RateLimit response evidence in a future authorized attempt; do not use alternate accounts, credentials or networks to evade the limit.

Next release owner must re-read current claims, actual registry hashes, stopped receipts, release permissions and gates, then use the existing protected recovery workflow for only missing original files when eligible. Stop again on any failure. Completion requires all 22 files plus successful fresh PyPI consumers on Linux, macOS and Windows.

No new library was selected: delivery of this existing portfolio remains incomplete. Native SDK and KGE lanes, and other open Python PRs, remain with their existing owners. Rollback of this evidence-only change is a normal reviewed revert; do not delete, overwrite or rebuild already-published files.
