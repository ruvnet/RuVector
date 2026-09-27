# @ruvector/kge — benchmarks and release gates

The measurement harness (ADR-003 / ADR-006): it trains the binding, evaluates
filtered ranking metrics, and gates a release on numbers, not on "tests pass".
See the package [README](../README.md) for the API and the FFT spike.

## Honest numbers, by construction

- **Filtered MRR / Hits@{1,3,10} with RANDOM tie-breaking** (ADR-003 §5; Sun et
  al. 2020, arXiv:1911.03903). Many older papers used TOP tie-breaking, which
  inflates scores — worst on WN18RR's symmetric relations. This harness asserts
  RANDOM in CI: an all-tied scorer must yield mean rank ≈ (|E|+1)/2, not 1.
  Because of this our numbers are directly comparable to LibKGE/PyKEEN and
  *lower* than TOP-inflated papers. That is correct.
- **Baselines are reproduced, then a margin is subtracted** — never invented.
  FB15k-237 and WN18RR gate on the LibKGE ComplEx MRR (0.348 / 0.475) minus 3
  points. CoDEx-M's baseline is read from arXiv:2009.07810 at build; its gate
  SKIPs until a maintainer records it with a source. The predict-latency gate is
  wired to `fft-spike-2026-09-21.json` (ADR-002): its native p95 threshold is
  live; the wasm threshold stays pending (wasm32 is compile-only in the spike).
- **HolE gate number = ComplEx's**, valid only because HolE ≡ ComplEx is proved
  by the crate unit test (ADR-002). That equivalence is a precondition, surfaced
  here as an external gate row.

## Running

```bash
node run.mjs --suite synthetic --tie-check --ann --adversarial --gate --report-only
node run.mjs --suite fb15k237 --scorer hole --dims 256 --epochs 100 --gate
node run.mjs --suite wn18rr --limit 5000 --ann --gate --report-only
```

Flags: `--suite synthetic|fb15k237|wn18rr|codexm|yago310|all`, `--scorer
hole|rotate`, `--dims`, `--epochs`, `--limit N`, `--ann`, `--adversarial`,
`--tie-check`, `--gate`, `--report-only`, `--baseline-receipt PATH`,
`--train-config JSON`, `--out`, `--help`.

The harness never crashes on an unavailable engine: each measured arm reports
"engine unavailable". Under `--gate` that is a FAIL (exit 1), as is a run where
every gate SKIPs; `--report-only` records the same FAIL verdict but exits 0. Every
call into the Model binding goes through `lib/arms.mjs`. For a real synthetic
run before `train`/`eval` land in the core, point the harness at the fake:

```bash
KGE_BENCH_BINDING=./test/fixtures/fake-binding.cjs \
  node run.mjs --suite synthetic --tie-check --ann --adversarial --gate --report-only
```

## Datasets (fetched at bench time, nothing redistributed)

| Suite | Source | Licence stance | `--limit` |
|---|---|---|---|
| `synthetic` | frozen `fixtures/synthetic-kg.json` (committed) | n/a | n/a |
| `fb15k237` | DeepGraphLearning/KnowledgeGraphEmbedding@2e440e0f `data/FB15k-237` (byte-identical to the ConvE tarball) | Freebase-derived; follows source | subgraph |
| `wn18rr` | DeepGraphLearning/KnowledgeGraphEmbedding@2e440e0f `data/wn18rr` (ConvE ids, 40,943 entities) | WordNet-derived; follows source | subgraph |
| `codexm` | tsafavi/codex `data/triples/codex-m` + hard negatives | code MIT (root LICENSE, hashed); triples CC BY 4.0 per the paper | subgraph |
| `yago310` | DeepGraphLearning/KnowledgeGraphEmbedding `data/YAGO3-10` (byte-identical to the ConvE tarball; lazy: only with `--suite yago310`) | YAGO-derived; follows source | subgraph |

`--limit N` keeps the top-N entities by train degree (tie-broken by id) and every
triple across all splits whose head and tail both survive — a connected subgraph,
so filtered ranking stays meaningful (a random-triple slice would orphan the
filter sets). Every fetched file is sha256-pinned; a file without a pin, or
whose bytes drift from it, fails closed (the suite is skipped as unavailable).

## Gates (ADR-006 §Release gates)

`gates.json` — each threshold carries a `source`. Numeric gates apply only on
their suite / when their flag is present; otherwise SKIP. A requested
`--tie-check` that cannot produce an all-tied model FAILs; the check zeroes the
tables through `toJson`/`fromJson`, so it runs on the real binding.
The synthetic suite exercises the tie-break, ANN-recall, adversarial and (native)
latency gates so CI measures something on every PR.

## Receipt

One `ruvector-kge-bench/receipt@1` per run: scorer, config, dataset/split
hashes, metrics per split, tie-break mode, ANN recall, the adversarial report,
predict latency, train throughput (`triple_epochs_per_sec`), host, the
binding's `statsJson`, and `provenance` (git SHA, engine-source sha256, config
sha256, dataset sha256, binding and native-binary sha256). `receipt_sha256`
seals the canonical body (`verifyReceipt` in `lib/receipt.mjs`). It records **no entity or relation ids** and **no triple text** (ADR-005) — hashes
and counts only.
