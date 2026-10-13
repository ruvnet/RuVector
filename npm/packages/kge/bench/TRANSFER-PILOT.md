# Exploratory KGE relation replay pilot, 2026-10-13

Source pretraining with mixed replay achieved mean filtered MRR **0.294088**, versus **0.272376** for equal exposure mixed training. All three paired seeds improved. This is a selected relation subset inside FB15k237, not cross-domain transfer or a release gate.

| Seed | Pretrain + replay | Cold, half exposure | Equal exposure | Warm minus equal |
| --- | ---: | ---: | ---: | ---: |
| 11 | 0.303951 | 0.148546 | 0.293179 | 0.010771 |
| 23 | 0.296900 | 0.276300 | 0.289516 | 0.007384 |
| 47 | 0.281414 | 0.118601 | 0.234432 | 0.046982 |

The mean paired gain is 0.021712 MRR (7.97% relative). Seed 47 supplies roughly 72% of aggregate gain. Three seeds do not establish robust significance or stable effect size.

## Protocol and controls

The pinned loader selects the top 500 entities by training degree. Relations ordered by training frequency are partitioned by alternating rank. Source and target each contain 1,024 deterministically selected training triples with disjoint relation sets and shared entities. There are 390 entities, 115 relations and 52 target relations.

Evaluation includes 159 target test triples from the admitted vocabulary. 324 of 483 subgraph test triples are excluded. This result does not describe full FB15k237 performance. All canonical truths in the admitted vocabulary are registered for filtered ranking after checkpoint freeze. Test triples are never trained.

Each seed (11, 23, 47) starts all arms from an identical initialized checkpoint. Target training facts are registered as validation during source training to freeze vocabulary, but source training consumes only Train tags. Inspection of the trainer found no validation-dependent early stopping or scheduling.

HolE uses 32 dimensions, batch size 128, learning rate 0.05 and N3 coefficient 0.001. Warm training uses 8 source epochs followed by 4 mixed epochs. Equal exposure uses two stages of 4 mixed epochs, matching optimizer resets. Cold uses 4 mixed epochs.

Warm and equal exposure each consume 16,384 positive examples and 128 batch updates. Their source/target composition differs: warm uses 12,288 source and 4,096 target examples; equal exposure uses 8,192 of each. Cold consumes 8,192 examples and 64 updates. The cold comparison cannot isolate transfer because its total budget is half as large.

All nine checkpoints are saved before evaluation. Evaluation uses filtered head and tail ranking with the same random tie seed across paired arms. Registration of evaluation facts asserts unchanged vocabulary and parameters. Reported training batch counts are per epoch; reported triple counts include registered validation entries and are not optimization exposure.

## Execution evidence and repair

The real native binding is npm `@ruvector/kge@0.2.0`, engine version 2.4.0. It was installed with lifecycle scripts disabled. npm integrity:
`sha512-WwfysfBa3/3HVmSLrwVvso+TuFp8rOubUJZIsoOUaWbLidzYvw35zuQthKhxSzH0wJE4CybJYVwOkCeyOZMCMA==`.

Linux x64 native binary SHA256:
`4a32175eab0256d0e48b2e619388fae21b3c25dfb0269c1988a74c48c0920463`.

Initial execution froze all checkpoints, then failed because the runner expected a flat MRR field. The binding returns `report.combined.mrr`. The repair changed extraction and added resume support. Evaluation resumed from hash-verified checkpoints without retraining or changing samples, parameters or the predefined metric. The receipt records both runner hashes. The retained manifest describes the original run; its binding path is redacted in the public receipt.

The executor verified successful completion after repair. Training took seconds on the existing CPU VM. No GPU or paid model API was used. Existing VM infrastructure cost was not available; no zero-total-cost claim is made.

## Reproduce

Install the exact published package into a separate prefix with scripts disabled, then supply its absolute `index.js` path:

```sh
KGE_BENCH_BINDING=/absolute/path/to/kge/index.js \
KGE_PILOT_OUT=/absolute/path/to/new-output \
timeout 60s node npm/packages/kge/bench/transfer-pilot.mjs
```

The runner refuses an existing manifest unless `KGE_PILOT_RESUME=1`. Resume verifies sample/config identity and every frozen checkpoint hash. It is intended to repair evaluation tooling, not tune against exposed test results. The shell timeout is required; the script itself does not enforce the declared runtime limit.

The loader downloads pinned files; data and model checkpoints are not committed. Dataset information: https://www.microsoft.com/en-gb/download/details.aspx?id=52312.

## Next gate

Preregister a fresh relation partition and untouched evaluation queries before further model selection. Reproduce the improvement at matched exposure, with more seeds and uncertainty estimates. Coding transfer remains a separate four-arm test requiring an approved model execution route. These exploratory results authorize neither package release nor general transfer claims.
