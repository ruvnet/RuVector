# KGE curriculum confirmation preregistration

Frozen before training and final evaluation on 2026-10-13. Extends the exploratory pilot in PR1211. No result is known at this commit.

## Question and decision

Does source pretraining with mixed replay beat mixed training at equal total positive-example exposure on a fresh selected query population inside FB15k237?

Primary metric: paired seed difference in filtered combined head/tail MRR, warm minus equal exposure. Accept only if the mean difference and the lower bound of a paired percentile bootstrap 95% interval both exceed zero. Use exactly 20,000 seed resamples, fixed LCG seed 20261013, sorted indices 499 and 19499. Seeds are exactly 101 through 110. Report all results, including a negative result. Do not choose another partition after seeing outcomes.

This interval measures training seed variability conditional on this partition and its selected queries. It is not a confidence interval over datasets or all graph queries.

## Fixed partition and exclusion

Use the existing pinned top500 training-degree subgraph. Order relations by SHA256 of `confirm-20261013-v1:` plus relation ID, alternating source and target. Select 1,024 source and 1,024 target facts by the original deterministic sampler. Target facts must share the source entity vocabulary.

Reconstruct all159 original pilot test triples and assert their frozen hash. Exclude those triples AND every new test fact sharing either a tail query (subject, relation) or head query (relation, object) with the old test population.

The resulting population is62 test facts,396 entities,112 relations,50 target relations.421 of483 subgraph test facts are excluded. The minimum was20, fixed before computing these counts; insufficient counts would abort, not trigger another salt.

## Controls and exposure

HolE,32 dimensions,batch128,lr0.05,N3 coefficient0.001. Each seed initializes all arms from the same serialized parameters and vocabulary. Source phase optimizes only source Train facts. Target Valid registrations freeze vocabulary and do not drive scheduling or early stopping.

Warm:8 source epochs,4 mixed epochs. Equal exposure:4 mixed epochs twice, matching optimizer resets. Cold:4 mixed epochs. Warm and equal each use16,384 positive examples and128 updates. Warm has12,288 source and4,096 target examples; equal has8,192 each. Therefore this is a comparison of curricula at equal total exposure, not an isolated pretraining effect. Cold has half the total budget.

Freeze all30 checkpoints before final evaluation. Verify checkpoint hashes, finite metrics, exact phase epochs and batch counts, train/test disjointness, and unchanged vocabulary/parameters after evaluation truth registration. Filter with all canonical known truths in the admitted vocabulary. Pair evaluation tie seeds across arms.

## Execution

Use the same pinned published native @ruvector/kge0.2.0, engine2.4.0, as the pilot. No provider API or GPU. Existing CPU infrastructure billing is unknown. A60 second external process timeout bounds each execution.

```sh
KGE_BENCH_BINDING=/absolute/path/to/kge/index.js \
KGE_PILOT_OUT=/absolute/path/to/new-confirmation \
KGE_CONFIRM_PHASE=freeze timeout 60s node npm/packages/kge/bench/transfer-confirmation.mjs
# Commit the frozen manifest and runner before continuing.
KGE_BENCH_BINDING=/absolute/path/to/kge/index.js \
KGE_PILOT_OUT=/absolute/path/to/new-confirmation \
KGE_CONFIRM_PHASE=run timeout 60s node npm/packages/kge/bench/transfer-confirmation.mjs
```

The runner rejects existing results or frozen checkpoints and changed sample/config/runner identities. Public manifest redacts only the local binding path. Original manifest hash7514a6c002cf4b53bf75d1d04e60a4869874bec9165aced09555a1823381bdea is over compact JSON; runner SHA256c831823f385a591bdac42345488a256fabe2f7dd21d2a813c675e893fd1e3dba.

A positive result permits only a conditional curriculum claim on this slice. No automatic merge, promotion or package publication.
