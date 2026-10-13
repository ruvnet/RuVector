# Confirmation result: NOT CONFIRMED

The preregistered fresh-query confirmation contradicts the pilot's positive direction.

| Arm | Mean filtered MRR |
| --- | ---: |
| Source pretraining plus replay | 0.174342 |
| Equal total exposure mixed training | 0.200159 |
| Cold mixed training, half total exposure | 0.161195 |

Warm minus equal exposure: **-0.025817 MRR**, or **-12.90% relative**. The fixed 20,000-resample paired seed bootstrap95% interval is **[-0.043433, -0.009094]**. Warm wins2 of10 seeds. The acceptance rule fails. On this selected partition, the measured curriculum is worse than equal total exposure mixed training.

## Evidence

Protocol and runner were committed before training at `b99552f583cd528ad9870769026dea06236411b4`. The exact remote commit was fetched, its runner hash verified, and its checkout was clean. All30 checkpoints froze before final evaluation. All invariants passed, including batch counts, finite metrics, prior query exclusion, vocabulary/parameter identity and train/test disjointness.

Execution used the published native @ruvector/kge0.2.0 engine2.4.0, existing CPU only, and took14.0649503 seconds. The trusted executor returned completionVersion1, completionVerifiedtrue,statusok,exitCode0. Provider and GPU usage were zero; existing VM billing remains unknown. An independent reviewer reproduced the mean and bootstrap interval exactly.

Public receipt includes every paired score, checkpoint and initial-state hashes, phase measurements, source commit, execution duration and raw-file hashes. Raw checkpoints remain in the scoped execution workspace.

## Limits and decision

This is62 selected test triples from one fresh relation partition, with421 of483 subgraph test triples excluded. The confidence interval is across training seeds, not datasets or independently sampled queries. Warm and equal have matched total positive-example/update budgets but different source/target proportions. This is a curriculum comparison within one dataset, not isolated pretraining or cross-domain transfer.

Preserve the original three-seed pilot as exploratory history. Do not present its7.97% relative gain as a validated improvement. Do not search more partitions after this outcome or tune against either exposed test set. No promotion, release or merge follows from these measurements.

Next useful work is strict benchmark evidence validation and a separately preregistered coding transfer experiment after its model and learned artifact requirements are established.

Acceptance replay: calculate the paired mean and fixed bootstrap from the committed receipt. It must reproduce the negative interval and `NOT_CONFIRMED` decision without retraining.
