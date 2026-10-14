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
`--tie-check`, `--gate`, `--report-only`, `--baseline-receipt PATH`, `--out`.

The harness never crashes on an unavailable engine: each measured arm reports
"engine unavailable" and, under `--report-only`, the run still exits 0. Every
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
| `fb15k237` | villmow/datasets_knowledge_embedding (standard splits) | Freebase-derived; follows source | subgraph |
| `wn18rr` | same mirror (`WN18RR/text`) | WordNet-derived; follows source | subgraph |
| `codexm` | tsafavi/codex `data/triples/codex-m` + hard negatives | code MIT (root LICENSE, hashed); triples CC BY 4.0 per the paper | subgraph |
| `yago310` | same mirror (lazy: only with `--suite yago310`) | YAGO-derived; follows source | subgraph |

`--limit N` keeps the top-N entities by train degree (tie-broken by id) and every
triple across all splits whose head and tail both survive — a connected subgraph,
so filtered ranking stays meaningful (a random-triple slice would orphan the
filter sets). Fetched files are sha256-pinned after the first download; the
fetcher prints the hash to record.

## Gates (ADR-006 §Release gates)

`gates.json` — each threshold carries a `source`. Numeric gates apply only on
their suite / when their flag is present; otherwise SKIP (never a silent pass).
The synthetic suite exercises the tie-break, ANN-recall, adversarial and (native)
latency gates so CI measures something on every PR.

## Receipt

One `ruvector-kge-bench/receipt@1` per run: scorer, config, dataset/split
hashes, metrics per split, tie-break mode, ANN recall, the adversarial report,
predict latency, train throughput, host, and the binding's `statsJson`. It
records **no entity or relation ids** and **no triple text** (ADR-005) — hashes
and counts only.

## Experimental cross-graph ULTRA adapter

`transfer/ultra_adapter.py` accepts RuVector `{s,r,o,split}` JSON exports and
provides exhaustive `predict({s,r,k})` / `predict({o,r,k})` style results through
an optional Python sidecar. It uses the original MIT ULTRA implementation and
the hash-pinned `ultra_3g.pth` checkpoint. It does not add a native Rust scorer,
change existing KGE behavior, or install anything automatically.

The first CPU pilot uses Nations, Kinships and UMLS from a recorded PyKEEN
revision. None is among the checkpoint's upstream-declared pretraining graphs
(FB15k237, WN18RR, CoDExMedium). This is evidence against that declared graph
list, not an independent audit of the original training process. Unlike the
legacy `carveTransfer` split, these are different whole graphs. Target support
edges are still available at inference: zero-shot means zero target updates,
not reasoning without observations.

The first 64 original test triples by SHA256 of JSON `[s,r,o]` are selected
without model scores. Other test facts stay filter-only. No valid or test
edges enter entity or relation message passing. All entities are candidates;
head and tail ranks use pessimistic ties. Random weights use seeds 101/202/303.
The native KGE control trains HolE for 50 epochs on target support only with
the same candidate universe, filters and queries. This is not an equal-total-
compute comparison: ULTRA brings prior external training. Source replay is
not measured here; earlier same-graph replay results cannot be substituted.

Reproduce from the repository root in an isolated Python environment with a
C++ compiler, using `transfer/requirements-cpu.txt` for the measured runtime:

```bash
python -m venv /tmp/ultra-env
source /tmp/ultra-env/bin/activate
pip install -r npm/packages/kge/bench/transfer/requirements-cpu.txt
git clone https://github.com/DeepGraphLearning/ULTRA.git /tmp/ultra
git -C /tmp/ultra checkout 427966ad8ed60420eef034063d44f3153addff90
git clone https://github.com/pykeen/pykeen.git /tmp/pykeen
git -C /tmp/pykeen checkout cbc1ea107afd7bebc531c1739c210c620f908fbf
python npm/packages/kge/bench/transfer/prepare_pykeen.py \
  --checkout /tmp/pykeen --dataset nations --output /tmp/nations.json
python npm/packages/kge/bench/transfer/ultra_adapter.py \
  --input /tmp/nations.json --upstream /tmp/ultra --output /tmp/ultra-result.json
node npm/packages/kge/bench/transfer/kge_control.mjs /tmp/nations.json \
  /absolute/path/to/installed/@ruvector/kge/index.js /tmp/kge-result.json
ULTRA_CHECKOUT=/tmp/ultra python -m unittest discover \
  -s npm/packages/kge/bench/transfer -p 'test_*.py'
```

Activate the environment so `ninja` is on PATH. The upstream CPU rspmm kernel
compiles on first inference. Setup/compilation time is excluded from recorded
evaluation time. Without `ULTRA_CHECKOUT`, the optional integration tests are explicitly skipped.

Pass `--query '{"s":"entity","r":"relation","k":5}'` to produce a ranked
prediction instead of a benchmark. Relation names must occur in target support;
they need not occur in checkpoint pretraining. Raw ULTRA and KGE scores are not
calibrated for combination. The adapter is sequential and experimental, not an
unrestricted inference service. All local model code, compiler and installed
dependencies must be trusted. Checkpoint bytes are SHA256-verified and loaded
with `weights_only=True`; no secrets or provider API are required.

Receipts in `transfer/pilot-results.json` preserve input/file/source hashes,
dependency versions, query hashes, paired head/tail ranks and controls. Positive
pretrained-versus-random results establish only a bounded exploratory transfer
signal. Promotion remains false. Acceptance for deployment requires separate
untouched graphs, target scratch and replay controls where meaningful,
equal target adaptation budgets, uncertainty estimates and a latency budget.

Measured pilot (2026-10-13), filtered MRR; controls averaged over three seeds:

| Graph | ULTRA zero-shot | Random ULTRA | Target-trained KGE |
|---|---:|---:|---:|
| Nations | 0.5753 | 0.2701 | 0.7396 |
| Kinships | 0.1858 | 0.0582 | 0.8128 |
| UMLS | 0.6767 | 0.0415 | 0.9276 |

ULTRA beats random weights on all three selected query sets and loses to
target-trained KGE on all three. Keep KGE as the incumbent; treat the structural
adapter as an experimental cold-start option. ULTRA evaluated 128 rankings per
graph in 1.92–2.28 seconds on this CPU environment, excluding initialization and
kernel compilation. KGE's 50 target epochs took 0.21–2.44 seconds per seed.
These are descriptive run timings, not a controlled latency benchmark.


## Frozen sparse target adaptation experiment

`transfer/sparse_transfer.py` compares frozen pretrained ULTRA, adapted
pretrained ULTRA, scratch ULTRA, native HolE with 50 epochs, and native HolE
with training duration calibrated on a discarded model. The protocol and
runner hashes were committed before execution in
`b1b445e3b4f95cfc93253f93c4181cb688da5aa4`. The original manifest is retained in
`transfer/sparse-manifest.json`.

Nested target support fractions are 1%, 5%, 10% and 25%, with a minimum of one
fact per relation. Nations' nominal 1% therefore uses 3.45%; all actual counts
and fractions are reported. Seeds are 101, 202 and 303. ULTRA uses AdamW,
learning rate 0.0005, batch size 8, 16 negatives and one training thread.
Training stops after crossing two seconds, checked between updates, or 1000
updates. The native timed arm estimates its epoch count from support-only
calibration. Actual durations determine whether the comparison meets the
10% timing tolerance. Fixed 50 epoch KGE is an operational reference and is
excluded from that timing gate. Setup and calibration are reported separately;
this is not equal FLOPs, equal examples, or equal total compute.

Only retained support facts enter training or message passing. Withheld
training facts become filter-only truths. Each ULTRA batch removes its
positive edges and their inverses before rebuilding the relation graph.
Negative sampling uses support only. All arms share the full candidate
vocabulary, including nodes with no retained support. Evaluation uses both
head and tail queries with exhaustive filtered pessimistic ranks.

Nations, Kinships and UMLS reuse the diagnostic pilot's 64 queries per graph.
Countries S1 adds an untouched whole-graph check using all 24 official test
facts. Countries source files are pinned by commit and hash; preparation
removes one exact within-training duplicate and rejects cross-split overlap.
The confirmation set is too small to establish broad transfer superiority.

Acceptance requires at least 10% relative aggregate MRR improvement over
scratch ULTRA and each KGE control, positive improvement over scratch at all
four support fractions, and every timed arm within 10% of two seconds.
Aggregate MRR weights seeds and fractions equally. No production promotion is
performed, regardless of this exploratory gate.

To reproduce, prepare the three PyKEEN datasets as above, and download
`train.txt`, `valid.txt` and `test.txt` from the pinned `BASE_URL` in
`transfer/prepare_countries.py`. Then:

```bash
python npm/packages/kge/bench/transfer/prepare_countries.py \
  --directory /tmp/countries-raw --output /tmp/countries_s1.json
python npm/packages/kge/bench/transfer/sparse_transfer.py --freeze \
  --inputs /tmp/nations.json /tmp/kinships.json /tmp/umls.json /tmp/countries_s1.json \
  --binding /absolute/path/to/installed/@ruvector/kge/index.js \
  --native /absolute/path/to/installed/@ruvector/kge/native/kge.linux-x64-gnu.node \
  --manifest /tmp/sparse-manifest.json
python npm/packages/kge/bench/transfer/sparse_transfer.py --run \
  --inputs /tmp/nations.json /tmp/kinships.json /tmp/umls.json /tmp/countries_s1.json \
  --binding /absolute/path/to/installed/@ruvector/kge/index.js \
  --native /absolute/path/to/installed/@ruvector/kge/native/kge.linux-x64-gnu.node \
  --manifest /tmp/sparse-manifest.json --upstream /tmp/ultra --output /tmp/sparse-run
python npm/packages/kge/bench/transfer/summarize_sparse.py \
  --receipt /tmp/sparse-run/receipt.json --output /tmp/sparse-summary.json
```

Use a new manifest when local absolute paths differ. Compare all input,
source and native binary hashes against the committed manifest before claiming
a reproduction. Wall-clock stopping can produce different update counts on
another run. The saved summary includes support coverage, timing mismatches,
per-fraction metrics and acceptance decisions. The compressed full receipt
preserves all 208 arm records and ULTRA paired ranks; checkpoint hashes are
recorded, while generated model weights are not committed.

Measured sparse pilot (2026-10-13), MRR averaged equally across four support
fractions and three training seeds. Frozen ULTRA has one result per fraction.

| Graph | Frozen ULTRA | Adapted ULTRA | Scratch ULTRA | KGE 50 epochs | KGE calibrated time |
|---|---:|---:|---:|---:|---:|
| nations | 0.4968 | 0.5223 | 0.5190 | 0.4087 | 0.4099 |
| kinships | 0.0638 | 0.0664 | 0.0538 | 0.1394 | 0.1419 |
| umls | 0.2921 | 0.3013 | 0.1760 | 0.1539 | 0.1547 |
| countries_s1 | 0.2677 | 0.2582 | 0.1677 | 0.0245 | 0.0300 |

No graph passes the frozen acceptance gate. Adapted ULTRA improves aggregate
MRR over scratch by 0.6% on Nations, 23.5% on Kinships, 71.2% on UMLS and 54.0%
on Countries, but every graph contains at least one support fraction where
scratch wins. KGE also remains stronger on Kinships. These aggregate gains do
not imply a uniform benefit from pretraining.

On the untouched Countries check, adapted ULTRA loses to scratch at 5%
support (0.1892 versus 0.2010). At 25%, adaptation improves over scratch
(0.5985 versus 0.2332), but frozen ULTRA is stronger still (0.6641).
Across all four fractions, frozen ULTRA averages 0.2677 versus adapted 0.2582.
The current adaptation recipe therefore does not demonstrate a benefit over
simply retaining the pretrained model on this holdout.

Timing also fails: 59 of 144 timed records fall outside 1.8 to 2.2 seconds.
ULTRA update granularity causes overruns, while native calibration sometimes
underuses the budget. This run cannot support an equal target time claim.
At the smallest Countries support level, none of the 24 test facts has both
endpoints observed in support; low coverage is part of this experiment, not
proof that every cold-start setting is handled.

`transfer/sparse-results.json` contains compact results and provenance;
`transfer/sparse-receipt.json.gz` contains the complete receipt. All 96 saved
ULTRA checkpoint hashes and recorded ULTRA MRR values were independently
recomputed from local checkpoint bytes and paired ranks. Countries preparation
was reproduced byte for byte. The next experiment should first validate a
longer timing budget on development graphs and compare frozen versus adapted
ULTRA explicitly. Any new transfer confirmation requires a fresh held-out
graph, since Countries has now been inspected. KGE remains the incumbent.


## Longer adaptation development gate

`transfer/adaptation_budget.py` tests whether increasing the target training
budget from two to twenty seconds is worth a new confirmation experiment.
This is development evidence only: Nations, Kinships and UMLS, 25% support,
64 previously inspected test facts per graph and seeds 101/202/303. The
selected support fraction must not be interpreted as representative of all
support levels. No new graph is loaded by this runner.

Frozen pretrained ULTRA is now an explicit comparator alongside scratch
ULTRA. The gate requires at least 10% relative mean MRR improvement over both
comparators on every development graph and every trained arm to finish in
18 to 22 seconds. All other training settings and the support sampler reuse
the earlier hash-pinned implementation. The output is 21 records, with MRR
recomputed from 128 head/tail ranks per record. A passing development gate
would justify freezing a separate protocol for an untouched confirmation
graph; it never promotes a model or establishes superiority over native KGE.

The manifest was written before execution. Its source and input hashes are
committed in `transfer/adaptation-manifest.json`; the development protocol
commit is `74ca7db0b614f5971b1133670ded26b5e54de641`.

```bash
python npm/packages/kge/bench/transfer/adaptation_budget.py --freeze \
  --inputs /tmp/nations.json /tmp/kinships.json /tmp/umls.json \
  --manifest /tmp/adaptation-manifest.json
python npm/packages/kge/bench/transfer/adaptation_budget.py --run \
  --inputs /tmp/nations.json /tmp/kinships.json /tmp/umls.json \
  --manifest /tmp/adaptation-manifest.json --upstream /tmp/ultra \
  --output /tmp/adaptation-run
```

Use the isolated environment described above with `ninja` on PATH. The
training ceiling is checked between updates, so actual duration and completed
update counts must accompany every result. Fresh output directories prevent
accidental overwrites. A failed development gate preserves the fresh holdout
and records the failure rather than trying new settings against that holdout.

Measured longer-budget development results (2026-10-13), MRR at 25% support:

| Graph | Frozen ULTRA | Adapted ULTRA | Scratch ULTRA | Adapted gain over frozen |
|---|---:|---:|---:|---:|
| nations | 0.5803 | 0.6293 | 0.5295 | 8.4% |
| kinships | 0.0940 | 0.1310 | 0.0746 | 39.4% |
| umls | 0.5067 | 0.6577 | 0.3764 | 29.8% |

All 18 training durations meet the frozen tolerance, ranging from
20.05 to 21.45 seconds, with 369.4 total training seconds.
The larger budget resolves the timing problem for these ULTRA arms; it says
nothing about native KGE calibration. There are 14 to 35 completed updates per arm.
Kinships and UMLS pass their development quality gates. Nations improves
8.4% over frozen, below the required 10%, so the overall gate fails.
No fresh confirmation graph was accessed and promotion remains false.

`transfer/adaptation-results.json` retains all 21 records, paired ranks,
source/input/checkpoint hashes and the gate decision. All 18 checkpoint hashes
were verified against saved weights. The useful finding is that more target
training helps adaptation on these development graphs. It does not establish
uniform gains, performance on a new graph, or superiority to native KGE.
Retain frozen ULTRA as the experimental baseline and leave adaptation opt-in
pending validation on the intended application graph. KGE remains incumbent.


## Validation-only adaptation selection

`validation_gate.py` and `guarded_adaptation.py` make adaptation conditional
on explicit validation evidence. The incumbent stays frozen unless candidate
MRR is at least 10% higher and the paired bootstrap lower 2.5 percentile of
MRR difference is positive. Resampling keeps head/tail ranks together per
fact, uses 2,000 replicates and a fixed seed. At least 32 facts are required;
the pilot selects 64 by query hash. This is a descriptive stability check,
not proof of independent graph generalization or a multiple-comparison correction.

The pilot reads pinned official PyKEEN `valid.txt` files. The earlier export's
`valid` tag also contains unused test facts, so it must not be treated as a
selection split. Selection filters contain retained support and official
validation truths only. Test facts, including unused test facts, never enter
these filters. The full candidate vocabulary remains fixed across arms.
Diagnostic test MRR retains the earlier full filtering policy, so validation
and diagnostic test MRR values should not be compared as if their filters
were identical.

The local checkpoint loader limits input to 64 MiB, checks the expected
SHA256 before restricted `weights_only=True` deserialization, requires finite
tensors, and validates keys, shapes and dtypes before changing weights.
The hash is an integrity check, not authentication. Use trusted local receipts
and weights only. This does not create an upload endpoint or inference service.
The sequential `apply_candidate` helper restores incumbent weights on any
rejection or exception and leaves accepted weights ready for `predict`.

```python
from guarded_adaptation import apply_candidate
from validation_gate import official_validation

support = [r for r in payload["triples"] if r["split"] == "train"]
queries, truths = official_validation(valid_bytes, expected_valid_sha256, support)
report = apply_candidate(adapter, candidate_path, expected_checkpoint_sha256,
                         queries, truths)
result = adapter.predict({"s": known_entity, "r": known_relation, "k": 5})
```

Here `adapter` starts with frozen pretrained weights. The validation file and
checkpoint hashes must come from trusted provenance. Do not supply test rows
to this API. The helper uses the existing model and adds no training run.

Reproduce the bounded development selector using saved checkpoints from the
longer adaptation run:

```bash
python npm/packages/kge/bench/transfer/evaluate_validation_gate.py --freeze \
  --run /tmp/adaptation-run --pykeen /tmp/pykeen --manifest /tmp/validation-manifest.json
python npm/packages/kge/bench/transfer/evaluate_validation_gate.py \
  --run /tmp/adaptation-run --pykeen /tmp/pykeen --manifest /tmp/validation-manifest.json \
  --upstream /tmp/ultra --output /tmp/validation-results.json
```

Measured selection (2026-10-13): one of nine adapted candidates was accepted,
Nations seed 202, with 13.26% validation MRR gain and a narrow bootstrap lower
bound of 0.000770. The other eight candidates retained frozen weights.
Selected diagnostic MRR averaged over seeds is 0.6061 for Nations versus
0.5803 frozen, 0.0940 for Kinships versus 0.0940 frozen, and 0.5067 for UMLS
versus 0.5067 frozen. Always adapting scored higher on these already inspected
test subsets, so this result does not establish selector superiority. It
shows that the earlier test gains are not consistently supported by this
separate validation check.

`transfer/validation-manifest.json` preserves the policy and source/input
hashes; `transfer/validation-results.json` contains all nine decisions,
validation ranks and diagnostic summaries. The manifest was written before
selection evaluation; implementation commit is
`71a7525be0edbc54d6b220baa526f8668b1358e9`. The policy was designed after earlier
test inspection, so all findings remain development evidence. No fresh
confirmation graph was accessed, no candidate was promoted and no KGE
superiority is claimed. Production use still requires an application graph,
separate validation and confirmation data, and measured serving cost.

### CPU software localization development pilot

`localization/` compares BM25 with frozen Salesforce/SweRankEmbed-Small on
six deterministic SWE-bench Lite cases: three each from Requests and Flask,
selected by the SHA256 of instance ID before retrieval. This is an issue to
file retrieval smoke experiment. It adds no reranker, graph scorer or training.
It does not establish unseen repository transfer or reproduce paper results.

The candidate corpus includes every production Python file under `requests/`
or `src/flask/` in each issue's base commit. Repository archives are read as
text, never extracted or executed. Chunks contain the first 80 module lines
and AST functions. Historical Python 2 syntax uses 80 line windows instead.
File coverage is complete for these cases, but nonfunction content beyond
line 80 can be absent from AST chunks. Gold production files come from the
benchmark patch and remain separate from encoder inputs. Predictions are
saved before scoring. Tests and patch text are not retrieval inputs.

SWERank uses the documented query prefix, normalized CLS embeddings and
maximum chunk score per file. The CPU variant uses float32, two Torch threads,
batch size eight and 512 tokens. BM25 uses the same chunks with k1=1.5 and
b=0.75, without token truncation. This difference in effective input is a
limitation. Record indexing cost separately from warm query encoding and
ranking. File ranking uses exact dot products in this small baseline.

Pinned sources:

* Model: `Salesforce/SweRankEmbed-Small`, revision
  `745d2a06103a66d3cfa600aa52fc0d3523010daa`.
* Model implementation: `Snowflake/snowflake-arctic-embed-m-long`, revision
  `92d97331f1f4b6a366c1f161354b9f3390cc219f`.
* Dataset: `princeton-nlp/SWE-bench_Lite`, revision
  `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`, test parquet.

Artifacts were downloaded using Hugging Face CLI with explicit revisions.
`pilot-manifest.json` records artifact, source, archive and input hashes and
was frozen before scoring. A hash checks integrity, not download origin.
The pinned local implementation is imported only after hash verification;
weights use safetensors. The model card declares CC BY NC 4.0. This is a
research comparator, not a commercial deployment authorization. Model weights
and target repository code are not vendored in this branch.

Reproduction requires Python with the six dependency versions in the manifest,
plus NumPy 1.26.4. Download the pinned model snapshot, the pinned implementation's
`configuration_hf_nomic_bert.py` and `modeling_hf_nomic_bert.py`, and the dataset
parquet. Generate a new manifest for local paths, then compare its content
hashes and pinned versions with the published manifest before running:

```bash
python npm/packages/kge/bench/localization/prepare.py \
  --parquet /tmp/swe-lite/data/test-00000-of-00001.parquet --output /tmp/localization-data
python npm/packages/kge/bench/localization/run.py --freeze \
  --data /tmp/localization-data --model /tmp/swerank-model --code /tmp/swerank-code \
  --manifest /tmp/localization-manifest.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python npm/packages/kge/bench/localization/run.py \
  --data /tmp/localization-data --model /tmp/swerank-model --code /tmp/swerank-code \
  --manifest /tmp/localization-manifest.json --output /tmp/localization-run
python -m unittest discover -s npm/packages/kge/bench/localization -p 'test_*.py'
```

Public benchmark pretraining overlap and issue edit chronology have not been
independently established. These six development cases cannot support a
production promotion or SOTA claim. A later RuVector graph comparison must use
separate development and confirmation repositories, identical candidates,
measured serving cost and a frozen acceptance threshold.

The lexical control sums a set of query tokens. Across Python hash seeds,
roundoff can change raw BM25 scores at floating point precision. Independent
recomputation of the first case preserved every file rank (maximum score
difference 7.11e-15); bitwise score reproducibility is not claimed.

Verify a completed run without encoding again. Preserve the Hugging Face CLI
`.cache/huggingface/download` metadata in each artifact directory:

```bash
python npm/packages/kge/bench/localization/verify.py \
  --run /tmp/localization-run --data /tmp/localization-data \
  --manifest /tmp/localization-manifest.json --model /tmp/swerank-model \
  --code /tmp/swerank-code --dataset /tmp/swe-lite \
  --source-commit 5957f51018cbb5d58e5f0af527e4b52342814860 \
  --output /tmp/localization-verified.json
```

The verifier checks complete predictions, rank arithmetic, all aggregate
metrics, source and input hashes, and download metadata checksums against
local artifact bytes. It embeds file rankings and gold paths for further
independent inspection. Metadata checks confirm local consistency; they do
not authenticate origin. The supplied source commit is declared provenance,
not remotely verified by this command. Run with Python assertions enabled.

Measured development results (2026-10-13), all six preselected cases:

| Method | MRR | Hit@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.8056 | 4/6 | 6/6 | 6/6 |
| SWERank CPU variant | 0.5988 | 3/6 | 5/6 | 6/6 |

Gold file ranks in selection order were BM25 `[1,1,2,1,1,3]` and SWERank
`[7,1,4,5,1,1]`. SWERank won one case, tied two and lost three. All six cases
have one gold production file. Macro Recall@10 has no remaining headroom in
this pilot. These small public development results favor keeping BM25 as the
baseline; they do not establish a general model ranking.

Dense indexing took 2265.65 seconds total (37.8 minutes) on two Torch CPU
threads, plus 4.50 seconds initialization. Mean warm dense query encoding and
ranking was 0.436 seconds (range 0.155 to 0.861). BM25 corpus scoring and ranking,
including lexical preparation on each query, averaged 0.0377 seconds. These
are single-query observations on a shared host, with light verification tasks
running alongside some indexing; they are not controlled serving percentiles.
No paid GPU job or fine tuning was used. Download/preparation time is excluded.

SWERank truncated 429 of 3013 chunks (14.2%) and one of six queries. Historical
syntax fallback preserved 37 chunks. There were 332 file candidates summed
across snapshots. `localization/pilot-results.json` includes the complete
receipt, predictions, gold paths and 17 download metadata checks. Seven focused
localization tests and 29 existing transfer tests pass. Python compilation and
independent metric verification pass. The pinned Ruflo 3.25.6 static scan
reported zero findings; Python and dependency advisory coverage is not
established. Its unrelated helper-file refresh was reverted.

Decision: retain the lexical baseline and keep the SWERank variant experimental.
The next comparison needs harder, separately held out repositories and a frozen
protocol for hybrid retrieval and RuVector graph scoring, with identical
candidates and measured latency. Unknown training overlap, partial chunk
coverage, truncation and the noncommercial model license remain material limits.
No graph arm has run and no transfer or SOTA claim is accepted.


## Frozen inductive code-graph follow-up

`localization/transfer-protocol.json` locks the pinned dataset revision and hash, repository/issue selection rules, resource caps, training split, and acceptance gate before confirmation scoring. It uses all 23 official development cases and selects one task from each of three confirmation repositories that are disjoint from development and the earlier Flask/Requests pilot.

`localization/graph_rerank.py` is a CPU-only experimental benchmark sidecar. It builds typed import, call, and inheritance relations from complete source text and fits one pairwise ridge reranker using development labels. It consumes BM25 and SWERank file ranks over the same candidate universe. It does not call `@ruvector/kge` and is not a production integration.

The protocol has not been scored. The complete pinned code snapshots and dense model artifacts must be restored and hash-verified before running it. Eleven focused localization tests currently pass with `python3 -m unittest discover -s localization -p 'test_*.py'`, including deterministic leave-one-repository-out L2 selection; passing those tests does not establish cross-repository transfer.
