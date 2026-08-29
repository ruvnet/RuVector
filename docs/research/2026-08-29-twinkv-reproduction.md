# TwinKV reproduction protocol

Source: arXiv 2608.27128, submitted 2026 08 27.

Evidence class: originating team report. No RuV model level reproduction yet.

## Baseline and candidate

Baseline: existing RuV KV eviction policy alone.

Candidate: identical policy and retained budget followed by `repair_retained_set`.

## Environment record

Pin model revision, tokenizer revision, RuVector commit, Rust toolchain, inference backend, driver, accelerator, operating system, cache precision, batch size, context length, and compression ratio.

## Workloads

1. LongBench representative tasks including one code task, one multi hop QA task, and TREC as an expected negative control.
2. RULER at 4K, 8K, and 16K context.
3. A short context accuracy control.
4. One real RuV agent workload with long tool and repository context.

## Models

1. Qwen3 4B, initial similarity threshold 0.85.
2. Llama 3.2 1B, initial similarity threshold 0.90.

Pin exact model revisions before running.

## Metrics

For every condition report baseline and candidate task score, absolute delta, relative delta, prefill latency, repair latency, decode throughput, cache bytes, peak memory, energy when measurable, swap count, orphan count, donor count, and failure count.

Use identical examples and seeds. Report variance across at least three seeds where decoding is stochastic.

## Ablations

1. Similarity threshold ladder around the paper value.
2. Local window 0, 16, 32, and 64.
3. Repair specific O(n K d) implementation versus full pairwise audit on a bounded context to confirm decision equivalence.
4. Max swap cap.
5. Random budget preserving swap negative control.

## Degradation tests

1. Zero norm keys.
2. Non finite keys.
3. Inconsistent dimensions.
4. Duplicate retained positions.
5. Missing protected sink or recent positions from the wrapped policy.
6. Extremely low and high compression.
7. TREC style repeated templates.
8. LongRoPE or another nonuniform rotary schedule before architecture specific correction is enabled.

## Acceptance gate

Graduate only when the target workload shows a positive mean quality effect, short context control does not regress by more than 0.5 absolute points, retained budget is preserved in every case, and measured repair overhead does not erase the end to end latency or memory benefit of eviction.

Record negative policies and task structures. Do not enable globally from a positive result on one model or benchmark.
