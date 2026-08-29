# ADR 330: KV redundancy repair as a composable cache optimization

Status: Proposed

Date: 2026 08 29

## Context

TwinKV, arXiv 2608.27128, reports that attention magnitude was statistically unrelated to leave one out causal utility in its controlled long context probe, with Spearman rho near zero. The paper proposes a training free repair pass that audits any existing KV eviction policy after selection.

The repair identifies two structural mistakes. An orphan is an evicted token with no sufficiently similar surviving key outside a local exclusion window. A donor is a retained unprotected token with a sufficiently similar surviving key. The repair swaps severe orphans for redundant donors while preserving the exact original cache budget.

The originating team reports mixed but useful results across Qwen3 4B and Llama 3.2 1B. It improves some wrapped policies substantially, is near even for others, and regresses on a few shot classification structure. This is not evidence that every cache policy should always enable the repair.

## Decision

Add an opt in TwinKV style repair primitive under `ruvllm::optimization`.

The RuV implementation computes best surviving similarity directly against the retained set. This follows the repair equation while avoiding construction of the full pairwise similarity matrix. Repair specific work is O(n K d), where n is context length, K is retained budget, and d is key dimension.

The primitive is policy agnostic. It accepts key vectors and a retained position set. It returns a repaired retained set plus explicit swap receipts. It does not own the underlying eviction score, cache allocation, model execution, or authority decisions.

## Invariants

1. The retained budget is exactly preserved.
2. Protected sink and recent positions are never selected as donors.
3. Local neighbors inside the exclusion window cannot establish redundancy.
4. Malformed, non finite, zero norm, inconsistent dimension, duplicate, and out of range inputs fail closed.
5. Equal inputs produce deterministic retained sets and swap receipts.
6. No model call or training step is introduced.
7. The feature remains opt in until matched benchmark reproduction establishes a positive workload specific effect.

## Contradictions and limits

The published effect is not universal. ExpectedAttention on Qwen3 4B often loses because the baseline is already near a ceiling. TREC style few shot exemplars also regress because template similarity can look like semantic redundancy.

The fixed similarity threshold is architecture sensitive. The paper uses 0.85 for Qwen3 4B and 0.90 for Llama 3.2 1B. Production integration therefore must keep threshold policy explicit and versioned rather than treating one threshold as universal.

Raw post RoPE similarity also needs special handling for nonuniform rotary schedules such as LongRoPE. This initial primitive assumes the caller supplies a representation where cosine similarity is meaningful for the target architecture.

## Validation plan

Reproduce the wrapped policy alone and wrapped policy plus repair on identical samples, seeds, model revisions, tokenizer revisions, cache ratios, and hardware.

Initial ladder:

1. Qwen3 4B at compression ratios 0.3, 0.5, and 0.7.
2. Llama 3.2 1B at the same ratios.
3. LongBench, RULER, and a short context no harm control.
4. At least one existing RuV cache policy.

Report task score, cache bytes, prefill latency, repair latency, decode throughput, peak memory, energy when available, swap count, orphan count, donor count, failures, and variance.

Promotion requires a positive mean quality effect on the target workload, no material short context regression, exact budget preservation, and repair overhead small enough that end to end latency remains favorable.

## Security and governance

This primitive only changes which cached positions survive. It must not alter execution permissions, model identity, policy authority, or evaluation state. Configuration and benchmark provenance should be recorded in RVF or the existing witness path before any adaptive threshold learning is enabled.

## Rollback

Disable the repair pass and retain the original eviction set. No state migration is required.
