# Preserve cosine ranking across finite coordinate scales in the CLI

Status: proposed for review

## Problem and evidence

The current JavaScript CLI squares coordinates for cached norms. Large finite values overflow the norm/dot arithmetic and serialize scores as null; very small finite values underflow to zero, producing false ties. Five real persisted CLI searches reproduce two failures at1e200/1e-200 and three passing controls at1e100/ordinary/zero vectors. Inserting an orthogonal vector before its matching vector makes the ranking failure visible rather than merely inspecting a helper.

An independent producer probe additionally found that a tiny positive cached norm paired with a large query reports1.0059877 for an exact aligned cosine. The sixth regression reproduces this against candidate9d608f05; the final guard routes subnormal squared norms through scaling.

## Decision

Retain the unrolled cached-norm calculation whenever dot and norm product are finite and both norms are large enough that squaring avoids subnormal precision. For overflow/underflow, scale each vector by its own largest absolute coordinate before computing the normalized dot product. Actual zero vectors retain score zero. This uses O(d) work and O(1) extra memory only for the fallback; it does not rescale stored vectors or change persistence formats.

## Validation and limits

Six real Node24 CLI subprocess tests pass with syntax/diff checks; independent exact-commit producer review is required. This targets finite equal-dimensional cosine inputs, not malformed vector validation, all distance metrics, arbitrary exact arithmetic or a WASM build. Standard floating-point rounding still applies. No throughput benchmark improvement is claimed.

## Changelog

Preserve finite cosine scores and correct ranking for large and tiny finite vectors after JSON reload.
