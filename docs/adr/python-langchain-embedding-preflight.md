# Validate LangChain embedding outputs before inserting documents

Status: proposed for review

## Problem and evidence

The actual RuVectorStore.add_texts API zips documents with embedding-provider output. Missing outputs silently omit documents, surplus outputs disappear, and a later wrong-dimensional or nonfinite vector raises only after inserting earlier documents. Five deterministic offline-provider cases reproduce those failures against upstream07933730 with its freshly built native extension; a valid metadata/ID control passes.

## Decision

Require one embedding per input text, then validate every embedding using the Collection float32 shape/finite-value validator before any insertion. Keep provider calls, explicit IDs, metadata convention and successful native insertion behavior unchanged. This adds O(n*d) validation and retains O(n*d) prepared arrays until insertion; it does not copy already contiguous float32 arrays unnecessarily.

## Validation and limits

The full Python suite passes301 tests with2 skips against the current079 native extension, including six real adapter/native-collection producer cases. Independent exact-commit review is required. This guards embedding cardinality, dimensions and finite values; it does not promise transactional rollback for arbitrary later native or malformed metadata failures, concurrency or provider service reliability. No paid model/network calls were used.

## Changelog

Reject malformed embedding batches before changing existing documents or caller-visible ID mappings.
