# ADR: Validate GNN checkpoint configuration and tensors before loading

## Status

Accepted for implementation; exact-source native execution, scoped checks and independent review are required before publication.

## Context

The Python GnnLayer envelope repeats dimensions, attention heads and dropout alongside a serialized RuvectorLayer. Serde validates the JSON representation but does not establish consistency between those declarations and the nested tensors. from_json accepts an envelope declaring hidden_dim=0 around a layer whose forward output has eight elements, inconsistent attention layouts, dimension-valid tensors incompatible with neighboring operations, nonpositive normalization epsilon, and finite JSON numbers that overflow f32 parameters. The wrapper relies on its envelope for pre-forward validation, so inconsistent loaded data bypasses that boundary.

## Decision

Add a reusable RuvectorLayer::validate_checkpoint operation and invoke it after deserialization before constructing the Python wrapper. Check positive/divisible declared dimensions, attention head count and head width, matching dropout, all linear matrix/bias shapes, normalization vector lengths, finite parameters and positive finite epsilon.

Retain the existing serialized format and every valid learned parameter. Validation does not construct fresh template weights, repair incompatible checkpoints, train a model or change forward-pass behavior. The implementation lives in a child validation module to keep the existing layer algorithm unchanged.

## Cost and limits

Validation scans stored parameters once at load time, adding O(parameter count) work with no additional parameter copies. It establishes structural configuration consistency and finite stored parameters. It does not prove trained-model quality or numerical stability for arbitrary finite weights or inference inputs. The generic core type still derives Deserialize; callers loading that type directly can explicitly invoke the validator. The Python checkpoint loader enforces it automatically.

## Validation

Twenty-six real Python/native cases cover envelope and attention mismatch, message/aggregation/GRU/attention/norm tensor inconsistencies, epsilon and dropout mismatch, parameter overflow, and both original and modified learned-weight round trips. Original source fails twenty-four cases and passes both valid controls. A guarded reversible draft overlay passed real Python-crate compilation and restored the previous clean source. Exact-commit native replay and broader Rust/Python checks are recorded before publication rather than inferred from the draft check.
