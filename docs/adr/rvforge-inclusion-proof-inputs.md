# RVForge inclusion proof input validation

Status: proposed

## Context

The public RVForge Merkle verifier accepts a one-leaf proof with a fractional or NaN leaf index, or matching buffers that are not SHA-256 digests. The registry wrapper also accepts a valid hexadecimal digest followed by an invalid suffix because Node truncates hexadecimal decoding. Such inputs do not describe an RFC 6962 inclusion proof.

## Decision

Require safe integer positions and positive tree sizes, exactly 32-byte binary hashes, and canonical `sha256:` wire digests before verification. Invalid inputs return false. Valid tree construction and audit paths are unchanged.

## Evidence and consequences

On unchanged upstream, four new regression cases fail while 29 focused cases pass. The fix preserves proof verification for every leaf of trees sized one through sixteen and the existing registry round trips. Validation covers software only; it does not establish registry trust or authenticate an unsigned tree head. The TypeScript package build and full package test suite are the local gates; hosted cross-platform and Rust parity checks remain separately reported.
