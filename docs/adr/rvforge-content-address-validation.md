# Validate RVForge registry content addresses

Status: proposed

## Context

The registry stores an object under any supplied digest and returns stored JSON without comparing its content to that address. A caller can overwrite an existing immutable release with different content, or a damaged object file can be returned as if it were the originally addressed release.

## Decision

Recompute the existing canonical object identity before both writes and reads. Reject content that does not match its address with the stable registry error. Keep the current identity convention, including signature exclusion, so adding a signature can still refresh the same object. Reject non-object payloads.

## Evidence and consequences

The new real-file tests fail on unchanged upstream for a mismatched write and a changed stored release. Tests assert that rejected writes leave both absent and existing object paths intact, and that signature updates remain compatible. Full RVForge package tests, build, and typecheck validate local software. This checks content integrity only; it does not authenticate signatures, prevent concurrent writers, or establish registry trust.
