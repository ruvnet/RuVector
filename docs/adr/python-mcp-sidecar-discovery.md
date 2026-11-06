# Discover persisted collections from their sidecars

## Context
The public list tool scans native index files, but empty RaBitQ collections persist only their sidecar until the first vector is added. They are usable through stats and insert but disappear from discovery after server restart. Conversely, an index lacking its required sidecar is advertised even though load rejects it. Collection.meta_path documents the sidecar as the existence contract.

## Decision
Enumerate the sidecar filenames instead of native indexes, stripping the complete suffix. Preserve sorted names and one entry per collection. This discovers empty collections and excludes index-only incomplete restores.

## Evidence
Public MCP producer tests create both backend types, clear the server cache and query discovery plus stats. A separate incomplete-index control verifies one complete collection is listed once. The unchanged upstream fails two regressions with 58 existing/control tests passing; the fix passes the full Python suite with 285 tests and 2 optional-integration skips. Runtime is current Python source plus the official published 0.1.2 macOS native extension and real MCP SDK.

## Limits
Listing does not validate every sidecar's contents or repair corrupt data. Native-source builds and other platforms are outside this local evidence.
