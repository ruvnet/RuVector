# Refresh graph indexes when replacing node, edge or hyperedge IDs

Status: proposed for review

## Problem and evidence

The public Rust create_node/create_edge/create_hyperedge methods currently replace objects with matching IDs while only adding new index memberships. Old labels/properties, relationship types/adjacency and hyperedge memberships continue to return the replacement under attributes it no longer has. Deleting a replaced node also leaves stale node IDs in its old label index. Five actual Rust public-consumer tests against source-identical upstream GraphDB reproduce four failures and one ordinary creation/update control.

## Decision

Keep the existing Rust upsert behavior. Hold the corresponding DashMap entry for one object ID, persist the candidate before changing live memory, remove the old object's index memberships, add new memberships and replace the entry. Same-ID replacements are serialized; existing missing-endpoint validation remains before relationship mutation. The Python create-only duplicate guard remains unchanged, with its explanation updated.

## Validation and limits

Seven focused public Rust tests additionally cover concurrent same-ID writers and rejected relationship replacement. Full applicable graph tests, formatting, scoped Clippy and independent exact-commit review are required. This is consistency after an operation completes, not a global transaction across index readers, other objects or concurrent node deletion. Existing shard lock ordering follows update_node; no callback executes under these new entry locks. Work is proportional to the replaced object's labels/properties/relationship membership, without cloning entire indexes.

## Changelog

Remove stale node, edge and hyperedge index memberships during existing-ID replacement.
