# ADR: Claim vacant memory-storage IDs for generated inserts

## Status

Accepted for this implementation; exact-revision independent review and scoped checks are required before publication.

## Context

Memory storage labels its counter-based IDs as unique, but explicit IDs do not advance that counter. Inserting a caller-owned vec_0 followed by an ID-less vector therefore overwrites the existing vector and metadata. The same collision occurs in batches and with concurrent generated inserts. A batch containing an ID-less entry before an explicit vec_0 reports duplicate IDs and loses a row.

## Decision

Retain explicit-ID upserts. For generated IDs, advance the counter with checked arithmetic and atomically claim a vacant DashMap vector entry. Skip occupied IDs rather than separately checking then overwriting them. Within a batch, skip every explicitly declared ID, including IDs appearing later in the batch. Refuse counter exhaustion rather than wrapping to zero.

This reserves the vector slot at insertion time. A future explicit insert can intentionally upsert that ID; it is not excluded forever. Metadata behavior and the existing storage/index boundary are unchanged. This change does not add a transaction across the vector map, metadata map, or subsequent indexing.

## Tradeoffs

An ID-less insert may probe several occupied or batch-reserved names. Each rejected name advances the monotonic counter; the work is bounded by the finite counter space and does not wait on occupied IDs. Batch reservation uses O(m) auxiliary references for m distinct declared IDs. Generated values range from zero through u64::MAX - 1; exhaustion at the terminal counter returns a storage error, while explicit inserts remain supported. Ordinary collision-free generated IDs retain their existing format and sequence.

## Validation

Five real public Rust cases cover single and batch collisions, later explicit IDs in a mixed batch, sixteen concurrent inserts around eight preexisting caller IDs, and normal allocation plus intentional explicit upserts. Original source fails four cases and passes the control; fixed source passes all five. A controlled private-counter boundary test demonstrates original wraparound behavior and fixed refusal without changing the old row. Broader scoped checks and exact-binary/source bindings are recorded before publication. No cross-platform or full transaction claim is made.
