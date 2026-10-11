# Keep vector identities stable during RaBitQ vacuum

## Context

Collection.vacuum promises physical reclamation but previously rebuilt RaBitQ rows with sequential IDs and reset the insertion counter. Adapters that retain document-to-vector mappings could retrieve documents without their original IDs or delete the wrong row after compaction. Emptying the collection also reused deleted identities.

## Decision

Pass retained external IDs into RabitqIndex.build and retain the existing next-ID counter. Filter metadata by retained identity rather than remapping it. When inserting into an index emptied by vacuum, pass the next assigned IDs explicitly to the native builder.

## Consequences

Vacuum retains its existing full reconstruction cost and returned dropped-row count. Row identity remains stable and IDs are not recycled after deleting all rows. This depends on the native external-ID export/save correction included in this change. It does not migrate already-renumbered collections or repair stale adapter maps caused by earlier compactions.

## Validation

An actual LangChain public producer adds three documents, deletes one, compacts, retrieves the next document with its original caller ID, and then deletes it correctly. Collection controls cover sparse identities, metadata, a no-op second vacuum, monotonic insertion, and compaction to empty. Tests bind to the source-built native artifact whose native source files are identical to this revision.
