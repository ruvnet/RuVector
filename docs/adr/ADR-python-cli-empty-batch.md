# Decision: report empty CLI batches without an ID range

The Collection API accepts an empty HNSW batch and returns no IDs. The public
insert-batch command saved the collection, then indexed that empty list while
printing success, causing a nonzero exit after a valid no-op.

Keep the existing Collection validation and save path. Print the row count and
omit the ID range when there are no returned IDs. Nonempty batches retain the
same output. A Click public-command regression verifies exit success and the
persisted zero row count. Tests use current Python source with the official
0.1.2 macOS native wheel; they do not claim a fresh native-source build.
