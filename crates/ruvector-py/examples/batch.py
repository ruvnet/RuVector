"""In-memory batch operations with explicit numeric vectors."""

from ruvector import VectorDB, VectorRecord

with VectorDB(2) as db:
    ids = db.insert_batch(
        VectorRecord(f"axis-{i}", vector, {"kind": "axis"})
        for i, vector in enumerate([[1, 0], [0, 1], [-1, 0]])
    )
    results = db.search_batch([[1, 0], [0, 1]], k=3, filter={"kind": "axis"})
    print([[result.id for result in row] for row in results])
    print(db.delete_batch(ids))
