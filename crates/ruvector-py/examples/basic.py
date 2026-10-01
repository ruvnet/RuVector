"""Run with: python examples/basic.py. Uses explicit vectors, not model embeddings."""

from pathlib import Path
from tempfile import TemporaryDirectory

from ruvector import VectorDB, VectorRecord

with TemporaryDirectory() as directory:
    path = Path(directory) / "memory.redb"
    with VectorDB(3, path=path) as db:
        db.insert_batch(
            [
                VectorRecord("first", [1, 0, 0], {"tenant": "acme"}),
                VectorRecord("second", [0, 1, 0], {"tenant": "other"}),
            ]
        )
        print(db.search([1, 0, 0], k=2, filter={"tenant": "acme"}))
        print("Deleted:", db.delete("second"))
    with VectorDB(path=path) as db:
        print("Reopened:", db.dimensions, len(db), db["first"])
