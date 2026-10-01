"""Type stub for the compiled PyO3 extension module ``ruvector._native``.

Hand-written per ``docs/sdk/02-strategy.md`` § "Type stubs" (same rationale
as ``__init__.pyi``, split into its own file so ``from ._native import ...``
in ``collection.py`` type-checks under ``mypy --strict`` — without this file
mypy cannot find an implementation for the compiled module and every
subclass of ``RuVectorError`` elsewhere resolves to ``Any``).
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

__version__: str

class RuVectorError(Exception):
    """Base class for every error raised by the ruvector extension."""

class RabitqIndex:
    """RaBitQ+ index — symmetric 1-bit scan with exact f32 rerank.

    Backed by ``ruvector_rabitq::RabitqPlusIndex``. Build with
    :meth:`build`, query with :meth:`search`, persist via :meth:`save` /
    :meth:`load`.
    """

    @staticmethod
    def build(
        vectors: NDArray[np.float32],
        *,
        ids: Optional[NDArray[np.uint64]] = ...,
        rerank_factor: int = ...,
        seed: int = ...,
    ) -> "RabitqIndex":
        """Build an index from an ``(n, dim)`` float32 array.

        ``vectors`` must be C-contiguous; non-contiguous arrays raise
        ``TypeError``. ``ids`` (optional) assigns the search-result id for
        each row instead of the default ``0..n`` row index; every id must
        fit in ``u32`` (the index's storage width) or this raises
        ``ValueError``. ``rerank_factor`` defaults to 20 (the ADR-154
        recommendation for 100% recall@10 at D=128). ``seed`` defaults
        to 42 for deterministic builds.
        """
        ...

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        *,
        rerank_factor: Optional[int] = ...,
    ) -> List[Tuple[int, float]]:
        """Search for the ``k`` nearest neighbours of ``query``.

        Returns a list of ``(id, score)`` tuples in ascending score
        order (squared L2). ``rerank_factor=None`` (the default) reuses
        the value the index was built with.
        """
        ...

    def save(self, path: str) -> None:
        """Persist the index to ``path`` in the ``.rbpx`` v1 format."""
        ...

    @staticmethod
    def load(path: str) -> "RabitqIndex":
        """Load an index previously written by :meth:`save`."""
        ...

    def add(self, id: int, vector: NDArray[np.float32]) -> None:
        """Append one vector in place (true incremental add, no rebuild).

        Not GIL-released — see ``docs/sdk/02-strategy.md`` § "GIL story".
        Prefer :meth:`add_batch` for more than a few inserts.
        """
        ...

    def add_batch(self, ids: NDArray[np.uint64], vectors: NDArray[np.float32]) -> None:
        """Append many vectors at once. Releases the GIL around the loop.

        Accepts u64 ids but, like :meth:`build`, every id must fit in
        ``u32`` (the index's storage width) or this raises ``ValueError``.
        """
        ...

    def export_items(self) -> List[Tuple[int, NDArray[np.float32]]]:
        """Return every ``(id, vector)`` pair currently held.

        Used by ``ruvector.Collection.vacuum()`` to physically drop
        tombstoned rows by rebuilding without them — there is no
        ``delete`` on the underlying index.
        """
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    @property
    def dim(self) -> int: ...
    @property
    def memory_bytes(self) -> int: ...
    @property
    def rerank_factor(self) -> int: ...

__all__ = ["RabitqIndex", "RuVectorError", "__version__"]
