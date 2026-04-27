"""Type stubs for ruvector M1.

Hand-written per docs/sdk/02-strategy.md § "Type stubs". Validates against
``mypy --strict`` and ``pyright``.
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
        rerank_factor: int = ...,
        seed: int = ...,
    ) -> "RabitqIndex":
        """Build an index from an ``(n, dim)`` float32 array.

        ``vectors`` must be C-contiguous; non-contiguous arrays raise
        ``TypeError``. ``rerank_factor`` defaults to 20 (the ADR-154
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

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

    @property
    def dim(self) -> int: ...
    @property
    def memory_bytes(self) -> int: ...
    @property
    def rerank_factor(self) -> int: ...

__all__ = ["RabitqIndex", "RuVectorError", "__version__"]
