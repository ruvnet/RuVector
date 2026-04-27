"""Type stubs for the `ruvector` package — mirrors python/ruvector/__init__.pyi.

PEP 561 has two valid stub layouts: stubs alongside the package (with
``py.typed`` and ``__init__.pyi``) or a separate stub-only package. We
ship both: the wheel includes ``python/ruvector/__init__.pyi`` for
in-package consumption, and this tree carries the same stubs for
tooling that reads PEP 561 stub-only packages directly.

Keep this file byte-identical (modulo this docstring) to
``python/ruvector/__init__.pyi``. A CI lint enforces the equivalence.
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

__version__: str

class RuVectorError(Exception):
    """Base class for every error raised by the ruvector extension."""

class RabitqIndex:
    """RaBitQ+ index — symmetric 1-bit scan with exact f32 rerank."""

    @staticmethod
    def build(
        vectors: NDArray[np.float32],
        *,
        rerank_factor: int = ...,
        seed: int = ...,
    ) -> "RabitqIndex": ...

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        *,
        rerank_factor: Optional[int] = ...,
    ) -> List[Tuple[int, float]]: ...

    def save(self, path: str) -> None: ...

    @staticmethod
    def load(path: str) -> "RabitqIndex": ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

    @property
    def dim(self) -> int: ...
    @property
    def memory_bytes(self) -> int: ...
    @property
    def rerank_factor(self) -> int: ...

__all__ = ["RabitqIndex", "RuVectorError", "__version__"]
