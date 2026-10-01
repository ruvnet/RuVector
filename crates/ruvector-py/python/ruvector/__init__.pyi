"""Type stubs for the ``ruvector`` package.

Hand-written per ``docs/sdk/02-strategy.md`` § "Type stubs". Validates
against ``mypy --strict`` and ``pyright``. The real symbols are declared in
``_native.pyi`` (compiled extension) and ``collection.py`` (plain annotated
Python, py.typed covers it directly); this file just re-exports both so
`ruvector.RabitqIndex` / `ruvector.Collection` resolve statically despite
`__init__.py` using PEP 562 `__getattr__` for the real lazy runtime lookup
(see `__init__.py`'s module docstring for why).
"""

from ruvector._native import RabitqIndex as RabitqIndex
from ruvector._native import RuVectorError as RuVectorError
from ruvector._native import __version__ as __version__
from ruvector.collection import Collection as Collection
from ruvector.collection import CollectionError as CollectionError
from ruvector.collection import CollectionStats as CollectionStats
from ruvector.collection import SearchHit as SearchHit

__all__ = [
    "RabitqIndex",
    "RuVectorError",
    "__version__",
    "Collection",
    "CollectionError",
    "CollectionStats",
    "SearchHit",
]
