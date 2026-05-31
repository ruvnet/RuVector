"""ruvector — vector similarity search via RaBitQ 1-bit quantization.

M1 surface only: a single ``RabitqIndex`` class plus the ``RuVectorError``
base exception. See ``docs/sdk/04-milestones.md`` for what M2/M3/M4 add
(RuLake, Embedder, A2aClient).
"""

from ruvector._native import RabitqIndex, RuVectorError, __version__

__all__ = ["RabitqIndex", "RuVectorError", "__version__"]
