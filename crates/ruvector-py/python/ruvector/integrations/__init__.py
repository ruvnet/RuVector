"""Third-party framework adapters for :class:`ruvector.Collection`.

Nothing is imported eagerly here: ``import ruvector.integrations`` (or
``import ruvector``) alone must not require ``langchain-core`` or
``llama-index-core``. Each adapter lives in its own submodule
(:mod:`ruvector.integrations.langchain`, :mod:`ruvector.integrations.llamaindex`)
and only pulls in its framework's dependency when that submodule itself is
imported — see each module's docstring for the exact boundary.
"""

from __future__ import annotations

__all__: list[str] = []
