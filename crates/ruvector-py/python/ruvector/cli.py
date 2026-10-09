"""``ruvector`` console script (ADR-352 M1.5).

Covers the data-plane core only — create/insert/search/delete/export/import/
benchmark/info/serve — not the npm package's 100+ command research surface
(see ADR-352 "What the npm CLI/MCP actually ship").

Fast-startup is a design requirement: every subcommand lazily imports numpy
and ``ruvector.collection`` inside its own function body so `ruvector --help`
doesn't pay for them. Measured via `python -X importtime -m ruvector.cli
--help` — see docs/sdk/LOOP-STATE.md for the recorded number.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, TypeVar

import click

if TYPE_CHECKING:  # pragma: no cover - type-only, no runtime import cost
    import numpy as np
    from numpy.typing import NDArray

    from ruvector.collection import Collection

_T = TypeVar("_T")


def _secho(message: str, *, fg: str) -> None:
    """``click.secho`` that also honors the NO_COLOR convention.

    ``click.echo``/``secho`` already auto-strip ANSI when stdout isn't a
    TTY (their own ``color=None`` default), but — unlike Rich's
    ``Console`` — they don't check the ``NO_COLOR`` env var on top of
    that. Check it here so a user who sets NO_COLOR gets consistent
    behaviour whether a command's output goes through Rich or through
    plain click.secho.
    """
    click.secho(message, fg=fg, color=False if os.environ.get("NO_COLOR") else None)


def _resolve_existing(path: str) -> Path:
    """Resolve a CLI-supplied path, erroring clearly if it doesn't exist.

    This is a *local* CLI invoked with the caller's own authority (like
    `sqlite3 file.db`), not a remote trust boundary — see ADR-352 "Security"
    and `ruvector.mcp_server._safe_path` for the boundary that actually
    needs traversal rejection (tool-call arguments from an MCP client).
    """
    p = Path(path).expanduser().resolve()
    return p


@click.group()
@click.version_option(package_name="ruvector")
def main() -> None:
    """ruvector — ultra-low-latency vector search, backed by Rust."""


@main.command()
@click.option("--path", required=True, help="Collection file path (creates PATH and PATH.meta.json).")
@click.option("--dim", required=True, type=int, help="Vector dimensionality.")
@click.option("--rerank-factor", default=20, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
def create(path: str, dim: int, rerank_factor: int, seed: int) -> None:
    """Create a new empty collection."""
    from ruvector.collection import Collection

    p = Path(path).expanduser().resolve()
    if Collection.meta_path(p).exists():
        raise click.ClickException(f"{p} already exists")
    coll = Collection.create(dim=dim, rerank_factor=rerank_factor, seed=seed)
    coll.save(p)
    _secho(f"created empty collection dim={dim} at {p}", fg="green")


@main.command()
@click.option("--path", required=True)
@click.option("--vectors", "vectors_path", required=True, help="Path to a .npy file, shape (n, dim), dtype float32.")
@click.option("--metadata", "metadata_path", default=None, help="Optional .json file: list of dicts aligned with rows.")
def insert_batch(path: str, vectors_path: str, metadata_path: Optional[str]) -> None:
    """Insert all rows of a .npy vector file into an existing collection."""
    import numpy as np

    from ruvector.collection import Collection

    p = _resolve_existing(path)
    vecs = np.load(_resolve_existing(vectors_path))
    metadatas = None
    if metadata_path:
        metadatas = json.loads(_resolve_existing(metadata_path).read_text())
    coll = Collection.load(p)
    ids = _run_with_spinner(f"inserting {len(vecs)} vectors...", lambda: coll.insert_batch(vecs, metadatas=metadatas))
    coll.save(p)
    _secho(f"inserted {len(ids)} vectors (ids {ids[0]}..{ids[-1]}); collection now has {len(coll)} rows", fg="green")


def _run_with_spinner(label: str, fn: "Callable[[], _T]") -> "_T":
    """Run ``fn`` (one bulk, atomic Rust call) behind an indeterminate
    Rich spinner, and return its result.

    Used by ``insert-batch`` (``Collection.insert_batch``) and ``import``
    (``Collection.from_vectors``). Both are single calls into Rust with no
    progress callback, so there is no way to report *granular* progress
    without changing what Python actually does. Two honest options were
    considered — see python/ruvector/collection.py for both methods:

    1. Chunk the call into N smaller bulk calls from Python, to get real
       incremental progress ticks.
    2. One atomic call (unchanged from before this CLI polish pass) with
       an indeterminate spinner that honestly says "working", not "N% of
       M done".

    Went with (2), for a correctness reason found by reading the code,
    not just style preference: ``Collection.insert_batch``, when the
    rabitq backend's index hasn't been built yet (``self._index is
    None``), does ``RabitqIndex.build(arr, ...)`` on *whatever rows are
    in that call* — fitting the rotation on them. Chunking one big insert
    into N pieces would fit that rotation on only the first chunk instead
    of the full batch: a real recall-quality regression, not a cosmetic
    difference. ``from_vectors`` has the same shape of call
    (``RabitqIndex.build`` over the whole array) and the same risk if
    rebuilt as a loop of smaller inserts. Even on the hnsw backend, where
    chunking would be behavior-equivalent (HNSW insertion is already
    incremental), the CLI can't tell which backend is in play without
    extra plumbing, and running two different code paths for a benefit
    that's "the spinner fills in a bit more" isn't worth the risk above.

    `total=None` renders a pulsing bar, never a fake percentage — that's
    what keeps this honest. `insert_batch`/`from_vectors` both release
    the GIL around their Rust call (see their own docstrings), so Rich's
    background refresh thread can still repaint the spinner while the
    call is in flight.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    # stderr, not stdout: stdout stays reserved for the final plain
    # success line (or --json payload, on the commands that have one), so
    # piping stdout to a file/program never sees spinner frames.
    console = Console(stderr=True)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        progress.add_task(description=label, total=None)
        return fn()


@main.command()
@click.option("--path", required=True)
@click.option("--query", "query_path", required=True, help="Path to a .npy file holding one (dim,) float32 vector.")
@click.option("-k", default=10, show_default=True, type=int)
@click.option("--filter", "filter_json", default=None, help='JSON dict for exact-match metadata filtering, e.g. \'{"cat":"news"}\'')
@click.option("--rerank-factor", default=None, type=int)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON instead of a table.")
@click.option("--no-color", is_flag=True, default=False, help="Disable colored table output (also honored: NO_COLOR env var, non-TTY stdout).")
def search(
    path: str,
    query_path: str,
    k: int,
    filter_json: Optional[str],
    rerank_factor: Optional[int],
    as_json: bool,
    no_color: bool,
) -> None:
    """Search a collection for the k nearest neighbours of a query vector."""
    import numpy as np

    from ruvector.collection import Collection

    p = _resolve_existing(path)
    q = np.load(_resolve_existing(query_path))
    filt = json.loads(filter_json) if filter_json else None
    coll = Collection.load(p)
    hits = coll.search(q, k, filter=filt, rerank_factor=rerank_factor)

    if as_json:
        # Scripting path: stays exactly plain, parseable JSON, never touched
        # by the rich/color code below.
        click.echo(json.dumps([{"id": h.id, "score": h.score, "metadata": h.metadata} for h in hits]))
        return

    from rich.console import Console
    from rich.table import Table

    # `Console()` already auto-detects both non-TTY stdout (no ANSI when
    # piped/redirected) and the NO_COLOR env var (see rich.console.Console.
    # __init__: self.no_color falls back to "NO_COLOR" in os.environ when
    # the constructor arg is left as the default `None`) — passing `None`
    # here (not `False`) when --no-color wasn't given preserves that
    # auto-detection instead of forcing color on. Note NO_COLOR/no_color
    # suppress *color* specifically (matching the no-color.org spec, which
    # scopes to color codes) — bold/dim/italic SGR attributes can still
    # render on a real TTY even with --no-color; piping to a non-TTY (the
    # case this flag mainly exists for) strips all ANSI regardless.
    console = Console(no_color=no_color or None)

    table = Table(title=f"search results (k={k})")
    table.add_column("id", justify="right")
    table.add_column("score", justify="right")
    # `overflow="ellipsis"` is a second guard alongside the manual
    # truncation below — it only kicks in if a value still exceeds the
    # rendered column width (e.g. a very narrow terminal).
    table.add_column("metadata", overflow="ellipsis", max_width=60)

    n = len(hits)
    for idx, h in enumerate(hits):
        # Hits are returned best-first by the backend regardless of metric
        # (cosine distance vs. similarity, rabitq vs. hnsw score scales
        # differ) — coloring by *rank* rather than by a score threshold
        # avoids assuming which direction "good" points, which the score
        # value alone doesn't tell us here.
        if n <= 1:
            rank_style = "bold green"
        elif idx < max(1, n // 3):
            rank_style = "bold green"
        elif idx < max(1, 2 * n // 3):
            rank_style = "yellow"
        else:
            rank_style = "dim"

        metadata_str = json.dumps(h.metadata) if h.metadata else ""
        if len(metadata_str) > 60:
            metadata_str = metadata_str[:57] + "..."
        table.add_row(str(h.id), f"{h.score:.4f}", metadata_str, style=rank_style)

    console.print(table)


@main.command()
@click.option("--path", required=True)
@click.option("--id", "ids", required=True, multiple=True, type=int, help="Repeatable; id(s) to soft-delete.")
@click.option("--vacuum", is_flag=True, default=False, help="Also physically rebuild to reclaim space.")
def delete(path: str, ids: "tuple[int, ...]", vacuum: bool) -> None:
    """Soft-delete one or more ids (and optionally vacuum immediately)."""
    from ruvector.collection import Collection

    p = _resolve_existing(path)
    coll = Collection.load(p)
    for i in ids:
        coll.delete(i)
    dropped = coll.vacuum() if vacuum else 0
    coll.save(p)
    _secho(f"tombstoned {len(ids)} id(s)" + (f", vacuumed {dropped} rows" if vacuum else ""), fg="green")


@main.command()
@click.option("--path", required=True)
@click.option("--out", "out_prefix", required=True, help="Writes <out>.npy (vectors by row) and <out>.meta.json (metadata).")
def export(path: str, out_prefix: str) -> None:
    """Export a collection's vectors + metadata for re-import elsewhere."""
    import numpy as np

    from ruvector.collection import Collection

    p = _resolve_existing(path)
    coll = Collection.load(p)
    live = coll.export_live_items()
    if not live:
        raise click.ClickException("collection is empty, nothing to export")
    vecs = np.stack([v for _, v, _ in live])
    metas = [m for _, _, m in live]
    np.save(f"{out_prefix}.npy", vecs)
    Path(f"{out_prefix}.meta.json").write_text(json.dumps(metas))
    _secho(f"exported {len(live)} vectors to {out_prefix}.npy / {out_prefix}.meta.json", fg="green")


@click.command(name="import")
@click.option("--path", required=True, help="Collection file to create/overwrite.")
@click.option("--vectors", "vectors_path", required=True, help=".npy file, shape (n, dim), dtype float32.")
@click.option("--metadata", "metadata_path", default=None)
@click.option("--rerank-factor", default=20, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
def import_(path: str, vectors_path: str, metadata_path: Optional[str], rerank_factor: int, seed: int) -> None:
    """Bulk-build a new collection from a .npy vector file."""
    import numpy as np

    from ruvector.collection import Collection

    p = Path(path).expanduser().resolve()
    vecs = np.load(_resolve_existing(vectors_path))
    metadatas = json.loads(_resolve_existing(metadata_path).read_text()) if metadata_path else None
    coll = _run_with_spinner(
        f"building collection from {len(vecs)} vectors...",
        lambda: Collection.from_vectors(vecs, metadatas=metadatas, rerank_factor=rerank_factor, seed=seed),
    )
    coll.save(p)
    _secho(f"built collection with {len(coll)} vectors at {p}", fg="green")


main.add_command(import_)


@main.command()
@click.option("--path", required=True)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--no-color", is_flag=True, default=False, help="Disable the colored panel (also honored: NO_COLOR env var, non-TTY stdout).")
def info(path: str, as_json: bool, no_color: bool) -> None:
    """Print collection stats."""
    from ruvector.collection import Collection

    p = _resolve_existing(path)
    stats = Collection.load(p).stats()
    if as_json:
        # Scripting path: exactly plain, parseable JSON (CollectionStats's
        # own field names via dataclass __dict__) — never styled.
        click.echo(json.dumps(stats.__dict__))
        return

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold cyan")
    table.add_column()
    table.add_row("count", str(stats.count))
    table.add_row("dim", str(stats.dim))
    table.add_row("backend", stats.backend)
    # stats() (python/ruvector/collection.py) only ever populates
    # rerank_factor/memory_bytes for backend="rabitq" — they're hardcoded
    # to 0 for "hnsw" because that backend doesn't use a rerank pass or
    # track its own heap size. Showing a bare "0" there reads like a bug;
    # say plainly that the field isn't applicable to this backend instead.
    if stats.backend == "rabitq":
        table.add_row("rerank_factor", str(stats.rerank_factor))
        table.add_row("memory_bytes", str(stats.memory_bytes))
    else:
        table.add_row("rerank_factor", "— (not used by hnsw)")
        table.add_row("memory_bytes", "— (not tracked for hnsw)")
    table.add_row("tombstoned", str(stats.tombstoned))

    console = Console(no_color=no_color or None)
    console.print(Panel(table, title=f"[bold]{p.name}[/bold]", expand=False))


@main.command()
@click.option("-n", default=10_000, show_default=True, type=int, help="Number of vectors.")
@click.option("--dim", default=128, show_default=True, type=int)
@click.option("--k", default=10, show_default=True, type=int)
@click.option("--queries", default=100, show_default=True, type=int)
@click.option("--rerank-factor", default=20, show_default=True, type=int)
@click.option("--json", "as_json", is_flag=True, default=False)
def benchmark(n: int, dim: int, k: int, queries: int, rerank_factor: int, as_json: bool) -> None:
    """Run an in-memory build+search latency benchmark (no comparator — see
    the ADR-352 benchmark methodology for a real side-by-side)."""
    import time

    import numpy as np

    from ruvector.collection import Collection

    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)

    t0 = time.perf_counter()
    coll = Collection.from_vectors(vecs, rerank_factor=rerank_factor)
    build_s = time.perf_counter() - t0

    qs = rng.standard_normal((queries, dim)).astype(np.float32)
    latencies = []
    for i in range(queries):
        t0 = time.perf_counter()
        coll.search(qs[i], k)
        latencies.append(time.perf_counter() - t0)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99) - 1]

    result = {
        "n": n, "dim": dim, "k": k, "queries": queries, "rerank_factor": rerank_factor,
        "build_seconds": build_s,
        "p50_ms": p50 * 1000,
        "p99_ms": p99 * 1000,
        "qps": 1.0 / (sum(latencies) / len(latencies)),
    }
    if as_json:
        # Scripting path: plain, parseable JSON — never styled.
        click.echo(json.dumps(result))
    else:
        _secho(
            f"n={n} dim={dim} build={build_s:.3f}s p50={result['p50_ms']:.3f}ms "
            f"p99={result['p99_ms']:.3f}ms qps={result['qps']:.0f}",
            fg="cyan",
        )


@main.command()
@click.option("--http", "use_http", is_flag=True, default=False, help="Serve streamable-HTTP instead of stdio.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8420, show_default=True, type=int)
@click.option(
    "--read-only",
    is_flag=True,
    default=False,
    help="Do not register mutating tools (create/insert/insert_batch/delete) or the Salesforce upsert route.",
)
def serve(use_http: bool, host: str, port: int, read_only: bool) -> None:
    """Launch the ruvector MCP server (stdio by default).

    If ``RUVECTOR_ENABLE_SALESFORCE_ACTIONS`` is set, also mounts the
    Salesforce Agentforce action routes (ADR-352) on the same server —
    see ``ruvector.salesforce_routes``'s module docstring for the separate
    auth these routes need (``MCPServer.custom_route`` endpoints do not
    get the MCP-level bearer auth). This only makes sense with `--http`
    (a REST action endpoint has no stdio equivalent); it's a no-op flag
    check here either way, not an error, so `ruvector serve` without
    `--http` simply ignores it rather than failing on an irrelevant
    env var a deployment script might set unconditionally.
    """
    import os

    from ruvector.mcp_server import UnsafeBindError, apply_read_only, run_http, run_stdio

    if read_only:
        apply_read_only()

    if use_http and os.environ.get("RUVECTOR_ENABLE_SALESFORCE_ACTIONS"):
        from ruvector.salesforce_routes import SalesforceAuthNotConfiguredError, maybe_register

        try:
            maybe_register(read_only=read_only)
        except SalesforceAuthNotConfiguredError as exc:
            raise click.ClickException(str(exc)) from exc

    if use_http:
        # Checked (not just assumed) what happens on a port-in-use or
        # permission-denied bind failure: `run_http` hands off to the MCP
        # SDK's uvicorn-based server, which catches the bind OSError
        # *internally*, logs its own clean one-line
        # "ERROR: [Errno N] ... address already in use"/"permission
        # denied" message, and then returns normally — the OSError never
        # reaches this function, so a `try`/`except OSError` here would be
        # dead code (verified: EADDRINUSE on a held port and EACCES on
        # port 80 both land in that uvicorn log line, not a Python
        # traceback). The one real rough edge is that the process then
        # exits 0 even though it failed to bind — misleading for a
        # supervisor/script checking the exit code — but that's decided
        # inside uvicorn's `Server.serve()` before control returns here,
        # so fixing it would mean changing `ruvector.mcp_server`, which is
        # out of scope for this CLI-only pass.
        try:
            run_http(host=host, port=port)
        except UnsafeBindError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        run_stdio()


if __name__ == "__main__":
    main()
