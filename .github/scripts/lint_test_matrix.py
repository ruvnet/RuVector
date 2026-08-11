#!/usr/bin/env python3
"""Fail if a test-matrix `packages` scalar contains a '#'.

`packages:` is a YAML folded scalar (`>-`). YAML does not recognise '#' as a
comment inside a block scalar, so any '#' written there is literal text. That
text is substituted verbatim into

    run: cargo nextest run --no-fail-fast ${{ matrix.packages }}

where bash *does* treat ' #' as starting a comment — silently discarding every
flag after it.

This is what happened to the `core-and-rest` shard between 578400d1d
(2026-06-21) and the commit that added this script: all 99 `--exclude` flags
were dropped, so the shard ran a bare `cargo nextest run --workspace`, compiled
the entire workspace including every crate the sharding exists to avoid, and
was cancelled at the 4h `timeout-minutes` cap having never started a test. It
failed this way on `main` and on every PR branch, for six weeks, while
presenting as a timeout rather than as a misconfiguration.

Nothing about that is visible in a diff or a log — the job runs, it just runs
the wrong command — so it needs a check.
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "ci.yml"


def main() -> int:
    doc = yaml.safe_load(WORKFLOW.read_text())
    include = doc["jobs"]["test"]["strategy"]["matrix"]["include"]

    failures = []
    for entry in include:
        name = entry.get("name", "<unnamed>")
        packages = entry.get("packages", "")
        if "#" not in packages:
            continue
        intended = packages.count("--exclude") + packages.count("-p ")
        surviving_text = packages.split(" #")[0]
        surviving = surviving_text.count("--exclude") + surviving_text.count("-p ")
        failures.append((name, intended, surviving, packages.index("#")))

    if not failures:
        print(f"ok: {len(include)} matrix shards, no '#' in any packages scalar")
        return 0

    print("FAIL: '#' found inside a matrix `packages` block scalar\n")
    for name, intended, surviving, at in failures:
        print(f"  shard {name!r}: '#' at offset {at}")
        print(f"    flags intended: {intended}, flags reaching cargo: {surviving}")
        if surviving < intended:
            print(f"    -> {intended - surviving} flag(s) SILENTLY DROPPED")
    print(
        "\nMove the comment above the `packages:` key, where YAML will treat it\n"
        "as a comment. Do not put '#' inside the `>-` scalar."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
