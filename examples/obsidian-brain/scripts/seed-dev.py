#!/usr/bin/env python3
"""Populate a dev vault + the local brain in one shot.

Responsibilities
----------------
1. POST each markdown file to the local brain and record the mapping
   between file path, content hash, and memory id so the plugin's
   `indexState` is *complete* — click-to-open works without running
   Bulk-sync first.
2. Surface AIDefence rejections (HTTP 422) as explicit log lines; the
   shell seed loop silently treated them as successes.
3. Optionally pull a slice of pi.ruv.io memories into the local brain
   as first-class notes tagged with `#brain/pi-<category>`.
4. Write the plugin's data.json with the full mapping + populated
   per-category seed counts.
5. Write .obsidian/graph.json color groups for every category seen.

Invocation
----------
    python3 scripts/seed-dev.py VAULT_DIR BRAIN_URL
        [--pi-url https://pi.ruv.io] [--pi-token $TOK] [--pi-limit 30]
        [--pi-query 'hnsw diskann']

Exits nonzero on infrastructure errors (brain unreachable). AIDefence
rejections are not errors.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import pathlib
import re
import sys
import urllib.request
import urllib.error
from typing import Any

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def strip_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw.lstrip()
    body = raw[m.end() :].lstrip()
    fm: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            fm[key] = [x.strip().strip('"\'') for x in val[1:-1].split(",") if x.strip()]
        elif (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            fm[key] = val[1:-1]
        else:
            fm[key] = val
    return fm, body


def hash_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def http(
    method: str, url: str, body: dict | None = None, headers: dict | None = None
) -> tuple[int, dict | list | None, str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            return resp.status, parsed, text
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return e.code, parsed, text


def seed_local_notes(
    vault: pathlib.Path,
    brain_url: str,
    index_state: dict[str, dict],
    categories: set[str],
) -> tuple[int, int]:
    indexed = 0
    blocked = 0
    for md in sorted(vault.glob("*.md")):
        raw = md.read_text(encoding="utf-8")
        fm, body = strip_frontmatter(raw)
        body = body.strip()
        if not body:
            continue
        category = str(
            fm.get("brain-category") or fm.get("category") or "obsidian"
        ).strip() or "obsidian"
        categories.add(category)
        status, parsed, text = http(
            "POST",
            f"{brain_url.rstrip('/')}/memories",
            {"category": category, "content": body},
        )
        if status == 201 and isinstance(parsed, dict):
            mem_id = parsed["id"]
            content_hash = parsed["content_hash"]
            rel_path = str(md.relative_to(vault))
            index_state["pathToHash"][rel_path] = content_hash
            index_state["hashToId"][content_hash] = mem_id
            index_state["idToPath"][mem_id] = rel_path
            print(f"   ✓ {md.stem} → {category}  ({mem_id[:12]})")
            indexed += 1
        elif status == 422 and isinstance(parsed, dict):
            threats_raw = parsed.get("threats", [])
            labels: list[str] = []
            for t in threats_raw:
                if isinstance(t, str):
                    labels.append(t)
                elif isinstance(t, dict):
                    labels.append(str(t.get("pattern") or t.get("category") or t))
                else:
                    labels.append(str(t))
            print(
                f"   ✗ {md.stem} BLOCKED by AIDefence "
                f"({parsed.get('threat_level', '?')}: {', '.join(labels) or '?'})"
            )
            blocked += 1
        else:
            print(f"   ! {md.stem} HTTP {status}: {text[:200]}")
    return indexed, blocked


def pull_pi(
    brain_url: str,
    index_state: dict[str, dict],
    pi_url: str,
    pi_token: str,
    limit: int,
    query: str | None,
    vault: pathlib.Path,
    categories: set[str],
) -> tuple[int, int, int]:
    """Pull N memories from pi.ruv.io and POST them into the local brain.

    Writes a companion stub note under Brain/Pi/<id>.md for each pulled
    memory so the graph has real nodes to show.
    """
    pulled: list[dict] = []
    if query:
        url = f"{pi_url.rstrip('/')}/v1/memories/search?q={urllib.parse.quote(query)}&limit={limit}"
    else:
        url = f"{pi_url.rstrip('/')}/v1/memories/list?limit={limit}"
    status, parsed, text = http(
        "GET", url, None, {"Authorization": f"Bearer {pi_token}"}
    )
    if status != 200:
        print(f"   ! pi.ruv.io {url} → {status}: {text[:200]}", file=sys.stderr)
        return 0, 0, 0
    if isinstance(parsed, dict) and "memories" in parsed:
        pulled = parsed["memories"]
    elif isinstance(parsed, list):
        pulled = parsed

    pi_dir = vault / "Brain" / "Pi"
    pi_dir.mkdir(parents=True, exist_ok=True)

    indexed = 0
    blocked = 0
    for mem in pulled:
        title = str(mem.get("title") or mem.get("id", "")).strip() or "untitled"
        content = str(mem.get("content", "")).strip()
        if not content:
            continue
        orig_cat = str(mem.get("category") or "pi").strip() or "pi"
        category = f"pi-{orig_cat}"
        categories.add(category)
        status, parsed, text = http(
            "POST",
            f"{brain_url.rstrip('/')}/memories",
            {"category": category, "content": content},
        )
        if status == 201 and isinstance(parsed, dict):
            mem_id = parsed["id"]
            content_hash = parsed["content_hash"]
            # Write a stub note so the user sees the content in the vault.
            safe_name = re.sub(r"[^A-Za-z0-9 _.-]+", "_", title)[:80] or mem_id
            stub = pi_dir / f"{safe_name}.md"
            tags_list = mem.get("tags") or []
            fm_tags = ",".join([f"pi/{orig_cat}"] + [str(t) for t in tags_list[:5]])
            stub.write_text(
                "---\n"
                f"brain-category: {category}\n"
                f"tags: [brain/{category}, {fm_tags}]\n"
                f'pi-id: "{mem["id"]}"\n'
                f'pi-source: "{pi_url}"\n'
                "---\n\n"
                f"# {title}\n\n"
                f"{content}\n",
                encoding="utf-8",
            )
            rel_path = str(stub.relative_to(vault))
            index_state["pathToHash"][rel_path] = content_hash
            index_state["hashToId"][content_hash] = mem_id
            index_state["idToPath"][mem_id] = rel_path
            print(f"   ✓ pi:{title[:50]} → {category}")
            indexed += 1
        elif status == 422:
            print(f"   ✗ pi:{title[:50]} BLOCKED (AIDefence)")
            blocked += 1
        else:
            print(f"   ! pi:{title[:50]} HTTP {status}")
    return indexed, blocked, len(pulled)


def hash_rgb(cat: str) -> int:
    h = 0
    for ch in cat:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360.0, 0.55, 0.55)
    return (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)


def write_graph_json(vault: pathlib.Path, categories: set[str]) -> int:
    gpath = vault / ".obsidian" / "graph.json"
    existing: dict[str, Any] = {}
    if gpath.exists():
        try:
            existing = json.loads(gpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    prior = [
        g
        for g in existing.get("colorGroups", [])
        if not (
            isinstance(g.get("query"), str)
            and g["query"].startswith("tag:#brain/")
        )
    ]
    groups = [
        {"query": f"tag:#brain/{c}", "color": {"a": 1, "rgb": hash_rgb(c)}}
        for c in sorted(categories)
    ]
    existing["colorGroups"] = prior + groups
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return len(groups)


def write_plugin_data(
    plugin_dir: pathlib.Path, settings: dict[str, Any], index_state: dict[str, dict]
) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    data_path = plugin_dir / "data.json"
    if data_path.exists():
        try:
            existing = json.loads(data_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    merged_settings = {**(existing.get("settings") or {}), **settings}
    data_path.write_text(
        json.dumps(
            {"settings": merged_settings, "indexState": index_state},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vault", type=pathlib.Path)
    ap.add_argument("brain_url")
    ap.add_argument("embedder_url")
    ap.add_argument("--pi-url", default="https://pi.ruv.io")
    ap.add_argument("--pi-token", default=os.environ.get("BRAIN_API_KEY", ""))
    ap.add_argument("--pi-limit", type=int, default=20)
    ap.add_argument("--pi-query", default="")
    args = ap.parse_args()

    vault: pathlib.Path = args.vault.resolve()
    if not vault.exists():
        print(f"vault does not exist: {vault}", file=sys.stderr)
        return 2

    # Sanity check: the brain should already be reachable.
    status, _, _ = http("GET", f"{args.brain_url.rstrip('/')}/health")
    if status != 200:
        print(f"brain unreachable at {args.brain_url}: {status}", file=sys.stderr)
        return 2

    # Dim-mismatch guard: DiskANN can't handle mixed-dimension vectors.
    # If the brain already has content at a different dim than the
    # embedder we're about to use, bail loudly rather than silently
    # corrupt the index.
    status, stats, _ = http(
        "GET", f"{args.brain_url.rstrip('/')}/brain/index_stats"
    )
    status2, eh, _ = http("GET", f"{args.embedder_url.rstrip('/')}/health")
    if (
        status == 200
        and isinstance(stats, dict)
        and int(stats.get("indexed_count") or 0) > 0
        and int(stats.get("dim") or 0) > 0
        and status2 == 200
        and isinstance(eh, dict)
        and int(eh.get("dim") or 0) > 0
        and int(stats["dim"]) != int(eh["dim"])
    ):
        print(
            f"refusing to seed: brain holds dim={stats['dim']} vectors but "
            f"embedder is dim={eh['dim']}. Wipe the brain data dir and "
            f"retry.",
            file=sys.stderr,
        )
        return 3

    index_state = {
        "pathToHash": {},
        "hashToId": {},
        "idToPath": {},
        "lastSync": 0,
    }
    categories: set[str] = set()

    print("==> seeding local brain from vault notes")
    indexed, blocked = seed_local_notes(vault, args.brain_url, index_state, categories)
    print(f"==> local: {indexed} indexed, {blocked} blocked by AIDefence")

    pi_pulled = 0
    pi_blocked = 0
    pi_total = 0
    if args.pi_token:
        print(f"==> pulling pi.ruv.io memories (limit={args.pi_limit})")
        pi_pulled, pi_blocked, pi_total = pull_pi(
            args.brain_url,
            index_state,
            args.pi_url,
            args.pi_token,
            args.pi_limit,
            args.pi_query or None,
            vault,
            categories,
        )
        print(
            f"==> pi.ruv.io: pulled {pi_pulled}/{pi_total}"
            + (f", {pi_blocked} blocked" if pi_blocked else "")
        )
    else:
        print("==> pi.ruv.io: skipped (no --pi-token / BRAIN_API_KEY)")

    groups = write_graph_json(vault, categories)
    print(f"==> wrote {groups} graph color groups")

    plugin_dir = vault / ".obsidian" / "plugins" / "obsidian-brain"
    import time

    index_state["lastSync"] = int(time.time())
    write_plugin_data(
        plugin_dir,
        {
            "brainUrl": args.brain_url,
            "embedderUrl": args.embedder_url,
            "defaultCategory": "obsidian",
            "autoIndex": True,
            "autoIndexDebounceMs": 3000,
            "indexMinChars": 20,
            "enableAIDefence": True,
            "searchLimit": 8,
            "relatedLimit": 8,
            "bulkSyncBatchSize": 16,
            "bulkSyncIncludeFolders": "",
            "bulkSyncExcludeFolders": ".obsidian,.trash,.brain-data,.obsidian-home",
            "storeMapping": {},
            "dpoDirection": "quality",
            "piUrl": args.pi_url,
            "piToken": args.pi_token,
            "piPullLimit": args.pi_limit,
            "piPullQuery": args.pi_query,
        },
        index_state,
    )
    print(
        f"==> wrote plugin data.json with {len(index_state['pathToHash'])} path→id mappings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
