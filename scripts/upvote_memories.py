#!/usr/bin/env python3
"""Batch upvote under-voted high-quality memories on pi.ruv.io brain."""

import json
import sys
import time
import urllib.request
import urllib.error

BASE = "https://pi.ruv.io/v1"
AUTH = "Bearer ruvector-swarm"
BATCH_SIZE = 100
TARGET_VOTES = 1500  # vote on all under-voted memories


def api_get(path):
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": AUTH})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def api_post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Authorization": AUTH, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def is_under_voted(qs):
    """A memory is under-voted if alpha + beta <= 2.0 (the prior, no observations)."""
    alpha = qs.get("alpha", 1.0)
    beta = qs.get("beta", 1.0)
    return (alpha + beta) <= 2.05  # small epsilon for float comparison


def main():
    # Get total count
    info = api_get("/memories/list?limit=1")
    total = info["total_count"]
    print(f"Total memories: {total}")

    under_voted_ids = []
    already_voted = 0
    offset = 0

    # Paginate and collect under-voted memory IDs
    while offset < total:
        batch = api_get(f"/memories/list?limit={BATCH_SIZE}&offset={offset}")
        memories = batch.get("memories", [])
        if not memories:
            break
        for m in memories:
            qs = m.get("quality_score", {"alpha": 1.0, "beta": 1.0})
            if is_under_voted(qs):
                under_voted_ids.append(m["id"])
            else:
                already_voted += 1
        offset += BATCH_SIZE
        sys.stdout.write(f"\rScanned {offset}/{total} — found {len(under_voted_ids)} under-voted, {already_voted} already voted")
        sys.stdout.flush()

    print(f"\n\nScan complete: {len(under_voted_ids)} under-voted, {already_voted} already voted")
    print(f"Current vote coverage: {already_voted}/{total} = {100*already_voted/total:.1f}%")

    # Upvote up to TARGET_VOTES
    to_vote = under_voted_ids[:TARGET_VOTES]
    print(f"\nUpvoting {len(to_vote)} memories...")

    success = 0
    errors = 0
    for i, mid in enumerate(to_vote):
        try:
            api_post(f"/memories/{mid}/vote", {"direction": "up"})
            success += 1
        except urllib.error.HTTPError as e:
            errors += 1
            if errors <= 3:
                print(f"\n  Error on {mid}: HTTP {e.code}")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"\n  Error on {mid}: {e}")

        if (i + 1) % 25 == 0:
            sys.stdout.write(f"\r  Voted: {success}/{i+1} (errors: {errors})")
            sys.stdout.flush()

    print(f"\n\nDone! Upvoted {success} memories ({errors} errors)")
    new_voted = already_voted + success
    print(f"New vote coverage: {new_voted}/{total} = {100*new_voted/total:.1f}%")


if __name__ == "__main__":
    main()
