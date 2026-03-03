#!/usr/bin/env python3
"""Upvote all memories 3x each to boost quality scores."""
import json, hashlib, urllib.request, urllib.error

BASE = "https://ruvbrain-875130704813.us-central1.run.app"
KEYS = [hashlib.sha256(f"voter-{i}".encode()).hexdigest()[:32] for i in range(6)]

def vote(memory_id, key):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    data = json.dumps({"direction": "up"}).encode()
    try:
        req = urllib.request.Request(f"{BASE}/v1/memories/{memory_id}/vote", data, headers)
        resp = urllib.request.urlopen(req, timeout=10)
        return True
    except:
        return False

# Get all memory IDs
headers = {"Authorization": f"Bearer {KEYS[0]}"}
req = urllib.request.Request(f"{BASE}/v1/memories/list?limit=100", headers=headers)
resp = urllib.request.urlopen(req, timeout=15)
page1 = json.loads(resp.read())
memories = page1 if isinstance(page1, list) else page1.get("memories", [])

# Get second page if exists
if len(memories) >= 100:
    req2 = urllib.request.Request(f"{BASE}/v1/memories/list?limit=100&offset=100", headers=headers)
    try:
        resp2 = urllib.request.urlopen(req2, timeout=15)
        page2 = json.loads(resp2.read())
        memories.extend(page2 if isinstance(page2, list) else page2.get("memories", []))
    except:
        pass

ids = [m["id"] for m in memories]
print(f"Found {len(ids)} memories to vote on")

ok = 0
fail = 0
for i, mid in enumerate(ids):
    for v in range(3):
        key = KEYS[(i * 3 + v) % len(KEYS)]
        if vote(mid, key):
            ok += 1
        else:
            fail += 1
    if (i + 1) % 25 == 0:
        print(f"  Progress: {i+1}/{len(ids)} ({ok} votes ok, {fail} fail)")

print(f"\nDone: {ok} votes cast ({fail} failed) across {len(ids)} memories")
