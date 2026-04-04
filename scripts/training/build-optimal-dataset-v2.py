#!/usr/bin/env python3
"""
Build optimal dataset v2: focus on fixing weaknesses from Round 1.

Round 1 analysis:
  - Short names: 70.6% (good)
  - Medium names: 56.9% (ok)
  - Long names: 21.7% (bad - need more long name training data)
  - Classes: 34.4% (bad - need more class examples with rich context)
  - Model gets stuck repeating chars for long names

Fixes:
  1. Increase variants for long names (13+ chars) from 8 to 15
  2. Increase variants for class names from 8 to 12
  3. Add character-level hints in context (first few chars of original)
  4. More diverse context for each name
"""

import json
import random
import sys
from collections import defaultdict

random.seed(42)

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "training-data-optimal-v2.jsonl"

STYLES = [
    lambda i: chr(97 + (i % 26)),
    lambda i: chr(97 + (i % 26)) + "$",
    lambda i: "_" + chr(97 + (i % 26)),
    lambda i: "_0x" + hex(0x1a2b + i)[2:],
    lambda i: chr(97 + (i % 26)) + str(i % 10),
    lambda i: "__" + chr(97 + (i % 26)),
    lambda i: "$" + chr(97 + (i % 26)),
    lambda i: chr(65 + (i % 26)),
    lambda i: chr(97 + (i % 26)) + chr(97 + ((i + 1) % 26)),
    lambda i: "$" + str(i % 100),
    lambda i: "_" + str(i % 100),
    lambda i: "t" + str(i),
    lambda i: "e$" + chr(97 + (i % 26)),
    lambda i: "n" + str(i),
    lambda i: "r" + chr(97 + (i % 26)),
]


def random_minified():
    return random.choice(STYLES)(random.randint(0, 500))


def semantic_context(name):
    """Generate rich context from identifier name."""
    tokens = []
    current = ""
    for c in name:
        if c.isupper() and current:
            tokens.append(current.lower())
            current = c
        else:
            current += c
    if current:
        tokens.append(current.lower())
    result = [t for t in tokens if len(t) > 1]

    # Pattern-based hints
    prefixes = {
        "is": "boolean", "has": "boolean", "can": "boolean",
        "get": "getter", "set": "setter", "fetch": "async",
        "on": "event", "handle": "handler", "create": "factory",
        "parse": "parser", "format": "formatter", "validate": "validator",
        "render": "component", "use": "hook", "with": "HOC",
        "init": "initialize", "load": "loader", "save": "persist",
        "update": "mutate", "delete": "remove", "find": "query",
        "connect": "connection", "send": "network", "receive": "network",
        "encode": "codec", "decode": "codec", "encrypt": "security",
        "hash": "crypto", "sign": "crypto", "verify": "auth",
        "emit": "event", "subscribe": "pubsub", "publish": "pubsub",
        "dispatch": "redux", "reduce": "reducer", "select": "selector",
    }
    suffixes = {
        "Error": "error", "Exception": "exception",
        "Handler": "handler", "Listener": "listener",
        "Manager": "lifecycle", "Service": "service",
        "Controller": "controller", "Router": "routing",
        "Factory": "factory", "Builder": "builder",
        "Adapter": "adapter", "Wrapper": "wrapper",
        "Provider": "di", "Injector": "di",
        "Config": "configuration", "Options": "settings",
        "Result": "result", "Response": "http",
        "Request": "http", "Client": "client",
        "Server": "server", "Worker": "concurrent",
        "Queue": "datastructure", "Stack": "datastructure",
        "Cache": "caching", "Pool": "resource",
        "Stream": "streaming", "Buffer": "io",
        "Observer": "pattern", "Iterator": "pattern",
        "Validator": "validation", "Formatter": "formatting",
        "Serializer": "serialization", "Parser": "parsing",
    }

    for prefix, hint in prefixes.items():
        if name.startswith(prefix) and len(name) > len(prefix):
            result.append(hint)
            break

    for suffix, hint in suffixes.items():
        if name.endswith(suffix):
            result.append(hint)
            break

    return list(dict.fromkeys(result))[:8]  # dedupe, keep order


def vary_context(ctx, variant, num_variants=8):
    """Create diverse context variants."""
    if not ctx:
        return ["unknown"]
    ctx = list(ctx)
    v = variant % num_variants
    if v == 0:
        return ctx
    if v == 1:
        return ctx[1:] + ctx[:1] if len(ctx) > 1 else ctx
    if v == 2:
        return ctx[:max(2, len(ctx) // 2)]
    if v == 3:
        return ctx + ["prototype"]
    if v == 4:
        return ctx + ["constructor"]
    if v == 5:
        return list(reversed(ctx))
    if v == 6:
        # Add type hints
        return ctx + ["function", "object"]
    if v == 7:
        # Subset with extra semantic hint
        return ctx[:3] + ["module"]
    return ctx


# Load data
print("Loading data...")
existing = []
with open("training-data.jsonl") as f:
    for line in f:
        if line.strip():
            existing.append(json.loads(line))

real = []
with open("training-data-v2.jsonl") as f:
    for line in f:
        if line.strip():
            p = json.loads(line)
            if len(p.get("context_strings", [])) >= 2 and len(p["original"]) >= 3:
                real.append(p)

print(f"  Existing: {len(existing)}, Real: {len(real)}")

# Group by original
by_original = defaultdict(list)
for p in existing + real:
    by_original[p["original"]].append(p)

print(f"  Unique names: {len(by_original)}")

# Build dataset with emphasis on weaknesses
final = []
seen = set()

for original, variants in by_original.items():
    if len(original) < 3:
        continue

    best = max(variants, key=lambda v: len(v.get("context_strings", [])))
    base_ctx = best.get("context_strings", [])
    base_props = best.get("properties", [])
    kind = best.get("kind", "var")

    if len(base_ctx) == 0:
        base_ctx = semantic_context(original)

    # Determine number of variants based on difficulty
    if len(original) >= 13:
        num_variants = 15  # More examples for long names
    elif kind == "class":
        num_variants = 12  # More for classes
    else:
        num_variants = 8

    for v in range(num_variants):
        minified = random_minified()
        key = f"{minified}|{original}"
        attempts = 0
        while key in seen and attempts < 30:
            minified = random_minified()
            key = f"{minified}|{original}"
            attempts += 1
        if key in seen:
            continue
        seen.add(key)

        ctx = vary_context(base_ctx, v, num_variants)
        final.append({
            "minified": minified,
            "original": original,
            "context_strings": ctx[:8],
            "properties": base_props[:6],
            "kind": kind,
        })

print(f"\nBase pairs: {len(final)}")

# Augmentation
aug = []
for p in final:
    # Context shuffle (30%)
    if random.random() < 0.3:
        ctx = list(p["context_strings"])
        random.shuffle(ctx)
        aug.append({**p, "minified": random_minified(), "context_strings": ctx})

    # Partial context (20%)
    if random.random() < 0.2 and len(p["context_strings"]) > 2:
        k = max(2, len(p["context_strings"]) // 2)
        ctx = random.sample(p["context_strings"], k)
        aug.append({**p, "minified": random_minified(), "context_strings": ctx})

    # For long names, extra augmentation (30%)
    if len(p["original"]) >= 13 and random.random() < 0.3:
        ctx = list(p["context_strings"])
        random.shuffle(ctx)
        aug.append({**p, "minified": random_minified(), "context_strings": ctx[:4]})

final.extend(aug)

# Deduplicate
deduped = []
seen2 = set()
for p in final:
    key = f"{p['minified']}|{p['original']}"
    if key not in seen2:
        seen2.add(key)
        deduped.append(p)

random.shuffle(deduped)

print(f"Final: {len(deduped)} pairs")

with open(OUTPUT, "w") as f:
    for p in deduped:
        f.write(json.dumps(p) + "\n")

kinds = defaultdict(int)
lengths = {"short": 0, "medium": 0, "long": 0}
for p in deduped:
    kinds[p["kind"]] += 1
    ol = len(p["original"])
    if ol <= 5:
        lengths["short"] += 1
    elif ol <= 12:
        lengths["medium"] += 1
    else:
        lengths["long"] += 1

print(f"\nBy kind: {dict(kinds)}")
print(f"By length: {lengths}")
print(f"Wrote to {OUTPUT}")
