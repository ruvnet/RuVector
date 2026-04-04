#!/usr/bin/env python3
"""
Build an optimal training dataset that balances learnability with diversity.

Strategy:
  1. Keep all original synthetic pairs (high signal-to-noise)
  2. Add real identifiers from node_modules but with enriched context
  3. Add many minified variants per original (the model needs to learn
     that context predicts the name, not the minified form)
  4. Ensure each original name has 5-10 variants with different minified names
  5. Target ~20K pairs for fast CPU training with good accuracy

Key insight: The model's job is context -> original_name. The minified
name is mostly noise (it's random). So we need many different minified
names mapping to the same original, with consistent context. This teaches
the model to rely on context, not the minified form.
"""

import json
import random
import sys
from collections import defaultdict

random.seed(42)

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "training-data-optimal.jsonl"

# Minifier styles
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
    i = random.randint(0, 500)
    s = random.choice(STYLES)
    return s(i)


def semantic_context(name):
    """Generate context from identifier name."""
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

    if name.startswith("is") or name.startswith("has"):
        result.append("boolean")
    if name.startswith("get") or name.startswith("fetch"):
        result.append("getter")
    if name.startswith("set"):
        result.append("setter")
    if name.startswith("on") or name.startswith("handle"):
        result.append("event")
    if name.startswith("create"):
        result.append("factory")
    if name.startswith("parse"):
        result.append("parse")
    if name.startswith("format"):
        result.append("format")
    if name.startswith("validate"):
        result.append("validate")
    if name.endswith("Error"):
        result.append("error")
    if name.endswith("Service"):
        result.append("service")
    if name.endswith("Handler"):
        result.append("handler")

    return result[:8]


def vary_context(ctx, variant):
    """Slightly vary context for training diversity."""
    if not ctx:
        return ["unknown"]
    ctx = list(ctx)
    v = variant % 6
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
        # Reverse order
        return list(reversed(ctx))
    return ctx


# Step 1: Load and analyze existing training data
print("Loading existing training data...")
existing_pairs = []
with open("training-data.jsonl") as f:
    for line in f:
        if line.strip():
            existing_pairs.append(json.loads(line))
print(f"  Existing: {len(existing_pairs)} pairs")

# Step 2: Load real identifiers from v2 data
print("Loading real identifier pairs from v2...")
real_pairs = []
with open("training-data-v2.jsonl") as f:
    for line in f:
        if line.strip():
            p = json.loads(line)
            if len(p.get("context_strings", [])) >= 2 and len(p["original"]) >= 3:
                real_pairs.append(p)
print(f"  Real pairs with good context: {len(real_pairs)}")

# Step 3: Group by original name
by_original = defaultdict(list)
for p in existing_pairs + real_pairs:
    by_original[p["original"]].append(p)
print(f"  Unique original names: {len(by_original)}")

# Step 4: Build optimal dataset
final_pairs = []
seen_keys = set()

for original, variants in by_original.items():
    if len(original) < 3:
        continue

    # Find the best context (most context strings)
    best = max(variants, key=lambda v: len(v.get("context_strings", [])))
    base_ctx = best.get("context_strings", [])
    base_props = best.get("properties", [])
    kind = best.get("kind", "var")

    if len(base_ctx) == 0:
        base_ctx = semantic_context(original)

    # Generate 8 variants
    for v in range(8):
        minified = random_minified()
        key = f"{minified}|{original}"
        attempts = 0
        while key in seen_keys and attempts < 20:
            minified = random_minified()
            key = f"{minified}|{original}"
            attempts += 1
        if key in seen_keys:
            continue
        seen_keys.add(key)

        ctx = vary_context(base_ctx, v)
        final_pairs.append({
            "minified": minified,
            "original": original,
            "context_strings": ctx[:8],
            "properties": base_props[:6],
            "kind": kind,
        })

print(f"\nGenerated {len(final_pairs)} pairs from {len(by_original)} unique names")

# Step 5: Augment
aug_pairs = []
for p in final_pairs:
    if random.random() < 0.25:
        ctx = list(p["context_strings"])
        random.shuffle(ctx)
        aug_pairs.append({
            **p, "minified": random_minified(), "context_strings": ctx,
        })
    if random.random() < 0.15 and len(p["context_strings"]) > 2:
        k = max(2, len(p["context_strings"]) // 2)
        ctx = random.sample(p["context_strings"], k)
        aug_pairs.append({
            **p, "minified": random_minified(), "context_strings": ctx,
        })

final_pairs.extend(aug_pairs)
print(f"After augmentation: {len(final_pairs)} pairs")

# Deduplicate
deduped = []
seen2 = set()
for p in final_pairs:
    key = f"{p['minified']}|{p['original']}"
    if key not in seen2:
        seen2.add(key)
        deduped.append(p)

random.shuffle(deduped)

# Cap at 25K
if len(deduped) > 25000:
    deduped = deduped[:25000]

print(f"Final: {len(deduped)} pairs")

with open(OUTPUT, "w") as f:
    for p in deduped:
        f.write(json.dumps(p) + "\n")
print(f"Wrote to {OUTPUT}")

kinds = defaultdict(int)
for p in deduped:
    kinds[p["kind"]] += 1
for k, v in sorted(kinds.items()):
    print(f"  {k}: {v}")

avg_ctx = sum(len(p["context_strings"]) for p in deduped) / len(deduped)
print(f"Avg context: {avg_ctx:.1f}")
