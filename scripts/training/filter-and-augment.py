#!/usr/bin/env python3
"""
Filter training data for quality and add augmentation.
Aims for ~30-50K high-quality, diverse pairs that train well on CPU.

Steps:
  1. Deduplicate by (original, context_hash) to remove near-duplicates
  2. Filter out low-quality pairs (no context, too-short names)
  3. Balance by kind (function/class/var)
  4. Augment with context shuffling, partial context, case variants
  5. Output filtered+augmented JSONL
"""

import json
import hashlib
import random
import sys
from collections import defaultdict

INPUT = sys.argv[1] if len(sys.argv) > 1 else "training-data-v2.jsonl"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "training-data-v2-filtered.jsonl"
TARGET_SIZE = 40000

random.seed(42)

# Load all pairs
print(f"Loading {INPUT}...")
pairs = []
with open(INPUT) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        pairs.append(json.loads(line))

print(f"Loaded {len(pairs)} pairs")

# Step 1: Quality filter
quality_pairs = []
for p in pairs:
    original = p["original"]
    ctx = p.get("context_strings", [])

    # Skip very short original names (not useful for learning)
    if len(original) < 3:
        continue

    # Skip if no context at all
    if len(ctx) == 0:
        continue

    # Skip names that are likely not real identifiers
    if not original[0].isalpha() and original[0] not in ("_", "$"):
        continue

    # Skip all-uppercase (likely constants, less interesting)
    if original.isupper() and len(original) > 5:
        continue

    quality_pairs.append(p)

print(f"After quality filter: {len(quality_pairs)}")

# Step 2: Deduplicate by original name - keep max 8 variants per name
by_original = defaultdict(list)
for p in quality_pairs:
    by_original[p["original"]].append(p)

deduped = []
for original, variants in by_original.items():
    # Keep up to 8 most diverse variants (by context)
    random.shuffle(variants)
    deduped.extend(variants[:8])

print(f"After dedup (max 8 per name): {len(deduped)}")

# Step 3: Balance by kind
by_kind = defaultdict(list)
for p in deduped:
    by_kind[p["kind"]].append(p)

print("By kind before balancing:")
for k, v in sorted(by_kind.items()):
    print(f"  {k}: {len(v)}")

# Cap each kind to prevent overwhelming dominance
max_per_kind = TARGET_SIZE // 2  # Allow some imbalance
balanced = []
for kind, items in by_kind.items():
    random.shuffle(items)
    balanced.extend(items[:max_per_kind])

random.shuffle(balanced)
print(f"After balancing: {len(balanced)}")

# Step 4: Augmentation
augmented = list(balanced)


def shuffle_context(p):
    """Shuffle context strings order."""
    ctx = list(p["context_strings"])
    random.shuffle(ctx)
    return {**p, "context_strings": ctx,
            "minified": random_minified()}


def partial_context(p):
    """Drop some context strings."""
    ctx = p["context_strings"]
    if len(ctx) <= 1:
        return None
    # Keep 50-80% of context
    keep = max(1, int(len(ctx) * random.uniform(0.5, 0.8)))
    new_ctx = random.sample(ctx, keep)
    return {**p, "context_strings": new_ctx,
            "minified": random_minified()}


def case_variant(p):
    """Generate case variant of the original name."""
    original = p["original"]
    variants = []

    # camelCase -> snake_case
    snake = ""
    for i, c in enumerate(original):
        if c.isupper() and i > 0:
            snake += "_" + c.lower()
        else:
            snake += c.lower()
    if snake != original and len(snake) > 3:
        variants.append(snake)

    # camelCase -> PascalCase
    pascal = original[0].upper() + original[1:]
    if pascal != original:
        variants.append(pascal)

    if not variants:
        return None

    chosen = random.choice(variants)
    return {**p, "original": chosen, "minified": random_minified()}


MINIFIED_CHARS = "abcdefghijklmnopqrstuvwxyz"

def random_minified():
    style = random.randint(0, 7)
    i = random.randint(0, 200)
    if style == 0:
        return MINIFIED_CHARS[i % 26]
    elif style == 1:
        return MINIFIED_CHARS[i % 26] + str(i % 10)
    elif style == 2:
        return "_" + MINIFIED_CHARS[i % 26]
    elif style == 3:
        return "_0x" + hex(0x1a2b + i)[2:]
    elif style == 4:
        return "$" + MINIFIED_CHARS[i % 26]
    elif style == 5:
        return "t" + str(i)
    elif style == 6:
        a = MINIFIED_CHARS[i % 26]
        b = MINIFIED_CHARS[(i + 1) % 26]
        return a + b
    else:
        return "n" + str(i)


# Generate augmented pairs
aug_count = 0
for p in balanced:
    # 30% chance of context shuffle augmentation
    if random.random() < 0.3:
        aug = shuffle_context(p)
        augmented.append(aug)
        aug_count += 1

    # 20% chance of partial context
    if random.random() < 0.2:
        aug = partial_context(p)
        if aug:
            augmented.append(aug)
            aug_count += 1

    # 10% chance of case variant
    if random.random() < 0.1:
        aug = case_variant(p)
        if aug:
            augmented.append(aug)
            aug_count += 1

print(f"Augmented pairs added: {aug_count}")

# Final dedup
seen = set()
final = []
for p in augmented:
    key = f"{p['minified']}|{p['original']}"
    if key not in seen:
        seen.add(key)
        final.append(p)

random.shuffle(final)

# Trim to target size if too large
if len(final) > TARGET_SIZE * 1.5:
    final = final[:int(TARGET_SIZE * 1.5)]

print(f"\nFinal dataset: {len(final)} pairs")

# Write output
with open(OUTPUT, "w") as f:
    for p in final:
        f.write(json.dumps(p) + "\n")

print(f"Wrote to {OUTPUT}")

# Stats
kinds = defaultdict(int)
for p in final:
    kinds[p["kind"]] += 1

print("\nFinal breakdown:")
for k, v in sorted(kinds.items()):
    print(f"  {k}: {v}")

avg_ctx = sum(len(p["context_strings"]) for p in final) / len(final)
avg_orig_len = sum(len(p["original"]) for p in final) / len(final)
print(f"Avg context strings: {avg_ctx:.1f}")
print(f"Avg original name length: {avg_orig_len:.1f}")
