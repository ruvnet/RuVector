// Adversarial symmetry-decoy generator (ADR-005 threat 1; ADR-006 §protocol 5).
// A JS re-implementation, for the harness's OWN use, of the symmetry-pattern
// poisoning attack (Bhardwaj et al., ACL-IJCNLP 2021, arXiv:2111.06345): decoy
// triples added elsewhere in the graph indirectly shift confidence on a target
// fact. Symmetry patterns generalize across every model/dataset the paper
// tested, which is why they are the CI regression vector.
//
// Simplification vs the paper: the paper selects decoy entities by a
// soft-truth / influence ranking; here decoy tails are chosen deterministically
// among entities not already linked to the anchor (selection simplified,
// pattern preserved). The gate is that mean confidence on the targeted facts
// DROPS under attack (ADR-006), not that the fact flips.

import { mulberry32 } from './metrics.mjs';

const triKey = (s, r, o) => `${s} ${r} ${o}`;

/**
 * Detect (near-)symmetric relations by reciprocal fraction: the share of
 * (s,r,o) triples (s≠o) whose reverse (o,r,s) is also present. A fully
 * reciprocal relation scores 1.0; a directed one scores ~0.
 *
 * @param triples  array of {s,r,o}
 * @param opts.threshold reciprocal fraction to call a relation symmetric (0.8)
 * @param opts.minCount  ignore relations with fewer than this many triples (10)
 * @returns array of { relation, fraction, count } sorted by fraction desc
 */
export function detectSymmetricRelations(triples, { threshold = 0.8, minCount = 10 } = {}) {
  const present = new Set(triples.map((t) => triKey(t.s, t.r, t.o)));
  const total = new Map();
  const recip = new Map();
  for (const t of triples) {
    if (t.s === t.o) continue;
    total.set(t.r, (total.get(t.r) ?? 0) + 1);
    if (present.has(triKey(t.o, t.r, t.s))) recip.set(t.r, (recip.get(t.r) ?? 0) + 1);
  }
  const out = [];
  for (const [r, n] of total) {
    if (n < minCount) continue;
    const fraction = (recip.get(r) ?? 0) / n;
    out.push({ relation: r, fraction, count: n, symmetric: fraction >= threshold });
  }
  return out.sort((a, b) => b.fraction - a.fraction);
}

/**
 * Detect inverse relation pairs by the share of (s,r1,o) whose (o,r2,s) exists.
 * Reported informationally in the receipt; not gated.
 * @returns array of { relation, inverse, fraction, count }
 */
export function detectInversePairs(triples, { threshold = 0.8, minCount = 10 } = {}) {
  const present = new Set(triples.map((t) => triKey(t.s, t.r, t.o)));
  const byRel = new Map();
  for (const t of triples) {
    if (t.s === t.o) continue;
    if (!byRel.has(t.r)) byRel.set(t.r, []);
    byRel.get(t.r).push(t);
  }
  const rels = [...byRel.keys()];
  const out = [];
  for (const r1 of rels) {
    const items = byRel.get(r1);
    const hits = new Map();
    for (const t of items) {
      for (const r2 of rels) {
        if (r2 === r1) continue;
        if (present.has(triKey(t.o, r2, t.s))) hits.set(r2, (hits.get(r2) ?? 0) + 1);
      }
    }
    for (const [r2, h] of hits) {
      const fraction = h / items.length;
      if (fraction >= threshold && items.length >= minCount) {
        out.push({ relation: r1, inverse: r2, fraction, count: items.length });
      }
    }
  }
  return out;
}

/**
 * Pick target facts to attack: test triples under a symmetric relation (the
 * generalizing vector). Deterministic: sorted by triple key, first `count`.
 */
export function pickTargets(testTriples, symmetricRelations, count = 20) {
  const symSet = new Set(symmetricRelations.filter((r) => r.symmetric).map((r) => r.relation));
  const eligible = testTriples
    .filter((t) => symSet.has(t.r) && t.s !== t.o)
    .sort((a, b) => triKey(a.s, a.r, a.o).localeCompare(triKey(b.s, b.r, b.o)));
  return eligible.slice(0, count);
}

/**
 * Generate symmetry decoys for the targets. For target (s,r,o) emit
 * `perTarget` decoys (o, r, s′) with s′ chosen among entities not already an
 * r-tail of o (and ≠ o, s), deterministically. Never emits a triple already in
 * `existing`.
 *
 * @param targets    array of {s,r,o}
 * @param entities   full entity id list
 * @param existing   array of {s,r,o} already in the graph
 * @param opts.perTarget decoys per target (default 3)
 * @param opts.seed  PRNG seed (default 1337)
 * @returns array of {s,r,o} decoy triples (deduped, disjoint from existing)
 */
export function generateSymmetryDecoys(targets, entities, existing, { perTarget = 3, seed = 1337 } = {}) {
  const present = new Set(existing.map((t) => triKey(t.s, t.r, t.o)));
  const tailsOf = new Map(); // `${o}|${r}` -> Set of existing tails
  for (const t of existing) {
    const key = `${t.o}|${t.r}`;
    if (!tailsOf.has(key)) tailsOf.set(key, new Set());
    tailsOf.get(key).add(t.s); // subjects that already point r->o
  }
  const rng = mulberry32(seed);
  const decoys = [];
  for (const t of targets) {
    const linked = tailsOf.get(`${t.o}|${t.r}`) ?? new Set();
    let made = 0;
    let attempts = 0;
    while (made < perTarget && attempts < entities.length * 2) {
      attempts++;
      const sPrime = entities[Math.floor(rng() * entities.length)];
      if (sPrime === t.o || sPrime === t.s) continue;
      if (linked.has(sPrime)) continue;
      const k = triKey(t.o, t.r, sPrime);
      if (present.has(k)) continue;
      present.add(k);
      linked.add(sPrime);
      decoys.push({ s: t.o, r: t.r, o: sPrime });
      made++;
    }
  }
  return decoys;
}
