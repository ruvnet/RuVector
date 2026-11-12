// Ranking metrics for the KGE benchmark (ADR-003 §5, ADR-006 §protocol 2,6).
// Pure functions over plain arrays so each is unit-testable against
// hand-computed values. No I/O, no dependencies.
//
// The load-bearing decision here is tie-breaking (Sun et al. 2020,
// arXiv:1911.03903): filtered MRR/Hits depend on how candidates with an
// IDENTICAL score to the true entity are ordered. TOP (true first) inflates,
// BOTTOM (true last) deflates, RANDOM (true uniformly among the tied block) is
// the LibKGE/PyKEEN default and the only honest one. `filteredRank` implements
// all three; the harness and CI assert RANDOM.

/** Deterministic PRNG (mulberry32). Seed with an integer; returns () => [0,1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Filtered rank (1-based) of `trueEntity` among scored candidates.
 *
 * @param candidates  array of { entity, score }; higher score = more plausible.
 * @param trueEntity  the gold tail (or head) whose rank we want.
 * @param opts.filter set/array of OTHER known-true entities for this query;
 *                    removed from the candidate pool (filtered setting).
 * @param opts.tieBreak 'random' | 'top' | 'bottom'.
 * @param opts.rng    a () => [0,1) source (mulberry32) for 'random'. Required
 *                    for reproducibility; defaults to Math.random if absent.
 * @returns integer rank in [1, |candidates after filtering|].
 */
export function filteredRank(candidates, trueEntity, { filter, tieBreak = 'random', rng } = {}) {
  const filterSet = filter instanceof Set ? filter : new Set(filter ?? []);
  let trueScore;
  for (const c of candidates) {
    if (c.entity === trueEntity) {
      trueScore = c.score;
      break;
    }
  }
  if (trueScore === undefined) {
    throw new Error(`filteredRank: trueEntity ${trueEntity} not among candidates`);
  }
  let higher = 0;
  let tied = 0; // tied with the true entity, excluding the true entity itself
  for (const c of candidates) {
    if (c.entity === trueEntity) continue;
    if (filterSet.has(c.entity)) continue; // filtered: a different known-true tail
    if (c.score > trueScore) higher++;
    else if (c.score === trueScore) tied++;
  }
  switch (tieBreak) {
    case 'top':
      return higher + 1;
    case 'bottom':
      return higher + tied + 1;
    case 'random': {
      const r = rng ?? Math.random;
      // uniform position within the tied block of size (tied + 1)
      const offset = Math.floor(r() * (tied + 1));
      return higher + 1 + offset;
    }
    default:
      throw new Error(`unknown tieBreak '${tieBreak}'`);
  }
}

/** Aggregate a list of 1-based ranks into MRR, MR and Hits@{1,3,10}. */
export function rankMetrics(ranks) {
  const n = ranks.length;
  if (n === 0) return { n: 0, mrr: 0, mr: 0, hits: { 1: 0, 3: 0, 10: 0 } };
  let mrr = 0;
  let mr = 0;
  const hits = { 1: 0, 3: 0, 10: 0 };
  for (const r of ranks) {
    mrr += 1 / r;
    mr += r;
    if (r <= 1) hits[1]++;
    if (r <= 3) hits[3]++;
    if (r <= 10) hits[10]++;
  }
  return {
    n,
    mrr: mrr / n,
    mr: mr / n,
    hits: { 1: hits[1] / n, 3: hits[3] / n, 10: hits[10] / n },
  };
}

/**
 * ANN recall@k: fraction of the exhaustive top-k that the ANN top-k recovers.
 * `annList`/`exactList` are ordered entity ids (best first). recall = overlap/k.
 */
export function recallAtK(annList, exactList, k) {
  const exactTop = new Set(exactList.slice(0, k));
  if (exactTop.size === 0) return 1;
  let hit = 0;
  for (const e of annList.slice(0, k)) if (exactTop.has(e)) hit++;
  return hit / exactTop.size;
}

/** Mean recall@k over an array of { ann, exact } ranked-list pairs. */
export function meanRecallAtK(pairs, k) {
  if (pairs.length === 0) return 0;
  let s = 0;
  for (const p of pairs) s += recallAtK(p.ann, p.exact, k);
  return s / pairs.length;
}

/** Mean of a numeric array (0 for empty). */
export function mean(xs) {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/** Percentile via linear interpolation on the sorted sample (NumPy default). */
export function percentile(xs, p) {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  if (s.length === 1) return s[0];
  const rank = (p / 100) * (s.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return s[lo];
  return s[lo] + (rank - lo) * (s[hi] - s[lo]);
}

/** Latency summary from per-query wall-clock samples (ms). */
export function latency(samplesMs) {
  return {
    n: samplesMs.length,
    p50: percentile(samplesMs, 50),
    p95: percentile(samplesMs, 95),
    mean: mean(samplesMs),
  };
}

/** Throughput: count over wallMs, as a per-second rate (0 for non-positive wall). */
export function ratePerSecond(count, wallMs) {
  const s = wallMs / 1000;
  return s > 0 ? count / s : 0;
}
