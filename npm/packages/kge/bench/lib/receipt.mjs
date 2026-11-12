// Receipt assembly (ADR-006 §Where it lives: "one signed receipt per run";
// ADR-005: triples' text is never logged — receipts store hashes and counts,
// never entity/relation ids or descriptions). A receipt records WHAT was
// measured and against WHAT: scorer, config, dataset/split hashes, filtered
// metrics per split, the tie-break mode, ANN recall, the adversarial report,
// predict latency, train throughput, host info and the binding's statsJson.

import os from 'node:os';
import { createHash } from 'node:crypto';

const sha256hex = (s) => createHash('sha256').update(String(s)).digest('hex');

/** Host + runtime provenance for reproducibility (no secrets, no paths). */
export function hostInfo() {
  return {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    cpus: os.cpus()?.length ?? null,
    cpu_model: os.cpus()?.[0]?.model ?? null,
    os_release: os.release(),
    total_mem_mb: Math.round(os.totalmem() / 1024 / 1024),
  };
}

/**
 * Assemble the receipt. `metrics` is { split: { mrr, mr, hits, perSide } } for
 * whichever splits were scored. Optional blocks (ann, adversarial, tieCheck,
 * hardNegatives, training) are included only when present. No labels ever.
 */
export function buildReceipt({
  suite,
  scorer,
  config,
  tieBreak,
  datasetHashes,
  splitsHash,
  counts,
  metrics,
  ann,
  adversarial,
  hardNegatives,
  tieCheck,
  latency,
  training,
  gates,
  binding,
  bindingSource,
  stats,
  extra,
}) {
  return {
    schema: 'ruvector-kge-bench/receipt@1',
    generated_at: new Date().toISOString(),
    suite,
    scorer: scorer ?? null,
    config: config ?? null, // {scorer, dims, seed, epochs, ...} — no data, only hyperparams
    tie_break: tieBreak ?? null,
    binding: binding ?? null, // { version, backend } or { unavailable, error }
    binding_source: bindingSource ?? null, // injected | path | env:KGE_BENCH_BINDING
    dataset: {
      hashes: datasetHashes ?? null, // { file: sha256 } — the frozen inputs, no data
      splits_hash: splitsHash ?? null,
      counts: counts ?? null,
    },
    metrics: metrics ?? {}, // { split: { mrr, mr, hits:{1,3,10}, perSide } }
    ann: ann ?? null, // { recall_at_10, queries, withIndex/withoutIndex latency }
    adversarial: adversarial ?? null, // { targets, decoys, mean_before, mean_after, drop, targets_hash }
    hard_negatives: hardNegatives ?? null,
    tie_check: tieCheck ?? null, // { available, topMr, randomMr, bottomMr, nEntities }
    latency_ms: latency ?? null, // predict p50/p95
    training: training ?? null, // { epochs, triplesPerSec }
    gates: gates ?? null,
    stats_json: stats ?? null,
    host: hostInfo(),
    ...(extra ? { extra } : {}),
  };
}

export { sha256hex };
