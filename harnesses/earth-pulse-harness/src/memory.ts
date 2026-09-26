// SPDX-License-Identifier: MIT
// Planetary memory for the Earth Pulse Observatory, backed by `agenticow`
// (ruvnet's "Git for Agent Memory" — Copy-On-Write vector branching over the
// ruVector/rvf engine). Concrete implementation of the ruVector planetary-memory
// decision in ADR-002.
//
// Why agenticow specifically: ADR-004 needs us to test a hypothesis against a
// counterfactual ocean state (a "storm week" vs a "calm week") WITHOUT mutating
// the base record of real events. agenticow's branch() forks the event memory
// in ~0.5 ms / 162 bytes regardless of base size, gives mutation isolation, and
// lets us diff()/promote() a scenario back only if it survives the gate — the
// freeze-the-physics / evolve-the-investigation pattern, in storage.
//
// The rvf engine keys vectors by i64 ids, so we map each string eventId to a
// sequential integer (shared across all branches) and carry the eventId in the
// text payload so query hits resolve back to a real event.

import { open, type AgenticMemory } from 'agenticow';
import type { EventEmbedding, PulseEvent } from './types.js';

export interface NeighborHit {
  eventId: string;
  distance: number;
  branch?: string;
}

/** Which sub-embedding to index/search (see ADR-002 — keep them separate). */
export type EmbeddingField = 'waveform' | 'environment' | 'source' | 'combined';

/** Shared id allocator so numeric ids stay consistent across base + branches. */
interface IdRegistry {
  next: number;
  byStr: Map<string, number>;
  byNum: Map<number, string>;
}

export class PlanetaryMemory {
  private constructor(
    private readonly mem: AgenticMemory,
    readonly field: EmbeddingField,
    readonly dimension: number,
    private readonly ids: IdRegistry,
  ) {}

  /** Open (or create) a planetary memory at `filePath` for a given embedding field. */
  static open(filePath: string, dimension: number, field: EmbeddingField = 'combined'): PlanetaryMemory {
    const mem = open(filePath, { dimension, metric: 'cosine' });
    return new PlanetaryMemory(mem, field, dimension, { next: 1, byStr: new Map(), byNum: new Map() });
  }

  private numericId(eventId: string): number {
    let n = this.ids.byStr.get(eventId);
    if (n === undefined) {
      n = this.ids.next++;
      this.ids.byStr.set(eventId, n);
      this.ids.byNum.set(n, eventId);
    }
    return n;
  }

  private vecOf(emb: EventEmbedding): number[] {
    return emb[this.field];
  }

  /** Ingest one event's embedding, keyed by its eventId, with a text summary. */
  ingestEvent(event: PulseEvent, emb: EventEmbedding): { accepted: number; rejected: number } {
    const vector = this.vecOf(emb);
    if (vector.length !== this.dimension) {
      throw new Error(
        `PlanetaryMemory: ${this.field} embedding has dim ${vector.length}, expected ${this.dimension}`,
      );
    }
    const id = this.numericId(event.eventId);
    const text = `${event.eventId} ${event.dominantPeriodS.toFixed(2)}s amp=${event.amplitude.toExponential(2)} ${event.sourceRegion}${event.glideDetected ? ' glide' : ''}`;
    const res = this.mem.ingest([{ id, vector, text }]);
    return { accepted: res.accepted, rejected: res.rejected };
  }

  /** Batch ingest. */
  ingestMany(items: { event: PulseEvent; emb: EventEmbedding }[]): { accepted: number; rejected: number } {
    let accepted = 0;
    let rejected = 0;
    for (const it of items) {
      const r = this.ingestEvent(it.event, it.emb);
      accepted += r.accepted;
      rejected += r.rejected;
    }
    return { accepted, rejected };
  }

  /** Nearest historical analogs to a query embedding (causal-neighbor search). */
  nearest(emb: EventEmbedding, k = 5): NeighborHit[] {
    const hits = this.mem.query(this.vecOf(emb), k) as { id: number; distance: number; branch?: string }[];
    return hits.map((h) => ({
      eventId: this.ids.byNum.get(h.id) ?? String(h.id),
      distance: h.distance,
      branch: h.branch,
    }));
  }

  /**
   * Branch the memory for a counterfactual scenario (e.g. "storm-week" vs
   * "calm-week"). COW-isolated: writes to the branch do not touch the base.
   * Shares the id registry so hits from either side resolve to real eventIds.
   */
  branch(label: string): PlanetaryMemory {
    const child = this.mem.branch(label) as AgenticMemory;
    return new PlanetaryMemory(child, this.field, this.dimension, this.ids);
  }

  /** Changes in this branch relative to its parent snapshot. */
  diff(): unknown {
    return this.mem.diff();
  }

  /** Status: counts, epoch, lineage depth. */
  status(): unknown {
    return this.mem.status();
  }

  /** Tag the current state so it can be rolled back to. */
  checkpoint(label: string): unknown {
    return this.mem.checkpoint(label);
  }

  close(): void {
    this.mem.close();
  }
}
