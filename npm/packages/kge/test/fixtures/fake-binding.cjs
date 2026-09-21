'use strict';
// Deterministic fake Model binding for the bench tests and for a real synthetic
// run without the native build (KGE_BENCH_BINDING=./test/fixtures/fake-binding.cjs).
// It is NOT the scorer — it is a toy graph model that ranks candidates by a
// seeded hash with a bonus on true triples, implements the filtered RANDOM/TOP/
// BOTTOM tie-break INDEPENDENTLY of bench/lib/metrics.mjs (so the tie-check gate
// genuinely cross-checks), and honours dev knobs so a test can drive each gate:
//
//   config._annDegraded    indexed predict returns a scrambled order  -> recall < 0.9
//   config._noConfidenceDrop  triple score ignores decoy dilution     -> adversarial drop = 0
//   config._badTieBreak    evalJson always uses TOP internally         -> RANDOM tie-break FAILS
//   config.epochs === 0    constant scorer (all candidates tied)       -> the tie-check model

function hash01(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return (h >>> 0) / 4294967296;
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Model {
  constructor(configJson) {
    this.cfg = JSON.parse(configJson || '{}');
    this.epochs = typeof this.cfg.epochs === 'number' ? this.cfg.epochs : 50;
    this.seed = (this.cfg.seed ?? 42) >>> 0;
    this.entities = new Set();
    this.relations = new Set();
    this.bySplit = { train: [], valid: [], transfer: [], test: [] };
    this.trueTails = new Map(); // `${s}|${r}` -> Set(o)
    this.trueHeads = new Map(); // `${r}|${o}` -> Set(s)
    this.incident = new Map(); // `${entity}|${r}` -> count (subject OR object)
    this.nTriples = 0;
  }

  addTriplesJson(json) {
    const arr = JSON.parse(json);
    for (const t of arr) {
      const { s, r, o, split } = t;
      this.entities.add(s); this.entities.add(o); this.relations.add(r);
      if (split && this.bySplit[split]) this.bySplit[split].push({ s, r, o });
      const tk = `${s}|${r}`; if (!this.trueTails.has(tk)) this.trueTails.set(tk, new Set()); this.trueTails.get(tk).add(o);
      const hk = `${r}|${o}`; if (!this.trueHeads.has(hk)) this.trueHeads.set(hk, new Set()); this.trueHeads.get(hk).add(s);
      this.incident.set(`${s}|${r}`, (this.incident.get(`${s}|${r}`) ?? 0) + 1);
      this.incident.set(`${o}|${r}`, (this.incident.get(`${o}|${r}`) ?? 0) + 1);
      this.nTriples++;
    }
    return JSON.stringify({ added: arr.length, entities: this.entities.size, relations: this.relations.size, triples: this.nTriples });
  }

  _score(s, r, o) {
    if (this.epochs === 0) return 0; // constant scorer: everything tied
    const isTrue = this.trueTails.get(`${s}|${r}`)?.has(o);
    let v = hash01(`${s}|${r}|${o}`) + (isTrue ? 0.5 : 0);
    if (!this.cfg._noConfidenceDrop) v -= 0.01 * (this.incident.get(`${o}|${r}`) ?? 0);
    return v;
  }

  _entArray() {
    if (!this._ea || this._ea.length !== this.entities.size) this._ea = [...this.entities].sort();
    return this._ea;
  }

  // Filtered rank of `trueEnt` among all entities as the given side's candidate.
  _rank(fixed, r, trueEnt, side, tieBreak, rng) {
    const ents = this._entArray();
    const filter = side === 'tail' ? this.trueTails.get(`${fixed}|${r}`) : this.trueHeads.get(`${r}|${fixed}`);
    const trueScore = side === 'tail' ? this._score(fixed, r, trueEnt) : this._score(trueEnt, r, fixed);
    let higher = 0, tied = 0;
    for (const e of ents) {
      if (e === trueEnt) continue;
      if (filter && filter.has(e)) continue;
      const sc = side === 'tail' ? this._score(fixed, r, e) : this._score(e, r, fixed);
      if (sc > trueScore) higher++;
      else if (sc === trueScore) tied++;
    }
    const mode = this.cfg._badTieBreak ? 'top' : tieBreak;
    if (mode === 'top') return higher + 1;
    if (mode === 'bottom') return higher + tied + 1;
    return higher + 1 + Math.floor(rng() * (tied + 1)); // random
  }

  evalJson(json) {
    const { split, tieBreak = 'random' } = JSON.parse(json);
    const triples = this.bySplit[split] ?? [];
    const rng = mulberry32(this.seed ^ (tieBreak === 'top' ? 1 : tieBreak === 'bottom' ? 2 : 3));
    const metricSet = (ranks) => {
      const n = ranks.length || 1;
      let mrr = 0, mr = 0, h1 = 0, h3 = 0, h10 = 0;
      for (const rk of ranks) { mrr += 1 / rk; mr += rk; if (rk <= 1) h1++; if (rk <= 3) h3++; if (rk <= 10) h10++; }
      return { count: ranks.length, mr: mr / n, mrr: mrr / n, hits1: h1 / n, hits3: h3 / n, hits10: h10 / n };
    };
    const tailRanks = [], headRanks = [];
    for (const t of triples) {
      tailRanks.push(this._rank(t.s, t.r, t.o, 'tail', tieBreak, rng));
      headRanks.push(this._rank(t.o, t.r, t.s, 'head', tieBreak, rng));
    }
    return JSON.stringify({
      report: { combined: metricSet([...tailRanks, ...headRanks]), head: metricSet(headRanks), tail: metricSet(tailRanks) },
      evalTriples: triples.length, split, splitSource: 'ingested split tags', filtered: true,
      note: 'fake double: filtered ranking over all entities with the requested tie-break',
    });
  }

  predictJson(json) {
    const q = JSON.parse(json);
    // Real contract: exactly ONE of s/o is open — no all-three scoring call.
    if (q.s !== undefined && q.o !== undefined) {
      return JSON.stringify({ error: { kind: 'invalid', message: 'predict leaves exactly one of "s"/"o" open' } });
    }
    const ents = this._entArray();
    const open = q.o !== undefined ? 'head' : 'tail'; // if o fixed, rank heads
    const scored = ents
      .map((e) => ({ entity: e, score: open === 'tail' ? this._score(q.s, q.r, e) : this._score(e, q.r, q.o) }))
      .sort((a, b) => b.score - a.score);
    let ranked = scored;
    if (q.useIndex && this.cfg._annDegraded) {
      // scramble the indexed order so recall@10 vs exhaustive drops well below 0.9
      ranked = scored.slice().reverse();
    }
    return JSON.stringify({ ann: !!q.useIndex, exact: !q.useIndex, candidates: ranked.slice(0, q.k ?? 10) });
  }

  trainJson(json) {
    const c = JSON.parse(json || '{}');
    return JSON.stringify({
      epoch: c.epochs ?? this.epochs, epochs: c.epochs ?? this.epochs, batches: 1,
      loss: 0.5, n3Penalty: 0.001, triplesPerSec: 100000,
      triples: this.nTriples, entities: this.entities.size, relations: this.relations.size,
    });
  }
  buildIndexJson() { return JSON.stringify({ built: true }); }
  toJson() {
    // A minimal envelope so client.save() works with the fake (the CLI writes it).
    return JSON.stringify({ sha256: 'fake', model: { entities: this.entities.size, triples: this.nTriples } });
  }
  optimizeJson(json) {
    // A plausible report shape (the fake does not run a real campaign) so the
    // TS client and CLI can exercise the --receipts / --out paths without the
    // native build.
    const spec = JSON.parse(json || '{}');
    const dims = this.cfg.dims ?? 256;
    const scorer = this.cfg.scorer ?? 'hole';
    const champion = {
      dims, lr: 0.1, optimizer: 'adam', loss: 'cross-entropy',
      neg_count: 100, temperature: 1.0, n3_lambda: 0.0, epochs: 20, scorer,
    };
    const receipts = JSON.stringify({ receipt: { seq: 0, decision: { decision: 'promote' } }, kge: { fake: true } });
    return JSON.stringify({
      champion, championId: 1, promoted: true, installed: true, paused: false,
      budgetConsumed: Math.min(spec.budget ?? 16, 64), splitSource: 'per-triple',
      proposalCount: 1,
      proposals: [{ id: 1, parent: 0, arm: 'hpo', decision: { decision: 'promote' } }],
      val: { baselineMrr: 0.20, championMrr: 0.50 },
      transfer: { baselineMrr: 0.30, championMrr: 0.31 },
      test: { baselineMrr: 0.25, championMrr: 0.45 },
      scorer, dims, receiptsCount: 1, receipts,
    });
  }
  statsJson() { return JSON.stringify({ backend: 'fake', entities: this.entities.size, relations: this.relations.size, triples: this.nTriples }); }
}

module.exports = { Model, version: () => '0.1.0-fake', backend: 'fake' };
