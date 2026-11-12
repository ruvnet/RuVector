/**
 * `kge demo` — a self-contained link-prediction walkthrough. It trains a HolE
 * model on a tiny bundled knowledge graph (people, the cities they live in,
 * the languages they speak, and the countries those cities are in), with three
 * `citizenOf` facts held out, then asks `(person, citizenOf, ?)` for each and
 * prints the ranked tails. Nothing is read from disk and nothing is fetched:
 * the triples are compiled in.
 */

import { createKge } from '../client';
import type { Triple } from '../types';
import { CliContext, flagNumber, ParsedArgs } from './args';

const CITY_COUNTRY: Record<string, string> = {
  London: 'UK',
  Manchester: 'UK',
  Paris: 'France',
  Lyon: 'France',
  Berlin: 'Germany',
  Munich: 'Germany',
};

const LANGUAGE: Record<string, string> = { UK: 'English', France: 'French', Germany: 'German' };

const LIVES_IN: Record<string, string> = {
  Ada: 'London',
  Alan: 'Manchester',
  Tim: 'London',
  Rosalind: 'Manchester',
  Marie: 'Paris',
  Pierre: 'Lyon',
  Simone: 'Paris',
  Sophie: 'Lyon',
  Albert: 'Berlin',
  Emmy: 'Munich',
  Max: 'Berlin',
  Lise: 'Munich',
};

/** People whose `citizenOf` fact is withheld from training and predicted. */
const HELD_OUT_PEOPLE = ['Tim', 'Simone', 'Max'];

function buildTriples(): Triple[] {
  const triples: Triple[] = [];
  for (const [city, country] of Object.entries(CITY_COUNTRY)) {
    triples.push({ s: city, r: 'locatedIn', o: country });
  }
  for (const [person, city] of Object.entries(LIVES_IN)) {
    const country = CITY_COUNTRY[city];
    triples.push({ s: person, r: 'livesIn', o: city });
    triples.push({ s: person, r: 'speaks', o: LANGUAGE[country] });
    if (!HELD_OUT_PEOPLE.includes(person)) {
      triples.push({ s: person, r: 'citizenOf', o: country });
    }
  }
  return triples;
}

/** The training triples (the held-out facts are not among them). */
export const DEMO_TRIPLES: readonly Triple[] = buildTriples();

/** The withheld facts: the demo predicts `o` from `(s, r, ?)`. */
export const HELD_OUT: readonly Triple[] = HELD_OUT_PEOPLE.map((s) => ({
  s,
  r: 'citizenOf',
  o: CITY_COUNTRY[LIVES_IN[s]],
}));

/** Settings that ranked every held-out answer first across seeds 1–8, 42, 99. */
const DIMS = 64;
const SEED = 42;
const TRAIN = { epochs: 500, lr: 0.1, n3_lambda: 0.05 };

/** Scores are rounded so native and WASM builds print the same thing. */
const round = (x: number): number => Math.round(x * 100) / 100;

/** `demo [-k 3] [--json]` */
export async function runDemo(flags: ParsedArgs['flags'], ctx: CliContext): Promise<number> {
  const k = flagNumber(flags, 'k') ?? 3;
  if (!Number.isInteger(k) || k < 1 || k > 10) {
    throw new Error('-k must be an integer from 1 to 10');
  }
  const kge = createKge({ scorer: 'hole', dims: DIMS, seed: SEED, binding: ctx.binding });
  const added = kge.addTriples(DEMO_TRIPLES);
  await kge.train(TRAIN);
  const total = added.entities;

  const queries = HELD_OUT.map((h) => {
    // Rank against every entity so the answer's rank is exact, then trim to k.
    const ranked = kge.predict({ s: h.s, r: h.r, k: total, useIndex: false }).candidates;
    const rank = ranked.findIndex((c) => c.entity === h.o) + 1;
    const candidates = ranked.slice(0, k).map((c) => ({ entity: c.entity, score: round(c.score) }));
    return { s: h.s, r: h.r, answer: h.o, rank, candidates };
  });

  if (flags.json === true) {
    ctx.out(
      JSON.stringify({
        backend: kge.backend,
        scorer: 'hole',
        dims: DIMS,
        ...TRAIN,
        triples: added.triples,
        entities: added.entities,
        relations: added.relations,
        queries,
      }),
    );
    return 0;
  }

  ctx.out(
    `Trained HolE (dims ${DIMS}, ${TRAIN.epochs} epochs) on ${added.triples} triples: ` +
      `${added.entities} entities, ${added.relations} relations.`,
  );
  ctx.out(`${HELD_OUT.length} citizenOf facts were held out of training. Predicting them:`);
  for (const q of queries) {
    ctx.out('');
    ctx.out(`(${q.s}, ${q.r}, ?)  held-out answer: ${q.answer}, rank ${q.rank} of ${total}`);
    q.candidates.forEach((c, i) => {
      const mark = c.entity === q.answer ? '  <- held out' : '';
      ctx.out(`  ${i + 1}. ${c.entity.padEnd(10)} ${c.score.toFixed(2).padStart(6)}${mark}`);
    });
  }
  return 0;
}
