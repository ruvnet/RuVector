#!/usr/bin/env node
/**
 * Deterministic text-corpus generator for the ADR-210 D2 recall benchmark
 * (acceptance gate 8).
 *
 * Constructs short real-English sentences from fixed word lists with a
 * seeded PRNG — no scraping, no licensed text, fully reproducible: running
 * this script always regenerates tests/fixtures/text-corpus.json
 * byte-identically. Sentences are grouped into ten topics so the embedded
 * corpus has the clustered, low-intrinsic-dimensionality structure of real
 * text (the regime the ADR-210 ANN floors must be measured in, as opposed
 * to the uniform-random worst case they were originally tuned on).
 *
 *   node tests/fixtures/generate-text-corpus.mjs
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

export const SEED = 0x210aD; // ADR-210
export const CORPUS_SIZE = 600; // 60 sentences per topic
export const QUERY_COUNT = 60; //  6 queries per topic

/** mulberry32 — tiny deterministic PRNG. */
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

const TOPICS = [
  {
    name: 'cooking',
    adjectives: ['fresh', 'spicy', 'roasted', 'tender', 'savory', 'crisp'],
    nouns: ['chef', 'baker', 'cook', 'sous chef', 'caterer', 'apprentice'],
    verbs: ['simmered', 'whisked', 'seasoned', 'grilled', 'kneaded', 'tasted'],
    objects: ['the tomato sauce', 'the sourdough loaf', 'the lamb stew', 'the lemon custard', 'the garlic broth', 'the herb butter'],
    places: ['in the busy kitchen', 'over a low flame', 'at the market stall', 'before the dinner rush', 'in a copper pan', 'beside the wood oven'],
  },
  {
    name: 'astronomy',
    adjectives: ['distant', 'bright', 'ancient', 'faint', 'massive', 'icy'],
    nouns: ['astronomer', 'telescope operator', 'stargazer', 'researcher', 'observatory team', 'student'],
    verbs: ['observed', 'tracked', 'photographed', 'measured', 'charted', 'discovered'],
    objects: ['the spiral galaxy', 'a passing comet', 'the lunar eclipse', 'a binary star', 'the meteor shower', 'a distant nebula'],
    places: ['through the telescope', 'from the mountain observatory', 'across the night sky', 'near the horizon', 'during the eclipse', 'in the southern sky'],
  },
  {
    name: 'finance',
    adjectives: ['volatile', 'steady', 'quarterly', 'rising', 'undervalued', 'risky'],
    nouns: ['investor', 'analyst', 'trader', 'fund manager', 'accountant', 'economist'],
    verbs: ['forecast', 'audited', 'hedged', 'rebalanced', 'reported', 'projected'],
    objects: ['the bond portfolio', 'the earnings report', 'the interest rate', 'the stock index', 'the annual budget', 'the currency exchange'],
    places: ['on the trading floor', 'before the market opened', 'during the board meeting', 'in the quarterly review', 'after the announcement', 'across global markets'],
  },
  {
    name: 'sports',
    adjectives: ['fast', 'tired', 'determined', 'young', 'veteran', 'agile'],
    nouns: ['striker', 'goalkeeper', 'sprinter', 'coach', 'midfielder', 'climber'],
    verbs: ['scored', 'defended', 'sprinted', 'trained', 'passed', 'won'],
    objects: ['the winning goal', 'the final match', 'the relay race', 'the championship title', 'the penalty kick', 'the marathon'],
    places: ['in the packed stadium', 'during overtime', 'on the home field', 'at the national finals', 'under the floodlights', 'in the last minute'],
  },
  {
    name: 'gardening',
    adjectives: ['blooming', 'hardy', 'fragrant', 'overgrown', 'young', 'wilted'],
    nouns: ['gardener', 'botanist', 'landscaper', 'farmer', 'florist', 'neighbor'],
    verbs: ['planted', 'pruned', 'watered', 'transplanted', 'harvested', 'mulched'],
    objects: ['the rose bushes', 'the tomato seedlings', 'the apple orchard', 'the herb garden', 'the climbing ivy', 'the tulip beds'],
    places: ['behind the greenhouse', 'along the garden path', 'in early spring', 'before the first frost', 'near the stone wall', 'under the old oak'],
  },
  {
    name: 'programming',
    adjectives: ['legacy', 'concurrent', 'recursive', 'flaky', 'optimized', 'deprecated'],
    nouns: ['developer', 'engineer', 'maintainer', 'reviewer', 'intern', 'architect'],
    verbs: ['refactored', 'debugged', 'deployed', 'profiled', 'merged', 'tested'],
    objects: ['the parser module', 'the failing test suite', 'the database migration', 'the caching layer', 'the build pipeline', 'the memory leak'],
    places: ['before the release', 'in the staging cluster', 'during code review', 'after the outage', 'on the main branch', 'late on friday'],
  },
  {
    name: 'music',
    adjectives: ['gentle', 'loud', 'melancholy', 'improvised', 'baroque', 'rhythmic'],
    nouns: ['violinist', 'composer', 'drummer', 'conductor', 'pianist', 'choir'],
    verbs: ['performed', 'rehearsed', 'recorded', 'arranged', 'tuned', 'conducted'],
    objects: ['the opening sonata', 'the jazz standard', 'the string quartet', 'the folk ballad', 'the final symphony', 'the choral piece'],
    places: ['in the concert hall', 'at the summer festival', 'during the encore', 'in the recording studio', 'before the premiere', 'for the radio broadcast'],
  },
  {
    name: 'weather',
    adjectives: ['heavy', 'sudden', 'mild', 'freezing', 'humid', 'gusty'],
    nouns: ['forecaster', 'meteorologist', 'storm chaser', 'pilot', 'sailor', 'farmer'],
    verbs: ['predicted', 'reported', 'monitored', 'warned about', 'recorded', 'expected'],
    objects: ['the approaching storm', 'the cold front', 'the morning fog', 'the hail shower', 'the heat wave', 'the rising flood'],
    places: ['over the coastal plain', 'across the valley', 'by late afternoon', 'throughout the weekend', 'near the river delta', 'along the ridge'],
  },
  {
    name: 'travel',
    adjectives: ['crowded', 'remote', 'scenic', 'delayed', 'overnight', 'narrow'],
    nouns: ['traveler', 'guide', 'backpacker', 'photographer', 'driver', 'tour group'],
    verbs: ['crossed', 'explored', 'boarded', 'hiked', 'mapped', 'visited'],
    objects: ['the mountain pass', 'the old harbor town', 'the night train', 'the desert trail', 'the island ferry', 'the ancient ruins'],
    places: ['at dawn', 'without a map', 'during the monsoon', 'on the second day', 'beyond the border', 'before the season ended'],
  },
  {
    name: 'medicine',
    adjectives: ['chronic', 'mild', 'acute', 'rare', 'recurring', 'treatable'],
    nouns: ['surgeon', 'nurse', 'physician', 'pharmacist', 'specialist', 'paramedic'],
    verbs: ['diagnosed', 'treated', 'monitored', 'prescribed medication for', 'examined', 'stabilized'],
    objects: ['the knee injury', 'the viral infection', 'the irregular heartbeat', 'the allergic reaction', 'the broken wrist', 'the migraine episode'],
    places: ['in the emergency ward', 'during the night shift', 'at the rural clinic', 'after the follow-up', 'before the operation', 'on the recovery floor'],
  },
];

function pick(rng, arr) {
  return arr[Math.floor(rng() * arr.length)];
}

function sentence(rng, topic) {
  const adj = pick(rng, topic.adjectives);
  const noun = pick(rng, topic.nouns);
  const verb = pick(rng, topic.verbs);
  const obj = pick(rng, topic.objects);
  const place = pick(rng, topic.places);
  return `The ${adj} ${noun} ${verb} ${obj} ${place}.`;
}

/** Generate the full fixture object (deterministic for a given seed). */
export function generateFixture(seed = SEED) {
  const rng = mulberry32(seed);
  const corpus = [];
  const queries = [];
  const perTopic = CORPUS_SIZE / TOPICS.length;
  const queriesPerTopic = QUERY_COUNT / TOPICS.length;
  for (const topic of TOPICS) {
    const seen = new Set();
    let made = 0;
    while (made < perTopic) {
      const s = sentence(rng, topic);
      if (seen.has(s)) continue; // dedupe within topic; rng advances either way
      seen.add(s);
      corpus.push(s);
      made++;
    }
    let q = 0;
    while (q < queriesPerTopic) {
      const s = sentence(rng, topic);
      if (seen.has(s)) continue; // queries are held out of the corpus
      seen.add(s);
      queries.push(s);
      q++;
    }
  }
  return {
    description:
      'Deterministic English text fixture for the ADR-210 D2 recall benchmark (gate 8). ' +
      'Generated by tests/fixtures/generate-text-corpus.mjs from fixed word lists with a ' +
      'seeded PRNG — regenerate with: node tests/fixtures/generate-text-corpus.mjs',
    seed,
    generator: 'tests/fixtures/generate-text-corpus.mjs',
    topics: TOPICS.map(t => t.name),
    corpus,
    queries,
  };
}

const __filename = fileURLToPath(import.meta.url);
if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  const fixture = generateFixture();
  const outPath = path.join(path.dirname(__filename), 'text-corpus.json');
  fs.writeFileSync(outPath, JSON.stringify(fixture, null, 2) + '\n');
  console.log(`Wrote ${fixture.corpus.length} corpus texts + ${fixture.queries.length} queries to ${outPath}`);
}
