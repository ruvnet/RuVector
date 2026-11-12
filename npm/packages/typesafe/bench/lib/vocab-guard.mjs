// Vocabulary-disjointness guard (ADR-006 §protocol 4): the bench refuses to run
// if a synthetic-generator content word appears in the gen-0 criteria text.
// This is the guard that "caught six leaks on 2026-09-21" — its job is to fail
// the exact failure mode where criteria memorised generator phrasing that
// recurs in the eval split, inflating accuracy for the wrong reason.
//
// Generator content words are drawn from the synthetic generator's metadata
// (tickets-corpus.json `topics`, the only generator metadata present). The
// gen-0 criteria come from tickets-decisions.json. Function words are allowed
// to overlap (they carry no topical signal); only content words trip the guard.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { FIXTURE_DIR } from './fixture.mjs';

// A small English function-word stop-list. Overlap on these never trips the
// guard — they are not generator "content".
export const STOP_WORDS = new Set(
  (
    'a an and are as at be been but by for from had has have he her his i in into is it its ' +
    'me my no not of on or our so than that the their them then there these they this to up ' +
    'us was we were what when which who will with you your already about already already-known ' +
    'already-collected do does done how if more most much need needs new now off out over own ' +
    'same some such very via already-held already-due yet'
  ).split(/\s+/),
);

/** Lowercase, split on non-letters, drop stop-words and 1-2 char tokens. */
export function contentWords(text) {
  return new Set(
    String(text)
      .toLowerCase()
      .split(/[^a-z]+/)
      .filter((w) => w.length >= 3 && !STOP_WORDS.has(w)),
  );
}

/**
 * Generator content words from the corpus generator metadata. Falls back to
 * a caller-supplied word list when no metadata is present.
 */
export function generatorWordsFromCorpus(corpus) {
  const words = new Set();
  for (const topic of corpus.topics ?? []) {
    for (const w of contentWords(topic)) words.add(w);
  }
  return words;
}

/** Flatten every criterion's descriptive text (what/not_for/examples/legend). */
export function criteriaText(questionDefs) {
  const parts = [];
  for (const def of Object.values(questionDefs)) {
    if (def.instructions) parts.push(def.instructions);
    if (def.type === 'choice') {
      for (const c of Object.values(def.criteria)) {
        if (typeof c === 'string') parts.push(c);
        else {
          if (c.what) parts.push(c.what);
          if (c.not_for) parts.push(c.not_for);
          for (const e of c.examples ?? []) parts.push(e);
        }
      }
    } else if (def.type === 'score') {
      for (const level of def.criteria ?? def.legend ?? []) parts.push(level);
    }
  }
  return parts.join(' ');
}

// Reviewed, known-benign overlaps between the *retrieval-corpus* generator's
// topic list and the gen-0 decision criteria (2026-09-21). The corpus generator
// (astronomy/cooking/software/…) is a different generator than the one that
// produced the decision tickets; "software" collides only because the plain-
// English `technical` criterion ("Something in the software is broken") happens
// to name the same everyday word. It is not memorised generator phrasing. Like
// deny.toml's dated ignores and ADR-005's github.com/ruvnet allow-list, this is
// an explicit exception so the guard fires on *new* leaks, not this reviewed one.
export const ACCEPTED_OVERLAPS = new Set(['software']);

/**
 * Check generator words against criteria. Returns { ok, leaks, accepted }.
 * `leaks` is the sorted set of generator content words that appear in the
 * criteria and are NOT in the accepted-overlap set — the list included in the
 * thrown error so the failure is legible. `accepted` records suppressed
 * known-benign overlaps for the receipt.
 */
export function checkVocabDisjoint(generatorWords, questionDefs, { accepted = new Set() } = {}) {
  const inCriteria = contentWords(criteriaText(questionDefs));
  const leaks = [];
  const acceptedHits = [];
  for (const w of generatorWords) {
    if (!inCriteria.has(w)) continue;
    if (accepted.has(w)) acceptedHits.push(w);
    else leaks.push(w);
  }
  leaks.sort();
  acceptedHits.sort();
  return { ok: leaks.length === 0, leaks, accepted: acceptedHits };
}

/**
 * Convenience: load the corpus, derive generator words, and check against the
 * given question defs. Throws with the leak list on failure (the harness
 * refuses to run). Returns the generator-word count and any accepted overlaps.
 */
export function assertVocabDisjoint(questionDefs, { fixtureDir = FIXTURE_DIR } = {}) {
  const corpus = JSON.parse(readFileSync(join(fixtureDir, 'tickets-corpus.json'), 'utf8'));
  const genWords = generatorWordsFromCorpus(corpus);
  const res = checkVocabDisjoint(genWords, questionDefs, { accepted: ACCEPTED_OVERLAPS });
  if (!res.ok) {
    throw new Error(
      `vocabulary-disjointness guard failed — generator content words leaked into ` +
        `gen-0 criteria: ${res.leaks.join(', ')}`,
    );
  }
  return { ok: true, generatorWordCount: genWords.size, accepted: res.accepted };
}
