// Novel-composition slice (ADR-007 §1b). The tickets generator is templated:
// many test items are stitched from sentences that also occur in the train
// pool. A test item is TEMPLATE if every one of its sentences occurs in the
// gradient pool, NOVEL otherwise. Tiers are reported on both.
//
//   pool      = excludeHeldOutText(bySplit).trainItems (136 rows — the only
//               ticket rows that ever produce gradients)
//   sentences = raw text split on /[.!?:]+/, each piece norm()ed (lib/norm.mjs),
//               kept iff it has ≥ 3 space-separated tokens
//   template  = test item with ≥ 1 kept sentence, all of them in the pool set
//   novel     = every other test item
//
// Measured at the 2026-09-26 freeze: 31 template / 119 novel (141/150 share at
// least one pool sentence). The ids are frozen in
// bench/fixtures/novel-slice-2026-09-26.json, pinned under HASHES.json
// `derived` (scripts/hash-fixtures.mjs). vs-jev recomputes and must match.
//
//   node bench/lib/novel-slice.mjs --write   # re-freeze (a reviewed event)

import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadTickets, excludeHeldOutText, FIXTURE_DIR } from './fixture.mjs';
import { norm } from './norm.mjs';

export const NOVEL_SLICE_FILE = 'fixtures/novel-slice-2026-09-26.json';
export const MIN_SENTENCE_TOKENS = 3;

/** norm()ed sentences of `text` with ≥ MIN_SENTENCE_TOKENS tokens. */
export function sentences(text) {
  return text
    .split(/[.!?:]+/)
    .map(norm)
    .filter((s) => s.split(' ').filter(Boolean).length >= MIN_SENTENCE_TOKENS);
}

/** Compute the split from loaded tickets. Ids are sorted lexically. */
export function computeNovelSlice(tickets) {
  const pool = excludeHeldOutText(tickets.bySplit).trainItems;
  const poolSentences = new Set(pool.flatMap((it) => sentences(it.text)));
  const template = [];
  const novel = [];
  let shareAny = 0;
  for (const it of tickets.bySplit.test) {
    const s = sentences(it.text);
    const hits = s.filter((x) => poolSentences.has(x)).length;
    if (hits > 0) shareAny++;
    (s.length > 0 && hits === s.length ? template : novel).push(it.id);
  }
  template.sort();
  novel.sort();
  return {
    pool_rows: pool.length,
    pool_sentences: poolSentences.size,
    test_rows: tickets.bySplit.test.length,
    share_any: shareAny,
    template_ids: template,
    novel_ids: novel,
  };
}

/** The frozen document written to NOVEL_SLICE_FILE. */
export function novelSliceDocument(tickets) {
  const s = computeNovelSlice(tickets);
  return {
    _comment:
      'ADR-007 §1b novel-composition slice of the tickets test split. template = every sentence ' +
      '(split on [.!?:]+, norm, >= 3 tokens) occurs in the 136-row gradient pool; novel = the rest. ' +
      'Frozen 2026-09-26; sha256 pinned in HASHES.json `derived`. Regenerate only on a reviewed re-freeze.',
    generated_by: 'bench/lib/novel-slice.mjs',
    splits_hash: tickets.splitsHash,
    definition: {
      pool: 'excludeHeldOutText(bySplit).trainItems',
      sentence_split: '[.!?:]+',
      normalization: 'bench/lib/norm.mjs norm()',
      min_tokens: MIN_SENTENCE_TOKENS,
    },
    counts: {
      pool_rows: s.pool_rows,
      pool_sentences: s.pool_sentences,
      test: s.test_rows,
      template: s.template_ids.length,
      novel: s.novel_ids.length,
      share_any_pool_sentence: s.share_any,
    },
    template_ids: s.template_ids,
    novel_ids: s.novel_ids,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  if (!process.argv.includes('--write')) {
    console.log(JSON.stringify(novelSliceDocument(loadTickets()).counts));
  } else {
    const doc = novelSliceDocument(loadTickets());
    const out = join(FIXTURE_DIR, '..', NOVEL_SLICE_FILE);
    writeFileSync(out, JSON.stringify(doc, null, 2) + '\n');
    console.log(`wrote ${out} ${JSON.stringify(doc.counts)} — now run scripts/hash-fixtures.mjs`);
  }
}
