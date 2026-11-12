// Dataset router. `loadDataset(suite)` dispatches to a fetcher and, on any
// download/parse failure, returns `{ skipped: 'unavailable: <reason>' }` so the
// harness marks that dataset skipped with the source URL — never a silent pass
// (ADR-006 §Where it lives). Datasets are fetched at bench time; nothing is
// committed to the package.

import { load as fb15k237 } from './fb15k237.mjs';
import { load as wn18rr } from './wn18rr.mjs';
import { load as codexm } from './codexm.mjs';
import { load as yago310 } from './yago310.mjs';

const FETCHERS = { fb15k237, wn18rr, codexm, yago310 };

export async function loadDataset(suite, opts = {}) {
  const fetcher = FETCHERS[suite];
  if (!fetcher) return { skipped: `unknown dataset '${suite}'` };
  try {
    return await fetcher(opts);
  } catch (e) {
    return { skipped: `unavailable: ${e && e.message ? e.message : e}` };
  }
}

export const DATASETS = Object.keys(FETCHERS);
