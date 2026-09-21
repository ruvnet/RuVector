// Dataset router. `loadDataset(suite)` dispatches to a fetcher and, on any
// download/parse failure, returns `{ skipped: 'unavailable: <reason>' }` so the
// harness marks that dataset skipped with the source URL — never a silent pass
// (ADR-006 §Where it lives).

import { load as banking77 } from './banking77.mjs';
import { load as clinc150 } from './clinc150.mjs';
import { load as hwu64 } from './hwu64.mjs';

const FETCHERS = { banking77, clinc150, hwu64 };

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
