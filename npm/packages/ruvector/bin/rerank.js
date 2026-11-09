'use strict';
/**
 * rerank.js: `hooks rag-context --rerank`.
 *
 * The flag was a silent no-op: rag-context never called loadNewModules(), so ExtendedWorkerPool was undefined; the pool
 * was never init()-ed (and is disabled unless RUVECTOR_PARALLEL=1), so execute() threw "Worker pool not initialized",
 * which an empty catch swallowed; rankContext() got its two arguments swapped; and its result (bare strings) would have
 * replaced the memories, dropping ids and scores. This reorders the memories by the worker's TF-IDF relevance, keeps
 * every field, adds `relevance`, and says whether it reranked. Test: test/rag-context-rerank.js.
 */
async function rerankMemories(memories, query, Pool) {
  const list = Array.isArray(memories) ? memories : [];
  if (!Pool) return { results: list, reranked: false, error: 'worker pool unavailable' };
  if (!list.length) return { results: list, reranked: true };
  const pool = new Pool({ enabled: true, numWorkers: 1 });
  try {
    await pool.init();
    const texts = list.map((m) => String(m && typeof m === 'object' ? (m.content ?? '') : (m ?? '')));
    const ranked = await pool.rankContext(texts, String(query ?? ''));
    const results = ranked.map((r) => {
      const m = list[r.index];
      return m && typeof m === 'object' ? { ...m, relevance: r.relevance } : { content: m, relevance: r.relevance };
    });
    return { results, reranked: true };
  } catch (e) {
    return { results: list, reranked: false, error: e && e.message ? e.message : String(e) };
  } finally {
    try { await pool.shutdown(); } catch (_) { /* nothing left to stop */ }
  }
}

module.exports = { rerankMemories };
