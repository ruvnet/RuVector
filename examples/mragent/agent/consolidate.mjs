// Memory consolidation / replay — the "sleep" phase of a self-reorganizing memory.
//
// Beyond MRAgent: the paper reconstructs over a STATIC graph. A 25-year-out memory
// system reshapes its own topology from workload — exactly the self-learning GNN
// RuVector describes ("pushes similarities back into the neighbor lists"). After a
// batch of successful reconstructions, we REPLAY the winning paths and lay down a
// direct Cue→shortcut→Content edge, so a query that needed a 3-hop traversal today
// resolves in 1 hop tomorrow. Embeddings/content (the frozen model) are untouched;
// only graph adjacency — the store's own learned index — changes.
//
// This is gated and deterministic: it consolidates only paths that already
// reconstruct the CORRECT content, so it never invents associations.

import { runReasoningLoop } from "./harness.mjs";

/**
 * Replay the corpus under `genome` and add shortcut edges for every task whose
 * correct content is currently reconstructed. Mutates the store's graph topology.
 * Returns { consolidated, hopsBefore } for reporting.
 *
 * @param {MemoryStore} store
 * @param {Array} tasks
 * @param {object} genome
 */
export function consolidate(store, tasks, genome) {
  const { cues, tags } = store.graph;
  let consolidated = 0;
  let hopsBefore = 0;
  let n = 0;

  for (const task of tasks) {
    if (task.answerable === false) continue;
    const r = runReasoningLoop(store.queryText(task.id), store, genome, task);
    hopsBefore += r.hops; n++;
    if (!r.correct) continue; // only consolidate paths that genuinely work

    const correctCue = cues.get(`cue:${task.id}:correct`);
    const correctContentId = `content:${task.id}`;
    if (!correctCue) continue;

    // Lay down a 1-hop shortcut tag the correct cue reaches immediately.
    const shortcutId = `tag:${task.id}-shortcut`;
    if (!tags.has(shortcutId)) {
      tags.set(shortcutId, {
        id: shortcutId, name: `${task.id}-shortcut`, text: "shortcut",
        toks: [], vec: new Float32Array(store.cueList[0].vec.length), content: [correctContentId], next: [],
      });
      // Prepend so it is the first link explored (reached at hop 1, fanout-safe).
      correctCue.links = [shortcutId, ...correctCue.links];
      consolidated++;
    }
  }

  return { consolidated, avgHopsBefore: hopsBefore / (n || 1) };
}
