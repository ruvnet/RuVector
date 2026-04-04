/**
 * module-tree.js - Hierarchical module tree builder.
 *
 * Builds a tree from co-reference density between modules using
 * agglomerative clustering and discriminative token naming.
 */

'use strict';

/**
 * Build a hierarchical module tree from co-reference density.
 *
 * 1. Build adjacency matrix from shared string references between modules.
 * 2. Agglomerative clustering by edge density.
 * 3. Name clusters from dominant discriminative strings.
 *
 * @param {Array<{name: string, content: string, fragments: number, confidence: number}>} modules
 * @param {string} source
 * @returns {{name: string, path: string, modules: Array, children: Array, depth: number}}
 */
function buildModuleTree(modules, source) {
  if (modules.length <= 1) {
    return {
      name: 'src',
      path: 'src',
      modules,
      children: [],
      depth: 0,
    };
  }

  // Extract string tokens from each module's content.
  const moduleTokens = modules.map((m) => {
    const tokens = new Set();
    const re = /["']([a-zA-Z_]\w{2,30})["']/g;
    let match;
    while ((match = re.exec(m.content)) !== null) {
      tokens.add(match[1]);
    }
    return tokens;
  });

  // Build adjacency: weight = number of shared tokens.
  const weights = new Map();
  for (let i = 0; i < modules.length; i++) {
    for (let j = i + 1; j < modules.length; j++) {
      let shared = 0;
      for (const tok of moduleTokens[i]) {
        if (moduleTokens[j].has(tok)) shared++;
      }
      if (shared > 0) {
        weights.set(`${i}:${j}`, shared);
      }
    }
  }

  // Agglomerative clustering.
  let clusters = modules.map((_, i) => [i]);

  while (clusters.length > 3) {
    let bestI = 0, bestJ = 1, bestW = -1;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const w = clusterWeight(clusters[i], clusters[j], weights);
        const norm = w / (clusters[i].length + clusters[j].length);
        if (norm > bestW) {
          bestW = norm;
          bestI = i;
          bestJ = j;
        }
      }
    }
    if (bestW <= 0) break;
    const merged = [...clusters[bestI], ...clusters[bestJ]];
    clusters.splice(bestJ, 1);
    clusters.splice(bestI, 1);
    clusters.push(merged);
  }

  // Name each cluster from discriminative tokens.
  const children = clusters.map((group) => {
    const groupModules = group.map((i) => modules[i]);
    const name = inferGroupName(group, moduleTokens, modules);
    return {
      name,
      path: `src/${name}`,
      modules: groupModules,
      children: [],
      depth: 1,
    };
  });

  return {
    name: 'src',
    path: 'src',
    modules: [],
    children,
    depth: 0,
  };
}

/** Compute total shared-token weight between two clusters. */
function clusterWeight(a, b, weights) {
  let total = 0;
  for (const ai of a) {
    for (const bi of b) {
      const key = ai < bi ? `${ai}:${bi}` : `${bi}:${ai}`;
      total += weights.get(key) || 0;
    }
  }
  return total;
}

/** Infer a group name from discriminative tokens. */
function inferGroupName(group, moduleTokens, modules) {
  const freq = new Map();
  for (const i of group) {
    for (const tok of moduleTokens[i]) {
      freq.set(tok, (freq.get(tok) || 0) + 1);
    }
  }
  const globalFreq = new Map();
  for (const tokens of moduleTokens) {
    for (const tok of tokens) {
      globalFreq.set(tok, (globalFreq.get(tok) || 0) + 1);
    }
  }
  let best = null, bestScore = -1;
  for (const [tok, count] of freq) {
    const global = globalFreq.get(tok) || 0;
    const score = (count / (global + 1)) * Math.log(count + 1);
    if (score > bestScore && tok.length >= 3) {
      bestScore = score;
      best = tok;
    }
  }
  if (best) return best.toLowerCase().replace(/[^a-z0-9_-]/g, '_');
  if (group.length > 0) return modules[group[0]].name;
  return 'group';
}

module.exports = { buildModuleTree };
