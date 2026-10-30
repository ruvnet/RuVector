// Community detection on graphs with planted groups: louvainCommunities() must end and find them, and
// spectralClustering() must find them the same way every run. Each case runs in a child process with a time
// limit, so a non-terminating implementation fails here instead of hanging the suite.
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const modulePath = require.resolve('../dist/core/graph-algorithms');

// Run `body` (a function source taking (g, planted)) in a child process; returns its JSON result.
function inChild(body, timeoutMs = 10000) {
  const src = `
    const g = require(${JSON.stringify(modulePath)});
    // Planted partition: k groups of n/k nodes, dense inside, sparse across; seeded, so every run sees the same graph.
    function planted(n, k, seed = 7) {
      let s = seed;
      const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
      const group = (i) => Math.floor((i * k) / n);
      const nodes = Array.from({ length: n }, (_, i) => 'n' + i);
      const pIn = Math.min(0.5, 12 / (n / k)), pOut = pIn / 40;
      const edges = [];
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          if (rnd() < (group(i) === group(j) ? pIn : pOut)) edges.push({ from: nodes[i], to: nodes[j] });
      return { nodes, edges, group };
    }
    // true when every planted group has one label of its own
    function recovered(labelOf, p, k) {
      const labels = Array.from({ length: k }, (_, c) => new Set(p.nodes.filter((_, i) => p.group(i) === c).map(labelOf)));
      return labels.every((s) => s.size === 1) && new Set(labels.map((s) => [...s][0])).size === k;
    }
    const out = (${body})(g, planted, recovered);
    process.stdout.write(JSON.stringify(out));`;
  const r = spawnSync(process.execPath, ['-e', src], { encoding: 'utf8', timeout: timeoutMs });
  if (r.error && r.error.code === 'ETIMEDOUT') return { timedOut: true };
  if (r.status !== 0) return { error: (r.stderr || '').split('\n').slice(0, 3).join(' ') };
  return JSON.parse(r.stdout);
}

const cases = [
  ['louvain ends and finds two planted groups of 25 (50 nodes)', () => {
    const r = inChild(`(g, planted, recovered) => { const p = planted(50, 2); const c = g.louvainCommunities(g.buildGraph(p.nodes, p.edges));
      return { recovered: recovered((n) => c.get(n), p, 2), communities: new Set(c.values()).size }; }`);
    assert.ok(!r.timedOut, 'louvainCommunities did not return within 10 s');
    assert.ok(!r.error, r.error);
    assert.deepEqual(r, { recovered: true, communities: 2 });
  }],
  ['louvain ends and finds two planted groups of 200 (400 nodes)', () => {
    const r = inChild(`(g, planted, recovered) => { const p = planted(400, 2); const c = g.louvainCommunities(g.buildGraph(p.nodes, p.edges));
      return { recovered: recovered((n) => c.get(n), p, 2), communities: new Set(c.values()).size }; }`);
    assert.ok(!r.timedOut, 'louvainCommunities did not return within 10 s');
    assert.ok(!r.error, r.error);
    assert.deepEqual(r, { recovered: true, communities: 2 });
  }],
  ['louvain: an edgeless graph keeps every node alone; a triangle is one community', () => {
    const r = inChild(`(g) => {
      const e = g.louvainCommunities(g.buildGraph(['a', 'b', 'c'], []));
      const t = g.louvainCommunities(g.buildGraph(['a', 'b', 'c'], [{ from: 'a', to: 'b' }, { from: 'b', to: 'c' }, { from: 'c', to: 'a' }]));
      return { edgeless: new Set(e.values()).size, triangle: new Set(t.values()).size }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { edgeless: 3, triangle: 1 });
  }],
  ['spectral finds two planted groups (50 nodes) and three (90 nodes), smallest eigenvalue ~0', () => {
    const r = inChild(`(g, planted, recovered) => {
      const p2 = planted(50, 2), s2 = g.spectralClustering(g.buildGraph(p2.nodes, p2.edges), 2);
      const p3 = planted(90, 3), s3 = g.spectralClustering(g.buildGraph(p3.nodes, p3.edges), 3);
      return { two: recovered((n) => s2.clusters.get(n), p2, 2), three: recovered((n) => s3.clusters.get(n), p3, 3),
               firstEigenvalueNearZero: Math.abs(s2.eigenvalues[0]) < 1e-6 }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { two: true, three: true, firstEigenvalueNearZero: true });
  }],
  ['spectral gives the same clusters on every run', () => {
    const r = inChild(`(g, planted) => { const p = planted(60, 2);
      const a = g.spectralClustering(g.buildGraph(p.nodes, p.edges), 2), b = g.spectralClustering(g.buildGraph(p.nodes, p.edges), 2);
      return { same: p.nodes.every((n) => a.clusters.get(n) === b.clusters.get(n)) }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { same: true });
  }],
  ['louvain: three disconnected pairs are three communities; heavy edges decide when weights differ', () => {
    const r = inChild(`(g) => {
      const label = (c) => JSON.stringify([...c.values()]);
      const pairs = g.louvainCommunities(g.buildGraph(['a', 'b', 'c', 'd', 'e', 'f'], [{ from: 'a', to: 'b' }, { from: 'c', to: 'd' }, { from: 'e', to: 'f' }]));
      const w = g.louvainCommunities(g.buildGraph(['a', 'b', 'c', 'd'], [{ from: 'a', to: 'b', weight: 10 }, { from: 'c', to: 'd', weight: 10 }, { from: 'b', to: 'c', weight: 1 }]));
      return { pairs: label(pairs), weighted: label(w) }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { pairs: '[0,0,1,1,2,2]', weighted: '[0,0,1,1]' });
  }],
  ['louvain: empty graph, self-loop, duplicate edges and an edge to an unlisted node give every listed node a label 0..k-1', () => {
    const r = inChild(`(g) => {
      const ok = (nodes, c) => nodes.every((n) => Number.isInteger(c.get(n))) && [...new Set(c.values())].sort((a, b) => a - b).every((v, i) => v === i);
      const cases = [
        [[], []],
        [['a', 'b'], [{ from: 'a', to: 'a' }, { from: 'a', to: 'b' }]],
        [['a', 'b', 'c'], [{ from: 'a', to: 'b' }, { from: 'a', to: 'b' }, { from: 'b', to: 'c' }]],
        [['a', 'b'], [{ from: 'a', to: 'b' }, { from: 'a', to: 'z' }]],
      ];
      return cases.map(([n, e]) => ok(n, g.louvainCommunities(g.buildGraph(n, e)))); }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, [true, true, true, true]);
  }],
  ['louvain on 12 random planted graphs (30–120 nodes, 2–4 groups): modularity at least the planted partition', () => {
    const r = inChild(`(g, planted) => {
      const out = [];
      for (let t = 0; t < 12; t++) {
        const k = 2 + (t % 3), n = k * (10 + 3 * t);
        const p = planted(n, k, 101 + t), graph = g.buildGraph(p.nodes, p.edges);
        const c = g.louvainCommunities(graph);
        const found = {}; for (const [node, l] of c) (found[l] = found[l] || []).push(node);
        const plantedGroups = Array.from({ length: k }, (_, j) => p.nodes.filter((_, i) => p.group(i) === j));
        out.push(g.calculateModularity(graph, Object.values(found)) >= g.calculateModularity(graph, plantedGroups) - 1e-9);
      }
      return out; }`, 20000);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, Array(12).fill(true));
  }],
  ['spectral: two disconnected triangles split exactly; eigenvalues ascending within [0, 2]', () => {
    const r = inChild(`(g) => {
      const s = g.spectralClustering(g.buildGraph(['a', 'b', 'c', 'd', 'e', 'f'],
        [{ from: 'a', to: 'b' }, { from: 'b', to: 'c' }, { from: 'c', to: 'a' }, { from: 'd', to: 'e' }, { from: 'e', to: 'f' }, { from: 'f', to: 'd' }]), 2);
      const l = ['a', 'b', 'c', 'd', 'e', 'f'].map((n) => s.clusters.get(n));
      const ev = s.eigenvalues;
      return { split: l[0] === l[1] && l[1] === l[2] && l[3] === l[4] && l[4] === l[5] && l[0] !== l[3],
               eigen: ev.every((v, i) => v >= -1e-9 && v <= 2 + 1e-9 && (i === 0 || v >= ev[i - 1] - 1e-9)) }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { split: true, eigen: true });
  }],
  ['spectral: empty graph, k = 1, k > n, an isolated node and an edge to an unlisted node', () => {
    const r = inChild(`(g) => {
      const S = (n, e, k) => g.spectralClustering(g.buildGraph(n, e), k).clusters;
      const empty = S([], [], 2).size;
      const one = new Set(S(['a', 'b', 'c'], [{ from: 'a', to: 'b' }], 1).values()).size;
      const big = S(['a', 'b'], [{ from: 'a', to: 'b' }], 5);
      const iso = S(['a', 'b', 'c'], [{ from: 'a', to: 'b' }], 2);
      const unk = S(['a', 'b'], [{ from: 'a', to: 'b' }, { from: 'a', to: 'z' }], 2);
      return { empty, one, bigLabelled: big.size === 2, isolatedApart: iso.get('a') === iso.get('b') && iso.get('c') !== iso.get('a'), unknownIgnored: unk.size === 2 && !unk.has('z') }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { empty: 0, one: 1, bigLabelled: true, isolatedApart: true, unknownIgnored: true });
  }],
  ['minCut: two triangles joined by one edge are cut at that edge (weight 1)', () => {
    const r = inChild(`(g) => {
      const m = g.minCut(g.buildGraph(['a', 'b', 'c', 'd', 'e', 'f'], [{ from: 'a', to: 'b' }, { from: 'b', to: 'c' }, { from: 'c', to: 'a' },
        { from: 'd', to: 'e' }, { from: 'e', to: 'f' }, { from: 'f', to: 'd' }, { from: 'c', to: 'd' }]));
      return { groups: m.groups.map((x) => [...x].sort().join('')).sort(), cut: m.cutWeight }; }`);
    assert.ok(!r.timedOut && !r.error, r.error || 'timed out');
    assert.deepEqual(r, { groups: ['abc', 'def'], cut: 1 });
  }],
];

const failures = [];
for (const [name, fn] of cases) {
  try { fn(); console.log(`  PASS  ${name}`); } catch (error) { failures.push(`${name}: ${error.message}`); console.log(`  FAIL  ${name}`); }
}
assert.deepEqual(failures, []);
