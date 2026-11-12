/**
 * package-banner-motifs.mjs — the small animated motifs drawn in the right-hand
 * panel of each npm package banner (see scripts/gen-package-banners.mjs).
 *
 * Every motif is a pure function of its options: no Date, no Math.random, so the
 * generator output is byte-for-byte reproducible. Motifs draw inside the panel
 * box x 600..970, y 58..208 and only use the shared CSS classes defined in
 * CLASS_CSS (the generator emits just the rules a banner actually uses).
 */

export const C = {
  accent: '#5cf2c3', violet: '#bd9bff', amber: '#ffbe73', text: '#f2f6ff',
  sub: '#b8c7da', dim: '#374354', node: '#162238', faint: '#8297b1',
};

/** CSS for each animation class, keyed by class name. */
export const CLASS_CSS = {
  wire: `.wire{fill:none;stroke:${C.accent};stroke-width:2;stroke-dasharray:8 7;animation:flow 3s linear infinite}`,
  lit: '.lit{animation:lit 6s ease-in-out infinite}',
  pulse: '.pulse{transform-box:fill-box;transform-origin:center;animation:pulse 2.4s ease-out infinite}',
  grow: '.grow{transform-box:fill-box;transform-origin:50% 100%;animation:grow 4s ease-in-out infinite}',
  col: '.col{transform-box:fill-box;transform-origin:50% 100%;animation:col 6s ease-in-out infinite}',
  bit: '.bit{animation:bit 6s ease-in-out infinite}',
  scan: '.scan{animation:scan 3s ease-in-out infinite alternate}',
  spk: '.spk{fill:none;stroke-width:12;animation:flow 2.4s linear infinite}',
};
export const KEYFRAMES = {
  wire: '@keyframes flow{to{stroke-dashoffset:-60}}',
  spk: '@keyframes flow{to{stroke-dashoffset:-60}}',
  lit: '@keyframes lit{0%,100%{opacity:.3}12%,40%{opacity:1}}',
  pulse: '@keyframes pulse{0%{transform:scale(.6);opacity:.9}100%{transform:scale(2);opacity:0}}',
  grow: '@keyframes grow{0%,100%{transform:scaleY(.35)}50%{transform:scaleY(1)}}',
  col: '@keyframes col{0%,35%{transform:none}50%,90%{transform:scaleY(.12)}}',
  bit: '@keyframes bit{0%,40%{opacity:.15}55%,90%{opacity:1}}',
  scan: '@keyframes scan{to{transform:translateY(84px)}}',
};

const r1 = (n) => Math.round(n * 10) / 10;
const lit = (d) => `class="lit" style="animation-delay:${r1(d)}s"`;
const dly = (cls, d) => `class="${cls}" style="animation-delay:${r1(d)}s"`;
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const t = (x, y, s, size = 12, fill = C.sub, extra = '') =>
  `<text x="${x}" y="${y}" fill="${fill}" font-size="${size}"${extra}>${esc(s)}</text>`;
const tm = (x, y, s, size = 12, fill = C.sub) => t(x, y, s, size, fill, ' text-anchor="middle"');
const dot = (x, y, r = 6, fill = C.accent, attrs = '') => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" ${attrs}/>`;
const chip = (x, y, w, h, label, stroke = C.accent, attrs = '', size = 12) =>
  `<g ${attrs}><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${C.node}" stroke="${stroke}"/>${tm(x + w / 2, y + h / 2 + size * 0.36, label, size, C.text)}</g>`;
const line = (x1, y1, x2, y2, stroke = C.dim, attrs = '') =>
  `<path d="M${x1} ${y1}L${x2} ${y2}" stroke="${stroke}" ${attrs}/>`;
const cyl = (x, y, w, h, label, stroke = C.violet) =>
  `<g fill="${C.node}" stroke="${stroke}"><path d="M${x} ${y}v${h}a${w / 2} 7 0 0 0 ${w} 0v-${h}"/><ellipse cx="${x + w / 2}" cy="${y}" rx="${w / 2}" ry="7"/></g>${tm(x + w / 2, y + h / 2 + 8, label, 12, C.text)}`;

/** Draw a node/edge graph; edges light in order, nodes light after their edge. */
function graph(nodes, edges, { cap, edgeColor = C.accent, step = 0.6 } = {}) {
  let s = edges.map(([a, b]) => line(nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1])).join('');
  s += edges.map(([a, b], i) => line(nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], edgeColor, `stroke-width="2" ${lit(i * step)}`)).join('');
  s += nodes.map(([x, y], i) => dot(x, y, 6, C.accent, lit(i * step))).join('');
  return s + (cap ? t(616, 200, cap, 12, C.faint) : '');
}

const M = {
  knn() { // query point, nearest neighbours light up
    const pts = [[640, 90], [672, 150], [700, 78], [720, 180], [745, 110], [770, 160], [800, 85], [812, 128], [790, 140], [835, 175], [860, 100], [885, 150], [910, 80], [930, 175], [760, 128], [828, 105]];
    const q = [795, 125], near = [7, 8, 14, 15, 5];
    let s = pts.map(([x, y]) => dot(x, y, 3.5, C.faint)).join('');
    s += near.map((i, k) => line(q[0], q[1], pts[i][0], pts[i][1], C.accent, `stroke-width="1.5" ${lit(k * 0.5)}`) + dot(pts[i][0], pts[i][1], 5, C.accent, lit(k * 0.5))).join('');
    s += `<circle cx="${q[0]}" cy="${q[1]}" r="12" fill="none" stroke="${C.amber}" class="pulse"/>` + dot(q[0], q[1], 5, C.amber);
    return s + t(616, 200, 'query → top-k neighbours + memory', 12, C.faint);
  },
  hnsw() { // layered HNSW descent
    const L = [[78, [700, 880]], [118, [650, 740, 820, 920]], [160, [630, 680, 730, 780, 830, 880, 935]]];
    let s = L.map(([y], i) => `<path d="M615 ${y}H955" stroke="${C.dim}" stroke-dasharray="2 4"/>${t(958, y + 4, 'L' + (2 - i), 10, C.faint)}`).join('');
    s += L.map(([y, xs]) => xs.map((x) => dot(x, y, 4, C.faint)).join('')).join('');
    s += '<path class="wire" d="M700 78L880 78L820 118L780 118L780 160L830 160"/>';
    s += [[700, 78], [880, 78], [820, 118], [780, 118], [780, 160], [830, 160]].map(([x, y], i) => dot(x, y, 6, C.accent, lit(i * 0.7))).join('');
    return s + t(616, 200, 'greedy descent through HNSW layers', 12, C.faint);
  },
  gnn() { // message passing
    const n = [[660, 90], [660, 170], [760, 130], [860, 85], [860, 175], [930, 130]];
    let s = graph(n, [[0, 2], [1, 2], [2, 3], [2, 4], [3, 5], [4, 5]], { cap: 'HNSW search + GNN message passing' });
    return s + '<path class="wire" d="M660 90L760 130L860 85L930 130"/>';
  },
  bundle({ items }) { // packages gathered into one install
    let s = items.map((it, i) => chip(615 + (i % 3) * 120, 66 + Math.floor(i / 3) * 34, 110, 26, it, C.accent, lit(i * 0.5), 11)).join('');
    s += '<path class="wire" d="M785 136V156"/>';
    return s + chip(680, 158, 210, 34, 'npm i @ruvector/all', C.amber, '', 14);
  },
  loop({ steps }) { // circular adaptation loop
    const cx = 785, cy = 130, rx = 120, ry = 50;
    let s = `<ellipse class="wire" cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"/>`;
    const pos = [[cx, cy - ry], [cx + rx, cy], [cx, cy + ry], [cx - rx, cy]];
    s += steps.map((st, i) => chip(pos[i][0] - 58, pos[i][1] - 13, 116, 26, st, i % 2 ? C.violet : C.accent, lit(i * 1.4), 11)).join('');
    return s;
  },
  segments({ labels, cap, top = 'query' }) { // RVF container segments
    const w = 330 / labels.length;
    let s = `<rect x="620" y="104" width="330" height="48" rx="10" fill="none" stroke="${C.violet}"/>`;
    s += labels.map((l, i) => `<g ${lit(i * 0.9)}><rect x="${r1(624 + i * w)}" y="108" width="${r1(w - 8)}" height="40" rx="6" fill="${C.node}" stroke="${C.accent}"/>${tm(r1(620 + i * w + w / 2), 133, l, 11, C.text)}</g>`).join('');
    s += `<path class="wire" d="M640 80H930"/>${t(620, 74, top, 11, C.faint)}`;
    return s + t(616, 182, cap, 12, C.faint);
  },
  solver() { // Thompson sampling arms
    const arms = ['policy A', 'policy B', 'policy C', 'policy D'];
    let s = arms.map((a, i) => `<rect x="${640 + i * 82}" y="76" width="40" height="84" rx="5" fill="${i === 2 ? C.accent : C.violet}" ${dly('grow', i * 0.9)}/>${tm(660 + i * 82, 176, a, 11)}`).join('');
    s += `<circle cx="824" cy="68" r="5" fill="none" stroke="${C.amber}" class="pulse"/>`;
    return s + t(616, 202, 'sample arms → pick → update posterior', 11, C.faint);
  },
  fanout({ src, targets, cap, broken = -1 }) { // one request, routed to one target
    const ys = targets.map((_, i) => 68 + i * (96 / (targets.length - 1)));
    let s = ys.map((y) => line(700, 128, 840, y + 12)).join('');
    s += ys.map((y, i) => line(700, 128, 840, y + 12, i === broken ? C.amber : C.accent, `stroke-width="2" ${lit(i * 1.5)}`)).join('');
    s += targets.map((tg, i) => chip(840, ys[i], 112, 24, tg, i === broken ? C.amber : C.accent, lit(i * 1.5), 11)).join('');
    s += chip(612, 114, 88, 28, src, C.violet, '', 12) + `<circle cx="700" cy="128" r="6" fill="none" stroke="${C.accent}" class="pulse"/>`;
    return s + t(612, 202, cap, 11, C.faint);
  },
  bits() { // float bars collapse to 1-bit codes
    const h = [60, 28, 72, 44, 18, 66, 36, 52];
    let s = h.map((v, i) => `<rect x="${620 + i * 18}" y="${160 - v}" width="12" height="${v}" rx="2" fill="${v > 40 ? C.accent : C.violet}" ${dly('col', i * 0.12)}/>`).join('');
    s += `<path class="wire" d="M775 125H815"/>${tm(795, 112, '32×', 12, C.amber)}`;
    s += h.map((v, i) => `<g ${dly('bit', i * 0.12)}><rect x="${828 + (i % 4) * 34}" y="${96 + Math.floor(i / 4) * 36}" width="28" height="28" rx="4" fill="${v > 40 ? C.accent : C.node}" stroke="${C.accent}"/>${tm(842 + (i % 4) * 34, 115 + Math.floor(i / 4) * 36, v > 40 ? '1' : '0', 13, v > 40 ? '#0c1324' : C.text)}</g>`).join('');
    return s + t(616, 196, 'f32 coordinates → sign bits → rerank', 12, C.faint);
  },
  diskhops() { // graph hops in RAM, fetches from SSD
    const n = [[630, 80], [690, 110], [750, 78], [810, 108], [870, 82], [930, 112]];
    let s = graph(n, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], { step: 0.8 });
    s += t(612, 66, 'PQ codes in RAM', 11, C.faint);
    s += [1, 3, 5].map((i, k) => `<path class="wire" d="M${n[i][0]} ${n[i][1] + 8}V160" style="animation-delay:${k * 0.8}s"/>`).join('');
    return s + `<path d="M620 160H950" stroke="${C.violet}"/>` + cyl(860, 168, 90, 14, 'SSD', C.violet) + t(616, 202, 'Vamana hops → full vectors on disk', 11, C.faint);
  },
  graph({ cap }) { // nodes and edges lighting up
    const n = [[640, 100], [700, 160], [745, 80], [800, 135], [860, 75], [880, 170], [935, 120]];
    return graph(n, [[0, 2], [0, 1], [1, 3], [2, 3], [2, 4], [3, 5], [4, 6], [5, 6], [3, 4]], { cap, edgeColor: C.violet, step: 0.5 });
  },
  lanes() { // SQL / Cypher / SPARQL into one store
    const q = ['SQL', 'Cypher', 'SPARQL'];
    let s = q.map((l, i) => chip(616, 70 + i * 40, 96, 28, l, [C.accent, C.violet, C.amber][i], lit(i * 1.6), 13)).join('');
    s += q.map((_, i) => `<path class="wire" d="M712 ${84 + i * 40}C780 ${84 + i * 40} 780 124 840 124"/>`).join('');
    return s + cyl(845, 100, 100, 40, 'rvlite', C.accent) + t(616, 202, 'one store · three query languages', 11, C.faint);
  },
  heartbeat() { // leader heartbeats to followers
    const f = [[650, 85], [650, 170], [920, 85], [920, 170]];
    let s = f.map(([x, y], i) => `<path class="wire" d="M785 128L${x} ${y}" style="animation-delay:${i * 0.3}s"/>`).join('');
    s += f.map(([x, y], i) => chip(x - 40, y - 12, 80, 24, 'follower', C.violet, lit(i * 0.8), 11)).join('');
    s += `<circle cx="785" cy="128" r="20" fill="none" stroke="${C.accent}" class="pulse"/>` + chip(740, 114, 90, 28, 'leader', C.accent, '', 13);
    return s + tm(785, 200, 'heartbeat · append entries · commit on quorum', 11, C.faint);
  },
  clock() { // channel deltas -> 7-state health
    let s = [80, 115, 150].map((y, i) => `<path class="wire" style="animation-delay:${i * 0.4}s" d="M615 ${y}c15 -14 30 14 45 0s30 14 45 0s30 -14 45 0s30 14 45 0"/>${t(615, y - 12, 'ch ' + i, 10, C.faint)}`).join('');
    s += '<path class="wire" d="M800 115H835"/>';
    s += Array.from({ length: 7 }, (_, i) => `<rect x="845" y="${66 + i * 17}" width="100" height="13" rx="6" fill="${C.node}" stroke="${i === 3 ? C.amber : C.accent}" ${lit(i * 0.8)}/>`).join('');
    return s + t(616, 200, 'per-channel deltas → agent time + 7-state health', 11, C.faint);
  },
  layers() { // image → conv stack → embedding
    let s = Array.from({ length: 16 }, (_, i) => `<rect x="${620 + (i % 4) * 14}" y="${92 + Math.floor(i / 4) * 14}" width="12" height="12" fill="${(i * 7) % 5 > 1 ? C.violet : C.dim}"/>`).join('');
    s += [[700, 76, 26, 100], [745, 88, 22, 76], [785, 100, 18, 52]].map(([x, y, w, h], i) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${C.node}" stroke="${C.accent}" ${lit(i * 1)}/>`).join('');
    s += '<path class="wire" d="M808 126H840"/>';
    s += Array.from({ length: 8 }, (_, i) => `<rect x="${848 + i * 13}" y="116" width="10" height="20" rx="2" fill="${C.accent}" ${lit(3 + i * 0.2)}/>`).join('');
    return s + t(616, 200, 'pixels → SIMD convolutions → image embedding', 11, C.faint);
  },
  spikes() { // spike-train raster
    let s = [78, 106, 134, 162].map((y, i) => `${dot(630, y, 7, C.violet, lit(i * 0.9))}<path class="spk" stroke="${i % 2 ? C.amber : C.accent}" stroke-dasharray="${[2, 3, 2, 4][i]} ${[22, 15, 30, 18][i]}" d="M645 ${y}H950"/><path d="M645 ${y}H950" stroke="${C.dim}" stroke-width=".5"/>`).join('');
    return s + `<path d="M645 ${78 - 8}V170" stroke="${C.dim}"/>` + t(616, 200, 'leaky integrate-and-fire neurons · STDP learning', 11, C.faint);
  },
  dag() { // topological order + critical path
    const n = [[630, 128], [710, 80], [710, 176], [800, 128], [880, 80], [940, 140]];
    const e = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 5]];
    let s = e.map(([a, b]) => line(n[a][0], n[a][1], n[b][0], n[b][1])).join('');
    s += '<path class="wire" style="stroke:#ffbe73" d="M630 128L710 176L800 128L880 80L940 140"/>';
    s += n.map(([x, y], i) => `<g ${lit(i * 0.7)}>${dot(x, y, 11, C.node, `stroke="${C.accent}"`)}${tm(x, y + 4, String(i + 1), 11, C.text)}</g>`).join('');
    return s + t(616, 206, 'topological order · critical path', 11, C.faint);
  },
  burst() { // traffic spike, capacity follows
    const v = [20, 22, 25, 24, 70, 96, 88, 30, 26, 22];
    let s = v.map((h, i) => `<rect x="${625 + i * 32}" y="${180 - h}" width="22" height="${h}" rx="3" fill="${h > 60 ? C.amber : C.violet}" ${h > 60 ? dly('grow', 0) : ''}/>`).join('');
    s += `<path class="wire" d="M620 150H740V78H850V150H950"/>`;
    return s + t(616, 200, '10–50× spike → predictive + reactive scale-out', 11, C.faint);
  },
  ocr() { // scan a page, emit LaTeX / MathML
    let s = `<rect x="625" y="66" width="120" height="120" rx="6" fill="${C.node}" stroke="${C.sub}"/>`;
    s += [84, 100, 132, 148, 164].map((y) => `<path d="M640 ${y}H${y === 132 ? 690 : 730}" stroke="${C.dim}" stroke-width="3"/>`).join('');
    s += tm(712, 136, 'a/b', 12, C.text) + `<path class="scan" d="M628 74H742" stroke="${C.accent}" stroke-width="2"/>`;
    s += '<path class="wire" d="M755 126H790"/>';
    s += chip(800, 84, 150, 30, '\\frac{a}{b}', C.accent, lit(0), 13) + chip(800, 132, 150, 30, '<mfrac>…</mfrac>', C.violet, lit(2), 12);
    return s + t(616, 202, 'equations → LaTeX + MathML', 11, C.faint);
  },
  mcp() { // agents ↔ MCP server ↔ RVF store
    let s = ['Claude Code', 'Cursor', 'any MCP client'].map((a, i) => chip(612, 80 + i * 38, 104, 26, a, C.violet, lit(i * 1.2), 11)).join('');
    s += [93, 131, 169].map((y, i) => `<path class="wire" style="animation-delay:${i * 0.4}s" d="M716 ${y}C745 ${y} 745 132 770 132"/>`).join('');
    s += chip(770, 110, 96, 44, 'MCP server', C.accent, '', 12) + '<path class="wire" d="M866 132H884"/>';
    return s + cyl(888, 114, 66, 30, 'RVF', C.amber) + tm(818, 176, 'stdio · SSE', 11, C.faint);
  },
  brain() { // sessions share into a shared brain
    const p = [[640, 80], [640, 176], [785, 74], [930, 80], [930, 176]];
    let s = p.map(([x, y], i) => `<path class="wire" style="animation-delay:${i * 0.5}s" d="M${x} ${y}L785 128"/>`).join('');
    s += p.map(([x, y], i) => chip(x - 34, y - 11, 68, 22, 'session', C.violet, lit(i * 1), 10)).join('');
    s += `<circle cx="785" cy="128" r="22" fill="none" stroke="${C.accent}" class="pulse"/>` + dot(785, 128, 20, C.node, `stroke="${C.accent}"`) + tm(785, 135, 'π', 20, C.accent);
    return s + tm(785, 204, 'share · search · transfer, with a witness chain', 11, C.faint);
  },
  pipeline({ cap }) { // screen frames → embed → memory → search
    let s = [0, 1, 2].map((i) => `<rect x="${618 + i * 8}" y="${78 + i * 8}" width="70" height="48" rx="4" fill="${C.node}" stroke="${C.sub}" ${lit(i * 0.6)}/>`).join('');
    s += '<path class="wire" d="M710 110H740"/>' + chip(744, 96, 66, 28, 'embed', C.accent, lit(2), 12);
    s += '<path class="wire" d="M810 110H838"/>';
    s += Array.from({ length: 9 }, (_, i) => dot(852 + (i % 3) * 22, 88 + Math.floor(i / 3) * 22, 5, i === 4 ? C.amber : C.accent, lit(3 + i * 0.25))).join('');
    return s + t(616, 196, cap, 11, C.faint);
  },
  filter() { // predicate-filtered HNSW search
    const n = Array.from({ length: 15 }, (_, i) => [640 + (i % 5) * 75, 82 + Math.floor(i / 5) * 42]);
    const ok = new Set([0, 2, 6, 8, 9, 12, 14]);
    let s = n.map(([x, y], i) => dot(x, y, 6, ok.has(i) ? C.accent : C.dim)).join('');
    s += '<path class="wire" d="M640 82L715 124L790 166L865 124L940 124L940 166"/>';
    s += [0, 6, 12, 8, 9, 14].map((i, k) => `<circle cx="${n[i][0]}" cy="${n[i][1]}" r="10" fill="none" stroke="${C.amber}" ${lit(k * 0.8)}/>`).join('');
    return s + t(616, 200, 'filter: lang = rust — search stays on matching nodes', 11, C.faint);
  },
  synth() { // schema → generated rows
    let s = chip(616, 92, 100, 66, '{ schema }', C.violet, '', 13) + '<path class="wire" d="M716 125H752"/>';
    s += Array.from({ length: 5 }, (_, i) => `<g ${lit(i * 0.9)}><rect x="760" y="${70 + i * 22}" width="190" height="16" rx="4" fill="${C.node}" stroke="${C.accent}"/><path d="M768 ${78 + i * 22}H${800 + (i * 37) % 60}M${850} ${78 + i * 22}H${880 + (i * 23) % 50}" stroke="${C.sub}" stroke-width="3"/></g>`).join('');
    return s + t(616, 200, 'schema-driven synthetic records for training + RAG', 11, C.faint);
  },
  swarm() { // regional agents under one coordinator
    const R = [['us-east', 655], ['eu-west', 785], ['asia-east', 915]];
    let s = R.map(([, x], i) => `<path class="wire" style="animation-delay:${i * 0.5}s" d="M785 88L${x} 140"/>`).join('');
    s += R.map(([n, x], i) => `<g ${lit(i * 1.5)}><rect x="${x - 52}" y="132" width="104" height="44" rx="10" fill="${C.node}" stroke="${C.violet}"/>${[0, 1, 2].map((k) => dot(x - 22 + k * 22, 147, 5, C.accent)).join('')}${tm(x, 169, n, 10, C.sub)}</g>`).join('');
    return s + chip(725, 66, 120, 26, 'coordinator', C.accent, '', 12) + tm(785, 202, 'regional agents · failover · consensus', 11, C.faint);
  },
  llm() { // tokens → chat template → two-tier KV cache
    let s = ['<|user|>', 'hello'].map((k, i) => chip(612 + i * 68, 72, 62, 24, k, C.violet, lit(i * 0.6), 10)).join('');
    s += '<path class="wire" d="M680 100V114"/>' + chip(624, 116, 112, 28, 'chat template', C.accent, '', 11);
    s += '<path class="wire" d="M736 130H770"/>';
    s += Array.from({ length: 12 }, (_, i) => `<rect x="${778 + (i % 6) * 29}" y="${i < 6 ? 98 : 134}" width="24" height="24" rx="4" fill="${i < 6 ? C.accent : C.violet}" ${lit(1 + i * 0.35)}/>`).join('');
    s += t(778, 92, 'fp32 tail', 10, C.faint) + t(778, 172, 'u8 quantized store', 10, C.faint);
    return s + t(612, 202, 'KV cache · chat templates · HNSW routing · MicroLoRA', 11, C.faint);
  },
  sql() { // CLI → Postgres
    let s = `<rect x="612" y="74" width="230" height="104" rx="8" fill="#08101e" stroke="${C.dim}"/>`;
    s += [['$ ruvector-pg search', C.accent], ['SELECT id FROM docs', C.text], ['ORDER BY v <=> $1', C.text], ['LIMIT 5;', C.text]].map(([l, c], i) => `<g ${lit(i * 0.8)}>${t(624, 98 + i * 22, l, 12, c)}</g>`).join('');
    s += '<path class="wire" d="M846 126H870"/>';
    return s + cyl(876, 104, 76, 40, 'Postgres', C.violet) + t(612, 202, 'pgvector-compatible SQL from the terminal', 11, C.faint);
  },
  bot() { // chat with vector memory
    let s = [[616, 70, 160, 'How do I deploy?', C.violet], [700, 104, 160, 'Recalled 3 memories…', C.accent], [616, 138, 130, 'Thanks!', C.violet]].map(([x, y, w, m, c], i) => chip(x, y, w, 26, m, c, lit(i * 1.5), 11)).join('');
    s += '<path class="wire" d="M864 117H886"/>';
    return s + cyl(890, 100, 64, 36, 'memory', C.amber) + t(616, 200, 'self-learning assistant · vector memory · multi-LLM', 11, C.faint);
  },
  extensions() { // modules plugging into ruvector
    const m = [['embeddings', 650, 78], ['persistence', 920, 78], ['exports', 650, 176], ['temporal', 920, 176], ['UI', 785, 196]];
    let s = m.map(([, x, y], i) => `<path class="wire" style="animation-delay:${i * 0.4}s" d="M${x} ${y}L785 128"/>`).join('');
    s += m.map(([l, x, y], i) => chip(x - 48, y - 12, 96, 24, l, C.violet, lit(i * 1), 11)).join('');
    return s + chip(735, 112, 100, 32, 'ruvector', C.accent, '', 13);
  },
};

/** Render one motif by key; throws on an unknown key so typos fail loudly. */
export function renderMotif(key, opts = {}) {
  const fn = M[key];
  if (!fn) throw new Error(`unknown motif: ${key}`);
  return fn(opts);
}
export { esc };
