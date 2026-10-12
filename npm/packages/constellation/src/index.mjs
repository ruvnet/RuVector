import { createHash } from 'node:crypto';

export const REGISTRY = 'https://registry.npmjs.org';
export const DEPENDENCY_KINDS = ['dependencies', 'optionalDependencies', 'peerDependencies', 'devDependencies'];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

export async function jsonRequest(url, options = {}, fetcher = fetch) {
  for (let attempt = 0; attempt < 4; attempt++) {
    const response = await fetcher(url, { ...options, signal: AbortSignal.timeout(30_000) });
    if (response.ok) return response.json();
    if ((response.status === 429 || response.status >= 500) && attempt < 3) {
      await sleep(500 * 2 ** attempt);
      continue;
    }
    throw new Error(`HTTP ${response.status}: ${new URL(url).origin}${new URL(url).pathname}`);
  }
}

export async function discoverPackages(maintainer = 'ruvnet', request = jsonRequest) {
  if (!/^[a-z0-9][a-z0-9._-]*$/i.test(maintainer)) throw new Error('Invalid maintainer');
  const names = new Set();
  let expected = 0;
  for (let offset = 0; ; offset += 250) {
    const result = await request(`${REGISTRY}/-/v1/search?text=${encodeURIComponent(`maintainer:${maintainer}`)}&size=250&from=${offset}`);
    if (!Array.isArray(result.objects) || !Number.isSafeInteger(result.total)) throw new Error('Invalid registry search response');
    expected = result.total;
    for (const entry of result.objects) names.add(entry.package.name);
    if (offset + result.objects.length >= expected) break;
    if (!result.objects.length) throw new Error('Registry search ended before total');
  }
  return { names: [...names].sort(), searchTotal: expected, uniqueCount: names.size, complete: names.size === expected };
}

export function normalizePackage(packument, maintainer = 'ruvnet') {
  const version = packument['dist-tags']?.latest;
  const manifest = packument.versions?.[version];
  if (!manifest) throw new Error(`No latest manifest: ${packument.name}`);
  const maintained = packument.maintainers?.some(m => m.name === maintainer) === true;
  const repository = typeof manifest.repository === 'string' ? manifest.repository : manifest.repository?.url ?? null;
  const pkg = {
    name: manifest.name, version, maintained, repository,
    directory: manifest.repository?.directory ?? null,
    description: manifest.description ?? '', license: manifest.license ?? null,
    homepage: manifest.homepage ?? null, keywords: manifest.keywords ?? [],
    deprecated: manifest.deprecated ?? null, bin: manifest.bin ?? {},
    engines: manifest.engines ?? {}, os: manifest.os ?? [], cpu: manifest.cpu ?? [],
    files: manifest.files ?? null,
    skills: manifest.skills ?? manifest.pi?.skills ?? null,
    integrity: manifest.dist?.integrity ?? null,
    attestations: Boolean(manifest.dist?.attestations),
    publishedAt: packument.time?.[version] ?? null,
    lifecycleScripts: Object.keys(manifest.scripts ?? {}).filter(k => ['preinstall','install','postinstall','prepare','prepublishOnly'].includes(k)),
    peerDependenciesMeta: manifest.peerDependenciesMeta ?? {},
    ...Object.fromEntries(DEPENDENCY_KINDS.map(k => [k, manifest[k] ?? {}])),
  };
  pkg.findings = [];
  if (!maintained) pkg.findings.push('maintainer-unverified');
  if (!pkg.repository) pkg.findings.push('missing-repository');
  if (!pkg.license) pkg.findings.push('missing-license');
  if (!pkg.integrity) pkg.findings.push('missing-integrity');
  if (!pkg.attestations) pkg.findings.push('no-registry-attestation');
  if (pkg.deprecated) pkg.findings.push('deprecated');
  if (pkg.lifecycleScripts.some(s => /install/.test(s))) pkg.findings.push('install-script-review');
  for (const kind of DEPENDENCY_KINDS) {
    for (const [name, range] of Object.entries(pkg[kind])) {
      if (/^(file:|link:|workspace:|git\+|https?:)/.test(range)) pkg.findings.push(`${kind}:non-registry:${name}`);
      if (range === '*' || range === 'latest') pkg.findings.push(`${kind}:unbounded:${name}`);
    }
  }
  return pkg;
}

export function buildGraph(packages) {
  const names = new Set(packages.map(p => p.name));
  const edges = [];
  for (const p of packages) for (const kind of DEPENDENCY_KINDS) {
    for (const [target, range] of Object.entries(p[kind] ?? {})) if (names.has(target)) {
      edges.push({ source: p.name, target, kind, range, optional: kind === 'optionalDependencies' || (kind === 'peerDependencies' && p.peerDependenciesMeta?.[target]?.optional === true) });
    }
  }
  // Strongly connected components on installed runtime edges; dev/peer graphs remain visible separately.
  const adjacency = new Map([...names].map(n => [n, []]));
  for (const e of edges) if (['dependencies','optionalDependencies'].includes(e.kind)) adjacency.get(e.source).push(e.target);
  let index = 0;
  const indices = new Map(), low = new Map(), stack = [], active = new Set(), cycles = [];
  function visit(n) {
    indices.set(n, index); low.set(n, index++); stack.push(n); active.add(n);
    for (const next of adjacency.get(n)) {
      if (!indices.has(next)) { visit(next); low.set(n, Math.min(low.get(n), low.get(next))); }
      else if (active.has(next)) low.set(n, Math.min(low.get(n), indices.get(next)));
    }
    if (low.get(n) === indices.get(n)) {
      const component = []; let x;
      do { x = stack.pop(); active.delete(x); component.push(x); } while (x !== n);
      if (component.length > 1 || adjacency.get(n).includes(n)) cycles.push(component.sort());
    }
  }
  for (const n of names) if (!indices.has(n)) visit(n);
  return { nodes: [...names].sort(), edges, runtimeCycles: cycles };
}

export function compareReports(current, previous) {
  const old = new Map((previous?.packages ?? []).map(p => [p.name, p]));
  return {
    baselineAvailable: Boolean(previous),
    added: current.packages.filter(p => !old.has(p.name)).map(p => p.name),
    changed: current.packages.filter(p => old.has(p.name) && old.get(p.name).version !== p.version).map(p => ({name:p.name, before:old.get(p.name).version, after:p.version})),
    removed: [...old.keys()].filter(n => !current.packages.some(p => p.name === n)),
    resolvedFindings: current.packages.flatMap(p => (old.get(p.name)?.findings ?? []).filter(f => !p.findings.includes(f)).map(f => ({name:p.name, finding:f}))),
  };
}

export function planImprovements(report, previous = null) {
  const incoming = new Map();
  for (const e of report.graph.edges) if (e.kind !== 'devDependencies') incoming.set(e.target, (incoming.get(e.target) ?? 0) + 1);
  const queue = report.packages.map(p => {
    const advisories = report.advisories?.[p.name] ?? [];
    const severity = advisories.reduce((n,a) => Math.max(n, ({critical:100,high:70,moderate:30,low:10})[a.severity] ?? 0), 0);
    const findings = p.findings.length;
    return { name:p.name, version:p.version, score:severity + Math.min(incoming.get(p.name) ?? 0,30) + findings * 2,
      incomingEdges:incoming.get(p.name) ?? 0, findings:p.findings, advisories:advisories.map(a => ({id:a.id,severity:a.severity,title:a.title,url:a.url})),
      nextAction:severity ? 'reproduce-advisory-and-patch' : findings ? 'inspect-package-and-repair-metadata-or-release-proof' : 'measure-sdk-mcp-skills-integration',
      publishAuthority:false };
  }).sort((a,b) => b.score-a.score || a.name.localeCompare(b.name));
  return { policy:'risk-and-impact-v1', learnedModel:false, delta:compareReports(report,previous), queue };
}

export async function sweep({ maintainer = 'ruvnet', concurrency = 8, request = jsonRequest, previous = null } = {}) {
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 16) throw new Error('concurrency must be 1..16');
  const discovery = await discoverPackages(maintainer, request);
  const packages = [], failures = [];
  let cursor = 0;
  await Promise.all(Array.from({length:concurrency}, async () => {
    while (cursor < discovery.names.length) {
      const name = discovery.names[cursor++];
      try {
        const packument = await request(`${REGISTRY}/${encodeURIComponent(name)}`);
        packages.push(normalizePackage(packument, maintainer));
      } catch (error) { failures.push({name,error:error.message}); }
    }
  }));
  packages.sort((a,b) => a.name.localeCompare(b.name));
  const report = { schemaVersion:1, observedAt:new Date().toISOString(), maintainer,
    coverage:{...discovery, names:undefined, fetched:packages.length, errors:failures, scope:'public-maintainer-search-latest-versions', privatePackages:'not-enumerated', transitiveAudit:false},
    packages, graph:buildGraph(packages), advisories:{}, advisoryStatus:'unknown' };
  try {
    report.advisories = await request(`${REGISTRY}/-/npm/v1/security/advisories/bulk`, { method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(Object.fromEntries(packages.map(p => [p.name,[p.version]]))) });
    report.advisoryStatus = 'queried-direct-package-versions';
  } catch (error) { report.advisoryError = error.message; }
  report.complete = discovery.complete && failures.length === 0 && report.advisoryStatus !== 'unknown';
  report.plan = planImprovements(report, previous);
  report.contentDigest = createHash('sha256').update(JSON.stringify(report.packages)).digest('hex');
  return report;
}

export function reportMarkdown(report) {
  const esc = value => String(value ?? '').replace(/[|\n\r]/g,' ');
  return [ '# RuvNet npm constellation', '', `Observed: ${report.observedAt}`, '',
    `Public packages: ${report.packages.length}/${report.coverage.searchTotal}. Complete scan: ${report.complete}. Runtime cycles: ${report.graph.runtimeCycles.length}.`,
    'Coverage is public maintainer search and latest manifests. Private packages, transitives, source security, and publish permissions are not certified.', '',
    '| Package | Version | Findings | Runtime ecosystem dependencies |', '|---|---|---|---|',
    ...report.packages.map(p => `| ${esc(p.name)} | ${esc(p.version)} | ${esc(p.findings.join(', '))} | ${esc(report.graph.edges.filter(e => e.source === p.name && e.kind !== 'devDependencies').map(e => `${e.target} (${e.kind})`).join(', '))} |`), '',
    '## Prioritized next work', '', ...report.plan.queue.slice(0,30).map(p => `- ${p.name}: ${p.nextAction}; priority ${p.score}.`), '',
    '## Runtime cycles', '', ...report.graph.runtimeCycles.map(c => `- ${c.join(' → ')}`), '',
  ].join('\n');
}
