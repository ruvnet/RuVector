import { accessSync, constants, statSync } from 'node:fs';
import { isAbsolute } from 'node:path';

const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value)
  && [Object.prototype, null].includes(Object.getPrototypeOf(value));
const finite = value => typeof value === 'number' && Number.isFinite(value);

/** Advisory routing only: consumes measured labels; never executes the selected model. */
export async function routeWithMetaHarness(
  { rows, prices, embedding, qualityBar = 0.8, k = 5 },
  { loader = () => import('@metaharness/router') } = {},
) {
  if (!finite(qualityBar) || qualityBar < 0 || qualityBar > 1) throw new TypeError('qualityBar must be in [0, 1]');
  if (!Number.isSafeInteger(k) || k < 1) throw new TypeError('k must be a positive safe integer');
  if (!Array.isArray(embedding) || embedding.length === 0 || !Array.from(embedding).every(finite)) throw new TypeError('embedding must be a nonempty finite vector');
  if (!Array.isArray(rows) || rows.length === 0 || !record(prices)) throw new TypeError('Measured rows and a price map are required');
  const ids = new Set();
  for (const row of rows) {
    if (!record(row) || !Array.isArray(row.embedding) || row.embedding.length !== embedding.length || !Array.from(row.embedding).every(finite)) throw new TypeError('All embeddings must have the same finite dimensions');
    if (!record(row.scores) || Object.keys(row.scores).length === 0) throw new TypeError('Each row needs measured quality labels');
    for (const [id, quality] of Object.entries(row.scores)) {
      if (!id.trim() || !finite(quality) || quality < 0 || quality > 1) throw new TypeError('Quality labels must be finite values in [0, 1]');
      if (!own(prices, id) || !finite(prices[id]) || prices[id] < 0) throw new TypeError(`Missing or invalid price for ${id}`);
      ids.add(id);
    }
  }
  for (const price of Object.values(prices)) if (!finite(price) || price < 0) throw new TypeError('Prices must be finite and nonnegative');
  let module;
  try { module = await loader(); } catch (cause) {
    if (cause?.code !== 'ERR_MODULE_NOT_FOUND' && cause?.code !== 'MODULE_NOT_FOUND') throw cause;
    throw Object.assign(new Error('Optional @metaharness/router integration is unavailable; install a compatible peer before routing', { cause }), { code: 'OPTIONAL_INTEGRATION_UNAVAILABLE' });
  }
  if (typeof module?.Router?.fromExamples !== 'function') throw new TypeError('Incompatible @metaharness/router API');
  const router = module.Router.fromExamples(structuredClone(rows), { ...prices }, { k, qualityBar });
  const result = router.route([...embedding]);
  if (!record(result) || !ids.has(result.id) || !finite(result.predictedQuality) || result.predictedQuality < 0 || result.predictedQuality > 1
      || !finite(result.costPerMTok) || result.costPerMTok !== prices[result.id]) throw new TypeError('Invalid router result');
  if (result.metBar !== true || result.predictedQuality < qualityBar) {
    throw Object.assign(new Error('No candidate clears the required quality bar'), { code: 'QUALITY_BAR_NOT_MET' });
  }
  return { id: result.id, predictedQuality: result.predictedQuality, costPerMTok: result.costPerMTok, metBar: true };
}

function absolutePath(value, label) {
  if (typeof value !== 'string' || !isAbsolute(value) || value.includes('\0')) throw new TypeError(`${label} must be an absolute path`);
}

/** Returns configuration only. Does not write configuration, install packages, or launch servers. */
export function createMcpConfiguration({ existing = {}, executables, reportPath }) {
  if (!record(existing) || (existing.mcpServers !== undefined && !record(existing.mcpServers))) throw new TypeError('existing must contain an object mcpServers map');
  if (!record(executables)) throw new TypeError('executables must be a map of installed executable paths');
  const supported = ['constellation', 'ruvector', 'ruflo'];
  if (Object.keys(executables).length === 0 || Object.keys(executables).some(key => !supported.includes(key))) throw new TypeError('Supply constellation, ruvector, or ruflo executable paths');
  const config = structuredClone(existing);
  config.mcpServers ??= {};
  for (const [kind, executable] of Object.entries(executables)) {
    absolutePath(executable, kind);
    if (!statSync(executable).isFile()) throw new TypeError(`${kind} executable must be a file`);
    accessSync(executable, constants.X_OK);
    const name = kind === 'constellation' ? 'ruvector-constellation' : kind;
    if (own(config.mcpServers, name)) throw new Error(`MCP server ${name} already exists; existing configuration was preserved`);
    if (kind === 'constellation') absolutePath(reportPath, 'reportPath');
    config.mcpServers[name] = { command: executable, args: kind === 'constellation' ? ['mcp', reportPath] : ['mcp', 'start'] };
  }
  return config;
}
