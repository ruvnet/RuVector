import { createHash, createHmac, timingSafeEqual } from 'node:crypto';
import { IntegrationError, loadOptionalPackage } from './optional-package';

const PACKAGE = '@ruvector/kge';
const FORMAT = 'ruvector-graph-reasoner-v1';
export interface GraphReasonerOptions { scorer?: 'hole' | 'rotate'; dims?: number; seed?: number }
export interface GraphTriple { s: string; r: string; o: string; split?: 'train' | 'valid' | 'test' | 'transfer' }
export type GraphPredictQuery = { r: string; k?: number; useIndex?: boolean } &
  ({ s: string; o?: never } | { o: string; s?: never });
export interface GraphPrediction { candidates: { entity: string; score: number }[]; exact: boolean; ann: boolean }
export interface GraphTrainingReport {
  epoch: number; epochs: number; batches: number; loss: number;
  entities: number; relations: number; triples: number; [key: string]: unknown;
}
export interface GraphSaveOptions { key?: string }
interface Client {
  backend: 'native' | 'wasm'; version: string;
  addTriples(triples: readonly GraphTriple[]): Record<string, number>;
  train(config?: Record<string, unknown>): Promise<GraphTrainingReport>;
  predict(query: GraphPredictQuery): GraphPrediction;
  similarRelations(query: { r: string; k?: number }): { relations: { relation: string; score: number }[] };
  compose(query: { r1: string; r2: string; s: string; k?: number }): GraphPrediction;
  save(): string;
  stats(): Record<string, unknown>;
}
interface Receipt { report: GraphTrainingReport; modelSha256: string }
interface Envelope { format: typeof FORMAT; model: string; training: Receipt; hmacSha256?: string }
const hash = (text: string): string => createHash('sha256').update(text).digest('hex');
function invalid(message: string): never {
  throw new IntegrationError('ERR_INTEGRATION_INVALID', PACKAGE, message);
}
function untrained(): never {
  throw new IntegrationError('ERR_INTEGRATION_UNTRAINED', PACKAGE,
    'Train this graph before reasoning, or load a trained save() from this adapter. Training completion does not establish prediction quality.');
}
function keyOf(options: GraphSaveOptions): string | undefined {
  if (!Object.prototype.hasOwnProperty.call(options, 'key')) return undefined;
  if (typeof options.key !== 'string' || Buffer.byteLength(options.key) < 16) {
    invalid('A signing key must be a string of at least 16 bytes.');
  }
  return options.key;
}
function bodyOf(envelope: Envelope): string {
  return JSON.stringify({ format: envelope.format, model: envelope.model, training: envelope.training });
}
function validateReport(report: GraphTrainingReport, stats: Record<string, unknown>): void {
  if (!report || !Number.isSafeInteger(report.epochs) || report.epochs < 1 ||
      report.epoch !== report.epochs - 1 || !Number.isSafeInteger(report.batches) || report.batches < 1 ||
      typeof report.loss !== 'number' || !Number.isFinite(report.loss)) invalid('Missing or invalid completed training report.');
  for (const field of ['entities', 'relations', 'triples'] as const) {
    if (!Number.isSafeInteger(report[field]) || report[field] < 1 || report[field] !== stats[field]) {
      invalid(`Training report ${field} does not match the loaded model.`);
    }
  }
}
/** Validate the actual upstream model state, not a caller-supplied trained flag. */
function validateTables(serialized: string): void {
  let model: any;
  try { model = JSON.parse(serialized).model; } catch { invalid('Invalid model JSON.'); }
  const dims = model?.config?.dims;
  if (model?.version !== 1 || !Number.isSafeInteger(dims) || dims <= 0 || dims % 2 ||
      !['hole', 'rotate'].includes(model?.config?.scorer)) invalid('Unsupported model schema or dimensions.');
  for (const field of ['entities', 'relations'] as const) {
    const labels = model?.[field]?.labels;
    const values = model?.tables?.[field];
    if (!Array.isArray(labels) || labels.length === 0 ||
        !labels.every((v: unknown) => typeof v === 'string') || new Set(labels).size !== labels.length ||
        model?.tables?.dims !== dims || !Array.isArray(values) || values.length !== labels.length * dims ||
        !values.every((v: unknown) => typeof v === 'number' && Number.isFinite(v))) {
      invalid(`Saved ${field} embeddings are missing or malformed.`);
    }
  }
  if (!Array.isArray(model.triples) || model.triples.length === 0) invalid('Saved graph has no training facts.');
  for (const triple of model.triples) {
    for (const [field, count] of [['s', model.entities.labels.length], ['o', model.entities.labels.length],
      ['r', model.relations.labels.length]] as const) {
      if (!Number.isSafeInteger(triple?.[field]) || triple[field] < 0 || triple[field] >= count) {
        invalid('Saved graph contains an out-of-range entity or relation reference.');
      }
    }
  }
}
async function api(): Promise<Record<string, any>> {
  const module = await loadOptionalPackage(PACKAGE);
  if (typeof module.createKge !== 'function' || typeof module.loadKge !== 'function') {
    throw new IntegrationError('ERR_INTEGRATION_API', PACKAGE, 'Installed @ruvector/kge must expose createKge and loadKge.');
  }
  return module;
}

/** Opt-in graph reasoning. Untrained models never produce rankings through this adapter.
 * Saved receipts establish consistency; use a trusted signing key for authenticity.
 */
export interface GraphReasoner {
  readonly backend: 'native' | 'wasm';
  readonly version: string;
  readonly trained: boolean;
  stats(): Record<string, unknown>;
  addTriples(triples: readonly GraphTriple[]): Record<string, number>;
  train(config?: Record<string, unknown>): Promise<GraphTrainingReport>;
  predict(query: GraphPredictQuery): GraphPrediction;
  similarRelations(query: { r: string; k?: number }): { relations: { relation: string; score: number }[] };
  compose(query: { r1: string; r2: string; s: string; k?: number }): GraphPrediction;
  save(options?: GraphSaveOptions): string;
}
class LocalGraphReasoner implements GraphReasoner {
  private training?: Receipt;
  private busy = false;
  /** @internal Use createGraphReasoner or loadGraphReasoner. */
  constructor(private readonly client: Client, training?: Receipt) { this.training = training; }
  get backend(): 'native' | 'wasm' { return this.client.backend; }
  get version(): string { return this.client.version; }
  get trained(): boolean { return !!this.training && !this.busy; }
  stats(): Record<string, unknown> { return { ...this.client.stats(), trained: this.trained }; }
  addTriples(triples: readonly GraphTriple[]): Record<string, number> {
    this.assertIdle();
    this.training = undefined;
    return this.client.addTriples(triples);
  }
  async train(config?: Record<string, unknown>): Promise<GraphTrainingReport> {
    this.assertIdle();
    this.training = undefined;
    this.busy = true;
    try {
      const report = await this.client.train(config);
      const model = this.client.save();
      validateTables(model);
      validateReport(report, this.client.stats());
      // Clone the receipt so modifying a returned report cannot authorize a save.
      this.training = { report: JSON.parse(JSON.stringify(report)), modelSha256: hash(model) };
      return report;
    } finally { this.busy = false; }
  }
  predict(query: GraphPredictQuery): GraphPrediction { this.assertTrained(); return this.client.predict(query); }
  similarRelations(query: { r: string; k?: number }): { relations: { relation: string; score: number }[] } {
    this.assertTrained(); return this.client.similarRelations(query);
  }
  compose(query: { r1: string; r2: string; s: string; k?: number }): GraphPrediction {
    this.assertTrained(); return this.client.compose(query);
  }
  save(options: GraphSaveOptions = {}): string {
    this.assertTrained();
    const key = keyOf(options);
    const model = this.client.save();
    if (hash(model) !== this.training!.modelSha256) invalid('Model changed since training; retrain before saving.');
    const envelope: Envelope = { format: FORMAT, model, training: this.training! };
    if (key !== undefined) envelope.hmacSha256 = createHmac('sha256', key).update(bodyOf(envelope)).digest('hex');
    return JSON.stringify(envelope);
  }
  private assertIdle(): void { if (this.busy) invalid('Training is in progress; wait before modifying the graph.'); }
  private assertTrained(): void { if (!this.trained) untrained(); }
}

export async function createGraphReasoner(options: GraphReasonerOptions = {}): Promise<GraphReasoner> {
  const module = await api();
  // Forward only documented model options; never accept a caller trained flag or binding override.
  return new LocalGraphReasoner(module.createKge({ scorer: options.scorer, dims: options.dims, seed: options.seed }));
}

/** Only adapter saves carry training evidence; upstream raw saves have no trained marker. */
export async function loadGraphReasoner(serialized: string, options: GraphSaveOptions = {}): Promise<GraphReasoner> {
  const key = keyOf(options);
  let envelope: Envelope;
  try { envelope = JSON.parse(serialized); } catch { invalid('Invalid graph reasoner JSON.'); }
  if (!envelope || envelope.format !== FORMAT || typeof envelope.model !== 'string' || !envelope.training) {
    invalid('Expected a trained graph reasoner save; raw @ruvector/kge saves do not contain training evidence.');
  }
  if (Object.prototype.hasOwnProperty.call(envelope, 'hmacSha256') && key === undefined) {
    invalid('This graph reasoner save is signed; supply its signing key to verify it.');
  }
  if (key !== undefined) {
    const expected = createHmac('sha256', key).update(bodyOf(envelope)).digest('hex');
    const actual = envelope.hmacSha256;
    if (typeof actual !== 'string' || !/^[a-f0-9]{64}$/.test(actual) ||
        !timingSafeEqual(Buffer.from(actual), Buffer.from(expected))) invalid('Graph reasoner signature is missing or invalid.');
  }
  if (hash(envelope.model) !== envelope.training.modelSha256) invalid('Training evidence does not match the model.');
  validateTables(envelope.model);
  const module = await api();
  const client: Client = module.loadKge(envelope.model); // validates upstream model checksum
  const actual = client.save();
  if (hash(actual) !== envelope.training.modelSha256) invalid('Loaded model differs from its training evidence.');
  validateReport(envelope.training.report, client.stats());
  return new LocalGraphReasoner(client, envelope.training);
}
