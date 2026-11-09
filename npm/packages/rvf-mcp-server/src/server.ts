/**
 * RVF MCP Server — core server implementation.
 *
 * Registers all RVF tools, resources, and prompts with the MCP SDK. Stores
 * are real on-disk `.rvf` files (via @ruvector/rvf) confined to a data dir,
 * so queries are index-backed and data survives server restarts.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { RvfDatabase } from '@ruvector/rvf';
import * as fs from 'node:fs';
import { z } from 'zod';
import {
  META_SUFFIX,
  MetadataLog,
  defaultDataDir,
  matchesFilter,
  resolveStorePath,
  storedMetric,
  type Metadata,
} from './stores.js';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface RvfMcpServerOptions {
  /** Server name shown to MCP clients. Default: 'rvf-mcp-server'. */
  name?: string;
  /** Server version. Default: '0.1.0'. */
  version?: string;
  /** Default vector dimensions for new stores. Default: 128. */
  defaultDimensions?: number;
  /** Maximum open stores. Default: 64. */
  maxStores?: number;
  /**
   * Directory that holds every store. Tool `path` arguments are resolved
   * inside it and may not escape it. Default: $RVF_MCP_DATA_DIR, else
   * ~/.rvf-mcp-server.
   */
  dataDir?: string;
  /** Auto-compaction thresholds after ingest. Default: {@link AUTO_COMPACT}. */
  autoCompact?: { minBytes: number; ratio: number };
}

/** Upper bounds on model-supplied sizes (tool arguments are untrusted). */
export const LIMITS = {
  maxDimensions: 65_536,
  maxK: 10_000,
  maxBatch: 10_000,
} as const;

/** Auto-compaction after ingest: file larger than `minBytes` AND `ratio` x live vector bytes. */
export const AUTO_COMPACT = {
  minBytes: 64 * 1024 * 1024,
  ratio: 8,
} as const;

const vectorSchema = z.array(z.number()).max(LIMITS.maxDimensions);
const metadataSchema = z.record(z.string(), z.union([z.string(), z.number(), z.boolean()]));

type Metric = 'l2' | 'cosine' | 'dotproduct';

interface StoreHandle {
  id: string;
  path: string;
  file: string;
  dimensions: number;
  metric: string;
  readOnly: boolean;
  db: RvfDatabase;
  meta: MetadataLog;
  createdAt: number;
}

type ToolResult = { content: Array<{ type: 'text'; text: string }> };

const reply = (body: unknown): ToolResult => ({
  content: [{ type: 'text' as const, text: JSON.stringify(body, null, 2) }],
});
const fail = (message: string): ToolResult => ({
  content: [{ type: 'text' as const, text: `Error: ${message}` }],
});
const errMessage = (err: unknown): string => (err instanceof Error ? err.message : String(err));

// ─── Server ─────────────────────────────────────────────────────────────────

export class RvfMcpServer {
  readonly mcp: McpServer;
  private stores = new Map<string, StoreHandle>();
  private nextId = 1;
  private opts: Required<RvfMcpServerOptions>;

  constructor(options?: RvfMcpServerOptions) {
    this.opts = {
      name: options?.name ?? 'rvf-mcp-server',
      version: options?.version ?? '0.1.0',
      defaultDimensions: options?.defaultDimensions ?? 128,
      maxStores: options?.maxStores ?? 64,
      dataDir: options?.dataDir ?? defaultDataDir(),
      autoCompact: options?.autoCompact ?? AUTO_COMPACT,
    };

    this.mcp = new McpServer(
      { name: this.opts.name, version: this.opts.version },
      {
        capabilities: {
          resources: {},
          tools: {},
          prompts: {},
        },
      },
    );

    this.registerTools();
    this.registerResources();
    this.registerPrompts();
  }

  /** Directory that holds this server's stores. */
  get dataDir(): string {
    return this.opts.dataDir;
  }

  private withStore(
    storeId: string,
    write: boolean,
    fn: (h: StoreHandle) => Promise<ToolResult>,
  ): Promise<ToolResult> {
    const handle = this.stores.get(storeId);
    if (!handle) return Promise.resolve(fail(`store ${storeId} not found`));
    if (write && handle.readOnly) return Promise.resolve(fail('store is read-only'));
    return fn(handle).catch((err) => fail(errMessage(err)));
  }

  private register(handle: Omit<StoreHandle, 'id' | 'createdAt'>): StoreHandle {
    const full: StoreHandle = { ...handle, id: `store_${this.nextId++}`, createdAt: Date.now() };
    this.stores.set(full.id, full);
    return full;
  }

  /** k-NN with exact-match metadata filtering applied to an over-fetched result. */
  private async search(h: StoreHandle, vector: number[], k: number, filter?: Metadata) {
    const q = new Float32Array(vector);
    if (!filter) {
      const results = await h.db.query(q, k);
      return { results, scanned: results.length };
    }
    const candidates = [...h.meta.entries.values()].filter((m) => matchesFilter(m, filter)).length;
    if (candidates === 0) return { results: [], scanned: 0 };
    const total = (await h.db.status()).totalVectors;
    let fetch = Math.min(total, Math.max(k * 4, 64));
    for (;;) {
      const raw = await h.db.query(q, fetch);
      const hits = raw.filter((r) => matchesFilter(h.meta.entries.get(r.id), filter));
      if (hits.length >= Math.min(k, candidates) || fetch >= total) {
        return { results: hits.slice(0, k), scanned: raw.length };
      }
      fetch = Math.min(total, fetch * 4);
    }
  }

  // ─── Tool Registration ──────────────────────────────────────────────────

  private registerTools(): void {
    // ── rvf_create_store ──────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_create_store',
      'Create a new RVF vector store at the given path (relative to the server data dir)',
      {
        path: z.string().describe('File path for the new .rvf store, relative to the data dir'),
        dimensions: z.number().int().positive().max(LIMITS.maxDimensions).describe('Vector dimensionality'),
        metric: z.enum(['l2', 'cosine', 'dotproduct']).default('l2').describe('Distance metric'),
      },
      async ({ path, dimensions, metric }) => {
        if (this.stores.size >= this.opts.maxStores) {
          return fail(`max stores (${this.opts.maxStores}) reached`);
        }
        try {
          const file = resolveStorePath(this.opts.dataDir, path, true);
          const meta = new MetadataLog(file + META_SUFFIX, false);
          if (fs.existsSync(meta.file)) throw new Error(`${path} already has metadata; use rvf_open_store`);
          const db = await RvfDatabase.create(file, { dimensions, metric: metric as Metric }, 'node');
          meta.header = { dimensions, metric };
          meta.rewrite();
          const h = this.register({ path, file, dimensions, metric, readOnly: false, db, meta });
          return reply({ storeId: h.id, path, dimensions, metric, status: 'created' });
        } catch (err) {
          return fail(errMessage(err));
        }
      },
    );

    // ── rvf_open_store ────────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_open_store',
      'Open an existing RVF store (relative to the server data dir) for reading and writing',
      {
        path: z.string().describe('Path to existing .rvf file, relative to the data dir'),
        readOnly: z.boolean().default(false).describe('Open in read-only mode'),
      },
      async ({ path, readOnly }) => {
        if (this.stores.size >= this.opts.maxStores) {
          return fail(`max stores (${this.opts.maxStores}) reached`);
        }
        try {
          const file = resolveStorePath(this.opts.dataDir, path, false);
          if (!fs.existsSync(file)) throw new Error(`store ${path} does not exist`);
          const db = readOnly
            ? await RvfDatabase.openReadonly(file, 'node')
            : await RvfDatabase.open(file, 'node');
          const meta = new MetadataLog(file + META_SUFFIX, readOnly);
          try {
            meta.load();
          } catch (err) {
            await db.close();
            throw err;
          }
          const dimensions = await db.dimension();
          // The file is the source of truth; a store built outside this server has no sidecar.
          const metric = storedMetric(db) ?? meta.header?.metric ?? 'unknown';
          meta.header ??= { dimensions, metric };
          const h = this.register({ path, file, dimensions, metric, readOnly, db, meta });
          return reply({ storeId: h.id, path, readOnly, dimensions, metric, status: 'opened' });
        } catch (err) {
          return fail(errMessage(err));
        }
      },
    );

    // ── rvf_close_store ───────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_close_store',
      'Close an open RVF store, releasing the writer lock',
      {
        storeId: z.string().describe('Store ID returned by create/open'),
      },
      async ({ storeId }) => this.withStore(storeId, false, async (h) => {
        this.stores.delete(storeId);
        await this.closeHandle(h);
        return reply({ storeId, status: 'closed', path: h.path });
      }),
    );

    // ── rvf_ingest ────────────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_ingest',
      'Insert vectors into an RVF store',
      {
        storeId: z.string().describe('Target store ID'),
        entries: z.array(z.object({
          id: z.string().describe('Unique vector ID'),
          vector: vectorSchema.describe('Embedding vector (must match store dimensions)'),
          metadata: metadataSchema.optional()
            .describe('Optional metadata key-value pairs'),
        })).max(LIMITS.maxBatch).describe('Vectors to insert'),
      },
      async ({ storeId, entries }) => this.withStore(storeId, true, async (h) => {
        // Last write wins for duplicate ids within a batch (upsert semantics).
        const batch = new Map<string, { vector: number[]; metadata?: Metadata }>();
        let rejected = 0;
        for (const e of entries) {
          if (e.vector.length !== h.dimensions) rejected++;
          else batch.set(e.id, { vector: e.vector, metadata: e.metadata });
        }
        if (batch.size > 0) {
          await h.db.ingestBatch([...batch].map(([id, e]) => ({ id, vector: new Float32Array(e.vector) })));
          const sets: Array<[string, Metadata]> = [];
          const cleared: string[] = [];
          for (const [id, e] of batch) {
            if (e.metadata && Object.keys(e.metadata).length > 0) sets.push([id, e.metadata]);
            else cleared.push(id);
          }
          h.meta.apply(sets, cleared);
        }
        let status = await h.db.status();
        // rvf-runtime appends a manifest listing every segment on each ingest,
        // so many small ingest calls grow the file quadratically (2,500 single
        // ingests -> 231 MB for 1.3 MB of vectors). Bound it by compacting
        // once the file is both large and mostly overhead. Compaction drops
        // the in-memory HNSW, which is rebuilt lazily on the next query.
        const liveBytes = status.totalVectors * h.dimensions * 4;
        let autoCompacted = false;
        // Stat the file directly: status().fileSizeBytes reads 0 in @ruvector/rvf <= 0.4.2.
        const fileBytes = (await fs.promises.stat(h.file)).size;
        const { minBytes, ratio } = this.opts.autoCompact;
        if (fileBytes > minBytes && fileBytes > ratio * Math.max(liveBytes, 1)) {
          await h.db.compact();
          status = await h.db.status();
          autoCompacted = true;
        }
        return reply({ accepted: batch.size, rejected, totalVectors: status.totalVectors, autoCompacted });
      }),
    );

    // ── rvf_query ─────────────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_query',
      'k-NN vector similarity search',
      {
        storeId: z.string().describe('Store ID to query'),
        vector: vectorSchema.describe('Query embedding vector'),
        k: z.number().int().positive().max(LIMITS.maxK).default(10).describe('Number of nearest neighbors'),
        filter: metadataSchema.optional()
          .describe('Metadata filter (exact match on fields)'),
      },
      async ({ storeId, vector, k, filter }) => this.withStore(storeId, false, async (h) => {
        if (vector.length !== h.dimensions) {
          return fail(`dimension mismatch (query=${vector.length}, store=${h.dimensions})`);
        }
        const { results, scanned } = await this.search(h, vector, k, filter);
        return reply({ results, totalScanned: scanned, metric: h.metric });
      }),
    );

    // ── rvf_delete ────────────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_delete',
      'Delete vectors by their IDs',
      {
        storeId: z.string().describe('Store ID'),
        ids: z.array(z.string()).max(LIMITS.maxBatch).describe('Vector IDs to delete'),
      },
      async ({ storeId, ids }) => this.withStore(storeId, true, async (h) => {
        const { deleted } = await h.db.delete(ids);
        h.meta.apply([], ids);
        const status = await h.db.status();
        return reply({ deleted, remaining: status.totalVectors });
      }),
    );

    // ── rvf_delete_filter ─────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_delete_filter',
      'Delete vectors matching a metadata filter',
      {
        storeId: z.string().describe('Store ID'),
        filter: metadataSchema
          .describe('Metadata filter — all matching vectors will be deleted'),
      },
      async ({ storeId, filter }) => this.withStore(storeId, true, async (h) => {
        const ids = [...h.meta.entries].filter(([, m]) => matchesFilter(m, filter)).map(([id]) => id);
        const deleted = ids.length > 0 ? (await h.db.delete(ids)).deleted : 0;
        h.meta.apply([], ids);
        const status = await h.db.status();
        return reply({ deleted, remaining: status.totalVectors });
      }),
    );

    // ── rvf_compact ───────────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_compact',
      'Compact store to reclaim dead space from deleted vectors',
      {
        storeId: z.string().describe('Store ID'),
      },
      async ({ storeId }) => this.withStore(storeId, true, async (h) => {
        const result = await h.db.compact();
        h.meta.rewrite();
        const status = await h.db.status();
        return reply({
          storeId,
          compacted: true,
          segmentsCompacted: result.segmentsCompacted,
          bytesReclaimed: result.bytesReclaimed,
          totalVectors: status.totalVectors,
        });
      }),
    );

    // ── rvf_status ────────────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_status',
      'Get the current status of an RVF store',
      {
        storeId: z.string().describe('Store ID'),
      },
      async ({ storeId }) => this.withStore(storeId, false, async (h) => {
        const status = await h.db.status();
        return reply({
          storeId: h.id,
          path: h.path,
          dimensions: h.dimensions,
          metric: h.metric,
          readOnly: h.readOnly,
          totalVectors: status.totalVectors,
          totalSegments: status.totalSegments,
          fileSizeBytes: status.fileSizeBytes,
          epoch: status.epoch,
          createdAt: new Date(h.createdAt).toISOString(),
        });
      }),
    );

    // ── rvf_list_stores ───────────────────────────────────────────────────
    this.mcp.tool(
      'rvf_list_stores',
      'List all open RVF stores',
      {},
      async () => {
        const list = await this.listStores();
        return reply({
          stores: list.map((h) => ({
            storeId: h.storeId,
            path: h.path,
            dimensions: h.dimensions,
            metric: h.metric,
            totalVectors: h.totalVectors,
            readOnly: h.readOnly,
          })),
          count: list.length,
        });
      },
    );
  }

  private async listStores() {
    const out = [];
    for (const h of this.stores.values()) {
      let totalVectors = 0;
      try {
        totalVectors = (await h.db.status()).totalVectors;
      } catch {
        // report the store even if its status call fails
      }
      out.push({
        storeId: h.id,
        path: h.path,
        dimensions: h.dimensions,
        metric: h.metric,
        readOnly: h.readOnly,
        totalVectors,
      });
    }
    return out;
  }

  private async closeHandle(h: StoreHandle): Promise<void> {
    try {
      h.meta.rewrite();
    } finally {
      await h.db.close();
    }
  }

  // ─── Resource Registration ──────────────────────────────────────────────

  private registerResources(): void {
    // List of open stores
    this.mcp.resource(
      'stores-list',
      'rvf://stores',
      { description: 'List all open RVF stores and their status' },
      async () => {
        const list = (await this.listStores()).map((h) => ({
          storeId: h.storeId,
          path: h.path,
          dimensions: h.dimensions,
          totalVectors: h.totalVectors,
        }));

        return {
          contents: [{
            uri: 'rvf://stores',
            mimeType: 'application/json',
            text: JSON.stringify({ stores: list }, null, 2),
          }],
        };
      },
    );
  }

  // ─── Prompt Registration ────────────────────────────────────────────────

  private registerPrompts(): void {
    this.mcp.prompt(
      'rvf-search',
      'Search for similar vectors in an RVF store',
      {
        storeId: z.string().describe('Store ID to search'),
        description: z.string().describe('Natural language description of what to search for'),
      },
      async ({ storeId, description }) => ({
        messages: [{
          role: 'user' as const,
          content: {
            type: 'text' as const,
            text: `Search the RVF store "${storeId}" for vectors similar to: "${description}". ` +
              'Use the rvf_query tool to perform the search. If you need to create an embedding ' +
              'from the description first, generate a suitable vector representation.',
          },
        }],
      }),
    );

    this.mcp.prompt(
      'rvf-ingest',
      'Ingest data into an RVF store',
      {
        storeId: z.string().describe('Store ID to ingest into'),
        data: z.string().describe('Data to embed and ingest'),
      },
      async ({ storeId, data }) => ({
        messages: [{
          role: 'user' as const,
          content: {
            type: 'text' as const,
            text: `Ingest the following data into RVF store "${storeId}": ${data}. ` +
              'Generate appropriate vector embeddings and metadata, then use the rvf_ingest tool.',
          },
        }],
      }),
    );
  }

  // ─── Connection ─────────────────────────────────────────────────────────

  async connect(transport: Parameters<McpServer['connect']>[0]): Promise<void> {
    await this.mcp.connect(transport);
  }

  async close(): Promise<void> {
    // Close every store so native writer locks are released and metadata
    // logs are compacted before the transport goes away.
    const handles = [...this.stores.values()];
    this.stores.clear();
    let failure: unknown;
    for (const h of handles) {
      try {
        await this.closeHandle(h);
      } catch (err) {
        failure ??= err;
      }
    }
    await this.mcp.close();
    if (failure) throw failure;
  }

  get storeCount(): number {
    return this.stores.size;
  }
}
