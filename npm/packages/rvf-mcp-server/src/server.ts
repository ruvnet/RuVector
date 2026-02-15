/**
 * RVF MCP Server — core server implementation.
 *
 * Wires all 17 MCP tools to the real RvfDatabase async API
 * for disk-backed, HNSW-indexed persistence.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { RvfDatabase, RvfError, RvfErrorCode } from '@ruvector/rvf';
import type { RvfFilterExpr, RvfFilterValue, DistanceMetric } from '@ruvector/rvf';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface RvfMcpServerOptions {
  /** Server name shown to MCP clients. Default: 'rvf-mcp-server'. */
  name?: string;
  /** Server version. Default: '0.1.0'. */
  version?: string;
  /** Maximum open stores. Default: 64. */
  maxStores?: number;
}

interface StoreHandle {
  id: string;
  path: string;
  db: RvfDatabase;
  readOnly: boolean;
  openedAt: number;
}

type ToolResult = { content: Array<{ type: 'text'; text: string }> };

// Reusable Zod schemas pulled out to avoid TS2589 deep-instantiation errors
// when inline inside mcp.tool() generic overload resolution.
const MetadataFilter = z.record(z.string(), z.any()).optional()
  .describe('Metadata filter (exact match on fields)');
const MetadataFilterRequired = z.record(z.string(), z.any())
  .describe('Metadata filter — all matching vectors will be deleted');
const IngestEntrySchema = z.object({
  id: z.string().describe('Unique vector ID'),
  vector: z.array(z.number()).describe('Embedding vector (must match store dimensions)'),
  metadata: z.record(z.string(), z.any()).optional()
    .describe('Optional metadata key-value pairs'),
});

// ─── Helpers ────────────────────────────────────────────────────────────────

/**
 * FNV-1a hash of a field name to produce a stable u32 fieldId.
 * This is a deterministic mapping that works without maintaining state.
 */
function fnv1aFieldId(name: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0; // ensure unsigned 32-bit
}

/**
 * Convert a simple `Record<string, value>` filter (as accepted by MCP tools)
 * into an `RvfFilterExpr` tree using `eq` operators joined by `and`.
 */
function buildFilterExpr(filter: Record<string, unknown>): RvfFilterExpr {
  const entries = Object.entries(filter);
  if (entries.length === 1) {
    const [key, val] = entries[0];
    return { op: 'eq', fieldId: fnv1aFieldId(key), value: val as string | number | boolean };
  }
  return {
    op: 'and',
    exprs: entries.map(([key, val]) => ({
      op: 'eq' as const,
      fieldId: fnv1aFieldId(key),
      value: val as string | number | boolean,
    })),
  };
}

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
      maxStores: options?.maxStores ?? 64,
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

  // ─── Internal helpers ───────────────────────────────────────────────────

  private errorResponse(msg: string): ToolResult {
    return { content: [{ type: 'text' as const, text: `Error: ${msg}` }] };
  }

  private async withStore<T>(
    storeId: string,
    fn: (handle: StoreHandle) => Promise<T>,
  ): Promise<T | ToolResult> {
    const handle = this.stores.get(storeId);
    if (!handle) {
      return this.errorResponse(`store ${storeId} not found`);
    }
    try {
      return await fn(handle);
    } catch (err) {
      if (err instanceof RvfError) {
        return this.errorResponse(`[${RvfErrorCode[err.code]}] ${err.message}`);
      }
      return this.errorResponse(String(err));
    }
  }

  // ─── Tool Registration ──────────────────────────────────────────────────

  private registerTools(): void {
    // ── rvf_create_store ──────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_create_store',
      'Create a new RVF vector store at the given path',
      {
        path: z.string().describe('File path for the new .rvf store'),
        dimensions: z.number().int().positive().describe('Vector dimensionality'),
        metric: z.enum(['l2', 'cosine', 'dotproduct']).default('l2').describe('Distance metric'),
      },
      async ({ path, dimensions, metric }) => {
        if (this.stores.size >= this.opts.maxStores) {
          return this.errorResponse(`max stores (${this.opts.maxStores}) reached`);
        }

        try {
          const db = await RvfDatabase.create(path, {
            dimensions,
            metric: metric as DistanceMetric,
          });

          const id = `store_${this.nextId++}`;
          this.stores.set(id, {
            id,
            path,
            db,
            readOnly: false,
            openedAt: Date.now(),
          });

          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({
                storeId: id,
                path,
                dimensions,
                metric,
                status: 'created',
              }, null, 2),
            }],
          };
        } catch (err) {
          if (err instanceof RvfError) {
            return this.errorResponse(`[${RvfErrorCode[err.code]}] ${err.message}`);
          }
          return this.errorResponse(String(err));
        }
      },
    );

    // ── rvf_open_store ────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_open_store',
      'Open an existing RVF store for reading and writing',
      {
        path: z.string().describe('Path to existing .rvf file'),
        readOnly: z.boolean().default(false).describe('Open in read-only mode'),
      },
      async ({ path, readOnly }) => {
        if (this.stores.size >= this.opts.maxStores) {
          return this.errorResponse(`max stores (${this.opts.maxStores}) reached`);
        }

        try {
          const db = readOnly
            ? await RvfDatabase.openReadonly(path)
            : await RvfDatabase.open(path);

          const id = `store_${this.nextId++}`;
          this.stores.set(id, {
            id,
            path,
            db,
            readOnly,
            openedAt: Date.now(),
          });

          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({
                storeId: id,
                path,
                readOnly,
                status: 'opened',
              }, null, 2),
            }],
          };
        } catch (err) {
          if (err instanceof RvfError) {
            return this.errorResponse(`[${RvfErrorCode[err.code]}] ${err.message}`);
          }
          return this.errorResponse(String(err));
        }
      },
    );

    // ── rvf_close_store ───────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_close_store',
      'Close an open RVF store, releasing the writer lock',
      {
        storeId: z.string().describe('Store ID returned by create/open'),
      },
      async ({ storeId }) => {
        const result = await this.withStore(storeId, async (handle) => {
          await handle.db.close();
          this.stores.delete(storeId);
          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({ storeId, status: 'closed', path: handle.path }, null, 2),
            }],
          };
        });
        return result as ToolResult;
      },
    );

    // ── rvf_ingest ────────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_ingest',
      'Insert vectors into an RVF store',
      {
        storeId: z.string().describe('Target store ID'),
        entries: z.array(IngestEntrySchema).describe('Vectors to insert'),
      },
      // @ts-ignore — TS2589: deep inference from nested Zod schema
      async ({ storeId, entries }: { storeId: string; entries: Array<{ id: string; vector: number[]; metadata?: Record<string, unknown> }> }) => {
        const result = await this.withStore(storeId, async (handle) => {
          if (handle.readOnly) {
            return this.errorResponse('store is read-only');
          }

          const ingestResult = await handle.db.ingestBatch(
            entries.map((e) => ({
              id: e.id,
              vector: e.vector,
              metadata: e.metadata as Record<string, RvfFilterValue> | undefined,
            })),
          );

          const status = await handle.db.status();

          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({
                accepted: ingestResult.accepted,
                rejected: ingestResult.rejected,
                epoch: ingestResult.epoch,
                totalVectors: status.totalVectors,
              }, null, 2),
            }],
          };
        });
        return result as ToolResult;
      },
    );

    // ── rvf_query ─────────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_query',
      'k-NN vector similarity search',
      {
        storeId: z.string().describe('Store ID to query'),
        vector: z.array(z.number()).describe('Query embedding vector'),
        k: z.number().int().positive().default(10).describe('Number of nearest neighbors'),
        filter: MetadataFilter,
      },
      // @ts-ignore — TS2589: deep inference from optional filter schema
      async ({ storeId, vector, k, filter }: { storeId: string; vector: number[]; k: number; filter?: Record<string, unknown> }) => {
        const result = await this.withStore(storeId, async (handle) => {
          const queryOpts = filter
            ? { filter: buildFilterExpr(filter) }
            : undefined;

          const results = await handle.db.query(vector, k, queryOpts);

          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({
                results: results.map((r) => ({ id: r.id, distance: r.distance })),
                count: results.length,
              }, null, 2),
            }],
          };
        });
        return result as ToolResult;
      },
    );

    // ── rvf_delete ────────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_delete',
      'Delete vectors by their IDs',
      {
        storeId: z.string().describe('Store ID'),
        ids: z.array(z.string()).describe('Vector IDs to delete'),
      },
      async ({ storeId, ids }) => {
        const result = await this.withStore(storeId, async (handle) => {
          if (handle.readOnly) {
            return this.errorResponse('store is read-only');
          }

          const delResult = await handle.db.delete(ids);
          const status = await handle.db.status();

          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({
                deleted: delResult.deleted,
                epoch: delResult.epoch,
                remaining: status.totalVectors,
              }, null, 2),
            }],
          };
        });
        return result as ToolResult;
      },
    );

    // ── rvf_delete_filter ─────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_delete_filter',
      'Delete vectors matching a metadata filter',
      {
        storeId: z.string().describe('Store ID'),
        filter: MetadataFilterRequired,
      },
      // @ts-ignore — TS2589: deep inference from filter schema
      async ({ storeId, filter }: { storeId: string; filter: Record<string, unknown> }) => {
        const result = await this.withStore(storeId, async (handle) => {
          if (handle.readOnly) {
            return this.errorResponse('store is read-only');
          }

          const filterExpr = buildFilterExpr(filter);
          const delResult = await handle.db.deleteByFilter(filterExpr);
          const status = await handle.db.status();

          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({
                deleted: delResult.deleted,
                epoch: delResult.epoch,
                remaining: status.totalVectors,
              }, null, 2),
            }],
          };
        });
        return result as ToolResult;
      },
    );

    // ── rvf_compact ───────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_compact',
      'Compact store to reclaim dead space from deleted vectors',
      {
        storeId: z.string().describe('Store ID'),
      },
      async ({ storeId }) => {
        const result = await this.withStore(storeId, async (handle) => {
          const compactResult = await handle.db.compact();

          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({
                storeId,
                segmentsCompacted: compactResult.segmentsCompacted,
                bytesReclaimed: compactResult.bytesReclaimed,
                epoch: compactResult.epoch,
              }, null, 2),
            }],
          };
        });
        return result as ToolResult;
      },
    );

    // ── rvf_status ────────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_status',
      'Get the current status of an RVF store',
      {
        storeId: z.string().describe('Store ID'),
      },
      async ({ storeId }) => {
        const result = await this.withStore(storeId, async (handle) => {
          const [status, dimension] = await Promise.all([
            handle.db.status(),
            handle.db.dimension(),
          ]);

          return {
            content: [{
              type: 'text' as const,
              text: JSON.stringify({
                storeId: handle.id,
                path: handle.path,
                dimensions: dimension,
                totalVectors: status.totalVectors,
                totalSegments: status.totalSegments,
                fileSizeBytes: status.fileSizeBytes,
                epoch: status.epoch,
                compactionState: status.compactionState,
                deadSpaceRatio: status.deadSpaceRatio,
                readOnly: status.readOnly,
                openedAt: new Date(handle.openedAt).toISOString(),
              }, null, 2),
            }],
          };
        });
        return result as ToolResult;
      },
    );

    // ── rvf_list_stores ───────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_list_stores',
      'List all open RVF stores',
      {},
      async () => {
        const handles = Array.from(this.stores.values());
        const list = await Promise.all(
          handles.map(async (h) => {
            try {
              const [status, dimension] = await Promise.all([
                h.db.status(),
                h.db.dimension(),
              ]);
              return {
                storeId: h.id,
                path: h.path,
                dimensions: dimension,
                totalVectors: status.totalVectors,
                readOnly: status.readOnly,
              };
            } catch {
              return {
                storeId: h.id,
                path: h.path,
                dimensions: null,
                totalVectors: null,
                readOnly: h.readOnly,
                error: 'failed to read status',
              };
            }
          }),
        );

        return {
          content: [{
            type: 'text' as const,
            text: JSON.stringify({ stores: list, count: list.length }, null, 2),
          }],
        };
      },
    );

    // ── rvf_put_meta ─────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_put_meta',
      'Store a key-value pair in store-level metadata (persisted in META_SEG)',
      {
        storeId: z.string().describe('Store ID from rvf_create_store or rvf_open_store'),
        key: z.string().describe('Metadata key'),
        value: z.string().describe('Metadata value (stored as UTF-8 bytes)'),
      },
      async ({ storeId, key, value }): Promise<ToolResult> => {
        const handle = this.stores.get(storeId);
        if (!handle) return { content: [{ type: 'text', text: `Store '${storeId}' not found` }] };
        try {
          await handle.db.putMeta(key, new TextEncoder().encode(value));
          return { content: [{ type: 'text', text: JSON.stringify({ success: true, key }) }] };
        } catch (e: any) {
          return { content: [{ type: 'text', text: `Error: ${e.message}` }] };
        }
      },
    );

    // ── rvf_get_meta ─────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_get_meta',
      'Retrieve a value from store-level metadata by key',
      {
        storeId: z.string().describe('Store ID'),
        key: z.string().describe('Metadata key to retrieve'),
      },
      async ({ storeId, key }): Promise<ToolResult> => {
        const handle = this.stores.get(storeId);
        if (!handle) return { content: [{ type: 'text', text: `Store '${storeId}' not found` }] };
        try {
          const result = await handle.db.getMeta(key);
          if (result === null) {
            return { content: [{ type: 'text', text: JSON.stringify({ found: false, key }) }] };
          }
          return { content: [{ type: 'text', text: JSON.stringify({ found: true, key, value: new TextDecoder().decode(result) }) }] };
        } catch (e: any) {
          return { content: [{ type: 'text', text: `Error: ${e.message}` }] };
        }
      },
    );

    // ── rvf_list_meta_keys ───────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_list_meta_keys',
      'List all keys in store-level metadata',
      {
        storeId: z.string().describe('Store ID'),
      },
      async ({ storeId }): Promise<ToolResult> => {
        const handle = this.stores.get(storeId);
        if (!handle) return { content: [{ type: 'text', text: `Store '${storeId}' not found` }] };
        try {
          const keys = await handle.db.listMetaKeys();
          return { content: [{ type: 'text', text: JSON.stringify({ keys, count: keys.length }) }] };
        } catch (e: any) {
          return { content: [{ type: 'text', text: `Error: ${e.message}` }] };
        }
      },
    );

    // ── rvf_branch ───────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_branch',
      'Create a COW (copy-on-write) branch of a store',
      {
        storeId: z.string().describe('Parent store ID'),
        childPath: z.string().describe('File path for the child branch'),
      },
      async ({ storeId, childPath }): Promise<ToolResult> => {
        const handle = this.stores.get(storeId);
        if (!handle) return { content: [{ type: 'text', text: `Store '${storeId}' not found` }] };
        try {
          const childDb = await handle.db.branch(childPath);
          const childId = `store_${this.nextId++}`;
          this.stores.set(childId, { id: childId, path: childPath, db: childDb, readOnly: false, openedAt: Date.now() });
          return { content: [{ type: 'text', text: JSON.stringify({ success: true, childStoreId: childId, childPath }) }] };
        } catch (e: any) {
          return { content: [{ type: 'text', text: `Error: ${e.message}` }] };
        }
      },
    );

    // ── rvf_freeze ───────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_freeze',
      'Freeze a COW branch (make it immutable)',
      {
        storeId: z.string().describe('Store ID to freeze'),
      },
      async ({ storeId }): Promise<ToolResult> => {
        const handle = this.stores.get(storeId);
        if (!handle) return { content: [{ type: 'text', text: `Store '${storeId}' not found` }] };
        try {
          await handle.db.freeze();
          return { content: [{ type: 'text', text: JSON.stringify({ success: true, frozen: true }) }] };
        } catch (e: any) {
          return { content: [{ type: 'text', text: `Error: ${e.message}` }] };
        }
      },
    );

    // ── rvf_cow_stats ────────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_cow_stats',
      'Get COW (copy-on-write) statistics for a store',
      {
        storeId: z.string().describe('Store ID'),
      },
      async ({ storeId }): Promise<ToolResult> => {
        const handle = this.stores.get(storeId);
        if (!handle) return { content: [{ type: 'text', text: `Store '${storeId}' not found` }] };
        try {
          const stats = await handle.db.cowStats();
          return { content: [{ type: 'text', text: JSON.stringify(stats) }] };
        } catch (e: any) {
          return { content: [{ type: 'text', text: `Error: ${e.message}` }] };
        }
      },
    );

    // ── rvf_witness_hash ─────────────────────────────────────────────────
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
    this.mcp.tool(
      'rvf_witness_hash',
      'Get the last witness hash from the audit chain',
      {
        storeId: z.string().describe('Store ID'),
      },
      async ({ storeId }): Promise<ToolResult> => {
        const handle = this.stores.get(storeId);
        if (!handle) return { content: [{ type: 'text', text: `Store '${storeId}' not found` }] };
        try {
          const hash = await handle.db.lastWitnessHash();
          const hex = Array.from(hash).map(b => b.toString(16).padStart(2, '0')).join('');
          return { content: [{ type: 'text', text: JSON.stringify({ witnessHash: hex }) }] };
        } catch (e: any) {
          return { content: [{ type: 'text', text: `Error: ${e.message}` }] };
        }
      },
    );
  }

  // ─── Resource Registration ──────────────────────────────────────────────

  private registerResources(): void {
    this.mcp.resource(
      'stores-list',
      'rvf://stores',
      { description: 'List all open RVF stores and their status' },
      async () => {
        const handles = Array.from(this.stores.values());
        const list = await Promise.all(
          handles.map(async (h) => {
            try {
              const [status, dimension] = await Promise.all([
                h.db.status(),
                h.db.dimension(),
              ]);
              return {
                storeId: h.id,
                path: h.path,
                dimensions: dimension,
                totalVectors: status.totalVectors,
              };
            } catch {
              return {
                storeId: h.id,
                path: h.path,
                dimensions: null,
                totalVectors: null,
              };
            }
          }),
        );

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
    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
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

    // @ts-ignore — TS2589: MCP SDK overload + Zod v4 deep inference
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
    // Close all native database handles before shutting down MCP
    const handles = Array.from(this.stores.values());
    await Promise.all(
      handles.map(async (h) => {
        try {
          await h.db.close();
        } catch {
          // best-effort on shutdown
        }
      }),
    );
    this.stores.clear();
    await this.mcp.close();
  }

  get storeCount(): number {
    return this.stores.size;
  }
}
