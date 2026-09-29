import { validateVector, type VectorRecord } from './vector.ts';
import type { AuthEnv } from './auth.ts';

export type StoredVector = VectorRecord & { dimension: number };
const NAME = /^[A-Za-z0-9_-]{1,64}$/;

export function validateCollection(collection: string): void {
  if (!NAME.test(collection)) throw new RangeError('invalid_collection');
}

export async function loadVectors(env: AuthEnv, tenantId: string, collection: string): Promise<StoredVector[]> {
  validateCollection(collection);
  const { results } = await env.DB.prepare(
    'SELECT id, dimension, vector_json, metadata_json FROM vectors WHERE tenant_id = ? AND collection = ? ORDER BY id LIMIT 501',
  ).bind(tenantId, collection).all<{
    id: string; dimension: number; vector_json: string; metadata_json: string;
  }>();
  if (results.length > 500) throw new RangeError('collection_search_limit_exceeded');
  return results.map(row => {
    const vector = JSON.parse(row.vector_json) as number[];
    validateVector(vector, row.dimension);
    return { id: row.id, dimension: row.dimension, vector, metadata: JSON.parse(row.metadata_json) };
  });
}

export async function storeVectors(
  env: AuthEnv, tenantId: string, collection: string, records: VectorRecord[],
): Promise<number> {
  validateCollection(collection);
  if (records.length < 1 || records.length > 24 || new Set(records.map(r => r.id)).size !== records.length) {
    throw new RangeError('invalid_batch');
  }
  const dimension = records[0].vector.length;
  for (const record of records) {
    if (!NAME.test(record.id)) throw new RangeError('invalid_record_id');
    validateVector(record.vector, dimension);
    if (JSON.stringify(record.metadata ?? {}).length > 2048) throw new RangeError('metadata_too_large');
  }
  const existing = await env.DB.prepare(
    'SELECT dimension FROM vectors WHERE tenant_id = ? AND collection = ? LIMIT 1',
  ).bind(tenantId, collection).first<{ dimension: number }>();
  if (existing && existing.dimension !== dimension) throw new RangeError('dimension_mismatch');
  const statements = records.map(record => env.DB.prepare(
    `INSERT INTO vectors (tenant_id, collection, id, dimension, vector_json, metadata_json)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(tenant_id, collection, id) DO UPDATE SET
       vector_json = excluded.vector_json, metadata_json = excluded.metadata_json,
       updated_at = CURRENT_TIMESTAMP
     WHERE vectors.dimension = excluded.dimension`,
  ).bind(tenantId, collection, record.id, dimension, JSON.stringify(record.vector), JSON.stringify(record.metadata ?? {})));
  await env.DB.batch(statements);
  return records.length;
}
