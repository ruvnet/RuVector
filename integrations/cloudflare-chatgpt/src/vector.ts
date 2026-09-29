export type VectorRecord = { id: string; vector: number[]; metadata?: Record<string, unknown> };
export type SearchHit = { id: string; distance: number; metadata?: Record<string, unknown> };

export function validateVector(vector: number[], dimension?: number): void {
  if (!Array.isArray(vector) || vector.length < 1 || vector.length > 1536 ||
    (dimension !== undefined && vector.length !== dimension) ||
    vector.some(value => typeof value !== 'number' || !Number.isFinite(value))) {
    throw new RangeError('invalid_vector');
  }
}

// RVF's in-memory store performs exact kNN. Instantiate per call: its handle registry
// never persists across tenants or requests. The binary is built in crates/rvf/rvf-wasm.
export async function searchExact(
  module: WebAssembly.Module,
  records: VectorRecord[],
  query: number[],
  k: number,
  metric: 'cosine' | 'l2' = 'cosine',
): Promise<SearchHit[]> {
  validateVector(query);
  if (!Number.isSafeInteger(k) || k < 1 || k > 20 || records.length > 500) throw new RangeError('invalid_search_limit');
  for (const record of records) validateVector(record.vector, query.length);
  if (records.length === 0) return [];

  const instance = await WebAssembly.instantiate(module, {});
  const wasm = instance.exports as unknown as {
    memory: WebAssembly.Memory;
    rvf_alloc(size: number): number;
    rvf_free(ptr: number, size: number): void;
    rvf_store_create(dim: number, metric: number): number;
    rvf_store_ingest(handle: number, vectors: number, ids: number, count: number): number;
    rvf_store_query(handle: number, query: number, k: number, metric: number, out: number): number;
    rvf_store_close(handle: number): number;
  };
  const metricCode = metric === 'cosine' ? 2 : 0;
  const handle = wasm.rvf_store_create(query.length, metricCode);
  if (handle <= 0) throw new Error('rvf_store_create_failed');
  const pointers: Array<[number, number]> = [];
  const alloc = (size: number) => {
    const ptr = wasm.rvf_alloc(size);
    if (ptr <= 0) throw new Error('rvf_alloc_failed');
    pointers.push([ptr, size]);
    return ptr;
  };
  try {
    const vectorPtr = alloc(records.length * query.length * 4);
    const idPtr = alloc(records.length * 8);
    let view = new DataView(wasm.memory.buffer);
    records.forEach((record, index) => {
      view.setBigUint64(idPtr + index * 8, BigInt(index), true);
      record.vector.forEach((value, dimension) => view.setFloat32(vectorPtr + (index * query.length + dimension) * 4, value, true));
    });
    if (wasm.rvf_store_ingest(handle, vectorPtr, idPtr, records.length) !== records.length) {
      throw new Error('rvf_ingest_failed');
    }
    const queryPtr = alloc(query.length * 4);
    const outPtr = alloc(Math.min(k, records.length) * 12);
    view = new DataView(wasm.memory.buffer);
    query.forEach((value, index) => view.setFloat32(queryPtr + index * 4, value, true));
    const count = wasm.rvf_store_query(handle, queryPtr, k, metricCode, outPtr);
    if (count < 0 || count > Math.min(k, records.length)) throw new Error('rvf_query_failed');
    view = new DataView(wasm.memory.buffer);
    return Array.from({ length: count }, (_, index) => {
      const position = Number(view.getBigUint64(outPtr + index * 12, true));
      if (!records[position]) throw new Error('rvf_result_out_of_bounds');
      return {
        id: records[position].id,
        distance: view.getFloat32(outPtr + index * 12 + 8, true),
        metadata: records[position].metadata,
      };
    });
  } finally {
    for (const [ptr, size] of pointers.reverse()) wasm.rvf_free(ptr, size);
    wasm.rvf_store_close(handle);
  }
}
