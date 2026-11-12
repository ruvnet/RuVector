// Independent seeded input generator and exhaustive cosine oracle.
export function vectors(count, dimensions, seed = 0x51a7) {
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  return Array.from({ length: count }, () =>
    Float32Array.from({ length: dimensions }, () => random() * 2 - 1));
}
export function exactIds(data, query, k) {
  let qnorm = 0;
  for (const x of query) qnorm += x * x;
  return data.map((vector, id) => {
    let dot = 0, norm = 0;
    for (let j = 0; j < vector.length; j++) {
      dot += vector[j] * query[j];
      norm += vector[j] * vector[j];
    }
    return { id: String(id), distance: 1 - dot / Math.sqrt(norm * qnorm) };
  }).sort((a, b) => a.distance - b.distance).slice(0, k).map(x => x.id);
}
export function recall(actual, expected) {
  const truth = new Set(expected);
  return actual.filter(id => truth.has(id)).length / truth.size;
}
export function readResults(db, query, k, filter) {
  const raw = db.search(query, k, filter);
  return raw.map(result => {
    const row = { id: result.id, score: result.score, metadata: result.metadata };
    result.free?.();
    return row;
  });
}
