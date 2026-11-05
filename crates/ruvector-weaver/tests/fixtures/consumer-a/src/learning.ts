// Excerpt shaped like claude-flow v3/plugins/ruvector-upstream/src/bridges/learning.ts
export async function loadLearning() {
  const wasmModule = await import('@ruvector/learning-wasm' as string).catch(() => null);
  return wasmModule;
}
