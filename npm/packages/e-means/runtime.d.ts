export { fit, predict } from '@ruvector/e-means';
export type { EMeansOptions, EMeansModel } from '@ruvector/e-means';
/** No filesystem/default URL loader. Supply bytes or a precompiled module. */
export function init(source: BufferSource | WebAssembly.Module): Promise<void>;
