/** Node-only build tooling; requires caller-installed esbuild. Never deploys. */
export function buildWorker(outDirectory: string, entryPoint?: string): Promise<{
  script: string;
  wasm: string;
}>;
