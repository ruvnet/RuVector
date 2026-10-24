import { pathToFileURL } from 'node:url';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

export type IntegrationErrorCode =
  | 'ERR_INTEGRATION_MISSING'
  | 'ERR_INTEGRATION_LOAD'
  | 'ERR_INTEGRATION_API'
  | 'ERR_INTEGRATION_UNTRAINED'
  | 'ERR_INTEGRATION_INVALID';

export class IntegrationError extends Error {
  constructor(
    public readonly code: IntegrationErrorCode,
    public readonly packageName: string,
    message: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = 'IntegrationError';
  }
}

// Keep genuine import() when tsc emits CommonJS. Resolve relative to this SDK,
// not process.cwd(), so this also works for installed packages and ESM peers.
const nativeImport = new Function('url', 'return import(url)') as
  (url: string) => Promise<Record<string, any>>;
const pending = new Map<string, Promise<Record<string, any>>>();

/** Load only on explicit feature use. Never install dependencies or swallow a
 * broken installed package as if the feature simply were not installed. */
export async function loadOptionalPackage(
  packageName: string,
  entry = packageName,
  developmentOnly = false,
): Promise<Record<string, any>> {
  let resolved: string;
  try {
    resolved = require.resolve(entry);
  } catch (cause: any) {
    if (cause?.code === 'MODULE_NOT_FOUND') {
      const installed = (require.resolve.paths(packageName) ?? []).some(directory =>
        existsSync(join(directory, packageName, 'package.json')));
      if (installed) {
        throw new IntegrationError('ERR_INTEGRATION_API', packageName,
          `Installed ${packageName} is missing the entry point ${entry}. Reinstall a compatible package.`, cause);
      }
      throw new IntegrationError('ERR_INTEGRATION_MISSING', packageName,
        `Install this feature explicitly: npm install ${developmentOnly ? '--save-dev ' : ''}${packageName}. ` +
        'Core vector operations do not require it.', cause);
    }
    throw new IntegrationError('ERR_INTEGRATION_API', packageName,
      `Installed ${packageName} does not expose the supported entry point ${entry}.`, cause);
  }
  let result = pending.get(resolved);
  if (!result) {
    result = nativeImport(pathToFileURL(resolved).href).then(module => {
      // CommonJS packages may expose the full API only as their default export.
      return module.default && typeof module.default === 'object'
        ? { ...module.default, ...module }
        : module;
    }).catch(cause => {
      pending.delete(resolved);
      throw new IntegrationError('ERR_INTEGRATION_LOAD', packageName,
        `Installed ${packageName} failed to load: ${cause instanceof Error ? cause.message : String(cause)}`, cause);
    });
    pending.set(resolved, result);
  }
  return result;
}
