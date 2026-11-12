/** Query the state of a hosted build (`GET /v1/builds/{id}`). */

import { ForgeApiClient, type BuildRecord, type ForgeApiOptions } from './api';
import { ForgeError, ForgeErrorCode } from './errors';

/** Build ids are opaque to the CLI, but must be safe to place in a URL path. */
const BUILD_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

export interface StatusOptions {
  api?: ForgeApiOptions;
}

export interface StatusResult {
  endpoint: string;
  build: BuildRecord;
  /** True once the build can no longer change state. */
  terminal: boolean;
}

export function assertBuildId(id: string): string {
  if (!BUILD_ID.test(id)) {
    throw new ForgeError(
      ForgeErrorCode.USAGE,
      `"${id}" is not a valid build id (letters, digits, dot, dash, underscore; 1-128 chars).`,
      { buildId: id },
    );
  }
  return id;
}

/**
 * @throws {ForgeError} `FORGE_E_USAGE` for a malformed id, `FORGE_E_NOT_FOUND`
 * when the build is unknown, `FORGE_E_NETWORK` when the service is unreachable.
 */
export async function runStatus(id: string, options: StatusOptions = {}): Promise<StatusResult> {
  const buildId = assertBuildId(id);
  const client = new ForgeApiClient(options.api);
  const build = await client.getBuild(buildId);
  return {
    endpoint: client.endpoint,
    build,
    terminal: ['succeeded', 'failed', 'cancelled'].includes(build.state),
  };
}
