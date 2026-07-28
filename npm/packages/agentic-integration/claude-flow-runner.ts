import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
const NPX_COMMAND = process.platform === 'win32' ? 'npx.cmd' : 'npx';

/**
 * Run a claude-flow hook without invoking a shell.
 *
 * Agent identifiers, regions, request IDs, and memory keys may originate from
 * remote coordination messages. Passing them as distinct argv entries keeps
 * shell metacharacters inert.
 */
export async function runClaudeFlow(args: Array<string | number | boolean>): Promise<void> {
  await execFileAsync(
    NPX_COMMAND,
    ['claude-flow@alpha', ...args.map(String)],
    {
      timeout: 30_000,
      windowsHide: true,
    },
  );
}
