import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
const NPX_COMMAND = process.platform === 'win32' ? 'npx.cmd' : 'npx';

/**
 * Send a claude-flow `hooks notify` message without invoking a shell.
 *
 * Messages embed region names, event names and scaling reasons that may come
 * from callers, metrics sources or event calendars. Passing the message as a
 * single argv entry keeps shell metacharacters (`$()`, backticks, `;`) inert.
 */
export async function notifyClaudeFlow(message: string): Promise<void> {
  await execFileAsync(
    NPX_COMMAND,
    ['claude-flow@alpha', 'hooks', 'notify', '--message', String(message)],
    {
      timeout: 30_000,
      windowsHide: true,
    },
  );
}
