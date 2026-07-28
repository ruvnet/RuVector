"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.runClaudeFlow = runClaudeFlow;
const child_process_1 = require("child_process");
const util_1 = require("util");
const execFileAsync = (0, util_1.promisify)(child_process_1.execFile);
const NPX_COMMAND = process.platform === 'win32' ? 'npx.cmd' : 'npx';
/**
 * Run a claude-flow hook without invoking a shell.
 *
 * Agent identifiers, regions, request IDs, and memory keys may originate from
 * remote coordination messages. Passing them as distinct argv entries keeps
 * shell metacharacters inert.
 */
async function runClaudeFlow(args) {
    await execFileAsync(NPX_COMMAND, ['claude-flow@alpha', ...args.map(String)], {
        timeout: 30000,
        windowsHide: true,
    });
}
//# sourceMappingURL=claude-flow-runner.js.map