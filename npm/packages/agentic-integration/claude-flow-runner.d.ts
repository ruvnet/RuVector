/**
 * Run a claude-flow hook without invoking a shell.
 *
 * Agent identifiers, regions, request IDs, and memory keys may originate from
 * remote coordination messages. Passing them as distinct argv entries keeps
 * shell metacharacters inert.
 */
export declare function runClaudeFlow(args: Array<string | number | boolean>): Promise<void>;
//# sourceMappingURL=claude-flow-runner.d.ts.map