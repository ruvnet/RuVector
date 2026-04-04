/**
 * Context window optimization based on Claude Code's compaction internals.
 *
 * The decompilation of v2.1.91 revealed:
 * - Compaction triggers at ~80% of the context window
 * - The `clear_tool_uses_20250919` API feature enables server-side compaction
 * - Deferred tool loading via ToolSearch saves ~2000 tokens per unused tool
 * - Prompt cache sharing (`promptCacheSharingEnabled`) caches static prefixes
 *
 * This module helps RVAgent structure prompts and manage context to
 * maximise cache hits and minimise unnecessary token usage.
 */

'use strict';

const { TOOL_NAMES } = require('./tool-schemas');

/** Default context window sizes by model family. */
const MODEL_CONTEXT_SIZES = {
  'claude-sonnet-4-20250514': 200000,
  'claude-opus-4-20250514': 200000,
  'claude-haiku-3.5': 200000,
  default: 200000,
};

/** Approximate token overhead per tool schema sent in the system prompt. */
const TOKENS_PER_TOOL_SCHEMA = 120;

/**
 * Build a deferred tool list — instead of sending all 25+ tool schemas
 * up front (costing ~3000 tokens), only include the tools the task
 * actually needs. The rest can be fetched on demand via ToolSearch.
 *
 * @param {string[]} requiredTools - tool names this task needs
 * @returns {{ included: string[], deferred: string[], tokensSaved: number }}
 */
function buildDeferredToolList(requiredTools) {
  const required = new Set(requiredTools || []);
  const included = [];
  const deferred = [];

  for (const tool of TOOL_NAMES) {
    if (required.has(tool)) {
      included.push(tool);
    } else {
      deferred.push(tool);
    }
  }

  const tokensSaved = deferred.length * TOKENS_PER_TOOL_SCHEMA;
  return { included, deferred, tokensSaved };
}

/**
 * Build a cache-optimised prompt structure.
 *
 * Claude Code's `promptCacheSharingEnabled` caches the longest common
 * prefix across requests. To maximise cache hits:
 * 1. Put static content first (tool schemas, CLAUDE.md rules)
 * 2. Put dynamic/session-specific content after
 *
 * @param {string} staticRules  - stable content (tool schemas, rules)
 * @param {string} dynamicContext - session-specific content (files, history)
 * @returns {{ cached: string, dynamic: string, totalTokensEstimate: number }}
 */
function buildCacheOptimizedPrompt(staticRules, dynamicContext) {
  const staticTokens = estimateTokens(staticRules);
  const dynamicTokens = estimateTokens(dynamicContext);

  return {
    cached: staticRules,
    dynamic: dynamicContext,
    totalTokensEstimate: staticTokens + dynamicTokens,
    cacheableTokens: staticTokens,
  };
}

/**
 * Determine whether compaction should be triggered.
 *
 * From the decompiled source, Claude Code triggers compaction when the
 * token count reaches approximately 80% of the context window. This
 * function mirrors that threshold.
 *
 * @param {number} tokenCount   - current token usage
 * @param {number} [maxContext] - max context window (default 200000)
 * @returns {boolean}
 */
function shouldCompact(tokenCount, maxContext) {
  const max = maxContext || MODEL_CONTEXT_SIZES.default;
  return tokenCount > max * 0.8;
}

/**
 * Estimate the number of tokens in a string.
 * Uses the common heuristic of ~4 characters per token for English text.
 *
 * @param {string} text
 * @returns {number}
 */
function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

/**
 * Recommend which tools to include for a given task type.
 * This avoids sending all tool schemas when only a subset is needed.
 *
 * @param {string} taskType
 * @returns {string[]}
 */
function recommendToolsForTask(taskType) {
  const base = ['Read', 'Bash', 'Glob', 'Grep'];

  const taskTools = {
    coding: [...base, 'Edit', 'Write', 'Agent', 'ToolSearch'],
    research: [...base, 'WebFetch', 'WebSearch', 'Agent'],
    quickfix: [...base, 'Edit'],
    planning: [...base, 'Agent', 'TodoWrite'],
    background: [...base, 'Edit', 'Write', 'Agent'],
    swarm: [...base, 'Edit', 'Write', 'Agent', 'ToolSearch'],
    review: [...base],
    ci: [...base, 'Edit', 'Write'],
  };

  return taskTools[taskType] || base;
}

/**
 * Get context window size for a model identifier.
 *
 * @param {string} model
 * @returns {number}
 */
function getContextSize(model) {
  return MODEL_CONTEXT_SIZES[model] || MODEL_CONTEXT_SIZES.default;
}

module.exports = {
  MODEL_CONTEXT_SIZES,
  TOKENS_PER_TOOL_SCHEMA,
  buildDeferredToolList,
  buildCacheOptimizedPrompt,
  shouldCompact,
  estimateTokens,
  recommendToolsForTask,
  getContextSize,
};
