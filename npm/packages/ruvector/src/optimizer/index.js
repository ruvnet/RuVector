/**
 * RVAgent Optimizer — Claude Code configuration profiles (ADR-139).
 *
 * Maps a task type (coding, research, quickfix, …) to an optimal set of
 * `CLAUDE_CODE_*` environment variables and a permission mode. Profiles are
 * derived from the decompiled Claude Code intelligence (see ADR-139) and are
 * consumed by `npx ruvector optimize` and the settings generator.
 *
 * Public API:
 *   - PERMISSION_MODES            — valid permission-mode strings
 *   - listProfiles()              — array of profile names
 *   - getProfile(name)            — { description, permissionMode, env } | null
 *   - applyProfile(name)          — sets process.env, returns { applied, permissionMode } | null
 *   - detectTaskType(prompt)      — infer a profile name from a free-text prompt
 */

'use strict';

/** Permission modes understood by Claude Code's harness. */
const PERMISSION_MODES = [
  'default',
  'acceptEdits',
  'bypassPermissions',
  'plan',
  'dontAsk',
  'auto',
];

/**
 * Task profiles. Every env key MUST start with `CLAUDE_CODE_` and every
 * `permissionMode` MUST be a member of {@link PERMISSION_MODES}.
 */
const PROFILES = {
  coding: {
    description: 'Implementation work — edits with checkpointing and thinking enabled',
    permissionMode: 'acceptEdits',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'high',
      CLAUDE_CODE_THINKING: '1',
      CLAUDE_CODE_AUTO_COMPACT: '1',
      CLAUDE_CODE_FILE_CHECKPOINTING: '1',
    },
  },
  research: {
    description: 'Investigation and analysis — read-heavy, no edits, deep reasoning',
    permissionMode: 'default',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'high',
      CLAUDE_CODE_THINKING: '1',
      CLAUDE_CODE_AUTO_COMPACT: '1',
    },
  },
  quickfix: {
    description: 'Small, fast fixes — brief output, low effort, quick turnaround',
    permissionMode: 'acceptEdits',
    env: {
      CLAUDE_CODE_BRIEF: '1',
      CLAUDE_CODE_EFFORT_LEVEL: 'low',
      CLAUDE_CODE_THINKING: '0',
    },
  },
  planning: {
    description: 'Architecture and design — plan mode, no edits until approved',
    permissionMode: 'plan',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'high',
      CLAUDE_CODE_THINKING: '1',
    },
  },
  background: {
    description: 'Long-running background/daemon work — autonomous, minimal prompts',
    permissionMode: 'bypassPermissions',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'medium',
      CLAUDE_CODE_AUTO_COMPACT: '1',
      CLAUDE_CODE_PROMPT_SUGGESTION: '0',
    },
  },
  swarm: {
    description: 'Multi-agent swarm coordination — autonomous, hooks-driven',
    permissionMode: 'bypassPermissions',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'medium',
      CLAUDE_CODE_AUTO_COMPACT: '1',
      CLAUDE_CODE_THINKING: '1',
    },
  },
  review: {
    description: 'Code review — read-only, no edits, thorough reasoning',
    permissionMode: 'default',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'high',
      CLAUDE_CODE_THINKING: '1',
      CLAUDE_CODE_BRIEF: '0',
    },
  },
  ci: {
    description: 'CI / automation pipelines — fast, non-interactive, no suggestions',
    permissionMode: 'dontAsk',
    env: {
      CLAUDE_CODE_FAST_MODE: '1',
      CLAUDE_CODE_EFFORT_LEVEL: 'low',
      CLAUDE_CODE_PROMPT_SUGGESTION: '0',
    },
  },
};

/**
 * Keyword → profile rules, evaluated in order. The first rule whose regex
 * matches the prompt wins, so more specific task types are listed first.
 */
const DETECTION_RULES = [
  { type: 'quickfix', re: /\b(typo|quick\s?-?fix|one-?liner|small fix|rename)\b/i },
  { type: 'ci', re: /\b(ci|cd|pipeline|github actions|workflow|deploy)\b/i },
  { type: 'swarm', re: /\b(swarm|multi-?agent|coordinate|orchestrat\w*|hive)\b/i },
  { type: 'background', re: /\b(background|daemon|monitor\w*|watch\b|long-?running)\b/i },
  { type: 'review', re: /\b(review|pull request|\bpr\b|audit|critique)\b/i },
  { type: 'planning', re: /\b(plan\w*|architect\w*|design|roadmap|strategy)\b/i },
  { type: 'research', re: /\b(research|investigat\w*|explore|compare|survey)\b/i },
  { type: 'coding', re: /\b(implement|code|build|add|write|refactor|create|fix|function|feature)\b/i },
];

/** List all available profile names. */
function listProfiles() {
  return Object.keys(PROFILES);
}

/** Return a profile by name, or `null` if it does not exist. */
function getProfile(name) {
  if (!name || !Object.prototype.hasOwnProperty.call(PROFILES, name)) {
    return null;
  }
  return PROFILES[name];
}

/**
 * Apply a profile's env vars to `process.env`. Returns `{ applied, permissionMode }`
 * where `applied` is the map of env vars that were set, or `null` if the profile
 * is unknown.
 */
function applyProfile(name) {
  const profile = getProfile(name);
  if (!profile) {
    return null;
  }
  const applied = {};
  for (const [key, val] of Object.entries(profile.env)) {
    process.env[key] = val;
    applied[key] = val;
  }
  return { applied, permissionMode: profile.permissionMode };
}

/**
 * Infer the most likely task type from a free-text prompt. Defaults to
 * `'coding'` for empty/unrecognised input.
 */
function detectTaskType(prompt) {
  if (!prompt || typeof prompt !== 'string') {
    return 'coding';
  }
  for (const rule of DETECTION_RULES) {
    if (rule.re.test(prompt)) {
      return rule.type;
    }
  }
  return 'coding';
}

module.exports = {
  PERMISSION_MODES,
  PROFILES,
  listProfiles,
  getProfile,
  applyProfile,
  detectTaskType,
};
