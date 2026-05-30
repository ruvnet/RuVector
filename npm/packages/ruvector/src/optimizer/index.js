'use strict';

/**
 * RVAgent Optimizer — profile management and task-type detection.
 *
 * Profiles map task types to optimal CLAUDE_CODE_* env vars and a
 * permission mode so callers can apply the right settings without
 * knowing the full schema.
 */

const PERMISSION_MODES = ['default', 'acceptEdits', 'auto', 'bypassPermissions'];

const PROFILES = {
  coding: {
    description: 'General coding tasks: implement, refactor, add features',
    permissionMode: 'acceptEdits',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'high',
      CLAUDE_CODE_BRIEF: '0',
      CLAUDE_CODE_AUTO_COMPACT: '1',
    },
  },
  research: {
    description: 'Research and information gathering tasks',
    permissionMode: 'default',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'high',
      CLAUDE_CODE_BRIEF: '0',
      CLAUDE_CODE_AUTO_COMPACT: '1',
    },
  },
  quickfix: {
    description: 'Small, fast fixes: typos, simple patches, single-line changes',
    permissionMode: 'acceptEdits',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'low',
      CLAUDE_CODE_BRIEF: '1',
      CLAUDE_CODE_AUTO_COMPACT: '0',
    },
  },
  planning: {
    description: 'Architecture and planning tasks: design, spec, roadmap',
    permissionMode: 'default',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'high',
      CLAUDE_CODE_BRIEF: '0',
      CLAUDE_CODE_AUTO_COMPACT: '1',
    },
  },
  background: {
    description: 'Background / daemon / monitoring tasks',
    permissionMode: 'auto',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'medium',
      CLAUDE_CODE_BRIEF: '1',
      CLAUDE_CODE_AUTO_COMPACT: '1',
    },
  },
  swarm: {
    description: 'Multi-agent swarm coordination tasks',
    permissionMode: 'bypassPermissions',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'high',
      CLAUDE_CODE_BRIEF: '0',
      CLAUDE_CODE_AUTO_COMPACT: '1',
    },
  },
  review: {
    description: 'Code review and audit tasks',
    permissionMode: 'default',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'high',
      CLAUDE_CODE_BRIEF: '0',
      CLAUDE_CODE_AUTO_COMPACT: '0',
    },
  },
  ci: {
    description: 'CI/CD pipeline and DevOps tasks',
    permissionMode: 'acceptEdits',
    env: {
      CLAUDE_CODE_EFFORT_LEVEL: 'medium',
      CLAUDE_CODE_BRIEF: '1',
      CLAUDE_CODE_AUTO_COMPACT: '1',
    },
  },
};

// Task-type detection: ordered keyword rules, first match wins.
const TASK_TYPE_RULES = [
  { type: 'quickfix',   keywords: [['fix', 'typo'], ['fix', 'bug'], ['patch', 'small'], ['hotfix']] },
  { type: 'swarm',      keywords: [['swarm'], ['multi-agent'], ['coordinate', 'agent']] },
  { type: 'review',     keywords: [['review'], ['audit', 'code'], ['code review']] },
  { type: 'ci',         keywords: [['ci', 'pipeline'], ['github actions'], ['workflow', 'action'], ['deploy', 'pipeline']] },
  { type: 'background', keywords: [['background'], ['daemon'], ['monitor', 'run'], ['cron']] },
  { type: 'planning',   keywords: [['plan', 'architect'], ['architecture'], ['design', 'system'], ['roadmap'], ['spec']] },
  { type: 'research',   keywords: [['research'], ['investigate'], ['explore', 'pattern'], ['find', 'best']] },
  { type: 'coding',     keywords: [['implement'], ['build'], ['create', 'module'], ['refactor'], ['add', 'feature']] },
];

/**
 * Detect the task type from a natural-language prompt.
 * Returns 'coding' as the default when no rule matches.
 */
function detectTaskType(prompt) {
  if (!prompt || typeof prompt !== 'string') return 'coding';
  const lower = prompt.toLowerCase();
  for (const rule of TASK_TYPE_RULES) {
    for (const keywordSet of rule.keywords) {
      if (keywordSet.every(kw => lower.includes(kw))) return rule.type;
    }
  }
  return 'coding';
}

/** Return an array of all profile names. */
function listProfiles() {
  return Object.keys(PROFILES);
}

/**
 * Return a profile by name, or null if unknown.
 * The returned object is a shallow copy — safe to mutate.
 */
function getProfile(name) {
  const p = PROFILES[name];
  if (!p) return null;
  return { ...p, env: { ...p.env } };
}

/**
 * Apply a profile by name: write env vars into process.env.
 * Returns { applied: { KEY: value, … } } or null when the profile is unknown.
 */
function applyProfile(name) {
  const profile = getProfile(name);
  if (!profile) return null;
  const applied = {};
  for (const [key, value] of Object.entries(profile.env)) {
    process.env[key] = value;
    applied[key] = value;
  }
  return { applied, profile };
}

module.exports = {
  PERMISSION_MODES,
  listProfiles,
  getProfile,
  applyProfile,
  detectTaskType,
};
