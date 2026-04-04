/**
 * Settings generator — produces optimal .claude/settings.json for a
 * given task profile. Based on decompiled Claude Code v2.1.91 schema.
 *
 * The generated settings include:
 * - Permission mode matching the task type
 * - Hooks for RVAgent integration (pre-task optimise, post-session learn)
 * - Feature flags (thinking, fast mode, auto-compact, checkpointing)
 */

'use strict';

/**
 * Generate a complete settings.json object for a given profile.
 *
 * @param {object} profile
 * @param {string} profile.taskType       - one of the known task types
 * @param {string} profile.permissionMode - permission mode for this task
 * @param {string} [profile.description]  - human-readable description
 * @param {object} [profile.env]          - environment variables
 * @returns {object} A settings.json-compatible object
 */
function generateSettings(profile) {
  if (!profile || !profile.taskType) {
    throw new Error('profile.taskType is required');
  }

  const isInteractive = !['ci', 'background'].includes(profile.taskType);
  const needsThinking = ['planning', 'research', 'coding'].includes(profile.taskType);
  const isFast = ['quickfix', 'ci'].includes(profile.taskType);

  const settings = {
    permissions: {
      defaultMode: profile.permissionMode || 'default',
    },
    autoCompactEnabled: true,
    fileCheckpointingEnabled: true,
    promptSuggestionEnabled: isInteractive,
  };

  if (needsThinking) {
    settings.alwaysThinkingEnabled = true;
  }

  if (isFast) {
    settings.fastMode = true;
  }

  // Hooks for RVAgent integration
  settings.hooks = buildHooks(profile.taskType);

  return settings;
}

/**
 * Build the hooks section for a given task type.
 *
 * @param {string} taskType
 * @returns {object}
 */
function buildHooks(taskType) {
  const hooks = {};

  // Pre-tool-use hook: optimise before file-mutating tools
  hooks.PreToolUse = [
    {
      matcher: 'Bash|Edit|Write',
      hooks: [
        {
          type: 'command',
          command: 'npx @claude-flow/cli@latest hooks pre-task --optimize',
        },
      ],
    },
  ];

  // Post-session hook: persist learnings
  hooks.Stop = [
    {
      matcher: '',
      hooks: [
        {
          type: 'command',
          command: 'npx @claude-flow/cli@latest hooks session-end --learn',
        },
      ],
    },
  ];

  // Swarm tasks get an additional session-start hook for team discovery
  if (taskType === 'swarm') {
    hooks.SessionStart = [
      {
        matcher: '',
        hooks: [
          {
            type: 'command',
            command: 'npx @claude-flow/cli@latest hooks session-start --optimize --team',
          },
        ],
      },
    ];
  }

  return hooks;
}

/**
 * Merge generated settings into an existing settings object, preserving
 * user customisations that do not conflict.
 *
 * @param {object} existing - the current settings.json content
 * @param {object} generated - output from generateSettings()
 * @returns {object} merged settings
 */
function mergeSettings(existing, generated) {
  if (!existing) return generated;

  const merged = { ...existing };

  // Merge permissions
  merged.permissions = {
    ...(existing.permissions || {}),
    ...(generated.permissions || {}),
  };

  // Merge hooks (append, do not replace)
  if (generated.hooks) {
    merged.hooks = merged.hooks || {};
    for (const [event, hookList] of Object.entries(generated.hooks)) {
      const existingHooks = merged.hooks[event] || [];
      // Avoid duplicates by checking command strings
      const existingCmds = new Set(
        existingHooks.flatMap((h) =>
          (h.hooks || []).map((inner) => inner.command)
        )
      );
      const newHooks = hookList.filter((h) =>
        (h.hooks || []).some((inner) => !existingCmds.has(inner.command))
      );
      merged.hooks[event] = [...existingHooks, ...newHooks];
    }
  }

  // Copy feature flags (generated wins)
  for (const key of [
    'autoCompactEnabled',
    'fileCheckpointingEnabled',
    'promptSuggestionEnabled',
    'alwaysThinkingEnabled',
    'fastMode',
  ]) {
    if (generated[key] !== undefined) {
      merged[key] = generated[key];
    }
  }

  return merged;
}

/**
 * Serialise settings to a formatted JSON string.
 *
 * @param {object} settings
 * @returns {string}
 */
function formatSettings(settings) {
  return JSON.stringify(settings, null, 2);
}

module.exports = {
  generateSettings,
  buildHooks,
  mergeSettings,
  formatSettings,
};
