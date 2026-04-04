#!/usr/bin/env node

/**
 * Tests for the RVAgent Optimizer (ADR-139).
 *
 * Validates:
 * - Each profile produces valid env vars
 * - Settings generator produces valid JSON
 * - Task type detection from prompts
 * - Tool schema validation
 * - Context optimizer calculations
 */

'use strict';

const path = require('path');
const optimizerPath = path.join(__dirname, '..', 'src', 'optimizer');

const optimizer = require(path.join(optimizerPath, 'index.js'));
const toolSchemas = require(path.join(optimizerPath, 'tool-schemas.js'));
const context = require(path.join(optimizerPath, 'context.js'));
const settingsGen = require(path.join(optimizerPath, 'settings-generator.js'));

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    passed++;
    console.log(`  \u2713 ${message}`);
  } else {
    failed++;
    console.error(`  \u2717 FAIL: ${message}`);
  }
}

function section(name) {
  console.log(`\n${name}`);
  console.log('='.repeat(name.length));
}

// ---------------------------------------------------------------------------
// Optimizer profiles
// ---------------------------------------------------------------------------

section('Optimizer Profiles');

const profiles = optimizer.listProfiles();
assert(profiles.length >= 8, `At least 8 profiles defined (got ${profiles.length})`);

for (const name of profiles) {
  const profile = optimizer.getProfile(name);
  assert(profile !== null, `getProfile('${name}') returns non-null`);
  assert(typeof profile.env === 'object', `Profile '${name}' has env object`);
  assert(typeof profile.permissionMode === 'string', `Profile '${name}' has permissionMode string`);
  assert(typeof profile.description === 'string', `Profile '${name}' has description`);

  // All env var keys should start with CLAUDE_CODE_
  for (const key of Object.keys(profile.env)) {
    assert(
      key.startsWith('CLAUDE_CODE_'),
      `Profile '${name}' env key '${key}' starts with CLAUDE_CODE_`
    );
  }

  // Permission mode must be a known mode
  assert(
    optimizer.PERMISSION_MODES.includes(profile.permissionMode),
    `Profile '${name}' permissionMode '${profile.permissionMode}' is valid`
  );
}

assert(optimizer.getProfile('nonexistent') === null, 'Unknown profile returns null');

// ---------------------------------------------------------------------------
// applyProfile
// ---------------------------------------------------------------------------

section('Apply Profile');

// Save and restore env
const savedEnv = { ...process.env };

const result = optimizer.applyProfile('quickfix');
assert(result !== null, 'applyProfile("quickfix") returns non-null');
assert(result.applied.CLAUDE_CODE_BRIEF === '1', 'BRIEF set to 1 for quickfix');
assert(result.applied.CLAUDE_CODE_EFFORT_LEVEL === 'low', 'EFFORT_LEVEL set to low for quickfix');
assert(process.env.CLAUDE_CODE_BRIEF === '1', 'process.env.CLAUDE_CODE_BRIEF is set');

// Restore env
for (const key of Object.keys(result.applied)) {
  delete process.env[key];
}
Object.assign(process.env, savedEnv);

assert(optimizer.applyProfile('nonexistent') === null, 'applyProfile unknown returns null');

// ---------------------------------------------------------------------------
// detectTaskType
// ---------------------------------------------------------------------------

section('Task Type Detection');

assert(
  optimizer.detectTaskType('Please implement a new auth module') === 'coding',
  'Detects "implement" as coding'
);
assert(
  optimizer.detectTaskType('Research the best database patterns') === 'research',
  'Detects "research" as research'
);
assert(
  optimizer.detectTaskType('Fix this typo in the readme') === 'quickfix',
  'Detects "fix" + "typo" as quickfix'
);
assert(
  optimizer.detectTaskType('Plan the new architecture for v3') === 'planning',
  'Detects "plan" + "architect" as planning'
);
assert(
  optimizer.detectTaskType('Run background monitoring daemon') === 'background',
  'Detects "background" + "daemon" as background'
);
assert(
  optimizer.detectTaskType('Spawn a multi-agent swarm to coordinate') === 'swarm',
  'Detects "swarm" + "coordinate" as swarm'
);
assert(
  optimizer.detectTaskType('Review this pull request') === 'review',
  'Detects "review" as review'
);
assert(
  optimizer.detectTaskType('Set up CI pipeline with GitHub Actions') === 'ci',
  'Detects "ci" + "pipeline" as ci'
);
assert(
  optimizer.detectTaskType('') === 'coding',
  'Empty prompt defaults to coding'
);
assert(
  optimizer.detectTaskType(null) === 'coding',
  'Null prompt defaults to coding'
);

// ---------------------------------------------------------------------------
// Tool Schemas
// ---------------------------------------------------------------------------

section('Tool Schemas');

assert(toolSchemas.TOOL_NAMES.length >= 15, `At least 15 tool schemas (got ${toolSchemas.TOOL_NAMES.length})`);
assert(toolSchemas.TOOL_SCHEMAS.Bash !== undefined, 'Bash schema exists');
assert(toolSchemas.TOOL_SCHEMAS.Read !== undefined, 'Read schema exists');
assert(toolSchemas.TOOL_SCHEMAS.Edit !== undefined, 'Edit schema exists');
assert(toolSchemas.TOOL_SCHEMAS.Write !== undefined, 'Write schema exists');
assert(toolSchemas.TOOL_SCHEMAS.Glob !== undefined, 'Glob schema exists');
assert(toolSchemas.TOOL_SCHEMAS.Grep !== undefined, 'Grep schema exists');

// Validate a correct tool call
const validBash = toolSchemas.validateToolCall('Bash', { command: 'ls -la' });
assert(validBash.valid === true, 'Valid Bash call passes validation');

// Validate missing required field
const invalidBash = toolSchemas.validateToolCall('Bash', {});
assert(invalidBash.valid === false, 'Missing required field fails validation');
assert(
  invalidBash.errors.some((e) => e.includes('command')),
  'Error mentions missing "command" field'
);

// Validate wrong type
const wrongType = toolSchemas.validateToolCall('Read', { file_path: 123 });
assert(wrongType.valid === false, 'Wrong type fails validation');

// Unknown tool passes (MCP extension)
const unknown = toolSchemas.validateToolCall('CustomMcpTool', { foo: 'bar' });
assert(unknown.valid === true, 'Unknown tool name passes validation');

// Null args fails
const nullArgs = toolSchemas.validateToolCall('Bash', null);
assert(nullArgs.valid === false, 'Null args fails validation');

// Optional fields
const readOptional = toolSchemas.validateToolCall('Read', { file_path: '/tmp/test' });
assert(readOptional.valid === true, 'Optional fields can be omitted');

const readFull = toolSchemas.validateToolCall('Read', {
  file_path: '/tmp/test',
  offset: 10,
  limit: 50,
});
assert(readFull.valid === true, 'All fields provided passes validation');

// Cached patterns
const cached = toolSchemas.getCachedPattern('Glob', '**/*.ts');
assert(cached !== null, 'Cached pattern for Glob:**/*.ts exists');
assert(toolSchemas.getCachedPattern('Foo', 'bar') === null, 'Unknown cached pattern returns null');

// ---------------------------------------------------------------------------
// Context Optimizer
// ---------------------------------------------------------------------------

section('Context Optimizer');

assert(
  context.shouldCompact(170000, 200000) === true,
  '170k/200k triggers compaction (85% > 80%)'
);
assert(
  context.shouldCompact(100000, 200000) === false,
  '100k/200k does not trigger compaction (50% < 80%)'
);
assert(
  context.shouldCompact(160001, 200000) === true,
  '160001/200k triggers compaction (just over 80%)'
);
assert(
  context.shouldCompact(160000, 200000) === false,
  '160000/200k does not trigger (exactly 80%)'
);

const deferred = context.buildDeferredToolList(['Read', 'Edit', 'Bash']);
assert(deferred.included.length === 3, 'Deferred: 3 tools included');
assert(deferred.deferred.length > 0, 'Deferred: remaining tools deferred');
assert(deferred.tokensSaved > 0, 'Deferred: tokens saved > 0');
assert(
  deferred.included.includes('Read') && deferred.included.includes('Edit'),
  'Deferred: correct tools included'
);

const prompt = context.buildCacheOptimizedPrompt('static rules here', 'dynamic context');
assert(prompt.cached === 'static rules here', 'Cache prompt: static part preserved');
assert(prompt.dynamic === 'dynamic context', 'Cache prompt: dynamic part preserved');
assert(prompt.totalTokensEstimate > 0, 'Cache prompt: token estimate > 0');
assert(prompt.cacheableTokens > 0, 'Cache prompt: cacheable tokens > 0');

assert(context.estimateTokens('hello world') > 0, 'estimateTokens returns positive');
assert(context.estimateTokens('') === 0, 'estimateTokens empty string is 0');
assert(context.estimateTokens(null) === 0, 'estimateTokens null is 0');

const codingTools = context.recommendToolsForTask('coding');
assert(codingTools.includes('Edit'), 'Coding task includes Edit tool');
assert(codingTools.includes('Write'), 'Coding task includes Write tool');

const reviewTools = context.recommendToolsForTask('review');
assert(!reviewTools.includes('Edit'), 'Review task does not include Edit');
assert(reviewTools.includes('Read'), 'Review task includes Read');

assert(context.getContextSize('claude-sonnet-4-20250514') === 200000, 'Known model context size');
assert(context.getContextSize('unknown-model') === 200000, 'Unknown model falls back to default');

// ---------------------------------------------------------------------------
// Settings Generator
// ---------------------------------------------------------------------------

section('Settings Generator');

const codingSettings = settingsGen.generateSettings({
  taskType: 'coding',
  permissionMode: 'acceptEdits',
});
assert(typeof codingSettings === 'object', 'generateSettings returns object');
assert(
  codingSettings.permissions.defaultMode === 'acceptEdits',
  'Coding settings: acceptEdits mode'
);
assert(codingSettings.autoCompactEnabled === true, 'Auto-compact enabled');
assert(codingSettings.fileCheckpointingEnabled === true, 'File checkpointing enabled');
assert(codingSettings.alwaysThinkingEnabled === true, 'Coding enables thinking');
assert(codingSettings.hooks !== undefined, 'Settings include hooks');

const ciSettings = settingsGen.generateSettings({
  taskType: 'ci',
  permissionMode: 'dontAsk',
});
assert(ciSettings.fastMode === true, 'CI settings: fast mode');
assert(ciSettings.promptSuggestionEnabled === false, 'CI: no prompt suggestions');

const swarmSettings = settingsGen.generateSettings({
  taskType: 'swarm',
  permissionMode: 'bypassPermissions',
});
assert(swarmSettings.hooks.SessionStart !== undefined, 'Swarm settings include SessionStart hook');

// JSON serialisation
const json = settingsGen.formatSettings(codingSettings);
assert(typeof json === 'string', 'formatSettings returns string');
const parsed = JSON.parse(json);
assert(parsed.permissions !== undefined, 'Formatted JSON is valid and parseable');

// Merge settings
const existing = {
  permissions: { defaultMode: 'default' },
  customField: 'preserved',
  hooks: {
    PreToolUse: [
      { matcher: 'custom', hooks: [{ type: 'command', command: 'echo custom' }] },
    ],
  },
};
const merged = settingsGen.mergeSettings(existing, codingSettings);
assert(merged.customField === 'preserved', 'Merge preserves existing custom fields');
assert(
  merged.permissions.defaultMode === 'acceptEdits',
  'Merge updates permission mode'
);
assert(merged.hooks.PreToolUse.length >= 2, 'Merge appends hooks without duplicating');

// Merge with null existing
const fromNull = settingsGen.mergeSettings(null, codingSettings);
assert(
  fromNull.permissions.defaultMode === 'acceptEdits',
  'Merge with null existing returns generated'
);

// Error case
let threw = false;
try {
  settingsGen.generateSettings({});
} catch (e) {
  threw = true;
}
assert(threw, 'generateSettings throws when taskType is missing');

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${'='.repeat(40)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
console.log('='.repeat(40));

if (failed > 0) {
  process.exit(1);
}
