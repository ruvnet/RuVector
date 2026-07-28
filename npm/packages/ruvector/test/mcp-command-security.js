#!/usr/bin/env node

/**
 * Regression guard for MCP command injection.
 *
 * Tool parameters arrive from an MCP client and must never be interpolated
 * into a shell command. The server intentionally uses execFileSync with an
 * argument array for both the local CLI and optional worker integration.
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const serverPath = path.join(__dirname, '..', 'bin', 'mcp-server.js');
const source = fs.readFileSync(serverPath, 'utf8');

assert.doesNotMatch(
  source,
  /\bexecSync\s*\(/,
  'MCP server must not execute shell command strings',
);
assert.doesNotMatch(
  source,
  /\bshell\s*:\s*true\b/,
  'MCP child processes must not enable a shell',
);
assert.match(
  source,
  /execFileSync\(process\.execPath,\s*\[RUVECTOR_CLI,\s*\.\.\.args\.map\(String\)\]/,
  'local hook commands must use the Node executable with an argument array',
);
assert.match(
  source,
  /execFileSync\(NPX_COMMAND,\s*\[packageSpec,\s*\.\.\.args\.map\(String\)\]/,
  'worker commands must use npx with an argument array',
);

console.log('MCP command execution is shell-free');
