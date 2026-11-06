#!/usr/bin/env node
/**
 * `ruvector skills list|install`: the shipped skills are valid SKILL.md files
 * (name + description frontmatter, the format both Claude Code and Codex read),
 * `skills/` is in the published `files`, and install copies them into
 * .claude/skills and .agents/skills without clobbering or path tricks.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'bin', 'cli.js');
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-skills-'));
const run = (args) => {
  const r = spawnSync(process.execPath, [CLI, ...args], { cwd: work, env: { ...process.env, HOME: work, NO_COLOR: '1' }, encoding: 'utf8' });
  return { status: r.status, json: JSON.parse(r.stdout.trim().split(/\r?\n/).pop()) };
};

try {
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
  assert.ok(pkg.files.includes('skills/'), 'skills/ must be published');

  const { json: listed } = run(['skills', 'list', '--json']);
  const names = listed.skills.map((s) => s.name);
  assert.ok(names.includes('ruvector-integrations'), `shipped skills: ${names}`);
  for (const name of names) {
    const body = fs.readFileSync(path.join(ROOT, 'skills', name, 'SKILL.md'), 'utf8');
    const front = body.match(/^---\n([\s\S]*?)\n---\n/);
    assert.ok(front, `${name}: frontmatter`);
    assert.match(front[1], new RegExp(`^name: ${name}$`, 'm'), `${name}: name matches directory`);
    assert.match(front[1], /^description: .{40,}$/m, `${name}: description`);
  }
  console.log(`  PASS  shipped skills have valid frontmatter: ${names.join(', ')}`);

  let r = run(['skills', 'install', '--json']);
  assert.strictEqual(r.json.success, true);
  for (const rel of ['.claude/skills/ruvector-integrations/SKILL.md', '.agents/skills/ruvector-integrations/SKILL.md']) {
    assert.ok(fs.existsSync(path.join(work, rel)), `${rel} installed`);
  }
  const mine = path.join(work, '.claude/skills/ruvector-integrations/SKILL.md');
  fs.writeFileSync(mine, 'local edits');
  r = run(['skills', 'install', '--target', 'claude', '--json']);
  assert.strictEqual(r.json.results[0].status, 'exists');
  assert.strictEqual(fs.readFileSync(mine, 'utf8'), 'local edits', 'no clobber without --force');
  r = run(['skills', 'install', '--target', 'claude', '--force', '--json']);
  assert.notStrictEqual(fs.readFileSync(mine, 'utf8'), 'local edits', '--force overwrites');
  console.log('  PASS  skills install writes .claude/skills + .agents/skills, respects existing files');

  assert.strictEqual(run(['skills', 'install', '../../etc', '--json']).json.success, false);
  assert.strictEqual(run(['skills', 'install', '--target', 'vim', '--json']).json.success, false);
  console.log('  PASS  unknown skill names and targets are rejected');
} catch (e) {
  console.error(`  FAIL  ${e.stack || e.message}`);
  process.exitCode = 1;
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
