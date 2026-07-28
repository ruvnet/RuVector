import { readFileSync } from 'fs';
import { join } from 'path';

const PACKAGE_ROOT = __dirname;
const COMMAND_SOURCES = [
  'agent-coordinator',
  'regional-agent',
  'swarm-manager',
  'coordination-protocol',
].flatMap((stem) => [`${stem}.ts`, `${stem}.js`]);

describe('claude-flow command security', () => {
  test.each(COMMAND_SOURCES)('%s does not invoke a shell', (file) => {
    const source = readFileSync(join(PACKAGE_ROOT, file), 'utf8');
    expect(source).not.toMatch(/\bexec\s*\(/);
    expect(source).not.toMatch(/\bexecAsync\s*\(/);
    expect(source).not.toMatch(/\bshell\s*:\s*true\b/);
  });

  test.each(['claude-flow-runner.ts', 'claude-flow-runner.js'])(
    '%s uses execFile with an argument array',
    (file) => {
      const source = readFileSync(join(PACKAGE_ROOT, file), 'utf8');
      expect(source).toMatch(/\bexecFileAsync\s*\(/);
      expect(source).toMatch(/\['claude-flow@alpha',\s*\.\.\.args\.map\(String\)\]/);
      expect(source).not.toMatch(/\bexec\s*\(/);
      expect(source).not.toMatch(/\bshell\s*:\s*true\b/);
    },
  );
});
