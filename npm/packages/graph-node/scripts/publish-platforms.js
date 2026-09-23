#!/usr/bin/env node
/**
 * Publish platform-specific @ruvector/graph-node packages to npm
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const platforms = [
  { name: 'linux-x64-gnu', nodeFile: 'index.linux-x64-gnu.node' },
  { name: 'linux-arm64-gnu', nodeFile: 'index.linux-arm64-gnu.node' },
  { name: 'darwin-x64', nodeFile: 'index.darwin-x64.node' },
  { name: 'darwin-arm64', nodeFile: 'index.darwin-arm64.node' },
  { name: 'win32-x64-msvc', nodeFile: 'index.win32-x64-msvc.node' },
];

const rootDir = path.join(__dirname, '..');
const version = require(path.join(rootDir, 'package.json')).version;

console.log('Publishing @ruvector/graph-node platform packages v' + version + '\n');

// A missing binary or a failed publish must fail the job: the main package's
// optionalDependencies pin every platform at this exact version, so a skipped
// or swallowed platform ships an uninstallable release (same class as #1007).
const missing = platforms.filter((p) => !fs.existsSync(path.join(rootDir, p.nodeFile)));
if (missing.length) {
  console.error('Missing platform binaries: ' + missing.map((p) => p.nodeFile).join(', '));
  process.exit(1);
}

function isPublished(pkgName) {
  try {
    execSync('npm view ' + pkgName + '@' + version + ' version', { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

let failed = 0;
for (const platform of platforms) {
  const pkgName = '@ruvector/graph-node-' + platform.name;
  const nodeFile = path.join(rootDir, platform.nodeFile);

  if (isPublished(pkgName)) {
    console.log(pkgName + '@' + version + ' already on npm - skipping\n');
    continue;
  }

  const tmpDir = path.join(rootDir, 'npm', platform.name);
  fs.mkdirSync(tmpDir, { recursive: true });

  // Create package.json for platform package
  const pkgJson = {
    name: pkgName,
    version: version,
    description: 'RuVector Graph Node.js bindings for ' + platform.name,
    main: 'ruvector-graph.node',
    files: ['ruvector-graph.node'],
    os: platform.name.includes('linux') ? ['linux'] :
        platform.name.includes('darwin') ? ['darwin'] :
        platform.name.includes('win32') ? ['win32'] : [],
    cpu: platform.name.includes('x64') ? ['x64'] :
         platform.name.includes('arm64') ? ['arm64'] : [],
    engines: { node: '>=18.0.0' },
    license: 'MIT',
    repository: {
      type: 'git',
      url: 'https://github.com/ruvnet/ruvector.git',
      directory: 'npm/packages/graph-node'
    },
    publishConfig: { access: 'public' }
  };

  fs.writeFileSync(
    path.join(tmpDir, 'package.json'),
    JSON.stringify(pkgJson, null, 2)
  );

  // Copy the .node file
  fs.copyFileSync(nodeFile, path.join(tmpDir, 'ruvector-graph.node'));

  // Publish
  console.log('Publishing ' + pkgName + '@' + version + '...');
  try {
    execSync('npm publish --access public', { cwd: tmpDir, stdio: 'inherit' });
    console.log('Published ' + pkgName + '@' + version + '\n');
  } catch (e) {
    console.error('Failed to publish ' + pkgName + ': ' + e.message + '\n');
    failed++;
  }

  // Cleanup
  fs.rmSync(tmpDir, { recursive: true, force: true });
}

if (failed) {
  console.error(failed + ' platform package(s) failed to publish');
  process.exit(1);
}
console.log('Done!');
