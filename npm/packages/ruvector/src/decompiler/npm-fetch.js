/**
 * npm-fetch.js - Fetch package info and files from the npm registry.
 *
 * Uses the built-in Node 18+ fetch API. Retrieves package metadata from
 * registry.npmjs.org and file contents from unpkg.com / jsdelivr.
 */

'use strict';

const REGISTRY_BASE = 'https://registry.npmjs.org';
const JSDELIVR_BASE = 'https://data.jsdelivr.com/v1/package/npm';
const UNPKG_BASE = 'https://unpkg.com';

/**
 * Fetch package metadata from the npm registry.
 * @param {string} packageName - npm package name (e.g. 'express', '@anthropic-ai/claude-code')
 * @returns {Promise<{name: string, description: string, versions: string[], latest: string, distTags: object, packageJson: object}>}
 */
async function fetchPackageInfo(packageName) {
  const url = `${REGISTRY_BASE}/${encodeURIComponent(packageName).replace('%40', '@')}`;
  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
  });

  if (!resp.ok) {
    throw new Error(`Package "${packageName}" not found (HTTP ${resp.status})`);
  }

  const data = await resp.json();
  const versions = Object.keys(data.versions || {}).reverse();
  const distTags = data['dist-tags'] || {};
  const latest = distTags.latest || versions[0] || '';

  return {
    name: data.name,
    description: data.description || '',
    versions,
    latest,
    distTags,
    packageJson: data.versions?.[latest] || {},
  };
}

/**
 * Fetch the file list for a specific package version via jsDelivr.
 * @param {string} packageName
 * @param {string} version
 * @returns {Promise<Array<{name: string, hash: string, size: number}>>}
 */
async function fetchPackageFileList(packageName, version) {
  const encodedName = encodeURIComponent(packageName).replace('%40', '@');
  const url = `${JSDELIVR_BASE}/${encodedName}@${version}/flat`;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Could not list files for ${packageName}@${version}`);
  }
  const data = await resp.json();
  return (data.files || []);
}

/**
 * Fetch the content of a single file from unpkg.
 * @param {string} packageName
 * @param {string} version
 * @param {string} filePath - e.g. '/dist/index.js'
 * @returns {Promise<string>}
 */
async function fetchFileContent(packageName, version, filePath) {
  const cleanPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
  const url = `${UNPKG_BASE}/${packageName}@${version}/${cleanPath}`;
  const resp = await fetch(url, { redirect: 'follow' });
  if (!resp.ok) {
    throw new Error(`Could not fetch ${cleanPath} from ${packageName}@${version} (HTTP ${resp.status})`);
  }
  return resp.text();
}

/**
 * Find the main JS bundle file from a file list.
 * Checks common locations and falls back to the largest .js file.
 * @param {Array<{name: string, size: number}>} files
 * @param {object} [packageJson] - optional package.json to check main/module/browser fields
 * @returns {string|null}
 */
function findMainBundle(files, packageJson) {
  // Check package.json fields first
  if (packageJson) {
    const fields = ['browser', 'module', 'main'];
    for (const field of fields) {
      if (typeof packageJson[field] === 'string') {
        const normalized = packageJson[field].startsWith('/')
          ? packageJson[field]
          : '/' + packageJson[field];
        if (files.some((f) => f.name === normalized)) {
          return normalized;
        }
      }
    }
  }

  // Well-known candidate paths
  const candidates = [
    '/dist/cli.js',
    '/dist/index.js',
    '/dist/main.js',
    '/dist/bundle.js',
    '/lib/index.js',
    '/lib/cli.js',
    '/index.js',
    '/cli.js',
  ];

  for (const candidate of candidates) {
    if (files.some((f) => f.name === candidate)) {
      return candidate;
    }
  }

  // Fallback: largest JS file (prefer non-minified)
  const jsFiles = files
    .filter((f) => f.name.endsWith('.js') && !f.name.endsWith('.min.js'))
    .sort((a, b) => b.size - a.size);

  return jsFiles.length > 0 ? jsFiles[0].name : null;
}

/**
 * Parse a target string into its components.
 * Handles: 'express', 'express@4.18.2', '@scope/pkg@1.0.0',
 * './local.js', 'https://...'.
 * @param {string} target
 * @returns {{type: 'npm'|'file'|'url', name?: string, version?: string, path?: string, url?: string}}
 */
function parseTarget(target) {
  if (target.startsWith('http://') || target.startsWith('https://')) {
    return { type: 'url', url: target };
  }

  if (target.startsWith('.') || target.startsWith('/')) {
    return { type: 'file', path: target };
  }

  // npm package: @scope/name@version or name@version
  let name = target;
  let version = undefined;
  if (target.startsWith('@')) {
    // scoped: @scope/name@version
    const afterScope = target.indexOf('/', 1);
    if (afterScope > 0) {
      const atIdx = target.indexOf('@', afterScope + 1);
      if (atIdx > 0) {
        name = target.slice(0, atIdx);
        version = target.slice(atIdx + 1);
      }
    }
  } else {
    const atIdx = target.indexOf('@');
    if (atIdx > 0) {
      name = target.slice(0, atIdx);
      version = target.slice(atIdx + 1);
    }
  }

  return { type: 'npm', name, version };
}

module.exports = {
  fetchPackageInfo,
  fetchPackageFileList,
  fetchFileContent,
  findMainBundle,
  parseTarget,
  REGISTRY_BASE,
  UNPKG_BASE,
  JSDELIVR_BASE,
};
