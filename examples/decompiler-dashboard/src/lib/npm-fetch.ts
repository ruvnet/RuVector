import type { NpmPackageInfo } from '../types';

const REGISTRY_BASE = 'https://registry.npmjs.org';
const JSDELIVR_BASE = 'https://data.jsdelivr.com/v1/package/npm';
const UNPKG_BASE = 'https://unpkg.com';

export async function fetchPackageInfo(packageName: string): Promise<NpmPackageInfo> {
  const url = `${REGISTRY_BASE}/${encodeURIComponent(packageName)}`;
  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
  });

  if (!resp.ok) {
    throw new Error(`Package "${packageName}" not found (HTTP ${resp.status})`);
  }

  const data = await resp.json();
  const versions = Object.keys(data.versions || {}).reverse();
  const distTags = data['dist-tags'] || {};

  return {
    name: data.name,
    description: data.description || '',
    versions,
    latest: distTags.latest || versions[0] || '',
    distTags,
  };
}

export interface FileListEntry {
  name: string;
  hash: string;
  size: number;
}

export async function fetchPackageFileList(
  packageName: string,
  version: string,
): Promise<FileListEntry[]> {
  const url = `${JSDELIVR_BASE}/${encodeURIComponent(packageName)}@${version}/flat`;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Could not list files for ${packageName}@${version}`);
  }
  const data = await resp.json();
  return (data.files || []) as FileListEntry[];
}

export async function fetchFileContent(
  packageName: string,
  version: string,
  filePath: string,
): Promise<string> {
  // Remove leading slash if present
  const cleanPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
  const url = `${UNPKG_BASE}/${packageName}@${version}/${cleanPath}`;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Could not fetch ${cleanPath} from ${packageName}@${version}`);
  }
  return resp.text();
}

/**
 * Find the main JS bundle file from the package file list.
 * Looks for common patterns: dist/index.js, lib/index.js, cli.js, etc.
 */
export function findMainBundle(files: FileListEntry[]): string | null {
  const candidates = [
    '/dist/cli.js',
    '/dist/index.js',
    '/dist/main.js',
    '/lib/index.js',
    '/lib/cli.js',
    '/index.js',
    '/cli.js',
    '/dist/bundle.js',
  ];

  for (const candidate of candidates) {
    if (files.some((f) => f.name === candidate)) {
      return candidate;
    }
  }

  // Find the largest JS file as a fallback
  const jsFiles = files
    .filter((f) => f.name.endsWith('.js') && !f.name.includes('.min.'))
    .sort((a, b) => b.size - a.size);

  return jsFiles.length > 0 ? jsFiles[0].name : null;
}
