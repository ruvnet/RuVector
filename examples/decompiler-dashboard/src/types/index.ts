export interface ModuleMetrics {
  fragments: number;
  sizeBytes: number;
}

export interface VersionMetrics {
  version: string;
  sizeBytes: number;
  lines: number;
  functions: number;
  asyncFunctions: number;
  arrowFunctions: number;
  classes: number;
  extends: number;
  sourceFile: string;
  extractedAt: string;
  modules: Record<string, ModuleMetrics>;
}

export interface VersionData {
  id: string;
  label: string;
  metrics: VersionMetrics;
  modules: Record<string, string>;
  readme: string;
  manifest?: RvfManifest;
}

export interface RvfManifest {
  format: string;
  version: string;
  fileId: string;
  dimensions: number;
  metric: string;
  totalVectors: number;
  totalSegments: number;
  fileSizeBytes: number;
  epoch: number;
  segments: RvfSegment[];
}

export interface RvfSegment {
  id: number;
  type: string;
  offset: number;
  payloadLength: number;
}

export interface NpmPackageInfo {
  name: string;
  description: string;
  versions: string[];
  latest: string;
  distTags: Record<string, string>;
}

export interface DecompileResult {
  packageName: string;
  version: string;
  metrics: VersionMetrics;
  modules: Record<string, string>;
  originalSource: string;
  beautifiedSource: string;
}

export interface SearchResult {
  versionId: string;
  moduleName: string;
  lineNumber: number;
  lineContent: string;
  context: string[];
}

export type PageId = 'explorer' | 'decompiler' | 'rvf-viewer';
