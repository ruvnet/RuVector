import type { RvfManifest, RvfSegment } from '../types';

/**
 * Parse an RVF manifest JSON file.
 * The actual binary RVF parsing would require WASM; this handles the
 * manifest sidecar that accompanies each .rvf file.
 */
export function parseRvfManifest(json: string): RvfManifest {
  const data = JSON.parse(json);
  return {
    format: data.format || 'unknown',
    version: data.version || '0',
    fileId: data.fileId || '',
    dimensions: data.dimensions || 0,
    metric: data.metric || 'unknown',
    totalVectors: data.totalVectors || 0,
    totalSegments: data.totalSegments || 0,
    fileSizeBytes: data.fileSizeBytes || 0,
    epoch: data.epoch || 0,
    segments: (data.segments || []).map((s: Record<string, unknown>) => ({
      id: s.id as number,
      type: s.type as string,
      offset: s.offset as number,
      payloadLength: s.payloadLength as number,
    })),
  };
}

/**
 * Parse a binary RVF file header to extract basic info.
 * The RVF format starts with a magic number and header.
 */
export function parseRvfBinaryHeader(buffer: ArrayBuffer): Partial<RvfManifest> {
  const view = new DataView(buffer);
  const decoder = new TextDecoder();

  // Try to read magic bytes - RVF files start with "RVF\x00"
  const magic = decoder.decode(new Uint8Array(buffer, 0, 3));
  if (magic !== 'RVF') {
    throw new Error('Not a valid RVF file (bad magic bytes)');
  }

  // Basic header parsing - version byte at offset 3
  const headerVersion = view.getUint8(3);

  return {
    format: 'rvf-binary',
    version: String(headerVersion),
    fileSizeBytes: buffer.byteLength,
  };
}

export function getSegmentTypeColor(type: string): string {
  const colors: Record<string, string> = {
    manifest: '#00e5ff',
    vec: '#b388ff',
    witness: '#ff80ab',
    hnsw: '#69f0ae',
    metadata: '#ffd740',
  };
  return colors[type] || '#90a4ae';
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
