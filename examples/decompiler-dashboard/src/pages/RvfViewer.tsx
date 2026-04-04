import { useState, useCallback } from 'react';
import type { VersionData, RvfManifest } from '../types';
import { parseRvfManifest, parseRvfBinaryHeader, getSegmentTypeColor, formatBytes } from '../lib/rvf-parser';
import { MetricsCard } from '../components/MetricsCard';

interface RvfViewerProps {
  versions: VersionData[];
  showToast: (msg: string) => void;
}

export function RvfViewer({ versions, showToast }: RvfViewerProps) {
  const [manifest, setManifest] = useState<RvfManifest | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'preloaded' | 'upload'>('preloaded');
  const [selectedVersionIdx, setSelectedVersionIdx] = useState(0);

  const handleFileDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [],
  );

  const handleFile = useCallback((file: File) => {
    setFileName(file.name);

    if (file.name.endsWith('.json')) {
      file.text().then((text) => {
        try {
          const parsed = parseRvfManifest(text);
          setManifest(parsed);
        } catch (err) {
          console.error('Failed to parse manifest:', err);
        }
      });
    } else if (file.name.endsWith('.rvf')) {
      file.arrayBuffer().then((buf) => {
        try {
          const header = parseRvfBinaryHeader(buf);
          setManifest({
            format: header.format || 'rvf-binary',
            version: header.version || '?',
            fileId: 'uploaded',
            dimensions: 0,
            metric: 'unknown',
            totalVectors: 0,
            totalSegments: 0,
            fileSizeBytes: header.fileSizeBytes || buf.byteLength,
            epoch: 0,
            segments: [],
          });
        } catch (err) {
          console.error('Failed to parse RVF binary:', err);
        }
      });
    }
  }, []);

  const preloadedManifests = versions.filter((v) => v.manifest);

  const handleLoadPreloaded = useCallback(
    (idx: number) => {
      const v = preloadedManifests[idx];
      if (v?.manifest) {
        setManifest(v.manifest);
        setFileName(`claude-code-${v.id}.rvf`);
        setSelectedVersionIdx(idx);
      }
    },
    [preloadedManifests],
  );

  return (
    <div className="max-w-[1600px] mx-auto p-4">
      <h2 className="text-lg font-semibold text-gray-100 mb-4">
        RVF <span className="text-accent-purple glow-purple">Viewer</span>
      </h2>

      {/* Tabs */}
      <div className="flex gap-1 mb-4">
        <button
          onClick={() => setActiveTab('preloaded')}
          className={`px-3 py-1.5 rounded-md text-sm transition-colors ${
            activeTab === 'preloaded'
              ? 'bg-accent-purple/10 text-accent-purple'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          Pre-loaded Versions
        </button>
        <button
          onClick={() => setActiveTab('upload')}
          className={`px-3 py-1.5 rounded-md text-sm transition-colors ${
            activeTab === 'upload'
              ? 'bg-accent-purple/10 text-accent-purple'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          Upload File
        </button>
      </div>

      {activeTab === 'preloaded' && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          {preloadedManifests.map((v, i) => (
            <button
              key={v.id}
              onClick={() => handleLoadPreloaded(i)}
              className={`p-4 rounded-lg border text-left transition-colors ${
                selectedVersionIdx === i && manifest
                  ? 'border-accent-purple/30 bg-accent-purple/5'
                  : 'border-surface-600 bg-surface-800 hover:bg-surface-700'
              }`}
            >
              <div className="text-sm font-mono font-semibold text-gray-200">{v.id}</div>
              <div className="text-xs text-gray-500 mt-1">
                {v.manifest?.totalVectors || 0} vectors
              </div>
              <div className="text-xs text-gray-500">
                {formatBytes(v.manifest?.fileSizeBytes || 0)}
              </div>
            </button>
          ))}
          {preloadedManifests.length === 0 && (
            <div className="col-span-4 text-center py-8 text-gray-500 text-sm">
              No pre-loaded manifest data available.
              <br />
              Place .rvf.manifest.json files in public/data/ to enable.
            </div>
          )}
        </div>
      )}

      {activeTab === 'upload' && (
        <div
          className="mb-6 border-2 border-dashed border-surface-600 rounded-lg p-12 text-center hover:border-accent-purple/30 transition-colors cursor-pointer"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleFileDrop}
          onClick={() => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.rvf,.json';
            input.onchange = (e) => {
              const file = (e.target as HTMLInputElement).files?.[0];
              if (file) handleFile(file);
            };
            input.click();
          }}
        >
          <div className="text-3xl opacity-20 mb-3">RVF</div>
          <p className="text-sm text-gray-500">
            Drop an .rvf or .rvf.manifest.json file here, or click to browse
          </p>
        </div>
      )}

      {/* Manifest display */}
      {manifest && (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <h3 className="text-sm font-mono text-accent-purple">{fileName}</h3>
            <span className="text-xs px-2 py-0.5 bg-accent-purple/10 rounded text-accent-purple">
              {manifest.format}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <MetricsCard
              title="Container"
              items={[
                { label: 'File Size', value: formatBytes(manifest.fileSizeBytes), color: '#b388ff' },
                { label: 'Format Version', value: manifest.version, color: '#00e5ff' },
              ]}
            />
            <MetricsCard
              title="Vectors"
              items={[
                { label: 'Total Vectors', value: manifest.totalVectors, color: '#b388ff' },
                { label: 'Dimensions', value: manifest.dimensions, color: '#00e5ff' },
                { label: 'Metric', value: manifest.metric, color: '#69f0ae' },
              ]}
            />
            <MetricsCard
              title="Structure"
              items={[
                { label: 'Segments', value: manifest.totalSegments, color: '#ff80ab' },
                { label: 'Epoch', value: manifest.epoch, color: '#ffd740' },
                { label: 'File ID', value: manifest.fileId.slice(0, 8) + '...', color: '#90a4ae' },
              ]}
            />
          </div>

          {/* Segment visualization */}
          {manifest.segments.length > 0 && (
            <div className="rounded-lg border border-surface-600 bg-surface-800 p-4">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                Segment Map ({manifest.segments.length} segments)
              </h3>

              {/* Visual bar */}
              <div className="flex h-8 rounded overflow-hidden mb-4">
                {manifest.segments.map((seg) => {
                  const pct = Math.max(
                    0.5,
                    (seg.payloadLength / manifest.fileSizeBytes) * 100,
                  );
                  return (
                    <div
                      key={seg.id}
                      className="relative group"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: getSegmentTypeColor(seg.type),
                        opacity: 0.6,
                      }}
                      title={`${seg.type} #${seg.id}: ${formatBytes(seg.payloadLength)}`}
                    >
                      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 bg-white/10 transition-opacity" />
                    </div>
                  );
                })}
              </div>

              {/* Legend */}
              <div className="flex gap-4 mb-4">
                {Array.from(new Set(manifest.segments.map((s) => s.type))).map((type) => (
                  <div key={type} className="flex items-center gap-1.5 text-xs text-gray-400">
                    <div
                      className="w-3 h-3 rounded-sm"
                      style={{ backgroundColor: getSegmentTypeColor(type), opacity: 0.6 }}
                    />
                    {type}
                  </div>
                ))}
              </div>

              {/* Segment table */}
              <div className="overflow-auto max-h-64">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-gray-500 border-b border-surface-600">
                      <th className="text-left py-1.5 px-2">ID</th>
                      <th className="text-left py-1.5 px-2">Type</th>
                      <th className="text-right py-1.5 px-2">Offset</th>
                      <th className="text-right py-1.5 px-2">Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {manifest.segments.map((seg) => (
                      <tr
                        key={seg.id}
                        className="border-b border-surface-700 hover:bg-surface-700"
                      >
                        <td className="py-1.5 px-2 text-gray-400">{seg.id}</td>
                        <td className="py-1.5 px-2">
                          <span style={{ color: getSegmentTypeColor(seg.type) }}>{seg.type}</span>
                        </td>
                        <td className="py-1.5 px-2 text-right text-gray-500">
                          0x{seg.offset.toString(16)}
                        </td>
                        <td className="py-1.5 px-2 text-right text-gray-400">
                          {formatBytes(seg.payloadLength)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
