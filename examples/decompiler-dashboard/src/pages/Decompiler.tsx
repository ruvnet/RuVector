import { useState, useCallback } from 'react';
import type { DecompileResult, NpmPackageInfo } from '../types';
import { fetchPackageInfo } from '../lib/npm-fetch';
import { decompilePackage } from '../lib/decompiler';
import { CodeViewer } from '../components/CodeViewer';
import { ModuleTree } from '../components/ModuleTree';
import { VersionMetricsCards } from '../components/MetricsCard';
import { DownloadMenu } from '../components/DownloadMenu';

interface DecompilerProps {
  showToast: (msg: string) => void;
}

export function Decompiler({ showToast }: DecompilerProps) {
  const [packageName, setPackageName] = useState('');
  const [packageInfo, setPackageInfo] = useState<NpmPackageInfo | null>(null);
  const [selectedVersion, setSelectedVersion] = useState('');
  const [result, setResult] = useState<DecompileResult | null>(null);
  const [selectedModule, setSelectedModule] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ message: '', pct: 0 });
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'modules' | 'beautified' | 'original'>('modules');

  const handleLookup = useCallback(async () => {
    if (!packageName.trim()) return;
    setError(null);
    setLoading(true);
    try {
      const info = await fetchPackageInfo(packageName.trim());
      setPackageInfo(info);
      setSelectedVersion(info.latest);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch package info');
    } finally {
      setLoading(false);
    }
  }, [packageName]);

  const handleDecompile = useCallback(async () => {
    if (!packageName.trim()) return;
    setError(null);
    setLoading(true);
    setResult(null);

    try {
      const res = await decompilePackage(
        packageName.trim(),
        selectedVersion,
        (msg, pct) => setProgress({ message: msg, pct }),
      );
      setResult(res);
      showToast(`Decompiled ${res.packageName}@${res.version}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Decompilation failed');
    } finally {
      setLoading(false);
    }
  }, [packageName, selectedVersion, showToast]);

  const moduleSource = selectedModule && result ? result.modules[selectedModule] : null;

  return (
    <div className="max-w-[1600px] mx-auto p-4">
      {/* Input section */}
      <div className="rounded-lg border border-surface-600 bg-surface-800 p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-100 mb-4">
          NPM Package <span className="text-accent-cyan">Decompiler</span>
        </h2>
        <div className="flex gap-3">
          <input
            type="text"
            value={packageName}
            onChange={(e) => setPackageName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleLookup()}
            placeholder="Package name (e.g., @anthropic-ai/claude-code, express, react)"
            className="flex-1 bg-surface-700 border border-surface-600 rounded-lg px-4 py-2.5 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-accent-cyan/50 focus:ring-1 focus:ring-accent-cyan/30 font-mono"
          />
          <button
            onClick={handleLookup}
            disabled={loading || !packageName.trim()}
            className="px-4 py-2.5 bg-surface-700 border border-surface-600 rounded-lg text-sm text-gray-300 hover:bg-surface-600 hover:text-accent-cyan transition-colors disabled:opacity-50"
          >
            Lookup
          </button>
        </div>

        {/* Package info + version selector */}
        {packageInfo && (
          <div className="mt-4 flex items-end gap-4">
            <div className="flex-1">
              <p className="text-sm text-gray-400 mb-2">{packageInfo.description}</p>
              <div className="flex items-center gap-3">
                <select
                  value={selectedVersion}
                  onChange={(e) => setSelectedVersion(e.target.value)}
                  className="bg-surface-700 border border-surface-600 rounded-md px-3 py-1.5 text-sm text-gray-200 font-mono focus:outline-none focus:border-accent-cyan/50"
                >
                  {packageInfo.versions.slice(0, 50).map((v) => (
                    <option key={v} value={v}>
                      {v}
                      {packageInfo.distTags.latest === v ? ' (latest)' : ''}
                    </option>
                  ))}
                </select>
                <span className="text-xs text-gray-500">
                  {packageInfo.versions.length} versions available
                </span>
              </div>
            </div>
            <button
              onClick={handleDecompile}
              disabled={loading}
              className="px-6 py-2.5 bg-accent-cyan/10 border border-accent-cyan/30 rounded-lg text-sm text-accent-cyan hover:bg-accent-cyan/20 transition-colors disabled:opacity-50 font-semibold"
            >
              {loading ? 'Working...' : 'Decompile'}
            </button>
          </div>
        )}

        {/* Progress */}
        {loading && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
              <span>{progress.message}</span>
              <span>{progress.pct}%</span>
            </div>
            <div className="h-1.5 bg-surface-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-accent-cyan rounded-full transition-all duration-300"
                style={{ width: `${progress.pct}%` }}
              />
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 px-4 py-2 bg-red-900/20 border border-red-500/30 rounded-lg text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          <VersionMetricsCards metrics={result.metrics} />

          <div className="flex items-center justify-between mt-4 mb-4">
            <div className="flex gap-1">
              {(['modules', 'beautified', 'original'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-1.5 rounded-md text-sm capitalize transition-colors ${
                    viewMode === mode
                      ? 'bg-accent-cyan/10 text-accent-cyan'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
            <DownloadMenu
              modules={result.modules}
              originalSource={result.originalSource}
              metricsJson={JSON.stringify(result.metrics, null, 2)}
              packageLabel={`${result.packageName}-${result.version}`}
            />
          </div>

          {viewMode === 'modules' ? (
            <div className="flex gap-4">
              <div className="w-64 flex-shrink-0">
                <ModuleTree
                  modules={result.metrics.modules}
                  selectedModule={selectedModule}
                  onSelectModule={setSelectedModule}
                />
              </div>
              <div className="flex-1 min-w-0">
                {moduleSource ? (
                  <CodeViewer code={moduleSource} />
                ) : (
                  <div className="flex items-center justify-center h-64 text-gray-500 border border-surface-600 rounded-lg bg-surface-800">
                    Select a module to view
                  </div>
                )}
              </div>
            </div>
          ) : (
            <CodeViewer
              code={viewMode === 'beautified' ? result.beautifiedSource : result.originalSource}
              maxHeight="800px"
            />
          )}
        </>
      )}
    </div>
  );
}
