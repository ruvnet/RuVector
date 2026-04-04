import { useState, useCallback } from 'react';
import type { VersionData, SearchResult } from '../types';
import { ModuleTree } from '../components/ModuleTree';
import { CodeViewer } from '../components/CodeViewer';
import { DiffViewer } from '../components/DiffViewer';
import { VersionMetricsCards } from '../components/MetricsCard';
import { SearchBar, SearchResults } from '../components/SearchBar';
import { VersionSelector } from '../components/VersionSelector';
import { DownloadMenu } from '../components/DownloadMenu';

interface ExplorerProps {
  versions: VersionData[];
  showToast: (msg: string) => void;
}

export function Explorer({ versions, showToast }: ExplorerProps) {
  const [selectedVersionIdx, setSelectedVersionIdx] = useState(versions.length - 1);
  const [selectedModule, setSelectedModule] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [compareMode, setCompareMode] = useState(false);
  const [compareVersionIdx, setCompareVersionIdx] = useState(0);

  const version = versions[selectedVersionIdx];
  const compareVersion = versions[compareVersionIdx];

  if (!version) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-500">
        No version data loaded. Place research data in public/data/.
      </div>
    );
  }

  const handleSearchNavigate = useCallback(
    (versionId: string, moduleName: string) => {
      const idx = versions.findIndex((v) => v.id === versionId);
      if (idx >= 0) {
        setSelectedVersionIdx(idx);
        setSelectedModule(moduleName);
        setSearchResults([]);
      }
    },
    [versions],
  );

  const moduleSource = selectedModule ? version.modules[selectedModule] : null;
  const compareModuleSource =
    compareMode && selectedModule ? compareVersion?.modules[selectedModule] : null;

  return (
    <div className="max-w-[1600px] mx-auto p-4">
      {/* Search */}
      <div className="mb-4">
        <SearchBar versions={versions} onResults={setSearchResults} />
        <SearchResults results={searchResults} onNavigate={handleSearchNavigate} />
      </div>

      {/* Metrics */}
      <VersionMetricsCards metrics={version.metrics} />

      {/* Controls bar */}
      <div className="flex items-center justify-between mt-4 mb-4">
        <div className="flex items-center gap-4">
          <VersionSelector
            versions={versions.map((v) => v.id)}
            selected={version.id}
            onChange={(id) => {
              const idx = versions.findIndex((v) => v.id === id);
              if (idx >= 0) setSelectedVersionIdx(idx);
            }}
            label="Version"
          />
          <button
            onClick={() => setCompareMode(!compareMode)}
            className={`px-3 py-1.5 rounded-md text-sm border transition-colors ${
              compareMode
                ? 'bg-accent-purple/10 border-accent-purple/30 text-accent-purple'
                : 'border-surface-600 text-gray-400 hover:text-gray-200'
            }`}
          >
            Compare
          </button>
          {compareMode && (
            <VersionSelector
              versions={versions.map((v) => v.id)}
              selected={compareVersion?.id || versions[0].id}
              onChange={(id) => {
                const idx = versions.findIndex((v) => v.id === id);
                if (idx >= 0) setCompareVersionIdx(idx);
              }}
              label="vs"
            />
          )}
        </div>
        {selectedModule && (
          <DownloadMenu
            modules={version.modules}
            metricsJson={JSON.stringify(version.metrics, null, 2)}
            packageLabel={`claude-code-${version.id}`}
          />
        )}
      </div>

      {/* Main layout: sidebar + code viewer */}
      <div className="flex gap-4">
        {/* Sidebar */}
        <div className="w-64 flex-shrink-0">
          <div className="sticky top-20">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 px-3">
              Modules ({Object.keys(version.metrics.modules).length})
            </h2>
            <ModuleTree
              modules={version.metrics.modules}
              selectedModule={selectedModule}
              onSelectModule={setSelectedModule}
            />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {selectedModule && moduleSource ? (
            <>
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-mono text-gray-300">
                  <span className="text-accent-cyan">{version.id}</span>
                  <span className="text-gray-600 mx-1">/</span>
                  {selectedModule}.js
                </h2>
                <span className="text-xs text-gray-500">
                  {moduleSource.split('\n').length} lines
                </span>
              </div>
              {compareMode && compareModuleSource ? (
                <DiffViewer
                  leftCode={compareModuleSource}
                  rightCode={moduleSource}
                  leftLabel={`${compareVersion.id}/${selectedModule}.js`}
                  rightLabel={`${version.id}/${selectedModule}.js`}
                />
              ) : (
                <CodeViewer code={moduleSource} />
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-96 text-gray-500 border border-surface-600 rounded-lg bg-surface-800">
              <div className="text-center">
                <div className="text-4xl mb-3 opacity-20">{ '<>' }</div>
                <p className="text-sm">Select a module to view its source</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
