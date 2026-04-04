import { useState, useCallback } from 'react';
import type { SearchResult, VersionData } from '../types';

interface SearchBarProps {
  versions: VersionData[];
  onResults: (results: SearchResult[]) => void;
}

export function SearchBar({ versions, onResults }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);

  const doSearch = useCallback(
    (q: string) => {
      if (q.length < 2) {
        onResults([]);
        return;
      }

      setSearching(true);
      const results: SearchResult[] = [];
      const lowerQ = q.toLowerCase();

      for (const version of versions) {
        for (const [moduleName, source] of Object.entries(version.modules)) {
          const lines = source.split('\n');
          for (let i = 0; i < lines.length; i++) {
            if (lines[i].toLowerCase().includes(lowerQ)) {
              const contextStart = Math.max(0, i - 1);
              const contextEnd = Math.min(lines.length, i + 2);
              results.push({
                versionId: version.id,
                moduleName,
                lineNumber: i + 1,
                lineContent: lines[i],
                context: lines.slice(contextStart, contextEnd),
              });
              if (results.length >= 100) break;
            }
          }
          if (results.length >= 100) break;
        }
        if (results.length >= 100) break;
      }

      onResults(results);
      setSearching(false);
    },
    [versions, onResults],
  );

  return (
    <div className="relative">
      <input
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          doSearch(e.target.value);
        }}
        placeholder="Search across all versions and modules..."
        className="w-full bg-surface-700 border border-surface-600 rounded-lg px-4 py-2.5 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-accent-cyan/50 focus:ring-1 focus:ring-accent-cyan/30 font-mono"
      />
      {searching && (
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          <div className="w-4 h-4 border-2 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin" />
        </div>
      )}
    </div>
  );
}

interface SearchResultsProps {
  results: SearchResult[];
  onNavigate: (versionId: string, moduleName: string) => void;
}

export function SearchResults({ results, onNavigate }: SearchResultsProps) {
  if (results.length === 0) return null;

  return (
    <div className="mt-3 rounded-lg border border-surface-600 bg-surface-800 max-h-80 overflow-auto">
      {results.slice(0, 50).map((r, i) => (
        <button
          key={i}
          onClick={() => onNavigate(r.versionId, r.moduleName)}
          className="w-full text-left px-4 py-2 border-b border-surface-700 hover:bg-surface-700 transition-colors"
        >
          <div className="flex items-center gap-2 text-xs mb-1">
            <span className="px-1.5 py-0.5 bg-accent-cyan/10 text-accent-cyan rounded text-[10px]">
              {r.versionId}
            </span>
            <span className="text-gray-500">{r.moduleName}.js</span>
            <span className="text-gray-600">L{r.lineNumber}</span>
          </div>
          <pre className="text-xs font-mono text-gray-400 truncate">{r.lineContent.trim()}</pre>
        </button>
      ))}
      {results.length > 50 && (
        <div className="px-4 py-2 text-xs text-gray-500 text-center">
          Showing 50 of {results.length} results
        </div>
      )}
    </div>
  );
}
