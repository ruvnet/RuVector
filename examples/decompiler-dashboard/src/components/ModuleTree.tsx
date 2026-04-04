import type { ModuleMetrics } from '../types';
import { formatBytes } from '../lib/rvf-parser';

interface ModuleTreeProps {
  modules: Record<string, ModuleMetrics>;
  selectedModule: string | null;
  onSelectModule: (name: string) => void;
}

const MODULE_ICONS: Record<string, string> = {
  'tool-dispatch': 'T',
  'permission-system': 'P',
  'mcp-client': 'M',
  'streaming-handler': 'S',
  'context-manager': 'C',
  'agent-loop': 'A',
  telemetry: 'L',
  commands: 'K',
  'class-hierarchy': 'H',
};

export function ModuleTree({ modules, selectedModule, onSelectModule }: ModuleTreeProps) {
  const entries = Object.entries(modules).sort(
    ([, a], [, b]) => b.sizeBytes - a.sizeBytes,
  );

  return (
    <div className="space-y-1">
      {entries.map(([name, metrics]) => {
        const isSelected = name === selectedModule;
        return (
          <button
            key={name}
            onClick={() => onSelectModule(name)}
            className={`w-full flex items-center gap-2 px-3 py-2 rounded-md text-left text-sm transition-colors ${
              isSelected
                ? 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30'
                : 'hover:bg-surface-600 text-gray-300'
            }`}
          >
            <span
              className={`flex-shrink-0 w-6 h-6 rounded flex items-center justify-center text-xs font-bold ${
                isSelected ? 'bg-accent-cyan/20 text-accent-cyan' : 'bg-surface-600 text-gray-500'
              }`}
            >
              {MODULE_ICONS[name] || name[0].toUpperCase()}
            </span>
            <div className="flex-1 min-w-0">
              <div className="truncate font-mono text-xs">{name}.js</div>
              <div className="text-[10px] text-gray-500">
                {metrics.fragments} frags &middot; {formatBytes(metrics.sizeBytes)}
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
