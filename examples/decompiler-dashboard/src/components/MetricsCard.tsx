import { formatBytes } from '../lib/rvf-parser';

interface MetricItem {
  label: string;
  value: string | number;
  color?: string;
}

interface MetricsCardProps {
  title: string;
  items: MetricItem[];
}

export function MetricsCard({ title, items }: MetricsCardProps) {
  return (
    <div className="rounded-lg border border-surface-600 bg-surface-800 p-4">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
        {title}
      </h3>
      <div className="grid grid-cols-2 gap-3">
        {items.map((item) => (
          <div key={item.label}>
            <div
              className="text-lg font-bold font-mono"
              style={{ color: item.color || '#00e5ff' }}
            >
              {typeof item.value === 'number' ? item.value.toLocaleString() : item.value}
            </div>
            <div className="text-xs text-gray-500">{item.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function VersionMetricsCards({ metrics }: { metrics: { sizeBytes: number; lines: number; functions: number; asyncFunctions: number; arrowFunctions: number; classes: number; extends: number } }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <MetricsCard
        title="Bundle Size"
        items={[
          { label: 'Size', value: formatBytes(metrics.sizeBytes), color: '#00e5ff' },
          { label: 'Lines', value: metrics.lines, color: '#b388ff' },
        ]}
      />
      <MetricsCard
        title="Functions"
        items={[
          { label: 'Total', value: metrics.functions, color: '#00e5ff' },
          { label: 'Async', value: metrics.asyncFunctions, color: '#69f0ae' },
          { label: 'Arrow', value: metrics.arrowFunctions, color: '#ffd740' },
        ]}
      />
      <MetricsCard
        title="Classes"
        items={[
          { label: 'Classes', value: metrics.classes, color: '#ff80ab' },
          { label: 'Extends', value: metrics.extends, color: '#b388ff' },
        ]}
      />
    </div>
  );
}
