import { useMemo } from 'react';
import { diffLines } from 'diff';

interface DiffViewerProps {
  leftCode: string;
  rightCode: string;
  leftLabel: string;
  rightLabel: string;
  maxHeight?: string;
}

export function DiffViewer({
  leftCode,
  rightCode,
  leftLabel,
  rightLabel,
  maxHeight = '600px',
}: DiffViewerProps) {
  const changes = useMemo(
    () => diffLines(leftCode, rightCode),
    [leftCode, rightCode],
  );

  const stats = useMemo(() => {
    let added = 0;
    let removed = 0;
    for (const change of changes) {
      const lineCount = change.value.split('\n').filter(Boolean).length;
      if (change.added) added += lineCount;
      if (change.removed) removed += lineCount;
    }
    return { added, removed };
  }, [changes]);

  return (
    <div className="rounded-lg border border-surface-600 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 bg-surface-700 border-b border-surface-600">
        <div className="flex gap-6 text-sm">
          <span className="text-gray-400">{leftLabel}</span>
          <span className="text-gray-500">vs</span>
          <span className="text-gray-400">{rightLabel}</span>
        </div>
        <div className="flex gap-4 text-xs">
          <span className="text-green-400">+{stats.added} lines</span>
          <span className="text-red-400">-{stats.removed} lines</span>
        </div>
      </div>
      <div className="overflow-auto bg-surface-900" style={{ maxHeight }}>
        <pre className="p-3 text-xs font-mono leading-5">
          {changes.map((change, i) => {
            const lines = change.value.split('\n').filter((l, idx, arr) =>
              idx < arr.length - 1 || l.length > 0
            );
            return lines.map((line, j) => (
              <div
                key={`${i}-${j}`}
                className={
                  change.added
                    ? 'bg-green-900/20 text-green-300'
                    : change.removed
                      ? 'bg-red-900/20 text-red-300'
                      : 'text-gray-400'
                }
              >
                <span className="inline-block w-5 text-center select-none opacity-50">
                  {change.added ? '+' : change.removed ? '-' : ' '}
                </span>
                {line}
              </div>
            ));
          })}
        </pre>
      </div>
    </div>
  );
}
