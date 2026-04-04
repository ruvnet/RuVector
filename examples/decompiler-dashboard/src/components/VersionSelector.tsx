interface VersionSelectorProps {
  versions: string[];
  selected: string;
  onChange: (version: string) => void;
  label?: string;
}

export function VersionSelector({ versions, selected, onChange, label }: VersionSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      {label && <span className="text-xs text-gray-500">{label}</span>}
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        className="bg-surface-700 border border-surface-600 rounded-md px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-accent-cyan/50 font-mono"
      >
        {versions.map((v) => (
          <option key={v} value={v}>
            {v}
          </option>
        ))}
      </select>
    </div>
  );
}
