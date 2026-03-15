export interface TierStatus {
  tier: number;
  label: string;
  totalBytes: number;
  downloadedBytes: number;
  status: 'pending' | 'downloading' | 'complete' | 'error';
  eta?: string;
}

const STATUS_COLORS: Record<string, string> = {
  pending: '#484F58',
  downloading: '#00E5FF',
  complete: '#2ECC71',
  error: '#FF4D4D',
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export class DownloadProgress {
  private root: HTMLElement;
  private listEl: HTMLElement;

  constructor(container: HTMLElement) {
    this.root = document.createElement('div');
    this.root.className = 'chart-container';

    const header = document.createElement('div');
    header.className = 'panel-header';
    header.textContent = 'Progressive Download';
    this.root.appendChild(header);

    this.listEl = document.createElement('div');
    this.listEl.style.display = 'flex';
    this.listEl.style.flexDirection = 'column';
    this.listEl.style.gap = '10px';
    this.listEl.style.padding = '8px';
    this.root.appendChild(this.listEl);

    container.appendChild(this.root);
  }

  update(tiers: TierStatus[]): void {
    this.listEl.innerHTML = '';

    for (const tier of tiers) {
      const pct = tier.totalBytes > 0
        ? Math.min(100, (tier.downloadedBytes / tier.totalBytes) * 100)
        : 0;
      const color = STATUS_COLORS[tier.status] ?? STATUS_COLORS.pending;

      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.flexDirection = 'column';
      row.style.gap = '4px';

      // Label row: tier label + status badge + ETA
      const labelRow = document.createElement('div');
      labelRow.style.display = 'flex';
      labelRow.style.justifyContent = 'space-between';
      labelRow.style.alignItems = 'center';

      const label = document.createElement('span');
      label.className = 'progress-label';
      label.textContent = `T${tier.tier}: ${tier.label}`;
      label.style.color = '#C9D1D9';
      label.style.fontSize = '12px';
      labelRow.appendChild(label);

      const rightInfo = document.createElement('span');
      rightInfo.style.fontSize = '11px';
      rightInfo.style.fontFamily = 'var(--font-mono)';
      const badge = `[${tier.status.toUpperCase()}]`;
      const etaText = tier.eta ? ` ETA ${tier.eta}` : '';
      rightInfo.textContent = `${formatBytes(tier.downloadedBytes)}/${formatBytes(tier.totalBytes)} ${badge}${etaText}`;
      rightInfo.style.color = color;
      labelRow.appendChild(rightInfo);

      row.appendChild(labelRow);

      // Progress bar
      const bar = document.createElement('div');
      bar.className = 'progress-bar';
      bar.style.height = '6px';
      bar.style.background = '#161B22';
      bar.style.borderRadius = '3px';
      bar.style.overflow = 'hidden';

      const fill = document.createElement('div');
      fill.className = 'progress-fill';
      fill.style.width = `${pct}%`;
      fill.style.height = '100%';
      fill.style.background = color;
      fill.style.borderRadius = '3px';
      fill.style.transition = 'width 0.3s ease';
      bar.appendChild(fill);

      row.appendChild(bar);
      this.listEl.appendChild(row);
    }
  }

  destroy(): void {
    this.root.remove();
  }
}
