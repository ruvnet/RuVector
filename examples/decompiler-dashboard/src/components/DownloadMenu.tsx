import { useState } from 'react';
import JSZip from 'jszip';
import { beautifyJS } from '../lib/beautifier';

interface DownloadMenuProps {
  modules: Record<string, string>;
  originalSource?: string;
  metricsJson?: string;
  packageLabel: string;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadText(content: string, filename: string, mime = 'text/javascript') {
  downloadBlob(new Blob([content], { type: mime }), filename);
}

export function DownloadMenu({ modules, originalSource, metricsJson, packageLabel }: DownloadMenuProps) {
  const [open, setOpen] = useState(false);
  const [generating, setGenerating] = useState(false);

  const safeLabel = packageLabel.replace(/[^a-zA-Z0-9._-]/g, '_');

  async function downloadModuleZip() {
    setGenerating(true);
    try {
      const zip = new JSZip();
      const folder = zip.folder(safeLabel) || zip;
      for (const [name, code] of Object.entries(modules)) {
        folder.file(`${name}.js`, code);
      }
      const blob = await zip.generateAsync({ type: 'blob' });
      downloadBlob(blob, `${safeLabel}-modules.zip`);
    } finally {
      setGenerating(false);
      setOpen(false);
    }
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 bg-surface-700 border border-surface-600 rounded-md text-sm text-gray-300 hover:bg-surface-600 hover:text-accent-cyan transition-colors"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        Download
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-56 bg-surface-700 border border-surface-600 rounded-lg shadow-xl z-50">
          {originalSource && (
            <button
              onClick={() => { downloadText(originalSource, `${safeLabel}-original.js`); setOpen(false); }}
              className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-surface-600 hover:text-accent-cyan"
            >
              Original bundle (.js)
            </button>
          )}
          {originalSource && (
            <button
              onClick={() => { downloadText(beautifyJS(originalSource), `${safeLabel}-beautified.js`); setOpen(false); }}
              className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-surface-600 hover:text-accent-cyan"
            >
              Beautified (.js)
            </button>
          )}
          <button
            onClick={downloadModuleZip}
            disabled={generating}
            className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-surface-600 hover:text-accent-cyan disabled:opacity-50"
          >
            {generating ? 'Generating...' : 'Module split (.zip)'}
          </button>
          {metricsJson && (
            <button
              onClick={() => { downloadText(metricsJson, `${safeLabel}-metrics.json`, 'application/json'); setOpen(false); }}
              className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-surface-600 hover:text-accent-cyan"
            >
              Metrics (.json)
            </button>
          )}
        </div>
      )}

      {open && (
        <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
      )}
    </div>
  );
}
