<p align="center">
  <strong>ruDevolution Dashboard</strong> — <em>Visual Decompiler Explorer</em>
</p>

<p align="center">
  🔍 Browse Decompiled Code &bull;
  📦 Decompile Any npm Package &bull;
  📊 RVF Container Viewer &bull;
  ⬇️ Download Results
</p>

<p align="center">
  <img alt="Vite" src="https://img.shields.io/badge/vite-6.x-646CFF?style=flat-square&logo=vite" />
  <img alt="React" src="https://img.shields.io/badge/react-19-61DAFB?style=flat-square&logo=react" />
  <img alt="TypeScript" src="https://img.shields.io/badge/typescript-5.x-3178C6?style=flat-square&logo=typescript" />
  <img alt="Tailwind" src="https://img.shields.io/badge/tailwind-3.x-06B6D4?style=flat-square&logo=tailwindcss" />
</p>

---

## 🚀 Quick Start

```bash
cd examples/decompiler-dashboard
npm install
npm run dev
# Open http://localhost:5173
```

Production build:
```bash
npm run build    # → dist/ (434KB, 133KB gzipped)
npm run preview  # Preview production build
```

---

## 📖 What It Does

The ruDevolution Dashboard is a fully client-side web app for exploring decompiled JavaScript source code. No server needed — everything runs in the browser.

### Three Pages

#### 🔍 Explorer — Browse Claude Code Versions

Browse decompiled Claude Code across all major releases (v0.2 → v2.1):

- **Version tree** — select any version from the sidebar
- **Module list** — see all extracted modules with metrics (lines, classes, functions)
- **Source viewer** — syntax-highlighted decompiled code with line numbers
- **Side-by-side diff** — compare the same module across two versions
- **Global search** — search across all versions and modules

Pre-loaded with data from `docs/research/claude-code-rvsource/versions/`:
- v0.2.x (300 vectors, 6.9MB bundle)
- v1.0.x (482 vectors, 8.9MB bundle)
- v2.0.x (785 vectors, 10.5MB bundle)
- v2.1.x (2,068 vectors, 12.6MB bundle)

#### 📦 Decompiler — Decompile Any npm Package

Paste any npm package name and decompile it in your browser:

1. Type a package name (e.g., `express`, `lodash`, `@anthropic-ai/claude-code`)
2. Select a version from the dropdown (fetched from npm registry)
3. Click **Decompile** — the app fetches, beautifies, and splits into modules
4. Browse the results with syntax highlighting
5. Download: original, beautified, module split (zip), or metrics JSON

**How it works** (fully client-side):
- Package info from `registry.npmjs.org` (CORS-enabled)
- File listing from `data.jsdelivr.com`
- File contents from `unpkg.com`
- Beautification via `js-beautify` (in-browser)
- Module splitting via ported `module-splitter.ts`

#### 📊 RVF Viewer — Inspect RVF Containers

View RVF (RuVector Format) binary containers:

- **Pre-loaded** manifests from the Claude Code RVF corpus
- **Drag-and-drop** any `.rvf` file to inspect it
- **Segment visualization** — color-coded bar chart showing segment types and sizes
- **Segment table** — detailed breakdown (type, offset, payload length)
- **Manifest details** — dimensions, vector count, metric, file ID

---

## 🏗️ Architecture

```
src/
├── main.tsx                    # React entry point
├── App.tsx                     # Router, data loader, toast system
├── index.css                   # Tailwind + dark theme + glow utilities
│
├── pages/
│   ├── Explorer.tsx            # Version browser with diff comparison
│   ├── Decompiler.tsx          # npm package decompiler
│   └── RvfViewer.tsx           # RVF container inspector
│
├── components/
│   ├── CodeViewer.tsx          # highlight.js syntax highlighting
│   ├── DiffViewer.tsx          # Side-by-side diff (uses `diff` lib)
│   ├── ModuleTree.tsx          # File tree with badges
│   ├── MetricsCard.tsx         # Stats cards (functions, classes, size)
│   ├── SearchBar.tsx           # Full-text search across versions
│   ├── VersionSelector.tsx     # Dropdown version picker
│   └── DownloadMenu.tsx        # Download options (zip, json, js)
│
├── lib/
│   ├── decompiler.ts           # Orchestrates fetch → beautify → split
│   ├── npm-fetch.ts            # npm registry + unpkg + jsdelivr
│   ├── module-splitter.ts      # Browser port of scripts/lib/module-splitter.mjs
│   ├── beautifier.ts           # js-beautify wrapper
│   └── rvf-parser.ts           # Parse RVF manifests + binary headers
│
├── types/
│   └── index.ts                # TypeScript interfaces
│
public/
└── data/                       # Pre-built version data
    ├── v0.2.x/                 # metrics.json, source/*.js, README.md
    ├── v1.0.x/
    ├── v2.0.x/
    └── v2.1.x/
```

---

## 🎨 Design

- **Dark theme** — `#0a0a0f` background with cyan (`#06b6d4`) and purple (`#a855f7`) accents
- **Monospace code** — JetBrains Mono / system monospace
- **Responsive** — works on desktop and tablet
- **Matches pi.ruv.io aesthetic** — glow effects, clean typography

---

## 🔗 Integration with ruDevolution

This dashboard is the visual frontend for the `ruvector-decompiler` Rust crate:

| Dashboard Feature | Powered By |
|-------------------|-----------|
| Module splitting | `scripts/lib/module-splitter.mjs` (ported to TS) |
| Version data | `docs/research/claude-code-rvsource/versions/` |
| RVF parsing | `crates/rvf/` manifest format |
| Decompilation | `crates/ruvector-decompiler/` (Rust, for CLI/API) |
| Neural inference | `model/deobfuscator.onnx` (future: ONNX.js in browser) |

### Connecting to the Rust Decompiler

For deeper analysis (MinCut partitioning, witness chains, neural inference), use the Rust CLI:

```bash
# Decompile with full pipeline (MinCut + AI + witness chains)
cargo run --release -p ruvector-decompiler --example run_on_cli -- bundle.min.js

# Then view results in the dashboard
cp -r output/ examples/decompiler-dashboard/public/data/custom/
npm run dev
```

### Future: WASM Integration

The dashboard will eventually run `ruvector-decompiler` compiled to WASM for full in-browser decompilation with MinCut partitioning and witness chains — no server needed.

---

## 📚 Related

| Resource | Description |
|----------|-------------|
| [`crates/ruvector-decompiler/`](../../crates/ruvector-decompiler/) | Rust decompiler crate (MinCut + AI + witness) |
| [`crates/ruvector-decompiler/README.md`](../../crates/ruvector-decompiler/README.md) | ruDevolution full documentation |
| [`docs/research/claude-code-rvsource/`](../../docs/research/claude-code-rvsource/) | 21 research documents |
| [`scripts/claude-code-decompile.sh`](../../scripts/claude-code-decompile.sh) | CLI decompiler script |
| [`scripts/claude-code-rvf-corpus.sh`](../../scripts/claude-code-rvf-corpus.sh) | RVF corpus builder |
| [ADR-135](../../docs/adr/ADR-135-mincut-decompiler-with-witness-chains.md) | Decompiler architecture |
| [ADR-136](../../docs/adr/ADR-136-gpu-trained-deobfuscation-model.md) | GPU training pipeline |
