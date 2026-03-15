# ADR-040a: Causal Atlas Dashboard Specification

**Status:** Proposed
**Date:** 2026-02-18
**Parent:** ADR-040
**Related:** ADR-008 (Chat UI RVF)

## Context

This sub-ADR extracts the dashboard specification from ADR-040 to keep
individual files under 500 lines per project guidelines. It covers the
Three.js visualization dashboard architecture, all five dashboard views
(V1-V5), the WebSocket live stream protocol, Vite build configuration,
and the design decision for embedding the dashboard in the RVF artifact.

The REST and CLI API definitions remain in the main ADR-040 as they are
part of the core runtime interface.

## Dashboard Architecture

The RVF embeds a Vite-bundled Three.js dashboard in `DASHBOARD_SEG`. The
runtime HTTP server serves it at `/` (root). All visualizations are driven
by the same API endpoints the CLI uses, so every rendered frame corresponds
to queryable, witness-backed data.

```
DASHBOARD_SEG (inside RVF)
  dist/
    index.html            # Vite SPA entry
    assets/
      main.[hash].js      # Three.js + D3 + app logic (tree-shaken)
      main.[hash].css     # Tailwind/minimal styles
      worker.js           # Web Worker for graph layout

Runtime serves:
  GET /                   -> DASHBOARD_SEG/dist/index.html
  GET /assets/*           -> DASHBOARD_SEG/dist/assets/*
  GET /api/*              -> JSON API (atlas, coherence, candidates, etc.)
  WS  /ws/live            -> Live streaming of boundary alerts and pipeline progress
```

**Build pipeline**: Vite builds the dashboard at package time into a single
tree-shaken bundle. The bundle is embedded into `DASHBOARD_SEG` during RVF
assembly. No Node.js required at runtime — the dashboard is pure static
assets served by the existing HTTP server.

## Dashboard Views

### V1: Causal Atlas Explorer (Three.js 3D)

Interactive 3D force-directed graph of the causal atlas.

| Feature | Implementation |
|---------|---------------|
| **Node rendering** | `THREE.InstancedMesh` for events — color by domain (Kepler=blue, TESS=cyan, JWST=gold, derived=white) |
| **Edge rendering** | `THREE.LineSegments` with opacity mapped to edge weight |
| **Causal flow** | Animated particles along causal edges showing temporal direction |
| **Scale selector** | Toggle between window scales (2h, 12h, 3d, 27d) — re-layouts graph |
| **Candidate highlight** | Click candidate in sidebar to trace its causal chain in 3D, dimming unrelated nodes |
| **Witness replay** | Step through witness chain entries, animating graph state forward/backward |
| **LOD** | Level-of-detail: far=boundary nodes only, mid=top-k events, close=full subgraph |

Data source: `GET /api/atlas/query`, `GET /api/atlas/trace`

### V2: Coherence Field Heatmap (Three.js + shader)

Real-time coherence field rendered as a colored surface over the atlas graph.

| Feature | Implementation |
|---------|---------------|
| **Field surface** | `THREE.PlaneGeometry` subdivided grid, vertex colors from coherence values |
| **Cut pressure** | Red hotspots where cut pressure is high, cool blue where stable |
| **Partition boundaries** | Glowing wireframe lines at partition cuts |
| **Time scrubber** | Scrub through epochs to see coherence evolution |
| **Drift overlay** | Toggle to show embedding drift as animated vector arrows |
| **Alert markers** | Pulsing icons at boundary alert locations |

Data source: `GET /api/coherence`, `GET /api/boundary/timeline`, `WS /ws/live`

### V3: Planet Candidate Dashboard (2D panels + 3D orbit)

Split view combining data panels with 3D orbital visualization.

| Panel | Content |
|-------|---------|
| **Ranked list** | Sortable table: candidate ID, score, uncertainty, period, SNR, publishable status |
| **Light curve viewer** | Interactive D3 chart: raw flux, detrended flux, transit model overlay, per-window score |
| **Phase-folded plot** | All transits folded at detected period, with confidence band |
| **3D orbit preview** | `THREE.Line` showing inferred orbital path around host star, sized by uncertainty |
| **Evidence trace** | Expandable tree showing witness chain from raw data to final score |
| **Score breakdown** | Radar chart: SNR, shape consistency, period stability, coherence stability |

Data source: `GET /api/candidates/planet`, `GET /api/candidates/:id/trace`

### V4: Life Candidate Dashboard (2D panels + 3D molecule)

Split view for spectral disequilibrium analysis.

| Panel | Content |
|-------|---------|
| **Ranked list** | Sortable table: candidate ID, disequilibrium score, uncertainty, molecule flags, publishable |
| **Spectrum viewer** | Interactive D3 chart: wavelength vs flux, molecule absorption bands highlighted |
| **Molecule presence matrix** | Heatmap of detected molecule families vs confidence |
| **3D molecule overlay** | `THREE.Sprite` labels at absorption wavelengths in a 3D wavelength space |
| **Reaction graph** | Force-directed graph of molecule co-occurrences vs equilibrium expectations |
| **Confound panel** | Bar chart: stellar activity penalty, contamination risk, repeatability score |

Data source: `GET /api/candidates/life`, `GET /api/candidates/:id/trace`

### V5: System Status Dashboard

Operational health and download progress.

| Panel | Content |
|-------|---------|
| **Download progress** | Per-tier progress bars with byte counts and ETA |
| **Segment sizes** | Stacked bar chart of RVF segment utilization |
| **Memory tiers** | S/M/L tier fill levels and compaction history |
| **Witness chain** | Scrolling log of recent witness entries with hash preview |
| **Pipeline status** | P0/P1/P2 and L0/L1/L2 stage indicators with event counts |
| **Performance** | Query latency histogram, events/second throughput |

Data source: `GET /api/status`, `GET /api/memory/tiers`, `WS /ws/live`

## WebSocket Live Stream

```typescript
// WS /ws/live — server pushes events as they happen
interface LiveEvent {
  type: 'boundary_alert' | 'candidate_new' | 'candidate_update' |
        'download_progress' | 'witness_commit' | 'pipeline_stage' |
        'coherence_update';
  timestamp: string;
  data: Record<string, unknown>;
}
```

The dashboard subscribes on connect and updates all views in real-time as
pipelines process data and boundaries evolve.

## Vite Build Configuration

```typescript
// vite.config.ts for dashboard build
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'dist/dashboard',
    assetsDir: 'assets',
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three'],         // ~150 KB gzipped
          d3: ['d3-scale', 'd3-axis', 'd3-shape', 'd3-selection'],
        },
      },
    },
  },
});
```

**Bundle budget**: < 500 KB gzipped total (Three.js ~150 KB, D3 subset ~30 KB,
app logic ~50 KB, styles ~10 KB). The dashboard adds minimal overhead to the
RVF artifact.

## Design Decision: D5 — Dashboard Embedded in RVF

The Three.js dashboard is bundled at build time and embedded in `DASHBOARD_SEG`
rather than served from an external CDN or requiring a separate install. This
ensures:

1. **Fully offline**: Works without network after boot
2. **Version-locked**: Dashboard always matches the API version it queries
3. **Single artifact**: One RVF file = runtime + data + visualization
4. **Witness-aligned**: Dashboard renders exactly the data the witness chain
   can verify

## Dashboard Package Structure

```
packages/agentdb-causal-atlas/
  dashboard/                    # Vite + Three.js visualization app
    vite.config.ts              # Build config — outputs to dist/dashboard/
    index.html                  # SPA entry point
    src/
      main.ts                   # App bootstrap, router, WS connection
      api.ts                    # Typed fetch wrappers for /api/* endpoints
      ws.ts                     # WebSocket client for /ws/live
      views/
        AtlasExplorer.ts        # V1: 3D causal atlas (Three.js force graph)
        CoherenceHeatmap.ts     # V2: Coherence field surface + cut pressure
        PlanetDashboard.ts      # V3: Planet candidates + light curves + 3D orbit
        LifeDashboard.ts        # V4: Life candidates + spectra + molecule graph
        StatusDashboard.ts      # V5: System health, downloads, witness log
      three/
        AtlasGraph.ts           # InstancedMesh nodes, LineSegments edges, particles
        CoherenceSurface.ts     # PlaneGeometry with vertex-colored field
        OrbitPreview.ts         # Orbital path visualization
        CausalFlow.ts           # Animated particles along causal edges
        LODController.ts        # Level-of-detail: boundary -> top-k -> full
      charts/
        LightCurveChart.ts      # D3 flux time series with transit overlay
        SpectrumChart.ts        # D3 wavelength vs flux with molecule bands
        RadarChart.ts           # Score breakdown radar
        MoleculeMatrix.ts       # Heatmap of molecule presence vs confidence
      components/
        Sidebar.ts              # Candidate list, filters, search
        TimeScrubber.ts         # Epoch scrubber for coherence replay
        WitnessLog.ts           # Scrolling witness chain entries
        DownloadProgress.ts     # Tier progress bars
      styles/
        main.css                # Minimal Tailwind or hand-rolled styles
  tests/
    dashboard.test.ts           # Dashboard build + API integration tests
```

## References

1. ADR-040: Causal Atlas RVF Runtime — Planet Detection & Life Candidate Scoring
2. ADR-008: Chat UI RVF Kernel Embedding
3. [Three.js Documentation](https://threejs.org/docs/)
4. [Vite Build Tool](https://vitejs.dev/)
5. [D3.js](https://d3js.org/)
