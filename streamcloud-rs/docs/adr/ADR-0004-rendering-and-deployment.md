# ADR-0004: Rendering (wgpu native + WebGPU/WASM) & GitHub Pages Deploy

- **Status**: Accepted
- **Date**: 2026-06-27

## Context

The deliverable includes a demo UI that runs the streaming reconstruction and
visualizes the resulting 3D point cloud, available both as a native app and in
the browser, deployed to GitHub Pages.

## Decision

- **Single renderer, two targets** via [`wgpu`]. `wgpu` targets native (Vulkan/
  Metal/DX12) and the browser (WebGPU) from one codebase, so the point-cloud
  renderer is written once.
  - Native: `streamcloud-cli` opens a `winit` window (interactive) or runs headless,
    rendering frames to an offscreen texture for PNG/MP4 export.
  - Web: `streamcloud-wasm` compiles to `wasm32-unknown-unknown`, exposes
    wasm-bindgen entry points, and drives a `<canvas>` WebGPU context.
- **Static site** in `demo/`: HTML/JS shell + the wasm bundle + a small set of
  bundled synthetic frames so the page is interactive without the 4.63 GB model.
- **Deploy** via `.github/workflows/streamcloud-pages.yml`:
  - Triggers: `workflow_dispatch` + push filtered to `streamcloud-rs/**` (so it
    never runs on unrelated monorepo changes).
  - Builds wasm with `wasm-pack`, assembles `demo/`, publishes with
    `actions/deploy-pages`.

## Rationale

- One renderer (wgpu) avoids maintaining separate native and web graphics paths.
- A path-filtered, dispatchable workflow coexists with the monorepo's existing
  CI without firing on every push.
- Bundling synthetic frames keeps the public demo self-contained and fast.

## Consequences

- WebGPU requires a recent browser; the page detects absence and shows a notice.
- The native viewer's interactive mode needs a display; headless export does
  not, so CI/tests use the headless path.
- GitHub Pages must be enabled for the repo (source: GitHub Actions) for the
  deploy job to publish.
