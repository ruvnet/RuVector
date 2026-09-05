# StreamCloud WebGPU demo

A static site that runs the StreamCloud pipeline (compiled to WASM) and
renders the streaming 3D reconstruction with **WebGPU** (2D canvas fallback when
WebGPU is unavailable).

## Build locally

```bash
# from streamcloud-rs/
cargo install wasm-pack            # once
wasm-pack build crates/streamcloud-wasm --target web --release --out-dir ../../demo/pkg
python3 -m http.server -d demo 8080
# open http://localhost:8080
```

`demo/pkg/` is generated (git-ignored). CI builds it and publishes the site via
`.github/workflows/streamcloud-pages.yml` (manual `workflow_dispatch` or on pushes
touching `streamcloud-rs/**`). Enable GitHub Pages with source = GitHub Actions.

## What it shows

Each animation frame, the WASM module advances a synthetic scene, encodes a
keyframe, retrieves the top-K structurally similar past keyframes (brute-force
cosine in the browser; lock-free HNSW via `ruvector-core` natively), reconstructs
a colored 3D point cloud, and streams it to the GPU. The orbiting camera reveals
the reconstructed depth.

> The browser uses the synthetic backend (ADR-0006): illustrative geometry, not
> a faithful reconstruction. The real 4.63 GB checkpoint loads via the native
> `candle` backend.
