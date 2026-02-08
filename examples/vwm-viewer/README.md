# RuVector VWM Viewer

## What is this?

A WebGPU-based 4D Gaussian splatting viewer that demonstrates the Visual World Model architecture. Opens in any browser with WebGPU support (Chrome 113+, Edge 113+, Firefox Nightly).

## Quick Start

```bash
cd examples/vwm-viewer
npx serve .
# Open http://localhost:3000 in Chrome
```

## How it Works

### Demo Mode (no WASM required)

The viewer runs immediately with synthetic demo data:
- Orbiting colored Gaussians demonstrating temporal motion
- Static background field showing spatial tiling
- Time scrubber for 4D playback

### With WASM Module

```bash
# Build the WASM module
cd crates/ruvector-vwm-wasm
wasm-pack build --target web --out-dir ../../examples/vwm-viewer/pkg

# Then serve
cd examples/vwm-viewer
npx serve .
```

## Controls

- **Left drag**: Orbit camera
- **Scroll**: Zoom in/out
- **Time slider**: Scrub through time
- **Search box**: Query entities (demo mode: filters by color name)

## Architecture

```
Camera Pose → [CPU] Project & Sort → [GPU] Splat Render
                     ↑
              Active Mask Filter
                     ↑
            Time Slider → Select active Gaussians
```

### Files

- `src/main.js` — Application entry, WebGPU init, render loop
- `src/renderer.js` — WGSL shaders, GPU pipeline, instanced quad rendering
- `src/camera.js` — Orbit camera with mouse controls
- `src/demo-data.js` — Synthetic Gaussian generator
- `src/ui.js` — Time slider, FPS counter, search box

## WebGPU Pipeline

### Vertex Shader

Each Gaussian is rendered as a screen-aligned quad (2 triangles, 6 vertices per instance). The quad is scaled by the Gaussian's screen-space radius.

### Fragment Shader

Evaluates the 2D Gaussian kernel: `G = exp(-0.5 * (a*dx² + 2*b*dx*dy + c*dy²))` where `[a,b,c]` is the inverse covariance (conic) matrix. Applies alpha blending with pre-multiplied alpha.

### Sorting

Gaussians are sorted back-to-front on CPU before upload. This gives correct alpha compositing for overlapping splats.

## Browser Support

- Chrome 113+ (recommended)
- Edge 113+
- Firefox Nightly (with `dom.webgpu.enabled`)
- Safari — WebGPU in Technology Preview

Fallback: Shows a message if WebGPU is unavailable.

## License

MIT
