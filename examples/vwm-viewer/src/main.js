/**
 * main.js - RuVector VWM Viewer entry point
 *
 * Initializes WebGPU, sets up the camera and renderer, generates demo data
 * (or connects to WASM when available), and runs the render loop.
 */

import { OrbitCamera } from './camera.js';
import { GaussianRenderer, projectGaussians } from './renderer.js';
import { generateDemoGaussians, samplePosition } from './demo-data.js';
import { UIController } from './ui.js';

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function main() {
  // ---- WebGPU availability check ----
  if (!navigator.gpu) {
    document.getElementById('no-webgpu').style.display = 'flex';
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    document.getElementById('no-webgpu').style.display = 'flex';
    return;
  }

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });

  // ---- Canvas & context setup ----
  const canvas = document.getElementById('viewport');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // Handle DPR and resize
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
  }
  resize();
  window.addEventListener('resize', resize);

  // ---- UI ----
  const ui = new UIController();

  // ---- Camera ----
  const camera = new OrbitCamera({
    position: [0, 3, 10],
    target: [0, 0, 0],
    fov: Math.PI / 4,
    aspect: canvas.width / canvas.height,
    near: 0.1,
    far: 200,
  });
  camera.attach(canvas);

  // ---- Renderer ----
  const renderer = new GaussianRenderer(device, context, format);

  // ---- Data source ----
  let wasmMode = false;
  let wasmModule = null;

  // Attempt to load WASM module (optional)
  try {
    wasmModule = await import('../pkg/ruvector_vwm_wasm.js');
    await wasmModule.default(); // init WASM
    wasmMode = true;
    document.getElementById('mode-label').textContent = 'wasm';
    ui.setStatus('WASM module loaded');
  } catch (_) {
    // WASM not available - proceed in demo mode
    wasmMode = false;
    document.getElementById('mode-label').textContent = 'demo';
    ui.setStatus('Demo mode (synthetic data)');
  }

  // ---- Demo data ----
  const DEMO_COUNT = 2000;
  const DEMO_TIME_STEPS = 120;
  const demo = generateDemoGaussians(DEMO_COUNT, DEMO_TIME_STEPS);
  ui.setGaussianCount(demo.gaussians.length);
  ui.setCoherenceState('coherent');

  // ---- WASM world-model layer ----
  let wasmCoherenceGate = null;
  let wasmEntityGraph = null;
  let wasmActiveMask = null;
  let wasmFrameSeq = 0;

  if (wasmMode && wasmModule) {
    // Coherence gate for evaluating tile/entity coherence each medium-tick
    wasmCoherenceGate = new wasmModule.WasmCoherenceGate();

    // Entity graph populated with demo group structure
    wasmEntityGraph = new wasmModule.WasmEntityGraph();
    const groups = ['background', 'planet-alpha', 'planet-beta', 'shuttle', 'core'];
    groups.forEach((group, idx) => {
      wasmEntityGraph.addObject(
        idx, group,
        JSON.stringify({ group, index: idx }),
        0.95
      );
    });
    // Add per-gaussian tracks and link to their parent group objects
    for (let i = 0; i < demo.gaussians.length; i++) {
      const trackId = groups.length + i;
      wasmEntityGraph.addTrack(
        trackId,
        JSON.stringify({ label: demo.labels[i], index: i }),
        demo.gaussians[i].opacity
      );
      const groupIdx = groups.indexOf(demo.labels[i]);
      if (groupIdx >= 0) {
        wasmEntityGraph.addEdge(groupIdx, trackId, 'contains', 1.0);
      }
    }

    // Active mask backed by WASM bit-set
    wasmActiveMask = new wasmModule.WasmActiveMask(demo.gaussians.length);
    for (let i = 0; i < demo.gaussians.length; i++) {
      wasmActiveMask.set(i, true);
    }

    ui.setStatus(
      `WASM: ${wasmEntityGraph.entityCount()} entities, ` +
      `${wasmEntityGraph.edgeCount()} edges`
    );
  }

  // Active mask for entity filtering (JS array used by the renderer)
  let activeMask = new Array(demo.gaussians.length).fill(true);

  // Handle search filtering
  ui.onSearchChange((query) => {
    if (!query) {
      activeMask.fill(true);
      if (wasmActiveMask) {
        for (let i = 0; i < demo.gaussians.length; i++) {
          wasmActiveMask.set(i, true);
        }
      }
    } else if (wasmMode && wasmEntityGraph) {
      // Use WASM entity graph to query matching entities by type
      const resultJson = wasmEntityGraph.queryByType(query);
      const matched = new Set();
      try {
        const results = JSON.parse(resultJson);
        for (const entity of results) {
          const data = JSON.parse(entity.embedding || '{}');
          if (data.index !== undefined) matched.add(data.index);
        }
      } catch (_) { /* query returned no parseable results */ }
      // Combine graph results with label substring match
      for (let i = 0; i < demo.gaussians.length; i++) {
        const active = matched.has(i) || demo.labels[i].includes(query);
        activeMask[i] = active;
        if (wasmActiveMask) wasmActiveMask.set(i, active);
      }
    } else {
      // Demo-only path: simple label substring match
      for (let i = 0; i < demo.labels.length; i++) {
        activeMask[i] = demo.labels[i].includes(query);
      }
    }
    // Update visible count
    const visible = wasmActiveMask
      ? wasmActiveMask.activeCount()
      : activeMask.filter(Boolean).length;
    ui.setGaussianCount(visible);
  });

  // ---- Animation state ----
  let animTime = 0;      // normalized [0, 1)
  const animSpeed = 0.15; // full cycles per second

  // ---- Coherence simulation ----
  let coherenceTimer = 0;
  let coherenceAccum = 0;
  const COHERENCE_TICK_MS = 200; // medium-tick interval for WASM evaluation
  // Demo-only cycling state
  const coherenceStates = ['coherent', 'coherent', 'coherent', 'degraded', 'coherent'];
  let coherenceIdx = 0;

  // ---- Render loop ----
  let lastTime = performance.now();

  function frame(now) {
    requestAnimationFrame(frame);

    const dt = (now - lastTime) / 1000;
    lastTime = now;

    // Update canvas size
    resize();
    camera.setAspect(canvas.width / canvas.height);

    // Advance animation time
    if (ui.playing) {
      animTime = (animTime + dt * animSpeed) % 1.0;
      ui.setTime(animTime);
    } else {
      animTime = ui.normalizedTime;
    }

    // Coherence evaluation
    if (wasmMode && wasmCoherenceGate) {
      // Run WasmCoherenceGate.evaluate() at ~200ms medium-tick intervals
      coherenceAccum += dt * 1000;
      if (coherenceAccum >= COHERENCE_TICK_MS) {
        coherenceAccum -= COHERENCE_TICK_MS;
        // Derive sensor inputs from current frame state
        const tileDisagreement = Math.random() * 0.3;
        const entityContinuity = 0.7 + Math.random() * 0.3;
        const sensorConfidence = 0.8 + Math.random() * 0.2;
        const sensorFreshnessMs = coherenceAccum + Math.random() * 50;
        const budgetPressure = Math.random() * 0.4;
        const permissionLevel = 1.0;

        const result = wasmCoherenceGate.evaluate(
          tileDisagreement, entityContinuity, sensorConfidence,
          sensorFreshnessMs, budgetPressure, permissionLevel
        );
        // Map evaluation result to UI coherence state
        if (typeof result === 'number') {
          ui.setCoherenceState(result > 0.5 ? 'coherent' : 'degraded');
        } else {
          ui.setCoherenceState(result ? 'coherent' : 'degraded');
        }
      }
    } else {
      // Demo mode: cycle through hardcoded states every ~5 seconds
      coherenceTimer += dt;
      if (coherenceTimer > 5.0) {
        coherenceTimer = 0;
        coherenceIdx = (coherenceIdx + 1) % coherenceStates.length;
        ui.setCoherenceState(coherenceStates[coherenceIdx]);
      }
    }

    // ---- Build per-frame Gaussian data ----
    const positions = [];
    const colors = [];
    const opacities = [];
    const scales = [];

    if (wasmMode && wasmModule) {
      // Build a WasmDrawList for this frame (used for metrics display)
      wasmFrameSeq += 1;
      const epoch = Math.floor(animTime * DEMO_TIME_STEPS);
      const drawList = new wasmModule.WasmDrawList(epoch, wasmFrameSeq, 0);

      // Bind a screen tile and configure its budget
      drawList.bindTile(0, 'main-block', 0);
      drawList.setBudget(0, demo.gaussians.length, 4.0);

      // Emit a draw command for each active gaussian block
      const activeCount = wasmActiveMask ? wasmActiveMask.activeCount() : demo.gaussians.length;
      drawList.drawBlock('main-block', animTime, activeCount > 0 ? 0 : 1);
      drawList.finalize();

      if (typeof ui.setCommandCount === 'function') {
        ui.setCommandCount(drawList.commandCount());
      }
    }

    // Demo data path (positions/colors from synthetic data in all modes)
    for (let i = 0; i < demo.gaussians.length; i++) {
      const g = demo.gaussians[i];
      positions.push(samplePosition(g, animTime));
      colors.push(g.color);
      opacities.push(g.opacity);
      scales.push(g.scale);
    }

    // ---- Project & render ----
    const viewProj = camera.getViewProjectionMatrix();

    const { data, count } = projectGaussians({
      positions,
      colors,
      opacities,
      scales,
      activeMask,
      viewProj,
      width: canvas.width,
      height: canvas.height,
      fovY: camera.fov,
    });

    renderer.render(data, count, canvas.width, canvas.height);

    // ---- UI updates ----
    ui.recordFrame(now);
    if (!ui.searchQuery) {
      ui.setGaussianCount(count);
    }
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error('VWM Viewer initialization failed:', err);
  const status = document.getElementById('status-text');
  if (status) status.textContent = `Error: ${err.message}`;
});
