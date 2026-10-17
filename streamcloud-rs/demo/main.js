// StreamCloud — WebGPU browser demo.
//
// Loads the Rust pipeline (compiled to WASM) and renders the streaming 3D
// reconstruction with WebGPU. Falls back to a 2D canvas renderer when WebGPU is
// unavailable so the page always shows something.

import init, { StreamCloudDemo } from './pkg/streamcloud_wasm.js';

const SCENE_W = 256, SCENE_H = 192, TOP_K = 16;
const STRIDE = 6;                       // floats per point: x,y,z,r,g,b
const MAX_POINTS = (SCENE_W / 4) * (SCENE_H / 4) + 16;

const el = (id) => document.getElementById(id);
const ui = {
  backend: el('backend'), frame: el('frame'), points: el('points'),
  anchors: el('anchors'), keyframes: el('keyframes'),
  toggle: el('toggle'), orbit: el('orbit'),
};

let running = true;
let angle = 0;

// ---- tiny mat4 helpers (column-major) ----
const mul = (a, b) => {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      o[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] + a[1 * 4 + r] * b[c * 4 + 1]
                   + a[2 * 4 + r] * b[c * 4 + 2] + a[3 * 4 + r] * b[c * 4 + 3];
  return o;
};
const perspective = (fovy, aspect, near, far) => {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
};
const lookAt = (eye, center, up) => {
  const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const dot = (a, b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const z = norm(sub(eye, center));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
};
function mvpMatrix(aspect) {
  const r = 7.5;
  const eye = [Math.sin(angle) * r, 1.5, 5 + Math.cos(angle) * r];
  const proj = perspective(50 * Math.PI / 180, aspect, 0.1, 100);
  const view = lookAt(eye, [0, 0, 5], [0, 1, 0]);
  return mul(proj, view);
}

// ---- WebGPU renderer ----
async function tryWebGPU(canvas) {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const shader = device.createShaderModule({
    code: `
struct U { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: U;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec3<f32> };
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) c: vec3<f32>) -> VSOut {
  var o: VSOut;
  o.pos = u.mvp * vec4<f32>(p, 1.0);
  o.color = c;
  return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}`,
  });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shader, entryPoint: 'vs',
      buffers: [{
        arrayStride: STRIDE * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
        ],
      }],
    },
    fragment: { module: shader, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'point-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const vbuf = device.createBuffer({
    size: MAX_POINTS * STRIDE * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  const ubuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: ubuf } }],
  });
  let depth = device.createTexture({
    size: [canvas.width, canvas.height], format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  return {
    name: 'WebGPU',
    draw(data, count) {
      device.queue.writeBuffer(vbuf, 0, data, 0, count * STRIDE);
      device.queue.writeBuffer(ubuf, 0, mvpMatrix(canvas.width / canvas.height));
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: ctx.getCurrentTexture().createView(),
          clearValue: { r: 0.03, g: 0.04, b: 0.07, a: 1 },
          loadOp: 'clear', storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: depth.createView(),
          depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.setVertexBuffer(0, vbuf);
      pass.draw(count);
      pass.end();
      device.queue.submit([enc.finish()]);
    },
  };
}

// ---- 2D fallback renderer ----
function fallback2D(canvas) {
  const g = canvas.getContext('2d');
  return {
    name: 'Canvas2D (no WebGPU)',
    draw(data, count) {
      const W = canvas.width, H = canvas.height;
      g.fillStyle = '#05070d'; g.fillRect(0, 0, W, H);
      const mvp = mvpMatrix(W / H);
      for (let i = 0; i < count; i++) {
        const o = i * STRIDE;
        const x = data[o], y = data[o + 1], z = data[o + 2];
        const cx = mvp[0]*x + mvp[4]*y + mvp[8]*z + mvp[12];
        const cy = mvp[1]*x + mvp[5]*y + mvp[9]*z + mvp[13];
        const cw = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
        if (cw <= 0) continue;
        const sx = (cx / cw * 0.5 + 0.5) * W;
        const sy = (1 - (cy / cw * 0.5 + 0.5)) * H;
        if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
        g.fillStyle = `rgb(${data[o+3]*255|0},${data[o+4]*255|0},${data[o+5]*255|0})`;
        g.fillRect(sx, sy, 2, 2);
      }
    },
  };
}

async function main() {
  await init();
  const demo = new StreamCloudDemo(SCENE_W, SCENE_H, TOP_K);
  const canvas = el('gpu');
  const renderer = (await tryWebGPU(canvas)) || fallback2D(canvas);
  ui.backend.textContent = renderer.name;
  if (renderer.name.startsWith('Canvas')) ui.backend.classList.add('warn');

  ui.toggle.onclick = () => {
    running = !running;
    ui.toggle.textContent = running ? 'Pause' : 'Resume';
  };

  function loop() {
    if (running) {
      const data = demo.next_frame();           // Float32Array [x,y,z,r,g,b]...
      const count = demo.point_count();
      renderer.draw(data, count);
      angle += parseFloat(ui.orbit.value);
      ui.frame.textContent = demo.frame_index();
      ui.points.textContent = count;
      ui.anchors.textContent = demo.anchors_used();
      ui.keyframes.textContent = demo.keyframes();
    }
    requestAnimationFrame(loop);
  }
  loop();
}

main().catch((e) => {
  el('backend').textContent = 'error: ' + e;
  el('backend').classList.add('warn');
  console.error(e);
});
