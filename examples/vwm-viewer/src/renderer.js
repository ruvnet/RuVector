/**
 * renderer.js - WebGPU Gaussian splatting renderer
 *
 * Pipeline overview:
 *   1. CPU: project Gaussian centers into screen space, compute 2D conic,
 *      sort by depth (front-to-back is fine for alpha blending with
 *      pre-multiplied alpha, but we sort back-to-front for correct
 *      compositing with standard alpha blending).
 *   2. GPU: instanced draw of screen-aligned quads, one per visible Gaussian.
 *      The fragment shader evaluates the 2D Gaussian kernel and applies
 *      alpha-blended color.
 */

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

const VERTEX_WGSL = /* wgsl */ `
struct Uniforms {
  viewport : vec2f,      // canvas width, height
  pad0     : f32,
  pad1     : f32,
};

struct SplatInstance {
  // Screen-space center (pixels), packed into first two floats
  center   : vec2f,
  // Upper-triangle of 2D inverse covariance (conic) matrix
  // conic.x = a, conic.y = b, conic.z = c  where Q = a*dx^2 + 2*b*dx*dy + c*dy^2
  conic    : vec3f,
  // Splat color (linear RGB) and opacity
  color    : vec4f,
  // Screen-space radius for the quad (pixels)
  radius   : f32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> splats : array<SplatInstance>;

struct VsOut {
  @builtin(position) pos : vec4f,
  @location(0) delta     : vec2f,  // offset from splat center in pixels
  @location(1) color     : vec4f,
  @location(2) conic     : vec3f,
};

// Quad vertices: two triangles covering [-1, -1] to [1, 1]
const QUAD_POS = array<vec2f, 6>(
  vec2f(-1.0, -1.0),
  vec2f( 1.0, -1.0),
  vec2f(-1.0,  1.0),
  vec2f(-1.0,  1.0),
  vec2f( 1.0, -1.0),
  vec2f( 1.0,  1.0),
);

@vertex
fn vs_main(
  @builtin(vertex_index)   vid : u32,
  @builtin(instance_index) iid : u32,
) -> VsOut {
  let splat = splats[iid];
  let qv    = QUAD_POS[vid];

  // Offset in pixels from splat center
  let delta = qv * splat.radius;

  // Screen-space position of this vertex
  let screen = splat.center + delta;

  // Convert to clip space: x in [-1,1], y in [-1,1]
  let clip = vec2f(
    (screen.x / uniforms.viewport.x) * 2.0 - 1.0,
    1.0 - (screen.y / uniforms.viewport.y) * 2.0,
  );

  var out : VsOut;
  out.pos   = vec4f(clip, 0.0, 1.0);
  out.delta = delta;
  out.color = splat.color;
  out.conic = splat.conic;
  return out;
}
`;

const FRAGMENT_WGSL = /* wgsl */ `
struct VsOut {
  @builtin(position) pos : vec4f,
  @location(0) delta     : vec2f,
  @location(1) color     : vec4f,
  @location(2) conic     : vec3f,
};

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4f {
  let dx = in.delta.x;
  let dy = in.delta.y;
  // Evaluate 2D Gaussian:  G = exp(-0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2))
  let power = -0.5 * (in.conic.x * dx * dx
                     + 2.0 * in.conic.y * dx * dy
                     + in.conic.z * dy * dy);
  // Clamp to avoid extreme values
  if (power > 0.0) { discard; }
  let alpha = min(0.99, in.color.a * exp(power));
  if (alpha < 1.0 / 255.0) { discard; }
  // Pre-multiplied alpha output
  return vec4f(in.color.rgb * alpha, alpha);
}
`;

// ---------------------------------------------------------------------------
// Renderer class
// ---------------------------------------------------------------------------

export class GaussianRenderer {
  /**
   * @param {GPUDevice} device
   * @param {GPUCanvasContext} context
   * @param {GPUTextureFormat} format
   */
  constructor(device, context, format) {
    this.device = device;
    this.context = context;
    this.format = format;

    // Maximum splats we can render per frame
    this.maxSplats = 100_000;

    // Current splat count for the active frame
    this.activeSplatCount = 0;

    this._initPipeline();
  }

  // ---- Public API ----

  /**
   * Upload projected splat data and render one frame.
   *
   * @param {Float32Array} splatData  - Packed splat instances (see SplatInstance struct)
   * @param {number}       count      - Number of splats
   * @param {number}       width      - Canvas width in pixels
   * @param {number}       height     - Canvas height in pixels
   */
  render(splatData, count, width, height) {
    if (count === 0) return;
    this.activeSplatCount = count;

    // Update uniform buffer (viewport size)
    const uniformData = new Float32Array([width, height, 0, 0]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // Upload splat instance data
    const byteLength = count * this.SPLAT_STRIDE_BYTES;
    this.device.queue.writeBuffer(this.splatBuffer, 0, splatData, 0, count * this.SPLAT_STRIDE_FLOATS);

    // Encode render pass
    const encoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.04, g: 0.04, b: 0.06, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, count, 0, 0); // 6 vertices per quad, `count` instances
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // ---- Internal setup ----

  _initPipeline() {
    // Each SplatInstance has: center(2) + conic(3) + color(4) + radius(1) = 10 floats
    // However we need 16-byte alignment for vec3/vec4 in storage buffers,
    // so we pad to 12 floats per instance:
    //   [center.x, center.y, pad, pad,  conic.x, conic.y, conic.z, pad,  r, g, b, a,  radius, pad, pad, pad]
    // That is 16 floats = 64 bytes per instance.
    this.SPLAT_STRIDE_FLOATS = 16;
    this.SPLAT_STRIDE_BYTES = this.SPLAT_STRIDE_FLOATS * 4;

    // Uniform buffer
    this.uniformBuffer = this.device.createBuffer({
      size: 16, // vec2f + 2 padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Storage buffer for splat instances
    this.splatBuffer = this.device.createBuffer({
      size: this.maxSplats * this.SPLAT_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Shader modules
    const vsModule = this.device.createShaderModule({ code: VERTEX_WGSL });
    const fsModule = this.device.createShaderModule({ code: FRAGMENT_WGSL });

    // Bind group layout
    const bglayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bglayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.splatBuffer } },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bglayout],
    });

    // Render pipeline with alpha blending
    this.pipeline = this.device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: vsModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: fsModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
          },
        }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
  }
}

// ---------------------------------------------------------------------------
// CPU-side projection helpers
// ---------------------------------------------------------------------------

/**
 * Project 3D Gaussians into screen-space splat instances.
 *
 * This performs:
 *   - View-projection transform of the center
 *   - Approximate 2D covariance from the 3D scale (simplified: isotropic)
 *   - Depth sorting (back-to-front for correct alpha compositing)
 *   - Active-mask filtering
 *
 * @param {object}       opts
 * @param {number[][]}   opts.positions    - Array of [x,y,z] world positions
 * @param {number[][]}   opts.colors       - Array of [r,g,b]
 * @param {number[]}     opts.opacities    - Array of opacity values
 * @param {number[][]}   opts.scales       - Array of [sx,sy,sz]
 * @param {boolean[]}    opts.activeMask   - Per-Gaussian visibility mask (optional)
 * @param {Float32Array} opts.viewProj     - 4x4 view-projection matrix (column-major)
 * @param {number}       opts.width        - Canvas width
 * @param {number}       opts.height       - Canvas height
 * @param {number}       opts.fovY         - Vertical FOV in radians
 * @returns {{ data: Float32Array, count: number }}
 */
export function projectGaussians(opts) {
  const {
    positions, colors, opacities, scales,
    activeMask, viewProj, width, height, fovY,
  } = opts;

  const n = positions.length;
  const focal = height / (2.0 * Math.tan(fovY / 2.0));

  // Step 1: project centers, compute depth, filter
  const projected = [];
  for (let i = 0; i < n; i++) {
    if (activeMask && !activeMask[i]) continue;

    const [wx, wy, wz] = positions[i];
    // Multiply by viewProj (column-major)
    const cx = viewProj[0] * wx + viewProj[4] * wy + viewProj[8]  * wz + viewProj[12];
    const cy = viewProj[1] * wx + viewProj[5] * wy + viewProj[9]  * wz + viewProj[13];
    const cz = viewProj[2] * wx + viewProj[6] * wy + viewProj[10] * wz + viewProj[14];
    const cw = viewProj[3] * wx + viewProj[7] * wy + viewProj[11] * wz + viewProj[15];

    if (cw <= 0.001) continue; // behind camera

    // NDC
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    const depth = cz / cw;

    // Screen space
    const sx = (ndcX * 0.5 + 0.5) * width;
    const sy = (1.0 - (ndcY * 0.5 + 0.5)) * height;

    // Frustum cull (generous margin)
    if (sx < -200 || sx > width + 200 || sy < -200 || sy > height + 200) continue;

    // Approximate 2D radius from 3D scale projected through perspective
    const avgScale = (scales[i][0] + scales[i][1] + scales[i][2]) / 3.0;
    const projectedRadius = (focal * avgScale) / (cw);
    if (projectedRadius < 0.3) continue; // too small to see

    // Conic for an isotropic Gaussian: a = c = 1/sigma^2, b = 0
    const sigma = Math.max(projectedRadius * 0.5, 0.5);
    const invSigma2 = 1.0 / (sigma * sigma);

    projected.push({
      depth,
      sx, sy,
      conicA: invSigma2,
      conicB: 0.0,
      conicC: invSigma2,
      r: colors[i][0],
      g: colors[i][1],
      b: colors[i][2],
      opacity: opacities[i],
      radius: Math.min(sigma * 3.0, 512), // clamp splat size
    });
  }

  // Step 2: sort back-to-front (descending depth)
  projected.sort((a, b) => b.depth - a.depth);

  // Step 3: pack into Float32Array (16 floats per splat, matching WGSL struct)
  const count = projected.length;
  const data = new Float32Array(count * 16);
  for (let i = 0; i < count; i++) {
    const p = projected[i];
    const off = i * 16;
    // center (vec2f) + 2 pad
    data[off + 0] = p.sx;
    data[off + 1] = p.sy;
    data[off + 2] = 0;
    data[off + 3] = 0;
    // conic (vec3f) + 1 pad
    data[off + 4] = p.conicA;
    data[off + 5] = p.conicB;
    data[off + 6] = p.conicC;
    data[off + 7] = 0;
    // color (vec4f)
    data[off + 8]  = p.r;
    data[off + 9]  = p.g;
    data[off + 10] = p.b;
    data[off + 11] = p.opacity;
    // radius + 3 pad
    data[off + 12] = p.radius;
    data[off + 13] = 0;
    data[off + 14] = 0;
    data[off + 15] = 0;
  }

  return { data, count };
}
