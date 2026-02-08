/**
 * camera.js - Orbit camera with mouse/touch controls
 *
 * Provides a simple orbit camera that rotates around a target point.
 * Returns view and projection matrices as Float32Arrays suitable for
 * uploading directly to WebGPU uniform buffers.
 */

export class OrbitCamera {
  /**
   * @param {Object} opts
   * @param {number[]} opts.position  - Initial eye position [x, y, z]
   * @param {number[]} opts.target    - Look-at target [x, y, z]
   * @param {number[]} opts.up        - World up vector [x, y, z]
   * @param {number}   opts.fov       - Vertical field of view in radians
   * @param {number}   opts.aspect    - Viewport width / height
   * @param {number}   opts.near      - Near clip plane
   * @param {number}   opts.far       - Far clip plane
   */
  constructor(opts = {}) {
    this.target = new Float32Array(opts.target ?? [0, 0, 0]);
    this.up = new Float32Array(opts.up ?? [0, 1, 0]);
    this.fov = opts.fov ?? Math.PI / 4;
    this.aspect = opts.aspect ?? 1;
    this.near = opts.near ?? 0.1;
    this.far = opts.far ?? 200.0;

    // Spherical coordinates around target
    const pos = opts.position ?? [0, 2, 8];
    const dx = pos[0] - this.target[0];
    const dy = pos[1] - this.target[1];
    const dz = pos[2] - this.target[2];
    this.radius = Math.sqrt(dx * dx + dy * dy + dz * dz);
    this.theta = Math.atan2(dx, dz); // azimuth
    this.phi = Math.asin(Math.min(1, Math.max(-1, dy / this.radius))); // elevation

    // Interaction state
    this._dragging = false;
    this._lastX = 0;
    this._lastY = 0;

    // Pre-allocated output matrices (column-major, 4x4)
    this._view = new Float32Array(16);
    this._proj = new Float32Array(16);
    this._viewProj = new Float32Array(16);
  }

  /** Attach mouse and wheel listeners to the given canvas element. */
  attach(canvas) {
    canvas.addEventListener('mousedown', (e) => {
      this._dragging = true;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => {
      this._dragging = false;
    });
    window.addEventListener('mousemove', (e) => {
      if (!this._dragging) return;
      const dx = e.clientX - this._lastX;
      const dy = e.clientY - this._lastY;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      this.theta -= dx * 0.005;
      this.phi += dy * 0.005;
      // Clamp phi to avoid flipping
      this.phi = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.phi));
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.radius *= 1 + e.deltaY * 0.001;
      this.radius = Math.max(0.5, Math.min(100, this.radius));
    }, { passive: false });

    // Touch support
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this._dragging = true;
        this._lastX = e.touches[0].clientX;
        this._lastY = e.touches[0].clientY;
      }
    });
    canvas.addEventListener('touchend', () => {
      this._dragging = false;
    });
    canvas.addEventListener('touchmove', (e) => {
      if (!this._dragging || e.touches.length !== 1) return;
      const dx = e.touches[0].clientX - this._lastX;
      const dy = e.touches[0].clientY - this._lastY;
      this._lastX = e.touches[0].clientX;
      this._lastY = e.touches[0].clientY;
      this.theta -= dx * 0.005;
      this.phi += dy * 0.005;
      this.phi = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.phi));
    });
  }

  /** Current eye position derived from spherical coordinates. */
  getPosition() {
    return [
      this.target[0] + this.radius * Math.cos(this.phi) * Math.sin(this.theta),
      this.target[1] + this.radius * Math.sin(this.phi),
      this.target[2] + this.radius * Math.cos(this.phi) * Math.cos(this.theta),
    ];
  }

  /** Update aspect ratio (call on canvas resize). */
  setAspect(aspect) {
    this.aspect = aspect;
  }

  // ---- Matrix builders (column-major for WebGPU/WGSL) ----

  /** Compute a look-at view matrix and write into this._view. */
  getViewMatrix() {
    const eye = this.getPosition();
    const t = this.target;
    // Forward (z axis points from target to eye in right-hand)
    let fx = eye[0] - t[0], fy = eye[1] - t[1], fz = eye[2] - t[2];
    let len = Math.sqrt(fx * fx + fy * fy + fz * fz);
    fx /= len; fy /= len; fz /= len;
    // Right = up x forward
    let rx = this.up[1] * fz - this.up[2] * fy;
    let ry = this.up[2] * fx - this.up[0] * fz;
    let rz = this.up[0] * fy - this.up[1] * fx;
    len = Math.sqrt(rx * rx + ry * ry + rz * rz);
    rx /= len; ry /= len; rz /= len;
    // True up = forward x right
    const ux = fy * rz - fz * ry;
    const uy = fz * rx - fx * rz;
    const uz = fx * ry - fy * rx;

    const m = this._view;
    // Column 0
    m[0] = rx; m[1] = ux; m[2] = fx; m[3] = 0;
    // Column 1
    m[4] = ry; m[5] = uy; m[6] = fy; m[7] = 0;
    // Column 2
    m[8] = rz; m[9] = uz; m[10] = fz; m[11] = 0;
    // Column 3
    m[12] = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);
    m[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
    m[14] = -(fx * eye[0] + fy * eye[1] + fz * eye[2]);
    m[15] = 1;
    return m;
  }

  /** Compute a perspective projection matrix and write into this._proj. */
  getProjectionMatrix() {
    const f = 1.0 / Math.tan(this.fov / 2);
    const rangeInv = 1.0 / (this.near - this.far);
    const m = this._proj;
    m[0] = f / this.aspect; m[1] = 0; m[2] = 0; m[3] = 0;
    m[4] = 0; m[5] = f; m[6] = 0; m[7] = 0;
    m[8] = 0; m[9] = 0; m[10] = this.far * rangeInv; m[11] = -1;
    m[12] = 0; m[13] = 0; m[14] = this.far * this.near * rangeInv; m[15] = 0;
    return m;
  }

  /** Returns the combined view-projection matrix (proj * view). */
  getViewProjectionMatrix() {
    const v = this.getViewMatrix();
    const p = this.getProjectionMatrix();
    const o = this._viewProj;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        o[j * 4 + i] =
          p[0 * 4 + i] * v[j * 4 + 0] +
          p[1 * 4 + i] * v[j * 4 + 1] +
          p[2 * 4 + i] * v[j * 4 + 2] +
          p[3 * 4 + i] * v[j * 4 + 3];
      }
    }
    return o;
  }
}
