/**
 * demo-data.js - Synthetic 4D Gaussian data generator
 *
 * Generates a set of Gaussians with temporal animation for testing the
 * viewer without a WASM backend. Each Gaussian has:
 *   position (x, y, z), color (r, g, b), opacity, scale (sx, sy, sz),
 *   rotation quaternion (qw, qx, qy, qz), and an entity label.
 *
 * Temporal keyframes are stored so the viewer can interpolate at any time t.
 */

/**
 * Generate demo Gaussians with temporal motion.
 *
 * @param {number} count      - Total number of Gaussians
 * @param {number} timeSteps  - Number of discrete time steps
 * @returns {{ gaussians: object[], timeSteps: number, labels: string[] }}
 */
export function generateDemoGaussians(count = 2000, timeSteps = 120) {
  const gaussians = [];
  const labels = [];

  // ---- Static background cloud ----
  const bgCount = Math.floor(count * 0.6);
  for (let i = 0; i < bgCount; i++) {
    gaussians.push(makeStaticGaussian(i));
    labels.push('background');
  }

  // ---- Orbiting sphere cluster ("planet") ----
  const orbitCount = Math.floor(count * 0.15);
  for (let i = 0; i < orbitCount; i++) {
    gaussians.push(makeOrbitGaussian(i, orbitCount, timeSteps, 3.0, 0));
    labels.push('planet-alpha');
  }

  // ---- Second orbiting cluster, different radius/color ----
  const orbit2Count = Math.floor(count * 0.1);
  for (let i = 0; i < orbit2Count; i++) {
    gaussians.push(makeOrbitGaussian(i, orbit2Count, timeSteps, 5.0, 1));
    labels.push('planet-beta');
  }

  // ---- Linear mover ("shuttle") ----
  const shuttleCount = Math.floor(count * 0.05);
  for (let i = 0; i < shuttleCount; i++) {
    gaussians.push(makeLinearGaussian(i, shuttleCount, timeSteps));
    labels.push('shuttle');
  }

  // ---- Pulsing center object ("core") ----
  const coreCount = count - bgCount - orbitCount - orbit2Count - shuttleCount;
  for (let i = 0; i < coreCount; i++) {
    gaussians.push(makePulsingGaussian(i, coreCount, timeSteps));
    labels.push('core');
  }

  return { gaussians, timeSteps, labels };
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function makeStaticGaussian(idx) {
  // Scattered around origin in a loose sphere
  const r = rand(2, 12);
  const theta = rand(0, Math.PI * 2);
  const phi = rand(-Math.PI / 2, Math.PI / 2);
  const x = r * Math.cos(phi) * Math.sin(theta);
  const y = r * Math.sin(phi);
  const z = r * Math.cos(phi) * Math.cos(theta);

  return {
    positions: [[x, y, z]], // single keyframe = static
    color: [rand(0.15, 0.35), rand(0.15, 0.35), rand(0.4, 0.7)],
    opacity: rand(0.3, 0.7),
    scale: [rand(0.05, 0.15), rand(0.05, 0.15), rand(0.05, 0.15)],
    rotation: [1, 0, 0, 0],
  };
}

function makeOrbitGaussian(idx, total, timeSteps, orbitRadius, colorSet) {
  // Each Gaussian is offset from the cluster center
  const clusterSpread = 0.4;
  const ox = rand(-clusterSpread, clusterSpread);
  const oy = rand(-clusterSpread, clusterSpread);
  const oz = rand(-clusterSpread, clusterSpread);

  const positions = [];
  for (let t = 0; t < timeSteps; t++) {
    const angle = (t / timeSteps) * Math.PI * 2;
    const cx = orbitRadius * Math.sin(angle);
    const cy = Math.sin(angle * 2) * 0.5; // slight vertical bob
    const cz = orbitRadius * Math.cos(angle);
    positions.push([cx + ox, cy + oy, cz + oz]);
  }

  const colors = [
    [rand(0.8, 1.0), rand(0.3, 0.5), rand(0.1, 0.3)],
    [rand(0.2, 0.4), rand(0.8, 1.0), rand(0.3, 0.5)],
  ];

  return {
    positions,
    color: colors[colorSet % colors.length],
    opacity: rand(0.6, 0.95),
    scale: [rand(0.08, 0.2), rand(0.08, 0.2), rand(0.08, 0.2)],
    rotation: [1, 0, 0, 0],
  };
}

function makeLinearGaussian(idx, total, timeSteps) {
  const spread = 0.2;
  const ox = rand(-spread, spread);
  const oy = rand(-spread, spread);
  const oz = rand(-spread, spread);

  const positions = [];
  for (let t = 0; t < timeSteps; t++) {
    const frac = t / timeSteps;
    // Shuttle moves along x axis, back and forth
    const cx = (frac < 0.5 ? frac * 2 - 0.5 : 1.5 - frac * 2) * 10.0;
    positions.push([cx + ox, 2.0 + oy, oz]);
  }

  return {
    positions,
    color: [rand(0.9, 1.0), rand(0.9, 1.0), rand(0.3, 0.5)],
    opacity: rand(0.7, 1.0),
    scale: [rand(0.05, 0.12), rand(0.05, 0.12), rand(0.15, 0.3)],
    rotation: [1, 0, 0, 0],
  };
}

function makePulsingGaussian(idx, total, timeSteps) {
  const angle = (idx / total) * Math.PI * 2;
  const baseR = rand(0.2, 0.6);
  const x = baseR * Math.cos(angle);
  const z = baseR * Math.sin(angle);

  const positions = [];
  for (let t = 0; t < timeSteps; t++) {
    const pulse = 1.0 + 0.3 * Math.sin((t / timeSteps) * Math.PI * 4);
    positions.push([x * pulse, rand(-0.1, 0.1), z * pulse]);
  }

  return {
    positions,
    color: [rand(0.9, 1.0), rand(0.5, 0.7), rand(0.8, 1.0)],
    opacity: rand(0.7, 1.0),
    scale: [rand(0.1, 0.25), rand(0.1, 0.25), rand(0.1, 0.25)],
    rotation: [1, 0, 0, 0],
  };
}

/**
 * Sample the position of a Gaussian at fractional time t in [0, 1).
 * Linearly interpolates between keyframes.
 *
 * @param {object} g    - A gaussian object from generateDemoGaussians
 * @param {number} t    - Normalized time in [0, 1)
 * @returns {number[]}  - [x, y, z]
 */
export function samplePosition(g, t) {
  const positions = g.positions;
  if (positions.length === 1) return positions[0];
  const ft = t * (positions.length - 1);
  const i0 = Math.floor(ft);
  const i1 = Math.min(i0 + 1, positions.length - 1);
  const frac = ft - i0;
  return [
    positions[i0][0] + (positions[i1][0] - positions[i0][0]) * frac,
    positions[i0][1] + (positions[i1][1] - positions[i0][1]) * frac,
    positions[i0][2] + (positions[i1][2] - positions[i0][2]) * frac,
  ];
}
