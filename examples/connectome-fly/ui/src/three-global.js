// Expose Three.js as `window.THREE` for the original IIFE modules
// (scene.js, fly.js, …) that were written against the CDN global.
//
// This lives in its own module so its side effect runs BEFORE any
// downstream module that reads `window.THREE`. ES modules are
// evaluated depth-first in import order, so importing this file at
// the very top of main.js guarantees the assignment lands before the
// module graph reaches scene.js.

import * as THREE from 'three';

window.THREE = THREE;
