// Connectome OS — Vite entry point.
//
// The existing modules under ./modules/ are IIFEs that attach their
// public API to `window.<Name>` (window.Scene, window.Dynamics, …).
// Importing them here runs their side effects in Vite's bundle,
// preserving the original load order expected by the HTML.

import * as THREE from 'three';

import './styles/tokens.css';
import './styles/layout.css';
import './styles/views.css';
import './styles/help.css';
import './styles/mobile.css';
import './styles/overlays.css';

// Expose Three.js on window so scene.js (and any other module that
// references the global `THREE` shipped by the original CDN script)
// keeps working unchanged.
window.THREE = THREE;

// Load order matches the original HTML.
import './modules/ui.js';
import './modules/nav.js';
import './modules/views.js';
import './modules/overlays.js';
import './modules/help.js';
import './modules/scene.js';
import './modules/dynamics.js';
import './modules/fly.js';
import './modules/actions.js';
