// Connectome OS — Vite entry point.
//
// The existing modules under ./modules/ are IIFEs that attach their
// public API to `window.<Name>` (window.Scene, window.Dynamics, …).
// Importing them here runs their side effects in Vite's bundle,
// preserving the original load order expected by the HTML.
//
// three-global.js MUST stay first — ES module imports are hoisted
// and evaluated in source order, so any downstream module that reads
// `window.THREE` needs this side effect to have already run.

import './three-global.js';

import './styles/tokens.css';
import './styles/layout.css';
import './styles/views.css';
import './styles/help.css';
import './styles/mobile.css';
import './styles/overlays.css';

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
import './modules/fly-sim.js';
import './modules/welcome.js';
