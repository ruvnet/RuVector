// Connectome OS — nav wiring & view switching
(function () {
  const VIEWS = {
    graph: {
      title: 'Connectome — co-firing graph',
      sub: '208 neurons · 4 modules · SBM fixture · partition live',
      canvasVisible: true
    },
    structure: {
      title: 'Structural layer — static adjacency',
      sub: 'typed directed graph · 139K target · FlyWire v783 fixture',
      canvasVisible: true
    },
    dynamics: {
      title: 'Dynamics — event-driven LIF',
      sub: 'wheel-based delivery · SoA · f32x8 lanes',
      canvasVisible: true
    },
    motifs: {
      title: 'Motif index — SDPA embeddings',
      sub: '100ms spike windows · HNSW kNN · brute-force fallback',
      canvasVisible: true
    },
    causal: {
      title: 'Causal perturbation',
      sub: 'targeted vs random null · σ-separation',
      canvasVisible: true
    },
    acceptance: {
      title: 'Acceptance suite — AT-1..5',
      sub: '68 tests · 0 fail · commit bd26c4ee4',
      canvasVisible: true
    },
    benchmarks: {
      title: 'Benchmarks — vs Brian2 / Auryn / NEST',
      sub: 'Ryzen · 1 thread · release · sparse + saturated',
      canvasVisible: true
    },
    embodiment: {
      title: 'Embodiment — motor I/O',
      sub: 'fly body · tripod gait · wing 200 Hz · sensory ↔ motor closed loop',
      canvasVisible: true
    },
    console: {
      title: 'Console — runtime introspection',
      sub: 'live trace · 11 measurement discoveries',
      canvasVisible: true
    },
    settings: {
      title: 'Session settings',
      sub: 'seed · engine flags · reproducibility',
      canvasVisible: true
    }
  };

  const titleEl = document.querySelector('.canvas-title h2');
  const subEl = document.querySelector('.canvas-title .sub');

  // Embodiment fly scene (lazy init on first activation)
  let flyInst = null;
  function ensureFly() {
    if (flyInst) return flyInst;
    const host = document.getElementById('fly-canvas');
    if (!host || !window.FlyScene) return null;
    flyInst = window.FlyScene.create(host);
    return flyInst;
  }

  function setEmbodimentVisible(on) {
    const wrap = document.querySelector('.canvas-wrap');
    const host = document.getElementById('fly-canvas');
    if (!host || !wrap) return;
    host.classList.toggle('active', on);
    wrap.classList.toggle('embodiment', on);
    if (on) {
      const f = ensureFly();
      if (f) { f.play(); f.resize(); }
    } else if (flyInst) {
      flyInst.pause();
    }
  }

  function activate(view) {
    const v = VIEWS[view] || VIEWS.graph;
    if (titleEl) {
      // Preserve the help icon when swapping title text
      const helpBtn = document.getElementById('canvas-help');
      titleEl.textContent = v.title;
      if (helpBtn) {
        helpBtn.dataset.help = 'view_' + view;
        titleEl.appendChild(helpBtn);
      }
    }
    if (subEl) subEl.innerHTML = v.sub.replace('partition live', '<span style="color:var(--signal)">partition live</span>');

    // Update rail active state
    document.querySelectorAll('.rail-item').forEach((el) => {
      el.classList.toggle('active', el.dataset.view === view);
    });
    document.querySelectorAll('.m-nav .item').forEach((el) => {
      el.classList.toggle('active', el.dataset.view === view);
    });

    // Highlight + scroll right-rail panel matching view
    const panels = document.querySelectorAll('.right-rail .panel');
    panels.forEach((p) => {
      p.classList.toggle('panel-focus', p.dataset.view === view);
    });
    const focused = document.querySelector(`.right-rail .panel[data-view="${view}"]`);
    if (focused) {
      const rail = document.querySelector('.right-rail');
      if (rail) {
        const top = focused.getBoundingClientRect().top - rail.getBoundingClientRect().top + rail.scrollTop - 12;
        rail.scrollTop = top;
        try { rail.scrollTo({ top, behavior: 'smooth' }); } catch (e) {}
      }
    }

    // Flash a view-indicator badge on the canvas
    let badge = document.getElementById('view-indicator');
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'view-indicator';
      badge.className = 'view-indicator';
      document.querySelector('.canvas-wrap')?.appendChild(badge);
    }
    badge.textContent = v.title;
    badge.classList.remove('show');
    // Force reflow then re-add
    void badge.offsetWidth;
    badge.classList.add('show');

    // Pulse the graph briefly
    window.ConnectomeScene?.pulseBurst(20);

    // Toggle embodiment scene
    setEmbodimentVisible(view === 'embodiment');

    // Swap view-specific content overlay
    if (window.ViewContent) window.ViewContent.setView(view);
    // Re-scan for help-icon triggers in newly-injected view content
    if (window.ConnectomeHelp) window.ConnectomeHelp.scan();
  }

  // Wire rail items
  document.querySelectorAll('.rail-item[data-view]').forEach((el) => {
    el.addEventListener('click', () => activate(el.dataset.view));
  });
  // Wire mobile nav
  document.querySelectorAll('.m-nav .item[data-view]').forEach((el) => {
    el.addEventListener('click', () => activate(el.dataset.view));
  });

  // Canvas header buttons
  document.querySelectorAll('.cc-btn[data-action]').forEach((b) => {
    b.addEventListener('click', () => {
      const a = b.dataset.action;
      if (a === 'reset-cam') {
        window.ConnectomeScene?.reset();
        window.FlyScene?.reset?.();
      }
      if (a === 'burst') window.ConnectomeScene?.pulseBurst(80);
      if (a === 'toggle-edges') {
        document.querySelectorAll('.cc-btn[data-action="toggle-edges"]').forEach(x => x.classList.toggle('active'));
      }
    });
  });

  // Initial
  activate('graph');

  // Live embodiment metrics ticker (cheap, always runs)
  const embRefs = {
    step: document.getElementById('emb-step'),
    wing: document.getElementById('emb-wing'),
    motor: document.getElementById('emb-motor'),
    sensory: document.getElementById('emb-sensory'),
    lat: document.getElementById('emb-lat'),
  };
  function fmt(n, unit, digits = 1) {
    return `${n.toFixed(digits)}<em> ${unit}</em>`;
  }
  let embT = 0;
  setInterval(() => {
    embT += 0.4;
    if (!embRefs.step) return;
    const step = 5.5 + Math.sin(embT * 0.7) * 0.6;
    const wing = 198 + Math.sin(embT * 1.3) * 3.2;
    const motor = 14.2 + Math.sin(embT * 0.5) * 2.1;
    const sensory = 22.8 + Math.sin(embT * 0.9 + 1) * 3.4;
    const lat = 3.8 + Math.sin(embT * 1.1) * 0.4;
    embRefs.step.innerHTML = fmt(step, 'Hz');
    embRefs.wing.innerHTML = fmt(wing, 'Hz', 0);
    embRefs.motor.innerHTML = fmt(motor, 'kHz');
    embRefs.sensory.innerHTML = fmt(sensory, 'kHz');
    embRefs.lat.innerHTML = fmt(lat, 'ms');
  }, 400);
})();
