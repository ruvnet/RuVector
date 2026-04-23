// Connectome OS — Dedicated Fly Simulation view.
//
// A standalone view (data-view="fly-sim") that reuses the procedural
// 3D fly in `fly.js` and wires it to the live spike data from
// `dynamics.js`. The view is a persistent full-canvas overlay inside
// .canvas-wrap; visibility is toggled when the rail item is activated,
// so the Three.js renderer stays warm and doesn't re-initialise on
// every view switch.
//
// Live-data mapping:
//   sensory in  →  sensory-burst pulse into FlyScene (antennae / eyes)
//   motor out   →  wing-beat + leg-step frequency (derived from spike rate)
//   fiedler     →  global body tint (coherent green ↔ fragmenting amber)
//
// When the SSE stream isn't connected and the JS mock is driving the
// raster, the same window._fiedler / window._real_spikes_total globals
// still update — so this view works on GitHub Pages too.

(function () {
  const W = window;

  function ensureHost() {
    let host = document.getElementById('fly-sim-root');
    if (host) return host;
    const wrap = document.querySelector('.canvas-wrap');
    if (!wrap) return null;
    host = document.createElement('div');
    host.id = 'fly-sim-root';
    host.className = 'fly-sim-root';
    host.innerHTML = `
      <div class="fs-head">
        <div class="fs-title">Fly simulation <span class="fs-sub">· real-time embodiment</span></div>
        <div class="fs-scenarios" role="toolbar" aria-label="Scenario">
          <button data-fly-scenario="normal"      class="fs-pill active">Normal</button>
          <button data-fly-scenario="saturated"   class="fs-pill">Saturated</button>
          <button data-fly-scenario="fragmenting" class="fs-pill">Fragmenting</button>
        </div>
      </div>
      <div class="fs-body">
        <div class="fs-stage" id="fly-sim-stage"></div>
        <aside class="fs-side">
          <div class="fs-card">
            <div class="fs-k">live source</div>
            <div class="fs-v" id="fs-src">–</div>
            <div class="fs-hint">"real" = Rust backend · "mock" = JS fallback</div>
          </div>
          <div class="fs-card">
            <div class="fs-k">sim clock</div>
            <div class="fs-v tnum" id="fs-clock">0 ms</div>
            <div class="fs-bar"><i id="fs-clock-bar" style="width:0%"></i></div>
          </div>
          <div class="fs-card">
            <div class="fs-k">spikes total</div>
            <div class="fs-v tnum" id="fs-spikes">0</div>
            <div class="fs-hint">counter on the live engine</div>
          </div>
          <div class="fs-card">
            <div class="fs-k">spike rate (1 s window)</div>
            <div class="fs-v tnum" id="fs-rate">0 <em>sp/s</em></div>
            <div class="fs-bar"><i id="fs-rate-bar" style="width:0%"></i></div>
          </div>
          <div class="fs-card">
            <div class="fs-k">wing beat</div>
            <div class="fs-v tnum" id="fs-wing">0 <em>Hz</em></div>
            <div class="fs-bar"><i id="fs-wing-bar" style="width:0%"></i></div>
          </div>
          <div class="fs-card">
            <div class="fs-k">fiedler λ₂</div>
            <div class="fs-v tnum" id="fs-fiedler">–</div>
            <div class="fs-hint" id="fs-fiedler-hint">coherence-collapse detector</div>
          </div>
          <div class="fs-card fs-card-dim">
            The fly body is procedural, not a render of a real fly —
            it translates live spike rates into motor behaviour (wing
            beat, leg tripod, antenna twitch) so you can see the engine
            driving the body in real time.
          </div>
        </aside>
      </div>
    `;
    wrap.appendChild(host);
    return host;
  }

  function mount() {
    const host = ensureHost();
    if (!host) return;
    const stage = host.querySelector('#fly-sim-stage');

    // Lazy-create the FlyScene when the view is first shown, so the
    // Three.js renderer + WebGL context aren't allocated until needed.
    let fly = null;
    function ensureFly() {
      if (!fly && W.FlyScene && stage) {
        fly = W.FlyScene.create(stage);
      }
      return fly;
    }

    function setVisible(on) {
      host.classList.toggle('active', !!on);
      if (on) {
        ensureFly();
        // If a FlyScene's internal resize hook exists, trigger it.
        W.dispatchEvent(new Event('resize'));
      }
    }

    // Rail activation — hook after nav.js has wired its clicks.
    document.querySelectorAll('[data-view="fly-sim"]').forEach((el) => {
      el.addEventListener('click', () => setVisible(true));
    });
    document.querySelectorAll('.rail-item[data-view], .m-nav .item[data-view]').forEach((el) => {
      el.addEventListener('click', () => {
        if (el.dataset.view !== 'fly-sim') setVisible(false);
      });
    });

    // Scenario pills — forward to the dynamics module (mock worker path).
    host.querySelectorAll('[data-fly-scenario]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const name = btn.dataset.flyScenario;
        host.querySelectorAll('[data-fly-scenario]').forEach((b) =>
          b.classList.toggle('active', b === btn)
        );
        W.Dynamics?.setScenario?.(name);
      });
    });

    // Live-readout update loop — reads the same globals dynamics.js
    // writes (window._real_spikes_total, _fiedler, _tick, _sim_ms).
    let prevTotal = 0;
    let prevT = performance.now();
    let rateHz = 0;
    let wingHz = 0;
    function tick() {
      const total = W._real_spikes_total || 0;
      const now = performance.now();
      const dt = Math.max(1e-3, (now - prevT) / 1000);
      if (dt >= 0.25) {
        const delta = Math.max(0, total - prevTotal);
        rateHz = delta / dt;
        prevTotal = total;
        prevT = now;
        // Wing-beat proxy: map spike-rate log to ~0–220 Hz.
        wingHz = Math.min(220, 30 + Math.log1p(rateHz) * 18);
        if (fly?.setWingHz) fly.setWingHz(wingHz);
        // Sensory burst proxy: sudden jumps in rate drive antennae.
        const burst = Math.min(1, Math.log1p(delta) / 10);
        fly?.setSensoryBurst?.(burst);
      }
      const srcEl = host.querySelector('#fs-src');
      if (srcEl) {
        const src = W._source || (W.Dynamics?.isMock?.() ? 'mock' : 'pending');
        srcEl.textContent = src;
        srcEl.dataset.src = src;
      }
      const clockEl = host.querySelector('#fs-clock');
      if (clockEl) {
        const t = W._sim_ms ?? W._tick ?? 0;
        clockEl.textContent = Number(t).toLocaleString() + ' ms';
      }
      const clockBar = host.querySelector('#fs-clock-bar');
      if (clockBar) {
        const pct = Math.min(100, ((W._tick || 0) % 10_000) / 100);
        clockBar.style.width = pct.toFixed(0) + '%';
      }
      const spEl = host.querySelector('#fs-spikes');
      if (spEl) spEl.textContent = total.toLocaleString();
      const rateEl = host.querySelector('#fs-rate');
      if (rateEl) rateEl.innerHTML = Math.round(rateHz).toLocaleString() + ' <em>sp/s</em>';
      const rateBar = host.querySelector('#fs-rate-bar');
      if (rateBar) {
        const pct = Math.min(100, Math.log1p(rateHz) * 10);
        rateBar.style.width = pct.toFixed(0) + '%';
      }
      const wingEl = host.querySelector('#fs-wing');
      if (wingEl) wingEl.innerHTML = Math.round(wingHz) + ' <em>Hz</em>';
      const wingBar = host.querySelector('#fs-wing-bar');
      if (wingBar) wingBar.style.width = Math.min(100, (wingHz / 220) * 100).toFixed(0) + '%';
      const fEl = host.querySelector('#fs-fiedler');
      const fiedler = W._fiedler;
      if (fEl) {
        fEl.textContent = Number.isFinite(fiedler) ? fiedler.toFixed(3) : '–';
      }
      const fHint = host.querySelector('#fs-fiedler-hint');
      if (fHint && Number.isFinite(fiedler)) {
        fHint.textContent = fiedler < 0.18
          ? 'fragmenting — below 0.18 collapse threshold'
          : fiedler < 0.3
          ? 'drifting — monitor'
          : 'stable — coherent co-firing';
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
