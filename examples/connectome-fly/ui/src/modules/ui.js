// Connectome OS — UI panels + perturbation + motifs + IndexedDB + tweaks

(function () {
  // === IndexedDB for perturbation history ================================
  const DB_NAME = 'connectome-os';
  const DB_VERSION = 1;
  let db = null;
  const dbReq = indexedDB.open(DB_NAME, DB_VERSION);
  dbReq.onupgradeneeded = (e) => {
    const d = e.target.result;
    if (!d.objectStoreNames.contains('perturbations')) {
      d.createObjectStore('perturbations', { keyPath: 'id', autoIncrement: true });
    }
    if (!d.objectStoreNames.contains('settings')) {
      d.createObjectStore('settings', { keyPath: 'key' });
    }
  };
  dbReq.onsuccess = (e) => {
    db = e.target.result;
    loadSettings();
    loadPerturbations();
  };

  function putSetting(key, value) {
    if (!db) return;
    const tx = db.transaction('settings', 'readwrite');
    tx.objectStore('settings').put({ key, value });
  }
  function getSetting(key) {
    return new Promise((resolve) => {
      if (!db) return resolve(null);
      const tx = db.transaction('settings', 'readonly');
      const req = tx.objectStore('settings').get(key);
      req.onsuccess = () => resolve(req.result ? req.result.value : null);
    });
  }
  function putPerturbation(rec) {
    if (!db) return;
    const tx = db.transaction('perturbations', 'readwrite');
    tx.objectStore('perturbations').add({ ...rec, ts: Date.now() });
  }
  function allPerturbations() {
    return new Promise((resolve) => {
      if (!db) return resolve([]);
      const tx = db.transaction('perturbations', 'readonly');
      const req = tx.objectStore('perturbations').getAll();
      req.onsuccess = () => resolve(req.result || []);
    });
  }

  async function loadSettings() {
    const cut = await getSetting('cut');
    if (cut) {
      window.ConnectomeScene?.setCut(cut[0], cut[1]);
      window.setCutModules?.(cut[0], cut[1]);
      updateCutUI(cut[0], cut[1]);
    }
    const scenario = await getSetting('scenario');
    if (scenario) {
      window.Dynamics?.setScenario(scenario);
      updateScenarioUI(scenario);
    }
  }

  async function loadPerturbations() {
    const recs = await allPerturbations();
    if (recs.length === 0) return;
    const list = document.getElementById('perturb-history');
    if (!list) return;
    list.innerHTML = '';
    recs.slice(-5).reverse().forEach((r) => {
      const el = document.createElement('div');
      el.className = 'cut-row';
      el.innerHTML = `
        <span class="idx">#${String(r.id).padStart(3, '0')}</span>
        <span class="edge">cut M${r.from}→M${r.to}, k=${r.k}</span>
        <span class="w">${r.sigma.toFixed(2)}σ</span>
        <span class="${r.status}">${r.status}</span>
      `;
      list.appendChild(el);
    });
  }

  // === CUT / PARTITION UI ================================================
  let curCut = [0, 1];
  const cutSelect = document.getElementById('cut-select');
  function updateCutUI(from, to) {
    curCut = [from, to];
    const labelEl = document.getElementById('cut-label');
    if (labelEl) labelEl.textContent = `M${from} → M${to}`;
    const kEl = document.getElementById('cut-boundary-count');
    const stats = window.ConnectomeScene?.stats();
    if (kEl && stats) kEl.textContent = String(stats.boundary);
  }

  document.querySelectorAll('.cut-row[data-cut]').forEach((row) => {
    row.addEventListener('click', () => {
      const [a, b] = row.dataset.cut.split('-').map(Number);
      document.querySelectorAll('.cut-row[data-cut]').forEach((r) => r.classList.remove('sel'));
      row.classList.add('sel');
      window.ConnectomeScene?.setCut(a, b);
      window.setCutModules?.(a, b);
      updateCutUI(a, b);
      putSetting('cut', [a, b]);
    });
  });

  // === Motif panel =======================================================
  const motifEls = document.querySelectorAll('.motif');
  motifEls.forEach((m) => {
    m.addEventListener('click', () => {
      motifEls.forEach((o) => o.classList.remove('sel'));
      m.classList.add('sel');
    });
    // paint a tiny raster into the motif preview
    const c = m.querySelector('canvas');
    if (c) drawMotifRaster(c, m.dataset.seed || '1');
  });

  function drawMotifRaster(canvas, seed) {
    const ctx = canvas.getContext('2d');
    const r = canvas.getBoundingClientRect();
    canvas.width = r.width * devicePixelRatio;
    canvas.height = r.height * devicePixelRatio;
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);
    let s = Number(seed) * 91237;
    const ROWS = 8;
    for (let c = 0; c < 60; c++) {
      for (let row = 0; row < ROWS; row++) {
        s = (s * 1664525 + 1013904223) >>> 0;
        if ((s & 0xff) < 30 + (row % 3) * 10) {
          ctx.fillStyle = row < 4 ? 'rgba(184,255,60,0.9)' : 'rgba(174,184,177,0.55)';
          const x = (c / 60) * W;
          const y = (row / ROWS) * H;
          ctx.fillRect(x, y, Math.max(1, W / 60 - 1), Math.max(1, H / ROWS - 1));
        }
      }
    }
  }

  // === Perturbation controls ============================================
  const slider = document.getElementById('slider-k');
  const kLabel = document.getElementById('slider-k-val');
  if (slider && kLabel) {
    slider.addEventListener('input', () => {
      kLabel.textContent = slider.value;
    });
  }

  const runBtn = document.getElementById('run-perturb');
  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      runBtn.disabled = true;
      runBtn.textContent = 'Running 30 trials…';
      document.getElementById('sigma-out').style.display = 'none';
      const k = Number(slider.value);
      // Visual: burst pulses, temporarily reduce health of cut modules
      window.ConnectomeScene?.pulseBurst(60);
      const base = [1, 1, 1, 1];
      const cut = base.slice();
      cut[curCut[0]] = 0.25; cut[curCut[1]] = 0.35;
      window.Dynamics?.setHealth(cut);

      // Animate divergence bars over ~2.2s
      const targetBarEl = document.getElementById('bar-targeted');
      const randomBarEl = document.getElementById('bar-random');
      const targetValEl = document.getElementById('val-targeted');
      const randomValEl = document.getElementById('val-random');

      const finalSigmaCut = 4.8 + Math.random() * 1.6;     // ~5.5
      const finalSigmaRand = 0.9 + Math.random() * 0.9;    // ~1.5
      const finalMeanCut = 0.32 + Math.random() * 0.2;
      const finalMeanRand = 0.07 + Math.random() * 0.07;

      const steps = 44;
      for (let i = 1; i <= steps; i++) {
        const t = i / steps;
        const ease = 1 - Math.pow(1 - t, 3);
        targetBarEl.style.width = (ease * Math.min(100, finalSigmaCut / 7 * 100)) + '%';
        randomBarEl.style.width = (ease * Math.min(100, finalSigmaRand / 7 * 100)) + '%';
        targetValEl.textContent = (ease * finalSigmaCut).toFixed(2) + 'σ';
        randomValEl.textContent = (ease * finalSigmaRand).toFixed(2) + 'σ';
        await sleep(50);
      }

      // Record result
      const sigmaSep = finalSigmaCut - finalSigmaRand;
      const status = finalSigmaCut >= 5 ? 'pass' : 'partial';

      document.getElementById('sigma-out').style.display = 'block';
      document.getElementById('sigma-sep-val').textContent = sigmaSep.toFixed(2) + 'σ';
      document.getElementById('sigma-conclusion').textContent =
        finalSigmaCut >= 5 ?
        'Targeted cut hits 5σ threshold — structural causality confirmed.' :
        'Targeted cut > random by ' + sigmaSep.toFixed(2) + 'σ.';

      putPerturbation({ from: curCut[0], to: curCut[1], k, sigma: finalSigmaCut, status });
      loadPerturbations();

      // Restore
      window.Dynamics?.setHealth([1, 1, 1, 1]);
      runBtn.disabled = false;
      runBtn.textContent = 'Run 30 paired trials';
    });
  }

  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  // === Scenario buttons ==================================================
  function updateScenarioUI(name) {
    document.querySelectorAll('[data-scenario]').forEach((b) => {
      b.classList.toggle('active', b.dataset.scenario === name);
    });
  }

  document.querySelectorAll('[data-scenario]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const s = btn.dataset.scenario;
      window.Dynamics?.setScenario(s);
      updateScenarioUI(s);
      putSetting('scenario', s);
      if (s === 'fragmenting') {
        setTimeout(() => {
          const el = document.getElementById('fiedler-alert');
          if (el) el.classList.add('on');
        }, 800);
        setTimeout(() => {
          const el = document.getElementById('fiedler-alert');
          if (el) el.classList.remove('on');
        }, 8000);
      }
    });
  });

  // Play/pause
  let playing = true;
  document.getElementById('play-toggle')?.addEventListener('click', (e) => {
    playing = !playing;
    if (playing) window.Dynamics?.play();
    else window.Dynamics?.pause();
    e.currentTarget.classList.toggle('active', playing);
    e.currentTarget.querySelector('.txt').textContent = playing ? 'PAUSE' : 'PLAY';
  });

  // === Fiedler alert ====================================================
  window.addEventListener('fiedler-alert', (e) => {
    const el = document.getElementById('fiedler-alert');
    if (el) {
      el.classList.add('on');
      setTimeout(() => el.classList.remove('on'), 4500);
    }
  });

  // === Tweaks ===========================================================
  const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
    "accent": "#B8FF3C",
    "cameraDrift": true,
    "fogDensity": 0.6
  }/*EDITMODE-END*/;

  window.addEventListener('message', (e) => {
    if (e.data?.type === '__activate_edit_mode') document.getElementById('tweaks')?.classList.add('open');
    if (e.data?.type === '__deactivate_edit_mode') document.getElementById('tweaks')?.classList.remove('open');
  });
  window.parent?.postMessage({ type: '__edit_mode_available' }, '*');

  // Accent color swatches
  document.querySelectorAll('.sw-btn[data-color]').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.sw-btn').forEach((o) => o.classList.remove('sel'));
      b.classList.add('sel');
      const c = b.dataset.color;
      document.documentElement.style.setProperty('--signal', c);
      window.parent?.postMessage({ type: '__edit_mode_set_keys', edits: { accent: c } }, '*');
    });
  });

  // Scenario select in tweaks
  document.getElementById('tweak-density')?.addEventListener('change', (e) => {
    document.documentElement.style.setProperty('--ambient-opacity', e.target.value);
  });

  // === Tweaks collapse ==================================================
  const tweaksPanel = document.getElementById('tweaks');
  const tweaksToggle = document.getElementById('tweaks-toggle');
  const COLLAPSE_KEY = 'tweaks-collapsed';
  // Restore state
  try {
    if (localStorage.getItem(COLLAPSE_KEY) === '1') {
      tweaksPanel?.classList.add('collapsed');
    }
  } catch (e) {}
  tweaksToggle?.addEventListener('click', () => {
    const collapsed = tweaksPanel.classList.toggle('collapsed');
    try { localStorage.setItem(COLLAPSE_KEY, collapsed ? '1' : '0'); } catch (e) {}
  });
})();
