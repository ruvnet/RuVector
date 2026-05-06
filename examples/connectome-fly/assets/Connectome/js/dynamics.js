// Connectome OS — Dynamics layer (spike raster + Fiedler + motifs)
// Runs a small LIF-ish simulation in a Web Worker; renders rasters on 2D canvases.

(function () {
  // === Spawn worker for spike generation =================================
  const workerSrc = `
    // Minimal LIF-style spike generator with 4 modules and state machine
    let running = true;
    let modeIdx = 0;         // 0 normal, 1 saturating, 2 fragmenting
    const MODES = ['normal', 'saturated', 'fragmenting'];
    const N = 208;           // visible neurons
    const MODS = 4;
    const perMod = N / MODS;
    const V = new Float32Array(N);
    const refr = new Int16Array(N);
    const rates = new Float32Array(N);

    // Module coupling weights (used to derive a co-firing signal)
    const modHealth = new Float32Array(MODS); // 1 = coherent, 0 = cut
    for (let i=0;i<MODS;i++) modHealth[i] = 1.0;

    let tick = 0;
    let fragmentAt = -1;   // when to inject fragmentation
    let collapseAt = -1;   // when behavior "fails"

    function setScenario(name) {
      modeIdx = MODES.indexOf(name);
      if (modeIdx === -1) modeIdx = 0;
      tick = 0;
      if (modeIdx === 2) { fragmentAt = 30; collapseAt = 80; }
      else { fragmentAt = -1; collapseAt = -1; }
    }
    setScenario('normal');

    function rand() { return Math.random(); }

    function stepOnce() {
      const mode = MODES[modeIdx];
      const spikes = [];
      // firing probability per module
      const baseP = new Float32Array(MODS);
      for (let m=0;m<MODS;m++) {
        if (mode === 'normal') baseP[m] = 0.06 + 0.02 * Math.sin(tick * 0.05 + m);
        else if (mode === 'saturated') baseP[m] = 0.35 + 0.05 * Math.sin(tick * 0.08 + m);
        else {
          // Fragmenting: module 0 starts desynchronizing ~fragmentAt
          let p = 0.08;
          if (tick >= fragmentAt && m === 0) p = 0.02 + Math.random() * 0.1;
          if (tick >= collapseAt) p = 0.01 + Math.random() * 0.03;
          baseP[m] = p;
        }
      }

      // Per-module coherence: measure of how many spike together this tick
      const modSpikes = new Int32Array(MODS);

      for (let i = 0; i < N; i++) {
        if (refr[i] > 0) { refr[i]--; continue; }
        const m = (i / perMod) | 0;
        const p = baseP[m] * modHealth[m];
        if (rand() < p) {
          spikes.push(i);
          refr[i] = 3;
          modSpikes[m]++;
        }
      }

      // Fiedler proxy: inverse of average intra-module synchrony variance
      let sum = 0, sum2 = 0;
      for (let m=0;m<MODS;m++) {
        const r = modSpikes[m] / perMod;
        sum += r; sum2 += r*r;
      }
      const mean = sum / MODS;
      const varc = sum2 / MODS - mean * mean;
      // Fiedler-ish value: coherent clusters → high; collapsing → low.
      // baseline around 0.3–0.5, drops sharply during fragmentation
      let fiedler = 0.08 + Math.max(0, mean * 1.1) - Math.abs(varc - 0.005) * 6;
      if (mode === 'fragmenting' && tick >= fragmentAt) {
        fiedler *= Math.max(0.15, 1 - (tick - fragmentAt) / 80);
      }
      if (mode === 'saturated') fiedler *= 1.4;
      fiedler = Math.max(0.02, Math.min(0.65, fiedler));

      tick++;
      return { spikes, fiedler, tick, modSpikes: Array.from(modSpikes) };
    }

    self.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === 'setScenario') setScenario(msg.scenario);
      else if (msg.type === 'setHealth') {
        for (let m=0;m<MODS;m++) modHealth[m] = msg.health[m];
      }
      else if (msg.type === 'pause') running = false;
      else if (msg.type === 'play') running = true;
    };

    // Run ~50 Hz
    setInterval(() => {
      if (!running) return;
      const out = stepOnce();
      self.postMessage(out);
    }, 40);
  `;

  const blob = new Blob([workerSrc], { type: 'application/javascript' });
  const worker = new Worker(URL.createObjectURL(blob));

  // === Raster rendering ==================================================
  const raster = document.getElementById('raster-canvas');
  const mRaster = document.getElementById('m-raster-canvas'); // optional mobile
  const rctx = raster ? raster.getContext('2d') : null;

  // Scrolling buffer: columns of spike events, 240 columns wide.
  const COLS = 240;
  const ROWS = 208;
  let col = 0;
  const buffer = new Uint8Array(COLS * ROWS);

  function drawRaster() {
    if (!rctx) return;
    const r = raster.getBoundingClientRect();
    if (raster.width !== r.width * devicePixelRatio || raster.height !== r.height * devicePixelRatio) {
      raster.width = r.width * devicePixelRatio;
      raster.height = r.height * devicePixelRatio;
    }
    const W = raster.width, H = raster.height;
    rctx.fillStyle = '#04080A';
    rctx.fillRect(0, 0, W, H);

    // module dividers
    rctx.strokeStyle = 'rgba(255,255,255,0.04)';
    rctx.lineWidth = 1;
    for (let m = 1; m < 4; m++) {
      const y = (ROWS / 4) * m / ROWS * H;
      rctx.beginPath(); rctx.moveTo(0, y); rctx.lineTo(W, y); rctx.stroke();
    }

    const cw = W / COLS;
    const ch = H / ROWS;

    for (let c = 0; c < COLS; c++) {
      const colIdx = (col + c) % COLS;
      for (let row = 0; row < ROWS; row++) {
        const v = buffer[colIdx * ROWS + row];
        if (v) {
          const m = (row / (ROWS / 4)) | 0;
          const isCut = (m === CUR_CUT[0] || m === CUR_CUT[1]);
          // Signal for cut modules, dim white for others
          if (isCut) {
            rctx.fillStyle = m === CUR_CUT[0] ? 'rgba(184,255,60,0.95)' : 'rgba(124,255,122,0.9)';
          } else {
            rctx.fillStyle = 'rgba(174,184,177,0.55)';
          }
          rctx.fillRect(c * cw, row * ch, Math.max(1, cw - 0.5), Math.max(1, ch - 0.3));
        }
      }
    }
  }

  let CUR_CUT = [0, 1];
  window.setCutModules = (a, b) => { CUR_CUT = [a, b]; };

  // === Fiedler rendering =================================================
  const fc = document.getElementById('fiedler-canvas');
  const fctx = fc ? fc.getContext('2d') : null;
  const FHIST = 180;
  const fHist = new Float32Array(FHIST);
  let fHead = 0;
  let fVal = 0.35;
  const FIEDLER_THRESHOLD = 0.18;
  let fiedlerAlerted = false;

  function drawFiedler() {
    if (!fctx) return;
    const r = fc.getBoundingClientRect();
    if (fc.width !== r.width * devicePixelRatio || fc.height !== r.height * devicePixelRatio) {
      fc.width = r.width * devicePixelRatio;
      fc.height = r.height * devicePixelRatio;
    }
    const W = fc.width, H = fc.height;
    fctx.fillStyle = 'rgba(0,0,0,0)';
    fctx.clearRect(0, 0, W, H);

    // Threshold line
    const yThr = H - (FIEDLER_THRESHOLD / 0.7) * H;
    fctx.setLineDash([3, 3]);
    fctx.strokeStyle = 'rgba(246,196,69,0.45)';
    fctx.lineWidth = 1;
    fctx.beginPath(); fctx.moveTo(0, yThr); fctx.lineTo(W, yThr); fctx.stroke();
    fctx.setLineDash([]);

    // Label threshold
    fctx.fillStyle = 'rgba(246,196,69,0.6)';
    fctx.font = `${9 * devicePixelRatio}px "JetBrains Mono", monospace`;
    fctx.fillText('fragility λ₂ < 0.18', 6, yThr - 4);

    // Fill area
    fctx.beginPath();
    fctx.moveTo(0, H);
    for (let i = 0; i < FHIST; i++) {
      const v = fHist[(fHead + i) % FHIST];
      const x = (i / (FHIST - 1)) * W;
      const y = H - (v / 0.7) * H;
      fctx.lineTo(x, y);
    }
    fctx.lineTo(W, H);
    const grad = fctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(184,255,60,0.28)');
    grad.addColorStop(1, 'rgba(184,255,60,0)');
    fctx.fillStyle = grad;
    fctx.fill();

    // Stroke
    fctx.beginPath();
    for (let i = 0; i < FHIST; i++) {
      const v = fHist[(fHead + i) % FHIST];
      const x = (i / (FHIST - 1)) * W;
      const y = H - (v / 0.7) * H;
      if (i === 0) fctx.moveTo(x, y); else fctx.lineTo(x, y);
    }
    fctx.lineWidth = 1.5 * devicePixelRatio;
    fctx.strokeStyle = fVal < FIEDLER_THRESHOLD ? '#F6C445' : '#B8FF3C';
    fctx.shadowBlur = 8;
    fctx.shadowColor = fVal < FIEDLER_THRESHOLD ? 'rgba(246,196,69,0.5)' : 'rgba(184,255,60,0.5)';
    fctx.stroke();
    fctx.shadowBlur = 0;
  }

  // === Mobile fiedler canvas ============================================
  const mfc = document.getElementById('m-fiedler-canvas');
  const mfctx = mfc ? mfc.getContext('2d') : null;
  function drawMobileFiedler() {
    if (!mfctx) return;
    const r = mfc.getBoundingClientRect();
    if (mfc.width !== r.width * devicePixelRatio || mfc.height !== r.height * devicePixelRatio) {
      mfc.width = r.width * devicePixelRatio;
      mfc.height = r.height * devicePixelRatio;
    }
    const W = mfc.width, H = mfc.height;
    mfctx.clearRect(0, 0, W, H);
    mfctx.beginPath();
    mfctx.moveTo(0, H);
    for (let i = 0; i < FHIST; i++) {
      const v = fHist[(fHead + i) % FHIST];
      const x = (i / (FHIST - 1)) * W;
      const y = H - (v / 0.7) * H;
      mfctx.lineTo(x, y);
    }
    mfctx.lineTo(W, H);
    const grad = mfctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(184,255,60,0.3)');
    grad.addColorStop(1, 'rgba(184,255,60,0)');
    mfctx.fillStyle = grad;
    mfctx.fill();
    mfctx.beginPath();
    for (let i = 0; i < FHIST; i++) {
      const v = fHist[(fHead + i) % FHIST];
      const x = (i / (FHIST - 1)) * W;
      const y = H - (v / 0.7) * H;
      if (i === 0) mfctx.moveTo(x, y); else mfctx.lineTo(x, y);
    }
    mfctx.lineWidth = 2 * devicePixelRatio;
    mfctx.strokeStyle = '#B8FF3C';
    mfctx.shadowBlur = 10;
    mfctx.shadowColor = 'rgba(184,255,60,0.6)';
    mfctx.stroke();
    mfctx.shadowBlur = 0;
  }

  // === Receive spikes from worker =======================================
  let spikeBudget = 0;
  worker.onmessage = (e) => {
    const { spikes, fiedler, tick, modSpikes } = e.data;
    // write into buffer at current col
    const colBase = col * ROWS;
    for (let i = 0; i < ROWS; i++) buffer[colBase + i] = 0;
    for (let s = 0; s < spikes.length; s++) buffer[colBase + spikes[s]] = 1;
    col = (col + 1) % COLS;
    spikeBudget += spikes.length;

    // Fiedler history
    fVal = fiedler;
    fHist[fHead] = fiedler;
    fHead = (fHead + 1) % FHIST;

    if (fVal < FIEDLER_THRESHOLD && !fiedlerAlerted) {
      fiedlerAlerted = true;
      dispatchEvent(new CustomEvent('fiedler-alert', { detail: { value: fVal } }));
    }
    if (fVal > 0.25) fiedlerAlerted = false;

    // expose to UI
    window._fiedler = fiedler;
    window._modSpikes = modSpikes;
    window._tick = tick;
  };

  // === UI render tick ====================================================
  let spikesPerSec = 0;
  let lastNow = performance.now();
  function uiTick() {
    drawRaster();
    drawFiedler();
    drawMobileFiedler();

    const now = performance.now();
    if (now - lastNow > 1000) {
      spikesPerSec = spikeBudget;
      spikeBudget = 0;
      lastNow = now;
    }
    // Update Fiedler hero text
    const heroEl = document.getElementById('fiedler-hero');
    if (heroEl) heroEl.textContent = fVal.toFixed(3);
    const mHeroEl = document.getElementById('m-fiedler-hero');
    if (mHeroEl) mHeroEl.textContent = fVal.toFixed(3);

    const deltaEl = document.getElementById('fiedler-delta');
    if (deltaEl) {
      const prev = fHist[(fHead + FHIST - 30) % FHIST];
      const d = fVal - prev;
      deltaEl.textContent = (d >= 0 ? '+' : '') + d.toFixed(3);
      deltaEl.className = 'delta ' + (d >= 0 ? 'pos' : 'neg');
    }

    // Throughput
    const throughputEl = document.getElementById('stat-throughput');
    if (throughputEl) throughputEl.textContent = (spikesPerSec * 1.0).toLocaleString() + ' sp/s';
    const tickEl = document.getElementById('stat-tick');
    if (tickEl) tickEl.textContent = 't=' + (window._tick || 0);

    requestAnimationFrame(uiTick);
  }
  requestAnimationFrame(uiTick);

  // === Public API ========================================================
  window.Dynamics = {
    setScenario(name) { worker.postMessage({ type: 'setScenario', scenario: name }); },
    setHealth(arr) { worker.postMessage({ type: 'setHealth', health: arr }); },
    pause() { worker.postMessage({ type: 'pause' }); },
    play() { worker.postMessage({ type: 'play' }); },
    getFiedler() { return fVal; }
  };
})();
