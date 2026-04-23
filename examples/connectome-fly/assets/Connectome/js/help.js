// Connectome OS — help popover system + content library
// Provides ?-icon popovers explaining every metric, panel, and concept in plain English.

(function () {
  // ---- Help content library ----------------------------------------------
  // Each entry: { title, body } where body can contain <b>, <code>, <p>.
  const HELP = {
    // === VIEWS (left rail) ===
    view_structure: {
      title: 'Structure',
      body: `<p>The <b>static wiring diagram</b> — which neuron connects to which, and how strongly.</p>
        <p>No time, no firing — just the graph. Think of it as the map before any traffic flows.</p>
        <p>Source: <code>FlyWire v783</code>, ~139K typed neurons, 6.12M edges.</p>`
    },
    view_graph: {
      title: 'Graph — co-firing view',
      body: `<p>Same neurons as Structure, but colored by <b>what fires together</b> in a short rolling window.</p>
        <p>The highlighted boundary is the current <b>mincut</b> — the edges the system thinks would, if severed, split the network cleanest.</p>`
    },
    view_dynamics: {
      title: 'Dynamics',
      body: `<p>The live <b>simulator</b>. Each neuron is a leaky integrate-and-fire (LIF) unit; spikes travel along edges with per-synapse delays.</p>
        <p>Uses a <code>delivery wheel</code> (a ring of delay slots) and SIMD <code>f32x8</code> lanes for throughput.</p>`
    },
    view_motifs: {
      title: 'Motifs',
      body: `<p>Finds <b>recurring patterns</b> in the spike stream.</p>
        <p>Every 100ms window is embedded by a small attention encoder (SDPA), then indexed in an HNSW graph so any new window can retrieve its nearest neighbors in milliseconds.</p>`
    },
    view_causal: {
      title: 'Causal perturbation',
      body: `<p>Asks: "if we <b>cut these specific edges</b>, how much does behavior change, versus cutting the same number of <b>random</b> edges?"</p>
        <p>A large gap (measured in σ) is evidence the targeted edges were actually <b>causal</b>, not just correlated.</p>`
    },
    view_acceptance: {
      title: 'Acceptance suite',
      body: `<p>The <b>test battery</b> (AT-1 through AT-5) that defines whether a build is fit for use.</p>
        <p>Covers repeatability, motif emergence, structural & functional cuts, coherence prediction, and causal effect size.</p>`
    },
    view_embodiment: {
      title: 'Embodiment',
      body: `<p>Hooks the simulator to a <b>body</b> — currently a virtual fly.</p>
        <p>Sensory spikes come in from eyes and antennae; motor pool readouts drive legs (tripod gait, ~5 Hz) and wings (~200 Hz). Closed-loop latency is tracked live.</p>`
    },
    view_benchmarks: {
      title: 'Benchmarks',
      body: `<p>Throughput (spikes per second) vs mainstream simulators on matched networks.</p>
        <p>Same hardware (Ryzen, 1 thread, release build), same seed, same connectivity. Reported for both <b>sparse</b> and <b>saturated</b> firing regimes.</p>`
    },
    view_console: {
      title: 'Console',
      body: `<p>Raw engine output — init order, discovery log, test results, REPL.</p>
        <p>Useful when something's off and the panels aren't telling you why.</p>`
    },
    view_settings: {
      title: 'Settings',
      body: `<p>Session seed, engine flags, reproducibility. Everything that determines whether two runs produce the same spikes.</p>`
    },

    // === METRICS ===
    fiedler: {
      title: 'λ₂ — Fiedler value',
      body: `<p>The <b>algebraic connectivity</b> of the co-firing graph. Computed from the second-smallest eigenvalue of the graph Laplacian.</p>
        <p><b>Higher</b> = the network is well-connected, firing coherently. <b>Lower</b> = it's about to split into independent pieces.</p>
        <p>A sharp drop typically <b>precedes</b> a visible desynchronization by 50–80ms — our earliest warning signal.</p>`
    },
    mincut: {
      title: 'Mincut boundary',
      body: `<p>The smallest set of edges whose removal would disconnect one module from another.</p>
        <p>We track which pair of modules has the weakest link (<b>M0 ↔ M1</b> right now) and update it every few ms. That boundary is the target for causal tests.</p>`
    },
    ari: {
      title: 'ARI — Adjusted Rand Index',
      body: `<p>Measures how well our <b>discovered partition</b> matches a known ground-truth partition. <b>1.0</b> = exact match, <b>0</b> = random.</p>
        <p>0.78 vs the SBM hub assignment means we recover the intended modular structure with high fidelity.</p>`
    },
    l1_sep: {
      title: 'L1 separation',
      body: `<p>L1 distance between the average firing-rate vectors of two partitions.</p>
        <p>Higher = the two sides are doing <b>different things</b>, not just sitting on different synapses.</p>`
    },
    sigma_sep: {
      title: 'σ-separation',
      body: `<p>The number of standard deviations between the <b>targeted cut</b> effect and the <b>random cut</b> null distribution.</p>
        <p><b>&gt;3σ</b> — targeted edges carry specific causal load. <b>&lt;2σ</b> — no evidence beyond chance.</p>`
    },
    precision_at_5: {
      title: 'precision@5',
      body: `<p>Of the 5 nearest-neighbor motifs retrieved for a query window, how many share the <b>same ground-truth label</b>?</p>
        <p>Target for AC-2 is <b>0.80</b>. We're at 0.60 — motif labels are coherent but noisy.</p>`
    },
    throughput: {
      title: 'Throughput',
      body: `<p>Spikes delivered and integrated per wall-clock second on one CPU thread.</p>
        <p><code>sparse</code> = 10 Hz mean firing, fan-out 100. <code>saturated</code> = 50 Hz, fan-out 1000.</p>`
    },
    tick: {
      title: 'Simulation tick',
      body: `<p>Current simulated time, in milliseconds. Independent of wall clock — tick 1000 might take 40ms or 4s depending on the scenario.</p>`
    },
    throughput_stat: {
      title: 'Σ — live throughput',
      body: `<p>Total spikes emitted across the entire graph in the last second of simulated time.</p>`
    },
    loop_latency: {
      title: 'Closed-loop latency',
      body: `<p>Time from a <b>sensory spike</b> entering the network to the resulting <b>motor pool readout</b> reaching the body.</p>
        <p>Under 5ms is what a real fly achieves for evasive maneuvers.</p>`
    },
    tripod_gait: {
      title: 'Tripod gait',
      body: `<p>The standard 6-legged walking pattern: legs L1·R2·L3 lift together, alternating with R1·L2·R3.</p>
        <p>Emerges from a motor central pattern generator (CPG) we drive with cortical module M3.</p>`
    },
    wing_beat: {
      title: 'Wing beat',
      body: `<p>Flap frequency of the virtual wings, driven by a 200 Hz oscillator in the thoracic motor pool.</p>`
    },

    // === SCENARIOS ===
    scenario: {
      title: 'Scenario',
      body: `<p>Presets that reshape the input drive:</p>
        <p><b>NORMAL</b> — baseline Poisson drive, coherence stable.<br/>
        <b>SATURATED</b> — all modules hammered at 50 Hz; tests the wheel & SIMD under load.<br/>
        <b>FRAGMENT</b> — drive diverges between modules; λ₂ collapses on purpose so you can see the early-warning fire.</p>`
    },

    // === HEADER ACTIONS ===
    cut_boundary_toggle: {
      title: 'Cut boundary highlight',
      body: `<p>Toggles the lime-green highlighting of the current mincut edges in the 3D graph.</p>`
    },
    spike_overlay: {
      title: 'Spike burst',
      body: `<p>Triggers a visual pulse along boundary edges — useful for screenshots and for showing people what co-firing looks like.</p>`
    },
    camera_reset: {
      title: 'Reset camera',
      body: `<p>Returns the 3D view to its default angle and zoom. (Double-click the canvas does the same thing.)</p>`
    },

    // === ACCEPTANCE TESTS ===
    ac_1: { title: 'AC-1 Repeatability',
      body: `<p>Same seed + same inputs must produce <b>bit-identical</b> spike trains across machines.</p><p>Currently: 194,784 spikes reproduced exactly.</p>` },
    ac_2: { title: 'AC-2 Motif emergence',
      body: `<p>Recurring activity patterns must be retrievable by nearest-neighbor with precision@5 ≥ 0.80.</p><p>Currently at <b>0.60</b> — partial pass.</p>` },
    ac_3a: { title: 'AC-3a Structural cut',
      body: `<p>The mincut partition must match the ground-truth SBM structure with ARI ≥ 0.70.</p><p>Currently: <b>0.78</b> ✓.</p>` },
    ac_3b: { title: 'AC-3b Functional cut',
      body: `<p>L1 distance between the firing signatures of sensory-side vs motor-side partitions.</p><p>Currently: <b>0.41</b> ✓.</p>` },
    ac_4: { title: 'AC-4 Coherence lead',
      body: `<p>On ≥70% of 30 trials, the λ₂ dip must precede desync by ≥50ms.</p><p>Currently: <b>74%</b> ✓.</p>` },
    ac_5: { title: 'AC-5 Causal perturbation',
      body: `<p>Targeted mincut edge removal must separate from random removal by ≥3σ.</p><p>Currently: 5.55σ / 1.57σ — partial until random tightens.</p>` },

    // === DYNAMICS PANELS ===
    spike_raster: {
      title: 'Spike raster',
      body: `<p>Each row is a neuron; each dot is a spike. Time runs left to right over the last 240ms.</p>
        <p>Vertical bands = coordinated bursts. Diagonal streaks = traveling waves along the wiring.</p>`
    },
    system_state: {
      title: 'System state',
      body: `<p>Eleven named measurement discoveries that fire when specific regime changes happen (λ₂ collapse, motif re-index, module fragment, etc).</p>
        <p>Each active segment = one discovery currently armed.</p>`
    },

    // === TOPBAR ===
    engine: {
      title: 'Engine',
      body: `<p>The simulator build currently running. <code>lif-wheel-soa</code> means: leaky integrate-and-fire neurons, delivery wheel, struct-of-arrays memory layout.</p>`
    },
    breadcrumbs: {
      title: 'Session',
      body: `<p><b>tier-1</b> — execution mode. <b>fly-fixture-v783</b> — loaded connectivity graph. <b>session.0x…</b> — seed hash; identical seed reproduces everything exactly.</p>`
    },
  };

  // ---- Popover element ---------------------------------------------------
  const pop = document.createElement('div');
  pop.id = 'help-popover';
  pop.setAttribute('role', 'tooltip');
  pop.innerHTML = '<div class="hp-title"></div><div class="hp-body"></div><div class="hp-foot"><span>hover · click</span><span>? help</span></div>';
  document.body.appendChild(pop);
  const popTitle = pop.querySelector('.hp-title');
  const popBody = pop.querySelector('.hp-body');

  let hideTimer = null;
  let currentAnchor = null;

  function positionPop(anchor) {
    const r = anchor.getBoundingClientRect();
    // Default: below-right of anchor
    const pr = pop.getBoundingClientRect();
    let left = r.right + 10;
    let top = r.top + r.height / 2 - pr.height / 2;
    // If it would overflow the right edge, put it on the left
    if (left + pr.width > window.innerWidth - 10) {
      left = r.left - pr.width - 10;
    }
    // Clamp vertically
    if (top < 10) top = 10;
    if (top + pr.height > window.innerHeight - 10) top = window.innerHeight - pr.height - 10;
    pop.style.left = left + 'px';
    pop.style.top = top + 'px';
  }

  function showHelp(anchor, key) {
    const entry = HELP[key];
    if (!entry) return;
    clearTimeout(hideTimer);
    currentAnchor = anchor;
    popTitle.textContent = entry.title;
    popBody.innerHTML = entry.body;
    // Paint before measuring
    pop.style.left = '-9999px';
    pop.style.top = '-9999px';
    pop.classList.add('show');
    // Two frames so styles settle
    requestAnimationFrame(() => requestAnimationFrame(() => positionPop(anchor)));
    anchor.classList.add('open');
  }

  function hideHelp(immediate = false) {
    const go = () => {
      pop.classList.remove('show');
      if (currentAnchor) currentAnchor.classList.remove('open');
      currentAnchor = null;
    };
    clearTimeout(hideTimer);
    if (immediate) go();
    else hideTimer = setTimeout(go, 120);
  }

  // Keep visible while hovered
  pop.addEventListener('mouseenter', () => clearTimeout(hideTimer));
  pop.addEventListener('mouseleave', () => hideHelp());

  // Click-outside to close
  document.addEventListener('click', (e) => {
    if (currentAnchor && !currentAnchor.contains(e.target) && !pop.contains(e.target)) {
      hideHelp(true);
    }
  });

  // ---- Attach help behavior ---------------------------------------------
  function attach(el, key) {
    if (!el || el.dataset.helpAttached) return;
    el.dataset.helpAttached = '1';
    el.dataset.helpKey = key;
    el.addEventListener('mouseenter', () => showHelp(el, key));
    el.addEventListener('mouseleave', () => hideHelp());
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      if (currentAnchor === el) hideHelp(true);
      else showHelp(el, key);
    });
    el.addEventListener('focus', () => showHelp(el, key));
    el.addEventListener('blur', () => hideHelp());
  }

  // Create a help icon element
  function makeIcon(key, opts = {}) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'help-icon' + (opts.large ? ' lg' : '');
    b.setAttribute('aria-label', 'help · ' + (HELP[key]?.title || key));
    b.setAttribute('tabindex', '0');
    attach(b, key);
    return b;
  }

  // ---- Auto-wire: anything with data-help="key" gets icon behavior ------
  function scan(root = document) {
    root.querySelectorAll('[data-help]:not([data-help-attached])').forEach(el => {
      const key = el.dataset.help;
      if (!HELP[key]) return;
      el.dataset.helpAttached = '1';
      // If the element itself is already a trigger (kpi, panel-head title), just attach behavior
      if (el.classList.contains('help-icon') || el.classList.contains('rail-item') || el.tagName === 'BUTTON') {
        attach(el, key);
      } else if (el.dataset.helpIcon === 'inline') {
        // Append an inline icon
        el.appendChild(makeIcon(key));
      } else {
        // Default: attach hover behavior to the element itself
        attach(el, key);
      }
    });
  }

  // Expose
  window.ConnectomeHelp = { HELP, attach, makeIcon, scan, show: showHelp, hide: hideHelp };

  // Initial scan after load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => scan());
  } else {
    scan();
  }
})();
