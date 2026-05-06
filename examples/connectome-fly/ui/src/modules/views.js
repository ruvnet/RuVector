// Connectome OS — view content overlays
// Each nav view gets a distinct content panel that overlays the canvas.

(function () {
  const W = window;

  // Build content for each view
  const CONTENT = {
    structure: () => `
      <div class="vc-grid">
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">S1</span>Typed directed graph</div>
          <div class="vc-stat-row">
            <div class="vc-stat"><div class="k">nodes</div><div class="v">139,255</div></div>
            <div class="vc-stat"><div class="k">edges</div><div class="v">6.12M</div></div>
            <div class="vc-stat"><div class="k">avg deg</div><div class="v">44.0</div></div>
            <div class="vc-stat"><div class="k">density</div><div class="v">3.2e-5</div></div>
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">S2</span>Cell-type typology</div>
          <div class="vc-bars">
            <div class="vc-bar"><span>Kenyon cells</span><i style="--w:82%"></i><b>42,321</b></div>
            <div class="vc-bar"><span>Projection neurons</span><i style="--w:54%"></i><b>28,110</b></div>
            <div class="vc-bar"><span>Local interneurons</span><i style="--w:41%"></i><b>21,480</b></div>
            <div class="vc-bar"><span>Motor neurons</span><i style="--w:18%"></i><b>9,215</b></div>
            <div class="vc-bar"><span>Sensory neurons</span><i style="--w:33%"></i><b>17,602</b></div>
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">S3</span>Adjacency provenance</div>
          <div class="vc-list">
            <div class="vc-item"><span class="dot ok"></span>FlyWire v783 · proofread · 2025-11-08</div>
            <div class="vc-item"><span class="dot ok"></span>Synaptic weights · cleft area · μ=82 nm²</div>
            <div class="vc-item"><span class="dot warn"></span>Gap junctions · heuristic · 11% coverage</div>
          </div>
        </div>
      </div>`,

    dynamics: () => `
      <div class="vc-grid">
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">D1</span>LIF engine state</div>
          <div class="vc-stat-row">
            <div class="vc-stat"><div class="k">dt</div><div class="v">0.1<em>ms</em></div></div>
            <div class="vc-stat"><div class="k">wheel</div><div class="v">128<em>slots</em></div></div>
            <div class="vc-stat"><div class="k">spikes/s</div><div class="v">1.84M</div></div>
            <div class="vc-stat"><div class="k">cpu</div><div class="v">47<em>%</em></div></div>
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">D2</span>SIMD lane occupancy</div>
          <div class="vc-lanes">
            ${Array.from({length:8}).map((_,i)=>`<div class="lane" style="--h:${40+Math.random()*55}%"><b>λ${i}</b></div>`).join('')}
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">D3</span>Delivery queue</div>
          <div class="vc-flow">
            <div class="flow-step"><b>emit</b><span>1.84M/s</span></div>
            <div class="flow-arrow"></div>
            <div class="flow-step"><b>route</b><span>delay bins</span></div>
            <div class="flow-arrow"></div>
            <div class="flow-step"><b>deliver</b><span>128 slots</span></div>
            <div class="flow-arrow"></div>
            <div class="flow-step"><b>integrate</b><span>f32x8</span></div>
          </div>
        </div>
      </div>`,

    motifs: () => `
      <div class="vc-grid">
        <div class="vc-card wide">
          <div class="vc-head"><span class="vc-num">M1</span>Query · 100ms spike window<button class="help-icon" data-help="view_motifs"></button></div>
          <div class="vc-query">
            <div class="qgrid">
              ${Array.from({length:40}).map(()=>{
                const on = Math.random() > 0.72;
                return `<i class="${on?'on':''}"></i>`;
              }).join('')}
            </div>
            <div class="qmeta">
              <div><span>window</span><b>t=2.84s → 2.94s</b></div>
              <div><span>active</span><b>11 / 40</b></div>
              <div><span>embed</span><b>SDPA · dim=64</b></div>
            </div>
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">M2</span>Top-5 neighbors · HNSW</div>
          <div class="vc-list">
            <div class="vc-item neighbor"><b>#2041</b><span>t=0.41s · ||Δ||=0.08</span><em>0.992</em></div>
            <div class="vc-item neighbor"><b>#1887</b><span>t=4.20s · ||Δ||=0.14</span><em>0.978</em></div>
            <div class="vc-item neighbor"><b>#3102</b><span>t=7.65s · ||Δ||=0.19</span><em>0.961</em></div>
            <div class="vc-item neighbor"><b>#0544</b><span>t=1.12s · ||Δ||=0.22</span><em>0.944</em></div>
            <div class="vc-item neighbor"><b>#2996</b><span>t=5.83s · ||Δ||=0.28</span><em>0.921</em></div>
          </div>
        </div>
      </div>`,

    causal: () => `
      <div class="vc-grid">
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">C1</span>Targeted cut · M0↔M1<button class="help-icon" data-help="sigma_sep"></button></div>
          <div class="vc-stat-row">
            <div class="vc-stat"><div class="k">z_cut</div><div class="v ok">5.55σ</div></div>
            <div class="vc-stat"><div class="k">p</div><div class="v">&lt;10⁻⁷</div></div>
            <div class="vc-stat"><div class="k">edges</div><div class="v">132</div></div>
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">C2</span>Random null</div>
          <div class="vc-stat-row">
            <div class="vc-stat"><div class="k">z_rand</div><div class="v dim">1.57σ</div></div>
            <div class="vc-stat"><div class="k">p</div><div class="v dim">0.12</div></div>
            <div class="vc-stat"><div class="k">trials</div><div class="v dim">512</div></div>
          </div>
        </div>
        <div class="vc-card wide">
          <div class="vc-head"><span class="vc-num">C3</span>Effect distribution · σ-separation</div>
          <div class="vc-dist">
            <div class="dist-bar null"><b>null</b><i style="--l:10%;--w:40%"></i></div>
            <div class="dist-bar random"><b>random</b><i style="--l:32%;--w:14%"></i></div>
            <div class="dist-bar targeted"><b>targeted</b><i style="--l:68%;--w:8%"></i></div>
            <div class="dist-scale"><span>-2σ</span><span>0</span><span>+2σ</span><span>+5σ</span></div>
          </div>
        </div>
      </div>`,

    acceptance: () => `
      <div class="vc-grid">
        <div class="vc-card wide">
          <div class="vc-head"><span class="vc-num">AC</span>Acceptance suite · 68 tests · 0 fail<button class="help-icon" data-help="view_acceptance"></button></div>
          <div class="vc-ac">
            <div class="ac-cell pass"><b>AC-1</b><span>Repeatability</span><em>pass</em></div>
            <div class="ac-cell partial"><b>AC-2</b><span>Motif emergence</span><em>partial</em></div>
            <div class="ac-cell pass"><b>AC-3a</b><span>Structural cut</span><em>pass</em></div>
            <div class="ac-cell pass"><b>AC-3b</b><span>Functional cut</span><em>pass</em></div>
            <div class="ac-cell pass"><b>AC-4</b><span>Coherence lead</span><em>pass</em></div>
            <div class="ac-cell partial"><b>AC-5</b><span>Causal perturb.</span><em>partial</em></div>
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">CI</span>Commit</div>
          <div class="vc-commit">
            <b>bd26c4ee4</b>
            <span>main @ origin/main</span>
            <em>2 hours ago</em>
          </div>
        </div>
      </div>`,

    benchmarks: () => `
      <div class="vc-grid">
        <div class="vc-card wide">
          <div class="vc-head"><span class="vc-num">B1</span>Throughput · spikes · s⁻¹<button class="help-icon" data-help="throughput"></button></div>
          <div class="vc-bench">
            <div class="bench-row"><b>Connectome</b><i style="--w:96%"></i><em>1.84M</em></div>
            <div class="bench-row"><b>NEST 3.7</b><i style="--w:42%"></i><em>0.81M</em></div>
            <div class="bench-row"><b>Auryn</b><i style="--w:61%"></i><em>1.17M</em></div>
            <div class="bench-row"><b>Brian2</b><i style="--w:18%"></i><em>0.34M</em></div>
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">B2</span>Conditions</div>
          <div class="vc-list">
            <div class="vc-item">Ryzen 7950X · 1 thread · release</div>
            <div class="vc-item">N=10⁵ LIF · p=0.02 · 30s sim</div>
            <div class="vc-item">Saturated regime (firing 10 Hz mean)</div>
          </div>
        </div>
      </div>`,

    console: () => `
      <div class="vc-grid">
        <div class="vc-card wide console">
          <div class="vc-head"><span class="vc-num">$</span>Runtime introspection</div>
          <pre class="vc-term">
<span class="ok">✓</span> engine.init    seed=0xdeadbeef · 139255 neurons · 6.12M edges
<span class="ok">✓</span> wheel.alloc    128 slots × dt=0.1ms · ring resident 1.2 MiB
<span class="ok">✓</span> simd.probe     avx2 f32x8 · fma=yes
<span class="warn">!</span> gap_junction.heuristic  11% coverage · flag AC-3b warn
<span class="ok">✓</span> motif.index    SDPA encoder loaded · dim=64
<span class="ok">✓</span> mincut.stream  λ₂ tracker online · window=50ms
<span class="dim">…</span> trace.attach   ring buffer 64 MiB · 11 measurement discoveries
<span class="ok">✓</span> accept.AT-1    repeatability · 194784 spikes · bit-exact
<span class="warn">!</span> accept.AT-2    motif p@5 = 0.60 (target 0.80) · partial
<span class="ok">✓</span> accept.AT-3a   ARI = 0.78 vs SBM hubs
<span class="ok">✓</span> accept.AT-4    coherence lead 74% of 30 trials
<span class="prompt">connectome:/sim#</span> <span class="cursor">_</span>
          </pre>
        </div>
      </div>`,

    settings: () => `
      <div class="vc-grid">
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">ST</span>Session</div>
          <div class="vc-kv">
            <div><span>seed</span><b>0xdeadbeef</b></div>
            <div><span>commit</span><b>bd26c4ee4</b></div>
            <div><span>engine</span><b>connectome-1.4.0</b></div>
            <div><span>wheel.slots</span><b>128</b></div>
            <div><span>dt</span><b>0.1 ms</b></div>
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">FL</span>Engine flags</div>
          <div class="vc-flags">
            <label><input type="checkbox" checked/><span>simd.f32x8</span></label>
            <label><input type="checkbox" checked/><span>delivery.wheel</span></label>
            <label><input type="checkbox" checked/><span>mincut.stream</span></label>
            <label><input type="checkbox"/><span>trace.full</span></label>
            <label><input type="checkbox"/><span>determinism.strict</span></label>
          </div>
        </div>
        <div class="vc-card">
          <div class="vc-head"><span class="vc-num">R</span>Reproducibility</div>
          <div class="vc-list">
            <div class="vc-item"><span class="dot ok"></span>Bit-exact across runs (seed fixed)</div>
            <div class="vc-item"><span class="dot ok"></span>Graph hash verified · sha256:7a3f…</div>
            <div class="vc-item"><span class="dot ok"></span>Reduction order deterministic</div>
          </div>
        </div>
      </div>`,
  };

  // Create overlay host
  const wrap = document.querySelector('.canvas-wrap');
  if (!wrap) return;
  const overlay = document.createElement('div');
  overlay.id = 'view-content';
  overlay.className = 'view-content';
  wrap.appendChild(overlay);

  function setView(view) {
    const builder = CONTENT[view];
    if (!builder) {
      overlay.classList.remove('active');
      wrap.classList.remove('view-content-active');
      overlay.innerHTML = '';
      return;
    }
    overlay.innerHTML = builder();
    overlay.classList.add('active');
    wrap.classList.add('view-content-active');
  }

  // Expose
  W.ViewContent = { setView };
})();
