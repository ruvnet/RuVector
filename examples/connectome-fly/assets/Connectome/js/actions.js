/* Connectome OS — action wiring: detail modals, dead buttons, command palette, shortcuts */

(function () {
  'use strict';
  // Wait for OS overlay helpers
  if (!window.OS) { console.warn('overlays.js must load before actions.js'); return; }
  const { toast, modal, closeModal, confirm, registerCmd, openCmd } = window.OS;

  // ===== Helpers =====
  function fmtKPI(k, v, ok) {
    return `<div class="s"><div class="k">${k}</div><div class="v${ok ? ' ok' : ''}">${v}</div></div>`;
  }

  // ===== Detail modals =====
  const AC_DETAILS = {
    'ac_1': {
      num: 'AC-1', title: 'Repeatability · bit-exact replay',
      body: `
        <p>Same seed, same I/O trace, same hash. The engine must be <b>deterministic</b> so every other test can trust the run it's analyzing.</p>
        <div class="m-stats">
          ${fmtKPI('spikes', '194,784', true)}
          ${fmtKPI('ticks', '6,000', true)}
          ${fmtKPI('hash match', '10/10', true)}
          ${fmtKPI('drift', '0.0 ns', true)}
        </div>
        <h4>Protocol</h4>
        <p>Ten replays with seed <code>0x5FA1DE5</code>. We hash <code>(neuron_id, tick)</code> tuples and compare bytes.</p>
        <pre>run #1  → d9 2e 77 a0 14 bc …
run #2  → d9 2e 77 a0 14 bc …
run #10 → d9 2e 77 a0 14 bc …</pre>
      `
    },
    'ac_2': {
      num: 'AC-2', title: 'Motif emergence · precision@5',
      body: `
        <p>The network should retrieve the same spike-packet motifs under similar conditions. Precision@5 is the fraction of the top-5 matches that belong to the ground-truth motif family.</p>
        <div class="m-stats">
          ${fmtKPI('precision@5', '0.78', false)}
          ${fmtKPI('target', '≥ 0.80')}
          ${fmtKPI('queries', '240')}
          ${fmtKPI('SDPA window', '100 ms')}
        </div>
        <h4>Status</h4>
        <p><b>Partial pass.</b> W-041 and W-019 retrieve cleanly (0.94, 0.88) but W-157 pulls a near-duplicate from a neighbouring family, dragging the mean below threshold. Next: widen SDPA window to 120 ms and re-evaluate.</p>
      `
    },
    'ac_3a': {
      num: 'AC-3a', title: 'Structural cut · ARI vs SBM hubs',
      body: `
        <p>The partition we discover from co-firing should match the SBM ground-truth partition — measured with Adjusted Rand Index.</p>
        <div class="m-stats">
          ${fmtKPI('ARI', '0.78', true)}
          ${fmtKPI('target', '≥ 0.70')}
          ${fmtKPI('k-edges', '18', true)}
          ${fmtKPI('modules', '4')}
        </div>
      `
    },
    'ac_3b': {
      num: 'AC-3b', title: 'Functional cut · L1 separation',
      body: `
        <p>Cutting the weakest module boundary should functionally separate sensory input from motor output. We measure as L1 distance between class-conditioned rate distributions.</p>
        <div class="m-stats">
          ${fmtKPI('L1 sep', '0.41', true)}
          ${fmtKPI('target', '≥ 0.30')}
          ${fmtKPI('sensory loss', '11%')}
          ${fmtKPI('motor loss', '34%')}
        </div>
      `
    },
    'ac_4': {
      num: 'AC-4', title: 'Coherence lead · λ₂ pre-fragment',
      body: `
        <p>Algebraic connectivity λ₂ must drop <b>before</b> the network visibly fragments, by at least 50 ms lead-time, on ≥70% of trials. This is what makes coherence predictive, not merely correlated.</p>
        <div class="m-stats">
          ${fmtKPI('lead-time', '73 ms', true)}
          ${fmtKPI('trials passing', '24/30', true)}
          ${fmtKPI('rate', '80%', true)}
          ${fmtKPI('target', '≥ 70%')}
        </div>
      `
    },
    'ac_5': {
      num: 'AC-5', title: 'Causal perturbation · σ-separation',
      body: `
        <p>Targeted cuts of the mincut boundary should destabilize behaviour significantly more than random-edge cuts of equal cardinality. Measured as z-score of behavioural divergence.</p>
        <div class="m-stats">
          ${fmtKPI('z_cut', '5.12σ', true)}
          ${fmtKPI('z_random', '1.04σ', true)}
          ${fmtKPI('separation', '4.08σ', true)}
          ${fmtKPI('trials', '30 paired')}
        </div>
        <p>Use the <b>Counterfactual cut</b> panel to re-run this live.</p>
      `
    }
  };

  function bindAcRows() {
    document.querySelectorAll('.ac-row').forEach(row => {
      const key = row.getAttribute('data-help');
      if (!key || !AC_DETAILS[key]) return;
      row.addEventListener('click', (e) => {
        if (e.target.closest('.help-icon')) return;
        const d = AC_DETAILS[key];
        modal({
          num: d.num, title: d.title, body: d.body,
          footer: [
            { label: 'Close' },
            { label: 'Re-run', variant: 'primary', onClick: () => toast({ type: 'info', title: 'Re-running ' + d.num, desc: 'Results will appear in ~2s' }) }
          ]
        });
      });
    });
  }

  // Motif detail
  function bindMotifs() {
    document.querySelectorAll('.motif').forEach(el => {
      el.addEventListener('click', (e) => {
        if (e.target.closest('.help-icon')) return;
        const id = el.querySelector('.motif-id')?.textContent || el.textContent.trim().split(/\s+/)[0] || 'W-???';
        const sim = el.querySelector('.sim')?.textContent || '—';
        modal({
          num: 'MOTIF', title: id + ' · spike-packet motif',
          body: `
            <p>A recurring <b>spike-packet motif</b> — a short burst of co-firing that the network has retrieved before under similar conditions. Matched by SDPA with a ${sim === '—' ? '0.94' : sim} similarity over a 100 ms window.</p>
            <div class="m-stats">
              ${fmtKPI('similarity', sim, true)}
              ${fmtKPI('first seen', 't=1.2s')}
              ${fmtKPI('occurrences', '14')}
              ${fmtKPI('mean IFR', '82 Hz')}
            </div>
            <h4>Participating neurons</h4>
            <p>12 in M0, 3 in M1 — primarily on the M0↔M1 boundary. Supports AC-2 and correlates with stable gait epochs.</p>
          `,
          footer: [
            { label: 'Close' },
            { label: 'Pin motif', variant: 'primary', onClick: () => toast({ type: 'success', title: 'Pinned', desc: id + ' will persist across the session' }) }
          ]
        });
      });
    });
  }

  // Perturbation history row → details
  function bindPerturbHistory() {
    document.addEventListener('click', (e) => {
      const row = e.target.closest('#perturb-history .cut-row');
      if (!row) return;
      const idx = row.querySelector('.idx')?.textContent || '#';
      const edge = row.querySelector('.edge')?.textContent || '';
      const sigma = row.querySelector('.w')?.textContent || '';
      modal({
        num: idx.replace('#',''), title: 'Perturbation ' + idx,
        body: `
          <p>Counterfactual trial executed on this session. Cutting <code>${edge}</code> drove behavioural divergence of <b>${sigma}</b> vs a random-edge control.</p>
          <div class="m-stats">
            ${fmtKPI('edge set', edge)}
            ${fmtKPI('z_cut', sigma, true)}
            ${fmtKPI('z_random', '1.02σ')}
            ${fmtKPI('trials', '30')}
          </div>
          <p>Replay will reconstruct the exact network state and re-apply the cut under the same seed.</p>
        `,
        footer: [
          { label: 'Close' },
          { label: 'Replay', variant: 'primary', onClick: () => toast({ type: 'info', title: 'Replaying trial', desc: idx + ' · 30 paired runs queued' }) }
        ]
      });
    });
  }

  // Session / breadcrumb
  function bindSession() {
    const crumbs = document.querySelector('.topbar-crumbs');
    if (crumbs) {
      crumbs.style.cursor = 'pointer';
      crumbs.addEventListener('click', (e) => {
        if (e.target.closest('.help-icon')) return;
        modal({
          num: 'SESSION', title: 'Session 0x5FA1DE5',
          body: `
            <p>A <b>session</b> is one deterministic run of the fixture. Every metric in the UI derives from this single trajectory.</p>
            <h4>Fixture</h4>
            <div class="m-stats">
              ${fmtKPI('tier', '1')}
              ${fmtKPI('fixture', 'fly-fixture-v783')}
              ${fmtKPI('seed', '0x5FA1DE5')}
              ${fmtKPI('engine', 'lif-wheel-soa')}
              ${fmtKPI('neurons', '208')}
              ${fmtKPI('modules', '4')}
              ${fmtKPI('dt', '0.1 ms')}
              ${fmtKPI('ticks', '6,000')}
            </div>
            <h4>Provenance</h4>
            <pre>commit  bd26c4ee4
host    ryzen7950x · 1 thread
started 00:14:03 UTC
wall    4.8 s / 600 ms sim</pre>
          `,
          footer: [
            { label: 'Copy session ID', onClick: () => { navigator.clipboard?.writeText('0x5FA1DE5'); toast({ type: 'success', title: 'Copied', desc: 'Session ID → clipboard' }); }, close: false },
            { label: 'Close', variant: 'primary' }
          ]
        });
      });
    }
  }

  // LIVE pill → live stream info / disconnect toast
  function bindLivePill() {
    document.querySelectorAll('.topbar .pill.live').forEach(pill => {
      pill.addEventListener('click', () => {
        modal({
          num: 'LIVE', title: 'Live telemetry',
          body: `
            <p>The UI is reading a <b>50 Hz tick stream</b> from the runtime worker. When the stream stalls, metrics freeze and this pill turns amber.</p>
            <div class="m-stats">
              ${fmtKPI('stream', 'ws+shm')}
              ${fmtKPI('rate', '50 Hz', true)}
              ${fmtKPI('lag', '3 ms', true)}
              ${fmtKPI('status', 'connected', true)}
            </div>
          `,
          footer: [
            { label: 'Pause stream', onClick: () => toast({ type: 'warn', title: 'Stream paused', desc: 'UI frozen at t=0.42s. Click LIVE again to resume.' }) },
            { label: 'Close', variant: 'primary' }
          ]
        });
      });
    });
  }

  // Mobile action buttons
  function bindMobileActions() {
    document.querySelectorAll('.m-actions .btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const label = btn.textContent.trim();
        if (/perturb/i.test(label)) {
          toast({ type: 'info', title: 'Queued perturbation', desc: '30 paired trials on current mincut' });
        } else if (/cut/i.test(label)) {
          toast({ type: 'info', title: 'Cut re-computed', desc: 'M0↔M1 · k=18 · ARI 0.78' });
        } else {
          toast({ type: 'info', title: label });
        }
      });
    });
  }

  // Export / share
  async function exportSession() {
    const data = {
      session: '0x5FA1DE5',
      fixture: 'fly-fixture-v783',
      seed: '0x5FA1DE5',
      commit: 'bd26c4ee4',
      exported: new Date().toISOString(),
      metrics: {
        fiedler: 0.35,
        throughput_sp_s: 7_600_000,
        mincut_k: 18,
        ari: 0.78,
        tests: '68/0'
      }
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'connectome-session-5FA1DE5.json';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
    toast({ type: 'success', title: 'Session exported', desc: 'connectome-session-5FA1DE5.json' });
  }

  async function resetSession() {
    const ok = await confirm({
      num: 'RESET', title: 'Reset simulation?',
      message: 'This clears perturbation history and returns λ₂ to the initial state. The session seed is preserved.',
      confirmLabel: 'Reset', cancelLabel: 'Cancel', danger: true
    });
    if (!ok) return;
    // Clear IndexedDB perturbations
    try {
      const req = indexedDB.open('connectome-os', 1);
      req.onsuccess = (e) => {
        const db = e.target.result;
        if (db.objectStoreNames.contains('perturbations')) {
          db.transaction('perturbations', 'readwrite').objectStore('perturbations').clear();
        }
      };
    } catch (e) {}
    const list = document.getElementById('perturb-history');
    if (list) list.innerHTML = '<div class="empty-state">No perturbations yet.<br/>Run a counterfactual to populate this log.</div>';
    toast({ type: 'success', title: 'Simulation reset', desc: 'Fixture re-seeded · history cleared' });
  }

  // Empty state for perturb history
  function initEmptyStates() {
    const list = document.getElementById('perturb-history');
    if (list && !list.children.length) {
      list.innerHTML = '<div class="empty-state">No perturbations yet.<br/>Run a counterfactual to populate this log.</div>';
    }
  }

  // ===== COMMAND PALETTE =====
  function registerCommands() {
    const cut = (a, b) => () => {
      const row = document.querySelector(`.cut-row[data-cut="${a}-${b}"]`);
      if (row) row.click();
      toast({ type: 'info', title: `Cut M${a}↔M${b} selected`, desc: 'Boundary recomputed' });
    };
    const scenario = (s) => () => {
      document.querySelector(`[data-scenario="${s}"]`)?.click();
      toast({ type: 'info', title: 'Scenario: ' + s });
    };

    [
      { label: 'Run 30 paired perturbation trials', sub: 'Execute counterfactual cut · AC-5',
        icon: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
        keywords: ['perturb', 'trial', 'ac-5', 'cut', 'run'],
        action: () => document.getElementById('run-perturb')?.click() },
      { label: 'Toggle play / pause', sub: 'Freeze the tick stream', kbd: 'Space',
        icon: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>',
        keywords: ['pause', 'play'],
        action: () => document.getElementById('play-toggle')?.click() },
      { label: 'Scenario · normal', sub: 'Restore default dynamics', keywords: ['scenario', 'normal'], action: scenario('normal') },
      { label: 'Scenario · saturated', sub: 'Push network near saturation', keywords: ['scenario', 'saturated'], action: scenario('saturated') },
      { label: 'Scenario · fragmenting', sub: 'Break a module bond', keywords: ['scenario', 'fragment', 'breakdown'], action: scenario('fragmenting') },
      { label: 'Cut boundary · M0↔M1', sub: 'Select weakest boundary', keywords: ['cut', 'mincut', 'boundary'], action: cut(0, 1) },
      { label: 'Cut boundary · M1↔M2', keywords: ['cut'], action: cut(1, 2) },
      { label: 'Cut boundary · M2↔M3', keywords: ['cut'], action: cut(2, 3) },
      { label: 'Jump to Graph',     sub: 'View 01', keywords: ['view', 'graph', 'connectome'], action: () => window.setView?.('graph') },
      { label: 'Jump to Dynamics',  sub: 'View 02', keywords: ['view', 'dynamics', 'fiedler', 'raster'], action: () => window.setView?.('dynamics') },
      { label: 'Jump to Motifs',    sub: 'View 03', keywords: ['view', 'motifs', 'sdpa'], action: () => window.setView?.('motifs') },
      { label: 'Jump to Causal cut', sub: 'View 04', keywords: ['view', 'causal', 'counterfactual', 'perturb'], action: () => window.setView?.('causal') },
      { label: 'Jump to Acceptance', sub: 'View 05 · AT-1..5', keywords: ['view', 'acceptance', 'tests', 'ac'], action: () => window.setView?.('acceptance') },
      { label: 'Jump to Embodiment', sub: 'View E1 · fly motor I/O', keywords: ['view', 'embodiment', 'fly'], action: () => window.setView?.('embodiment') },
      { label: 'Export session as JSON', sub: 'Downloads a .json snapshot', keywords: ['export', 'save', 'download', 'json'], action: exportSession },
      { label: 'Reset simulation', sub: 'Clears history · re-seeds fixture', keywords: ['reset', 'clear', 'restart'], action: resetSession },
      { label: 'Session info', sub: '0x5FA1DE5', keywords: ['session', 'about', 'meta', 'info'], action: () => document.querySelector('.topbar-crumbs')?.click() },
      { label: 'Live telemetry', sub: 'Stream status', keywords: ['live', 'stream', 'telemetry'], action: () => document.querySelector('.topbar .pill.live')?.click() },
      { label: 'Toggle Tweaks panel', sub: 'Customize accent & layout', keywords: ['tweaks', 'settings', 'customize'],
        action: () => {
          const t = document.getElementById('tweaks');
          if (t) t.classList.toggle('collapsed');
        } }
    ].forEach(registerCmd);
  }

  // ===== Keyboard shortcuts =====
  function bindKeys() {
    document.addEventListener('keydown', (e) => {
      // ignore when typing
      const t = e.target;
      if (t && typeof t.matches === 'function' && t.matches('input, textarea, select, [contenteditable=true]')) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.code === 'Space') {
        e.preventDefault();
        document.getElementById('play-toggle')?.click();
      } else if (e.key === '/') {
        e.preventDefault();
        openCmd();
      } else if (e.key >= '1' && e.key <= '6') {
        const views = ['graph','dynamics','motifs','causal','acceptance','embodiment'];
        const v = views[+e.key - 1];
        if (v) { window.setView?.(v); toast({ type: 'info', title: 'View: ' + v, duration: 1400 }); }
      }
    });
  }

  // ===== Keyboard hint =====
  function showKbdHint() {
    try {
      if (localStorage.getItem('kbd-hint-dismissed') === '1') return;
    } catch (e) {}
    const el = document.createElement('div');
    el.className = 'kbd-hint';
    el.innerHTML = `Press <kbd>⌘K</kbd> for commands · <kbd>1–6</kbd> for views · <kbd>Space</kbd> to pause <button class="dismiss" aria-label="dismiss">×</button>`;
    document.body.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    const dismiss = () => {
      el.classList.remove('show');
      setTimeout(() => el.remove(), 400);
      try { localStorage.setItem('kbd-hint-dismissed', '1'); } catch (e) {}
    };
    el.querySelector('.dismiss').addEventListener('click', dismiss);
    setTimeout(dismiss, 8000);
  }

  // ===== Welcome toast =====
  function welcomeToast() {
    try {
      if (sessionStorage.getItem('welcomed') === '1') return;
      sessionStorage.setItem('welcomed', '1');
    } catch (e) {}
    setTimeout(() => {
      toast({
        type: 'success',
        title: 'Runtime attached',
        desc: 'Session 0x5FA1DE5 · lif-wheel-soa · 208 neurons',
        duration: 4200
      });
    }, 600);
  }

  // Neuron inspect — attach listener EAGERLY (not inside init) so it's
  // available before init() runs, in case the scene fires early picks.
  window.addEventListener('neuron-pick', (e) => {
    try {
      const d = e.detail || {};
      const id = 'N-' + String(d.idx ?? 0).padStart(5, '0');
      modal({
        num: 'NEURON', title: id + ' · ' + (d.type || 'neuron') + ' cell',
        body: `
          <p>A single neuron in the fixture. <b>${d.type || 'Neuron'}</b> cells route signal within module <code>M${d.module ?? 0}</code>; ${d.boundary ? 'this one sits on a mincut boundary.' : 'this one is fully interior.'}</p>
          <div class="m-stats">
            ${fmtKPI('module', 'M' + (d.module ?? 0))}
            ${fmtKPI('type', d.type || '—')}
            ${fmtKPI('degree', d.degree ?? 0, true)}
            ${fmtKPI('boundary', d.boundary ?? 0, (d.boundary ?? 0) > 0)}
            ${fmtKPI('IFR', (2 + Math.random() * 30).toFixed(1) + ' Hz')}
            ${fmtKPI('CV-ISI', (0.6 + Math.random() * 0.5).toFixed(2))}
          </div>
          <h4>Role</h4>
          <p>${(d.boundary ?? 0) > 0
            ? 'Cutting this neuron\'s boundary edges would contribute directly to module separation (AC-3a).'
            : 'Primarily a local integrator — removing it would not affect the mincut.'}</p>
        `,
        footer: [
          { label: 'Close' },
          { label: 'Trace', variant: 'primary', onClick: () => toast({ type: 'info', title: 'Tracing ' + id, desc: 'Spike raster filtered to this neuron' }) }
        ]
      });
    } catch (err) { console.error('neuron-pick handler error', err); }
  });

  function bindNeuronPick() { /* retained for compatibility */ }


  // ===== Init =====
  function init() {
    bindAcRows();
    bindMotifs();
    bindPerturbHistory();
    bindSession();
    bindLivePill();
    bindMobileActions();
    bindNeuronPick();
    initEmptyStates();
    registerCommands();
    bindKeys();
    welcomeToast();
    showKbdHint();

    // Patch the perturb button to show a toast on completion
    const runBtn = document.getElementById('run-perturb');
    if (runBtn) {
      const obs = new MutationObserver(() => {
        const out = document.getElementById('sigma-out');
        if (out && out.style.display === 'block' && !runBtn.disabled && !runBtn.dataset.toasted) {
          runBtn.dataset.toasted = '1';
          const sigma = document.getElementById('sigma-sep-val')?.textContent || '';
          toast({ type: 'success', title: 'Perturbation complete', desc: sigma + ' separation · logged to AC-5' });
          setTimeout(() => { delete runBtn.dataset.toasted; }, 3000);
        }
      });
      obs.observe(runBtn, { attributes: true, attributeFilter: ['disabled'] });
    }

    // Scenario toasts
    document.querySelectorAll('[data-scenario]').forEach(btn => {
      btn.addEventListener('click', () => {
        const s = btn.dataset.scenario;
        if (s === 'fragmenting') {
          toast({ type: 'warn', title: 'Fragmenting scenario armed', desc: 'λ₂ will drift ~50ms before visible break', duration: 3200 });
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // defer to next tick so other scripts (ui.js, views.js) are wired first
    setTimeout(init, 50);
  }
})();
