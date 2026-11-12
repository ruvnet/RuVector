// Agent dock frontend (ADR-295).
//
// This file renders dock chrome from what the Rust commands return and nothing
// else. Two rules hold everywhere below:
//
// 1. Agent-supplied strings reach the DOM only through `textContent`, only
//    inside an element carrying `.agent-region`, and only next to the
//    dock-drawn "Agent-reported" label. There is no path from agent text to
//    markup, to a class name, or to a state indicator.
// 2. Nothing here decides a state, a badge, or a permission. Pause and
//    terminate call the Rust commands and re-read the roster; the answer is
//    whatever Rust says it is.
//
// Wrapped in an IIFE so its bindings do not collide with main.js, which shares
// the classic-script global scope.

(function () {
  const invoke = (cmd, args) => {
    const api = window.__TAURI__;
    if (!api) {
      return Promise.reject(
        new Error("Tauri API unavailable (open this page through the app)."),
      );
    }
    return api.core.invoke(cmd, args);
  };

  const $ = (id) => document.getElementById(id);
  const dock = { expandedAgent: null };

  function fail(err) {
    const box = $("error");
    box.textContent = String(err);
    box.hidden = false;
  }

  function clearError() {
    $("error").hidden = true;
  }

  function bytes(n) {
    if (!n) return "0 B";
    const units = ["B", "KiB", "MiB", "GiB"];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i += 1;
    }
    return `${i === 0 ? v : v.toFixed(1)} ${units[i]}`;
  }

  const money = (micros) => `$${(micros / 1_000_000).toFixed(2)}`;

  // Agent text goes here and nowhere else. `suspicious` is a Rust verdict; the
  // agent cannot set it, and when it is set the withheld original is shown as a
  // title so a curious user can still read what was claimed.
  function renderAgentContent(regionEl, textEl, content) {
    textEl.textContent = content.text;
    regionEl.classList.toggle("suspicious", content.suspicious);
    regionEl.title = content.suspicious
      ? `Withheld — the agent said: ${content.withheld ?? ""}`
      : "";
    const label = regionEl.querySelector(".agent-label");
    if (label) label.textContent = content.label;
  }

  function renderPill(pill) {
    $("dock-empty").hidden = true;
    $("dock-pill").hidden = false;

    // System-owned: manifest identity and measured state.
    $("pill-icon").textContent = "◆";
    $("pill-name").textContent = pill.name;

    renderAgentContent($("pill-task-region"), $("pill-task"), pill.task);

    $("pill-progress-bar").value = pill.progress;
    $("pill-progress-text").textContent = `${pill.progress}%`;

    const chip = $("pill-state");
    chip.className = `state-chip ${pill.state_color}`;
    $("pill-state-glyph").textContent = pill.state_glyph;
    $("pill-state-label").textContent = pill.state_label;

    // Pause and Terminate are present in every non-terminal state; the list
    // comes from Rust, so the webview cannot hide a control.
    const controls = pill.controls;
    const pause = $("pill-pause");
    const resumable = controls.includes("resume");
    pause.textContent = resumable ? "Resume" : "Pause";
    pause.hidden = !(controls.includes("pause") || resumable);
    $("pill-stop").hidden = !controls.includes("terminate");
    $("pill-expand").hidden = !controls.includes("expand");

    $("dock-pill").dataset.agentId = pill.agent_id;
  }

  function renderSecondary(pills) {
    const box = $("dock-secondary");
    box.replaceChildren();
    for (const pill of pills) {
      const btn = document.createElement("button");
      btn.className = "secondary-icon";
      btn.title = `${pill.name} — ${pill.state_label}`;

      const dot = document.createElement("span");
      dot.className = `dot ${pill.state_color}`;
      const name = document.createElement("span");
      name.textContent = pill.name;

      btn.append(dot, name);
      btn.addEventListener("click", () => expand(pill.agent_id));
      box.append(btn);
    }
  }

  function renderView(view) {
    if (view.active) {
      renderPill(view.active);
    } else {
      $("dock-pill").hidden = true;
      $("dock-empty").hidden = false;
    }
    renderSecondary(view.secondary);

    const overflow = $("dock-overflow");
    overflow.hidden = view.overflow_count === 0;
    overflow.textContent = `+${view.overflow_count} more`;

    // Aggregates cover every agent, including the ones with no slot.
    $("agg-cpu").textContent = `${view.aggregate.cpu_millis_per_sec} mCPU`;
    $("agg-mem").textContent = bytes(view.aggregate.memory_bytes);
    $("agg-cost").textContent = money(view.aggregate.estimated_cost_micros);
    $("agg-approvals").textContent =
      view.pending_approvals === 1
        ? "1 approval"
        : `${view.pending_approvals} approvals`;
  }

  function renderActions(actions) {
    const ul = $("exp-actions");
    ul.replaceChildren();
    for (const action of actions.slice(-12).reverse()) {
      const li = document.createElement("li");
      // Agent-authored lines keep the agent styling inside the list too.
      li.className = action.agent_authored ? "agent-authored" : "";
      li.textContent = action.text.display;
      if (action.text.suspicious && action.text.withheld) {
        li.title = `Withheld — the agent said: ${action.text.withheld}`;
      }
      ul.append(li);
    }
  }

  function renderList(id, items, emptyText) {
    const ul = $(id);
    ul.replaceChildren();
    if (!items.length) {
      const li = document.createElement("li");
      li.textContent = emptyText;
      ul.append(li);
      return;
    }
    for (const item of items) {
      const li = document.createElement("li");
      li.textContent = item;
      ul.append(li);
    }
  }

  function renderCapabilityCard(card) {
    const ul = $("exp-card");
    ul.replaceChildren();
    for (const line of card.lines) {
      const li = document.createElement("li");
      li.className = line.satisfied ? "yes" : "no";

      const mark = document.createElement("span");
      mark.className = "mark";
      mark.textContent = line.satisfied ? "✓" : "✕";

      const label = document.createElement("span");
      label.textContent = line.label;

      const detail = document.createElement("span");
      detail.className = "detail";
      detail.textContent = line.detail;

      li.append(mark, label, detail);
      ul.append(li);
    }
  }

  function renderExpanded(view) {
    dock.expandedAgent = view.pill.agent_id;
    renderPill(view.pill);
    renderAgentContent(
      $("exp-objective").parentElement,
      $("exp-objective"),
      view.objective,
    );

    renderActions(view.recent_actions);
    renderList("exp-approvals", view.pending_approvals, "None pending.");
    renderList("exp-grants", view.capability_grants, "Nothing is granted.");

    $("exp-model").textContent = view.model ?? "no model attached";
    $("exp-tokens").textContent = `${view.tokens_in} in · ${view.tokens_out} out`;
    $("exp-cpu").textContent = `${view.cpu_millis_per_sec} mCPU`;
    $("exp-mem").textContent = bytes(view.memory_bytes);
    $("exp-network").textContent = view.network;
    $("exp-witness").textContent = view.witness;
    $("exp-witness").className =
      view.witness === "valid" ? "status-ok" : "status-unverified";
    $("exp-cost").textContent = money(view.estimated_cost_micros);

    $("exp-instruction").disabled = !view.instruction_field_enabled;
    $("exp-send").disabled = !view.instruction_field_enabled;

    renderCapabilityCard(view.capability_card);

    $("dock-expanded").hidden = false;
    $("pill-expand").setAttribute("aria-expanded", "true");
  }

  async function refresh() {
    try {
      renderView(await invoke("dock_view"));
      if (dock.expandedAgent) {
        renderExpanded(await invoke("dock_expand", { agentId: dock.expandedAgent }));
      }
    } catch (err) {
      fail(err);
    }
  }

  async function expand(agentId) {
    clearError();
    try {
      renderExpanded(await invoke("dock_expand", { agentId }));
    } catch (err) {
      fail(err);
    }
  }

  function activeAgentId() {
    return $("dock-pill").dataset.agentId || null;
  }

  // One action. No confirmation chain, no expand first.
  async function control(command) {
    const agentId = activeAgentId();
    if (!agentId) return;
    clearError();
    try {
      await invoke(command, { agentId });
      await refresh();
    } catch (err) {
      fail(err);
    }
  }

  $("pill-pause").addEventListener("click", () => control("dock_pause"));
  $("pill-stop").addEventListener("click", () => control("dock_terminate"));
  $("pill-expand").addEventListener("click", () => {
    const panel = $("dock-expanded");
    if (!panel.hidden) {
      panel.hidden = true;
      dock.expandedAgent = null;
      $("pill-expand").setAttribute("aria-expanded", "false");
      return;
    }
    const agentId = activeAgentId();
    if (agentId) expand(agentId);
  });

  $("dock-overflow").addEventListener("click", async () => {
    clearError();
    try {
      const pills = await invoke("dock_swarm_view");
      renderSecondary(pills.slice(1));
      $("dock-overflow").hidden = true;
    } catch (err) {
      fail(err);
    }
  });

  // Rebuild the roster from the library. Everything the pill shows about an
  // installed agent — state, trust badge, network, permissions, witness — is
  // re-derived in Rust from the install record and the verified chain, so this
  // button cannot produce a friendlier-looking dock than the facts support.
  $("dock-sync").addEventListener("click", async () => {
    clearError();
    try {
      renderView(await invoke("dock_sync"));
      dock.expandedAgent = null;
      $("dock-expanded").hidden = true;
    } catch (err) {
      fail(err);
    }
  });

  // Development harness. `dock_add_scaffold_agent` reports unknown values for
  // every security-bearing field, and `dock_agent_report` is the agent channel:
  // task text and progress, both sanitized in Rust.

  $("dock-add").addEventListener("click", async () => {
    const name = $("dock-new-name").value.trim() || "Unnamed agent";
    clearError();
    try {
      renderView(
        await invoke("dock_add_scaffold_agent", {
          agentId: `agent-${Date.now()}`,
          name,
          icon: "analyst",
        }),
      );
      $("dock-new-name").value = "";
    } catch (err) {
      fail(err);
    }
  });

  $("dock-agent-send").addEventListener("click", async () => {
    const agentId = activeAgentId();
    if (!agentId) {
      fail("Register an agent first.");
      return;
    }
    const progress = Number.parseInt($("dock-agent-progress").value, 10);
    clearError();
    try {
      await invoke("dock_agent_report", {
        agentId,
        task: $("dock-agent-task").value,
        progress: Number.isNaN(progress) ? 0 : progress,
      });
      await refresh();
    } catch (err) {
      fail(err);
    }
  });

  $("exp-send").addEventListener("click", () =>
    fail("Instructions reach the agent once rvm-ffi is wired in."),
  );

  const dockStep = document.querySelector('.step[data-goto="dock"]');
  if (dockStep) dockStep.addEventListener("click", refresh);
})();
