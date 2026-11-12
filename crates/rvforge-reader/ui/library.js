// Library frontend (requirements P7, P9).
//
// Same rule as everywhere else in this app: nothing here decides anything. The
// state transitions, the lineage check, the approval gate, and the rollback
// refusal all live in Rust; this file renders their answers and surfaces their
// errors verbatim rather than softening them into a friendlier message.
//
// Two behaviours are deliberate:
//
// 1. Uninstall offers three separate choices — keep state, export then remove,
//    remove without exporting — because collapsing them into one button is how
//    a user loses state they did not agree to lose.
// 2. The update panel shows the permission difference *before* the approve
//    button does anything, and the button submits only the classes the plan
//    listed as expanding.
//
// Wrapped in an IIFE so its bindings do not collide with main.js.

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
  const lib = { entries: [], selected: null, plan: null };

  function fail(err) {
    const box = $("error");
    box.textContent = String(err);
    box.hidden = false;
  }

  function clearError() {
    $("error").hidden = true;
  }

  function humanBytes(n) {
    if (!n) return "none";
    const units = ["bytes", "KiB", "MiB", "GiB"];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i += 1;
    }
    return `${i === 0 ? v : v.toFixed(1)} ${units[i]}`;
  }

  function button(label, title, onClick) {
    const b = document.createElement("button");
    b.className = "ctl";
    b.textContent = label;
    if (title) b.title = title;
    b.addEventListener("click", onClick);
    return b;
  }

  function cell(row, text, className) {
    const td = document.createElement("td");
    td.textContent = text;
    if (className) td.className = className;
    row.append(td);
    return td;
  }

  async function run(command, args, note) {
    clearError();
    try {
      const result = await invoke(command, args);
      await refresh();
      if (note) {
        const box = $("lib-count");
        box.textContent = typeof note === "function" ? note(result) : note;
      }
      return result;
    } catch (err) {
      fail(err);
      await refresh();
      return null;
    }
  }

  async function refresh() {
    clearError();
    try {
      lib.entries = await invoke("library_list");
    } catch (err) {
      fail(err);
      lib.entries = [];
    }
    renderRows();
  }

  function renderRows() {
    const body = $("lib-rows");
    body.replaceChildren();
    $("lib-count").textContent = lib.entries.length
      ? `${lib.entries.length} installed`
      : "Nothing is installed yet.";

    for (const entry of lib.entries) {
      const record = entry.record;
      const tr = document.createElement("tr");

      cell(tr, record.agentName);
      cell(tr, entry.state_label, entry.state === "running" ? "yes" : "");
      cell(tr, record.baseIdentity.slice(0, 20) + "…", "mono").title =
        record.baseIdentity;
      cell(
        tr,
        record.grantedClasses.length ? record.grantedClasses.join(", ") : "nothing",
      );
      cell(tr, humanBytes(entry.state_bytes));

      const actions = document.createElement("td");
      const id = record.installId;
      actions.append(
        button("Open", "Re-verifies the artifact first", () =>
          run("library_open", { installId: id }),
        ),
        button("Pause", "", () => run("library_pause", { installId: id })),
        button("Terminate", "", () => run("library_terminate", { installId: id })),
        button("Verify", "", () =>
          run("library_verify", { installId: id }, (r) => `Verification: ${r}`),
        ),
        button("Reset", "Discards state; the base artifact is untouched", () =>
          run("library_reset", { installId: id }, (n) => `${n} state record(s) discarded`),
        ),
        button("Export state", "", () => exportState(id)),
        button("Import state", "", () => importState(id)),
        button("Archive", "", () => run("library_archive", { installId: id })),
        button("Uninstall", "", () => uninstall(entry)),
        button("Update…", "", () => select(entry)),
      );
      tr.append(actions);
      body.append(tr);
    }
  }

  async function askPath(message) {
    // No file picker for a directory in this build; a typed path keeps the
    // operation explicit rather than guessing a default location.
    const answer = window.prompt(message);
    return answer && answer.trim() ? answer.trim() : null;
  }

  async function exportState(installId) {
    const dest = await askPath("Export the sealed state capsule to which directory?");
    if (!dest) return;
    await run("library_export_state", { installId, dest }, (t) =>
      `Exported ${t.files} sealed record(s) to ${t.path}`,
    );
  }

  async function importState(installId) {
    const source = await askPath("Import a sealed state capsule from which directory?");
    if (!source) return;
    await run("library_import_state", { installId, source }, (t) =>
      `Imported ${t.files} sealed record(s)`,
    );
  }

  // Three distinct choices. The default keeps state, and removing it is never
  // a side effect of removing the runtime (ADR-294).
  async function uninstall(entry) {
    const id = entry.record.installId;
    const held = entry.state_bytes > 0;
    const answer = window.prompt(
      held
        ? `'${entry.record.agentName}' holds ${humanBytes(entry.state_bytes)} of sealed state.\n\n` +
          "Type 'keep' to remove the runtime only,\n" +
          "a directory path to export the state and then remove it,\n" +
          "or 'discard' to delete the state without exporting."
        : `Remove '${entry.record.agentName}'? Type 'keep' to confirm.`,
      "keep",
    );
    if (!answer) return;

    const choice = answer.trim();
    if (choice === "keep") {
      await run("library_uninstall", { installId: id }, (o) =>
        `Runtime removed; state retained at ${o.state_retained_at}`,
      );
    } else if (choice === "discard") {
      await run(
        "library_uninstall",
        { installId: id, removeState: true },
        "Runtime and state removed at your request",
      );
    } else {
      await run(
        "library_uninstall",
        { installId: id, removeState: true, exportTo: choice },
        (o) => `Exported ${o.exported_files} record(s) to ${o.exported_to}, then removed`,
      );
    }
  }

  function select(entry) {
    lib.selected = entry;
    lib.plan = null;
    $("upd-subject").textContent = `${entry.record.agentName} (${entry.record.baseIdentity.slice(0, 20)}…)`;
    $("upd-diff-box").hidden = true;
    $("lib-detail").hidden = false;
  }

  async function planUpdate() {
    if (!lib.selected) {
      fail("Choose an agent to update first.");
      return;
    }
    const path = $("upd-path").value.trim();
    if (!path) {
      fail("Give the path to the new .rvf.");
      return;
    }
    const schema = Number.parseInt($("upd-schema").value, 10);
    clearError();
    try {
      lib.plan = await invoke("update_plan", {
        installId: lib.selected.record.installId,
        path,
        versionLabel: $("upd-version").value.trim() || "unlabelled",
        // Lineage: the plan is rejected unless this matches what is installed.
        predecessorIdentity: lib.selected.record.baseIdentity,
        stateSchemaVersion: Number.isNaN(schema) ? null : schema,
        rollbackUnsafe: $("upd-rollback-unsafe").checked,
      });
      renderPlan(lib.plan);
    } catch (err) {
      fail(err);
      lib.plan = null;
      $("upd-diff-box").hidden = true;
    }
    await refresh();
  }

  function renderPlan(plan) {
    const list = $("upd-diff");
    list.replaceChildren();
    for (const line of renderedLines(plan.diff)) {
      const li = document.createElement("li");
      li.textContent = line;
      list.append(li);
    }

    const note = $("upd-approval");
    note.textContent = plan.requires_approval
      ? `Approving this update grants: ${plan.approval_classes.join(", ")}.` +
        (plan.rollback_available_after
          ? ""
          : " This release declares rollback unsafe, so it cannot be undone.")
      : "No capability expansion; this release stays within the existing contract.";
    note.className = plan.requires_approval ? "status-unverified" : "hint";

    $("upd-diff-box").hidden = false;
  }

  // `PermissionDiff` serializes its data, not its rendering, so the P9 lines
  // are rebuilt here from the same fields the Rust `lines()` uses.
  function renderedLines(diff) {
    const lines = diff.changes.map((c) => c.text);
    if (diff.state_schema_change) {
      lines.push(
        `Changes memory schema from ${diff.state_schema_change.from} to ${diff.state_schema_change.to}`,
      );
    }
    if (!lines.length) {
      lines.push("No capability change; this release stays within the existing contract.");
    }
    return lines;
  }

  async function applyUpdate() {
    if (!lib.plan) {
      fail("Compute the difference first.");
      return;
    }
    const schema = Number.parseInt($("upd-schema").value, 10);
    await run(
      "update_apply",
      {
        installId: lib.plan.install_id,
        path: $("upd-path").value.trim(),
        versionLabel: lib.plan.version_label,
        predecessorIdentity: lib.plan.from_identity,
        // Exactly what the plan asked for. Nothing extra is smuggled in here.
        approvedClasses: lib.plan.approval_classes,
        stateSchemaVersion: Number.isNaN(schema) ? null : schema,
        rollbackUnsafe: $("upd-rollback-unsafe").checked,
      },
      (r) => `Updated to ${r.baseIdentity.slice(0, 20)}…`,
    );
    lib.plan = null;
    $("upd-diff-box").hidden = true;
  }

  async function rollback() {
    if (!lib.selected) {
      fail("Choose an agent first.");
      return;
    }
    await run(
      "update_rollback",
      { installId: lib.selected.record.installId },
      (r) => `Rolled back to ${r.baseIdentity.slice(0, 20)}…`,
    );
  }

  $("lib-refresh").addEventListener("click", refresh);
  $("upd-plan").addEventListener("click", planUpdate);
  $("upd-apply").addEventListener("click", applyUpdate);
  $("upd-rollback").addEventListener("click", rollback);

  const libStep = document.querySelector('.step[data-goto="library"]');
  if (libStep) libStep.addEventListener("click", refresh);

  // main.js calls this after a successful install.
  window.rvforgeLibrary = { refresh };
})();
