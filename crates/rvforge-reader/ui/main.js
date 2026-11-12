// RVForge Reader frontend.
//
// No framework, no build step, no network. The three screens render exactly
// what the Rust commands return; the frontend makes no permission decisions of
// its own and never softens a status.

const invoke = (cmd, args) => {
  const api = window.__TAURI__;
  if (!api) {
    return Promise.reject(new Error("Tauri API unavailable (open this page through the app)."));
  }
  return api.core.invoke(cmd, args);
};

const $ = (id) => document.getElementById(id);
const state = { path: null, card: null, offer: null, verified: false, installed: null };

function showError(message) {
  const box = $("error");
  box.textContent = message;
  box.hidden = !message;
}

function goto(screen) {
  for (const el of document.querySelectorAll(".screen")) {
    el.classList.toggle("active", el.id === `screen-${screen}`);
  }
  for (const btn of document.querySelectorAll(".step")) {
    btn.setAttribute("aria-current", btn.dataset.goto === screen ? "true" : "false");
  }
}

function enableStep(screen) {
  const btn = document.querySelector(`.step[data-goto="${screen}"]`);
  if (btn) btn.disabled = false;
}

function fillNotes(id, notes) {
  const ul = $(id);
  ul.replaceChildren();
  for (const note of notes || []) {
    const li = document.createElement("li");
    li.textContent = note;
    ul.append(li);
  }
}

function humanBytes(n) {
  if (!n) return "0 bytes";
  const units = ["bytes", "KiB", "MiB", "GiB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${i === 0 ? v : v.toFixed(1)} ${units[i]}`;
}

// Screen 1 — open and verify.

async function pickFile() {
  const api = window.__TAURI__;
  if (!api || !api.dialog) {
    showError("File picker unavailable; type a path instead.");
    return;
  }
  const chosen = await api.dialog.open({
    multiple: false,
    filters: [{ name: "RVForge agent package", extensions: ["rvf"] }],
  });
  if (typeof chosen === "string") {
    $("path").value = chosen;
    await runInspect();
  }
}

async function runInspect() {
  const path = $("path").value.trim();
  if (!path) {
    showError("Choose a .rvf file or type a path.");
    return;
  }
  showError("");
  try {
    const summary = await invoke("inspect_rvf", { path });
    state.path = path;
    state.verified = false;
    renderInspection(summary);
    // Reviewing capabilities stays closed until verification has actually run:
    // there is nothing to permit until the package is known to be intact.
    $("to-capability").disabled = true;
  } catch (err) {
    showError(String(err));
  }
}

async function runVerify() {
  if (!state.path) {
    showError("Inspect a package first.");
    return;
  }
  showError("");
  try {
    const outcome = await invoke("verify_rvf", { path: state.path });
    state.verified = outcome.summary.verification === "verified";
    renderInspection(outcome.summary);
    if (state.verified) {
      $("to-capability").disabled = false;
      $("to-capability").title = "";
      enableStep("capability");
    } else {
      showError("This package did not verify. It cannot be installed.");
    }
  } catch (err) {
    showError(String(err));
  }
}

function renderInspection(s) {
  $("f-name").textContent = s.file_name || s.path;
  $("f-size").textContent = s.exists ? humanBytes(s.size_bytes) : "file not found";
  $("f-identity").textContent = s.identity || "not a readable container";
  $("f-publisher").textContent = s.publisher || "unknown";
  $("f-segments").textContent = s.segments.length
    ? s.segments.map((seg) => seg.segment_type).join(", ")
    : "none read";
  $("f-capabilities").textContent = s.declared_capabilities.length
    ? s.declared_capabilities.join(", ")
    : "nothing (default deny)";

  // Anything short of a completed, passing check is rendered as a warning.
  // "not-checked" is a warning too: a signature nobody verified is not a
  // signature the user can rely on.
  const sig = $("f-signature");
  sig.textContent = s.signature;
  sig.className = s.signature === "verified" ? "status-ok" : "status-unverified";

  const ver = $("f-verification");
  ver.textContent = s.verification;
  ver.className = s.verification === "verified" ? "status-ok" : "status-unverified";

  fillNotes("f-notes", s.notes);
  $("open-result").hidden = false;
}

// Screen 2 — the P6 capability contract.

async function showCapabilities() {
  if (!state.path) {
    showError("Inspect a package first.");
    return;
  }
  if (!state.verified) {
    showError("Verify the package first. An unverified package has no capability card.");
    return;
  }
  showError("");
  try {
    // `install_offer` verifies the same bytes the install will copy and reports
    // exactly which classes the install will accept, so the screen cannot show
    // one contract and submit another.
    const offer = await invoke("install_offer", { path: state.path });
    state.offer = offer;
    state.card = offer.card;
    renderOffer(offer);
    goto("capability");
    enableStep("runtime");
  } catch (err) {
    showError(String(err));
  }
}

function renderOffer(offer) {
  $("cap-subject").textContent = state.path;
  const card = offer.card;

  // One checkbox per requested class. Clearing one narrows the grant; there is
  // no control that widens it, because the install would refuse anyway.
  const requests = $("cap-requests");
  requests.replaceChildren();
  if (!card.requests.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Nothing is granted.";
    requests.append(li);
  } else {
    for (const line of card.requests) {
      const li = document.createElement("li");
      const label = document.createElement("label");
      label.className = "inline";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = true;
      box.dataset.class = line.class;
      box.className = "grant";
      const text = document.createElement("span");
      text.textContent = line.text;
      label.append(box, text);
      li.append(label);
      requests.append(li);
    }
  }

  const cannot = $("cap-cannot");
  cannot.replaceChildren();
  for (const line of card.cannot) {
    const li = document.createElement("li");
    li.textContent = line.text;
    cannot.append(li);
  }

  const notes = [...(card.notes || [])];
  for (const trigger of card.manual_review_triggers || []) {
    notes.push(`Manual review trigger: ${trigger}`);
  }
  if (offer.runtime_profile) {
    notes.push(`Runtime selected for this host: ${offer.runtime_profile} (${offer.isolation_claim}).`);
  }
  if (offer.refusal) notes.push(offer.refusal);
  fillNotes("cap-notes", notes);

  if (!$("install-name").value) {
    $("install-name").value = ($("f-name").textContent || "Agent").replace(/\.rvf$/i, "");
  }

  // The refusal is the Rust side's, rendered rather than re-decided here.
  $("install").disabled = Boolean(offer.refusal);
  $("install").title = offer.refusal || "";
}

async function runInstall() {
  if (!state.offer || state.offer.refusal) {
    showError(state.offer?.refusal || "Review the capability contract first.");
    return;
  }
  const acceptedClasses = [...document.querySelectorAll("input.grant:checked")].map(
    (box) => box.dataset.class,
  );
  showError("");
  try {
    const entry = await invoke("install_agent", {
      path: state.path,
      name: $("install-name").value.trim() || "Unnamed agent",
      acceptedClasses,
      rollbackUnsafe: $("install-rollback-unsafe").checked,
    });
    state.installed = entry.record.installId;
    enableStep("library");
    goto("library");
    window.rvforgeLibrary?.refresh();
  } catch (err) {
    showError(String(err));
  }
}

// Screen 3 — runtime status.

async function showRuntime() {
  if (!state.path) {
    showError("Inspect a package first.");
    return;
  }
  showError("");
  try {
    const choice = await invoke("runtime_selection", { path: state.path });
    renderRuntime(choice);
    goto("runtime");
  } catch (err) {
    showError(String(err));
  }
}

function renderRuntime(c) {
  const unsupported = c.status === "unsupported";
  $("rt-profile").textContent = unsupported
    ? "unsupported — this host cannot run the agent"
    : c.profile;
  $("rt-profile").className = unsupported ? "status-unverified" : "";
  $("rt-isolation").textContent = c.isolation_claim || "none";
  $("rt-mechanisms").textContent = c.mechanisms.length
    ? c.mechanisms.join(", ")
    : "none engaged";
  $("rt-order").textContent = c.order.join(" → ");
  $("rt-policy").textContent =
    c.policy_source === "embedded-default"
      ? "embedded default (no signed policy override)"
      : c.policy_source;
  // The witness line is read from the verified chain, not written here. Until
  // it answers, the field says so rather than showing a status nobody checked.
  $("rt-witness").textContent = "reading the witness chain…";
  $("rt-witness").className = "status-unverified";
  invoke("witness_chain", { path: null })
    .then((chain) => {
      $("rt-witness").textContent = chain.summary;
      $("rt-witness").className =
        chain.label === "valid" ? "status-ok" : "status-unverified";
    })
    .catch((err) => {
      $("rt-witness").textContent = `the witness chain could not be read: ${err}`;
      $("rt-witness").className = "status-unverified";
    });

  const body = $("rt-eval");
  body.replaceChildren();
  for (const e of c.evaluated) {
    const tr = document.createElement("tr");
    const profile = document.createElement("td");
    profile.textContent = e.profile;
    const eligible = document.createElement("td");
    eligible.textContent = e.eligible ? "yes" : "no";
    eligible.className = e.eligible ? "yes" : "no";
    const reason = document.createElement("td");
    reason.textContent = e.reason;
    tr.append(profile, eligible, reason);
    body.append(tr);
  }
}

// Wiring.

$("pick").addEventListener("click", pickFile);
$("inspect").addEventListener("click", runInspect);
$("verify").addEventListener("click", runVerify);
$("path").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runInspect();
});
$("to-capability").addEventListener("click", showCapabilities);
$("cancel-install").addEventListener("click", () => {
  state.card = null;
  state.offer = null;
  goto("open");
});
$("install").addEventListener("click", runInstall);

for (const btn of document.querySelectorAll(".step")) {
  btn.addEventListener("click", () => {
    const target = btn.dataset.goto;
    if (target === "capability") showCapabilities();
    else if (target === "runtime") showRuntime();
    else goto(target);
  });
}
