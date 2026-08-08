// Witness viewer frontend (requirements P15.11, ADR-295 §2 element 8).
//
// This screen is entirely system-owned chrome: nothing on it comes from an
// agent, and nothing on it is a verdict this file reached. `witness_chain`
// recomputes every content address and re-checks every link in Rust; the code
// below only renders what it answered.
//
// Two rules follow from that:
//
// 1. A chain reads as good only when Rust said `valid`. Every other label —
//    `broken-at-N`, `empty`, `unchained` — renders as unverified, never as
//    neutral. There is no branch here that softens a status.
// 2. Where a chain broke is shown on the receipt that broke it, in place, with
//    the reason Rust gave. "Invalid" without a position would hide that the
//    records before it did verify.
//
// Wrapped in an IIFE so its bindings do not collide with main.js and dock.js,
// which share the classic-script global scope.

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

  function fail(err) {
    const box = $("error");
    box.textContent = String(err);
    box.hidden = false;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  async function load(path) {
    $("error").hidden = true;
    try {
      render(await invoke("witness_chain", { path: path || null }));
    } catch (err) {
      fail(err);
    }
  }

  function render(view) {
    const ok = view.label === "valid";

    $("wit-banner").hidden = false;
    $("wit-banner").className = `wit-banner ${ok ? "ok" : "unverified"}`;
    $("wit-label").textContent = view.label;
    $("wit-summary").textContent = view.summary;

    $("wit-source").hidden = false;
    $("wit-source").textContent = view.source;

    const container = $("wit-subjects");
    container.replaceChildren();
    for (const subject of view.subjects) {
      container.append(renderSubject(subject));
    }
    if (!view.subjects.length) {
      container.append(
        el("p", "hint", "No receipts have been recorded for any subject."),
      );
    }

    const malformed = $("wit-malformed");
    malformed.replaceChildren();
    for (const line of view.malformed) {
      malformed.append(
        el("li", null, `line ${line.line}: ${line.reason}`),
      );
    }
    $("wit-malformed-box").hidden = view.malformed.length === 0;
  }

  function renderSubject(subject) {
    const valid = subject.label === "valid";
    const section = el("section", "wit-subject");

    const head = el("div", "wit-subject-head");
    head.append(el("span", "mono", subject.subject_short));
    head.append(
      el("span", `wit-chip ${valid ? "ok" : "unverified"}`, subject.label),
    );
    head.append(
      el(
        "span",
        "hint",
        subject.receipts.length === 1 ? "1 receipt" : `${subject.receipts.length} receipts`,
      ),
    );
    section.append(head);

    const list = el("ol", "wit-chain");
    for (const receipt of subject.receipts) {
      list.append(renderReceipt(receipt));
    }
    section.append(list);
    return section;
  }

  function renderReceipt(receipt) {
    const item = el("li", receipt.break_reason ? "wit-receipt broken" : "wit-receipt");

    const head = el("div", "wit-receipt-head");
    head.append(el("span", "wit-event", receipt.event));
    head.append(
      el(
        "span",
        `wit-outcome ${receipt.outcome === "pass" ? "ok" : "unverified"}`,
        receipt.outcome,
      ),
    );
    head.append(el("span", "hint", receipt.actor));
    head.append(el("span", "hint", receipt.timestamp));
    head.append(el("span", "mono id", receipt.id_short));
    item.append(head);

    if (receipt.detail) item.append(el("p", "wit-detail", receipt.detail));

    // An unchained record is an audit entry, not a link. Saying so on the row
    // is the difference between "not verified" and "verified fine".
    if (!receipt.chained) {
      item.append(
        el("p", "wit-note", "Local receipt — carries no content address, so no link to check."),
      );
    }

    if (receipt.break_reason) {
      item.append(
        el("p", "wit-break", `Chain stops verifying here: ${receipt.break_reason}`),
      );
    }
    return item;
  }

  $("wit-load").addEventListener("click", () => load($("wit-path").value.trim()));
  $("wit-load-local").addEventListener("click", () => {
    $("wit-path").value = "";
    load(null);
  });
  $("wit-path").addEventListener("keydown", (e) => {
    if (e.key === "Enter") load($("wit-path").value.trim());
  });

  const step = document.querySelector('.step[data-goto="witness"]');
  if (step) step.addEventListener("click", () => load($("wit-path").value.trim()));
})();
