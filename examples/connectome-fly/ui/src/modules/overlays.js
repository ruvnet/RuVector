/* Overlay system: toasts, modals, confirm, command palette */

(function () {
  'use strict';

  // ===== HOSTS =====
  function ensureHost(id, className) {
    let el = document.getElementById(id);
    if (!el) {
      el = document.createElement('div');
      el.id = id;
      if (className) el.className = className;
      document.body.appendChild(el);
    }
    return el;
  }
  const toastHost = ensureHost('toast-host');
  const modalHost = ensureHost('modal-host');
  const cmdHost = ensureHost('cmd-host');

  // ===== TOASTS =====
  const TOAST_ICONS = {
    info: 'i', success: '✓', warn: '!', error: '×'
  };

  function toast(opts) {
    if (typeof opts === 'string') opts = { title: opts };
    const { type = 'info', title = '', desc = '', duration = 3800, action } = opts;
    const el = document.createElement('div');
    el.className = 'toast ' + type;
    el.innerHTML = `
      <span class="t-icon">${TOAST_ICONS[type] || 'i'}</span>
      <div class="t-body">
        <div class="t-title"></div>
        ${desc ? '<div class="t-desc"></div>' : ''}
      </div>
      <button class="t-close" aria-label="Dismiss">
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M2 2l6 6M8 2l-6 6"/></svg>
      </button>
      ${action ? '<div class="t-action"></div>' : ''}
    `;
    el.querySelector('.t-title').textContent = title;
    if (desc) el.querySelector('.t-desc').textContent = desc;
    if (action) {
      const wrap = el.querySelector('.t-action');
      (Array.isArray(action) ? action : [action]).forEach(a => {
        const b = document.createElement('button');
        b.textContent = a.label;
        b.addEventListener('click', () => { a.onClick && a.onClick(); close(); });
        wrap.appendChild(b);
      });
    }
    toastHost.appendChild(el);
    let closed = false;
    let timer = null;
    function close() {
      if (closed) return;
      closed = true;
      clearTimeout(timer);
      el.classList.add('closing');
      setTimeout(() => el.remove(), 220);
    }
    el.querySelector('.t-close').addEventListener('click', close);
    if (duration > 0) timer = setTimeout(close, duration);
    // pause on hover
    el.addEventListener('mouseenter', () => { if (timer) clearTimeout(timer); });
    el.addEventListener('mouseleave', () => { if (duration > 0 && !closed) timer = setTimeout(close, 1500); });
    return { close };
  }

  // ===== MODAL =====
  let currentModal = null;

  function showModal(opts) {
    closeModal();
    const { num, title, body, footer, wide, onClose } = opts;
    const modal = document.createElement('div');
    modal.className = 'modal' + (wide ? ' wide' : '');
    const numHtml = num ? `<span class="m-num">${num}</span>` : '';
    modal.innerHTML = `
      <div class="m-head">
        <div class="m-title">${numHtml}<span class="m-title-text"></span></div>
        <button class="m-close" aria-label="Close">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M3 3l6 6M9 3l-6 6"/></svg>
        </button>
      </div>
      <div class="m-body"></div>
      ${footer ? '<div class="m-foot"></div>' : ''}
    `;
    modal.querySelector('.m-title-text').textContent = title || '';
    const bodyEl = modal.querySelector('.m-body');
    if (typeof body === 'string') bodyEl.innerHTML = body;
    else if (body instanceof Node) bodyEl.appendChild(body);
    if (footer) {
      const footEl = modal.querySelector('.m-foot');
      (Array.isArray(footer) ? footer : [footer]).forEach(f => {
        const b = document.createElement('button');
        b.className = 'm-btn' + (f.variant ? ' ' + f.variant : '');
        b.textContent = f.label;
        b.addEventListener('click', () => {
          if (f.onClick) f.onClick();
          if (f.close !== false) closeModal();
        });
        footEl.appendChild(b);
      });
    }
    const backdrop = document.createElement('div');
    backdrop.className = 'backdrop';
    backdrop.addEventListener('click', closeModal);
    modalHost.innerHTML = '';
    modalHost.appendChild(backdrop);
    modalHost.appendChild(modal);
    modalHost.classList.add('open');
    modal.querySelector('.m-close').addEventListener('click', closeModal);
    currentModal = { el: modal, onClose };
    return { close: closeModal };
  }

  function closeModal() {
    if (!currentModal) return;
    const o = currentModal.onClose;
    currentModal = null;
    modalHost.classList.remove('open');
    modalHost.innerHTML = '';
    if (o) o();
  }

  function confirm(opts) {
    return new Promise((resolve) => {
      showModal({
        num: opts.num,
        title: opts.title || 'Confirm',
        body: `<p>${opts.message || 'Are you sure?'}</p>`,
        footer: [
          { label: opts.cancelLabel || 'Cancel', onClick: () => resolve(false) },
          {
            label: opts.confirmLabel || 'Confirm',
            variant: opts.danger ? 'danger' : 'primary',
            onClick: () => resolve(true)
          }
        ],
        onClose: () => resolve(false)
      });
    });
  }

  // ===== COMMAND PALETTE =====
  let cmdOpen = false;
  let cmdIndex = 0;
  let cmdFiltered = [];
  const CMDS = [];

  function registerCmd(c) {
    CMDS.push(c);
  }

  function openCmd() {
    if (cmdOpen) return;
    cmdOpen = true;
    cmdHost.innerHTML = `
      <div class="backdrop"></div>
      <div class="cmd">
        <input type="text" placeholder="Search commands…" autocomplete="off" spellcheck="false" />
        <div class="cmd-list"></div>
        <div class="cmd-foot">
          <span><span class="cmd-kbd">↑↓</span> navigate</span>
          <span><span class="cmd-kbd">↵</span> run</span>
          <span><span class="cmd-kbd">ESC</span> close</span>
        </div>
      </div>
    `;
    cmdHost.classList.add('open');
    const input = cmdHost.querySelector('input');
    const list = cmdHost.querySelector('.cmd-list');
    cmdHost.querySelector('.backdrop').addEventListener('click', closeCmd);
    input.addEventListener('input', () => renderCmd(input.value));
    input.addEventListener('keydown', handleCmdKey);
    cmdIndex = 0;
    renderCmd('');
    setTimeout(() => input.focus(), 10);
  }

  function closeCmd() {
    cmdOpen = false;
    cmdHost.classList.remove('open');
    cmdHost.innerHTML = '';
  }

  function renderCmd(query) {
    const q = query.toLowerCase().trim();
    cmdFiltered = q ? CMDS.filter(c =>
      c.label.toLowerCase().includes(q) ||
      (c.sub || '').toLowerCase().includes(q) ||
      (c.keywords || []).some(k => k.toLowerCase().includes(q))
    ) : CMDS.slice();
    cmdIndex = Math.min(cmdIndex, Math.max(0, cmdFiltered.length - 1));
    const list = cmdHost.querySelector('.cmd-list');
    if (!cmdFiltered.length) {
      list.innerHTML = '<div class="cmd-empty">No commands found</div>';
      return;
    }
    list.innerHTML = cmdFiltered.map((c, i) => `
      <div class="cmd-item${i === cmdIndex ? ' sel' : ''}" data-idx="${i}">
        <span class="cmd-icon">${c.icon || '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="4"/></svg>'}</span>
        <div>
          <div class="cmd-label">${escapeHtml(c.label)}</div>
          ${c.sub ? `<div class="cmd-sub">${escapeHtml(c.sub)}</div>` : ''}
        </div>
        ${c.kbd ? `<span class="cmd-kbd">${c.kbd}</span>` : '<span></span>'}
      </div>
    `).join('');
    list.querySelectorAll('.cmd-item').forEach(el => {
      el.addEventListener('mouseenter', () => {
        cmdIndex = parseInt(el.dataset.idx, 10);
        updateSel();
      });
      el.addEventListener('click', () => runCmd(parseInt(el.dataset.idx, 10)));
    });
  }

  function updateSel() {
    cmdHost.querySelectorAll('.cmd-item').forEach((el, i) => {
      el.classList.toggle('sel', i === cmdIndex);
    });
    const sel = cmdHost.querySelector('.cmd-item.sel');
    if (sel) sel.scrollIntoView({ block: 'nearest' });
  }

  function handleCmdKey(e) {
    if (e.key === 'Escape') { closeCmd(); e.preventDefault(); }
    else if (e.key === 'ArrowDown') { cmdIndex = Math.min(cmdFiltered.length - 1, cmdIndex + 1); updateSel(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { cmdIndex = Math.max(0, cmdIndex - 1); updateSel(); e.preventDefault(); }
    else if (e.key === 'Enter') { runCmd(cmdIndex); e.preventDefault(); }
  }

  function runCmd(idx) {
    const c = cmdFiltered[idx];
    if (!c) return;
    closeCmd();
    setTimeout(() => { try { c.action(); } catch (err) { console.error(err); toast({ type: 'error', title: 'Command failed', desc: String(err.message || err) }); } }, 10);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }

  // ===== GLOBAL KEYS =====
  document.addEventListener('keydown', (e) => {
    // cmd-k / ctrl-k
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      if (cmdOpen) closeCmd(); else openCmd();
      return;
    }
    // esc closes modal
    if (e.key === 'Escape') {
      if (cmdOpen) { closeCmd(); return; }
      if (currentModal) { closeModal(); return; }
    }
  });

  // ===== EXPORT =====
  window.OS = window.OS || {};
  window.OS.toast = toast;
  window.OS.modal = showModal;
  window.OS.closeModal = closeModal;
  window.OS.confirm = confirm;
  window.OS.openCmd = openCmd;
  window.OS.registerCmd = registerCmd;

})();
