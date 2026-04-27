// Welcome modal: intro + three tutorial cards.
// Opens on every page load. Dismiss via X, ESC, backdrop, or CTA —
// dismissal only applies to the current page load; a reload brings
// it back. The "?" button in the topbar also reopens it within a
// session. Prior localStorage keys from earlier builds are cleaned
// up on mount so they don't silently suppress the modal.

(function () {
  const LEGACY_STORAGE_KEY = 'connectome-os.welcome.dismissed.v1';

  function mount() {
    // Drop any stale dismissal state from the previous build so the
    // modal reliably opens on every load.
    try {
      localStorage.removeItem(LEGACY_STORAGE_KEY);
    } catch {
      /* private-mode: best effort */
    }
    const root = document.getElementById('welcome-modal');
    if (!root) return;
    const closeBtn = root.querySelector('.welcome-close');
    const startBtn = root.querySelector('.welcome-start');
    const backdrop = root.querySelector('.welcome-backdrop');
    const reopenBtn = document.getElementById('welcome-reopen');

    function open() {
      root.classList.remove('welcome-closing');
      root.classList.add('welcome-open');
      root.setAttribute('aria-hidden', 'false');
      document.addEventListener('keydown', onKey);
    }

    function close() {
      // Trigger exit animation; the timeout finalises the aria state.
      // No persistence — reload brings the modal back.
      root.classList.remove('welcome-open');
      root.classList.add('welcome-closing');
      document.removeEventListener('keydown', onKey);
      // Animation timing — must match overlays.css welcome-fade-out.
      const FALLBACK_MS = 520;
      setTimeout(() => {
        if (root.classList.contains('welcome-closing')) {
          root.classList.remove('welcome-closing');
          root.setAttribute('aria-hidden', 'true');
        }
      }, FALLBACK_MS);
    }

    function onKey(e) {
      if (e.key === 'Escape') close();
    }

    closeBtn?.addEventListener('click', close);
    startBtn?.addEventListener('click', close);
    backdrop?.addEventListener('click', close);
    reopenBtn?.addEventListener('click', (e) => {
      e.preventDefault();
      open();
    });

    // Always open on load. Slight delay so the real-backend status
    // fetch can populate the banner before the modal claims focus.
    setTimeout(open, 350);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
