// Welcome modal: first-visit intro + three tutorial cards.
// Dismiss via the X button, ESC, the backdrop, or the primary CTA.
// Remembers the choice in localStorage so repeat visitors aren't
// interrupted — but a small "?" button (#welcome-reopen) in the topbar
// reopens it on demand.

(function () {
  const STORAGE_KEY = 'connectome-os.welcome.dismissed.v1';

  function isDismissed() {
    try {
      return localStorage.getItem(STORAGE_KEY) === '1';
    } catch {
      return false;
    }
  }

  function markDismissed() {
    try {
      localStorage.setItem(STORAGE_KEY, '1');
    } catch {
      /* private mode — best effort */
    }
  }

  function clearDismissed() {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }

  function mount() {
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
      // Trigger exit animation; the 'animationend' listener removes
      // the node from interaction.
      root.classList.remove('welcome-open');
      root.classList.add('welcome-closing');
      document.removeEventListener('keydown', onKey);
      markDismissed();
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
      clearDismissed();
      open();
    });

    if (!isDismissed()) {
      // Slight delay so the real-backend status fetch can populate
      // the banner before the modal claims focus.
      setTimeout(open, 350);
    } else {
      root.setAttribute('aria-hidden', 'true');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
