/**
 * ui.js - UI controller for the VWM viewer overlay
 *
 * Wires up DOM controls (time slider, search, status indicators)
 * and provides methods for the render loop to push live stats.
 */

export class UIController {
  constructor() {
    // Grab DOM elements
    this.timeSlider = document.getElementById('time-slider');
    this.timeLabel = document.getElementById('time-label');
    this.fpsDisplay = document.getElementById('fps-value');
    this.gaussianCountDisplay = document.getElementById('gaussian-count');
    this.coherenceIndicator = document.getElementById('coherence-state');
    this.searchBox = document.getElementById('search-box');
    this.statusText = document.getElementById('status-text');

    // State
    this._normalizedTime = 0; // 0..1
    this._playing = true;
    this._playSpeed = 1.0;
    this._searchQuery = '';
    this._onSearchChange = null;
    this._onTimeChange = null;

    // FPS rolling average
    this._frameTimes = [];

    this._bindEvents();
  }

  // ---- Public API ----

  /** Current normalized time [0, 1). */
  get normalizedTime() {
    return this._normalizedTime;
  }

  /** Whether animation is playing. */
  get playing() {
    return this._playing;
  }

  /** Current search filter string (lowercase). */
  get searchQuery() {
    return this._searchQuery;
  }

  /** Register callback for search query changes. */
  onSearchChange(fn) {
    this._onSearchChange = fn;
  }

  /** Register callback for time scrub changes. */
  onTimeChange(fn) {
    this._onTimeChange = fn;
  }

  /** Update the time slider externally (e.g. from animation loop). */
  setTime(t) {
    this._normalizedTime = t;
    if (this.timeSlider) {
      this.timeSlider.value = Math.round(t * 1000);
    }
    if (this.timeLabel) {
      this.timeLabel.textContent = `t=${t.toFixed(3)}`;
    }
  }

  /** Push a frame timestamp to compute FPS. */
  recordFrame(nowMs) {
    this._frameTimes.push(nowMs);
    // Keep last 60 frames
    if (this._frameTimes.length > 60) this._frameTimes.shift();
    if (this._frameTimes.length > 1) {
      const dt =
        (this._frameTimes[this._frameTimes.length - 1] - this._frameTimes[0]) /
        (this._frameTimes.length - 1);
      const fps = 1000 / dt;
      if (this.fpsDisplay) {
        this.fpsDisplay.textContent = fps.toFixed(1);
      }
    }
  }

  /** Set the displayed Gaussian count. */
  setGaussianCount(n) {
    if (this.gaussianCountDisplay) {
      this.gaussianCountDisplay.textContent = n.toLocaleString();
    }
  }

  /** Update the coherence state indicator. */
  setCoherenceState(state) {
    if (!this.coherenceIndicator) return;
    this.coherenceIndicator.textContent = state;
    this.coherenceIndicator.className = 'coherence-badge';
    if (state === 'coherent') {
      this.coherenceIndicator.classList.add('coherent');
    } else if (state === 'degraded') {
      this.coherenceIndicator.classList.add('degraded');
    } else {
      this.coherenceIndicator.classList.add('unknown');
    }
  }

  /** Set status bar text. */
  setStatus(text) {
    if (this.statusText) this.statusText.textContent = text;
  }

  /** Toggle play/pause. */
  togglePlay() {
    this._playing = !this._playing;
    const btn = document.getElementById('play-btn');
    if (btn) btn.textContent = this._playing ? 'Pause' : 'Play';
  }

  // ---- Internal ----

  _bindEvents() {
    // Time slider
    if (this.timeSlider) {
      this.timeSlider.addEventListener('input', () => {
        this._normalizedTime = parseInt(this.timeSlider.value, 10) / 1000;
        this._playing = false;
        const btn = document.getElementById('play-btn');
        if (btn) btn.textContent = 'Play';
        if (this.timeLabel) {
          this.timeLabel.textContent = `t=${this._normalizedTime.toFixed(3)}`;
        }
        if (this._onTimeChange) this._onTimeChange(this._normalizedTime);
      });
    }

    // Play button
    const playBtn = document.getElementById('play-btn');
    if (playBtn) {
      playBtn.addEventListener('click', () => this.togglePlay());
    }

    // Search box
    if (this.searchBox) {
      this.searchBox.addEventListener('input', () => {
        this._searchQuery = this.searchBox.value.trim().toLowerCase();
        if (this._onSearchChange) this._onSearchChange(this._searchQuery);
      });
    }
  }
}
