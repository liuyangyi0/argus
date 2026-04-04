/**
 * Browser audio alert system for nuclear plant operators.
 * Plays a warning tone when HIGH severity alerts are detected.
 * Uses Web Audio API for reliable playback without external audio files.
 */

class AlertAudioSystem {
    constructor() {
        this.audioContext = null;
        this.enabled = true;
        this.lastAlertTime = 0;
        this.cooldownMs = 5000; // Don't repeat within 5 seconds
        this._originalTitle = document.title;
        this._flashInterval = null;
    }

    /**
     * Create AudioContext on first user interaction (browser autoplay policy).
     */
    init() {
        if (this.audioContext) return;
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.warn('Web Audio API not available:', e);
        }
    }

    /**
     * Generate warning beeps using oscillator.
     * HIGH:   three rapid beeps at 800Hz
     * MEDIUM: two beeps at 600Hz
     * LOW:    single beep at 400Hz
     */
    playWarning(severity) {
        if (!this.enabled) return;

        var now = Date.now();
        if (now - this.lastAlertTime < this.cooldownMs) return;
        this.lastAlertTime = now;

        // Ensure AudioContext is ready
        this.init();
        if (!this.audioContext) return;

        // Resume if suspended (browser policy)
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }

        var freq, beepCount;
        switch (severity) {
            case 'high':
                freq = 800;
                beepCount = 3;
                break;
            case 'medium':
                freq = 600;
                beepCount = 2;
                break;
            default:
                freq = 400;
                beepCount = 1;
                break;
        }

        this._playBeeps(freq, beepCount);
    }

    _playBeeps(freq, count) {
        var ctx = this.audioContext;
        var beepDuration = 0.15;  // seconds
        var gapDuration = 0.1;    // seconds between beeps
        var startTime = ctx.currentTime;

        for (var i = 0; i < count; i++) {
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.type = 'square';
            osc.frequency.value = freq;

            // Envelope: quick attack, sustain, quick release
            var t = startTime + i * (beepDuration + gapDuration);
            gain.gain.setValueAtTime(0, t);
            gain.gain.linearRampToValueAtTime(0.3, t + 0.01);
            gain.gain.setValueAtTime(0.3, t + beepDuration - 0.02);
            gain.gain.linearRampToValueAtTime(0, t + beepDuration);

            osc.start(t);
            osc.stop(t + beepDuration);
        }
    }

    /**
     * Flash browser tab title to draw operator attention.
     * Alternates between alert message and original title.
     */
    flashTitle(message) {
        // Clear any existing flash
        this.stopFlashTitle();

        var original = this._originalTitle;
        var showing = false;
        this._flashInterval = setInterval(function() {
            showing = !showing;
            document.title = showing ? message : original;
        }, 800);

        // Stop flashing after 10 seconds
        var self = this;
        setTimeout(function() {
            self.stopFlashTitle();
        }, 10000);
    }

    stopFlashTitle() {
        if (this._flashInterval) {
            clearInterval(this._flashInterval);
            this._flashInterval = null;
            document.title = this._originalTitle;
        }
    }

    toggle() {
        this.enabled = !this.enabled;
        this._updateToggleButton();
        return this.enabled;
    }

    _updateToggleButton() {
        var btn = document.getElementById('audio-toggle-btn');
        if (btn) {
            btn.textContent = this.enabled ? '\uD83D\uDD0A' : '\uD83D\uDD07';
            btn.title = this.enabled ? '\u5173\u95ED\u58F0\u97F3\u544A\u8B66' : '\u5F00\u542F\u58F0\u97F3\u544A\u8B66';
        }
    }
}

// Global instance
var alertAudio = new AlertAudioSystem();

// Initialize AudioContext on first user interaction
document.addEventListener('click', function() { alertAudio.init(); }, { once: true });
document.addEventListener('keydown', function() { alertAudio.init(); }, { once: true });

// Listen for HTMX events that indicate new alerts
document.body.addEventListener('htmx:afterSwap', function(evt) {
    var elt = evt.detail.elt;
    if (!elt || !elt.querySelectorAll) return;

    var highBadges = elt.querySelectorAll('.badge-high');
    if (highBadges.length > 0) {
        alertAudio.playWarning('high');
        alertAudio.flashTitle('\u26A0\uFE0F \u9AD8\u7EA7\u544A\u8B66');
        return;
    }

    var medBadges = elt.querySelectorAll('.badge-medium');
    if (medBadges.length > 0) {
        alertAudio.playWarning('medium');
    }
});
