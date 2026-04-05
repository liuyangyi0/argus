/* Sound alerts, browser notifications, and title flash for Argus dashboard.
 *
 * Polls /api/alerts/json?severity=high&limit=1 to detect new HIGH alerts.
 * Triggers: audio beep, desktop notification, title bar flash.
 */
(function() {
    'use strict';

    var _lastAlertId = null;
    var _flashInterval = null;
    var _originalTitle = document.title;
    var _soundEnabled = localStorage.getItem('argus_sound') !== 'off';
    var _notifyEnabled = localStorage.getItem('argus_notify') !== 'off';

    // Audio context for alert beep (no external file needed)
    function playAlertBeep() {
        if (!_soundEnabled) return;
        try {
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            // Three short beeps
            [0, 0.2, 0.4].forEach(function(offset) {
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.frequency.value = 880;
                osc.type = 'square';
                gain.gain.value = 0.15;
                osc.start(ctx.currentTime + offset);
                osc.stop(ctx.currentTime + offset + 0.12);
            });
        } catch(e) { /* Audio not available */ }
    }

    function showDesktopNotification(alertId, cameraId, score) {
        if (!_notifyEnabled) return;
        if (!('Notification' in window)) return;
        if (Notification.permission === 'granted') {
            new Notification('Argus - 高严重度告警', {
                body: cameraId + ' | 异常分数: ' + score,
                tag: 'argus-high-alert',
                requireInteraction: true
            });
        } else if (Notification.permission !== 'denied') {
            Notification.requestPermission();
        }
    }

    function startTitleFlash() {
        if (_flashInterval) return;
        var flash = false;
        _flashInterval = setInterval(function() {
            document.title = flash ? '!! 高告警 !!' : _originalTitle;
            flash = !flash;
        }, 1000);
    }

    function stopTitleFlash() {
        if (_flashInterval) {
            clearInterval(_flashInterval);
            _flashInterval = null;
            document.title = _originalTitle;
        }
    }

    // Stop flashing when user focuses window
    window.addEventListener('focus', stopTitleFlash);

    // Poll for new HIGH alerts
    function checkForHighAlerts() {
        fetch('/api/alerts/json?severity=high&limit=1')
            .then(function(r) { return r.ok ? r.json() : []; })
            .then(function(alerts) {
                if (!alerts || alerts.length === 0) return;
                var latest = alerts[0];
                if (latest.alert_id !== _lastAlertId) {
                    _lastAlertId = latest.alert_id;
                    playAlertBeep();
                    showDesktopNotification(latest.alert_id, latest.camera_id, latest.anomaly_score);
                    if (!document.hasFocus()) {
                        startTitleFlash();
                    }
                }
            })
            .catch(function() { /* ignore fetch errors */ });
    }

    // Request notification permission on first interaction
    document.addEventListener('click', function _once() {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
        document.removeEventListener('click', _once);
    });

    // Start polling (every 5 seconds)
    setInterval(checkForHighAlerts, 5000);
    // Initial check
    setTimeout(checkForHighAlerts, 2000);

    // Expose settings toggle
    window.argusToggleSound = function() {
        _soundEnabled = !_soundEnabled;
        localStorage.setItem('argus_sound', _soundEnabled ? 'on' : 'off');
        window.showToast(_soundEnabled ? '声音告警已开启' : '声音告警已关闭', 'info');
    };
    window.argusToggleNotify = function() {
        _notifyEnabled = !_notifyEnabled;
        localStorage.setItem('argus_notify', _notifyEnabled ? 'on' : 'off');
        window.showToast(_notifyEnabled ? '桌面通知已开启' : '桌面通知已关闭', 'info');
    };
})();
