/* Toast notification system for Argus dashboard.
 * Supports standard toasts + undo toasts with countdown.
 *
 * Server usage:
 *   headers = {"HX-Trigger": JSON.stringify({showToast: {message: "...", type: "success"}})}
 *
 * Undo toast (from JS):
 *   window.showUndoToast("已标记为误报", "/api/alerts/ALT-xxx/workflow", {status: "new"});
 */
(function() {
    var container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    function showToast(message, type) {
        type = type || 'info';
        var toast = document.createElement('div');
        toast.className = 'toast ' + type;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(function() {
            toast.classList.add('fade-out');
            setTimeout(function() { toast.remove(); }, 300);
        }, 3000);
    }

    function showUndoToast(message, undoUrl, undoData, timeoutMs) {
        timeoutMs = timeoutMs || 5000;
        var toast = document.createElement('div');
        toast.className = 'toast info';
        toast.style.display = 'flex';
        toast.style.justifyContent = 'space-between';
        toast.style.alignItems = 'center';
        toast.style.gap = 'var(--space-3)';

        var textSpan = document.createElement('span');
        textSpan.textContent = message;

        var undoBtn = document.createElement('button');
        undoBtn.textContent = '撤销';
        undoBtn.style.cssText = 'background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.3);' +
            'color:#fff;padding:2px 12px;border-radius:4px;cursor:pointer;font-size:13px;white-space:nowrap;';

        var cancelled = false;
        undoBtn.onclick = function() {
            cancelled = true;
            toast.remove();
            if (undoUrl) {
                fetch(undoUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(undoData || {}),
                }).then(function() {
                    showToast('已撤销', 'success');
                    if (window.htmx) {
                        htmx.ajax('GET', window.location.pathname, { target: '#content', swap: 'innerHTML' });
                    }
                });
            }
        };

        toast.appendChild(textSpan);
        toast.appendChild(undoBtn);
        container.appendChild(toast);

        setTimeout(function() {
            if (!cancelled) {
                toast.classList.add('fade-out');
                setTimeout(function() { toast.remove(); }, 300);
            }
        }, timeoutMs);
    }

    // Listen for HTMX showToast event
    document.addEventListener('showToast', function(e) {
        var detail = e.detail || {};
        showToast(detail.message || '操作完成', detail.type || 'success');
    });

    // Catch error responses
    document.body.addEventListener('htmx:afterRequest', function(e) {
        if (e.detail.xhr && e.detail.xhr.status >= 400) {
            try {
                var data = JSON.parse(e.detail.xhr.responseText);
                if (data.error) showToast(data.error, 'error');
            } catch(_) {}
        }
    });

    window.showToast = showToast;
    window.showUndoToast = showUndoToast;
})();
