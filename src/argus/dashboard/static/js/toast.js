/* Toast notification system for Argus dashboard.
 * Listens for HTMX showToast events triggered via HX-Trigger response header.
 *
 * Server usage:
 *   headers = {"HX-Trigger": JSON.stringify({showToast: {message: "...", type: "success"}})}
 */
(function() {
    // Create toast container
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    function showToast(message, type) {
        type = type || 'info';
        const toast = document.createElement('div');
        toast.className = 'toast ' + type;
        toast.textContent = message;
        container.appendChild(toast);

        // Auto-dismiss after 3 seconds
        setTimeout(function() {
            toast.classList.add('fade-out');
            setTimeout(function() { toast.remove(); }, 300);
        }, 3000);
    }

    // Listen for HTMX showToast event
    document.addEventListener('showToast', function(e) {
        const detail = e.detail || {};
        showToast(detail.message || '操作完成', detail.type || 'success');
    });

    // Also listen on body for htmx:afterRequest to catch HX-Trigger headers
    document.body.addEventListener('htmx:afterRequest', function(e) {
        // Check for error responses
        if (e.detail.xhr && e.detail.xhr.status >= 400) {
            try {
                const data = JSON.parse(e.detail.xhr.responseText);
                if (data.error) {
                    showToast(data.error, 'error');
                }
            } catch(_) {}
        }
    });

    // Expose globally for manual use
    window.showToast = showToast;
})();
