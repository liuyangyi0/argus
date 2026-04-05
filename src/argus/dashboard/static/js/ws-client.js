/**
 * Argus WebSocket Client
 *
 * Provides real-time push updates for the dashboard, replacing HTMX polling.
 * Features: auto-reconnect with exponential backoff, heartbeat, fallback to polling.
 */
(function () {
  'use strict';

  var ws = null;
  var reconnectDelay = 1000;
  var maxReconnectDelay = 30000;
  var heartbeatTimer = null;
  var reconnectTimer = null;
  var connected = false;

  // Topic handlers: topic -> array of callback functions
  var handlers = {};

  function getWsUrl() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = proto + '//' + location.host + '/ws';
    // Extract token from cookie or meta tag if auth is enabled
    var tokenMeta = document.querySelector('meta[name="ws-token"]');
    if (tokenMeta && tokenMeta.content) {
      url += '?token=' + encodeURIComponent(tokenMeta.content);
    }
    return url;
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
      return;
    }

    try {
      ws = new WebSocket(getWsUrl());
    } catch (e) {
      scheduleReconnect();
      return;
    }

    ws.onopen = function () {
      connected = true;
      reconnectDelay = 1000;
      updateIndicator('connected');
      startHeartbeat();
      // Remove polling attributes from HTMX elements since WS is active
      disablePolling();
    };

    ws.onmessage = function (event) {
      try {
        var msg = JSON.parse(event.data);
        if (msg.topic === 'ping') {
          // Respond with pong
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'pong' }));
          }
          return;
        }
        dispatch(msg.topic, msg.data);
      } catch (e) {
        // ignore parse errors
      }
    };

    ws.onclose = function (event) {
      connected = false;
      stopHeartbeat();
      updateIndicator('disconnected');

      // Auth failure — don't reconnect
      if (event.code === 4401) {
        updateIndicator('auth_failed');
        return;
      }

      // Re-enable polling as fallback
      enablePolling();
      scheduleReconnect();
    };

    ws.onerror = function () {
      // onclose will fire after this
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connect();
    }, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, maxReconnectDelay);
  }

  function startHeartbeat() {
    stopHeartbeat();
    heartbeatTimer = setInterval(function () {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'pong' }));
      }
    }, 25000);
  }

  function stopHeartbeat() {
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
  }

  /**
   * Dispatch a received message to registered handlers.
   */
  function dispatch(topic, data) {
    var topicHandlers = handlers[topic];
    if (topicHandlers) {
      for (var i = 0; i < topicHandlers.length; i++) {
        try {
          topicHandlers[i](data);
        } catch (e) {
          // handler error, ignore
        }
      }
    }

    // Generic DOM update: find elements with data-ws-topic matching this topic
    var elements = document.querySelectorAll('[data-ws-topic="' + topic + '"]');
    for (var j = 0; j < elements.length; j++) {
      var el = elements[j];
      var url = el.getAttribute('data-ws-refresh-url');
      if (url) {
        // Trigger an HTMX request to refresh this element's content
        htmx.ajax('GET', url, { target: el, swap: 'outerHTML' });
      }
    }
  }

  // Store original polling attributes so we can restore them
  var pollingElements = [];

  /**
   * Disable HTMX polling on elements that are now served by WebSocket.
   */
  function disablePolling() {
    var elements = document.querySelectorAll('[data-ws-topic]');
    for (var i = 0; i < elements.length; i++) {
      var el = elements[i];
      var trigger = el.getAttribute('hx-trigger');
      if (trigger && trigger.indexOf('every') !== -1) {
        pollingElements.push({ el: el, trigger: trigger });
        el.removeAttribute('hx-trigger');
        htmx.process(el);
      }
    }
  }

  /**
   * Re-enable polling as fallback when WebSocket is disconnected.
   */
  function enablePolling() {
    for (var i = 0; i < pollingElements.length; i++) {
      var item = pollingElements[i];
      if (document.contains(item.el)) {
        item.el.setAttribute('hx-trigger', item.trigger);
        htmx.process(item.el);
      }
    }
    pollingElements = [];
  }

  /**
   * Update the connection status indicator in the nav bar.
   */
  function updateIndicator(status) {
    var indicator = document.getElementById('ws-status');
    if (!indicator) {
      // Create indicator in nav-right
      var navRight = document.querySelector('.nav-right');
      if (!navRight) return;
      indicator = document.createElement('span');
      indicator.id = 'ws-status';
      indicator.style.cssText = 'font-size:11px;padding:2px 8px;border-radius:10px;margin-right:8px;';
      navRight.insertBefore(indicator, navRight.firstChild);
    }

    if (status === 'connected') {
      indicator.textContent = 'WS';
      indicator.style.background = '#2e7d32';
      indicator.style.color = '#fff';
      indicator.title = 'WebSocket \u5df2\u8fde\u63a5';
    } else if (status === 'disconnected') {
      indicator.textContent = 'WS';
      indicator.style.background = '#ff9800';
      indicator.style.color = '#fff';
      indicator.title = 'WebSocket \u91cd\u8fde\u4e2d...';
    } else if (status === 'auth_failed') {
      indicator.textContent = 'WS';
      indicator.style.background = '#f44336';
      indicator.style.color = '#fff';
      indicator.title = 'WebSocket \u8ba4\u8bc1\u5931\u8d25';
    }
  }

  // Public API
  window.ArgusWS = {
    /**
     * Register a handler for a WebSocket topic.
     * @param {string} topic - One of: health, cameras, alerts, tasks
     * @param {function} callback - Called with (data) when event received
     */
    on: function (topic, callback) {
      if (!handlers[topic]) {
        handlers[topic] = [];
      }
      handlers[topic].push(callback);
    },

    /**
     * Remove a handler for a topic.
     */
    off: function (topic, callback) {
      if (handlers[topic]) {
        handlers[topic] = handlers[topic].filter(function (h) { return h !== callback; });
      }
    },

    /**
     * Check if WebSocket is currently connected.
     */
    isConnected: function () {
      return connected;
    }
  };

  // Register default handlers for toast notifications on new alerts
  window.ArgusWS.on('alerts', function (data) {
    if (window.showToast && data.severity) {
      var severityLabels = { info: '\u63d0\u793a', low: '\u4f4e', medium: '\u4e2d', high: '\u9ad8' };
      var label = severityLabels[data.severity] || data.severity;
      var type = data.severity === 'high' ? 'error' : (data.severity === 'medium' ? 'warning' : 'info');
      window.showToast('\u65b0\u544a\u8b66 [' + label + '] ' + data.camera_id + ' - ' + data.zone_id, type);
    }
  });

  // Auto-connect when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', connect);
  } else {
    connect();
  }
})();
