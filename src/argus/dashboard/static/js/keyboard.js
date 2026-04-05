/* Argus keyboard shortcuts + Cmd+K command palette.
 *
 * Global:    Cmd+K=command palette, Esc=close, ?=help, 1-5=nav
 * Alerts:    j/k=navigate, Enter=detail, y=confirm, n=false positive, e=escalate
 * Detail:    y/n/e same as above
 */
(function() {
    'use strict';

    // ── Navigation targets (5 pages) ──
    var _navTargets = [
        '/',           // 1 = 总览
        '/cameras',    // 2 = 摄像头
        '/alerts',     // 3 = 告警
        '/models',     // 4 = 模型
        '/system',     // 5 = 系统
    ];

    var _navLabels = ['总览', '摄像头', '告警', '模型', '系统'];

    // ── Helpers ──
    function isInputFocused() {
        var el = document.activeElement;
        if (!el) return false;
        var tag = el.tagName.toLowerCase();
        return tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable;
    }

    function closeAllOverlays() {
        var modal = document.getElementById('alert-modal');
        if (modal) modal.classList.remove('active');
        var help = document.getElementById('keyboard-help');
        if (help) help.classList.remove('active');
        closePalette();
    }

    // ── Command Palette (Cmd+K) ──
    var _paletteOpen = false;

    function buildCommands() {
        var cmds = [];
        // Nav pages
        for (var i = 0; i < _navTargets.length; i++) {
            cmds.push({ label: '跳转到 ' + _navLabels[i], action: 'nav', url: _navTargets[i], icon: '→' });
        }
        // Camera-specific (dynamic)
        var camCards = document.querySelectorAll('[hx-push-url*="cam="]');
        camCards.forEach(function(card) {
            var url = card.getAttribute('hx-get');
            var pushUrl = card.getAttribute('hx-push-url');
            var camId = pushUrl ? pushUrl.split('cam=')[1] : '';
            if (camId && url) {
                cmds.push({ label: '跳转到 ' + camId, action: 'htmx', url: url, target: '#content', icon: '📹' });
                cmds.push({ label: camId + ' 最近告警', action: 'nav', url: '/alerts?camera_id=' + camId, icon: '🚨' });
            }
        });
        // Common actions
        cmds.push({ label: '查看告警', action: 'nav', url: '/alerts', icon: '🚨' });
        cmds.push({ label: '键盘快捷键帮助', action: 'fn', fn: toggleHelp, icon: '⌨' });
        return cmds;
    }

    function openPalette() {
        var overlay = document.getElementById('cmd-palette');
        if (!overlay) return;
        overlay.classList.add('active');
        _paletteOpen = true;
        var input = overlay.querySelector('.cmd-input');
        if (input) { input.value = ''; input.focus(); }
        filterPalette('');
    }

    function closePalette() {
        var overlay = document.getElementById('cmd-palette');
        if (overlay) overlay.classList.remove('active');
        _paletteOpen = false;
    }

    function filterPalette(query) {
        var list = document.getElementById('cmd-list');
        if (!list) return;
        var cmds = buildCommands();
        var q = query.toLowerCase();
        var filtered = q ? cmds.filter(function(c) {
            return c.label.toLowerCase().indexOf(q) !== -1;
        }) : cmds;

        list.innerHTML = '';
        filtered.slice(0, 10).forEach(function(cmd, idx) {
            var item = document.createElement('div');
            item.className = 'cmd-item' + (idx === 0 ? ' cmd-active' : '');
            item.innerHTML = '<span class="cmd-icon">' + (cmd.icon || '') + '</span>' +
                             '<span>' + cmd.label + '</span>';
            item.onclick = function() { executeCommand(cmd); };
            list.appendChild(item);
        });
    }

    function executeCommand(cmd) {
        closePalette();
        if (cmd.action === 'nav') {
            window.location.href = cmd.url;
        } else if (cmd.action === 'htmx' && window.htmx) {
            htmx.ajax('GET', cmd.url, { target: cmd.target || '#content', swap: 'innerHTML' });
        } else if (cmd.action === 'fn' && cmd.fn) {
            cmd.fn();
        }
    }

    function navigatePaletteItems(direction) {
        var items = document.querySelectorAll('.cmd-item');
        if (items.length === 0) return;
        var active = document.querySelector('.cmd-item.cmd-active');
        var idx = active ? Array.prototype.indexOf.call(items, active) : -1;
        var next = idx + direction;
        if (next < 0) next = items.length - 1;
        if (next >= items.length) next = 0;
        items.forEach(function(it) { it.classList.remove('cmd-active'); });
        items[next].classList.add('cmd-active');
        items[next].scrollIntoView({ block: 'nearest' });
    }

    function executeActivePaletteItem() {
        var active = document.querySelector('.cmd-item.cmd-active');
        if (active) active.click();
    }

    // ── Help overlay ──
    var _helpVisible = false;
    function toggleHelp() {
        var overlay = document.getElementById('keyboard-help');
        if (!overlay) return;
        _helpVisible = !_helpVisible;
        overlay.classList[_helpVisible ? 'add' : 'remove']('active');
    }

    // ── Alert row navigation ──
    function getAlertRows() { return document.querySelectorAll('#alerts-form tbody tr'); }
    function getSelectedRow() { return document.querySelector('#alerts-form tbody tr.kb-selected'); }

    function selectRow(row) {
        var prev = getSelectedRow();
        if (prev) prev.classList.remove('kb-selected');
        if (row) {
            row.classList.add('kb-selected');
            row.scrollIntoView({ block: 'nearest' });
        }
    }

    function navigateRow(dir) {
        var rows = getAlertRows();
        if (rows.length === 0) return;
        var cur = getSelectedRow();
        var idx = cur ? Array.prototype.indexOf.call(rows, cur) : -1;
        var next = Math.max(0, Math.min(rows.length - 1, idx + dir));
        selectRow(rows[next]);
    }

    function actionOnSelected(selector) {
        var row = getSelectedRow();
        if (!row) return;
        var btn = row.querySelector(selector);
        if (btn) btn.click();
    }

    // ── Main keydown handler ──
    document.addEventListener('keydown', function(e) {
        // Cmd+K / Ctrl+K — command palette (always active)
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            if (_paletteOpen) closePalette(); else openPalette();
            return;
        }

        // Palette navigation when open
        if (_paletteOpen) {
            if (e.key === 'Escape') { closePalette(); return; }
            if (e.key === 'ArrowDown') { e.preventDefault(); navigatePaletteItems(1); return; }
            if (e.key === 'ArrowUp') { e.preventDefault(); navigatePaletteItems(-1); return; }
            if (e.key === 'Enter') { e.preventDefault(); executeActivePaletteItem(); return; }
            return; // Let typing happen in input
        }

        // Don't intercept when typing in inputs
        if (isInputFocused()) return;

        var key = e.key;

        // Escape — close overlays
        if (key === 'Escape') { closeAllOverlays(); return; }

        // ? — help
        if (key === '?' || (e.shiftKey && key === '/')) {
            e.preventDefault(); toggleHelp(); return;
        }

        // 1-5 — nav pages
        if (key >= '1' && key <= '5' && !e.ctrlKey && !e.altKey && !e.metaKey) {
            window.location.href = _navTargets[parseInt(key) - 1];
            return;
        }

        // Alert shortcuts: j/k nav, y=confirm, n=false positive, e=escalate
        if (key === 'j') { navigateRow(1); return; }
        if (key === 'k') { navigateRow(-1); return; }
        if (key === 'y') { actionOnSelected('.btn-primary'); return; }
        if (key === 'n') { actionOnSelected('.btn-ghost'); return; }
        if (key === 'Enter') {
            var row = getSelectedRow();
            if (row) { var t = row.querySelector('.alert-thumb'); if (t) t.click(); }
            return;
        }
    });

    // ── DOM init ──
    document.addEventListener('DOMContentLoaded', function() {
        // Command palette overlay
        var palette = document.createElement('div');
        palette.id = 'cmd-palette';
        palette.className = 'modal-overlay';
        palette.onclick = function(e) { if (e.target === palette) closePalette(); };
        palette.innerHTML =
            '<div style="background:var(--bg-raised);border-radius:var(--radius-xl);' +
            'max-width:560px;width:90%;border:1px solid var(--border-strong);box-shadow:var(--shadow-xl);' +
            'margin-top:15vh;align-self:flex-start;">' +
            '<input class="cmd-input" type="text" placeholder="输入命令或搜索..." ' +
            'style="width:100%;padding:var(--space-4) var(--space-5);background:transparent;' +
            'border:none;border-bottom:1px solid var(--border-subtle);color:var(--text-primary);' +
            'font-size:var(--text-lg);outline:none;font-family:var(--font-sans);">' +
            '<div id="cmd-list" style="max-height:320px;overflow-y:auto;padding:var(--space-2) 0;"></div>' +
            '<div style="padding:var(--space-2) var(--space-4);border-top:1px solid var(--border-subtle);' +
            'font-size:var(--text-xs);color:var(--text-tertiary);">' +
            '↑↓ 导航 · Enter 执行 · Esc 关闭</div></div>';
        document.body.appendChild(palette);

        var cmdInput = palette.querySelector('.cmd-input');
        if (cmdInput) {
            cmdInput.addEventListener('input', function() { filterPalette(this.value); });
        }

        // Keyboard help overlay
        var help = document.createElement('div');
        help.id = 'keyboard-help';
        help.className = 'modal-overlay';
        help.onclick = function(e) { if (e.target === help) toggleHelp(); };
        help.innerHTML = '<div class="modal-content" style="max-width:480px;">' +
            '<button class="modal-close" onclick="document.getElementById(\'keyboard-help\').classList.remove(\'active\')">&times;</button>' +
            '<h3 style="color:var(--status-info-text);margin-bottom:var(--space-4);">键盘快捷键</h3>' +
            '<table>' +
            '<tr><td style="color:var(--text-secondary);padding:4px var(--space-4) 4px 0;">Cmd+K</td><td>命令面板</td></tr>' +
            '<tr><td style="color:var(--text-secondary);padding:4px var(--space-4) 4px 0;">?</td><td>显示/隐藏此帮助</td></tr>' +
            '<tr><td style="color:var(--text-secondary);padding:4px var(--space-4) 4px 0;">Esc</td><td>关闭弹窗</td></tr>' +
            '<tr><td style="color:var(--text-secondary);padding:4px var(--space-4) 4px 0;">1-5</td><td>切换导航页面</td></tr>' +
            '<tr><td colspan="2" style="padding:var(--space-2) 0;border-bottom:1px solid var(--border-subtle);"></td></tr>' +
            '<tr><td colspan="2" style="color:var(--text-secondary);padding:var(--space-2) 0;">告警页面：</td></tr>' +
            '<tr><td style="color:var(--text-secondary);padding:4px var(--space-4) 4px 0;">j / k</td><td>上下移动选中行</td></tr>' +
            '<tr><td style="color:var(--text-secondary);padding:4px var(--space-4) 4px 0;">y</td><td>确认真实</td></tr>' +
            '<tr><td style="color:var(--text-secondary);padding:4px var(--space-4) 4px 0;">n</td><td>标记误报</td></tr>' +
            '<tr><td style="color:var(--text-secondary);padding:4px var(--space-4) 4px 0;">Enter</td><td>查看详情</td></tr>' +
            '</table></div>';
        document.body.appendChild(help);

        // CSS for selected row + command palette items
        var style = document.createElement('style');
        style.textContent =
            '#alerts-form tbody tr.kb-selected { background: var(--status-info-bg) !important; outline: 1px solid var(--status-info); }' +
            '.cmd-item { display:flex;align-items:center;gap:var(--space-3);padding:var(--space-2) var(--space-5);' +
            'cursor:pointer;font-size:var(--text-sm);color:var(--text-primary);transition:background 0.1s; }' +
            '.cmd-item:hover, .cmd-item.cmd-active { background:var(--status-info-bg); }' +
            '.cmd-icon { width:20px;text-align:center;flex-shrink:0;font-size:var(--text-base); }';
        document.head.appendChild(style);
    });
})();
