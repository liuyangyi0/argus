/* Keyboard shortcuts for Argus dashboard.
 *
 * Global:    Escape=close modal, ?=show help, 1-8=nav tabs
 * Alerts:    j/k=navigate rows, a=acknowledge, f=mark false positive, Enter=detail
 */
(function() {
    'use strict';

    var _helpVisible = false;

    // Navigation targets mapped to number keys
    var _navTargets = [
        '/',            // 1 = 总览
        '/cameras',     // 2 = 摄像头
        '/baseline',    // 3 = 基线与模型
        '/zones',       // 4 = 检测区域
        '/alerts',      // 5 = 告警中心
        '/config',      // 6 = 系统设置
        '/backup',      // 7 = 数据备份
        '/audit',       // 8 = 审计日志
    ];

    function isInputFocused() {
        var el = document.activeElement;
        if (!el) return false;
        var tag = el.tagName.toLowerCase();
        return tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable;
    }

    function closeModal() {
        var modal = document.getElementById('alert-modal');
        if (modal) modal.classList.remove('active');
    }

    function toggleHelp() {
        var overlay = document.getElementById('keyboard-help');
        if (!overlay) return;
        _helpVisible = !_helpVisible;
        if (_helpVisible) {
            overlay.classList.add('active');
        } else {
            overlay.classList.remove('active');
        }
    }

    // Alert row navigation
    function getAlertRows() {
        return document.querySelectorAll('#alerts-form tbody tr');
    }

    function getSelectedRow() {
        return document.querySelector('#alerts-form tbody tr.kb-selected');
    }

    function selectRow(row) {
        var prev = getSelectedRow();
        if (prev) prev.classList.remove('kb-selected');
        if (row) {
            row.classList.add('kb-selected');
            row.scrollIntoView({ block: 'nearest' });
        }
    }

    function navigateRow(direction) {
        var rows = getAlertRows();
        if (rows.length === 0) return;
        var current = getSelectedRow();
        var idx = current ? Array.prototype.indexOf.call(rows, current) : -1;
        var next = idx + direction;
        if (next < 0) next = 0;
        if (next >= rows.length) next = rows.length - 1;
        selectRow(rows[next]);
    }

    function actionOnSelected(selector) {
        var row = getSelectedRow();
        if (!row) return;
        var btn = row.querySelector(selector);
        if (btn) btn.click();
    }

    document.addEventListener('keydown', function(e) {
        // Never intercept when typing in inputs
        if (isInputFocused()) return;

        var key = e.key;

        // Escape — close modal or help overlay
        if (key === 'Escape') {
            if (_helpVisible) { toggleHelp(); return; }
            closeModal();
            return;
        }

        // ? — toggle keyboard help
        if (key === '?' || (e.shiftKey && key === '/')) {
            e.preventDefault();
            toggleHelp();
            return;
        }

        // 1-8 — nav tabs
        if (key >= '1' && key <= '8' && !e.ctrlKey && !e.altKey && !e.metaKey) {
            var idx = parseInt(key) - 1;
            if (idx < _navTargets.length) {
                window.location.href = _navTargets[idx];
            }
            return;
        }

        // Alert page shortcuts
        if (key === 'j') { navigateRow(1); return; }
        if (key === 'k') { navigateRow(-1); return; }
        if (key === 'a') { actionOnSelected('.btn-primary'); return; }
        if (key === 'f') { actionOnSelected('.btn-ghost'); return; }
        if (key === 'Enter') {
            var row = getSelectedRow();
            if (row) {
                var thumb = row.querySelector('.alert-thumb');
                if (thumb) thumb.click();
            }
            return;
        }
    });

    // Inject help overlay and selected row styles
    document.addEventListener('DOMContentLoaded', function() {
        // Keyboard help overlay
        var help = document.createElement('div');
        help.id = 'keyboard-help';
        help.className = 'modal-overlay';
        help.onclick = function(e) { if (e.target === help) toggleHelp(); };
        help.innerHTML = '<div class="modal-content" style="max-width:480px;">' +
            '<button class="modal-close" onclick="document.getElementById(\'keyboard-help\').classList.remove(\'active\')">&times;</button>' +
            '<h3 style="color:#4fc3f7;margin-bottom:16px;">键盘快捷键</h3>' +
            '<table style="font-size:13px;">' +
            '<tr><td style="color:#8890a0;padding:4px 16px 4px 0;">?</td><td>显示/隐藏快捷键帮助</td></tr>' +
            '<tr><td style="color:#8890a0;padding:4px 16px 4px 0;">Esc</td><td>关闭弹窗</td></tr>' +
            '<tr><td style="color:#8890a0;padding:4px 16px 4px 0;">1-8</td><td>切换导航标签页</td></tr>' +
            '<tr><td colspan="2" style="padding:8px 0;border-bottom:1px solid #2a2d37;"></td></tr>' +
            '<tr><td colspan="2" style="color:#8890a0;padding:8px 0;">告警页面：</td></tr>' +
            '<tr><td style="color:#8890a0;padding:4px 16px 4px 0;">j / k</td><td>上下移动选中行</td></tr>' +
            '<tr><td style="color:#8890a0;padding:4px 16px 4px 0;">a</td><td>确认选中告警</td></tr>' +
            '<tr><td style="color:#8890a0;padding:4px 16px 4px 0;">f</td><td>标记误报</td></tr>' +
            '<tr><td style="color:#8890a0;padding:4px 16px 4px 0;">Enter</td><td>查看告警详情</td></tr>' +
            '</table></div>';
        document.body.appendChild(help);

        // CSS for selected row
        var style = document.createElement('style');
        style.textContent = '#alerts-form tbody tr.kb-selected { background: #1565c0 !important; outline: 1px solid #4fc3f7; }';
        document.head.appendChild(style);
    });
})();
