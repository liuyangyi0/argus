/**
 * Zone Editor — Canvas-based polygon drawing tool for defining
 * include/exclude zones on camera snapshots.
 *
 * Green polygon = include zone (only detect here)
 * Red polygon = exclude zone (never detect here)
 */

let currentCameraId = null;
let points = [];
let canvas, ctx;

function openZoneEditor(cameraId) {
    currentCameraId = cameraId;
    points = [];

    const modal = document.getElementById('zone-editor-modal');
    modal.style.display = 'block';
    document.getElementById('editor-title').textContent = 'Draw Zone — ' + cameraId;
    document.getElementById('zone-name').value = '';
    document.getElementById('editor-status').textContent = 'Loading camera snapshot...';

    canvas = document.getElementById('zone-canvas');
    ctx = canvas.getContext('2d');

    // Load camera snapshot
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function () {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        canvas._bgImage = img;
        document.getElementById('editor-status').textContent =
            'Click on the image to draw polygon vertices. Click Save when done.';
    };
    img.onerror = function () {
        ctx.fillStyle = '#12141c';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#616161';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Camera snapshot unavailable — draw on blank canvas', canvas.width / 2, canvas.height / 2);
        document.getElementById('editor-status').textContent =
            'Snapshot unavailable. You can still draw zones on the blank canvas.';
    };
    img.src = '/api/zones/snapshot/' + cameraId + '?t=' + Date.now();

    canvas.onclick = function (e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);
        points.push([x, y]);
        redraw();
        document.getElementById('editor-status').textContent =
            points.length + ' points. Click Save when done, or keep clicking to add more.';
    };
}

function closeZoneEditor() {
    document.getElementById('zone-editor-modal').style.display = 'none';
    points = [];
    currentCameraId = null;
}

function clearPoints() {
    points = [];
    redraw();
    document.getElementById('editor-status').textContent =
        'Points cleared. Click to start drawing again.';
}

function redraw() {
    if (!ctx) return;

    // Redraw background
    if (canvas._bgImage) {
        ctx.drawImage(canvas._bgImage, 0, 0);
    } else {
        ctx.fillStyle = '#12141c';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    if (points.length === 0) return;

    const zoneType = document.getElementById('zone-type').value;
    const color = zoneType === 'exclude' ? 'rgba(244, 67, 54, 0.3)' : 'rgba(76, 175, 80, 0.3)';
    const strokeColor = zoneType === 'exclude' ? '#f44336' : '#4caf50';

    // Draw filled polygon
    if (points.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1]);
        }
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
    }

    // Draw outline
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
    }
    if (points.length >= 3) ctx.closePath();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw vertices
    for (const [x, y] of points) {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = strokeColor;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// Redraw when zone type changes
document.getElementById('zone-type')?.addEventListener('change', redraw);

async function saveZone() {
    if (points.length < 3) {
        document.getElementById('editor-status').textContent =
            'Need at least 3 points to form a polygon.';
        return;
    }

    const name = document.getElementById('zone-name').value || 'Zone';
    const zoneType = document.getElementById('zone-type').value;
    const zoneId = zoneType + '_' + Date.now().toString(36);

    const body = {
        camera_id: currentCameraId,
        zone_id: zoneId,
        name: name,
        polygon: points,
        zone_type: zoneType,
        priority: 'standard',
        anomaly_threshold: 0.7,
    };

    try {
        const resp = await fetch('/api/zones', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (resp.ok) {
            document.getElementById('editor-status').textContent = 'Zone saved! Closing...';
            setTimeout(function () {
                closeZoneEditor();
                // Refresh the zones page
                htmx.ajax('GET', '/api/zones', { target: '#content', swap: 'innerHTML' });
            }, 500);
        } else {
            const err = await resp.json();
            document.getElementById('editor-status').textContent = 'Error: ' + (err.error || 'Unknown');
        }
    } catch (e) {
        document.getElementById('editor-status').textContent = 'Network error: ' + e.message;
    }
}
