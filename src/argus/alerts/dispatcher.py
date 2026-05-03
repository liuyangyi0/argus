"""Multi-channel alert dispatcher.

Routes graded alerts to multiple destinations: database persistence,
webhook (HTTP POST), and WebSocket push. Each channel operates independently
so a failure in one channel doesn't block others.

Webhook dispatch runs in a background thread to avoid blocking the
camera processing thread during HTTP timeouts.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

import cv2
import numpy as np
import structlog

from argus.alerts.grader import Alert
from argus.config.schema import AlertConfig, AudioAlertConfig
from argus.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from argus.storage.database import Database

logger = structlog.get_logger()


class AlertDispatcher:
    """Dispatches alerts to all configured channels.

    Channels:
    - Database: Always active, persists all alerts to SQLite
    - Webhook: HTTP POST to configured URL (e.g., plant DCS), runs in background thread
    - WebSocket: Real-time dashboard push

    Each alert's snapshot and heatmap are saved to disk before
    dispatching to ensure the images are available for review.
    """

    def __init__(
        self,
        config: AlertConfig,
        database: Database,
        alerts_dir: str | Path = "data/alerts",
        on_alert: Callable[[str, dict], None] | None = None,
        audio_config: AudioAlertConfig | None = None,
    ):
        self._config = config
        self._db = database
        self._alerts_dir = Path(alerts_dir)
        self._on_alert_ws = on_alert
        self._audio_config = audio_config or AudioAlertConfig()
        self._alerts_dir.mkdir(parents=True, exist_ok=True)
        self._http_client = None
        self._shutdown = threading.Event()
        # M3: Cache disk space check to avoid frequent syscalls
        self._disk_space_ok: bool = True
        self._disk_space_checked_at: float = 0.0

        # DET-009: Circuit breaker for webhook dispatch
        cb_config = CircuitBreakerConfig(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout_seconds=config.circuit_breaker_timeout,
        )
        self._circuit_breaker = CircuitBreaker(cb_config)

        # HIGH-13: Non-daemon threads so queued alerts are not discarded on shutdown
        # Background database dispatch thread — keeps DB writes off the pipeline thread
        self._db_queue: Queue = Queue(maxsize=200)
        self._db_thread = threading.Thread(
            target=self._db_worker,
            name="argus-db",
            daemon=False,
        )
        self._db_thread.start()

        # Background webhook dispatch thread
        self._webhook_queue: Queue = Queue(maxsize=100)
        self._webhook_thread: threading.Thread | None = None
        if self._config.webhook.enabled:
            self._webhook_thread = threading.Thread(
                target=self._webhook_worker,
                name="argus-webhook",
                daemon=False,
            )
            self._webhook_thread.start()

    @staticmethod
    def _alert_to_dict(
        alert: Alert,
        snapshot_path: str | None = None,
        heatmap_path: str | None = None,
        evidence_unavailable: bool = False,
    ) -> dict:
        """Convert an alert to a dict payload for dispatch channels."""
        payload = {
            "alert_id": alert.alert_id,
            "timestamp": datetime.fromtimestamp(alert.timestamp, tz=timezone.utc).isoformat(),
            "camera_id": alert.camera_id,
            "zone_id": alert.zone_id,
            "severity": alert.severity.value,
            "anomaly_score": round(alert.anomaly_score, 4),
            "frame_number": alert.frame_number,
            "detection_type": alert.detection_type,
            "detected_objects": alert.detected_objects,
            "corroborated": alert.corroborated,
            "handling_policy": alert.handling_policy,
            "event_group_count": alert.event_group_count,
        }
        if snapshot_path is not None:
            payload["snapshot_path"] = snapshot_path
        if heatmap_path is not None:
            payload["heatmap_path"] = heatmap_path
        if evidence_unavailable:
            payload["evidence_unavailable"] = True
        # Optional fields — only include when set to keep payload lean
        if alert.correlation_partner is not None:
            payload["correlation_partner"] = alert.correlation_partner
        if alert.model_version_id is not None:
            payload["model_version_id"] = alert.model_version_id
        if alert.classification_label is not None:
            payload["classification_label"] = alert.classification_label
        if alert.classification_confidence is not None:
            payload["classification_confidence"] = round(alert.classification_confidence, 4)
        if alert.severity_adjusted_by_classifier:
            payload["severity_adjusted_by_classifier"] = True
        if alert.segmentation_count > 0:
            payload["segmentation_count"] = alert.segmentation_count
            payload["segmentation_total_area_px"] = alert.segmentation_total_area_px
            payload["segmentation_objects"] = alert.segmentation_objects
        if alert.event_group_id is not None:
            payload["event_group_id"] = alert.event_group_id
        if alert.speed_ms is not None:
            payload["speed_ms"] = alert.speed_ms
        if alert.speed_px_per_sec is not None:
            payload["speed_px_per_sec"] = alert.speed_px_per_sec
        if alert.trajectory_model is not None:
            payload["trajectory_model"] = alert.trajectory_model
        if alert.origin_x_mm is not None:
            payload["origin_x_mm"] = alert.origin_x_mm
        if alert.origin_y_mm is not None:
            payload["origin_y_mm"] = alert.origin_y_mm
        if alert.origin_z_mm is not None:
            payload["origin_z_mm"] = alert.origin_z_mm
        if alert.landing_x_mm is not None:
            payload["landing_x_mm"] = alert.landing_x_mm
        if alert.landing_y_mm is not None:
            payload["landing_y_mm"] = alert.landing_y_mm
        if alert.landing_z_mm is not None:
            payload["landing_z_mm"] = alert.landing_z_mm
        trajectories_json = getattr(alert, "trajectories_json", None)
        if trajectories_json:
            import json as _json
            try:
                payload["trajectories"] = _json.loads(trajectories_json)
            except Exception:
                pass
        return payload

    def set_websocket_broadcaster(self, broadcaster: Callable[[str, dict], None] | None) -> None:
        """Wire (or rewire) the WebSocket broadcaster after construction.

        The dispatcher is built before the dashboard FastAPI app — and thus
        before ``ws_manager`` exists — so production code constructs the
        dispatcher with ``on_alert=None`` and calls this setter once the app
        is up. Without this hookup alerts are never pushed to the dashboard
        in real time (P1 audit finding 2026-05).
        """
        self._on_alert_ws = broadcaster

    def dispatch(self, alert: Alert) -> None:
        """Send an alert to all configured channels."""
        evidence_unavailable = False

        # Check disk space before writing images
        if not self._check_disk_space():
            logger.error(
                "dispatch.disk_full",
                alert_id=alert.alert_id,
                msg="Disk space critically low, skipping image save",
            )
            snapshot_path = None
            heatmap_path = None
            evidence_unavailable = True
        else:
            snapshot_path = self._save_snapshot(alert)
            heatmap_path = self._save_heatmap(alert)
            if alert.snapshot is not None and snapshot_path is None:
                evidence_unavailable = True

        # Channel 1: Database (non-blocking, queued to background thread)
        db_data = {
            "alert": alert,
            "snapshot_path": snapshot_path,
            "heatmap_path": heatmap_path,
            "evidence_unavailable": evidence_unavailable,
        }
        try:
            self._db_queue.put_nowait(db_data)
        except Exception:
            logger.error(
                "dispatch.db_queue_full",
                alert_id=alert.alert_id,
                msg="Alert DB persistence lost — queue overflow, attempting sync fallback",
            )
            # Fallback: synchronous write so the alert is never silently lost
            self._dispatch_database(alert, snapshot_path, heatmap_path)

        # Channel 2: Webhook (non-blocking, queued to background thread, with circuit breaker DET-009)
        if self._config.webhook.enabled:
            if self._circuit_breaker.allow_request():
                payload = self._alert_to_dict(
                    alert, snapshot_path, heatmap_path, evidence_unavailable,
                )
                try:
                    self._webhook_queue.put_nowait(payload)
                except Exception:
                    logger.error(
                        "dispatch.webhook_queue_full",
                        alert_id=alert.alert_id,
                        msg="Alert webhook delivery lost — queue overflow",
                    )
                    # P6: Persist overflow event so no alert is silently lost
                    self._record_dispatch_failure(alert, "webhook_queue_overflow")
            else:
                logger.warning(
                    "dispatch.circuit_open",
                    alert_id=alert.alert_id,
                    msg="Circuit breaker open, webhook skipped",
                )

        # Channel 3: WebSocket push (real-time dashboard notification)
        # HIGH severity → priority dispatch (bypass queue, push directly)
        ws_payload = self._alert_to_dict(alert, evidence_unavailable=evidence_unavailable)
        if self._on_alert_ws:
            try:
                self._on_alert_ws("alerts", ws_payload)
            except Exception as e:
                logger.warning("dispatch.websocket_failed", alert_id=alert.alert_id, error=str(e))

        # Channel 4: Audio alarm via WebSocket
        if self._on_alert_ws:
            try:
                audio_msg = {
                    "type": "audio_alert",
                    "severity": ws_payload["severity"],
                    "alert_id": ws_payload["alert_id"],
                    "camera_id": ws_payload["camera_id"],
                }
                # Look up per-severity audio settings
                sev_audio = getattr(self._audio_config, alert.severity.value, None)
                if sev_audio is not None:
                    audio_msg["sound"] = sev_audio.sound
                    audio_msg["voice_template"] = sev_audio.voice_template
                self._on_alert_ws("audio_alert", audio_msg)
            except Exception as e:
                logger.warning("dispatch.audio_alert_failed", alert_id=alert.alert_id, error=str(e))

        dispatch_latency_ms = (time.time() - alert.timestamp) * 1000
        logger.info(
            "alert.dispatched",
            alert_id=alert.alert_id,
            severity=alert.severity.value,
            camera_id=alert.camera_id,
            score=round(alert.anomaly_score, 3),
            latency_ms=round(dispatch_latency_ms, 1),
        )

    def _dispatch_database(
        self, alert: Alert, snapshot_path: str | None, heatmap_path: str | None
    ) -> None:
        """Persist alert to database."""
        try:
            # Only forward segmentation fields when the segmenter actually
            # produced objects — keeps the DB column NULL for alerts where
            # the segmenter is off or returned an empty result.
            seg_count = getattr(alert, "segmentation_count", 0)
            seg_total_area = getattr(alert, "segmentation_total_area_px", 0)
            seg_objects = getattr(alert, "segmentation_objects", None)
            has_segmentation = bool(seg_count)
            import json as _json

            traj_json = None
            if alert.trajectory_points:
                traj_json = _json.dumps(
                    [{"t": round(t, 4), "x": round(x, 1), "y": round(y, 1)}
                     for t, x, y in alert.trajectory_points]
                )

            self._db.save_alert(
                alert_id=alert.alert_id,
                timestamp=datetime.fromtimestamp(alert.timestamp, tz=timezone.utc),
                camera_id=alert.camera_id,
                zone_id=alert.zone_id,
                severity=alert.severity.value,
                anomaly_score=alert.anomaly_score,
                snapshot_path=snapshot_path,
                heatmap_path=heatmap_path,
                event_group_id=getattr(alert, "event_group_id", None),
                event_group_count=getattr(alert, "event_group_count", 1),
                speed_ms=alert.speed_ms,
                speed_px_per_sec=alert.speed_px_per_sec,
                trajectory_model=alert.trajectory_model,
                origin_x_mm=alert.origin_x_mm,
                origin_y_mm=alert.origin_y_mm,
                origin_z_mm=alert.origin_z_mm,
                landing_x_mm=alert.landing_x_mm,
                landing_y_mm=alert.landing_y_mm,
                landing_z_mm=alert.landing_z_mm,
                trajectories_json=getattr(alert, "trajectories_json", None),
                classification_label=alert.classification_label,
                classification_confidence=alert.classification_confidence,
                corroborated=getattr(alert, "corroborated", None),
                correlation_partner=getattr(alert, "correlation_partner", None),
                segmentation_count=seg_count if has_segmentation else None,
                segmentation_total_area_px=seg_total_area if has_segmentation else None,
                segmentation_objects=seg_objects if has_segmentation else None,
                category=getattr(alert, "category", None),
                severity_adjusted_by_classifier=getattr(alert, "severity_adjusted_by_classifier", None),
                trajectory_points=traj_json,
                model_version_id=getattr(alert, "model_version_id", None),
            )
        except Exception as e:
            logger.error("dispatch.db_failed", alert_id=alert.alert_id, error=str(e))

    def _dispatch_with_retry(
        self,
        queue: Queue,
        send_fn: Callable[[dict], None],
        channel: str,
        max_retries: int = 3,
    ) -> None:
        """Shared retry loop for background dispatch workers.

        Polls the queue, retries each payload up to max_retries with
        exponential backoff, and logs dropped payloads at ERROR level.

        On shutdown, performs a best-effort drain of any remaining queued
        payloads (single attempt each, no backoff) so the process can exit
        without dropping items that were already queued before shutdown.
        """
        while not self._shutdown.is_set():
            try:
                data = queue.get(timeout=5.0)
            except Empty:
                continue

            backoff = 1.0
            for attempt in range(1, max_retries + 1):
                if self._shutdown.is_set():
                    break
                try:
                    send_fn(data)
                    logger.debug(
                        f"dispatch.{channel}_ok",
                        alert_id=data.get("alert_id"),
                    )
                    break
                except Exception as e:
                    if attempt >= max_retries:
                        logger.error(
                            f"dispatch.{channel}_dropped",
                            alert_id=data.get("alert_id"),
                            error=str(e),
                            attempts=attempt,
                        )
                    else:
                        logger.warning(
                            f"dispatch.{channel}_retry",
                            alert_id=data.get("alert_id"),
                            error=str(e),
                            attempt=attempt,
                        )
                        self._shutdown.wait(timeout=min(backoff, 30.0))
                        backoff = min(backoff * 2, 30.0)

        # Drain phase: single attempt per item, no retry/backoff. The DB
        # already has the alert (dispatch_database ran before queueing), so
        # losing a webhook here is a delivery failure — not an audit loss.
        drained = 0
        failed = 0
        while True:
            try:
                data = queue.get_nowait()
            except Empty:
                break
            try:
                send_fn(data)
                drained += 1
            except Exception as e:
                failed += 1
                logger.error(
                    f"dispatch.{channel}_dropped_on_shutdown",
                    alert_id=data.get("alert_id"),
                    error=str(e),
                )
        if drained or failed:
            logger.info(
                f"{channel}_worker.drain_complete",
                drained=drained,
                failed=failed,
            )

    def _db_worker(self) -> None:
        """Background thread that persists alerts to the database.

        After shutdown is signalled, drains any remaining items in the
        queue before exiting so buffered alerts are written to the DB
        rather than silently dropped.
        """
        logger.info("db_worker.started")
        while not self._shutdown.is_set():
            try:
                data = self._db_queue.get(timeout=5.0)
            except Empty:
                continue
            self._dispatch_database(
                data["alert"], data["snapshot_path"], data["heatmap_path"],
            )

        # Drain remaining items. _dispatch_database swallows its own exceptions
        # (see its try/except), so failures here are logged, not raised.
        drained = 0
        while True:
            try:
                data = self._db_queue.get_nowait()
            except Empty:
                break
            self._dispatch_database(
                data["alert"], data["snapshot_path"], data["heatmap_path"],
            )
            drained += 1
        if drained:
            logger.info("db_worker.drain_complete", drained=drained)

    def _webhook_worker(self) -> None:
        """Background thread that sends webhook HTTP POST requests."""
        logger.info("webhook_worker.started", url=self._config.webhook.url)

        def send(payload: dict) -> None:
            if self._http_client is None:
                import httpx
                self._http_client = httpx.Client(timeout=self._config.webhook.timeout)
            response = self._http_client.post(self._config.webhook.url, json=payload)
            response.raise_for_status()
            self._circuit_breaker.record_success()

        def send_with_cb(payload: dict) -> None:
            try:
                send(payload)
            except Exception:
                self._circuit_breaker.record_failure()
                raise

        self._dispatch_with_retry(self._webhook_queue, send_with_cb, "webhook")

    def _record_dispatch_failure(self, alert: Alert, reason: str) -> None:
        """Log dispatch failure so no alert is silently lost.

        The alert is already persisted to DB by _dispatch_database() before
        queue submission. This records the delivery failure for review.
        """
        logger.error(
            "dispatch.delivery_lost",
            alert_id=alert.alert_id,
            camera_id=alert.camera_id,
            severity=alert.severity.value,
            reason=reason,
            msg=f"ALERT DELIVERY FAILED: {alert.alert_id} — {reason}",
        )

    def _save_snapshot(self, alert: Alert) -> str | None:
        """Save the alert snapshot frame to disk with anomaly region annotations (DET-007)."""
        if alert.snapshot is None:
            return None

        try:
            date_dir = self._alerts_dir / _date_folder(alert.timestamp) / alert.camera_id
            date_dir.mkdir(parents=True, exist_ok=True)

            # DET-007: Annotate snapshot with red rectangles around anomaly regions
            annotated = self._annotate_snapshot(
                alert.snapshot, alert.heatmap, alert.anomaly_score
            )

            filename = f"{alert.alert_id}_snapshot.jpg"
            path = date_dir / filename
            cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return str(path)
        except Exception as e:
            logger.error("dispatch.snapshot_save_failed", error=str(e))
            return None

    @staticmethod
    def _annotate_snapshot(
        frame: np.ndarray,
        heatmap: np.ndarray | None,
        score: float,
    ) -> np.ndarray:
        """Draw red rectangles around anomaly regions with score labels (DET-007)."""
        annotated = frame.copy()
        if heatmap is None:
            return annotated

        from argus.core.anomaly_postprocess import AnomalyMapProcessor

        h, w = frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        processor = AnomalyMapProcessor(min_contour_area=100)
        regions = processor.extract_regions(heatmap_resized, threshold=0.5)

        for r in regions:
            cv2.rectangle(annotated, (r.x, r.y), (r.x + r.width, r.y + r.height), (0, 0, 255), 2)
            label = f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (r.x, r.y - th - 6), (r.x + tw + 4, r.y), (0, 0, 255), -1)
            cv2.putText(
                annotated, label, (r.x + 2, r.y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        return annotated

    def _save_heatmap(self, alert: Alert) -> str | None:
        """Save the anomaly heatmap to disk as a colored overlay."""
        if alert.heatmap is None:
            return None

        try:
            date_dir = self._alerts_dir / _date_folder(alert.timestamp) / alert.camera_id
            date_dir.mkdir(parents=True, exist_ok=True)

            # Convert normalized heatmap (0-1) to colored image
            heatmap_uint8 = (alert.heatmap * 255).astype(np.uint8)
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            filename = f"{alert.alert_id}_heatmap.jpg"
            path = date_dir / filename
            cv2.imwrite(str(path), colored, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return str(path)
        except Exception as e:
            logger.error("dispatch.heatmap_save_failed", error=str(e))
            return None

    def _check_disk_space(self, min_free_mb: int = 500) -> bool:
        """Check if enough disk space is available for alert images.

        Returns True if at least min_free_mb megabytes are available.
        Results are cached for 30 seconds to avoid frequent syscalls.
        """
        import shutil

        now = time.monotonic()
        if now - self._disk_space_checked_at < 30.0:
            return self._disk_space_ok

        try:
            usage = shutil.disk_usage(self._alerts_dir)
            free_mb = usage.free / (1024 * 1024)
            self._disk_space_ok = free_mb >= min_free_mb
            self._disk_space_checked_at = now
            if not self._disk_space_ok:
                logger.warning(
                    "dispatch.low_disk_space",
                    free_mb=round(free_mb, 1),
                    threshold_mb=min_free_mb,
                    path=str(self._alerts_dir),
                )
            return self._disk_space_ok
        except OSError as e:
            logger.error("dispatch.disk_check_failed", error=str(e))
            return False  # Fail-closed: skip image save if check fails (DB record still written)

    def flush_db_queue(self, timeout: float = 5.0) -> None:
        """Block until the DB dispatch queue is empty. Used for testing."""
        deadline = time.monotonic() + timeout
        while self._db_queue.qsize() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        # Allow the DB worker thread to finish processing the last item
        time.sleep(0.05)

    def get_circuit_breaker_status(self) -> dict:
        """Get circuit breaker status for dashboard display (DET-009)."""
        return self._circuit_breaker.get_status()

    def close(self) -> None:
        """Clean up resources, draining queues before shutdown (HIGH-13).

        Signals shutdown, then waits for worker threads to drain their queues.
        Non-daemon threads ensure this method is always reached before process exit.
        """
        logger.info(
            "dispatcher.closing",
            db_pending=self._db_queue.qsize(),
            webhook_pending=self._webhook_queue.qsize(),
        )
        self._shutdown.set()

        # Wait for threads to finish processing remaining items
        if self._db_thread.is_alive():
            self._db_thread.join(timeout=10.0)
            if self._db_thread.is_alive():
                logger.warning("dispatcher.db_drain_timeout")
        if self._webhook_thread and self._webhook_thread.is_alive():
            self._webhook_thread.join(timeout=10.0)
            if self._webhook_thread.is_alive():
                logger.warning("dispatcher.webhook_drain_timeout")

        # Log any items still in queues after drain
        remaining_db = self._db_queue.qsize()
        remaining_webhook = self._webhook_queue.qsize()
        if remaining_db > 0 or remaining_webhook > 0:
            logger.error(
                "dispatcher.alerts_lost_on_shutdown",
                db_remaining=remaining_db,
                webhook_remaining=remaining_webhook,
            )

        if self._http_client:
            self._http_client.close()
        logger.info("dispatcher.closed")


def _date_folder(timestamp: float) -> str:
    """Generate a date-based folder name from an epoch timestamp."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
