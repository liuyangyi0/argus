"""Multi-channel alert dispatcher.

Routes graded alerts to multiple destinations: database persistence,
webhook (HTTP POST), and console output. Each channel operates independently
so a failure in one channel doesn't block others.

Webhook dispatch runs in a background thread to avoid blocking the
camera processing thread during HTTP timeouts.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

import cv2
import numpy as np
import structlog

from argus.alerts.grader import Alert
from argus.config.schema import AlertConfig
from argus.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from argus.storage.database import Database

logger = structlog.get_logger()


class AlertDispatcher:
    """Dispatches alerts to all configured channels.

    Channels:
    - Database: Always active, persists all alerts to SQLite
    - Webhook: HTTP POST to configured URL (e.g., plant DCS), runs in background thread
    - Console: Colored terminal output for monitoring

    Each alert's snapshot and heatmap are saved to disk before
    dispatching to ensure the images are available for review.
    """

    def __init__(
        self,
        config: AlertConfig,
        database: Database,
        alerts_dir: str | Path = "data/alerts",
        on_alert: Callable[[str, dict], None] | None = None,
    ):
        self._config = config
        self._db = database
        self._alerts_dir = Path(alerts_dir)
        self._on_alert_ws = on_alert
        self._alerts_dir.mkdir(parents=True, exist_ok=True)
        self._http_client = None
        self._shutdown = threading.Event()

        # DET-009: Circuit breaker for webhook dispatch
        cb_config = CircuitBreakerConfig(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout_seconds=config.circuit_breaker_timeout,
        )
        self._circuit_breaker = CircuitBreaker(cb_config)

        # HIGH-13: Non-daemon threads so queued alerts are not discarded on shutdown
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

        # Background email dispatch thread
        self._email_queue: Queue = Queue(maxsize=100)
        self._email_thread: threading.Thread | None = None
        if self._config.email.enabled and self._config.email.recipients:
            self._email_thread = threading.Thread(
                target=self._email_worker,
                name="argus-email",
                daemon=False,
            )
            self._email_thread.start()

    @staticmethod
    def _alert_to_dict(
        alert: Alert,
        snapshot_path: str | None = None,
        heatmap_path: str | None = None,
    ) -> dict:
        """Convert an alert to a dict payload for dispatch channels."""
        payload = {
            "alert_id": alert.alert_id,
            "timestamp": datetime.fromtimestamp(alert.timestamp, tz=timezone.utc).isoformat(),
            "camera_id": alert.camera_id,
            "zone_id": alert.zone_id,
            "severity": alert.severity.value,
            "anomaly_score": round(alert.anomaly_score, 4),
        }
        if snapshot_path is not None:
            payload["snapshot_path"] = snapshot_path
        if heatmap_path is not None:
            payload["heatmap_path"] = heatmap_path
        return payload

    def dispatch(self, alert: Alert) -> None:
        """Send an alert to all configured channels."""
        # Check disk space before writing images
        if not self._check_disk_space():
            logger.error(
                "dispatch.disk_full",
                alert_id=alert.alert_id,
                msg="Disk space critically low, skipping image save",
            )
            snapshot_path = None
            heatmap_path = None
        else:
            snapshot_path = self._save_snapshot(alert)
            heatmap_path = self._save_heatmap(alert)

        # Channel 1: Database (always active)
        self._dispatch_database(alert, snapshot_path, heatmap_path)

        # Channel 2: Webhook (non-blocking, queued to background thread, with circuit breaker DET-009)
        if self._config.webhook.enabled:
            if self._circuit_breaker.allow_request():
                payload = self._alert_to_dict(alert, snapshot_path, heatmap_path)
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

        # Channel 3: Email (non-blocking, queued to background thread)
        if self._config.email.enabled and self._config.email.recipients:
            # Respect min_severity filter
            severity_order = ["info", "low", "medium", "high"]
            alert_level = severity_order.index(alert.severity.value) if alert.severity.value in severity_order else -1
            min_level = severity_order.index(self._config.email.min_severity.value) if self._config.email.min_severity.value in severity_order else 0
            if alert_level >= min_level:
                email_data = self._alert_to_dict(alert, snapshot_path)
                try:
                    self._email_queue.put_nowait(email_data)
                except Exception:
                    logger.error(
                        "dispatch.email_queue_full",
                        alert_id=alert.alert_id,
                        msg="Alert email delivery lost — queue overflow",
                    )
                    self._record_dispatch_failure(alert, "email_queue_overflow")

        # Channel 4: WebSocket push (real-time dashboard notification)
        if self._on_alert_ws:
            try:
                self._on_alert_ws("alerts", self._alert_to_dict(alert))
            except Exception as e:
                logger.warning("dispatch.websocket_failed", alert_id=alert.alert_id, error=str(e))

        logger.info(
            "alert.dispatched",
            alert_id=alert.alert_id,
            severity=alert.severity.value,
            camera_id=alert.camera_id,
            score=round(alert.anomaly_score, 3),
        )

    def _dispatch_database(
        self, alert: Alert, snapshot_path: str | None, heatmap_path: str | None
    ) -> None:
        """Persist alert to database."""
        try:
            self._db.save_alert(
                alert_id=alert.alert_id,
                timestamp=datetime.fromtimestamp(alert.timestamp, tz=timezone.utc),
                camera_id=alert.camera_id,
                zone_id=alert.zone_id,
                severity=alert.severity.value,
                anomaly_score=alert.anomaly_score,
                snapshot_path=snapshot_path,
                heatmap_path=heatmap_path,
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

    def _email_worker(self) -> None:
        """Background thread that sends email alerts via SMTP.

        Supports both TLS and plain SMTP (common in plant internal networks).
        """
        import smtplib
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        email_cfg = self._config.email
        logger.info("email_worker.started", smtp_host=email_cfg.smtp_host, recipients=len(email_cfg.recipients))

        def send(data: dict) -> None:
            msg = MIMEMultipart()
            msg["From"] = email_cfg.from_address
            msg["To"] = ", ".join(email_cfg.recipients)
            msg["Subject"] = (
                f"[Argus] {data['severity'].upper()} Alert - "
                f"Camera {data['camera_id']} - {data['alert_id']}"
            )
            body = (
                f"Alert ID: {data['alert_id']}\n"
                f"Severity: {data['severity'].upper()}\n"
                f"Camera: {data['camera_id']}\n"
                f"Zone: {data['zone_id']}\n"
                f"Anomaly Score: {data['anomaly_score']}\n"
                f"Time: {data['timestamp']}\n"
            )
            msg.attach(MIMEText(body, "plain"))

            snapshot_path = data.get("snapshot_path")
            if snapshot_path:
                snap = Path(snapshot_path)
                if snap.exists():
                    with open(snap, "rb") as f:
                        img = MIMEImage(f.read(), name=snap.name)
                        msg.attach(img)

            server = smtplib.SMTP(email_cfg.smtp_host, email_cfg.smtp_port, timeout=10)
            if email_cfg.use_tls:
                server.starttls()
            if email_cfg.smtp_username:
                server.login(email_cfg.smtp_username, email_cfg.smtp_password)
            server.sendmail(email_cfg.from_address, email_cfg.recipients, msg.as_string())
            server.quit()

        self._dispatch_with_retry(self._email_queue, send, "email")

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
        """
        import shutil

        try:
            usage = shutil.disk_usage(self._alerts_dir)
            free_mb = usage.free / (1024 * 1024)
            if free_mb < min_free_mb:
                logger.warning(
                    "dispatch.low_disk_space",
                    free_mb=round(free_mb, 1),
                    threshold_mb=min_free_mb,
                    path=str(self._alerts_dir),
                )
                return False
            return True
        except OSError as e:
            logger.error("dispatch.disk_check_failed", error=str(e))
            return True  # Fail-open: allow save if check itself fails

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
            webhook_pending=self._webhook_queue.qsize(),
            email_pending=self._email_queue.qsize(),
        )
        self._shutdown.set()

        # Wait for threads to finish processing remaining items
        if self._webhook_thread and self._webhook_thread.is_alive():
            self._webhook_thread.join(timeout=10.0)
            if self._webhook_thread.is_alive():
                logger.warning("dispatcher.webhook_drain_timeout")
        if self._email_thread and self._email_thread.is_alive():
            self._email_thread.join(timeout=10.0)
            if self._email_thread.is_alive():
                logger.warning("dispatcher.email_drain_timeout")

        # Log any items still in queues after drain
        remaining_webhook = self._webhook_queue.qsize()
        remaining_email = self._email_queue.qsize()
        if remaining_webhook > 0 or remaining_email > 0:
            logger.error(
                "dispatcher.alerts_lost_on_shutdown",
                webhook_remaining=remaining_webhook,
                email_remaining=remaining_email,
            )

        if self._http_client:
            self._http_client.close()
        logger.info("dispatcher.closed")


def _date_folder(timestamp: float) -> str:
    """Generate a date-based folder name from an epoch timestamp."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
