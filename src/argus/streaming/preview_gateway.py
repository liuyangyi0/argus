"""Dashboard preview gateway for camera snapshots and browser streams."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any

import cv2
import structlog
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

logger = structlog.get_logger()

SourceProbe = Callable[[str | int, float], dict[str, Any]]


def _camera_attr(camera: Any, name: str, default: Any = None) -> Any:
    if isinstance(camera, dict):
        return camera.get(name, default)
    return getattr(camera, name, default)


def _manager_running(manager: Any) -> bool:
    return bool(getattr(manager, "running", False))


def _encode_jpeg(frame: Any, quality: int) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buffer.tobytes()


def probe_source_blocking(source: str | int, timeout: float = 5.0) -> dict[str, Any]:
    """Open a video source with a hard timeout and grab one frame."""
    import threading

    result: dict[str, Any] = {"ok": False}
    start = time.monotonic()

    def _worker() -> None:
        cap = None
        try:
            if isinstance(source, int):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                try:
                    cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
                except ValueError:
                    cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                result["error"] = "open_failed"
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                result["error"] = "no_frame"
                return
            result["ok"] = True
            result["resolution"] = [int(frame.shape[1]), int(frame.shape[0])]
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if cap is not None:
                cap.release()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        result.setdefault("error", "timeout")
    result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
    return result


class PreviewGateway:
    """Central access point for dashboard camera preview surfaces."""

    def __init__(
        self,
        state: Any,
        *,
        stream_executor: Executor | None = None,
        stream_semaphore: asyncio.Semaphore | None = None,
        max_stream_duration: float = 30 * 60,
        source_probe: SourceProbe = probe_source_blocking,
    ) -> None:
        self._state = state
        self._stream_executor = stream_executor
        self._stream_semaphore = stream_semaphore
        self._max_stream_duration = max_stream_duration
        self._source_probe = source_probe

    @classmethod
    def for_request(cls, request: Request, **kwargs: Any) -> PreviewGateway:
        return cls(request.app.state, **kwargs)

    def find_camera_config(self, camera_id: str) -> Any | None:
        """Find the persisted/original camera config for API responses."""
        app_config = getattr(self._state, "config", None)
        if app_config is not None:
            config = next(
                (
                    c
                    for c in getattr(app_config, "cameras", [])
                    if getattr(c, "camera_id", None) == camera_id
                ),
                None,
            )
            if config is not None:
                return config

        camera_manager = self.camera_manager
        if camera_manager is not None:
            get_config = getattr(camera_manager, "get_camera_config", None)
            if callable(get_config):
                config = get_config(camera_id)
                if config is not None and getattr(config, "camera_id", None) == camera_id:
                    return config
            config = next(
                (
                    c
                    for c in getattr(camera_manager, "_cameras", [])
                    if getattr(c, "camera_id", None) == camera_id
                ),
                None,
            )
            if config is not None:
                return config
        return None

    @property
    def camera_manager(self) -> Any | None:
        return getattr(self._state, "camera_manager", None)

    def stream_manager(self) -> Any | None:
        registry = self._stream_registry()
        if registry is not None:
            manager = getattr(registry, "manager", None)
            if manager is not None:
                return manager
        return getattr(self._state, "go2rtc", None)

    def stream_manager_running(self) -> bool:
        return _manager_running(self.stream_manager())

    def snapshot_response(self, camera_id: str) -> Response:
        """Return the latest camera frame as a raw JPEG response."""
        camera_manager = self.camera_manager
        if camera_manager is None:
            return Response(status_code=503)

        get_latest_frame = getattr(camera_manager, "get_latest_frame", None)
        if not callable(get_latest_frame):
            return Response(status_code=503)
        frame = get_latest_frame(camera_id)
        if frame is None:
            return Response(status_code=404)

        jpeg = _encode_jpeg(frame, 70)
        if jpeg is None:
            return Response(status_code=503)
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store"},
        )

    def latest_frame_stream_response(self, request: Request, camera_id: str) -> Response:
        """Return an MJPEG stream of the latest frames for a camera."""
        camera_manager = self.camera_manager
        if camera_manager is None:
            return Response(status_code=503)

        def _grab() -> bytes | None:
            frame = camera_manager.get_latest_frame(camera_id)
            if frame is None:
                return None
            return _encode_jpeg(frame, 60)

        return self._mjpeg_response(request, _grab)

    def heatmap_stream_response(self, request: Request, camera_id: str) -> Response:
        """Return an MJPEG stream with anomaly heatmap overlay."""
        camera_manager = self.camera_manager
        if camera_manager is None:
            return Response(status_code=503)

        def _grab() -> bytes | None:
            import numpy as np

            frame = camera_manager.get_latest_frame(camera_id)
            if frame is None:
                return None
            anomaly_map = camera_manager.get_latest_anomaly_map(camera_id)
            if anomaly_map is not None:
                h, w = frame.shape[:2]
                heatmap = cv2.resize(anomaly_map, (w, h))
                heatmap_u8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
                mask = heatmap > 0.3
                if mask.any():
                    mask_3ch = np.stack([mask] * 3, axis=-1)
                    frame = np.where(
                        mask_3ch,
                        cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0),
                        frame,
                    )
            return _encode_jpeg(frame, 60)

        return self._mjpeg_response(request, _grab)

    def streaming_info(self, request: Request, camera: Any) -> dict[str, Any]:
        """Return browser-facing streaming URLs for a camera."""
        camera_id = str(_camera_attr(camera, "camera_id", ""))
        fallback = f"/api/cameras/{camera_id}/stream"
        resolution = self.ensure_stream_registered(camera)
        if resolution is None or not getattr(resolution, "go2rtc_managed", False):
            return {
                "camera_id": camera_id,
                "go2rtc": False,
                "fallback": fallback,
            }

        manager = self.stream_manager()
        api_port = getattr(manager, "api_port", None)
        if api_port is None:
            return {
                "camera_id": camera_id,
                "go2rtc": False,
                "fallback": fallback,
            }

        stream_name = str(getattr(resolution, "stream_name", None) or camera_id)
        base = self._client_base_url(request, int(api_port))
        ws_scheme = "wss" if request.url.scheme == "https" else "ws"
        host = request.headers.get("host", "").split(":")[0] or "127.0.0.1"
        ws_base = f"{ws_scheme}://{host}:{api_port}"

        return {
            "camera_id": camera_id,
            "go2rtc": True,
            "webrtc_ws": f"{ws_base}/api/ws?src={stream_name}",
            "mse_ws": f"{ws_base}/api/ws?src={stream_name}&mode=mse",
            "hls": f"{base}/api/stream.m3u8?src={stream_name}",
            "mjpeg": f"{base}/api/frame.jpeg?src={stream_name}",
            "player": f"{base}/stream.html?src={stream_name}",
            "fallback": fallback,
        }

    def ensure_stream_registered(self, camera: Any) -> Any | None:
        """Register a camera stream through StreamRegistry or legacy go2rtc."""
        registry = self._stream_registry()
        if registry is not None:
            manager = getattr(registry, "manager", None)
            if manager is None:
                manager = getattr(self._state, "go2rtc", None)
            if not _manager_running(manager):
                return None
            ensure_registered = getattr(registry, "ensure_registered", None)
            if callable(ensure_registered):
                try:
                    return ensure_registered(camera, start_if_needed=False)
                except Exception:
                    logger.warning("preview.registry_register_failed", exc_info=True)
                    return None

        go2rtc = getattr(self._state, "go2rtc", None)
        if go2rtc is None or not _manager_running(go2rtc):
            return None

        from argus.streaming.go2rtc_manager import register_go2rtc_stream

        try:
            return register_go2rtc_stream(go2rtc, camera)
        except Exception:
            logger.warning("preview.go2rtc_register_failed", exc_info=True)
            return None

    def list_registered_streams(self) -> tuple[bool, dict[str, Any]]:
        manager = self.stream_manager()
        if manager is None or not _manager_running(manager):
            return False, {}
        list_streams = getattr(manager, "list_streams", None)
        if not callable(list_streams):
            return False, {}
        try:
            return True, list_streams()
        except Exception:
            logger.warning("preview.list_streams_failed", exc_info=True)
            return False, {}

    def probe_running_camera_connection(self, camera_id: str) -> dict[str, Any] | None:
        """Probe a running camera using its latest pipeline frame."""
        camera_manager = self.camera_manager
        if camera_manager is None:
            return None
        get_status = getattr(camera_manager, "get_status", None)
        if not callable(get_status):
            return None
        try:
            status = next(
                (
                    s
                    for s in get_status()
                    if getattr(s, "camera_id", None) == camera_id
                    and getattr(s, "running", False)
                ),
                None,
            )
        except Exception:
            return None
        if status is None:
            return None

        frame = None
        get_latest_frame = getattr(camera_manager, "get_latest_frame", None)
        if callable(get_latest_frame):
            frame = get_latest_frame(camera_id)
        if frame is not None:
            return {
                "ok": True,
                "latency_ms": 0.0,
                "resolution": [int(frame.shape[1]), int(frame.shape[0])],
                "source": "running_pipeline",
            }
        connected = bool(getattr(status, "connected", False))
        return {
            "ok": connected,
            "latency_ms": 0.0,
            "error": None if connected else "no_latest_frame",
            "source": "running_pipeline",
        }

    async def probe_source(self, source: str | int, timeout: float = 5.0) -> dict[str, Any]:
        return await asyncio.to_thread(self._source_probe, source, timeout)

    def _mjpeg_response(
        self,
        request: Request,
        grab_fn: Callable[[], bytes | None],
    ) -> Response:
        if self._stream_executor is None or self._stream_semaphore is None:
            return Response(status_code=503, content="Streaming resources unavailable")
        if self._stream_semaphore.locked():
            return Response(status_code=503, content="Too many active streams")

        async def _generate():
            loop = asyncio.get_running_loop()
            async with self._stream_semaphore:
                start = time.monotonic()
                try:
                    while True:
                        if await request.is_disconnected():
                            break
                        if time.monotonic() - start > self._max_stream_duration:
                            break
                        jpeg = await loop.run_in_executor(self._stream_executor, grab_fn)
                        if jpeg is not None:
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + jpeg
                                + b"\r\n"
                            )
                        await asyncio.sleep(0.2)
                except asyncio.CancelledError:
                    pass

        return StreamingResponse(
            _generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    def _stream_registry(self) -> Any | None:
        return getattr(self._state, "stream_registry", None)

    @staticmethod
    def _client_base_url(request: Request, port: int) -> str:
        host = request.headers.get("host", "").split(":")[0] or "127.0.0.1"
        scheme = "https" if request.url.scheme == "https" else "http"
        return f"{scheme}://{host}:{port}"
