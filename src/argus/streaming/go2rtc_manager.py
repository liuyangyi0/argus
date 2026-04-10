"""go2rtc process lifecycle manager and HTTP API client.

go2rtc is a standalone streaming proxy that converts RTSP camera feeds
into browser-friendly formats (WebRTC, MSE, HLS, MJPEG) with hardware
codec pass-through and sub-second latency.

This module manages go2rtc as a child process and provides a thin Python
client for its REST API so that Argus can dynamically register / remove
camera streams at runtime.

Ref: https://github.com/AlexxIT/go2rtc
"""

from __future__ import annotations

import atexit
import json
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_API_PORT = 1984
_DEFAULT_RTSP_PORT = 8554
_DEFAULT_WEBRTC_PORT = 8555
_HEALTH_POLL_INTERVAL = 0.5  # seconds
_HEALTH_POLL_TIMEOUT = 10  # seconds


def usb_to_go2rtc_source(device_index: str | int) -> str:
    """Convert a USB camera device index to a go2rtc ffmpeg source URL.

    go2rtc supports USB cameras via its built-in FFmpeg integration:
      ``ffmpeg:device?video=<index>#video=h264``

    This avoids the legacy MJPEG fallback path (which creates long-lived
    HTTP connections that exhaust the browser's per-origin connection limit).

    Ref: https://github.com/AlexxIT/go2rtc/issues/159
    """
    idx = int(device_index)
    return f"ffmpeg:device?video={idx}#video=h264"


def _find_go2rtc_binary() -> Path | None:
    """Locate the go2rtc binary on the system.

    Search order:
    1. ``bin/go2rtc`` relative to the project root
    2. ``go2rtc`` on ``$PATH``
    """
    # Project-local binary
    project_root = Path(__file__).resolve().parents[3]  # src/argus/streaming -> project root
    suffix = ".exe" if platform.system() == "Windows" else ""
    local_bin = project_root / "bin" / f"go2rtc{suffix}"
    if local_bin.is_file():
        return local_bin

    # System PATH
    found = shutil.which(f"go2rtc{suffix}")
    if found:
        return Path(found)

    return None


class Go2RTCManager:
    """Manages the go2rtc child process and exposes its REST API.

    Parameters
    ----------
    api_port:
        HTTP API / WebRTC signalling port (default 1984).
    rtsp_port:
        RTSP listener port (default 8554).
    webrtc_port:
        WebRTC ICE/UDP port (default 8555).
    binary_path:
        Explicit path to the go2rtc executable.  When *None* the manager
        searches ``bin/`` and ``$PATH``.
    config_dir:
        Directory where the generated ``go2rtc.yaml`` will be written.
        Defaults to a temporary directory.
    """

    def __init__(
        self,
        *,
        api_port: int = _DEFAULT_API_PORT,
        rtsp_port: int = _DEFAULT_RTSP_PORT,
        webrtc_port: int = _DEFAULT_WEBRTC_PORT,
        binary_path: str | Path | None = None,
        config_dir: str | Path | None = None,
    ) -> None:
        self.api_port = api_port
        self.rtsp_port = rtsp_port
        self.webrtc_port = webrtc_port
        self._base_url = f"http://127.0.0.1:{api_port}"

        if binary_path:
            self._binary = Path(binary_path)
        else:
            self._binary = _find_go2rtc_binary()

        if config_dir:
            self._config_dir = Path(config_dir)
        else:
            self._config_dir = Path(tempfile.mkdtemp(prefix="go2rtc_"))

        self._process: subprocess.Popen | None = None
        self._stdout_thread: threading.Thread | None = None
        self._auto_config_dir = config_dir is None  # track for cleanup
        self._http = httpx.Client(base_url=self._base_url, timeout=5.0)

        # Safety net: ensure cleanup even on unexpected interpreter shutdown
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def _write_config(self, streams: dict[str, str] | None = None) -> Path:
        """Write a minimal ``go2rtc.yaml`` and return its path."""
        cfg: dict[str, Any] = {
            "api": {
                "listen": f":{self.api_port}",
                "origin": "*",
            },
            "rtsp": {
                "listen": f":{self.rtsp_port}",
            },
            "webrtc": {
                "listen": f":{self.webrtc_port}",
            },
        }
        if streams:
            cfg["streams"] = streams

        config_path = self._config_dir / "go2rtc.json"
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        logger.info(
            "go2rtc.config_written",
            path=str(config_path),
            stream_count=len(streams or {}),
        )
        return config_path

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def _kill_stale_process(self) -> None:
        """Kill any orphaned go2rtc still occupying our API port."""
        try:
            resp = self._http.get("/api", timeout=2)
            if resp.status_code == 200:
                logger.warning("go2rtc.killing_stale", port=self.api_port)
                # Try graceful exit via API first
                try:
                    self._http.post("/api/exit", timeout=2)
                except Exception:
                    logger.debug("go2rtc.graceful_exit_failed", exc_info=True)
                time.sleep(1)
                # If still alive, find and kill by port (Windows)
                if platform.system() == "Windows":
                    try:
                        result = subprocess.run(
                            ["netstat", "-ano"],
                            capture_output=True, text=True, timeout=5,
                        )
                        for line in result.stdout.splitlines():
                            if f":{self.api_port}" in line and "LISTENING" in line:
                                pid = line.strip().split()[-1]
                                subprocess.run(
                                    ["taskkill", "/F", "/PID", pid],
                                    capture_output=True, timeout=5,
                                )
                                logger.info("go2rtc.stale_killed", pid=pid)
                                break
                    except Exception as exc:
                        logger.warning("go2rtc.stale_kill_failed", error=str(exc))
                else:
                    # Unix: use fuser or lsof
                    try:
                        subprocess.run(
                            ["fuser", "-k", f"{self.api_port}/tcp"],
                            capture_output=True, timeout=5,
                        )
                    except Exception:
                        logger.debug("go2rtc.unix_fuser_kill_failed", exc_info=True)
                time.sleep(0.5)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException):
            pass  # No stale process — port is free

    def start(self, initial_streams: dict[str, str] | None = None) -> None:
        """Start the go2rtc process.

        Parameters
        ----------
        initial_streams:
            Mapping of ``{stream_name: source_url}`` to pre-register.
            Example: ``{"cam_01": "rtsp://192.168.1.10:554/stream1"}``
        """
        if self._process and self._process.poll() is None:
            logger.warning("go2rtc.already_running", pid=self._process.pid)
            return

        # Kill orphaned go2rtc from a previous run that wasn't cleaned up
        self._kill_stale_process()

        if self._binary is None:
            logger.error(
                "go2rtc.binary_not_found",
                hint="Download from https://github.com/AlexxIT/go2rtc/releases "
                     "and place in bin/ or add to PATH",
            )
            raise FileNotFoundError(
                "go2rtc binary not found.  Download from "
                "https://github.com/AlexxIT/go2rtc/releases and place in bin/"
            )

        config_path = self._write_config(initial_streams)

        cmd = [str(self._binary), "-config", str(config_path)]
        logger.info("go2rtc.starting", cmd=cmd)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Drain stdout in a background thread to prevent pipe buffer deadlock
        self._stdout_thread = threading.Thread(
            target=self._drain_stdout, daemon=True, name="go2rtc-stdout",
        )
        self._stdout_thread.start()

        # Wait for the API to become reachable
        self._wait_for_ready()
        logger.info("go2rtc.started", pid=self._process.pid, api_port=self.api_port)

    def stop(self) -> None:
        """Stop the go2rtc process gracefully."""
        if self._process is None:
            return

        if self._process.poll() is not None:
            logger.info("go2rtc.already_stopped", returncode=self._process.returncode)
            self._process = None
            return

        logger.info("go2rtc.stopping", pid=self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("go2rtc.force_kill", pid=self._process.pid)
            self._process.kill()
            self._process.wait(timeout=3)

        self._process = None
        logger.info("go2rtc.stopped")

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _wait_for_ready(self) -> None:
        """Poll the health endpoint until go2rtc is responsive."""
        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT
        while time.monotonic() < deadline:
            try:
                resp = self._http.get("/api")
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL)

        raise TimeoutError(
            f"go2rtc did not become ready within {_HEALTH_POLL_TIMEOUT}s"
        )

    # ------------------------------------------------------------------
    # Stream management via REST API
    # ------------------------------------------------------------------

    def register_camera(
        self, camera_id: str, source: str, protocol: str,
    ) -> str | None:
        """Register a camera and return the RTSP re-stream URL.

        Handles protocol dispatch (RTSP pass-through, USB→ffmpeg conversion).
        Returns the ``rtsp://`` URL that the pipeline should read from,
        or ``None`` if the protocol is unsupported or registration fails.
        """
        if protocol == "rtsp":
            go2rtc_source = source
        elif protocol == "usb":
            go2rtc_source = usb_to_go2rtc_source(source)
        else:
            return None
        try:
            self.add_stream(camera_id, go2rtc_source)
        except Exception:
            logger.warning("go2rtc.register_failed", camera_id=camera_id)
            return None
        return f"rtsp://127.0.0.1:{self.rtsp_port}/{camera_id}"

    def add_stream(self, name: str, source: str) -> None:
        """Register a new stream source (e.g. RTSP URL).

        Parameters
        ----------
        name:
            Logical stream name (usually ``camera_id``).
        source:
            Source URL, e.g. ``rtsp://192.168.1.10:554/stream1``.
        """
        resp = self._http.put(
            "/api/streams",
            params={"name": name, "src": source},
        )
        resp.raise_for_status()
        logger.info("go2rtc.stream_added", name=name, source=source)

    def remove_stream(self, name: str) -> None:
        """Remove a stream by name."""
        resp = self._http.delete("/api/streams", params={"name": name})
        resp.raise_for_status()
        logger.info("go2rtc.stream_removed", name=name)

    def list_streams(self) -> dict[str, Any]:
        """Return all registered streams and their state."""
        resp = self._http.get("/api/streams")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # URL helpers for the frontend
    # ------------------------------------------------------------------

    def webrtc_url(self, stream_name: str) -> str:
        """Return the WebRTC signalling URL for a stream."""
        return f"{self._base_url}/api/ws?src={stream_name}"

    def mse_url(self, stream_name: str) -> str:
        """Return the MSE (Media Source Extensions) URL for a stream."""
        return f"{self._base_url}/api/ws?src={stream_name}&mode=mse"

    def hls_url(self, stream_name: str) -> str:
        """Return the HLS playlist URL for a stream."""
        return f"{self._base_url}/api/stream.m3u8?src={stream_name}"

    def mjpeg_url(self, stream_name: str) -> str:
        """Return the MJPEG fallback URL for a stream."""
        return f"{self._base_url}/api/frame.jpeg?src={stream_name}"

    def stream_html_url(self, stream_name: str) -> str:
        """Return the built-in player page URL (embeddable via iframe)."""
        return f"{self._base_url}/stream.html?src={stream_name}"

    # ------------------------------------------------------------------
    # Sync cameras from Argus config
    # ------------------------------------------------------------------

    def sync_cameras(self, cameras: list[dict[str, Any]]) -> None:
        """Register cameras from the Argus configuration with go2rtc.

        RTSP cameras are registered directly.  USB cameras are converted to
        go2rtc's ``ffmpeg:device`` source format.  Other protocols (e.g.
        ``file``) are skipped.

        Parameters
        ----------
        cameras:
            List of camera config dicts, each having at least
            ``camera_id``, ``source``, and ``protocol`` keys.
        """
        current = self.list_streams()
        registered = set(current.keys()) if current else set()

        for cam in cameras:
            cam_id = cam["camera_id"]
            if cam_id in registered:
                registered.discard(cam_id)
                continue
            self.register_camera(cam_id, cam["source"], cam.get("protocol", "rtsp"))

        # Remove stale streams that are no longer in the config
        for stale in registered:
            try:
                self.remove_stream(stale)
            except Exception:
                logger.warning("go2rtc.remove_stale_failed", name=stale)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _drain_stdout(self) -> None:
        """Read go2rtc stdout until EOF, forwarding lines to the logger."""
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                logger.debug("go2rtc.output", line=line.rstrip())
        except ValueError:
            pass  # stdout closed

    _closed = False

    def close(self) -> None:
        """Stop the process and release HTTP resources (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self.stop()
        try:
            self._http.close()
        except Exception:
            logger.debug("go2rtc.http_client_close_failed", exc_info=True)
        # Clean up auto-created temp config directory
        if self._auto_config_dir and self._config_dir.exists():
            shutil.rmtree(self._config_dir, ignore_errors=True)
        # Unregister atexit handler to avoid double-cleanup
        try:
            atexit.unregister(self.close)
        except Exception:
            logger.debug("go2rtc.atexit_unregister_failed", exc_info=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            logger.debug("go2rtc.destructor_close_failed", exc_info=True)
