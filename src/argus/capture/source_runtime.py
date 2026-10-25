"""One capture owner per camera, independent of analysis task lifetimes.

Capture factories must return unopened adapters.  The runtime never starts an
adapter's competing latest-frame thread.  Reconnect replaces the adapter only
after successful cleanup and advances the source epoch; failed cleanup blocks
replacement instead of opening another reader.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from argus.capture.base import CaptureAdapter
from argus.capture.camera import CameraState, FrameData
from argus.capture.frame_hub import (
    FrameHub,
    FrameSubscription,
    SourceToken,
    SubscriptionMode,
)
from argus.core.frame_identity import build_frame_identity


def create_source_capture(config, *, source_declared: str, stream_id: str | None = None):
    """Build an unopened adapter without constructing any detection models."""
    from argus.capture.camera import CameraCapture

    kwargs = {
        "camera_id": config.camera_id, "source": config.source,
        "fps_target": config.fps_target, "resolution": config.resolution,
        "reconnect_delay": config.reconnect_delay,
        "max_reconnect_attempts": config.max_reconnect_attempts,
        "source_declared": source_declared, "source_runtime": config.source,
        "stream_id": stream_id,
    }
    if config.protocol == "gige":
        from argus.capture.gige_capture import GigECapture

        return GigECapture(**kwargs, exposure=config.gige.exposure,
                           gain=config.gige.gain, pixel_format=config.gige.pixel_format)
    kwargs["protocol"] = config.protocol
    if config.protocol == "usb":
        kwargs.update(usb_backend=config.usb.preferred_backend,
                      usb_pixel_format=config.usb.pixel_format,
                      usb_min_runtime_fps=config.usb.min_runtime_fps)
    return CameraCapture(**kwargs)


@dataclass(frozen=True)
class SourceRuntimeState:
    camera_id: str
    state: str
    connected: bool
    running: bool
    source_session_id: str | None
    source_epoch: int
    generation: int
    cleanup_pending: bool
    error: str | None
    observer_error: str | None
    frames_published: int
    reconnect_count: int
    observer_dropped_frames: int
    reconnect_attempts: int
    reconnect_max_attempts: int
    reconnect_delay_seconds: float


class CameraSourceRuntime:
    """Own acquisition, broadcast, preview cache, and reconnect fencing."""

    def __init__(
        self,
        camera_id: str,
        capture_factory: Callable[[], CaptureAdapter],
        *,
        source: str = "",
        protocol: str = "rtsp",
        on_frame: Callable[[FrameData], None] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.protocol = protocol
        self.hub = FrameHub(camera_id)
        self._factory = capture_factory
        self._on_frame = on_frame
        self._lifecycle_lock = threading.RLock()
        self._lock = threading.RLock()
        self._capture: CaptureAdapter | None = None
        self._thread: threading.Thread | None = None
        self._observer_thread: threading.Thread | None = None
        self._observer_subscription: FrameSubscription | None = None
        self._stop_event = threading.Event()
        self._reconnect_event = threading.Event()
        self._state = "stopped"
        self._error: str | None = None
        self._observer_error: str | None = None
        self._cleanup_pending = False
        self._generation = 0
        self._session: str | None = None
        self._epoch = 0
        self._frames_published = 0
        self._reconnect_count = 0
        self._observer_dropped_frames = 0
        self._reconnect_attempts = 0
        self._reconnect_max_attempts = -1
        self._reconnect_initial_delay = 0.25
        self._reconnect_delay = self._reconnect_initial_delay
        self._identity_context: dict[str, str | None] = {}

    @property
    def source_token(self) -> SourceToken | None:
        return self.hub.source_token

    @property
    def latest_frame(self) -> FrameData | None:
        return self.hub.latest_frame

    @property
    def capture_state(self) -> CameraState:
        with self._lock:
            state = replace(self._capture.state) if self._capture else CameraState()
            if self._state not in ("running",):
                state.connected = False
            state.reconnect_count = self._reconnect_count
            if self._error:
                state.error = self._error
            return state

    @property
    def state(self) -> SourceRuntimeState:
        with self._lock:
            return SourceRuntimeState(
                self.camera_id, self._state,
                self._state == "running" and bool(
                    self._capture is not None and self._capture.state.connected
                ),
                self._thread is not None and self._thread.is_alive(),
                self._session, self._epoch, self._generation, self._cleanup_pending,
                self._error, self._observer_error, self._frames_published,
                self._reconnect_count,
                self._observer_subscription.status.dropped_frames
                if self._observer_subscription is not None else self._observer_dropped_frames,
                self._reconnect_attempts, self._reconnect_max_attempts,
                self._reconnect_delay,
            )

    def snapshot(self) -> dict:
        return asdict(self.state)

    def runtime_status(self) -> dict:
        """Expose the legacy capture diagnostic contract without SDK calls.

        The state property copies the currently owned adapter's already
        measured metadata.  Never delegate to arbitrary adapter status code
        while holding a source lock: status reads must not open/read devices.
        """
        state = self.capture_state
        return {
            "backend": state.backend,
            "requested_pixel_format": state.requested_pixel_format,
            "pixel_format": state.pixel_format,
            "requested_resolution": (
                tuple(state.requested_resolution) if state.requested_resolution is not None else None
            ),
            "actual_resolution": (
                tuple(state.actual_resolution) if state.actual_resolution is not None else None
            ),
            "requested_fps": state.requested_fps,
            "reported_fps": state.reported_fps,
            "actual_fps": state.actual_fps,
            "degraded": state.degraded,
            "degradation_reason": state.degradation_reason,
        }

    def commit_if_current(self, token: SourceToken, callback: Callable[[], None]) -> bool:
        """Linearize a short result commit against source stop/reconnect.

        Callbacks must not wait for workers or invoke source/task lifecycle
        operations.  Slow effects belong in a durable outbox outside this lock.
        """
        with self._lock:
            if (
                token != self.source_token or self._state != "running"
                or self._capture is None or not self._capture.state.connected
            ):
                return False
            callback()
            return True

    def subscribe(
        self, name: str, *, mode: SubscriptionMode = "latest", capacity: int = 32,
    ) -> FrameSubscription:
        return self.hub.subscribe(name, mode=mode, capacity=capacity)

    def configure_frame_identity(
        self, *, source_declared: str | None = None,
        source_runtime: str | None = None, stream_id: str | None = None,
    ) -> None:
        # Metadata configuration only: changing the physical input requires a
        # new source runtime.  The owning manager configures this before start.
        with self._lock:
            if source_declared is not None:
                self._identity_context["source_declared"] = source_declared
            if source_runtime is not None:
                self._identity_context["source_runtime"] = source_runtime
            self._identity_context["stream_id"] = stream_id

    def _new_capture(self) -> CaptureAdapter:
        capture = self._factory()
        if capture.camera_id != self.camera_id:
            raise ValueError("capture camera identity does not match its source owner")
        with self._lock:
            self._capture = capture
            self.source = capture.source
            self.protocol = capture.protocol
        configure = getattr(capture, "configure_frame_identity", None)
        if configure is not None and self._identity_context:
            configure(**self._identity_context)
        set_reconnect_handler = getattr(capture, "set_reconnect_handler", None)
        if set_reconnect_handler is not None:
            set_reconnect_handler(self.request_reconnect)
        return capture

    def start(self) -> bool:
        """Connect once, then start the sole reader; duplicate starts are no-ops."""
        with self._lifecycle_lock:
            with self._lock:
                if self._cleanup_pending:
                    return False
                if self._thread is not None and self._thread.is_alive():
                    return True
                if self._capture is not None:
                    # An unexpected worker exit never implies resource cleanup.
                    self._cleanup_pending = True
                    self._state = "cleanup_pending"
                    return False
                self._generation += 1
                generation = self._generation
                self._session = uuid.uuid4().hex
                self._state = "starting"
                self._error = None
                self._observer_error = None
                self._stop_event.clear()
                self._reconnect_event.clear()
            try:
                capture = self._new_capture()
                with self._lock:
                    if generation != self._generation or self._stop_event.is_set():
                        raise RuntimeError("source start was cancelled before connection")
                    self._reconnect_max_attempts = int(
                        getattr(capture, "max_reconnect_attempts", -1)
                    )
                    self._reconnect_initial_delay = max(
                        0.0, float(getattr(capture, "reconnect_delay", 0.25)),
                    )
                    self._reconnect_delay = self._reconnect_initial_delay
                    self._reconnect_attempts = 0
                if not capture.connect():
                    raise RuntimeError(capture.state.error or "source connection failed")
                with self._lock:
                    if generation != self._generation or self._stop_event.is_set():
                        raise RuntimeError("source start was cancelled before publication")
                    self._epoch += 1
                    self.hub.activate(self._session, self._epoch)
                    self._state = "running"
                    self._thread = threading.Thread(
                        target=self._read_loop, args=(generation,),
                        name=f"source-{self.camera_id}", daemon=True,
                    )
                    if self._on_frame is not None:
                        self._observer_subscription = self.subscribe(
                            "__source_observer__", mode="ordered_window", capacity=64,
                        )
                        self._observer_thread = threading.Thread(
                            target=self._observe_frames, args=(generation,),
                            name=f"source-observer-{self.camera_id}", daemon=True,
                        )
                        self._observer_thread.start()
                    self._thread.start()
                return True
            except Exception as exc:
                with self._lock:
                    self._error = f"source start failed: {exc}"
                error = self._error
                clean = self.stop()
                with self._lock:
                    if clean:
                        self._state = "failed"
                        self._error = error
                return False

    def request_reconnect(self) -> None:
        with self._lock:
            if self._state not in ("running", "waiting_source"):
                return
            self._state = "waiting_source"
            self.hub.invalidate()
            self._reconnect_event.set()

    def _replace_capture(self, generation: int) -> bool:
        """Runs only on the source reader, never concurrently with another read."""
        with self._lock:
            self._state = "waiting_source"
            self.hub.invalidate()
            self._reconnect_event.set()
            capture = self._capture
        try:
            if capture is not None:
                if capture.stop() is False:
                    raise RuntimeError("capture stop did not confirm cleanup")
            with self._lock:
                self._capture = None
        except Exception as exc:
            with self._lock:
                self._cleanup_pending = True
                self._state = "cleanup_pending"
                self._error = f"reconnect cleanup failed: {exc}"
            return False
        with self._lock:
            if (
                self._reconnect_max_attempts >= 0
                and self._reconnect_attempts >= self._reconnect_max_attempts
            ):
                self._state = "failed"
                self._cleanup_pending = True
                self._error = "source reconnect attempts exhausted; stop/start required"
                return False
            delay = self._reconnect_delay
            self._reconnect_attempts += 1
        if self._wait_for_reconnect(delay):
            return False
        with self._lock:
            if generation != self._generation or self._stop_event.is_set():
                return False
            self._reconnect_count += 1
        try:
            replacement = self._new_capture()
            connected = replacement.connect()
            with self._lock:
                if generation != self._generation or self._stop_event.is_set():
                    return False
                if not connected:
                    self._error = replacement.state.error or "source reconnection failed"
                    self._reconnect_delay = min(delay * 2, 60.0)
                    return True  # retry, after cleaning this unsuccessful adapter
                self._epoch += 1
                self.hub.activate(self._session, self._epoch)
                self._state = "running"
                self._error = None
                self._reconnect_attempts = 0
                self._reconnect_delay = self._reconnect_initial_delay
                self._reconnect_event.clear()
            return True
        except Exception as exc:
            with self._lock:
                self._error = f"source reconnection failed: {exc}"
                self._reconnect_delay = min(delay * 2, 60.0)
            return True

    def _wait_for_reconnect(self, delay: float) -> bool:
        return self._stop_event.wait(delay)

    def _read_loop(self, generation: int) -> None:
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    capture = self._capture
                    token = self.source_token
                if capture is None or not capture.state.connected or self._reconnect_event.is_set():
                    if not self._replace_capture(generation):
                        break
                    continue
                frame = capture.read()
                with self._lock:
                    if generation != self._generation or self._stop_event.is_set():
                        break
                    if self._reconnect_event.is_set() or token != self.source_token:
                        continue
                    if frame is not None and token is not None:
                        identity = frame.identity or build_frame_identity(
                            camera_id=self.camera_id,
                            source_declared=str(self._identity_context.get(
                                "source_declared", self.source,
                            )),
                            source_runtime=str(self._identity_context.get(
                                "source_runtime", self.source,
                            )),
                            stream_id=self._identity_context.get("stream_id"),
                            frame_index=frame.frame_number,
                            capture_monotonic_ts=frame.timestamp,
                            timestamp_provenance="source_runtime_receive_clock",
                        )
                        frame = replace(frame, identity=identity.with_updates(
                            source_session_id=token[0], source_epoch=token[1],
                        ))
                        if self.hub.publish(frame):
                            self._frames_published += 1
                    elif not capture.state.connected:
                        self._state = "waiting_source"
                        self.hub.invalidate()
                if frame is None:
                    self._stop_event.wait(0.01)
        except Exception as exc:
            with self._lock:
                self._error = f"source reader failed: {exc}"
                self._state = "failed"
                self._cleanup_pending = True
                self.hub.invalidate()

    def _observe_frames(self, generation: int) -> None:
        subscription = self._observer_subscription
        if subscription is None:
            return
        while not self._stop_event.is_set():
            frame = subscription.get(0.1)
            if frame is None:
                continue
            with self._lock:
                token = self.source_token
                if generation != self._generation or token is None:
                    continue
                if (frame.identity.source_session_id, frame.identity.source_epoch) != token:
                    continue
            try:
                self._on_frame(frame)
                with self._lock:
                    if generation == self._generation and token == self.source_token:
                        self._observer_error = None
            except Exception as exc:
                with self._lock:
                    # An old callback may finish after reconnect.  It carries
                    # its original frame identity, but may not overwrite the
                    # new source generation's observer health state.
                    if generation == self._generation and token == self.source_token:
                        self._observer_error = str(exc)

    def request_stop(self) -> None:
        """Fence a pending connect/read without waiting for lifecycle ownership."""
        with self._lock:
            self._generation += 1
            self._stop_event.set()
            self.hub.invalidate()
            self._state = "stopping"
            self._cleanup_pending = True
            capture = self._capture
        request = getattr(capture, "request_stop", None)
        if callable(request):
            try:
                request()
            except Exception as exc:
                with self._lock:
                    self._error = f"capture cancellation failed: {exc}"

    def stop(self, timeout: float = 5.0) -> bool:
        """Revoke generations, unblock capture, and require confirmed cleanup."""
        deadline = time.monotonic() + max(0.0, timeout)
        self.request_stop()
        if not self._lifecycle_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            with self._lock:
                self._error = "source lifecycle operation is still running; cleanup ownership retained"
            return False
        try:
            with self._lock:
                capture = self._capture
                threads = (self._thread, self._observer_thread)
                if self._observer_subscription is not None:
                    self._observer_dropped_frames = (
                        self._observer_subscription.status.dropped_frames
                    )
                    self.hub.unsubscribe(
                        "__source_observer__", subscription=self._observer_subscription,
                    )
            errors: list[str] = []
            deferred_cleanup = False
            if capture is not None:
                try:
                    request_stop = getattr(capture, "request_stop", None)
                    if request_stop is not None:
                        # Native SDK/OpenCV handles must not be released while
                        # read() is still running on the owning source thread.
                        request_stop()
                        deferred_cleanup = True
                    elif capture.stop() is False:
                        raise RuntimeError("capture stop did not confirm cleanup")
                except Exception as exc:
                    errors.append(f"capture cleanup failed: {exc}")
            for thread in threads:
                if thread is not None and thread.is_alive():
                    if thread is not threading.current_thread():
                        thread.join(max(0.0, deadline - time.monotonic()))
                    if thread.is_alive():
                        errors.append(f"worker still running: {thread.name}")
            if deferred_cleanup and not any(
                thread is not None and thread.is_alive() for thread in threads
            ):
                try:
                    if capture.stop() is False:
                        raise RuntimeError("capture stop did not confirm cleanup")
                except Exception as exc:
                    errors.append(f"capture cleanup failed: {exc}")
            with self._lock:
                # A connect may have completed late after stop captured its
                # previous adapter.  Never clear ownership of that new resource.
                late_capture = self._capture
            if (
                late_capture is not None and late_capture is not capture
                and not any(thread is not None and thread.is_alive() for thread in threads)
            ):
                try:
                    if late_capture.stop() is False:
                        raise RuntimeError("late capture cleanup not confirmed")
                except Exception as exc:
                    errors.append(f"late capture cleanup failed: {exc}")
            with self._lock:
                if errors:
                    self._state = "cleanup_pending"
                    self._error = "; ".join(errors)
                    return False
                self._capture = None
                self._thread = None
                self._observer_thread = None
                self._observer_subscription = None
                self._cleanup_pending = False
                self._state = "stopped"
                self._error = None
                return True
        finally:
            self._lifecycle_lock.release()


class SourceCaptureView:
    """Legacy CaptureAdapter view without permission to close its camera source."""

    def __init__(
        self, source_runtime: CameraSourceRuntime, name: str, *, timeout: float = 1.0,
    ) -> None:
        self._source_runtime = source_runtime
        self._name = name
        self._timeout = timeout
        self._subscription = source_runtime.subscribe(name)

    @property
    def camera_id(self) -> str:
        return self._source_runtime.camera_id

    @property
    def source(self) -> str:
        return self._source_runtime.source

    @property
    def protocol(self) -> str:
        return self._source_runtime.protocol

    @property
    def state(self) -> CameraState:
        return self._source_runtime.capture_state

    def runtime_status(self) -> dict:
        return self._source_runtime.runtime_status()

    def connect(self) -> bool:
        # Source lifecycle is owned by CameraManager, not task initialization.
        if self._subscription.status.closed:
            self._subscription = self._source_runtime.subscribe(self._name)
        return self.state.connected

    def read(self) -> FrameData | None:
        frame = self._subscription.get(self._timeout)
        # Legacy detection contains in-place OpenCV transforms.  Its adapter
        # owns writable pixels without weakening the shared hub's contract.
        return replace(frame, frame=frame.frame.copy()) if frame is not None else None

    def read_latest(self) -> FrameData | None:
        return self.read()

    def start_capture_thread(self) -> None:
        """No-op: the sole source reader is already owned by CameraSourceRuntime."""

    def request_reconnect(self) -> None:
        # A stale consumer request cannot restart a healthy shared source.
        if not self.state.connected:
            self._source_runtime.request_reconnect()

    def reconnect(self) -> bool:
        self._source_runtime.request_reconnect()
        return self.state.connected

    def configure_frame_identity(self, **kwargs: str | None) -> None:
        self._source_runtime.configure_frame_identity(**kwargs)

    def stop(self) -> None:
        self._source_runtime.hub.unsubscribe(self._name, subscription=self._subscription)

    def stop_capture_thread(self) -> None:
        self.stop()

    def disconnect(self) -> None:
        self.stop()

    def release(self) -> None:
        self.stop()
