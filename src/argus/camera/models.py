"""Runtime camera input plan models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DetectionBackend = Literal["opencv", "gige_sdk"]
Go2RTCRegistration = Literal["rest_api", "initial_config"]
PreviewMode = Literal["go2rtc", "latest_frame_mjpeg", "disabled"]
SnapshotMode = Literal["latest_frame_jpeg", "disabled"]


@dataclass(frozen=True, slots=True)
class DetectionInput:
    """Source that the detection pipeline should open."""

    source: str
    protocol: str
    backend: DetectionBackend
    via_go2rtc: bool = False
    stream_name: str | None = None


@dataclass(frozen=True, slots=True)
class PreviewInput:
    """Source family for browser live preview."""

    mode: PreviewMode
    stream_name: str | None = None
    fallback_path: str | None = None


@dataclass(frozen=True, slots=True)
class SnapshotInput:
    """Source family for still-frame snapshots."""

    mode: SnapshotMode
    path: str | None = None


@dataclass(frozen=True, slots=True)
class Go2RTCStreamSpec:
    """A stream that should be registered with go2rtc."""

    name: str
    source: str
    source_protocol: str
    runtime_rtsp_url: str
    registration: Go2RTCRegistration


@dataclass(frozen=True, slots=True)
class CameraRuntimePlan:
    """Separated runtime inputs for one persisted camera config."""

    camera_id: str
    original_source: str
    original_protocol: str
    detection: DetectionInput
    preview: PreviewInput
    snapshot: SnapshotInput
    go2rtc_stream: Go2RTCStreamSpec | None = None
