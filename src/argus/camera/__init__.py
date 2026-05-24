"""Camera runtime planning models."""

from argus.camera.models import (
    CameraRuntimePlan,
    DetectionInput,
    Go2RTCStreamSpec,
    PreviewInput,
    SnapshotInput,
)
from argus.camera.planner import CameraRuntimePlanner

__all__ = [
    "CameraRuntimePlan",
    "CameraRuntimePlanner",
    "DetectionInput",
    "Go2RTCStreamSpec",
    "PreviewInput",
    "SnapshotInput",
]
