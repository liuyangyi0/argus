"""Tests for camera runtime input planning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from argus.camera import CameraRuntimePlanner
from argus.config.schema import CameraConfig, GigEConfig
from argus.streaming.go2rtc_manager import (
    gige_to_go2rtc_source,
    usb_to_go2rtc_source,
)


def _plan(config: CameraConfig):
    before = config.model_dump(mode="json")
    plan = CameraRuntimePlanner.build(config)
    assert config.model_dump(mode="json") == before
    return plan


def test_usb_detection_reads_go2rtc_rtsp_without_mutating_config():
    config = CameraConfig(camera_id="usb_01", name="USB", source="0", protocol="usb")

    plan = _plan(config)

    assert config.source == "0"
    assert config.protocol == "usb"
    assert plan.original_source == "0"
    assert plan.original_protocol == "usb"
    assert plan.detection.source == "rtsp://127.0.0.1:8554/usb_01"
    assert plan.detection.protocol == "rtsp"
    assert plan.detection.backend == "opencv"
    assert plan.detection.via_go2rtc is True
    assert plan.detection.stream_name == "usb_01"
    assert plan.go2rtc_stream is not None
    assert plan.go2rtc_stream.source == usb_to_go2rtc_source("0")
    assert plan.go2rtc_stream.registration == "rest_api"
    assert plan.preview.mode == "go2rtc"
    assert plan.preview.stream_name == "usb_01"


def test_rtsp_detection_reads_original_source_while_preview_uses_go2rtc():
    source = "rtsp://192.168.1.10:554/stream1"
    config = CameraConfig(camera_id="rtsp_01", name="RTSP", source=source, protocol="rtsp")

    plan = _plan(config)

    assert plan.detection.source == source
    assert plan.detection.protocol == "rtsp"
    assert plan.detection.via_go2rtc is False
    assert plan.go2rtc_stream is not None
    assert plan.go2rtc_stream.source == source
    assert plan.go2rtc_stream.runtime_rtsp_url == "rtsp://127.0.0.1:8554/rtsp_01"
    assert plan.preview.mode == "go2rtc"
    assert plan.snapshot.mode == "latest_frame_jpeg"


def test_gige_detection_stays_on_sdk_and_exec_preview_is_planned():
    capture_script = "scripts/gige_capture.ps1"
    config = CameraConfig(
        camera_id="gige_01",
        name="GigE",
        source="192.168.1.20",
        protocol="gige",
        gige=GigEConfig(capture_script=capture_script),
    )

    plan = _plan(config)

    assert plan.detection.source == "192.168.1.20"
    assert plan.detection.protocol == "gige"
    assert plan.detection.backend == "gige_sdk"
    assert plan.detection.via_go2rtc is False
    assert plan.go2rtc_stream is not None
    assert plan.go2rtc_stream.source == gige_to_go2rtc_source(capture_script)
    assert plan.go2rtc_stream.registration == "initial_config"
    assert plan.preview.mode == "go2rtc"
    assert plan.preview.stream_name == "gige_01"


def test_file_detection_reads_file_and_preview_uses_latest_frame_mjpeg():
    config = CameraConfig(
        camera_id="file_01",
        name="File",
        source="data/samples/file.mp4",
        protocol="file",
    )

    plan = _plan(config)

    assert plan.detection.source == "data/samples/file.mp4"
    assert plan.detection.protocol == "file"
    assert plan.detection.backend == "opencv"
    assert plan.detection.via_go2rtc is False
    assert plan.go2rtc_stream is None
    assert plan.preview.mode == "latest_frame_mjpeg"
    assert plan.preview.fallback_path == "/api/cameras/file_01/stream"
    assert plan.snapshot.path == "/api/cameras/file_01/snapshot"


def test_runtime_plan_models_are_immutable():
    config = CameraConfig(
        camera_id="rtsp_02",
        name="RTSP",
        source="rtsp://example/s",
        protocol="rtsp",
    )
    plan = CameraRuntimePlanner.build(config)

    with pytest.raises(FrozenInstanceError):
        plan.detection.source = "rtsp://changed"  # type: ignore[misc]
