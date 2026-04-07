"""Tests for GlobalDegradationManager (UX v2 §5)."""

import time

import pytest

from argus.core.degradation import (
    DEGRADATION_TEMPLATES,
    DegradationLevel,
    GlobalDegradationEvent,
    GlobalDegradationManager,
)


class TestGlobalDegradationManager:
    def test_report_creates_event(self):
        mgr = GlobalDegradationManager()
        event_id = mgr.report("yolo_down", camera_id="cam_01")
        assert event_id.startswith("deg-")
        assert mgr.active_count == 1

    def test_report_with_format_kwargs(self):
        mgr = GlobalDegradationManager()
        event_id = mgr.report(
            "rtsp_broken",
            camera_id="cam_02",
            camera="cam_02",
            n=3,
            max=10,
            t=12,
        )
        active = mgr.get_active()
        assert len(active) == 1
        assert "cam_02" in active[0]["title"]

    def test_resolve_removes_event(self):
        mgr = GlobalDegradationManager()
        eid = mgr.report("yolo_down")
        assert mgr.active_count == 1

        result = mgr.resolve(eid)
        assert result is True
        assert mgr.active_count == 0

    def test_resolve_nonexistent_returns_false(self):
        mgr = GlobalDegradationManager()
        assert mgr.resolve("nonexistent") is False

    def test_resolve_by_category(self):
        mgr = GlobalDegradationManager()
        mgr.report("rtsp_broken", camera_id="cam_01", camera="cam_01", n=1, max=10, t=0)
        mgr.report("rtsp_broken", camera_id="cam_02", camera="cam_02", n=1, max=10, t=0)
        mgr.report("yolo_down", camera_id="cam_01")

        # Resolve only rtsp_broken for cam_01
        count = mgr.resolve_by_category("rtsp_broken", camera_id="cam_01")
        assert count == 1
        assert mgr.active_count == 2  # cam_02 rtsp + yolo still active

    def test_deduplication_same_category_camera(self):
        """Reporting same category+camera should replace, not duplicate."""
        mgr = GlobalDegradationManager()
        mgr.report("yolo_down", camera_id="cam_01")
        mgr.report("yolo_down", camera_id="cam_01")
        assert mgr.active_count == 1

    def test_different_cameras_not_deduplicated(self):
        mgr = GlobalDegradationManager()
        mgr.report("yolo_down", camera_id="cam_01")
        mgr.report("yolo_down", camera_id="cam_02")
        assert mgr.active_count == 2

    def test_get_active_sorted_by_severity(self):
        mgr = GlobalDegradationManager()
        mgr.report("baseline_drift", camera_id="cam_01", camera="cam_01")  # INFO
        mgr.report("storage_low", free=2)  # SEVERE
        mgr.report("circuit_breaker")  # MODERATE

        active = mgr.get_active()
        assert len(active) == 3
        # Severe should be first
        assert active[0]["level"] == "severe"
        assert active[1]["level"] == "moderate"
        assert active[2]["level"] == "info"

    def test_max_active_level(self):
        mgr = GlobalDegradationManager()
        assert mgr.max_active_level is None

        mgr.report("baseline_drift", camera_id="cam_01", camera="cam_01")
        assert mgr.max_active_level == DegradationLevel.INFO

        mgr.report("storage_low", free=1)
        assert mgr.max_active_level == DegradationLevel.SEVERE

    def test_history_includes_resolved(self):
        mgr = GlobalDegradationManager()
        eid = mgr.report("yolo_down")
        mgr.resolve(eid)

        history = mgr.get_history(days=1)
        assert len(history) == 1
        assert history[0]["resolved_at"] is not None

    def test_unknown_category_creates_generic(self):
        mgr = GlobalDegradationManager()
        eid = mgr.report("unknown_thing")
        assert mgr.active_count == 1
        active = mgr.get_active()
        assert "未知降级" in active[0]["title"]

    def test_all_templates_have_required_keys(self):
        for category, template in DEGRADATION_TEMPLATES.items():
            assert "level" in template, f"{category} missing level"
            assert "title" in template, f"{category} missing title"
            assert "impact" in template, f"{category} missing impact"
            assert "action" in template, f"{category} missing action"
            assert isinstance(template["level"], DegradationLevel)
