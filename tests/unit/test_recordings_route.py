"""Continuous-recording HTTP safety using fake bytes, never a media decoder."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import stat
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest
from starlette.requests import ClientDisconnect

from argus.config.schema import AuthConfig
from argus.dashboard.auth import AuthMiddleware, create_session_token
from argus.dashboard.routes import recordings


_SECRET = "recording-route-test-session-secret"


@pytest.fixture
def context(tmp_path):
    root = tmp_path / "recordings"
    root.mkdir()
    manager = SimpleNamespace(_output_dir=root, get_storage_stats=Mock(return_value={"total_segments": 0}))
    retention = SimpleNamespace(get_retention_stats=Mock(return_value={"retained": 0}),
                                run_cleanup=Mock(return_value={"removed": 0}))
    app = FastAPI()
    app.include_router(recordings.router, prefix="/api/recordings")
    app.state.recording_manager, app.state.retention_manager = manager, retention
    app.add_middleware(AuthMiddleware, config=AuthConfig(enabled=True), session_secret=_SECRET)
    with TestClient(app) as client:
        client.cookies.set("argus_session", create_session_token("recording-operator", "operator", _SECRET))
        yield SimpleNamespace(client=client, app=app, root=root, manager=manager, retention=retention)


def _segment(context, *, camera="cam_01", date="2026-10-28", filename="cam_01_12_00.mp4", data=b"fake-video-bytes"):
    path = context.root / date / camera / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _video(camera="cam_01", filename="cam_01_12_00.mp4"):
    return f"/api/recordings/{camera}/segment/{filename}/video"


@pytest.mark.parametrize("url", ["/api/recordings/cam_01/segments", _video(), "/api/recordings/storage-stats"])
def test_recording_reads_require_replay_permission(context, url):
    context.client.cookies.set("argus_session", create_session_token("recording-viewer", "viewer", _SECRET))

    response = context.client.get(url)

    assert response.status_code == 403
    assert response.json() == {"code": 40300, "msg": "权限不足", "data": None}
    context.manager.get_storage_stats.assert_not_called()
    context.retention.get_retention_stats.assert_not_called()


@pytest.mark.parametrize("url", ["/api/recordings/cam_01/segments", _video(), "/api/recordings/storage-stats"])
def test_recording_reads_require_authentication(context, url):
    context.client.cookies.clear()

    response = context.client.get(url)

    assert response.status_code == 401
    assert response.json()["code"] == 40100


def test_operator_reads_only_exact_camera_and_disambiguated_urls(context):
    _segment(context)
    _segment(context, camera="cam_02", filename="cam_02_12_00.mp4", data=b"other-camera")

    response = context.client.get("/api/recordings/cam_01/segments")

    assert response.status_code == 200
    rows = response.json()["data"]
    assert len(rows) == 1
    assert rows[0]["camera_id"] == "cam_01"
    assert rows[0]["date"] == "2026-10-28"
    assert rows[0]["video_url"] == _video() + "?date=2026-10-28"
    assert context.client.get(rows[0]["video_url"]).content == b"fake-video-bytes"


@pytest.mark.parametrize("camera, filename", [
    ("cam.01", "cam.01_12_00.mp4"),
    ("办公室相机", "办公室相机_12_00.mp4"),
    ("camera one", "camera one 12_00.MP4"),
])
def test_safe_existing_camera_and_filename_components_remain_readable(context, camera, filename):
    _segment(context, camera=camera, filename=filename, data=b"compatible-camera")

    listing = context.client.get(f"/api/recordings/{quote(camera, safe='')}/segments")

    assert listing.status_code == 200
    rows = listing.json()["data"]
    assert len(rows) == 1
    assert rows[0]["camera_id"] == camera
    assert rows[0]["filename"] == filename
    response = context.client.get(rows[0]["video_url"])
    assert response.status_code == 200
    assert response.content == b"compatible-camera"
    encoded = quote(filename, safe="")
    disposition = (f'inline; filename="{filename}"' if encoded == filename
                   else f"inline; filename*=utf-8''{encoded}")
    assert response.headers["content-disposition"] == disposition


def test_video_never_falls_back_to_another_camera_with_the_same_name(context):
    _segment(context, camera="cam_02", data=b"private-other-camera")

    response = context.client.get(_video())

    assert response.status_code == 404
    assert response.json()["code"] == 40400
    assert b"private-other-camera" not in response.content


def test_duplicate_name_requires_exact_recording_date(context):
    _segment(context, date="2026-10-27", data=b"yesterday")
    _segment(context, date="2026-10-28", data=b"today")

    ambiguous = context.client.get(_video())

    assert ambiguous.status_code == 409
    assert ambiguous.json()["code"] == 40900
    assert context.client.get(_video(), params={"date": "2026-10-27"}).content == b"yesterday"
    assert context.client.get(_video(), params={"date": "2026-10-28"}).content == b"today"
    assert context.client.get(_video(), params={"date": "2026-10-26"}).status_code == 404


@pytest.mark.parametrize("date", ["../2026-10-28", "2026-02-30", "2026-1-28", "2026-10-28/../", "*"])
def test_invalid_date_is_a_validation_envelope(context, date):
    _segment(context)

    response = context.client.get(_video(), params={"date": date})

    assert response.status_code == 400
    assert response.json()["code"] == 40000


@pytest.mark.parametrize("camera, filename", [
    ("cam_*", "cam_01_12_00.mp4"), ("cam_[12]", "cam_01_12_00.mp4"),
    ("..\\cam_02", "cam_01_12_00.mp4"), ("cam_01", "..\\secret.mp4"),
    ("cam_01", "*.mp4"), ("cam_01", "cam_01_12_00.mp4:stream"),
    ("cam_01", "CON.mp4"), ("NUL", "cam_01_12_00.mp4"),
])
def test_path_and_glob_identifiers_are_rejected(context, camera, filename):
    _segment(context)

    response = context.client.get(_video(camera, filename))

    assert response.status_code == 404
    assert response.json()["code"] == 40400


@pytest.mark.parametrize("component", [
    ".", "..", "cam/01", "cam\\01", "cam:01", "cam*01", "cam?01", "cam[01]",
    "cam\x00", "cam\r\n", "cam\u202e01", "cam\ud800", "cam.", "cam ", "CON", "LPT1",
])
def test_unsafe_single_components_are_rejected_before_filesystem_access(component):
    assert recordings._safe_component(component, camera=True) is False


@pytest.mark.parametrize("filename", [
    "../clip.mp4", "cam\\clip.mp4", "clip.mp4:stream", "*.mp4", "clip?.mp4", "clip[0].mp4",
    "clip\x00.mp4", "clip\r\n.mp4", "clip\u202e.mp4", "clip\ud800.mp4", "clip.mp4.", "clip.mp4 ",
    "CON.mp4", "LPT1.mp4",
])
def test_unsafe_filenames_are_rejected_before_filesystem_access(filename):
    assert recordings._safe_component(filename) is False


def test_resolved_sibling_with_same_root_prefix_is_not_contained(context, monkeypatch):
    path = _segment(context)
    sibling = context.root.with_name(context.root.name + "-private") / "2026-10-28" / "cam_01" / path.name
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"outside-root")
    original = Path.resolve

    def resolve(candidate, *args, **kwargs):
        return sibling if candidate == path else original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)

    assert context.client.get(_video()).status_code == 404
    assert context.client.get("/api/recordings/cam_01/segments").json()["data"] == []


@pytest.mark.parametrize("linked_part", ["date", "camera", "file"])
def test_descendant_links_are_neither_listed_nor_served(context, monkeypatch, linked_part):
    path = _segment(context)
    linked = {"date": path.parent.parent, "camera": path.parent, "file": path}[linked_part]
    original = recordings._is_link_or_reparse
    monkeypatch.setattr(recordings, "_is_link_or_reparse", lambda candidate: candidate == linked or original(candidate))

    assert context.client.get(_video()).status_code == 404
    assert context.client.get("/api/recordings/cam_01/segments").json()["data"] == []


def test_windows_junction_attribute_is_treated_as_a_link(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "lstat", lambda _: SimpleNamespace(
        st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
    ))

    assert recordings._is_link_or_reparse(tmp_path) is True


@pytest.mark.parametrize("camera", ["cam_01", "cam_02"])
def test_hard_link_aliases_are_not_evidence_of_camera_ownership(context, camera):
    path = _segment(context, data=b"one-camera-only")
    alias = context.root / "2026-10-28" / camera / "alias.mp4"
    alias.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(path, alias)
    except OSError:
        pytest.skip("filesystem does not support hard links")

    assert context.client.get(_video()).status_code == 404
    assert context.client.get(_video(camera, "alias.mp4")).status_code == 404
    assert context.client.get(f"/api/recordings/{camera}/segments").json()["data"] == []


def test_hard_link_created_during_open_is_rejected_and_handle_closed(context, monkeypatch):
    path = _segment(context)
    opened = []
    original_open, original_stat = Path.open, os.fstat

    def tracked_open(candidate, *args, **kwargs):
        handle = original_open(candidate, *args, **kwargs)
        if candidate == path:
            opened.append(handle)
        return handle

    def newly_linked(fd):
        value = original_stat(fd)
        if not opened or opened[0].closed or fd != opened[0].fileno():
            return value
        fields = {name: getattr(value, name) for name in dir(value) if name.startswith("st_")}
        return SimpleNamespace(**{**fields, "st_nlink": 2})

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(os, "fstat", newly_linked)
    assert context.client.get(_video()).status_code == 404
    assert len(opened) == 1 and opened[0].closed


def test_resolved_windows_short_alias_is_not_accepted_as_an_exact_name(context, monkeypatch):
    path = _segment(context, filename="long_camera_recording_name.mp4")
    alias = path.with_name("LONG_C~1.MP4")
    alias.write_bytes(b"synthetic-short-name")
    original = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda candidate, *args, **kwargs:
                        path if candidate == alias else original(candidate, *args, **kwargs))

    assert context.client.get(_video(filename=alias.name)).status_code == 404


@pytest.mark.parametrize("component", ["CONIN$", "CONOUT$", "COM¹", "LPT²", "NUL .mp4"])
def test_windows_reserved_aliases_remain_rejected(component):
    assert recordings._safe_component(component, camera=True) is False


def test_file_identity_change_after_open_is_rejected_and_handle_closed(context, monkeypatch):
    path = _segment(context)
    opened = []
    original_open = Path.open

    def tracked_open(candidate, *args, **kwargs):
        handle = original_open(candidate, *args, **kwargs)
        if candidate == path:
            opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(os.path, "samestat", lambda *_args: False)

    response = context.client.get(_video())

    assert response.status_code == 404
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize("range_value, expected, content_range", [
    ("bytes=2-5", b"2345", "bytes 2-5/10"),
    ("bytes=7-", b"789", "bytes 7-9/10"),
    ("bytes=-3", b"789", "bytes 7-9/10"),
    ("bytes=-100", b"0123456789", "bytes 0-9/10"),
])
def test_video_ranges_read_the_opened_fake_file(context, range_value, expected, content_range):
    _segment(context, data=b"0123456789")

    response = context.client.get(_video(), headers={"Range": range_value})

    assert response.status_code == 206
    assert response.content == expected
    assert response.headers["content-range"] == content_range
    assert response.headers["content-length"] == str(len(expected))
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("range_value", ["bytes=10-", "bytes=-0", "bytes=5-2", "bytes=0-1,3-4", "invalid"])
def test_invalid_range_uses_416_envelope(context, range_value):
    _segment(context, data=b"0123456789")

    response = context.client.get(_video(), headers={"Range": range_value})

    assert response.status_code == 416
    assert response.json()["code"] == 41600
    assert response.headers["content-range"] == "bytes */10"


def test_if_range_mismatch_returns_whole_representation(context):
    _segment(context, data=b"0123456789")

    response = context.client.get(_video(), headers={"Range": "bytes=2-3", "If-Range": '"old"'})

    assert response.status_code == 200
    assert response.content == b"0123456789"
    assert response.headers["etag"].startswith('W/"')
    matched = context.client.get(_video(), headers={"Range": "bytes=2-3", "If-Range": response.headers["etag"]})
    assert matched.status_code == 200
    assert matched.content == b"0123456789"


def test_same_second_date_validator_cannot_splice_changed_range_bytes(context):
    path = _segment(context, data=b"old-file-content")
    original_time_ns = 1_793_145_600_100_000_000
    os.utime(path, ns=(original_time_ns, original_time_ns))
    first = context.client.get(_video())
    path.write_bytes(b"new-file-content")
    changed_time_ns = original_time_ns + 100_000_000
    os.utime(path, ns=(changed_time_ns, changed_time_ns))

    changed = context.client.get(_video(), headers={
        "Range": "bytes=0-2", "If-Range": first.headers["last-modified"],
    })

    assert changed.headers["last-modified"] == first.headers["last-modified"]
    assert changed.status_code == 200
    assert changed.content == b"new-file-content"


def test_empty_file_range_is_rejected_without_losing_the_handle(context, monkeypatch):
    path = _segment(context, data=b"")
    original = Path.open
    opened = []

    def tracked_open(candidate, *args, **kwargs):
        result = original(candidate, *args, **kwargs)
        if candidate == path:
            opened.append(result)
        return result

    monkeypatch.setattr(Path, "open", tracked_open)
    response = context.client.get(_video(), headers={"Range": "bytes=0-"})
    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */0"
    assert len(opened) == 1 and opened[0].closed


def _direct_opened_response(context, monkeypatch):
    path = _segment(context)
    original = Path.open
    opened = []

    def tracked_open(candidate, *args, **kwargs):
        result = original(candidate, *args, **kwargs)
        if candidate == path:
            opened.append(result)
        return result

    monkeypatch.setattr(Path, "open", tracked_open)
    request = Request({"type": "http", "app": context.app, "headers": [],
                       "state": {"user": {"username": "operator", "role": "operator"}}})
    response = recordings.serve_segment_video(request, "cam_01", path.name, date=None)
    assert isinstance(response, recordings._RecordingVideoResponse)
    assert len(opened) == 1 and not opened[0].closed
    return response, opened[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["http.response.start", "http.response.body"])
async def test_send_failure_closes_the_file_even_before_the_generator_enters(context, monkeypatch, phase):
    response, handle = _direct_opened_response(context, monkeypatch)

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == phase:
            raise OSError("synthetic disconnected client")

    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert handle.closed


@pytest.mark.asyncio
async def test_cancellation_before_first_body_byte_closes_the_file(context, monkeypatch):
    response, handle = _direct_opened_response(context, monkeypatch)
    entered = asyncio.Event()

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert handle.closed


def test_response_construction_failure_closes_the_already_opened_handle(context, monkeypatch):
    path = _segment(context)
    original = Path.open
    opened = []

    def tracked_open(candidate, *args, **kwargs):
        result = original(candidate, *args, **kwargs)
        if candidate == path:
            opened.append(result)
        return result

    def fail_response(*args, **kwargs):
        raise RuntimeError("synthetic response construction failure")

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(recordings, "_RecordingVideoResponse", fail_response)
    with pytest.raises(RuntimeError, match="construction failure"):
        context.client.get(_video())
    assert len(opened) == 1 and opened[0].closed


def test_disk_scanning_read_handlers_use_fastapi_worker_threads():
    for handler in (recordings.list_segments, recordings.serve_segment_video, recordings.storage_stats):
        assert not inspect.iscoroutinefunction(handler)


def test_timezone_aware_segment_filters_preserve_the_instant(context):
    path = _segment(context)
    modified = datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(path, (modified, modified))

    response = context.client.get("/api/recordings/cam_01/segments", params={
        "start": "2026-10-28T08:00:00+08:00", "end": "2026-10-28T08:00:00+08:00",
    })

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


@pytest.mark.parametrize("params", [{"start": "bad"}, {"end": "bad"},
                                    {"start": "2026-10-29", "end": "2026-10-28"}])
def test_invalid_segment_filters_return_a_validation_envelope(context, params):
    response = context.client.get("/api/recordings/cam_01/segments", params=params)

    assert response.status_code == 400
    assert response.json()["code"] == 40000


def test_storage_statistics_require_permission_but_cleanup_still_requires_admin(context):
    response = context.client.get("/api/recordings/storage-stats")

    assert response.status_code == 200
    assert response.json()["data"] == {"recording": {"total_segments": 0}, "retention": {"retained": 0}}
    assert context.client.post("/api/recordings/cleanup").status_code == 403
    context.retention.run_cleanup.assert_not_called()
