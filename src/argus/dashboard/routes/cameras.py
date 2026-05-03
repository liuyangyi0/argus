"""Camera management API routes with live preview (Chinese UI)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time

import cv2
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse
from argus.dashboard.api_response import (
    api_conflict,
    api_internal_error,
    api_not_found,
    api_success,
    api_unavailable,
    api_validation_error,
)
from argus.dashboard.forms import htmx_toast_headers, parse_request_form

logger = structlog.get_logger()

# Maximum stream duration in seconds (30 minutes)
_MAX_STREAM_DURATION = 30 * 60

# ── MJPEG streaming isolation ──
# Dedicated thread pool for MJPEG frame encoding (cv2.imencode) so it can
# never starve the default asyncio thread pool used by regular API requests.
_STREAM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=12, thread_name_prefix="mjpeg-enc",
)

# Server-wide cap on concurrent MJPEG streams to prevent resource exhaustion.
_MAX_CONCURRENT_STREAMS = 8
_stream_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_STREAMS)


from typing import Callable


def _mjpeg_response(request: Request, grab_fn: Callable[[], bytes | None]) -> Response:
    """Build a StreamingResponse for an MJPEG stream, or 503 if overloaded.

    ``grab_fn`` is a **synchronous** callable executed in the dedicated MJPEG
    thread pool.  It should return JPEG bytes or ``None`` to skip a frame.
    """
    if _stream_semaphore.locked():
        return Response(status_code=503, content="Too many active streams")

    async def _generate():
        loop = asyncio.get_running_loop()
        async with _stream_semaphore:
            start = time.monotonic()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    if time.monotonic() - start > _MAX_STREAM_DURATION:
                        break
                    jpeg = await loop.run_in_executor(_STREAM_EXECUTOR, grab_fn)
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


router = APIRouter()


def _ensure_go2rtc_stream(request: Request, cam_config) -> None:
    """Register camera with go2rtc and redirect USB sources to RTSP re-stream.

    Delegates to ``Go2RTCManager.register_camera()`` for protocol dispatch.
    Safe to call multiple times — skips if already redirected or go2rtc unavailable.
    """
    go2rtc = getattr(request.app.state, "go2rtc", None)
    if go2rtc is None or not go2rtc.running:
        return

    # Already redirected by a previous call
    if cam_config.protocol == "rtsp" and cam_config.source.startswith("rtsp://127.0.0.1"):
        return

    original_protocol = cam_config.protocol
    rtsp_url = go2rtc.register_camera(cam_config.camera_id, cam_config.source, cam_config.protocol)
    if rtsp_url and original_protocol == "usb":
        cam_config.source = rtsp_url
        cam_config.protocol = "rtsp"


_STAGE_IDS = ["capture", "review", "training", "deploy", "inference"]
_STAGE_NAMES = {"capture": "采集", "review": "基线审查", "training": "训练", "deploy": "发布", "inference": "推理"}


def _find_camera_config(request: Request, camera_id: str):
    """Find camera config from the running manager first, then persisted app config."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is not None:
        config = next((c for c in getattr(camera_manager, "_cameras", []) if c.camera_id == camera_id), None)
        if config is not None:
            return config

    app_config = getattr(request.app.state, "config", None)
    if app_config is not None:
        return next((c for c in getattr(app_config, "cameras", []) if c.camera_id == camera_id), None)
    return None


def _get_region_info(request: Request, region_id: int | None) -> dict:
    """Resolve region display info from the database."""
    if region_id is None:
        return {
            "region_id": None,
            "region_name": None,
            "region_owner": None,
            "region_phone": None,
            "region_email": None,
        }

    db = getattr(request.app.state, "db", None)
    if not db:
        return {
            "region_id": region_id,
            "region_name": None,
            "region_owner": None,
            "region_phone": None,
            "region_email": None,
        }

    region = db.get_region(region_id)
    if region is None:
        return {
            "region_id": region_id,
            "region_name": None,
            "region_owner": None,
            "region_phone": None,
            "region_email": None,
        }

    return {
        "region_id": region.id,
        "region_name": region.name,
        "region_owner": region.owner,
        "region_phone": region.phone,
        "region_email": region.email,
    }


def _get_lifecycle_stages(request: Request, camera_id: str, *, cam_status=None) -> list[dict]:
    """Determine camera's current lifecycle stage for Pipeline Stepper.

    Args:
        cam_status: Pre-fetched CameraStatus to avoid redundant get_status() calls.
    """
    baseline_mgr = getattr(request.app.state, "baseline_manager", None)
    database = getattr(request.app.state, "database", None)
    camera_manager = getattr(request.app.state, "camera_manager", None)

    # Stage 1: Capture
    baseline_count = 0
    if baseline_mgr:
        baseline_count = baseline_mgr.count_images(camera_id)
    has_baselines = baseline_count > 0

    # Stage 2: Baseline verified (auto-pass for now, future: explicit approval)
    baseline_verified = has_baselines

    # Stage 3: Training
    training_done = False
    training_info = ""
    if database:
        latest = database.get_latest_training(camera_id)
        if latest and latest.status == "complete":
            training_done = True
            training_info = f"等级 {latest.quality_grade}" if latest.quality_grade else ""

    # Stage 4: Deployment
    deployed = False
    if camera_manager:
        try:
            det_status = camera_manager.get_detector_status(camera_id)
            if det_status and det_status.get("mode") == "anomalib":
                deployed = True
        except Exception:
            logger.debug("lifecycle.detector_status_failed", camera_id=camera_id, exc_info=True)

    # Stage 5: Inference running (use pre-fetched status to avoid N+1)
    inferring = False
    if cam_status is not None:
        inferring = bool(cam_status.connected and cam_status.stats and cam_status.stats.frames_analyzed > 0)
    elif camera_manager:
        for s in camera_manager.get_status():
            if s.camera_id == camera_id and s.connected and s.stats and s.stats.frames_analyzed > 0:
                inferring = True
                break

    # Determine status per stage
    stages_done = [has_baselines, baseline_verified, training_done, deployed, inferring]
    first_incomplete = next((i for i, done in enumerate(stages_done) if not done), None)

    def _status(idx, done):
        if done:
            return "completed"
        return "active" if first_incomplete == idx else "pending"

    infos = [
        f"{baseline_count} 帧" if baseline_count else "",
        "已通过" if baseline_verified else "",
        training_info,
        "已部署" if deployed else "",
        "运行中" if inferring else "",
    ]

    return [
        {"id": _STAGE_IDS[i], "name": _STAGE_NAMES[_STAGE_IDS[i]],
         "status": _status(i, stages_done[i]), "info": infos[i]}
        for i in range(5)
    ]


@router.post("")
async def add_camera(request: Request):
    """Add a new camera configuration."""
    camera_manager = request.app.state.camera_manager
    config = request.app.state.config
    if not camera_manager or not config:
        return api_unavailable("不可用")

    form = await parse_request_form(request)
    camera_id = form.get("camera_id", "").strip()
    name = form.get("name", "").strip()
    region_id_raw = form.get("region_id", "").strip()
    source = form.get("source", "").strip()
    protocol = form.get("protocol", "rtsp")
    fps_target = int(form.get("fps_target", 5))
    resolution_str = form.get("resolution", "1920,1080")

    region_id = None
    if region_id_raw:
        try:
            region_id = int(region_id_raw)
        except ValueError:
            return api_validation_error("区域 ID 无效")

    if not camera_id or not name or not source:
        return api_validation_error("请填写所有必填字段")

    manager_cameras = getattr(camera_manager, "_cameras", None)

    # Check for duplicate across config and running manager state
    existing_ids = {c.camera_id for c in config.cameras}
    if isinstance(manager_cameras, list):
        existing_ids.update(c.camera_id for c in manager_cameras)

    if camera_id in existing_ids:
        return api_conflict(f"摄像头 {camera_id} 已存在")

    # Parse resolution
    try:
        res_parts = resolution_str.split(",")
        resolution = (int(res_parts[0]), int(res_parts[1]))
    except (ValueError, IndexError):
        resolution = (1920, 1080)

    # Build GigE config if applicable
    from argus.config.schema import CameraConfig, GigEConfig
    gige_kwargs = {}
    if protocol == "gige":
        gige_kwargs["gige"] = GigEConfig(
            exposure=float(form.get("gige_exposure", 0)),
            gain=float(form.get("gige_gain", 0)),
            pixel_format=form.get("gige_pixel_format", "Mono8"),
            capture_script=form.get("gige_capture_script") or None,
        )

    cam_config = CameraConfig(
        camera_id=camera_id,
        name=name,
        region_id=region_id,
        source=source,
        protocol=protocol,
        fps_target=fps_target,
        resolution=resolution,
        **gige_kwargs,
    )

    config.cameras.append(cam_config)

    try:
        # Thread-safe addition to the running manager's camera list
        if camera_manager is not None and hasattr(camera_manager, "add_camera_config"):
            if camera_manager._cameras is not config.cameras:
                camera_manager.add_camera_config(cam_config)

        config_path = getattr(request.app.state, "config_path", None)
        if config_path:
            from argus.config.loader import save_config as _save_config

            _save_config(config, config_path)
    except Exception:
        config.cameras = [camera for camera in config.cameras if camera is not cam_config]
        if camera_manager is not None and hasattr(camera_manager, "remove_camera_config"):
            camera_manager.remove_camera_config(camera_id)
        logger.exception("camera.add_failed", camera_id=camera_id)
        return api_internal_error("摄像头配置保存失败")

    # Note: camera is added to config but not started. User must click "start".
    logger.info("camera.added", camera_id=camera_id, source=source)

    return api_success({"camera_id": camera_id})


@router.delete("/{camera_id}")
async def delete_camera(request: Request, camera_id: str):
    """Delete a camera configuration. Stops the camera first if running."""
    camera_manager = request.app.state.camera_manager
    config = request.app.state.config
    if not camera_manager or not config:
        return api_unavailable("不可用")

    # Verify camera exists
    cam_config = next((c for c in config.cameras if c.camera_id == camera_id), None)
    if cam_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    # Stop the camera if it's running
    try:
        await asyncio.to_thread(camera_manager.stop_camera, camera_id)
    except Exception:
        logger.debug("camera.stop_on_delete_failed", camera_id=camera_id, exc_info=True)

    # Remove from config list
    config.cameras = [c for c in config.cameras if c.camera_id != camera_id]

    # Remove from camera manager's runtime list
    if hasattr(camera_manager, "remove_camera_config"):
        camera_manager.remove_camera_config(camera_id)

    # Remove go2rtc stream registration
    go2rtc = getattr(request.app.state, "go2rtc_manager", None)
    if go2rtc and hasattr(go2rtc, "remove_stream"):
        try:
            go2rtc.remove_stream(camera_id)
        except Exception:
            logger.debug("camera.go2rtc_stream_remove_failed", camera_id=camera_id, exc_info=True)

    # Persist to config file
    config_path = getattr(request.app.state, "config_path", None)
    if config_path:
        try:
            from argus.config.loader import save_config as _save_config
            _save_config(config, config_path)
        except Exception:
            logger.exception("camera.delete_save_failed", camera_id=camera_id)

    logger.info("camera.deleted", camera_id=camera_id)
    return api_success({"camera_id": camera_id, "message": "已删除"})


@router.get("/{camera_id}/config")
async def get_camera_config(request: Request, camera_id: str):
    """Return the full configuration for a single camera (for edit form)."""
    cam_config = _find_camera_config(request, camera_id)
    if cam_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    gige = cam_config.gige
    region_info = _get_region_info(request, getattr(cam_config, "region_id", None))
    return api_success({
        "camera_id": cam_config.camera_id,
        "name": cam_config.name,
        **region_info,
        "source": cam_config.source,
        "protocol": cam_config.protocol,
        "fps_target": cam_config.fps_target,
        "resolution": list(cam_config.resolution),
        "gige_exposure": gige.exposure,
        "gige_gain": gige.gain,
        "gige_pixel_format": gige.pixel_format,
        "gige_capture_script": gige.capture_script or "",
    })


@router.put("/{camera_id}")
async def update_camera(request: Request, camera_id: str):
    """Update an existing camera's configuration."""
    camera_manager = request.app.state.camera_manager
    config = request.app.state.config
    if not camera_manager or not config:
        return api_unavailable("不可用")

    cam_config = next((c for c in config.cameras if c.camera_id == camera_id), None)
    if cam_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    form = await parse_request_form(request)

    # Update fields (only if provided)
    if form.get("name"):
        cam_config.name = form["name"].strip()
    if "region_id" in form:
        region_id_raw = form.get("region_id", "").strip()
        if region_id_raw:
            try:
                cam_config.region_id = int(region_id_raw)
            except ValueError:
                return api_validation_error("区域 ID 无效")
        else:
            cam_config.region_id = None
    if form.get("source"):
        cam_config.source = form["source"].strip()
    if form.get("protocol"):
        cam_config.protocol = form["protocol"]
    if form.get("fps_target"):
        cam_config.fps_target = int(form["fps_target"])
    if form.get("resolution"):
        try:
            parts = form["resolution"].split(",")
            cam_config.resolution = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            pass

    # GigE parameters (protocol already updated above if changed)
    from argus.config.schema import GigEConfig
    if cam_config.protocol == "gige":
        cam_config.gige = GigEConfig(
            exposure=float(form.get("gige_exposure", cam_config.gige.exposure)),
            gain=float(form.get("gige_gain", cam_config.gige.gain)),
            pixel_format=form.get("gige_pixel_format", cam_config.gige.pixel_format),
            capture_script=form.get("gige_capture_script") or cam_config.gige.capture_script,
        )

    # Sync the manager's runtime copy under its lock
    lock = getattr(camera_manager, "_lock", None)
    if lock:
        with lock:
            manager_cam = next(
                (c for c in getattr(camera_manager, "_cameras", []) if c.camera_id == camera_id),
                None,
            )
            if manager_cam is not None and manager_cam is not cam_config:
                manager_cam.name = cam_config.name
                manager_cam.region_id = getattr(cam_config, "region_id", None)
                manager_cam.source = cam_config.source
                manager_cam.protocol = cam_config.protocol
                manager_cam.fps_target = cam_config.fps_target
                manager_cam.resolution = cam_config.resolution
                manager_cam.gige = cam_config.gige

    # Persist
    config_path = getattr(request.app.state, "config_path", None)
    if config_path:
        try:
            from argus.config.loader import save_config as _save_config
            _save_config(config, config_path)
        except Exception:
            logger.exception("camera.update_save_failed", camera_id=camera_id)
            return api_internal_error("保存失败")

    # Check if camera is running — changes take effect after restart
    statuses = camera_manager.get_status() if hasattr(camera_manager, "get_status") else []
    is_running = any(s.camera_id == camera_id and s.running for s in statuses)

    logger.info("camera.updated", camera_id=camera_id)
    return api_success({
        "camera_id": camera_id,
        "needs_restart": is_running,
        "message": "配置已更新" + ("，重启摄像头后生效" if is_running else ""),
    })


@router.post("/{camera_id}/start")
async def start_camera(request: Request, camera_id: str):
    """Start a stopped camera."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    # Register with go2rtc and redirect USB → RTSP before pipeline opens the device
    cam_config = next((c for c in camera_manager._cameras if c.camera_id == camera_id), None)
    if cam_config:
        _ensure_go2rtc_stream(request, cam_config)

    try:
        success = await asyncio.to_thread(camera_manager.start_camera, camera_id)
    except RuntimeError:
        return api_unavailable("服务正在关闭")
    if success:
        return api_success({}, headers=htmx_toast_headers("摄像头已启动"))
    return api_internal_error("启动失败")


@router.post("/{camera_id}/stop")
async def stop_camera(request: Request, camera_id: str):
    """Stop a running camera."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    await asyncio.to_thread(camera_manager.stop_camera, camera_id)
    return api_success({}, headers=htmx_toast_headers("摄像头已停止"))


def _probe_source_blocking(source: str | int, timeout: float = 5.0) -> dict:
    """Open a video source with a hard timeout and grab one frame.

    Returns {ok, latency_ms, resolution, error}. Used by the test-connection
    endpoints (痛点 8: surface camera reachability before saving) without
    spinning up a full pipeline.
    """
    import threading

    result: dict = {"ok": False}
    start = time.monotonic()

    def _worker() -> None:
        cap = None
        try:
            # CAP_DSHOW for USB indices, default backend for URLs/files
            if isinstance(source, int):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                # Convert pure-int strings to int for USB index detection
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
        except Exception as e:  # noqa: BLE001
            result["error"] = f"{type(e).__name__}: {e}"
        finally:
            if cap is not None:
                cap.release()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        # cv2.VideoCapture doesn't honour SIGTERM mid-open on Windows; we
        # report timeout and let the thread clean up in the background.
        result.setdefault("error", "timeout")
    result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
    return result


@router.post("/{camera_id}/test-connection")
async def test_camera_connection(request: Request, camera_id: str):
    """痛点 8: live-probe an already-configured camera in 5 seconds."""
    cam_config = _find_camera_config(request, camera_id)
    if cam_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    try:
        result = await asyncio.to_thread(_probe_source_blocking, cam_config.source, 5.0)
    except RuntimeError:
        return api_unavailable("服务正在关闭")
    return api_success(result)


@router.post("/test-connection-draft")
async def test_camera_connection_draft(request: Request):
    """痛点 8: probe arbitrary source/url before the camera is saved.

    Body: { "source": str | int, "protocol"?: str }
    """
    try:
        body = await request.json()
    except Exception:
        return api_validation_error("无效的JSON请求")
    source = body.get("source")
    if source is None or source == "":
        return api_validation_error("source 不能为空")
    try:
        result = await asyncio.to_thread(_probe_source_blocking, source, 5.0)
    except RuntimeError:
        return api_unavailable("服务正在关闭")
    return api_success(result)


@router.get("/usb-devices")
async def usb_devices(request: Request):
    """Probe USB camera indices 0-9 and return available devices."""
    import asyncio

    camera_manager = getattr(request.app.state, "camera_manager", None)
    # Indices already occupied by running USB cameras
    in_use: set[int] = set()
    if camera_manager:
        for cfg in camera_manager._cameras:
            if cfg.protocol == "usb" and cfg.camera_id in camera_manager._threads:
                try:
                    in_use.add(int(cfg.source))
                except (ValueError, TypeError):
                    pass

    def _probe() -> list[dict]:
        results = []
        for idx in range(10):
            if idx in in_use:
                continue
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            try:
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    results.append({
                        "index": idx,
                        "name": f"USB Camera {idx}",
                        "width": w,
                        "height": h,
                    })
            finally:
                cap.release()
        return results

    devices = await asyncio.to_thread(_probe)
    return api_success({"devices": devices})


@router.get("/json")
def cameras_json(request: Request):
    """JSON API for camera status."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_success({"cameras": []})

    health_monitor = getattr(request.app.state, "health_monitor", None)

    def _row(s):
        # 痛点 8: include health + pipeline_mode so the list page can render
        # connectivity badges and (痛点 4) mode badges without a second roundtrip.
        health = (
            health_monitor.get_camera_health(s.camera_id)
            if health_monitor is not None else None
        )
        pipeline_mode = camera_manager.get_pipeline_mode(s.camera_id)
        if not isinstance(pipeline_mode, str):
            pipeline_mode = None
        return {
            "camera_id": s.camera_id,
            "name": s.name,
            **_get_region_info(request, getattr(_find_camera_config(request, s.camera_id), "region_id", None)),
            "connected": s.connected,
            "running": s.running,
            "pipeline_mode": pipeline_mode,
            "health": health,
            "stats": {
                "frames_captured": s.stats.frames_captured,
                "frames_analyzed": s.stats.frames_analyzed,
                "anomalies_detected": s.stats.anomalies_detected,
                "alerts_emitted": s.stats.alerts_emitted,
                "avg_latency_ms": round(s.stats.avg_latency_ms, 1),
            } if s.stats else None,
        }

    return api_success({"cameras": [_row(s) for s in camera_manager.get_status()]})


@router.get("/{camera_id}/detail/json")
def camera_detail_json(request: Request, camera_id: str):
    """Return a detailed camera payload for the detail page."""
    camera_manager = request.app.state.camera_manager
    camera_config = _find_camera_config(request, camera_id)
    if camera_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    status = None
    if camera_manager is not None:
        status = next((item for item in camera_manager.get_status() if item.camera_id == camera_id), None)

    runner = camera_manager.get_runner_snapshot(camera_id) if camera_manager else None
    detector = camera_manager.get_detector_status(camera_id) if camera_manager else None
    learning = camera_manager.get_learning_progress(camera_id) if camera_manager else None
    pipeline_mode = camera_manager.get_pipeline_mode(camera_id) if camera_manager else None
    anomaly_locked = camera_manager.is_anomaly_locked(camera_id) if camera_manager else False

    health_monitor = getattr(request.app.state, "health_monitor", None)
    health = health_monitor.get_camera_health(camera_id) if health_monitor is not None else None

    stats = None
    if status is not None and status.stats is not None:
        stats = {
            "frames_captured": status.stats.frames_captured,
            "frames_analyzed": status.stats.frames_analyzed,
            "anomalies_detected": status.stats.anomalies_detected,
            "alerts_emitted": status.stats.alerts_emitted,
            "avg_latency_ms": round(status.stats.avg_latency_ms, 1),
        }

    # Lifecycle stages for pipeline stepper
    stages = _get_lifecycle_stages(request, camera_id, cam_status=status)

    return api_success({
        "camera_id": camera_id,
        "name": camera_config.name,
        **_get_region_info(request, getattr(camera_config, "region_id", None)),
        "connected": status.connected if status is not None else False,
        "running": status.running if status is not None else False,
        "stats": stats,
        "stages": stages,
        "config": camera_config.model_dump(mode="json"),
        "runtime": {
            "pipeline_mode": pipeline_mode,
            "anomaly_locked": anomaly_locked,
            "learning_progress": learning,
        },
        "runner": {
            "model_ref": runner.model_ref,
            "health_status": runner.health_status,
            "cusum_state": runner.cusum_state,
            "lock_state": runner.lock_state.value,
            "last_heartbeat": runner.last_heartbeat,
            "version_tag": runner.version_tag,
            "degradation_state": runner.degradation_state.value,
            "consecutive_failures": runner.consecutive_failures,
        } if runner is not None else None,
        "detector": detector,
        "health": health,
    })


@router.get("/{camera_id}/runner")
def camera_runner_snapshot(request: Request, camera_id: str):
    """Get the inference runner state snapshot for a camera (5.1)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("摄像头管理器未运行")

    snapshot = camera_manager.get_runner_snapshot(camera_id)
    if snapshot is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    return api_success({
        "camera_id": snapshot.camera_id,
        "model_ref": snapshot.model_ref,
        "health_status": snapshot.health_status,
        "cusum_state": snapshot.cusum_state,
        "lock_state": snapshot.lock_state.value,
        "last_heartbeat": snapshot.last_heartbeat,
        "version_tag": snapshot.version_tag,
        "degradation_state": snapshot.degradation_state.value,
        "consecutive_failures": snapshot.consecutive_failures,
        "stats": {
            "frames_captured": snapshot.stats.frames_captured,
            "frames_analyzed": snapshot.stats.frames_analyzed,
            "anomalies_detected": snapshot.stats.anomalies_detected,
            "alerts_emitted": snapshot.stats.alerts_emitted,
            "avg_latency_ms": round(snapshot.stats.avg_latency_ms, 1),
        },
    })


@router.get("/{camera_id}/snapshot")
def camera_snapshot(request: Request, camera_id: str):
    """Get the latest frame from a camera as a JPEG image."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return Response(status_code=503)

    frame = camera_manager.get_latest_frame(camera_id)
    if frame is None:
        return Response(status_code=404)

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store"},
    )


@router.get("/{camera_id}/stream")
async def camera_stream(request: Request, camera_id: str):
    """MJPEG stream of the latest frames from a camera (~5 FPS)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return Response(status_code=503)

    def _grab():
        frame = camera_manager.get_latest_frame(camera_id)
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes()

    return _mjpeg_response(request, _grab)


@router.get("/{camera_id}/heatmap-stream")
async def camera_heatmap_stream(request: Request, camera_id: str):
    """MJPEG stream with anomaly heatmap overlay."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return Response(status_code=503)

    import numpy as np

    def _grab():
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
            blended = frame.copy()
            if mask.any():
                mask_3ch = np.stack([mask] * 3, axis=-1)
                blended = np.where(
                    mask_3ch,
                    cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0),
                    frame,
                )
            frame = blended
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes()

    return _mjpeg_response(request, _grab)


# ── Video Wall API (UX v2 §2) ──

@router.get("/wall/status")
async def wall_status(request: Request):
    """Return aggregated status for all cameras in the video wall.

    Response: {cameras: [{camera_id, name, status, model_version,
               current_score, score_sparkline, alert_count_today,
               active_alert, degradation}]}
    """
    camera_manager = request.app.state.camera_manager
    if camera_manager is None:
        return api_success({"cameras": []})

    db = getattr(request.app.state, "database", None)
    health_monitor = getattr(request.app.state, "health_monitor", None)

    def _build_wall_data():
        from datetime import datetime, timezone

        statuses = list(camera_manager.get_status())
        camera_ids = [cs.camera_id for cs in statuses]

        # P0 perf: one batched stats call + one batched DB call replaces the
        # previous 2*N + 1 per-camera queries. Before: 5 Hz poll * 4 clients *
        # (2N+1) = ~200 qps against SQLite contending with AlertDispatcher
        # INSERTs. After: ~2-3 queries per request regardless of N.
        bp_stats = camera_manager.get_backpressure_stats()

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        wall_batch: dict = {}
        if db is not None and camera_ids:
            try:
                wall_batch = db.get_wall_status_batch(camera_ids, since=today_start)
            except Exception:
                # WARNING not DEBUG: a silent failure here means the video
                # wall shows "0 alerts today" forever, which operators will
                # misread as a healthy system.
                logger.warning("cameras.wall_status_batch_failed", exc_info=True)
                wall_batch = {}

        cameras = []
        for cam_status in statuses:
            cam_id = cam_status.camera_id
            # Determine connection status: a camera is "online" if its
            # pipeline is running, even if not yet fully connected (USB
            # cameras may take a moment to initialize).
            is_online = getattr(cam_status, "connected", False) or getattr(cam_status, "running", False)
            tile: dict = {
                "camera_id": cam_id,
                "name": getattr(cam_status, "name", cam_id),
                "status": "online" if is_online else "offline",
                "model_version": getattr(cam_status, "model_version_id", None),
                "fps": None,
                "current_score": 0.0,
                "score_sparkline": [],
                "alert_count_today": 0,
                "active_alert": None,
                "degradation": None,
            }

            pipeline = camera_manager.get_pipeline(cam_id)
            if pipeline is not None and hasattr(pipeline, "get_wall_status"):
                wall_data = pipeline.get_wall_status()
                tile["current_score"] = wall_data.get("current_score", 0.0)
                tile["score_sparkline"] = wall_data.get("score_sparkline", [])

            # FPS from pipeline stats (computed every 2s in process_frame)
            stats = getattr(cam_status, "stats", None)
            if stats is not None and getattr(stats, "current_fps", 0) > 0:
                tile["fps"] = stats.current_fps

            # Backpressure visibility (P0-2) — snapshot fetched once above.
            bp = bp_stats.get(cam_id, {})
            tile["frames_dropped"] = bp.get("dropped", 0)
            tile["backpressured"] = bp.get("backpressured", False)

            # Alert aggregates from the single batched query.
            batch_row = wall_batch.get(cam_id)
            if batch_row is not None:
                tile["alert_count_today"] = batch_row.get("count", 0)
                active = batch_row.get("active")
                if active is not None:
                    tile["active_alert"] = active

            if health_monitor is not None:
                health = health_monitor.get_camera_health(cam_id)
                if health and not health.get("connected", True):
                    tile["degradation"] = "rtsp_broken"
                elif health and health.get("error"):
                    tile["degradation"] = "error"

            cameras.append(tile)
        return cameras

    try:
        cameras = await asyncio.to_thread(_build_wall_data)
    except RuntimeError:
        return api_unavailable("服务正在关闭")
    return api_success({"cameras": cameras})
