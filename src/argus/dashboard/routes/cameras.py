"""Camera management API routes with live preview (Chinese UI)."""

from __future__ import annotations

import asyncio
import concurrent.futures

import cv2
import structlog
from fastapi import APIRouter, Request

from argus.core.pipeline import PipelineMode
from argus.dashboard.api_contracts import (
    CameraDetailResponse,
    CameraLifecycleResponse,
    CameraListResponse,
    CameraModeRequest,
    CameraModeResponse,
)
from argus.dashboard.api_response import (
    ApiEnvelope,
    api_conflict,
    api_forbidden,
    api_internal_error,
    api_not_found,
    api_success,
    api_typed_success,
    api_unavailable,
    api_validation_error,
)
from argus.dashboard.auth import require_role
from argus.dashboard.forms import htmx_toast_headers, parse_request_form
from argus.streaming.preview_gateway import (
    PreviewGateway,
    probe_source_blocking as _probe_source_blocking,
)

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


def _preview_gateway(request: Request) -> PreviewGateway:
    return PreviewGateway.for_request(
        request,
        stream_executor=_STREAM_EXECUTOR,
        stream_semaphore=_stream_semaphore,
        max_stream_duration=_MAX_STREAM_DURATION,
        source_probe=_probe_source_blocking,
    )


def _active_orchestrator(request: Request):
    """Return the orchestrator only when its stream manager is already running."""
    orchestrator = getattr(request.app.state, "camera_orchestrator", None)
    if orchestrator is None:
        return None
    registry = getattr(orchestrator, "stream_registry", None)
    manager = getattr(registry, "manager", None)
    if manager is not None and getattr(manager, "running", False):
        return orchestrator
    return None


router = APIRouter()


def _ensure_go2rtc_stream(request: Request, cam_config):
    """Register camera with go2rtc without mutating the persisted config."""
    return _preview_gateway(request).ensure_stream_registered(cam_config)


_STAGE_IDS = ["capture", "review", "training", "deploy", "inference"]
_STAGE_NAMES = {"capture": "采集", "review": "基线审查", "training": "训练", "deploy": "发布", "inference": "推理"}


def _find_camera_config(request: Request, camera_id: str):
    """Find the persisted/original camera config for API responses."""
    return _preview_gateway(request).find_camera_config(camera_id)


def _manager_camera_status_payload(
    camera_manager,
    camera_id: str,
    *,
    status=None,
) -> dict | None:
    """Read the manager's canonical camera payload when it supports it."""
    if status is not None:
        serializer = getattr(camera_manager, "serialize_camera_status", None)
        if callable(serializer):
            try:
                payload = serializer(status)
            except Exception:
                logger.debug(
                    "camera.status_serialize_failed",
                    camera_id=camera_id,
                    exc_info=True,
                )
            else:
                if isinstance(payload, dict):
                    return payload

    getter = getattr(camera_manager, "get_camera_status_payload", None)
    if callable(getter):
        try:
            payload = getter(camera_id)
        except Exception:
            logger.debug(
                "camera.status_payload_failed",
                camera_id=camera_id,
                exc_info=True,
            )
        else:
            if isinstance(payload, dict):
                return payload
    return None


def _stream_resolution_for(request: Request, camera_id: str):
    registry = getattr(request.app.state, "stream_registry", None)
    resolutions = getattr(registry, "resolutions", None)
    if isinstance(resolutions, dict):
        return resolutions.get(camera_id)
    return None


def _source_consistency_status(
    *,
    declared_source: str,
    runtime_source: str,
    go2rtc_stream: str | None,
) -> str:
    if declared_source == runtime_source:
        return "direct"
    if go2rtc_stream and runtime_source.startswith("rtsp://"):
        return "go2rtc_redirect"
    return "mismatch"


def _public_scene_guard_status(status: dict | None) -> dict | None:
    """Normalize transitional runtime snapshots to the public safety DTO.

    Older runners and lightweight test doubles may omit diagnostic fields.
    The public response must still expose an unambiguous fail-closed state,
    model/profile binding and recovery progress instead of making clients
    infer them from missing keys.
    """
    if not isinstance(status, dict):
        return None

    usable = status.get("usable") is True
    enabled = status.get("enabled") is True
    state = str(status.get("state") or ("disabled" if not enabled else "blocked"))
    recovery = status.get("recovery_progress")
    if not isinstance(recovery, dict):
        recovery = {}

    def _count(value) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    current = _count(
        status.get("policy_recovery_count", recovery.get("current", 0))
    )
    required = _count(
        status.get("required_recovery_count", recovery.get("required", 0))
    )
    metrics = status.get("metrics")
    reference_available = (
        status.get("reference_available") is True
        if "reference_available" in status
        else isinstance(metrics, dict) and metrics.get("reference_available") is True
    )
    if not isinstance(metrics, dict):
        metrics = {"reference_available": reference_available}
    else:
        metrics = {**metrics}
        metrics.setdefault("reference_available", reference_available)

    return {
        "enabled": enabled,
        "usable": usable,
        "state": state,
        "raw_state": status.get("raw_state") or state,
        "reason": status.get("reason"),
        "profile_id": status.get("profile_id"),
        "bound_model_version": status.get("bound_model_version"),
        "bound_model_artifact_sha256": status.get(
            "bound_model_artifact_sha256"
        ),
        "runtime_model_artifact_sha256": status.get(
            "runtime_model_artifact_sha256"
        ),
        "reference_source": status.get("reference_source"),
        "reference_sha256": status.get("reference_sha256"),
        "reference_error": status.get("reference_error"),
        "runtime_error": status.get("runtime_error"),
        "check_interval_frames": status.get("check_interval_frames"),
        "check_interval_seconds": status.get("check_interval_seconds"),
        "last_checked_frame": status.get("last_checked_frame"),
        "quick_match_score": status.get("quick_match_score"),
        "quick_shift_px": status.get("quick_shift_px"),
        "reference_available": reference_available,
        "policy_recovery_count": current,
        "required_recovery_count": required,
        "recovery_progress": {
            "current": current,
            "required": required,
            "complete": bool(recovery.get("complete", usable)),
        },
        "metrics": metrics,
    }


def _serialize_diagnostic_record(record) -> dict:
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if isinstance(record, dict):
        return record

    def _serialize_stage(stage) -> dict:
        if hasattr(stage, "to_dict"):
            return stage.to_dict()
        if isinstance(stage, dict):
            return stage
        return {
            "stage_name": getattr(stage, "stage_name", ""),
            "duration_ms": getattr(stage, "duration_ms", 0.0),
            "skipped": getattr(stage, "skipped", False),
            "skip_reason": getattr(stage, "skip_reason", ""),
            "metadata": getattr(stage, "metadata", {}),
        }

    def _serialize_candidate(candidate) -> dict:
        if hasattr(candidate, "to_dict"):
            return candidate.to_dict()
        if isinstance(candidate, dict):
            return candidate
        return {
            "candidate_id": getattr(candidate, "candidate_id", ""),
            "candidate_type": getattr(candidate, "candidate_type", ""),
            "decision": getattr(candidate, "decision", ""),
            "stage": getattr(candidate, "stage", ""),
            "evidence": getattr(candidate, "evidence", {}),
            "suppression_reason": getattr(candidate, "suppression_reason", None),
        }

    stages = getattr(record, "stages", [])
    candidates = getattr(record, "candidate_records", [])
    return {
        "frame_number": getattr(record, "frame_number", None),
        "timestamp": getattr(record, "timestamp", None),
        "camera_id": getattr(record, "camera_id", None),
        "total_duration_ms": getattr(record, "total_duration_ms", 0.0),
        "stages": [_serialize_stage(stage) for stage in stages],
        "anomaly_score": getattr(record, "anomaly_score", 0.0),
        "is_anomalous": getattr(record, "is_anomalous", False),
        "alert_emitted": getattr(record, "alert_emitted", False),
        "pipeline_mode": getattr(record, "pipeline_mode", "active"),
        "low_light": getattr(record, "low_light", False),
        "frame_id": getattr(record, "frame_id", ""),
        "alignment_shift_px": list(getattr(record, "alignment_shift_px", (0.0, 0.0))),
        "scene_guard_state": getattr(record, "scene_guard_state", "disabled"),
        "final_decision": getattr(record, "final_decision", "normal"),
        "suppression_reason": getattr(record, "suppression_reason", None),
        "candidate_records": [
            _serialize_candidate(candidate) for candidate in candidates
        ],
    }


def _recent_camera_diagnostics(camera_manager, camera_id: str, limit: int = 20) -> list[dict]:
    if camera_manager is None:
        return []
    getter = getattr(camera_manager, "get_diagnostics", None)
    if not callable(getter):
        return []
    try:
        records = getter(camera_id, limit)
    except Exception:
        logger.debug(
            "camera.diagnostics_failed",
            camera_id=camera_id,
            exc_info=True,
        )
        return []
    if not isinstance(records, (list, tuple)):
        return []
    return [_serialize_diagnostic_record(record) for record in records]


def _manager_anomaly_backend_status(
    camera_manager,
    camera_id: str,
) -> tuple[str | None, str]:
    """Return backend status through the manager's public runtime contract."""
    if camera_manager is None:
        return None, "auto"
    getter = getattr(camera_manager, "get_anomaly_backend_status", None)
    if not callable(getter):
        return None, "auto"
    try:
        status = getter(camera_id)
    except Exception:
        logger.debug(
            "camera.anomaly_backend_status_failed",
            camera_id=camera_id,
            exc_info=True,
        )
        return None, "auto"
    if not isinstance(status, dict):
        return None, "auto"
    actual = status.get("actual")
    expected = status.get("expected")
    return (
        actual if isinstance(actual, str) else None,
        expected if isinstance(expected, str) else "auto",
    )


def _get_region_info(request: Request, region_id: int | None) -> dict:
    """Resolve region display info from the database."""
    if region_id is None:
        return {
            "region_id": None,
            "region_name": None,
            "region_owner": None,
            "region_phone": None,
        }

    db = getattr(request.app.state, "db", None)
    if not db:
        return {
            "region_id": region_id,
            "region_name": None,
            "region_owner": None,
            "region_phone": None,
        }

    region = db.get_region(region_id)
    if region is None:
        return {
            "region_id": region_id,
            "region_name": None,
            "region_owner": None,
            "region_phone": None,
        }

    return {
        "region_id": region.id,
        "region_name": region.name,
        "region_owner": region.owner,
        "region_phone": region.phone,
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
    if not require_role(request, "admin", "engineer"):
        return api_forbidden("权限不足")

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
            if camera_manager.get_camera_config(camera_id) is None:
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
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    camera_manager = request.app.state.camera_manager
    config = request.app.state.config
    if not camera_manager or not config:
        return api_unavailable("不可用")

    # Verify camera exists
    cam_config = next((c for c in config.cameras if c.camera_id == camera_id), None)
    if cam_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    # Stop the camera if it's running
    orchestrator = _active_orchestrator(request)
    try:
        if orchestrator is not None:
            await asyncio.to_thread(orchestrator.stop, camera_id)
        else:
            await asyncio.to_thread(camera_manager.stop_camera, camera_id)
    except Exception:
        logger.debug("camera.stop_on_delete_failed", camera_id=camera_id, exc_info=True)

    # Remove from config list
    config.cameras = [c for c in config.cameras if c.camera_id != camera_id]

    # Remove from camera manager's runtime list
    if hasattr(camera_manager, "remove_camera_config"):
        camera_manager.remove_camera_config(camera_id)

    # Remove go2rtc stream declaration/registration
    stream_registry = getattr(request.app.state, "stream_registry", None)
    stream_removed = False
    if stream_registry is not None and hasattr(stream_registry, "undeclare_camera"):
        try:
            stream_removed = bool(stream_registry.undeclare_camera(cam_config))
        except Exception:
            logger.debug("camera.stream_undeclare_failed", camera_id=camera_id, exc_info=True)
    if not stream_removed:
        go2rtc = getattr(request.app.state, "go2rtc", None)
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
    if not require_role(request, "admin", "engineer"):
        return api_forbidden("权限不足")

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


@router.post(
    "/{camera_id}/mode",
    response_model=ApiEnvelope[CameraModeResponse],
)
async def set_camera_mode(
    request: Request,
    camera_id: str,
    body: CameraModeRequest,
):
    """Switch a running camera pipeline between active/learning/maintenance modes."""
    if not require_role(request, "admin", "operator", "engineer"):
        return api_forbidden("权限不足")

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    mode = PipelineMode(body.mode)

    previous = (
        camera_manager.get_pipeline_mode(camera_id)
        if hasattr(camera_manager, "get_pipeline_mode") else None
    )
    if previous is None:
        return api_not_found(f"摄像头 {camera_id} 未运行")

    try:
        ok = await asyncio.to_thread(camera_manager.set_pipeline_mode, camera_id, mode)
    except RuntimeError:
        return api_unavailable("服务正在关闭")
    if not ok:
        return api_not_found(f"摄像头 {camera_id} 未运行")

    return api_typed_success(
        {
            "camera_id": camera_id,
            "previous_mode": previous,
            "pipeline_mode": mode.value,
        },
        CameraModeResponse,
    )


@router.post(
    "/{camera_id}/start",
    response_model=ApiEnvelope[CameraLifecycleResponse],
)
async def start_camera(request: Request, camera_id: str):
    """Start a stopped camera."""
    if not require_role(request, "admin", "operator", "engineer"):
        return api_forbidden("权限不足")

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    try:
        orchestrator = _active_orchestrator(request)
        if orchestrator is not None:
            success = await asyncio.to_thread(orchestrator.start, camera_id)
        else:
            # Register with go2rtc before the pipeline opens exclusive USB devices.
            cam_config = camera_manager.get_camera_config(camera_id)
            if cam_config:
                _ensure_go2rtc_stream(request, cam_config)
            success = await asyncio.to_thread(camera_manager.start_camera, camera_id)
    except RuntimeError:
        return api_unavailable("服务正在关闭")
    if success:
        payload = _manager_camera_status_payload(camera_manager, camera_id)
        return api_typed_success(
            payload or {"camera_id": camera_id},
            CameraLifecycleResponse,
            headers=htmx_toast_headers("摄像头已启动"),
        )
    return api_internal_error("启动失败")


@router.post(
    "/{camera_id}/stop",
    response_model=ApiEnvelope[CameraLifecycleResponse],
)
async def stop_camera(request: Request, camera_id: str):
    """Stop a running camera."""
    if not require_role(request, "admin", "operator", "engineer"):
        return api_forbidden("权限不足")

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    orchestrator = _active_orchestrator(request)
    if orchestrator is not None:
        await asyncio.to_thread(orchestrator.stop, camera_id)
    else:
        await asyncio.to_thread(camera_manager.stop_camera, camera_id)
    payload = _manager_camera_status_payload(camera_manager, camera_id)
    return api_typed_success(
        payload or {"camera_id": camera_id},
        CameraLifecycleResponse,
        headers=htmx_toast_headers("摄像头已停止"),
    )


@router.post("/{camera_id}/test-connection")
async def test_camera_connection(request: Request, camera_id: str):
    """痛点 8: live-probe an already-configured camera in 5 seconds."""
    if not require_role(request, "admin", "engineer"):
        return api_forbidden("权限不足")
    gateway = _preview_gateway(request)
    cam_config = gateway.find_camera_config(camera_id)
    if cam_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    running_probe = gateway.probe_running_camera_connection(camera_id)
    if running_probe is not None:
        return api_success(running_probe)

    try:
        result = await gateway.probe_source(
            cam_config.source,
            5.0,
            getattr(cam_config, "protocol", None),
        )
    except RuntimeError:
        return api_unavailable("服务正在关闭")
    return api_success(result)


@router.post("/test-connection-draft")
async def test_camera_connection_draft(request: Request):
    """痛点 8: probe arbitrary source/url before the camera is saved.

    Body: { "source": str | int, "protocol"?: str }
    """
    if not require_role(request, "admin", "engineer"):
        return api_forbidden("权限不足")
    try:
        body = await request.json()
    except Exception:
        return api_validation_error("无效的JSON请求")
    source = body.get("source")
    if source is None or source == "":
        return api_validation_error("source 不能为空")
    protocol = body.get("protocol")
    try:
        result = await _preview_gateway(request).probe_source(source, 5.0, protocol)
    except RuntimeError:
        return api_unavailable("服务正在关闭")
    return api_success(result)


@router.get("/usb-devices")
async def usb_devices(request: Request):
    """Probe USB camera indices 0-9 and return available devices."""
    import asyncio

    camera_manager = getattr(request.app.state, "camera_manager", None)
    app_config = getattr(request.app.state, "config", None)
    # Indices already configured or occupied by USB cameras. Running USB
    # cameras are usually owned by go2rtc, so probing them here can steal or
    # disturb the exclusive device handle.
    in_use: set[int] = set()
    configured_cameras = []
    if app_config is not None:
        configured_cameras.extend(getattr(app_config, "cameras", []) or [])
    if camera_manager:
        for cfg in getattr(camera_manager, "_cameras", []) or []:
            if not any(existing.camera_id == cfg.camera_id for existing in configured_cameras):
                configured_cameras.append(cfg)
    for cfg in configured_cameras:
        if getattr(cfg, "protocol", None) == "usb":
            try:
                in_use.add(int(getattr(cfg, "source", "")))
            except (ValueError, TypeError):
                pass
    if camera_manager:
        for cfg in getattr(camera_manager, "_cameras", []) or []:
            if cfg.protocol == "usb" and cfg.camera_id in getattr(camera_manager, "_threads", {}):
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


@router.get("/json", response_model=ApiEnvelope[CameraListResponse])
def cameras_json(request: Request):
    """JSON API for camera status."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_typed_success({"cameras": []}, CameraListResponse)

    health_monitor = getattr(request.app.state, "health_monitor", None)

    def _row(s):
        # 痛点 8: include health + pipeline_mode so the list page can render
        # connectivity badges and (痛点 4) mode badges without a second roundtrip.
        health = (
            health_monitor.get_camera_health(s.camera_id)
            if health_monitor is not None else None
        )
        canonical = _manager_camera_status_payload(
            camera_manager,
            s.camera_id,
            status=s,
        )
        if canonical is not None:
            canonical["health"] = health
            return {
                **canonical,
                **_get_region_info(
                    request,
                    getattr(_find_camera_config(request, s.camera_id), "region_id", None),
                ),
            }

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
            "capture_status": getattr(s, "capture_status", None),
            "stats": {
                "frames_captured": s.stats.frames_captured,
                "frames_analyzed": s.stats.frames_analyzed,
                "anomalies_detected": s.stats.anomalies_detected,
                "alerts_emitted": s.stats.alerts_emitted,
                "avg_latency_ms": round(s.stats.avg_latency_ms, 1),
            } if s.stats else None,
        }

    return api_typed_success(
        {"cameras": [_row(s) for s in camera_manager.get_status()]},
        CameraListResponse,
    )


@router.get(
    "/{camera_id}/detail/json",
    response_model=ApiEnvelope[CameraDetailResponse],
)
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
    scene_guard = None
    get_scene_guard_status = (
        getattr(camera_manager, "get_scene_guard_status", None)
        if camera_manager is not None
        else None
    )
    if callable(get_scene_guard_status):
        try:
            maybe_scene_guard = get_scene_guard_status(camera_id)
            if isinstance(maybe_scene_guard, dict):
                scene_guard = _public_scene_guard_status(maybe_scene_guard)
        except Exception:
            logger.debug(
                "camera.scene_guard_status_failed",
                camera_id=camera_id,
                exc_info=True,
            )
    model_backend, expected_model_backend = _manager_anomaly_backend_status(
        camera_manager,
        camera_id,
    )

    health_monitor = getattr(request.app.state, "health_monitor", None)
    health = health_monitor.get_camera_health(camera_id) if health_monitor is not None else None
    candidate_diagnostics = _recent_camera_diagnostics(
        camera_manager,
        camera_id,
        limit=10,
    )

    identity_status = None
    get_identity_status = (
        getattr(camera_manager, "get_frame_identity_status", None)
        if camera_manager is not None
        else None
    )
    if callable(get_identity_status):
        try:
            maybe_identity_status = get_identity_status(camera_id)
            if isinstance(maybe_identity_status, dict):
                identity_status = maybe_identity_status
        except Exception:
            logger.debug(
                "camera.frame_identity_status_failed",
                camera_id=camera_id,
                exc_info=True,
            )
    stream_resolution = _stream_resolution_for(request, camera_id)
    declared_source = str(
        (identity_status or {}).get("declared_source")
        or getattr(stream_resolution, "original_source", None)
        or camera_config.source
    )
    runtime_source = str(
        (identity_status or {}).get("runtime_source")
        or getattr(stream_resolution, "runtime_source", None)
        or camera_config.source
    )
    go2rtc_stream = (
        (identity_status or {}).get("go2rtc_stream")
        or getattr(stream_resolution, "stream_name", None)
    )

    stats = None
    fps_observed = None
    if status is not None and status.stats is not None:
        fps_observed = getattr(status.stats, "current_fps", None) or None
        stats = {
            "frames_captured": status.stats.frames_captured,
            "frames_analyzed": status.stats.frames_analyzed,
            "anomalies_detected": status.stats.anomalies_detected,
            "alerts_emitted": status.stats.alerts_emitted,
            "avg_latency_ms": round(status.stats.avg_latency_ms, 1),
        }
    capture_status = getattr(status, "capture_status", None) if status is not None else None
    if not fps_observed and capture_status:
        fps_observed = capture_status.get("actual_fps") or capture_status.get("reported_fps")

    dropped_frames = 0
    get_backpressure_stats = (
        getattr(camera_manager, "get_backpressure_stats", None)
        if camera_manager is not None
        else None
    )
    if callable(get_backpressure_stats):
        try:
            backpressure = get_backpressure_stats()
            if isinstance(backpressure, dict):
                dropped_frames = int(
                    backpressure.get(camera_id, {}).get("dropped", 0)
                )
        except Exception:
            dropped_frames = 0

    source_status = _source_consistency_status(
        declared_source=declared_source,
        runtime_source=runtime_source,
        go2rtc_stream=str(go2rtc_stream) if go2rtc_stream else None,
    )
    latest_identity = (identity_status or {}).get("latest")
    source_diagnostics = {
        "camera_id": camera_id,
        "declared_source": declared_source,
        "runtime_source": runtime_source,
        "go2rtc_stream": go2rtc_stream,
        "last_frame_index": (identity_status or {}).get("last_frame_index"),
        "last_capture_ts": (identity_status or {}).get("last_capture_ts"),
        "last_pipeline_ts": (identity_status or {}).get("last_pipeline_ts"),
        "last_snapshot_frame_index": (
            (identity_status or {}).get("last_snapshot_frame_index")
        ),
        "last_snapshot_ts": (identity_status or {}).get("last_snapshot_ts"),
        "last_snapshot_identity": (identity_status or {}).get("last_snapshot"),
        "last_preview_stream_frame_index": (
            (identity_status or {}).get("last_preview_stream_frame_index")
        ),
        "last_preview_stream_ts": (identity_status or {}).get("last_preview_stream_ts"),
        "last_preview_stream_identity": (
            (identity_status or {}).get("last_preview_stream")
        ),
        "fps_observed": fps_observed,
        "dropped_frames": dropped_frames,
        "source_consistency": source_status,
        "latest_frame_identity": latest_identity,
    }

    # Lifecycle stages for pipeline stepper
    stages = _get_lifecycle_stages(request, camera_id, cam_status=status)

    return api_typed_success({
        "camera_id": camera_id,
        "declared_source": declared_source,
        "runtime_source": runtime_source,
        "go2rtc_stream": go2rtc_stream,
        "last_frame_index": source_diagnostics["last_frame_index"],
        "last_capture_ts": source_diagnostics["last_capture_ts"],
        "last_pipeline_ts": source_diagnostics["last_pipeline_ts"],
        "last_snapshot_frame_index": source_diagnostics["last_snapshot_frame_index"],
        "last_snapshot_ts": source_diagnostics["last_snapshot_ts"],
        "last_preview_stream_frame_index": source_diagnostics["last_preview_stream_frame_index"],
        "last_preview_stream_ts": source_diagnostics["last_preview_stream_ts"],
        "fps_observed": fps_observed,
        "dropped_frames": dropped_frames,
        "source_consistency": source_status,
        "source_diagnostics": source_diagnostics,
        "name": camera_config.name,
        **_get_region_info(request, getattr(camera_config, "region_id", None)),
        "connected": status.connected if status is not None else False,
        "running": status.running if status is not None else False,
        "model_backend": model_backend,
        "expected_model_backend": expected_model_backend,
        # Public safety contract.  Keep the nested runtime copy during the
        # frontend migration, but new clients should read this top-level field.
        "scene_guard": scene_guard,
        "stats": stats,
        "stages": stages,
        "config": camera_config.model_dump(mode="json"),
        "runtime": {
            "pipeline_mode": pipeline_mode,
            "anomaly_locked": anomaly_locked,
            "learning_progress": learning,
            "capture_status": capture_status,
            "scene_guard": scene_guard,
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
            "latest_frame_identity": getattr(runner, "latest_frame_identity", None),
        } if runner is not None else None,
        "detector": detector,
        "candidate_diagnostics": candidate_diagnostics,
        "health": health,
    }, CameraDetailResponse)


@router.get("/{camera_id}/diagnostics")
def camera_diagnostics(request: Request, camera_id: str, limit: int = 50):
    """Get recent frame diagnostics with candidate evidence for a camera."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("摄像头管理器未运行")
    camera_config = _find_camera_config(request, camera_id)
    if camera_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")
    limit = max(1, min(int(limit), 200))
    return api_success({
        "camera_id": camera_id,
        "diagnostics": _recent_camera_diagnostics(
            camera_manager,
            camera_id,
            limit=limit,
        ),
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

    candidate_diagnostics = _recent_camera_diagnostics(
        camera_manager,
        camera_id,
        limit=10,
    )

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
        "latest_frame_identity": getattr(snapshot, "latest_frame_identity", None),
        "stats": {
            "frames_captured": snapshot.stats.frames_captured,
            "frames_analyzed": snapshot.stats.frames_analyzed,
            "anomalies_detected": snapshot.stats.anomalies_detected,
            "alerts_emitted": snapshot.stats.alerts_emitted,
            "avg_latency_ms": round(snapshot.stats.avg_latency_ms, 1),
        },
        "candidate_diagnostics": candidate_diagnostics,
    })


@router.get("/{camera_id}/snapshot")
def camera_snapshot(request: Request, camera_id: str):
    """Get the latest frame from a camera as a JPEG image."""
    return _preview_gateway(request).snapshot_response(camera_id)


@router.get("/{camera_id}/stream")
async def camera_stream(request: Request, camera_id: str):
    """MJPEG stream of the latest frames from a camera (~5 FPS)."""
    return _preview_gateway(request).latest_frame_stream_response(request, camera_id)


@router.get("/{camera_id}/heatmap-stream")
async def camera_heatmap_stream(request: Request, camera_id: str):
    """MJPEG stream with anomaly heatmap overlay."""
    return _preview_gateway(request).heatmap_stream_response(request, camera_id)


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

            get_wall_status = getattr(camera_manager, "get_wall_status", None)
            wall_data = get_wall_status(cam_id) if callable(get_wall_status) else None
            if isinstance(wall_data, dict):
                tile["current_score"] = wall_data.get("current_score", 0.0)
                tile["score_sparkline"] = wall_data.get("score_sparkline", [])
                actual_backend, expected_backend = _manager_anomaly_backend_status(
                    camera_manager,
                    cam_id,
                )
                if actual_backend is not None:
                    tile["model_backend"] = actual_backend
                    tile["expected_model_backend"] = expected_backend
                    if (
                        tile["degradation"] is None
                        and expected_backend != "auto"
                        and actual_backend != expected_backend
                    ):
                        tile["degradation"] = "model_backend_mismatch"
                    elif (
                        tile["degradation"] is None
                        and actual_backend in {"ssim-fallback", "none", "disabled"}
                    ):
                        tile["degradation"] = "model_fallback"

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

            capture_status = getattr(cam_status, "capture_status", None)
            if tile["degradation"] is None and capture_status and capture_status.get("degraded"):
                tile["degradation"] = "capture_degraded"
            if capture_status:
                tile["capture_status"] = capture_status

            cameras.append(tile)
        return cameras

    try:
        cameras = await asyncio.to_thread(_build_wall_data)
    except RuntimeError:
        return api_unavailable("服务正在关闭")
    return api_success({"cameras": cameras})
