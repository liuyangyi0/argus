"""Bridge pipeline degradation events into dashboard-wide degradation state."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


def publish_pipeline_degradation(
    ws_manager: object,
    degradation_manager: object | None,
    topic: str,
    payload: dict[str, Any],
) -> None:
    """Broadcast a pipeline degradation event and mirror it globally.

    ``DetectionPipeline`` emits anomaly-head fallback events on the
    ``system_degradation`` topic. The global degradation bar, however, is
    driven by ``GlobalDegradationManager`` on the ``degradation`` topic. This
    bridge keeps both surfaces consistent while preserving the original event.
    """
    if topic == "system_degradation" and payload.get("component") == "anomaly":
        camera_id = str(payload.get("camera_id") or "") or None
        try:
            if payload.get("type") == "entered" and degradation_manager is not None:
                degradation_manager.report(
                    category="model_fallback",
                    camera_id=camera_id,
                )
            elif payload.get("type") in {"recovered", "resolved"} and degradation_manager is not None:
                degradation_manager.resolve_by_category(
                    category="model_fallback",
                    camera_id=camera_id,
                )
        except Exception as exc:
            logger.warning(
                "dashboard.degradation_bridge_failed",
                topic=topic,
                camera_id=camera_id,
                event_type=payload.get("type"),
                error=str(exc),
            )

    broadcast = getattr(ws_manager, "broadcast", None)
    if callable(broadcast):
        broadcast(topic, payload)
