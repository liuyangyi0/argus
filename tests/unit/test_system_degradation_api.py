from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from argus.dashboard.app import create_app


class _Pipeline:
    def __init__(
        self,
        *,
        degraded: bool,
        reason: str | None = None,
        since: float | None = None,
    ) -> None:
        self._degraded = degraded
        self._reason = reason
        self._since = since

    def is_anomaly_degraded(self) -> bool:
        return self._degraded

    def get_anomaly_degradation_reason(self) -> str | None:
        return self._reason

    def get_anomaly_degradation_started_at(self) -> float | None:
        return self._since


def test_anomaly_degradation_endpoint_aggregates_pipeline_fallback(tmp_path):
    alerts_dir = tmp_path / "alerts"
    alerts_dir.mkdir()
    camera_manager = SimpleNamespace(
        _pipelines={
            "cam_01": _Pipeline(degraded=False),
            "cam_02": _Pipeline(
                degraded=True,
                reason="anomaly_head_failures",
                since=1234.5,
            ),
        }
    )
    client = TestClient(
        create_app(camera_manager=camera_manager, alerts_dir=str(alerts_dir))
    )

    response = client.get("/api/system/anomaly-degradation")

    assert response.status_code == 200
    data = response.json()["data"]["anomaly"]
    assert data["degraded"] is True
    assert data["reason"] == "anomaly_head_failures"
    assert data["since"] == 1234.5
    assert data["cameras"] == [
        {
            "camera_id": "cam_01",
            "degraded": False,
            "reason": None,
            "since": None,
        },
        {
            "camera_id": "cam_02",
            "degraded": True,
            "reason": "anomaly_head_failures",
            "since": 1234.5,
        },
    ]
