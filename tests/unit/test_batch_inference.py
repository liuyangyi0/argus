"""Tests for batch inference API endpoint (Phase 3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestBatchInferenceEndpoint:
    """Test POST /api/models/batch-inference."""

    def _make_request(self, body: dict, camera_manager=None):
        """Create a mock FastAPI Request."""
        request = MagicMock()
        request.json = AsyncMock(return_value=body)
        request.app.state.camera_manager = camera_manager
        return request

    @pytest.mark.asyncio
    async def test_missing_camera_id(self):
        from argus.dashboard.routes.models import batch_inference

        request = self._make_request({"image_paths": ["/tmp/a.jpg"]})
        response = await batch_inference(request)
        assert response.status_code == 400
        assert b"camera_id" in response.body

    @pytest.mark.asyncio
    async def test_empty_image_paths(self):
        from argus.dashboard.routes.models import batch_inference

        request = self._make_request({"camera_id": "cam_01", "image_paths": []})
        response = await batch_inference(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_exceeds_max_batch_size(self):
        from argus.dashboard.routes.models import batch_inference, MAX_BATCH_SIZE

        paths = [f"/tmp/img_{i}.jpg" for i in range(MAX_BATCH_SIZE + 1)]
        request = self._make_request({"camera_id": "cam_01", "image_paths": paths})
        response = await batch_inference(request)
        assert response.status_code == 400
        assert b"Maximum" in response.body

    @pytest.mark.asyncio
    async def test_camera_not_found(self):
        from argus.dashboard.routes.models import batch_inference

        mgr = MagicMock()
        mgr._pipelines = {}
        request = self._make_request(
            {"camera_id": "cam_99", "image_paths": ["/tmp/a.jpg"]},
            camera_manager=mgr,
        )
        response = await batch_inference(request)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_no_detector(self):
        from argus.dashboard.routes.models import batch_inference

        pipeline = MagicMock()
        pipeline._detector = None
        mgr = MagicMock()
        mgr._pipelines = {"cam_01": pipeline}
        request = self._make_request(
            {"camera_id": "cam_01", "image_paths": ["/tmp/a.jpg"]},
            camera_manager=mgr,
        )
        response = await batch_inference(request)
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_successful_scoring(self, tmp_path):
        from argus.dashboard.routes.models import batch_inference
        import json

        # Create a test image file
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\x00" * 100)

        detector = MagicMock()
        detector.threshold = 0.5
        detector.predict.return_value = {"score": 0.7}

        pipeline = MagicMock()
        pipeline._detector = detector

        mgr = MagicMock()
        mgr._pipelines = {"cam_01": pipeline}

        import numpy as np
        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)):
            request = self._make_request(
                {"camera_id": "cam_01", "image_paths": [str(img_file)]},
                camera_manager=mgr,
            )
            response = await batch_inference(request)

        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["total"] == 1
        assert data["scored"] == 1
        assert data["results"][0]["score"] == 0.7
        assert data["results"][0]["is_anomalous"] is True

    @pytest.mark.asyncio
    async def test_file_not_found_in_batch(self):
        from argus.dashboard.routes.models import batch_inference
        import json

        detector = MagicMock()
        detector.threshold = 0.5

        pipeline = MagicMock()
        pipeline._detector = detector

        mgr = MagicMock()
        mgr._pipelines = {"cam_01": pipeline}

        request = self._make_request(
            {"camera_id": "cam_01", "image_paths": ["/nonexistent/img.jpg"]},
            camera_manager=mgr,
        )
        response = await batch_inference(request)

        data = json.loads(response.body)
        assert data["total"] == 1
        assert data["scored"] == 0
        assert "error" in data["results"][0]

    @pytest.mark.asyncio
    async def test_no_camera_manager(self):
        from argus.dashboard.routes.models import batch_inference

        request = self._make_request(
            {"camera_id": "cam_01", "image_paths": ["/tmp/a.jpg"]},
            camera_manager=None,
        )
        response = await batch_inference(request)
        assert response.status_code == 503
