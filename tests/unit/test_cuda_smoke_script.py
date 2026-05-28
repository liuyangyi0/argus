from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.smoke_cuda import (
    parse_args,
    run_cuda_smoke,
    run_opencv_cuda_smoke,
    run_torch_cuda_smoke,
)


def test_parse_args_accepts_business_smoke_options() -> None:
    args = parse_args([
        "--business-smoke",
        "--business-browser",
        "required",
        "--matrix-size",
        "128",
        "--image-size",
        "96",
    ])

    assert args.business_smoke is True
    assert args.business_browser == "required"
    assert args.matrix_size == 128
    assert args.image_size == 96


@pytest.mark.parametrize("argv", [["--matrix-size", "0"], ["--image-size", "0"], ["--skip-torch", "--skip-opencv"]])
def test_parse_args_rejects_invalid_values(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_missing_cuda_fails_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.smoke_cuda.run_torch_cuda_smoke",
        lambda matrix_size: {"ok": False, "error": "torch missing"},
    )
    monkeypatch.setattr(
        "scripts.smoke_cuda.run_opencv_cuda_smoke",
        lambda image_size: {"ok": False, "error": "opencv missing"},
    )

    result = run_cuda_smoke(parse_args([]))

    assert result["ok"] is False
    assert result["errors"] == ["torch missing", "opencv missing"]


def test_allow_missing_cuda_reports_without_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.smoke_cuda.run_torch_cuda_smoke",
        lambda matrix_size: {"ok": False, "error": "torch missing"},
    )
    monkeypatch.setattr(
        "scripts.smoke_cuda.run_opencv_cuda_smoke",
        lambda image_size: {"ok": False, "error": "opencv missing"},
    )

    result = run_cuda_smoke(parse_args(["--allow-missing-cuda"]))

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["torch"]["error"] == "torch missing"
    assert result["opencv"]["error"] == "opencv missing"


class _FakeMean:
    def detach(self) -> "_FakeMean":
        return self

    def cpu(self) -> float:
        return 1.25

    def __float__(self) -> float:
        return 1.25


class _FakeTensor:
    device = "cuda:0"

    @property
    def T(self) -> "_FakeTensor":
        return self

    def mean(self) -> _FakeMean:
        return _FakeMean()


class _FakeTorch:
    __version__ = "2.fake"
    version = SimpleNamespace(cuda="13.0")

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(_idx: int) -> str:
            return "Fake GPU"

        @staticmethod
        def synchronize() -> None:
            return None

    @staticmethod
    def randn(_shape, *, device: str) -> _FakeTensor:
        assert device == "cuda"
        return _FakeTensor()

    @staticmethod
    def mm(_left: _FakeTensor, _right: _FakeTensor) -> _FakeTensor:
        return _FakeTensor()


def test_torch_cuda_smoke_executes_tensor_op() -> None:
    result = run_torch_cuda_smoke(matrix_size=32, torch_module=_FakeTorch)

    assert result["ok"] is True
    assert result["device"] == "Fake GPU"
    assert result["mean"] == 1.25
    assert result["output_device"] == "cuda:0"


class _FakeGpuMat:
    def __init__(self, data=None) -> None:
        self.data = data

    def upload(self, frame) -> None:
        self.data = frame

    def download(self):
        return self.data


class _FakeStream:
    def waitForCompletion(self) -> None:
        return None


class _FakeClahe:
    def apply(self, gray: _FakeGpuMat, _stream: _FakeStream) -> _FakeGpuMat:
        return gray


class _FakeMog:
    def apply(self, gpu: _FakeGpuMat, _lr: float, _stream: _FakeStream) -> _FakeGpuMat:
        return _FakeGpuMat(np.ones(gpu.data.shape[:2], dtype=np.uint8) * 255)


class _FakeCuda:
    @staticmethod
    def getCudaEnabledDeviceCount() -> int:
        return 1

    @staticmethod
    def Stream() -> _FakeStream:
        return _FakeStream()

    @staticmethod
    def cvtColor(gpu: _FakeGpuMat, _code: int) -> _FakeGpuMat:
        return _FakeGpuMat(gpu.data[:, :, 0])

    @staticmethod
    def createCLAHE(*, clipLimit: float, tileGridSize: tuple[int, int]) -> _FakeClahe:
        assert clipLimit == 2.0
        assert tileGridSize == (8, 8)
        return _FakeClahe()

    @staticmethod
    def createBackgroundSubtractorMOG2() -> _FakeMog:
        return _FakeMog()


class _FakeCv2:
    __version__ = "4.fake"
    cuda = _FakeCuda
    COLOR_BGR2GRAY = 6

    @staticmethod
    def cuda_GpuMat() -> _FakeGpuMat:
        return _FakeGpuMat()


def test_opencv_cuda_smoke_executes_cuda_ops() -> None:
    result = run_opencv_cuda_smoke(image_size=32, cv2_module=_FakeCv2, np_module=np)

    assert result["ok"] is True
    assert result["cuda_device_count"] == 1
    assert result["clahe_shape"] == [32, 32]
    assert result["mog2_shape"] == [32, 32]
    assert result["mog2_nonzero"] == 1024


def test_business_smoke_failure_marks_overall_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.smoke_cuda.run_torch_cuda_smoke", lambda matrix_size: {"ok": True})
    monkeypatch.setattr("scripts.smoke_cuda.run_opencv_cuda_smoke", lambda image_size: {"ok": True})
    monkeypatch.setattr(
        "scripts.smoke_cuda.run_business_training_smoke",
        lambda args: {"ok": False, "returncode": 1},
    )

    result = run_cuda_smoke(parse_args(["--business-smoke"]))

    assert result["ok"] is False
    assert result["errors"] == ["Dashboard business training smoke failed"]
