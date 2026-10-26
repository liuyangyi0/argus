"""No compute or real media: verify the ordinary-function test safety boundary."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.run_local_functional_tests import FunctionalGuard


@pytest.mark.parametrize("command,kwargs", [
    (["ffmpeg", "-i", "source.mp4", "output.mp4"], {}),
    (["/usr/bin/ffprobe", "source.mp4"], {}),
    (Path("/usr/bin/ffmpeg"), {}),
    ([r"C:\Media Tools\ffmpeg.exe", "-version"], {}),
    ('"C:\\Media Tools\\ffprobe.exe" -version', {}),
    (["ordinary-name", "-i", "source.mp4"], {"executable": Path("/usr/bin/ffmpeg")}),
])
def test_media_process_is_rejected_before_launch(command, kwargs):
    guard = FunctionalGuard()
    launch = Mock()
    with pytest.raises(RuntimeError, match="subprocess media tool"):
        guard.guard_process_launch(launch)(command, **kwargs)
    launch.assert_not_called()
    assert len(guard.violations) == 1


@pytest.mark.parametrize("executable", ["python", Path("python"), b"python"])
def test_ordinary_python_probe_preserves_process_arguments(executable):
    guard = FunctionalGuard()
    launch = Mock()
    args = [executable, "-I", "-c", "print('no model or media')"]
    result = guard.guard_process_launch(launch)(args, cwd="isolated", stdout=-1)
    assert result is launch.return_value
    launch.assert_called_once_with(args, cwd="isolated", stdout=-1)
    assert guard.violations == []


def test_subprocess_run_cannot_bypass_media_process_guard(monkeypatch):
    import subprocess

    guard = FunctionalGuard()
    launch = Mock()
    monkeypatch.setattr(subprocess, "Popen", guard.guard_process_launch(launch))
    with pytest.raises(RuntimeError, match="subprocess media tool"):
        subprocess.run(["ffprobe", "never-open.mp4"], check=True)
    launch.assert_not_called()


def test_scalar_python_path_is_forwarded_without_being_treated_as_argv():
    guard = FunctionalGuard()
    launch = Mock()
    executable = Path("/ordinary tools/python")
    assert guard.guard_process_launch(launch)(executable) is launch.return_value
    launch.assert_called_once_with(executable)
    assert guard.violations == []


def test_media_and_model_entry_points_are_guarded_without_opening_anything():
    """Configure a separate guard and restore every patched API afterwards."""
    import av
    import cv2

    from argus.anomaly.detector import AnomalibDetector, MultiScaleDetector
    from argus.anomaly.trainer import ModelTrainer
    from argus.core.pipeline import DetectionPipeline

    guard = FunctionalGuard()
    patches = [(av, "open"), (cv2, "VideoCapture"), (cv2, "VideoWriter")]
    import subprocess

    patches.append((subprocess, "Popen"))
    for cls, methods in (
        (ModelTrainer, ("train", "export")),
        (AnomalibDetector, ("load", "predict", "predict_batch")),
        (MultiScaleDetector, ("load", "predict")),
        (DetectionPipeline, ("initialize", "process_frame")),
    ):
        patches.extend((cls, method) for method in methods if hasattr(cls, method))
    with pytest.MonkeyPatch.context() as monkeypatch:
        for owner, name in patches:
            monkeypatch.setattr(owner, name, getattr(owner, name))
        guard.pytest_configure(None)
        for entry, args in (
            (av.open, ("never-open.mp4",)),
            (cv2.VideoCapture, (0,)),
            (cv2.VideoWriter, ("never-write.mp4",)),
            (ModelTrainer.train, (None,)),
            (DetectionPipeline.process_frame, (None, None)),
        ):
            with pytest.raises(RuntimeError, match="Local functional profile forbids"):
                entry(*args)
    assert set(guard.violations) == {
        "av.open", "cv2.VideoCapture", "cv2.VideoWriter",
        "ModelTrainer.train", "DetectionPipeline.process_frame",
    }
