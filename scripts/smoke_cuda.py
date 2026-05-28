"""GPU/CUDA runtime smoke for Argus deployment machines.

This check is intentionally separate from the normal Dashboard smoke: a
Windows USB workstation may be perfectly valid without CUDA, while an Ubuntu
training/inference box must prove that both PyTorch CUDA and OpenCV CUDA are
usable.  The smoke executes small real GPU operations instead of only checking
``is_available()`` flags.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


class CudaSmokeFailure(RuntimeError):
    """Raised when the CUDA smoke cannot prove a required GPU capability."""


def _tail(text: str, *, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_torch_cuda_smoke(
    *,
    matrix_size: int = 512,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Run a small CUDA matrix multiply through PyTorch."""
    result: dict[str, Any] = {
        "ok": False,
        "checked": True,
        "matrix_size": matrix_size,
    }
    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
    except Exception as exc:
        result["error"] = f"import torch failed: {exc}"
        return result

    result["version"] = getattr(torch, "__version__", None)
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    result["device_count"] = int(torch.cuda.device_count())

    if not result["cuda_available"]:
        result["error"] = "torch.cuda.is_available() is false"
        return result
    if result["device_count"] <= 0:
        result["error"] = "torch reports zero CUDA devices"
        return result

    try:
        result["device"] = torch.cuda.get_device_name(0)
        start = time.perf_counter()
        x = torch.randn((matrix_size, matrix_size), device="cuda")
        y = torch.mm(x, x.T)
        torch.cuda.synchronize()
        result["duration_ms"] = round((time.perf_counter() - start) * 1000, 3)
        result["mean"] = round(float(y.mean().detach().cpu()), 6)
        result["output_device"] = str(getattr(y, "device", "cuda"))
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"torch CUDA op failed: {exc}"
    return result


def run_opencv_cuda_smoke(
    *,
    image_size: int = 256,
    cv2_module: Any | None = None,
    np_module: Any | None = None,
) -> dict[str, Any]:
    """Run CUDA CLAHE and MOG2 through OpenCV using Argus' stream style."""
    result: dict[str, Any] = {
        "ok": False,
        "checked": True,
        "image_size": image_size,
    }
    try:
        cv2 = cv2_module
        if cv2 is None:
            import cv2 as cv2  # type: ignore[no-redef]
        np = np_module
        if np is None:
            import numpy as np  # type: ignore[no-redef]
    except Exception as exc:
        result["error"] = f"import cv2/numpy failed: {exc}"
        return result

    result["version"] = getattr(cv2, "__version__", None)
    cuda = getattr(cv2, "cuda", None)
    cuda_count = int(cuda.getCudaEnabledDeviceCount()) if cuda is not None else 0
    result["cuda_device_count"] = cuda_count
    if cuda_count <= 0:
        result["error"] = "cv2.cuda reports zero CUDA devices"
        return result

    try:
        frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        frame[image_size // 4 : image_size * 3 // 4, image_size * 3 // 8 : image_size * 5 // 8] = 180
        stream = cv2.cuda.Stream()
        gpu = cv2.cuda_GpuMat()
        gpu.upload(frame)

        start = time.perf_counter()
        gray = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)
        clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_out = clahe.apply(gray, stream)
        stream.waitForCompletion()
        clahe_result = clahe_out.download()

        mog = cv2.cuda.createBackgroundSubtractorMOG2()
        mask = mog.apply(gpu, -1, stream)
        stream.waitForCompletion()
        mask_result = mask.download()

        result["duration_ms"] = round((time.perf_counter() - start) * 1000, 3)
        result["clahe_shape"] = list(clahe_result.shape)
        result["clahe_mean"] = round(float(clahe_result.mean()), 3)
        result["mog2_shape"] = list(mask_result.shape)
        result["mog2_nonzero"] = int(np.count_nonzero(mask_result))
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"OpenCV CUDA op failed: {exc}"
    return result


def run_business_training_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Optionally chain the normal Dashboard business smoke."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "smoke_dashboard_business_flow.py"),
        "--training-mode",
        "normal",
        "--training-timeout",
        str(args.business_training_timeout),
        "--browser",
        args.business_browser,
        "--timeout",
        str(args.business_timeout),
        "--recording-timeout",
        str(args.business_recording_timeout),
    ]
    env = os.environ.copy()
    py_path = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
    env["PYTHONPATH"] = py_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=max(args.business_timeout + 30.0, args.business_training_timeout + 60.0),
    )
    return {
        "ok": completed.returncode == 0,
        "command": cmd,
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def run_cuda_smoke(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    result: dict[str, Any] = {
        "ok": False,
        "torch": None,
        "opencv": None,
        "business_smoke": None,
        "errors": errors,
    }

    if not args.skip_torch:
        torch_result = run_torch_cuda_smoke(matrix_size=args.matrix_size)
        result["torch"] = torch_result
        if not torch_result.get("ok") and not args.allow_missing_cuda:
            errors.append(str(torch_result.get("error") or "torch CUDA smoke failed"))

    if not args.skip_opencv:
        opencv_result = run_opencv_cuda_smoke(image_size=args.image_size)
        result["opencv"] = opencv_result
        if not opencv_result.get("ok") and not args.allow_missing_cuda:
            errors.append(str(opencv_result.get("error") or "OpenCV CUDA smoke failed"))

    if args.business_smoke:
        business = run_business_training_smoke(args)
        result["business_smoke"] = business
        if not business.get("ok"):
            errors.append("Dashboard business training smoke failed")

    result["ok"] = not errors
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Argus CUDA/GPU runtime smoke")
    parser.add_argument(
        "--allow-missing-cuda",
        action="store_true",
        help="Report missing CUDA without failing; useful on CPU-only dev workstations.",
    )
    parser.add_argument("--skip-torch", action="store_true", help="Skip PyTorch CUDA check")
    parser.add_argument("--skip-opencv", action="store_true", help="Skip OpenCV CUDA check")
    parser.add_argument("--matrix-size", type=int, default=512, help="PyTorch CUDA matrix size")
    parser.add_argument("--image-size", type=int, default=256, help="OpenCV CUDA synthetic image size")
    parser.add_argument(
        "--business-smoke",
        action="store_true",
        help="Also run smoke_dashboard_business_flow.py with normal training and browser off by default",
    )
    parser.add_argument("--business-timeout", type=float, default=90.0)
    parser.add_argument("--business-recording-timeout", type=float, default=90.0)
    parser.add_argument("--business-training-timeout", type=float, default=420.0)
    parser.add_argument("--business-browser", choices=["off", "auto", "required"], default="off")
    args = parser.parse_args(argv)
    if args.matrix_size <= 0:
        parser.error("--matrix-size must be > 0")
    if args.image_size <= 0:
        parser.error("--image-size must be > 0")
    if args.skip_torch and args.skip_opencv and not args.business_smoke:
        parser.error("nothing to check: all CUDA checks skipped and --business-smoke not set")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_cuda_smoke(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
