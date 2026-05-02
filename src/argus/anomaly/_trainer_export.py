"""Export and quantization helpers split out of ``ModelTrainer``.

These functions used to live as ``ModelTrainer`` static methods inside
``trainer.py``. They were moved here so the 1.8k-line trainer module is no
longer the only place that knows about the OpenVINO/ONNX export contract,
and so the OpenVINO static-shape regression guard (added in c55d7f5) sits
next to the export call that emits the IR.

Backward compatibility: ``trainer.py`` re-mounts these as static methods on
``ModelTrainer``, so existing call sites and tests
(``ModelTrainer._assert_openvino_static``, ``ModelTrainer._export_model``,
``ModelTrainer._quantize_int8``) keep working unchanged.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


# ── Calibration dataset (private to int8 quantization) ──────────────────────


def _list_images_for_calibration(directory: Path) -> list[Path]:
    """Sorted .png + .jpg files in ``directory``.

    Local copy (rather than reusing ``trainer._list_images``) so this module
    has no import dependency on ``trainer.py`` and the two cannot deadlock at
    import time.
    """
    return sorted(list(directory.glob("*.png")) + list(directory.glob("*.jpg")))


def build_calibration_dataset(
    ov_model,
    val_dir: Path | None,
    max_images: int = 100,
) -> list[np.ndarray]:
    """Build a calibration dataset for INT8 quantization from validation images.

    Returns a list of preprocessed numpy arrays matching the model's input shape.
    Each element is a single-batch NCHW float32 tensor ready for NNCF.
    """
    if val_dir is None or not val_dir.exists():
        return []

    val_normal = val_dir / "normal" if (val_dir / "normal").exists() else val_dir
    images = _list_images_for_calibration(val_normal)
    if not images:
        return []

    input_layer = ov_model.input(0)
    input_shape = input_layer.shape  # e.g. [1, 3, 256, 256]

    # Handle dynamic shapes — fall back to 256x256
    try:
        _, channels, height, width = [
            int(d) if not isinstance(d, int) and hasattr(d, "get_length") and d.get_length() != -1
            else int(d) if isinstance(d, int)
            else 256
            for d in input_shape
        ]
    except (ValueError, TypeError):
        channels, height, width = 3, 256, 256

    calibration_data: list[np.ndarray] = []
    for img_path in images[:max_images]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        resized = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # HWC -> NCHW float32 in [0, 1]
        blob = rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        calibration_data.append(blob)

    logger.info(
        "training.calibration_dataset_built",
        images=len(calibration_data),
        input_shape=f"{1}x{channels}x{height}x{width}",
    )
    return calibration_data


# ── Public export-side surface ──────────────────────────────────────────────


def assert_openvino_static(export_path: Path) -> None:
    """Raise if the exported OpenVINO IR has any dynamic input dim.

    Defensive guard against a regression of anomalib's default export config.
    Without this, dynamic-axes IR would crash CPU inference at the first
    ``predict()`` with ``Broadcast Check failed, Value -1 not in range`` —
    far away from the trainer that produced the broken IR.
    """
    try:
        import openvino as ov
    except ImportError:
        return
    xml_files = sorted(export_path.rglob("*.xml"))
    if not xml_files:
        return
    xml = xml_files[0]
    ov_model = ov.Core().read_model(str(xml))
    for input_tensor in ov_model.inputs:
        shape = input_tensor.partial_shape
        if not shape.is_static:
            name = input_tensor.any_name
            raise RuntimeError(
                f"Exported OpenVINO IR {xml} has non-static input shape "
                f"{shape} on '{name}'. CPU inference will crash with "
                "'Broadcast Check failed, Value -1 not in range'. "
                "Ensure engine.export() receives "
                "onnx_kwargs={'dynamo': False, 'dynamic_axes': {}} and "
                "input_size=(H, W)."
            )


def quantize_int8(
    export_path: Path,
    val_dir: Path | None = None,
    calibration_images: int = 100,
) -> bool:
    """Apply INT8 post-training quantization to an exported OpenVINO model.

    Uses NNCF's quantize() with calibration data from validation images.
    Falls back gracefully if nncf or openvino is not installed.

    Returns:
        True if quantization succeeded, False otherwise (logs the reason).
    """
    try:
        import nncf
    except ImportError:
        logger.warning(
            "training.int8_skipped",
            reason="nncf not installed — run: pip install argus[quantize]",
        )
        return False

    try:
        import openvino as ov
    except ImportError:
        logger.warning(
            "training.int8_skipped",
            reason="openvino not installed",
        )
        return False

    xml_files = sorted(export_path.rglob("*.xml"))
    if not xml_files:
        logger.warning("training.int8_skipped", reason="No .xml model found in export path")
        return False

    model_xml = xml_files[0]
    logger.info("training.int8_quantizing", model=str(model_xml))

    try:
        core = ov.Core()
        ov_model = core.read_model(model_xml)

        cal_images = build_calibration_dataset(
            ov_model=ov_model,
            val_dir=val_dir,
            max_images=calibration_images,
        )

        if not cal_images:
            logger.warning(
                "training.int8_skipped",
                reason="No calibration images available",
            )
            return False

        quantized_model = nncf.quantize(
            ov_model,
            nncf.Dataset(cal_images),
            subset_size=min(len(cal_images), calibration_images),
            model_type=nncf.ModelType.TRANSFORMER,
            preset=nncf.QuantizationPreset.MIXED,
        )

        # Backup FP model before overwriting with INT8 version
        model_bin = model_xml.with_suffix(".bin")
        fp_backup_xml = model_xml.with_name(model_xml.stem + "_fp32.xml")
        fp_backup_bin = model_xml.with_name(model_xml.stem + "_fp32.bin")
        try:
            shutil.copy2(str(model_xml), str(fp_backup_xml))
            if model_bin.exists():
                shutil.copy2(str(model_bin), str(fp_backup_bin))
            logger.info(
                "training.int8_fp_backup",
                xml=str(fp_backup_xml),
                bin=str(fp_backup_bin),
            )
        except OSError as backup_err:
            logger.warning("training.int8_backup_failed", error=str(backup_err))

        ov.save_model(quantized_model, str(model_xml))

        logger.info(
            "training.int8_complete",
            model=str(model_xml),
            original_size_mb=round(model_bin.stat().st_size / 1024 / 1024, 1)
            if model_bin.exists()
            else None,
        )
        return True

    except Exception as e:
        logger.error("training.int8_failed", error=str(e))
        logger.warning(
            "training.int8_fallback",
            msg="Keeping FP model — INT8 quantization failed",
        )
        return False


def export_model(
    engine,
    model,
    export_format: str,
    export_path: str,
    image_size: int = 256,
    quantization: str = "fp16",
    val_dir: Path | None = None,
    calibration_images: int = 100,
) -> dict[str, str]:
    """Export trained model to an optimized inference format.

    Pinned export config for OpenVINO/ONNX:

    1. ``dynamo=False``: PyTorch 2.11 + anomalib's dynamo exporter trips over
       dynamic_axes, so use the classic torchscript exporter.
    2. ``dynamic_axes={}``: Emit a fully static ONNX graph. Without this
       anomalib defaults to ``{"input": {0: "batch"}}`` which produces an
       OpenVINO IR with dynamic batch ``?``. Dinomaly2 IR then explodes at
       CPU inference with ``Broadcast Check failed, Value -1 not in range``.
    3. ``input_size=(H,W)``: Matches the single resolution the detector ever
       feeds at inference.

    Falls back to torch export if the requested format fails (except torch
    itself, which propagates the error).

    If quantization == "int8" and export_format == "openvino", runs NNCF
    post-training quantization on the exported FP model using validation
    images as calibration data.

    Returns:
        dict with ``actual_format`` and ``actual_quantization`` reflecting
        what was actually produced (may differ from request on fallback).
    """
    actual_format = export_format
    export_kwargs: dict = {}
    if export_format in {"openvino", "onnx"}:
        export_kwargs["onnx_kwargs"] = {
            "dynamo": False,
            "dynamic_axes": {},
        }
        export_kwargs["input_size"] = (image_size, image_size)
    try:
        engine.export(
            model=model,
            export_type=export_format,
            export_root=export_path,
            **export_kwargs,
        )
    except Exception as e:
        if export_format != "torch":
            logger.warning(
                "training.export_fallback",
                original_format=export_format,
                error=str(e),
                msg="Falling back to torch export",
            )
            actual_format = "torch"
            engine.export(
                model=model,
                export_type="torch",
                export_root=export_path,
            )
        else:
            raise
    logger.info("training.exported", format=actual_format, path=export_path)

    if actual_format == "openvino":
        assert_openvino_static(Path(export_path))

    actual_quantization = quantization
    if quantization == "int8" and actual_format == "openvino":
        int8_ok = quantize_int8(
            export_path=Path(export_path),
            val_dir=val_dir,
            calibration_images=calibration_images,
        )
        if not int8_ok:
            actual_quantization = "fp32"
            logger.warning(
                "training.quantization_downgraded",
                requested="int8",
                actual="fp32",
            )
    elif quantization == "int8" and actual_format != "openvino":
        # INT8 only supported with OpenVINO; record actual state
        actual_quantization = "fp32"

    return {
        "actual_format": actual_format,
        "actual_quantization": actual_quantization,
    }
