"""Logging + GPU-environment helpers used by ``argus.__main__``.

Extracted out of ``__main__.py`` so the entrypoint stays focused on
orchestration. Public surface:

- :func:`setup_file_logging` — attach a rotating file handler driven by
  the loaded :class:`LoggingConfig`. Console output is configured by the
  caller via ``structlog.configure``; this only adds the file sink.
- :func:`log_gpu_environment` — emit one structured log entry summarising
  PyTorch CUDA + OpenCV CUDA availability at startup, so an operator can
  tell from the log whether inference is GPU- or CPU-bound.
- :data:`SEVERITY_COLORS` / :data:`RESET` — ANSI colour table for the
  console alert banner.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from argus.config.schema import AlertSeverity

SEVERITY_COLORS = {
    AlertSeverity.INFO: "\033[36m",    # cyan
    AlertSeverity.LOW: "\033[33m",     # yellow
    AlertSeverity.MEDIUM: "\033[91m",  # light red
    AlertSeverity.HIGH: "\033[31;1m",  # bold red
}
RESET = "\033[0m"

logger = structlog.get_logger()


def setup_file_logging(config, log_level: int) -> None:
    """Configure rotating file log output alongside console.

    APScheduler is muted to WARNING because it emits two INFO lines per
    fire ("Running job" + "executed successfully"); on a 30s cadence
    across N jobs that dominates steady-state logs with no information
    value.
    """
    log_cfg = config.logging
    log_dir = Path(log_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(log_dir / "argus.log"),
        maxBytes=log_cfg.max_file_size_mb * 1024 * 1024,
        backupCount=log_cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)

    if log_cfg.log_format == "json":
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
            ],
        )

    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)

    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)


def log_gpu_environment() -> None:
    """Log GPU/CUDA availability at startup for operator visibility."""
    import cv2

    # PyTorch / CUDA
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            logger.info(
                "env.cuda_available",
                device=torch.cuda.get_device_name(0),
                memory_mb=getattr(dev, "total_memory", getattr(dev, "total_mem", 0)) // (1024 * 1024),
                cuda_version=torch.version.cuda,
                torch_version=torch.__version__,
            )
        else:
            logger.warning(
                "env.cuda_unavailable",
                msg="CUDA not available — inference will run on CPU (slow)",
            )
    except ImportError:
        logger.warning(
            "env.torch_missing",
            msg="PyTorch not installed — GPU acceleration disabled",
        )

    # OpenCV CUDA
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
    if cuda_count > 0:
        logger.info("env.opencv_cuda", devices=cuda_count)
    else:
        logger.info(
            "env.opencv_cuda_unavailable",
            msg="OpenCV CUDA not available — cv2 ops will use CPU",
        )
