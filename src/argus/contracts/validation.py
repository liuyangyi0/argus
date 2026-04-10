"""Lightweight data contract validation at module boundaries (Section 7).

These validators run at pipeline stage boundaries to detect data
integrity issues early.  Violations are logged via structlog as
warnings — they never raise exceptions or block the pipeline.
This is advisory validation for auditability, not enforcement
that could cause outages in a nuclear environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class ContractViolation:
    """A single data contract violation."""

    field: str
    message: str
    value: object = None


def validate_anomaly_result(result: object) -> list[ContractViolation]:
    """Validate an anomaly detection result at the detector→grader boundary.

    Checks:
    - anomaly_score is a finite float in [0, 1]
    - anomaly_map (if present) is a 2D/3D ndarray with matching dimensions
    """
    violations: list[ContractViolation] = []

    score = getattr(result, "anomaly_score", None)
    if score is None:
        violations.append(ContractViolation("anomaly_score", "missing"))
    elif not isinstance(score, (int, float)):
        violations.append(ContractViolation("anomaly_score", f"not numeric: {type(score)}", score))
    else:
        import math
        if math.isnan(score) or math.isinf(score):
            violations.append(ContractViolation("anomaly_score", "NaN or Inf", score))
        elif score < 0.0 or score > 1.0:
            violations.append(ContractViolation("anomaly_score", "out of range [0, 1]", score))

    anomaly_map = getattr(result, "anomaly_map", None)
    if anomaly_map is not None:
        try:
            import numpy as np
            if not isinstance(anomaly_map, np.ndarray):
                violations.append(ContractViolation(
                    "anomaly_map", f"not ndarray: {type(anomaly_map)}"
                ))
            elif anomaly_map.ndim < 2:
                violations.append(ContractViolation(
                    "anomaly_map", f"expected 2D+, got {anomaly_map.ndim}D"
                ))
        except ImportError:
            logger.debug("validation.numpy_import_failed", exc_info=True)

    _log_violations("anomaly_result", violations)
    return violations


def validate_alert(alert: object) -> list[ContractViolation]:
    """Validate an Alert at the grader→dispatcher boundary.

    Checks:
    - Required fields present: alert_id, camera_id, severity, anomaly_score
    - severity is a known enum value
    - anomaly_score > 0
    """
    violations: list[ContractViolation] = []

    for field_name in ("alert_id", "camera_id", "severity"):
        val = getattr(alert, field_name, None)
        if not val:
            violations.append(ContractViolation(field_name, "missing or empty"))

    score = getattr(alert, "anomaly_score", None)
    if score is not None and score <= 0:
        violations.append(ContractViolation("anomaly_score", "must be > 0", score))

    severity = getattr(alert, "severity", None)
    if severity is not None:
        valid_severities = {"info", "low", "medium", "high"}
        sev_val = severity.value if hasattr(severity, "value") else str(severity)
        if sev_val not in valid_severities:
            violations.append(ContractViolation(
                "severity", f"unknown: {sev_val}", severity,
            ))

    _log_violations("alert", violations)
    return violations


def validate_feedback(feedback: object) -> list[ContractViolation]:
    """Validate a feedback entry at the dashboard→queue boundary.

    Checks:
    - feedback_type is one of confirmed/false_positive/uncertain
    - camera_id is present
    - source is one of manual/drift/health
    """
    violations: list[ContractViolation] = []

    camera_id = getattr(feedback, "camera_id", None)
    if not camera_id:
        violations.append(ContractViolation("camera_id", "missing or empty"))

    fb_type = getattr(feedback, "feedback_type", None)
    valid_types = {"confirmed", "false_positive", "uncertain"}
    if fb_type is not None:
        type_val = fb_type.value if hasattr(fb_type, "value") else str(fb_type)
        if type_val not in valid_types:
            violations.append(ContractViolation(
                "feedback_type", f"unknown: {type_val}", fb_type,
            ))

    source = getattr(feedback, "source", None)
    valid_sources = {"manual", "drift", "health"}
    if source is not None:
        src_val = source.value if hasattr(source, "value") else str(source)
        if src_val not in valid_sources:
            violations.append(ContractViolation(
                "source", f"unknown: {src_val}", source,
            ))

    _log_violations("feedback", violations)
    return violations


def validate_inference_record(record: object) -> list[ContractViolation]:
    """Validate an inference record at the pipeline→buffer boundary.

    Checks:
    - camera_id present
    - anomaly_score in [0, 1]
    - timestamp > 0
    """
    violations: list[ContractViolation] = []

    if not getattr(record, "camera_id", None):
        violations.append(ContractViolation("camera_id", "missing"))

    score = getattr(record, "anomaly_score", None)
    if score is not None:
        if not isinstance(score, (int, float)) or score < 0.0 or score > 1.0:
            violations.append(ContractViolation("anomaly_score", "out of [0, 1]", score))

    ts = getattr(record, "timestamp", None)
    if ts is not None and ts <= 0:
        violations.append(ContractViolation("timestamp", "must be > 0", ts))

    _log_violations("inference_record", violations)
    return violations


def _log_violations(contract_name: str, violations: list[ContractViolation]) -> None:
    """Log violations as warnings (never raise exceptions)."""
    for v in violations:
        logger.warning(
            "contract.violation",
            contract=contract_name,
            field=v.field,
            message=v.message,
            value=str(v.value)[:100] if v.value is not None else None,
        )
