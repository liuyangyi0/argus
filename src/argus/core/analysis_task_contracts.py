"""Closed analysis-task catalog and data-only contracts, with no model imports."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

LEGACY_TEMPLATE_ID = "legacy_anomaly.v1"
LEGACY_AUTHORITY = "legacy"
MANAGED_AUTHORITY = "managed"
_TASK_NAMESPACE = UUID("a66c26ab-254d-5712-9044-9eb823b2a7cd")


def legacy_task_id(camera_id: str) -> str:
    """Return a stable, collision-resistant ID without normalizing source IDs."""
    if not isinstance(camera_id, str) or not camera_id:
        raise ValueError("camera_id must be a non-empty string")
    return f"legacy-{uuid5(_TASK_NAMESPACE, camera_id).hex}"


class ClosedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class TaskSamplingConfig(ClosedConfig):
    mode: Literal["latest", "ordered_window"] = "latest"
    fps_target: float = Field(default=5.0, gt=0, le=120)
    capacity: int = Field(default=32, ge=1, le=256)
    max_gap_seconds: float = Field(default=1.0, gt=0, le=30)


class TaskCollectionConfig(ClosedConfig):
    enabled: bool = False
    policy: Literal["periodic", "events", "disagreement", "low_confidence"] = "periodic"
    interval_seconds: float = Field(default=10.0, ge=0.1, le=86400)
    max_samples: int = Field(default=1000, ge=1, le=100000)
    max_storage_mb: int = Field(default=1024, ge=1, le=102400)


class TaskResourceConfig(ClosedConfig):
    device: Literal["cpu", "cuda"] = "cpu"
    load_timeout_seconds: float = Field(default=30.0, ge=1, le=300)
    timeout_seconds: float = Field(default=5.0, ge=0.05, le=120)
    stop_timeout_seconds: float = Field(default=5.0, ge=0.1, le=30)
    max_inflight: Literal[1] = 1
    memory_limit_mb: int = Field(default=2048, ge=128, le=32768)


class TaskSceneConfig(ClosedConfig):
    policy: Literal["none", "require_profile"] = "none"
    profile_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def profile_binding(self):
        if (self.policy == "require_profile") != (self.profile_id is not None):
            raise ValueError("require_profile needs a profile_id; none must not carry a profile")
        return self


class TaskRulesConfig(ClosedConfig):
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, gt=0, le=1)
    labels: list[str] = Field(default_factory=list, max_length=256)
    min_duration_seconds: float = Field(default=0.0, ge=0, le=3600)
    cooldown_seconds: float = Field(default=30.0, ge=0, le=86400)
    region_polygon: list[list[float]] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def closed_rules(self):
        _validate_labels(self.labels)
        if self.region_polygon and len(self.region_polygon) < 3:
            raise ValueError("region_polygon needs at least three normalized points")
        if any(len(point) != 2 or any(not 0 <= value <= 1 for value in point)
               for point in self.region_polygon):
            raise ValueError("region_polygon coordinates must be normalized x/y pairs")
        return self


class ManagedTaskConfig(ClosedConfig):
    sampling: TaskSamplingConfig = Field(default_factory=TaskSamplingConfig)
    collection: TaskCollectionConfig = Field(default_factory=TaskCollectionConfig)
    resources: TaskResourceConfig = Field(default_factory=TaskResourceConfig)
    scene: TaskSceneConfig = Field(default_factory=TaskSceneConfig)
    rules: TaskRulesConfig = Field(default_factory=TaskRulesConfig)


def _validate_labels(labels: list[str]) -> None:
    if len(labels) != len(set(labels)):
        raise ValueError("labels must be unique")
    if any(not item.strip() or len(item) > 100 or any(ord(char) < 32 for char in item)
           for item in labels):
        raise ValueError("labels must be non-empty printable names of at most 100 characters")


class TaskModelDescriptor(ClosedConfig):
    name: str = Field(default="ONNX model", min_length=1, max_length=200)
    adapter_id: Literal["onnx_detection.v1", "onnx_pose.v1"]
    output_schema: Literal["yolov8_detection.v1", "yolov8_pose.v1"]
    labels: list[str] = Field(min_length=1, max_length=256)
    input_size: list[int] = Field(min_length=2, max_length=2)
    input_layout: Literal["NCHW"] = "NCHW"
    input_dtype: Literal["float32"] = "float32"
    color_order: Literal["RGB"] = "RGB"

    @model_validator(mode="after")
    def compatible_schema(self):
        _validate_labels(self.labels)
        expected = {
            "onnx_detection.v1": "yolov8_detection.v1",
            "onnx_pose.v1": "yolov8_pose.v1",
        }[self.adapter_id]
        if self.output_schema != expected:
            raise ValueError("output_schema does not match the registered adapter")
        if self.adapter_id == "onnx_pose.v1" and self.labels != ["person"]:
            raise ValueError("yolov8_pose.v1 requires the COCO-17 person schema")
        if any(value < 16 or value > 4096 for value in self.input_size):
            raise ValueError("input_size must contain H/W dimensions from 16 to 4096")
        return self


_TEMPLATES = {
    LEGACY_TEMPLATE_ID: {"name": "Legacy anomaly detection", "authority": LEGACY_AUTHORITY,
                         "model_roles": {}, "managed": False},
    "object_detection.v1": {"name": "Object detection", "authority": MANAGED_AUTHORITY,
                            "model_roles": {"detector": "onnx_detection.v1"}, "managed": True},
    "pose_estimation.v1": {"name": "Pose estimation", "authority": MANAGED_AUTHORITY,
                           "model_roles": {"pose": "onnx_pose.v1"}, "managed": True},
    "ppe_compliance.v1": {"name": "PPE compliance", "authority": MANAGED_AUTHORITY,
                          "model_roles": {"detector": "onnx_detection.v1"}, "managed": True},
    "ppe_compliance_dual.v1": {"name": "PPE compliance (separate person/head models)",
                               "authority": MANAGED_AUTHORITY,
                               "model_roles": {"person_detector": "onnx_detection.v1",
                                               "head_detector": "onnx_detection.v1"}, "managed": True},
    "fast_motion.v1": {"name": "Fast motion", "authority": MANAGED_AUTHORITY,
                       "model_roles": {}, "managed": True},
}


def get_default_config(template_id: str) -> dict:
    if template_id not in _TEMPLATES or not _TEMPLATES[template_id]["managed"]:
        raise ValueError("template is not a managed task template")
    config = ManagedTaskConfig().model_dump(mode="json")
    if template_id == "fast_motion.v1":
        config["sampling"].update(mode="ordered_window", fps_target=60.0, capacity=64,
                                  max_gap_seconds=0.1)
    if template_id in {"ppe_compliance.v1", "ppe_compliance_dual.v1"}:
        config["rules"]["min_duration_seconds"] = 2.0
    return config


def template_specs() -> list[dict]:
    return [
        {
            "template_id": template_id, **deepcopy(spec),
            "model_role_requirements": {
                role: {"required_labels": (
                    ["person", "helmet", "bare_head"] if template_id == "ppe_compliance.v1"
                    else ["person"] if template_id == "ppe_compliance_dual.v1" and role == "person_detector"
                    else ["helmet", "bare_head"] if template_id == "ppe_compliance_dual.v1"
                    else []
                )} for role in spec["model_roles"]
            },
            "availability": "shadow" if spec["managed"] else "legacy_compatibility",
            "supports_creation": spec["managed"],
            "deployment_authority": spec["authority"],
            "safety_contract": "shadow_only_no_production_alerts" if spec["managed"]
            else "existing_scene_guard_and_model_release_service",
            "allowed_stages": ["shadow"] if spec["managed"] else [],
            "production_readiness": {
                "ready": False, "decision": "NO-GO",
                "reason": "Template-specific production evaluation and signing policy are not implemented"
                if spec["managed"] else "Legacy release service retains exclusive authority",
            },
            "default_config": get_default_config(template_id) if spec["managed"] else None,
        }
        for template_id, spec in _TEMPLATES.items()
    ]


def template_spec(template_id: str) -> dict:
    for spec in template_specs():
        if spec["template_id"] == template_id:
            return spec
    raise ValueError("unknown task template")


def normalize_task_config(template_id: str, config: dict) -> dict:
    default = get_default_config(template_id)
    if not isinstance(config, dict):
        raise ValueError("task configuration must be a JSON object")
    # Merge only the declared sections; extra keys remain visible to Pydantic
    # and are rejected rather than silently ignored.
    merged = {**default, **config}
    for section, value in config.items():
        if section in default and isinstance(value, dict):
            merged[section] = {**default[section], **value}
    normalized = ManagedTaskConfig.model_validate(merged).model_dump(mode="json")
    if template_id == "fast_motion.v1" and normalized["sampling"]["mode"] != "ordered_window":
        raise ValueError("fast_motion requires ordered_window sampling")
    if template_id == "fast_motion.v1" and normalized["rules"]["labels"] not in ([], ["fast_projectile"]):
        raise ValueError("fast_motion supports only the fast_projectile class")
    if (template_id == "ppe_compliance_dual.v1"
            and not set(normalized["rules"]["labels"]) <= {"person", "helmet", "bare_head"}):
        raise ValueError("dual PPE supports only person, helmet and bare_head labels")
    return normalized


def model_role_incompatibilities(template_id: str, role: str, descriptor: dict,
                                 task_labels: list[str]) -> list[str]:
    """Data-only role checks shared by listing, freezing and final resolution."""
    spec = template_spec(template_id)
    if role not in spec["model_roles"]:
        return ["ROLE_NOT_DECLARED"]
    reasons = []
    if descriptor.get("adapter_id") != spec["model_roles"][role]:
        reasons.append("ADAPTER_MISMATCH")
    labels = set(descriptor.get("labels", []))
    required = set(spec["model_role_requirements"][role]["required_labels"])
    if not required <= labels:
        reasons.append("REQUIRED_LABELS_MISSING")
    requested = set(task_labels)
    if template_id == "ppe_compliance_dual.v1":
        # Selection is across the task's output, not a requirement that a
        # person-only model also knows helmet classes (or vice versa).
        requested &= required
    if not requested <= labels:
        reasons.append("TASK_LABELS_MISSING")
    return reasons
