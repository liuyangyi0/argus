"""Adapter contracts use synthetic tensors and fake SDKs; no model executes."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from argus.core.analysis_task_contracts import get_default_config
from argus.core.task_adapters import (
    FastMotionAdapter,
    OnnxYoloAdapter,
    PpeAdapter,
    PpeRoleDetector,
    TaskAdapterError,
    _reject_external_onnx_data,
    build_task_adapter,
)


class FakeSession:
    def __init__(self, *, labels=2, pose=False, providers=None):
        self.providers = providers or ["CPUExecutionProvider"]
        self.inputs = [SimpleNamespace(type="tensor(float)", shape=[1, 3, 32, 32], name="images")]
        self.outputs = [SimpleNamespace(type="tensor(float)", shape=[1, 56 if pose else 4 + labels, 2])]
        self.raw = np.zeros(self.outputs[0].shape, dtype=np.float32)
        self.fallback_disabled = False

    def disable_fallback(self):
        self.fallback_disabled = True

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, names, values):
        assert self.fallback_disabled
        tensor = values["images"]
        assert tensor.shape == (1, 3, 32, 32) and tensor.dtype == np.float32
        return [self.raw]


class FakeOrt:
    __version__ = "1.23.2"

    def __init__(self, session, available=None):
        self.session = session
        self.available = available or ["CPUExecutionProvider"]
        self.options = []
        self.loads = 0
        self.preloads = []

    def preload_dlls(self, **kwargs):
        self.preloads.append(kwargs)

    def get_available_providers(self):
        return self.available

    def SessionOptions(self):
        return SimpleNamespace(add_session_config_entry=lambda *args: self.options.append(args))

    def InferenceSession(self, payload, *, sess_options, providers, enable_fallback, read_config_from_model):
        assert payload == b"synthetic-not-an-onnx-model"
        assert enable_fallback == 0 and read_config_from_model is False
        self.loads += 1
        return self.session


def descriptor(tmp_path, *, pose=False):
    payload = b"synthetic-not-an-onnx-model"
    path = tmp_path / "model.onnx"
    path.write_bytes(payload)
    return {"artifact_path": str(path), "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "adapter_id": "onnx_pose.v1" if pose else "onnx_detection.v1",
            "output_schema": "yolov8_pose.v1" if pose else "yolov8_detection.v1",
            "labels": ["person"] if pose else ["person", "box"], "input_size": [32, 32],
            "input_layout": "NCHW", "input_dtype": "float32", "color_order": "RGB"}


def make_adapter(tmp_path, *, pose=False, session=None, config=None):
    session = session or FakeSession(pose=pose)
    ort = FakeOrt(session)
    instance = OnnxYoloAdapter(descriptor(tmp_path, pose=pose),
                               config or get_default_config("pose_estimation.v1" if pose else "object_detection.v1"),
                               tmp_path, ort_module=ort, validator=lambda _: None)
    return instance, session, ort


@pytest.mark.parametrize("change", [
    lambda session: session.inputs[0].shape.__setitem__(2, "height"),
    lambda session: session.outputs[0].shape.__setitem__(2, "anchors"),
    lambda session: session.outputs[0].shape.__setitem__(1, 7),
    lambda session: setattr(session.outputs[0], "type", "tensor(double)"),
    lambda session: session.outputs.append(session.outputs[0]),
])
def test_onnx_static_input_and_output_schema_is_checked_before_run(tmp_path, change):
    session = FakeSession()
    change(session)
    with pytest.raises(TaskAdapterError) as failure:
        make_adapter(tmp_path, session=session)
    assert failure.value.code == "INVALID_BUNDLE"


def test_cuda_unavailability_does_not_load_cpu_model(tmp_path):
    ort = FakeOrt(FakeSession())
    config = get_default_config("object_detection.v1")
    config["resources"]["device"] = "cuda"
    with pytest.raises(TaskAdapterError) as failure:
        OnnxYoloAdapter(descriptor(tmp_path), config, tmp_path, ort_module=ort, validator=lambda _: None)
    assert failure.value.code == "UNAVAILABLE" and ort.loads == 0


def test_windows_cuda_preloads_dependencies_before_session_without_changing_provider(tmp_path, monkeypatch):
    monkeypatch.setattr("argus.core.task_adapters.sys.platform", "win32")
    ort = FakeOrt(FakeSession(providers=["CUDAExecutionProvider"]), available=["CUDAExecutionProvider"])
    config = get_default_config("object_detection.v1")
    config["resources"]["device"] = "cuda"
    load = ort.InferenceSession

    def checked_load(*args, **kwargs):
        assert ort.preloads == [{"cuda": True, "cudnn": True, "msvc": True, "directory": None}]
        return load(*args, **kwargs)

    ort.InferenceSession = checked_load
    adapter = OnnxYoloAdapter(descriptor(tmp_path), config, tmp_path, ort_module=ort, validator=lambda _: None)
    assert adapter.provider == "CUDAExecutionProvider" and ort.loads == 1
    assert ("session.disable_cpu_ep_fallback", "1") in ort.options


@pytest.mark.parametrize("loader", [None, "raises"])
def test_windows_cuda_missing_or_failed_preload_does_not_attempt_cpu_session(tmp_path, monkeypatch, loader):
    monkeypatch.setattr("argus.core.task_adapters.sys.platform", "win32")
    ort = FakeOrt(FakeSession(), available=["CUDAExecutionProvider"])

    def fail(**_kwargs):
        raise OSError("synthetic missing dependency")

    ort.preload_dlls = fail if loader == "raises" else None
    config = get_default_config("object_detection.v1")
    config["resources"]["device"] = "cuda"
    with pytest.raises(TaskAdapterError) as failure:
        OnnxYoloAdapter(descriptor(tmp_path), config, tmp_path, ort_module=ort, validator=lambda _: None)
    assert failure.value.code == "UNAVAILABLE" and ort.loads == 0


def test_cpu_session_does_not_preload_cuda_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr("argus.core.task_adapters.sys.platform", "win32")
    _, _, ort = make_adapter(tmp_path)
    assert ort.preloads == []


def test_cuda_wrapper_and_graph_cpu_fallback_are_both_disabled(tmp_path):
    session = FakeSession(providers=["CUDAExecutionProvider"])
    ort = FakeOrt(session, available=["CUDAExecutionProvider", "CPUExecutionProvider"])
    config = get_default_config("object_detection.v1")
    config["resources"]["device"] = "cuda"
    adapter = OnnxYoloAdapter(descriptor(tmp_path), config, tmp_path, ort_module=ort, validator=lambda _: None)
    assert session.fallback_disabled
    assert ("session.disable_cpu_ep_fallback", "1") in ort.options
    assert adapter.provider == "CUDAExecutionProvider"
    session.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    adapter = OnnxYoloAdapter(descriptor(tmp_path), config, tmp_path, ort_module=ort, validator=lambda _: None)
    assert adapter.provider == "CUDAExecutionProvider"


@pytest.mark.parametrize("providers", [
    ["CPUExecutionProvider"], ["CPUExecutionProvider", "CUDAExecutionProvider"],
    ["CUDAExecutionProvider", "AzureExecutionProvider"],
    ["CUDAExecutionProvider", "CPUExecutionProvider", "TensorrtExecutionProvider"],
])
def test_cuda_rejects_missing_cuda_wrong_precedence_or_other_providers(tmp_path, providers):
    session = FakeSession(providers=providers)
    ort = FakeOrt(session, available=["CUDAExecutionProvider", "CPUExecutionProvider"])
    config = get_default_config("object_detection.v1")
    config["resources"]["device"] = "cuda"
    with pytest.raises(TaskAdapterError):
        OnnxYoloAdapter(descriptor(tmp_path), config, tmp_path, ort_module=ort, validator=lambda _: None)


def test_cpu_task_rejects_additional_provider(tmp_path):
    with pytest.raises(TaskAdapterError):
        make_adapter(tmp_path, session=FakeSession(providers=["CPUExecutionProvider", "CUDAExecutionProvider"]))


def test_old_wrapper_that_ignores_constructor_fallback_flag_is_rejected_before_load(tmp_path):
    ort = FakeOrt(FakeSession())
    ort.__version__ = "1.22.0"
    with pytest.raises(TaskAdapterError) as failure:
        OnnxYoloAdapter(descriptor(tmp_path), get_default_config("object_detection.v1"), tmp_path,
                        ort_module=ort, validator=lambda _: None)
    assert failure.value.code == "UNAVAILABLE" and ort.loads == 0


def test_constructor_cuda_placement_failure_is_not_retried_with_cpu(tmp_path):
    ort = FakeOrt(FakeSession(), available=["CUDAExecutionProvider", "CPUExecutionProvider"])
    config = get_default_config("object_detection.v1")
    config["resources"]["device"] = "cuda"
    attempts = []

    def refuse_cpu_node(payload, **kwargs):
        attempts.append(kwargs)
        assert kwargs["enable_fallback"] == 0
        assert kwargs["providers"] == ["CUDAExecutionProvider"]
        raise RuntimeError("graph contains a CPU-assigned node")

    ort.InferenceSession = refuse_cpu_node
    with pytest.raises(RuntimeError, match="CPU-assigned"):
        OnnxYoloAdapter(descriptor(tmp_path), config, tmp_path, ort_module=ort, validator=lambda _: None)
    assert len(attempts) == 1


def test_model_bytes_sha_is_checked_before_sdk_load(tmp_path):
    value = descriptor(tmp_path)
    value["artifact_sha256"] = "0" * 64
    ort = FakeOrt(FakeSession())
    with pytest.raises(TaskAdapterError):
        OnnxYoloAdapter(value, get_default_config("object_detection.v1"), tmp_path,
                        ort_module=ort, validator=lambda _: None)
    assert ort.loads == 0


@pytest.mark.parametrize("buffer", ["input", "output"])
def test_declared_buffer_budget_blocks_oversized_static_shapes(tmp_path, buffer):
    value = descriptor(tmp_path)
    config = get_default_config("object_detection.v1")
    config["resources"]["memory_limit_mb"] = 128
    session = FakeSession()
    if buffer == "input":
        value["input_size"] = [4096, 4096]
    else:
        value["labels"] = [f"label-{index}" for index in range(256)]
        session.outputs[0].shape = [1, 260, 100000]
    ort = FakeOrt(session)
    with pytest.raises(TaskAdapterError) as failure:
        OnnxYoloAdapter(value, config, tmp_path, ort_module=ort, validator=lambda _: None)
    assert failure.value.code == "RESOURCE_EXHAUSTED"


def test_detector_fake_output_is_letterboxed_filtered_and_nms_limited(tmp_path):
    adapter, session, _ = make_adapter(tmp_path)
    session.raw[0, :, 0] = [16, 16, 16, 16, 0.1, 0.9]
    session.raw[0, :, 1] = [16, 16, 16, 16, 0.1, 0.8]
    result = adapter.infer(np.zeros((16, 32, 3), dtype=np.uint8), 1.0)
    assert len(result["detections"]) == 1
    assert result["detections"][0]["label"] == "box"
    assert result["detections"][0]["bbox_xyxy"] == [8, 0, 24, 16]


@pytest.mark.parametrize("invalid", ["nan", "negative_probability", "shape", "dtype"])
def test_invalid_model_output_never_becomes_normal_observation(tmp_path, invalid):
    adapter, session, _ = make_adapter(tmp_path)
    session.raw[0, :, 0] = [16, 16, 16, 16, 0.1, 0.9]
    if invalid == "nan":
        session.raw[0, 0, 0] = np.nan
    elif invalid == "negative_probability":
        session.raw[0, 4, 0] = -1
    elif invalid == "shape":
        session.raw = session.raw[:, :, :1]
    else:
        session.raw = session.raw.astype(np.float64)
    with pytest.raises(TaskAdapterError) as failure:
        adapter.infer(np.zeros((32, 32, 3), dtype=np.uint8), 1.0)
    assert failure.value.code == "INVALID_OUTPUT"


def test_pose_fake_tensor_preserves_coco17_keypoint_schema(tmp_path):
    adapter, session, _ = make_adapter(tmp_path, pose=True)
    session.raw[0, :5, 0] = [16, 16, 16, 16, 0.9]
    session.raw[0, 5:, 0] = np.tile([10, 12, 0.8], 17)
    result = adapter.infer(np.zeros((32, 32, 3), dtype=np.uint8), 1.0)
    detection = result["detections"][0]
    assert detection["keypoint_schema"] == "coco17.v1" and len(detection["keypoints"]) == 17


def test_external_sparse_onnx_constant_is_rejected_without_loading_runtime(monkeypatch):
    tensor = SimpleNamespace(external_data=["private-file"], data_location=1)
    attribute = SimpleNamespace(
        HasField=lambda name: name == "sparse_tensor", graphs=[], tensors=[], sparse_tensors=[],
        sparse_tensor=SimpleNamespace(values=tensor, indices=SimpleNamespace(external_data=[], data_location=0)),
    )
    graph = SimpleNamespace(initializer=[], sparse_initializer=[],
                            node=[SimpleNamespace(domain="", attribute=[attribute])])
    onnx = SimpleNamespace(load_model_from_string=lambda _: SimpleNamespace(functions=[], graph=graph))
    monkeypatch.setattr("argus.core.task_adapters.importlib.import_module", lambda name: onnx)
    with pytest.raises(TaskAdapterError, match="external ONNX"):
        _reject_external_onnx_data(b"synthetic")


def item(label, box):
    return {"label": label, "bbox_xyxy": box, "confidence": 0.9}


class FakeDetector:
    provider = "fake-provider"

    def __init__(self):
        self.detections = []

    def infer(self, frame, timestamp):
        return {"detections": self.detections}

    def reset(self, reason):
        pass


def test_ppe_uses_explicit_bare_head_duration_and_resets_after_sampling_gap():
    detector = FakeDetector()
    detector.detections = [item("person", [0, 0, 100, 200]), item("bare_head", [20, 0, 40, 20])]
    adapter = PpeAdapter(detector, get_default_config("ppe_compliance.v1"))
    assert adapter.infer(None, 0)["people"][0]["state"] == "pending"
    assert adapter.infer(None, 1)["people"][0]["state"] == "pending"
    assert adapter.infer(None, 2)["people"][0]["state"] == "violation"
    assert adapter.infer(None, 5)["people"][0]["state"] == "pending"
    detector.detections = [item("person", [0, 0, 100, 200])]
    assert adapter.infer(None, 6)["people"][0]["state"] == "unknown"


def test_ppe_conflicting_or_ambiguous_evidence_remains_unknown():
    detector = FakeDetector()
    adapter = PpeAdapter(detector, get_default_config("ppe_compliance.v1"))
    detector.detections = [item("person", [0, 0, 100, 200]), item("helmet", [20, 0, 40, 20]),
                           item("bare_head", [20, 0, 40, 20])]
    assert adapter.infer(None, 0)["people"][0]["reason"] == "conflicting_head_evidence"
    detector.detections.append(item("person", [10, 0, 110, 200]))
    assert all(person["state"] == "unknown" for person in adapter.infer(None, 1)["people"])


def dual_manifest(tmp_path):
    models = {}
    for role, labels in (("person_detector", ["person"]), ("head_detector", ["helmet", "bare_head"])):
        value = descriptor(tmp_path)
        path = tmp_path / f"{role}.onnx"
        path.write_bytes(b"synthetic-not-an-onnx-model")
        value.update(artifact_path=str(path), labels=labels)
        models[role] = value
    return {"template_id": "ppe_compliance_dual.v1", "models": models,
            "config": get_default_config("ppe_compliance_dual.v1")}


def test_dual_ppe_validates_every_artifact_before_loading_any_sdk(tmp_path, monkeypatch):
    manifest = dual_manifest(tmp_path)
    manifest["models"]["head_detector"]["artifact_sha256"] = "0" * 64
    loads = []
    monkeypatch.setattr("argus.core.task_adapters.OnnxYoloAdapter", lambda *args: loads.append(args))
    with pytest.raises(TaskAdapterError, match="SHA-256"):
        build_task_adapter(manifest, tmp_path)
    assert loads == []


def test_dual_ppe_uses_same_frame_only_after_all_models_ready_and_splits_the_budget(tmp_path, monkeypatch):
    manifest = dual_manifest(tmp_path)
    loaded, observations = [], []

    class Role:
        provider = "CPUExecutionProvider"

        def __init__(self, value, config, root):
            self.labels = value["labels"]
            assert config["resources"]["memory_limit_mb"] == 1024
            assert config["rules"]["labels"] == []
            loaded.append(self)

        def reset(self, reason):
            observations.append((self.labels, reason))

        def infer(self, frame, timestamp):
            assert len(loaded) == 2
            observations.append((frame, timestamp))
            return {"state": "observed", "detections": (
                [item("person", [0, 0, 100, 200]), item("helmet", [20, 0, 40, 20])]
                if self.labels == ["person"] else
                [item("bare_head", [20, 0, 40, 20]), item("person", [200, 0, 300, 200])]
            )}

    monkeypatch.setattr("argus.core.task_adapters.OnnxYoloAdapter", Role)
    adapter = build_task_adapter(manifest, tmp_path)
    frame = object()
    observed = adapter.infer(frame, 1)
    assert observations == [(frame, 1), (frame, 1)]
    assert [row["label"] for row in observed["detections"]] == ["person", "bare_head"]
    assert len(observed["people"]) == 1 and observed["people"][0]["state"] == "pending"
    adapter.reset("source-changed")
    assert observations[-2:] == [(["person"], "source-changed"), (["helmet", "bare_head"], "source-changed")]


def test_dual_ppe_second_model_load_failure_never_returns_an_adapter_or_runs_first_model(tmp_path, monkeypatch):
    loaded = []

    def load(value, config, root):
        if loaded:
            raise TaskAdapterError("UNAVAILABLE", "head role failed to load")
        first = SimpleNamespace(provider="CPUExecutionProvider", infer=lambda *args: pytest.fail("partial model ran"))
        loaded.append(first)
        return first

    monkeypatch.setattr("argus.core.task_adapters.OnnxYoloAdapter", load)
    with pytest.raises(TaskAdapterError, match="head role"):
        build_task_adapter(dual_manifest(tmp_path), tmp_path)
    assert len(loaded) == 1


def test_dual_ppe_infer_failure_does_not_return_partial_evidence_or_advance_confirmation():
    first = SimpleNamespace(provider="CPUExecutionProvider", reset=lambda _: None,
                            infer=lambda *args: {"state": "observed", "detections": [item("person", [0, 0, 100, 200])]})
    def fail(*args):
        raise TaskAdapterError("INFER_TIMEOUT", "head role timed out")
    second = SimpleNamespace(provider="CPUExecutionProvider", reset=lambda _: None, infer=fail)
    adapter = PpeAdapter(PpeRoleDetector(first, second), get_default_config("ppe_compliance_dual.v1"))
    with pytest.raises(TaskAdapterError, match="timed out"):
        adapter.infer(None, 1)
    assert adapter._tracks == [] and adapter._last_timestamp is None


class FakeMotion:
    def __init__(self, **kwargs):
        self.frames = self.resets = 0

    def process(self, frame, *, timestamp):
        self.frames += 1
        return SimpleNamespace(candidates=[], suppression_reason=None)

    def reset(self):
        self.resets += 1


def test_fast_motion_insufficient_sampling_latches_until_explicit_reset():
    adapter = FastMotionAdapter(get_default_config("fast_motion.v1"), detector_factory=FakeMotion)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert adapter.infer(frame, 1)["state"] == "warming_up"
    assert adapter.infer(frame, 1 + 1 / 60)["state"] == "observed"
    assert adapter.infer(frame, 1.1)["state"] == "sampling_insufficient"
    count = adapter.detector.frames
    assert adapter.infer(frame, 1.12)["state"] == "sampling_insufficient"
    assert adapter.detector.frames == count
    adapter.reset("source_changed")
    assert adapter.infer(frame, 2)["state"] == "warming_up"
