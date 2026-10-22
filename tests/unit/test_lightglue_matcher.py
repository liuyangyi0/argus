"""Offline model loading and coordinate contracts; no optional model is needed."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from argus.preprocessing import lightglue_matcher as adapter


class _Tensor:
    def __init__(self, array: np.ndarray, device: Any = "cpu") -> None:
        self.array = array
        self.device = device

    def to(self, **kwargs: Any) -> _Tensor:
        self.device = kwargs.get("device", self.device)
        return self

    def __truediv__(self, divisor: float) -> _Tensor:
        return _Tensor(self.array / divisor, self.device)

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class _Device:
    def __init__(self, name: str) -> None:
        self.name = name
        self.type = name.split(":")[0]
        self.index = int(name.split(":")[1]) if ":" in name else None

    def __str__(self) -> str:
        return self.name


class _Model:
    def __init__(self, state: dict[str, np.ndarray]) -> None:
        self.state = state
        self.loads: list[bool] = []

    def state_dict(self) -> dict[str, np.ndarray]:
        return self.state

    def load_state_dict(self, state: dict[str, np.ndarray], *, strict: bool) -> None:
        self.loads.append(strict)
        if strict and set(state) != set(self.state):
            raise RuntimeError("Missing or unexpected keys")
        for key in set(state).intersection(self.state):
            if state[key].shape != self.state[key].shape:
                raise RuntimeError("Shape mismatch")

    def eval(self) -> _Model:
        return self

    def to(self, _device: Any) -> _Model:
        return self


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    extractor_bytes = b"local-extractor"
    matcher_bytes = b"local-matcher"
    extractor_path = tmp_path / "aliked-n16.pth"
    matcher_path = tmp_path / "aliked_lightglue.pth"
    extractor_path.write_bytes(extractor_bytes)
    matcher_path.write_bytes(matcher_bytes)
    config = adapter.LightGlueMatcherConfig(
        extractor_weights=extractor_path,
        extractor_sha256=hashlib.sha256(extractor_bytes).hexdigest(),
        matcher_weights=matcher_path,
        matcher_sha256=hashlib.sha256(matcher_bytes).hexdigest(),
        device="cpu", process_width=64,
    )
    extractor_state = {"weight": np.ones(1)}
    matcher_state = {"transformers.0.self_attn.weight": np.ones(1)}
    loader_calls: list[dict[str, Any]] = []

    def safe_load(source, **kwargs):
        loader_calls.append(kwargs)
        return extractor_state if source.read() == extractor_bytes else matcher_state

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A real download was attempted")

    torch = SimpleNamespace(
        __version__="mock-torch", device=_Device, float32="float32",
        from_numpy=lambda array: _Tensor(array), inference_mode=nullcontext,
        is_tensor=lambda value: isinstance(value, np.ndarray), isfinite=np.isfinite,
        load=safe_load,
        cuda=SimpleNamespace(
            is_available=lambda: False, device_count=lambda: 0,
            synchronize=lambda _device: None,
        ),
        hub=SimpleNamespace(load_state_dict_from_url=forbidden, download_url_to_file=forbidden),
    )

    class Extractor(_Model):
        def __init__(self, **kwargs):
            super().__init__(extractor_state)
            self.config = kwargs
            self.points = np.array([[[8.0, 10.0], [20.0, 18.0]]])
            self.images: list[np.ndarray] = []
            self.resizes: list[Any] = []
            self.feature_mutator = None
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(adapter.ALIKED_WEIGHTS_URL), strict=True,
            )

        def extract(self, image, *, resize):
            self.images.append(image.array.copy())
            self.resizes.append(resize)
            features = {
                "keypoints": _Tensor(self.points.copy(), image.device),
                "descriptors": _Tensor(
                    np.ones((1, self.points.shape[1], 128)), image.device,
                ),
                "image_size": _Tensor(
                    np.array([[image.array.shape[2], image.array.shape[1]]]), image.device,
                ),
            }
            if self.feature_mutator is not None:
                self.feature_mutator(features)
            return features

    class Matcher(_Model):
        def __init__(self, **kwargs):
            super().__init__({**matcher_state, "confidence_thresholds": np.array([0.9])})
            self.config = kwargs
            self.pairs = np.array([[0, 0], [1, 1]], dtype=np.int64)
            self.scores = np.array([0.9, 0.8])
            self.calls = 0
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(adapter.LIGHTGLUE_WEIGHTS_URL), strict=False,
            )

        def __call__(self, features):
            self.calls += 1
            device = features["image0"]["keypoints"].device
            return {
                "matches": [_Tensor(self.pairs, device)],
                "scores": [_Tensor(self.scores, device)],
            }

    lightglue = SimpleNamespace(__version__="mock-lightglue", ALIKED=Extractor, LightGlue=Matcher)
    original_import = adapter.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "torch":
            return torch
        if name == "lightglue":
            return lightglue
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(adapter.importlib, "import_module", import_module)
    provenance = {
        "url": "https://github.com/cvg/LightGlue.git",
        "vcs_info": {"vcs": "git", "commit_id": adapter.LIGHTGLUE_SOURCE_COMMIT},
    }
    package_origin = tmp_path / "installed-lightglue" / "lightglue" / "__init__.py"
    package_origin.parent.mkdir(parents=True)
    package_origin.write_text("# fake package; never imported\n")
    monkeypatch.setattr(adapter.importlib.metadata, "distribution", lambda _name: SimpleNamespace(
        version="0.0", read_text=lambda _name: json.dumps(provenance),
        locate_file=lambda _path: package_origin,
    ))
    monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda _name: SimpleNamespace(
        origin=str(package_origin),
    ))
    return SimpleNamespace(
        config=config, torch=torch, lightglue=lightglue,
        loader_calls=loader_calls, original_loader=forbidden, provenance=provenance,
        package_origin=package_origin,
    )


def test_initialization_uses_verified_local_safe_states_and_restores_loader(runtime):
    matcher = adapter.LightGlueMatcher(runtime.config)

    assert runtime.loader_calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu", "weights_only": True},
    ]
    assert runtime.torch.hub.load_state_dict_from_url is runtime.original_loader
    assert runtime.torch.hub.download_url_to_file is runtime.original_loader
    assert matcher._extractor.loads == [True, True]
    assert matcher._matcher.loads[-1] is True
    assert matcher._matcher.config["features"] == "aliked"


def test_weight_hash_is_checked_before_optional_import(runtime, monkeypatch):
    runtime.config.extractor_weights.write_bytes(b"changed")

    def forbid_import(*_args, **_kwargs):
        raise AssertionError("Optional dependency imported before integrity validation")

    monkeypatch.setattr(adapter.importlib, "import_module", forbid_import)
    with pytest.raises(adapter.LightGlueError) as exc:
        adapter.LightGlueMatcher(runtime.config)
    assert exc.value.code == "weights_sha256_mismatch"


@pytest.mark.parametrize("provenance", [
    None,
    {"url": "https://github.com/cvg/LightGlue.git"},
    {"url": "https://github.com/cvg/LightGlue.git", "vcs_info": {
        "vcs": "git", "commit_id": "0" * 40,
    }},
    {"url": "https://github.com/not-official/LightGlue.git", "vcs_info": {
        "vcs": "git", "commit_id": adapter.LIGHTGLUE_SOURCE_COMMIT,
    }},
])
def test_unreviewed_runtime_is_rejected_before_optional_import(runtime, monkeypatch, provenance):
    monkeypatch.setattr(adapter.importlib.metadata, "distribution", lambda _name: SimpleNamespace(
        version="0.0", read_text=lambda _name: json.dumps(provenance),
    ))

    def forbid_import(*_args, **_kwargs):
        raise AssertionError("An unverified optional package was imported")

    monkeypatch.setattr(adapter.importlib, "import_module", forbid_import)
    with pytest.raises(adapter.LightGlueError) as exc:
        adapter.LightGlueMatcher(runtime.config)
    assert exc.value.code == "source_unverified"
    assert runtime.loader_calls == []


def test_missing_runtime_distribution_is_explicit_before_import(runtime, monkeypatch):
    def missing(_name):
        raise adapter.importlib.metadata.PackageNotFoundError("lightglue")

    monkeypatch.setattr(adapter.importlib.metadata, "distribution", missing)
    with pytest.raises(adapter.LightGlueError) as exc:
        adapter.LightGlueMatcher(runtime.config)
    assert exc.value.code == "dependency_missing"
    assert runtime.loader_calls == []


def test_same_named_package_cannot_shadow_pinned_installation(runtime, monkeypatch, tmp_path):
    shadow = tmp_path / "unreviewed" / "lightglue.py"
    shadow.parent.mkdir()
    shadow.write_text("# must never be imported\n")
    monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda _name: SimpleNamespace(
        origin=str(shadow),
    ))

    def forbid_import(*_args, **_kwargs):
        raise AssertionError("A shadowing optional package was imported")

    monkeypatch.setattr(adapter.importlib, "import_module", forbid_import)
    with pytest.raises(adapter.LightGlueError) as exc:
        adapter.LightGlueMatcher(runtime.config)
    assert exc.value.code == "source_unverified"
    assert runtime.loader_calls == []


def test_cuda_unavailable_never_becomes_cpu(runtime):
    config = adapter.LightGlueMatcherConfig(
        **{**runtime.config.__dict__, "device": "cuda:0"},
    )
    with pytest.raises(adapter.LightGlueError) as exc:
        adapter.LightGlueMatcher(config)
    assert exc.value.code == "device_unavailable"
    assert runtime.loader_calls == []


def test_scoped_loader_rejects_unknown_urls_and_restores_after_failure(runtime):
    with pytest.raises(adapter.LightGlueError) as exc:
        with adapter._local_weight_loader(runtime.torch, {}):
            runtime.torch.hub.load_state_dict_from_url("https://invalid.example/unapproved.pth")
    assert exc.value.code == "offline_weights_required"
    assert runtime.torch.hub.load_state_dict_from_url is runtime.original_loader
    assert runtime.torch.hub.download_url_to_file is runtime.original_loader


def test_strict_legacy_migration_does_not_allow_missing_learned_parameters():
    model = _Model({
        "transformers.0.self_attn.weight": np.ones(1),
        "confidence_thresholds": np.array([0.9]),
    })
    adapter._strict_matcher_state(model, {"self_attn.0.weight": np.ones(1)})
    assert model.loads == [True]
    with pytest.raises(adapter.LightGlueError) as exc:
        adapter._strict_matcher_state(model, {"other.weight": np.ones(1)})
    assert exc.value.code == "incomplete_checkpoint"


def test_strict_legacy_migration_rejects_key_collision():
    with pytest.raises(adapter.LightGlueError) as exc:
        adapter._strict_matcher_state(_Model({}), {
            "self_attn.0.weight": np.ones(1),
            "transformers.0.self_attn.weight": np.ones(1),
        })
    assert exc.value.code == "invalid_checkpoint"


def test_nonfinite_checkpoint_is_rejected(runtime):
    runtime.torch.load = lambda *_args, **_kwargs: {"weight": np.array([np.nan])}
    with pytest.raises(adapter.LightGlueError) as exc:
        adapter.LightGlueMatcher(runtime.config)
    assert exc.value.code == "invalid_checkpoint"


def test_coordinates_use_pixel_centers_and_independent_rounded_scales(runtime):
    matcher = adapter.LightGlueMatcher(runtime.config)
    reference = np.zeros((65, 129, 3), dtype=np.uint8)
    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    result = matcher.match(reference, frame)

    points = np.array([[8.0, 10.0], [20.0, 18.0]])
    np.testing.assert_allclose(result.reference_points, (points + 0.5) * [129 / 64, 65 / 32] - 0.5)
    np.testing.assert_allclose(result.frame_points, (points + 0.5) * 2 - 0.5)
    assert result.raw_matches == 2
    assert result.metadata["diagnostic_cpu"] is True
    assert result.metadata["reference_processed_size"] == [64, 32]
    assert matcher._extractor.resizes == [None, None]
    json.dumps(result.metadata, allow_nan=False)


def test_bgr_is_converted_to_rgb_and_reference_cache_is_content_based(runtime):
    matcher = adapter.LightGlueMatcher(runtime.config)
    reference = np.empty((32, 64, 3), dtype=np.uint8)
    reference[:] = [10, 20, 30]
    frame = reference.copy()

    first = matcher.match(reference, frame)
    second = matcher.match(reference.copy(), frame)
    reference[0, 0, 0] += 1
    third = matcher.match(reference, frame)

    np.testing.assert_allclose(
        matcher._extractor.images[0][:, 0, 0], np.array([30, 20, 10]) / 255.0,
    )
    assert first.metadata["reference_cached"] is False
    assert second.metadata["reference_cached"] is True
    assert third.metadata["reference_cached"] is False
    assert len(matcher._extractor.images) == 5


@pytest.mark.parametrize("pairs", [
    np.array([[0, 0], [1, 0]], dtype=np.int64),
    np.array([[0, 0], [0, 1]], dtype=np.int64),
    np.array([[0, -1]], dtype=np.int64),
    np.array([[0, 2]], dtype=np.int64),
    np.array([[0.0, 0.0]], dtype=np.float64),
])
def test_invalid_match_indices_are_errors_not_empty_success(runtime, pairs):
    matcher = adapter.LightGlueMatcher(runtime.config)
    matcher._matcher.pairs = pairs
    matcher._matcher.scores = np.ones(len(pairs))
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    with pytest.raises(adapter.LightGlueError) as exc:
        matcher.match(frame, frame)
    assert exc.value.code == "invalid_correspondences"


@pytest.mark.parametrize("scores", [np.array([np.nan, 0.9]), np.array([1.1, 0.8]), np.array([0.9])])
def test_invalid_scores_cannot_become_normal_empty_matches(runtime, scores):
    matcher = adapter.LightGlueMatcher(runtime.config)
    matcher._matcher.scores = scores
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    with pytest.raises(adapter.LightGlueError) as exc:
        matcher.match(frame, frame)
    assert exc.value.code == "invalid_correspondences"


def test_zero_keypoints_is_valid_and_does_not_call_matcher(runtime):
    matcher = adapter.LightGlueMatcher(runtime.config)
    matcher._extractor.points = np.empty((1, 0, 2))
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    result = matcher.match(frame, frame)
    assert result.raw_matches == 0
    assert result.frame_points.shape == (0, 2)
    assert result.reference_points.shape == (0, 2)
    assert matcher._matcher.calls == 0


def test_empty_correspondence_output_is_valid(runtime):
    matcher = adapter.LightGlueMatcher(runtime.config)
    matcher._matcher.pairs = np.empty((0, 2), dtype=np.int64)
    matcher._matcher.scores = np.empty(0)
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    result = matcher.match(frame, frame)
    assert result.reference_keypoints == result.frame_keypoints == 2
    assert result.raw_matches == 0


def test_backend_failure_is_explicit(runtime):
    matcher = adapter.LightGlueMatcher(runtime.config)

    def fail(_features):
        raise RuntimeError("CUDA inference failed")

    matcher._matcher = fail
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    with pytest.raises(adapter.LightGlueError) as exc:
        matcher.match(frame, frame)
    assert exc.value.code == "inference_failed"


@pytest.mark.parametrize("mutation", [
    "nan_descriptor", "complex_descriptor", "wrong_dimension", "missing_descriptor",
    "wrong_image_size", "too_many_keypoints",
])
def test_invalid_extractor_features_cannot_become_valid_empty_matches(runtime, mutation):
    matcher = adapter.LightGlueMatcher(runtime.config)

    def corrupt(features):
        if mutation == "nan_descriptor":
            features["descriptors"].array[0, 0, 0] = np.nan
        elif mutation == "complex_descriptor":
            features["descriptors"].array = features["descriptors"].array.astype(complex) + 1j
        elif mutation == "wrong_dimension":
            features["descriptors"].array = np.ones((1, 2, 64))
        elif mutation == "missing_descriptor":
            del features["descriptors"]
        elif mutation == "wrong_image_size":
            features["image_size"].array[0, 0] += 1
        elif mutation == "too_many_keypoints":
            count = runtime.config.max_keypoints + 1
            features["keypoints"].array = np.tile([[[8.0, 10.0]]], (1, count, 1))
            features["descriptors"].array = np.ones((1, count, 128))

    matcher._extractor.feature_mutator = corrupt
    # Even a matcher that would hide the corruption as no matches is not reached.
    matcher._matcher.pairs = np.empty((0, 2), dtype=np.int64)
    matcher._matcher.scores = np.empty(0)
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    with pytest.raises(adapter.LightGlueError) as exc:
        matcher.match(frame, frame)
    assert exc.value.code == "invalid_correspondences"
    assert matcher._matcher.calls == 0
    assert matcher._reference_key is None


@pytest.mark.parametrize("output_name", ["keypoints", "descriptors", "image_size"])
def test_cuda_extractor_outputs_cannot_silently_be_cpu(runtime, output_name):
    runtime.torch.cuda.is_available = lambda: True
    runtime.torch.cuda.device_count = lambda: 1
    config = adapter.LightGlueMatcherConfig(**{**runtime.config.__dict__, "device": "cuda:0"})
    matcher = adapter.LightGlueMatcher(config)

    def cpu_output(features):
        features[output_name].device = "cpu"

    matcher._extractor.feature_mutator = cpu_output
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    with pytest.raises(adapter.LightGlueError) as exc:
        matcher.match(frame, frame)
    assert exc.value.code == "device_mismatch"
