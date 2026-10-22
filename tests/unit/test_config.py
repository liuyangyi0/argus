"""Tests for configuration loading and validation."""


import pytest
import yaml
from pydantic import ValidationError

from argus.config.loader import load_config
from argus.config.schema import (
    ArgusConfig, ZonePriority,
    SeverityThresholds, MOG2Config, CameraConfig, TemporalConfirmation,
    AnomalyConfig, CalibrationConfig, FastMotionConfig, SimplexConfig,
    PersistentChangeConfig, WaterLeakConfig, PlumeConfig, SensorFusionConfig,
    AlignmentConfig, CandidateModelEvidenceConfig, SceneGuardConfig,
    StaticCandidateConfirmationConfig, LowLightConfig, PersonFilterConfig,
    DashboardConfig,
)


def _model_required_confirmation() -> StaticCandidateConfirmationConfig:
    return StaticCandidateConfirmationConfig(
        enabled=True,
        mode="enforce",
        model_evidence=CandidateModelEvidenceConfig(
            enabled=True,
            fusion_policy="model_required",
            calibration_id="cal-1",
            global_score_min=0.5,
            roi_p95_min=0.5,
            roi_context_delta_min=0.1,
        ),
    )


def _safe_model_required_camera(**overrides) -> CameraConfig:
    values = {
        "camera_id": "c1",
        "name": "test",
        "source": "test.mp4",
        "scene_guard": SceneGuardConfig(
            enabled=True,
            require_profile_model_binding=True,
            allow_alignable=False,
        ),
        "persistent_change": PersistentChangeConfig(enabled=True),
        "static_confirmation": _model_required_confirmation(),
        "low_light": LowLightConfig(enabled=False),
    }
    values.update(overrides)
    return CameraConfig(**values)


class TestConfigSchema:
    def test_default_config_is_valid(self):
        """Default config should pass validation."""
        config = ArgusConfig()
        assert config.node_id == "argus-edge-01"
        assert config.log_level == "INFO"

    def test_go2rtc_webrtc_listener_requires_valid_ipv4(self):
        assert DashboardConfig(
            go2rtc_webrtc_listen_host="127.0.0.1"
        ).go2rtc_webrtc_listen_host == "127.0.0.1"
        with pytest.raises(ValidationError, match="IPv4"):
            DashboardConfig(go2rtc_webrtc_listen_host="999.1.2.3")

    def test_severity_thresholds_ordering(self):
        """Severity thresholds should be monotonically increasing."""
        config = ArgusConfig()
        t = config.alerts.severity_thresholds
        assert t.info < t.low < t.medium < t.high

    def test_zone_priority_values(self):
        """Zone priority enum values should be correct."""
        assert ZonePriority.CRITICAL.value == "critical"
        assert ZonePriority.STANDARD.value == "standard"
        assert ZonePriority.LOW_PRIORITY.value == "low_priority"

    def test_inverted_severity_thresholds_rejected(self):
        """Inverted severity thresholds must be rejected."""
        with pytest.raises(ValidationError, match="ordered"):
            SeverityThresholds(info=0.95, low=0.85, medium=0.70, high=0.50)

    def test_equal_severity_thresholds_rejected(self):
        """Equal severity thresholds must be rejected."""
        with pytest.raises(ValidationError, match="ordered"):
            SeverityThresholds(info=0.5, low=0.5, medium=0.5, high=0.5)

    def test_mog2_history_too_low_rejected(self):
        """MOG2 history below minimum must be rejected."""
        with pytest.raises(ValidationError):
            MOG2Config(history=1)

    def test_mog2_heartbeat_too_low_rejected(self):
        """Heartbeat frames below minimum must be rejected."""
        with pytest.raises(ValidationError):
            MOG2Config(heartbeat_frames=1)

    def test_fps_target_zero_rejected(self):
        """FPS target of 0 must be rejected."""
        with pytest.raises(ValidationError):
            CameraConfig(camera_id="c1", name="t", source="x", fps_target=0)

    def test_scene_stable_geometry_cannot_exceed_alignment_capacity(self):
        with pytest.raises(ValidationError, match="stable_max_corner_displacement_px"):
            CameraConfig(
                camera_id="c1",
                name="t",
                source="x",
                alignment=AlignmentConfig(enabled=True, max_shift_px=3.0),
                scene_guard=SceneGuardConfig(
                    enabled=True,
                    stable_max_shift_px=2.0,
                    stable_max_corner_displacement_px=4.0,
                ),
            )

    def test_scene_guard_schema_defaults_fail_closed(self):
        scene_guard = SceneGuardConfig()

        assert scene_guard.enabled is True
        assert scene_guard.require_profile_model_binding is True
        assert scene_guard.require_operator_approved_profile is True
        assert scene_guard.fallback_to_baseline is False

    def test_evidence_threshold_too_low_rejected(self):
        """evidence_threshold below minimum must be rejected."""
        with pytest.raises(ValidationError):
            TemporalConfirmation(evidence_threshold=0.1)

    def test_valid_bounds_accepted(self):
        """Valid parameter values within bounds should be accepted."""
        t = SeverityThresholds(info=0.3, low=0.5, medium=0.7, high=0.9)
        assert t.info == 0.3
        m = MOG2Config(history=100, heartbeat_frames=50)
        assert m.history == 100

    def test_object_safety_alert_classes_are_explicit_and_detected(self):
        config = PersonFilterConfig(
            classes_to_detect=[0, 43],
            alert_class_ids=[43],
            alert_min_confidence=0.92,
        )

        assert config.alert_class_ids == [43]
        assert config.alert_min_confidence == 0.92
        assert PersonFilterConfig().alert_class_ids == []

    def test_object_safety_alert_typo_is_rejected(self):
        with pytest.raises(ValidationError, match="alert_clas_ids"):
            PersonFilterConfig(alert_clas_ids=[43])

    @pytest.mark.parametrize("alert_class_ids", [[43, 43], [0], [39]])
    def test_invalid_object_safety_alert_classes_are_rejected(
        self,
        alert_class_ids,
    ):
        with pytest.raises(ValidationError, match="alert_class_ids"):
            PersonFilterConfig(
                classes_to_detect=[0, 43],
                alert_class_ids=alert_class_ids,
            )


class TestConfigLoader:
    def test_load_valid_yaml(self, tmp_path):
        """Should load and validate a YAML config file."""
        config_data = {
            "node_id": "test-node",
            "cameras": [
                {
                    "camera_id": "cam_01",
                    "name": "Test Cam",
                    "source": "test.mp4",
                    "protocol": "file",
                }
            ],
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.node_id == "test-node"
        assert len(config.cameras) == 1
        assert config.cameras[0].camera_id == "cam_01"

    @pytest.mark.parametrize("misspelled", ["static_confirmaton", "scene_guad"])
    def test_camera_safety_block_name_typo_is_rejected(self, tmp_path, misspelled):
        config_data = {
            "cameras": [
                {
                    "camera_id": "cam_01",
                    "name": "Test Cam",
                    "source": "test.mp4",
                    misspelled: {"enabled": True},
                }
            ]
        }
        config_file = tmp_path / "typo.yaml"
        config_file.write_text(yaml.safe_dump(config_data), encoding="utf-8")

        with pytest.raises(ValidationError, match=misspelled):
            load_config(config_file)

    def test_load_missing_file_raises(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_empty_yaml(self, tmp_path):
        """Empty YAML should produce default config."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = load_config(config_file)
        assert config.node_id == "argus-edge-01"

    def test_load_default_config_file(self):
        """The default.yaml shipped with the project should load correctly."""
        config = load_config("configs/default.yaml")
        assert config.node_id == "argus-edge-01"
        assert len(config.cameras) >= 1
        assert config.cameras[0].camera_id == "c"
        assert config.cameras[0].protocol == "usb"
        assert config.cameras[0].usb.device_name == "OBSBOT Meet 2 StreamCamera"
        assert config.cameras[0].usb.pixel_format == "mjpeg"
        assert config.cameras[0].usb.min_runtime_fps == pytest.approx(50.0)
        scene_guard = config.cameras[0].scene_guard
        assert scene_guard.enabled is True
        assert scene_guard.require_profile_model_binding is True
        assert scene_guard.require_operator_approved_profile is True
        assert scene_guard.fallback_to_baseline is False
        assert config.dashboard.host == "127.0.0.1"


class TestDinomaly2Config:
    """B1-6: Dinomaly2 configuration tests."""

    def test_dinomaly2_config_valid(self):
        """model_type=dinomaly2 + backbone + layers 正确解析。"""
        config = AnomalyConfig(
            model_type="dinomaly2",
            dinomaly_backbone="dinov2_vitb14",
            dinomaly_encoder_layers=[2, 5, 8, 11],
        )
        assert config.model_type == "dinomaly2"
        assert config.dinomaly_backbone == "dinov2_vitb14"

    def test_dinomaly2_few_shot_minimum(self):
        """dinomaly_few_shot_images validates range."""
        config = AnomalyConfig(model_type="dinomaly2", dinomaly_few_shot_images=8)
        assert config.dinomaly_few_shot_images == 8

    def test_dinomaly2_multi_class_default_false(self):
        """dinomaly_multi_class defaults to False."""
        config = AnomalyConfig(model_type="dinomaly2")
        assert config.dinomaly_multi_class is False

    def test_release_shadow_evidence_gate_bounds(self):
        config = AnomalyConfig(
            min_shadow_samples=100,
            max_shadow_alert_rate_delta=0.05,
            max_shadow_score_divergence=0.2,
            max_shadow_latency_ms=1000.0,
        )
        assert config.min_shadow_samples == 100
        assert config.max_shadow_alert_rate_delta == pytest.approx(0.05)

        with pytest.raises(ValidationError):
            AnomalyConfig(max_shadow_alert_rate_delta=1.5)


class TestQuantizationConfig:
    """B2-4: Quantization configuration tests."""

    def test_quantization_config_values(self):
        """fp32/fp16/int8 都是合法值。"""
        for q in ("fp32", "fp16", "int8"):
            config = AnomalyConfig(quantization=q)
            assert config.quantization == q

    def test_int8_calibration_images_range(self):
        """calibration_images validates range 50-1000."""
        config = AnomalyConfig(quantization="int8", quantization_calibration_images=100)
        assert config.quantization_calibration_images == 100

        with pytest.raises(ValidationError):
            AnomalyConfig(quantization="int8", quantization_calibration_images=10)


class TestCUSUMConfig:
    """A1-3: CUSUM evidence config tests."""

    def test_evidence_lambda_range(self):
        """evidence_lambda must be 0.80-0.99."""
        config = TemporalConfirmation(evidence_lambda=0.95)
        assert config.evidence_lambda == 0.95

        with pytest.raises(ValidationError):
            TemporalConfirmation(evidence_lambda=0.5)

    def test_evidence_threshold_range(self):
        """evidence_threshold must be 0.5-20.0."""
        config = TemporalConfirmation(evidence_threshold=5.0)
        assert config.evidence_threshold == 5.0


class TestCalibrationConfig:
    """A2-2: Calibration config tests."""

    def test_calibration_default_disabled(self):
        """Calibration should be disabled by default."""
        config = CalibrationConfig()
        assert config.enabled is False

    def test_calibration_fprs(self):
        """FPR values should be within range."""
        config = CalibrationConfig(
            enabled=True,
            target_fpr_info=0.10,
            target_fpr_low=0.01,
            target_fpr_medium=0.001,
            target_fpr_high=0.0001,
        )
        assert config.target_fpr_info == 0.10


class TestSensorFusionConfig:
    """P9 external sensor fusion configuration tests."""

    def test_default_thresholds_are_ordered(self):
        config = SensorFusionConfig()
        assert config.gas_warning_ppm <= config.gas_danger_ppm
        assert config.pressure_drop_warning_kpa <= config.pressure_drop_danger_kpa
        assert config.humidity_warning_rh <= config.humidity_danger_rh
        assert config.temperature_warning_c <= config.temperature_danger_c
        assert config.warning_multiplier <= config.danger_multiplier

    def test_invalid_threshold_order_rejected(self):
        with pytest.raises(ValidationError, match="gas warning"):
            SensorFusionConfig(gas_warning_ppm=100.0, gas_danger_ppm=50.0)

    def test_default_yaml_exposes_sensor_fusion_rules(self):
        config = load_config("configs/default.yaml")
        assert config.sensor_fusion.enabled is False
        assert config.sensor_fusion.gas_warning_ppm == pytest.approx(10.0)
        assert config.sensor_fusion.danger_multiplier == pytest.approx(2.5)


class TestSimplexConfig:
    """A3-2: Simplex config tests."""

    def test_simplex_default_enabled(self):
        config = SimplexConfig()
        assert config.enabled is True

    def test_simplex_diff_threshold_range(self):
        with pytest.raises(ValidationError):
            SimplexConfig(diff_threshold=5)  # below 10


class TestPersistentChangeConfig:
    """P2 fixed-scene persistent change configuration tests."""

    def test_schema_default_disabled_for_compatibility(self):
        config = PersistentChangeConfig()
        assert config.enabled is False
        assert config.min_persistence_seconds == 2.0

    def test_default_yaml_enables_fixed_scene_head(self):
        config = load_config("configs/default.yaml")
        assert config.cameras[0].persistent_change.enabled is True
        assert config.cameras[0].persistent_change.alert_score == pytest.approx(0.82)

    def test_bounds_validation(self):
        with pytest.raises(ValidationError):
            PersistentChangeConfig(max_change_fraction=0.0)


class TestCandidateModelEvidenceConfig:
    def test_schema_default_is_backward_compatible(self):
        confirmation = StaticCandidateConfirmationConfig()

        assert confirmation.model_evidence.enabled is False
        assert confirmation.model_evidence.fusion_policy == "structure_only"

    def test_default_yaml_enables_shadow_collection_without_thresholds(self):
        config = load_config("configs/default.yaml")
        confirmation = config.cameras[0].static_confirmation

        assert confirmation.mode == "shadow"
        assert confirmation.model_evidence.enabled is True
        assert confirmation.model_evidence.fusion_policy == "model_required"
        assert confirmation.model_evidence.calibration_id is None

    def test_enforce_model_required_needs_bound_complete_calibration(self):
        with pytest.raises(ValidationError, match="calibration_id"):
            StaticCandidateConfirmationConfig(
                enabled=True,
                mode="enforce",
                model_evidence=CandidateModelEvidenceConfig(
                    enabled=True,
                    fusion_policy="model_required",
                ),
            )

    def test_enforce_model_required_cannot_rate_limit(self):
        with pytest.raises(ValidationError, match="cannot sample"):
            StaticCandidateConfirmationConfig(
                enabled=True,
                mode="enforce",
                model_evidence=CandidateModelEvidenceConfig(
                    enabled=True,
                    fusion_policy="model_required",
                    calibration_id="cal-1",
                    global_score_min=0.5,
                    roi_p95_min=0.5,
                    roi_context_delta_min=0.1,
                    sample_every_n_candidates=2,
                ),
            )

    def test_model_evidence_rejects_misspelled_fields(self):
        with pytest.raises(ValidationError, match="telemettry"):
            CandidateModelEvidenceConfig(telemettry=True)

    def test_static_confirmation_rejects_misspelled_model_evidence_block(self):
        with pytest.raises(ValidationError, match="model_evidnce"):
            StaticCandidateConfirmationConfig(model_evidnce={"enabled": True})

    def test_disabled_outer_gate_cannot_claim_model_required_enforcement(self):
        with pytest.raises(ValidationError, match="static_confirmation.enabled"):
            StaticCandidateConfirmationConfig(
                enabled=False,
                mode="enforce",
                model_evidence=CandidateModelEvidenceConfig(
                    enabled=True,
                    fusion_policy="model_required",
                    calibration_id="cal-1",
                    global_score_min=0.5,
                    roi_p95_min=0.5,
                    roi_context_delta_min=0.1,
                ),
            )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_model_thresholds_must_be_finite(self, value):
        with pytest.raises(ValidationError):
            CandidateModelEvidenceConfig(roi_context_delta_min=value)

    def test_enforce_requires_persistent_change_head(self):
        confirmation = StaticCandidateConfirmationConfig(
            enabled=True,
            mode="enforce",
        )

        with pytest.raises(ValidationError, match="persistent_change.enabled"):
            CameraConfig(
                camera_id="c1",
                name="test",
                source="test.mp4",
                persistent_change=PersistentChangeConfig(enabled=False),
                static_confirmation=confirmation,
            )

    def test_structure_only_enforce_requires_scene_guard_reference(self):
        confirmation = StaticCandidateConfirmationConfig(
            enabled=True,
            mode="enforce",
        )

        with pytest.raises(ValidationError, match="scene_guard.enabled"):
            CameraConfig(
                camera_id="c1",
                name="test",
                source="test.mp4",
                persistent_change=PersistentChangeConfig(enabled=True),
                scene_guard=SceneGuardConfig(enabled=False),
                static_confirmation=confirmation,
            )

    def test_model_required_enforce_requires_scene_guard(self):
        with pytest.raises(ValidationError, match="scene_guard.enabled"):
            _safe_model_required_camera(
                scene_guard=SceneGuardConfig(enabled=False),
            )

    def test_model_required_enforce_requires_immutable_profile_binding(self):
        with pytest.raises(ValidationError, match="require_profile_model_binding"):
            _safe_model_required_camera(
                scene_guard=SceneGuardConfig(
                    enabled=True,
                    require_profile_model_binding=False,
                ),
            )

    def test_model_required_enforce_rejects_alignable_scene_variant(self):
        with pytest.raises(ValidationError, match="allow_alignable=false"):
            _safe_model_required_camera(
                scene_guard=SceneGuardConfig(
                    enabled=True,
                    require_profile_model_binding=True,
                    allow_alignable=True,
                ),
            )

    def test_model_required_enforce_rejects_dynamic_low_light_clahe(self):
        with pytest.raises(ValidationError, match="low-light CLAHE"):
            _safe_model_required_camera(
                low_light=LowLightConfig(enabled=True, clahe_enabled=True),
            )

    def test_model_required_enforce_allows_low_light_without_clahe(self):
        camera = _safe_model_required_camera(
            low_light=LowLightConfig(enabled=True, clahe_enabled=False),
        )

        assert camera.low_light.enabled is True
        assert camera.low_light.clahe_enabled is False

    @pytest.mark.parametrize("skip_frame_on_person", [False, True])
    def test_model_required_enforce_accepts_either_person_policy(
        self,
        skip_frame_on_person,
    ):
        camera = _safe_model_required_camera(
            person_filter=PersonFilterConfig(
                skip_frame_on_person=skip_frame_on_person,
            ),
        )

        assert camera.person_filter.skip_frame_on_person is skip_frame_on_person

    def test_model_required_enforce_rejects_incompatible_ensemble(self):
        with pytest.raises(ValidationError, match="does not support anomaly ensemble"):
            _safe_model_required_camera(
                anomaly=AnomalyConfig(ensemble={"enabled": True}),
            )

    def test_model_required_enforce_rejects_shared_dinomaly_lifecycle(self):
        with pytest.raises(ValidationError, match="dinomaly_multi_class"):
            _safe_model_required_camera(
                anomaly=AnomalyConfig(
                    model_type="dinomaly2",
                    dinomaly_multi_class=True,
                ),
            )


class TestFastMotionConfig:
    """P5 high-speed projectile configuration tests."""

    def test_fragmented_streak_gap_default(self):
        config = FastMotionConfig()
        assert config.streak_gap_px == 4

    def test_default_yaml_enables_fragmented_streak_grouping(self):
        config = load_config("configs/default.yaml")
        fast_motion = config.cameras[0].fast_motion
        assert fast_motion.enabled is True
        assert fast_motion.streak_gap_px == 4

    def test_streak_gap_bounds_validation(self):
        with pytest.raises(ValidationError):
            FastMotionConfig(streak_gap_px=0)


class TestWaterLeakConfig:
    """P6 visible leak configuration tests."""

    def test_schema_default_disabled_for_compatibility(self):
        config = WaterLeakConfig()
        assert config.enabled is False
        assert config.alert_score == pytest.approx(0.84)

    def test_default_yaml_exposes_visible_leak_head_disabled(self):
        config = load_config("configs/default.yaml")
        water = config.cameras[0].water_leak
        assert water.enabled is False
        assert water.min_growth_area_px == 80

    def test_bounds_validation(self):
        with pytest.raises(ValidationError):
            WaterLeakConfig(max_change_fraction=0.0)


class TestPlumeConfig:
    """P7 visible plume configuration tests."""

    def test_schema_default_disabled_for_compatibility(self):
        config = PlumeConfig()
        assert config.enabled is False
        assert config.alert_score == pytest.approx(0.83)

    def test_default_yaml_exposes_visible_plume_head_disabled(self):
        config = load_config("configs/default.yaml")
        plume = config.cameras[0].plume
        assert plume.enabled is False
        assert plume.max_edge_density == pytest.approx(0.08)
        assert "visible_plume" in config.alerts.enabled_categories

    def test_bounds_validation(self):
        with pytest.raises(ValidationError):
            PlumeConfig(max_change_fraction=0.0)
