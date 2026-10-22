# Argus 源码索引

> 由 `python scripts/extract_source_index.py --output docs/source-index.md` 生成。
> 该文件用于压缩阅读上下文，写架构或模块文档前可先查看这里，再按需打开源码。

## 目录树

```text
src/argus/
- __init__.py
- __main__.py
  - alerts/__init__.py
  - alerts/calibration.py
  - alerts/dispatcher.py
  - alerts/evidence.py
  - alerts/feedback.py
  - alerts/grader.py
  - anomaly/__init__.py
  - anomaly/_trainer_export.py
  - anomaly/backbone_manager.py
  - anomaly/backbone_trainer.py
  - anomaly/baseline.py
  - anomaly/baseline_lifecycle.py
  - anomaly/candidate_model_evidence.py
  - anomaly/classifier.py
  - anomaly/dataset_selection.py
  - anomaly/detector.py
  - anomaly/drift.py
  - anomaly/ensemble.py
  - anomaly/job_executor.py
  - anomaly/metrics.py
  - anomaly/model_compare.py
  - anomaly/model_package.py
  - anomaly/model_router.py
  - anomaly/persistent_change.py
  - anomaly/plume.py
  - anomaly/quality.py
  - anomaly/segmenter.py
  - anomaly/shadow_runner.py
  - anomaly/static_candidate_confirmation.py
  - anomaly/trainer.py
  - anomaly/training_validator.py
  - anomaly/water_leak.py
  - camera/__init__.py
  - camera/models.py
  - camera/orchestrator.py
  - camera/planner.py
  - capture/__init__.py
  - capture/base.py
  - capture/baseline_job.py
  - capture/camera.py
  - capture/factory.py
  - capture/frame_buffer.py
  - capture/frame_filter.py
  - capture/gige_capture.py
  - capture/manager.py
  - capture/post_review.py
  - capture/quality.py
  - capture/samplers.py
  - config/__init__.py
  - config/loader.py
  - config/schema.py
  - contracts/__init__.py
  - contracts/validation.py
  - core/__init__.py
  - core/active_learning.py
  - core/adaptive_threshold.py
  - core/alert_ring_buffer.py
  - core/anomaly_postprocess.py
  - core/circuit_breaker.py
  - core/correlation.py
  - core/degradation.py
  - core/diagnostics.py
  - core/error_channel.py
  - core/event_bus.py
  - core/frame_identity.py
  - core/frame_quality.py
  - core/health.py
  - core/inference_queue.py
  - core/inference_record.py
  - core/metrics.py
  - core/model_discovery.py
  - core/model_status.py
  - core/pipeline.py
  - core/pipeline_mode_guard.py
  - core/runner.py
  - core/runtime.py
  - core/scheduler.py
  - core/temporal_tracker.py
  - core/video_encoder.py
  - core/zone_mask.py
  - dashboard/__init__.py
  - dashboard/api_contracts.py
  - dashboard/api_response.py
  - dashboard/app.py
  - dashboard/auth.py
  - dashboard/degradation_bridge.py
  - dashboard/forms.py
  - dashboard/model_runtime.py
    - dashboard/routes/__init__.py
    - dashboard/routes/alerts.py
    - dashboard/routes/audit.py
    - dashboard/routes/backup.py
    - dashboard/routes/baseline.py
    - dashboard/routes/calibration.py
    - dashboard/routes/cameras.py
    - dashboard/routes/config.py
    - dashboard/routes/degradation.py
    - dashboard/routes/labeling.py
    - dashboard/routes/me.py
    - dashboard/routes/models.py
    - dashboard/routes/physics.py
    - dashboard/routes/recordings.py
    - dashboard/routes/regions.py
    - dashboard/routes/replay.py
    - dashboard/routes/reports.py
    - dashboard/routes/sensors.py
    - dashboard/routes/streaming.py
    - dashboard/routes/system.py
    - dashboard/routes/tasks.py
    - dashboard/routes/training_jobs.py
    - dashboard/routes/users.py
    - dashboard/routes/zones.py
  - dashboard/tasks.py
  - dashboard/websocket.py
  - imaging/__init__.py
  - imaging/acquisition.py
  - imaging/fusion.py
  - imaging/polarization.py
  - person/__init__.py
  - person/detector.py
  - physics/__init__.py
  - physics/calibration.py
  - physics/multi_cam.py
  - physics/speed.py
  - physics/trajectory.py
  - prefilter/__init__.py
  - prefilter/fast_motion.py
  - prefilter/mog2.py
  - prefilter/simple_detector.py
  - preprocessing/__init__.py
  - preprocessing/alignment.py
  - preprocessing/scene_guard.py
  - preprocessing/scene_profile.py
  - runtime/__init__.py
  - runtime/dev_training.py
  - runtime/dev_video.py
  - runtime/logging_setup.py
  - runtime/training_job_wiring.py
  - sensors/__init__.py
  - sensors/fusion.py
  - storage/__init__.py
  - storage/alert_recording.py
  - storage/audit.py
  - storage/backup.py
  - storage/continuous_recorder.py
  - storage/database.py
  - storage/inference_buffer.py
  - storage/inference_store.py
  - storage/model_registry.py
  - storage/model_release_service.py
  - storage/models.py
  - storage/release_evidence.py
  - storage/release_pipeline.py
  - storage/retention.py
  - streaming/__init__.py
  - streaming/go2rtc_manager.py
  - streaming/preview_gateway.py
  - streaming/stream_registry.py
  - streaming/websocket_proxy.py
  - validation/__init__.py
  - validation/pipeline_video_eval.py
  - validation/recall_test.py
  - validation/strict_video_decoder.py
  - validation/synthetic.py
  - validation/video_eval.py

web/src/
- App.vue
    - api/__tests__/clientContract.test.ts
  - api/alerts.ts
  - api/baselines.ts
  - api/cameras.ts
  - api/client.ts
  - api/index.ts
  - api/labeling.ts
  - api/models.ts
  - api/physics.ts
  - api/regions.ts
  - api/replay.ts
  - api/reports.ts
  - api/system.ts
  - api/tasks.ts
  - api/training.ts
  - api/users.ts
  - api/zones.ts
  - assets/hero.png
  - assets/monitoring.png
  - assets/vite.svg
  - assets/vue.svg
  - components/AnnotationOverlay.vue
  - components/ContentSkeleton.vue
  - components/DegradationBar.vue
  - components/EmptyState.vue
  - components/ErrorBoundary.vue
  - components/ImageCompareSlider.vue
  - components/OverviewLiveStream.vue
  - components/ReplayPlayer.vue
  - components/SignalTrack.vue
  - components/Sparkline.vue
  - components/VideoTile.vue
  - components/VideoWall.vue
  - components/ZoneEditor.vue
    - components/__tests__/DegradationBar.test.ts
    - components/__tests__/ReplayPlayerState.test.ts
    - components/alerts/AlertDetailPanel.vue
    - components/alerts/AlertPhysicsPanel.vue
    - components/alerts/AlertsTable.vue
    - components/alerts/AlertsToolbar.vue
    - components/baseline/DatasetSelector.vue
    - components/calibration/CalibrationWizard.vue
    - components/common/AsyncState.vue
    - components/common/ConnectionTestResult.vue
    - components/common/DangerousActionConfirm.vue
    - components/common/ErrorDetailModal.vue
    - components/common/HealthBadge.vue
    - components/common/ModeBadge.vue
    - components/common/PageShell.vue
    - components/common/ResponsiveTable.vue
    - components/common/StatusBanner.vue
    - components/models/BackboneList.vue
    - components/models/BaselineCaptureModals.vue
    - components/models/BaselineImagesPanel.vue
    - components/models/BaselineTab.vue
    - components/models/BaselineTable.vue
    - components/models/BaselineVersionDrawer.vue
    - components/models/BatchInference.vue
    - components/models/ConfusionMatrix.vue
    - components/models/EventLog.vue
    - components/models/LabelingTab.vue
    - components/models/MetricsChart.vue
    - components/models/ModelCameraState.vue
    - components/models/ModelComparison.vue
    - components/models/ModelTable.vue
    - components/models/ModelsTab.vue
    - components/models/ThresholdPreview.vue
    - components/models/TrainingCreate.vue
    - components/models/TrainingHistory.vue
    - components/models/TrainingJobsList.vue
    - components/models/TrainingTab.vue
    - components/replay/ReplayCanvas.vue
    - components/replay/ReplayControls.vue
    - components/replay/ReplayReference.vue
    - components/replay/ReplaySignalTracks.vue
    - components/replay/StoryboardPlayer.vue
    - components/system/ClassifierPanel.vue
    - components/system/CrossCameraPanel.vue
    - components/system/ErrorCenterDrawer.vue
    - components/system/ImagingPanel.vue
    - components/system/ModelStatusPanel.vue
    - components/system/ModuleTogglePanel.vue
    - components/system/SegmenterPanel.vue
    - components/system/SystemAuditPanel.vue
    - components/system/SystemConfigPanel.vue
    - components/system/SystemDegradationPanel.vue
    - components/system/SystemOverviewPanel.vue
    - components/system/SystemRegionPanel.vue
    - components/system/SystemUserPanel.vue
    - components/trajectory/TrajectoryOverlay.vue
    - composables/__tests__/useGo2RTC.test.ts
    - composables/__tests__/useReplayController.test.ts
    - composables/__tests__/useStoryboardController.test.ts
    - composables/__tests__/useWebSocketContract.test.ts
  - composables/useCanvasCompositor.ts
  - composables/useConfigApplyToast.ts
  - composables/useGo2RTC.ts
  - composables/useHeatmapCache.ts
  - composables/useModelState.ts
  - composables/useReplayController.ts
  - composables/useStoryboardController.ts
  - composables/useSystemMode.ts
  - composables/useWebSocket.ts
  - config/constants.ts
    - constants/__tests__/roles.test.ts
  - constants/roles.ts
- env.d.ts
  - layouts/BasicLayout.vue
    - layouts/__tests__/BasicLayoutNavigation.test.ts
- main.ts
    - router/__tests__/routes.test.ts
  - router/index.ts
    - services/streaming/PlayerFactory.ts
    - services/streaming/StreamManager.ts
    - stores/__tests__/theme.test.ts
    - stores/__tests__/useAlertStore.test.ts
  - stores/theme.ts
  - stores/useAlertStore.ts
  - stores/useAuthStore.ts
  - stores/useBaselineStore.ts
  - stores/useErrorStore.ts
  - stores/useWallStore.ts
- style.css
  - types/api.ts
  - types/generated-openapi.ts
    - utils/__tests__/moduleToggleFeedback.test.ts
    - utils/__tests__/trainingJobPolling.test.ts
  - utils/colors.ts
  - utils/error.ts
  - utils/logger.ts
  - utils/moduleToggleFeedback.ts
  - utils/time.ts
  - utils/trainingJobPolling.ts
  - views/Alerts.vue
  - views/CameraDetail.vue
  - views/Cameras.vue
  - views/CollectionsView.vue
  - views/Models.vue
  - views/Overview.vue
  - views/ReplayView.vue
  - views/Reports.vue
  - views/StoryboardReplay.vue
  - views/System.vue
  - views/Warehouse3D.vue
    - views/__tests__/AlertsRealtimeFlow.test.ts
    - views/__tests__/CameraDetailStream.test.ts
    - views/__tests__/CamerasAddFlow.test.ts
    - views/__tests__/CollectionsRoleActions.test.ts
    - views/__tests__/OverviewDeepLink.test.ts
    - views/__tests__/ReplayViewRoute.test.ts
    - views/__tests__/ReportsEvidence.test.ts
    - views/__tests__/StoryboardReplayRoute.test.ts
    - views/models/BaselineView.vue
    - views/models/ComparisonView.vue
    - views/models/LabelingView.vue
    - views/models/ModelsRegistryView.vue
    - views/models/ThresholdView.vue
    - views/models/TrainingView.vue
    - views/system/AuditView.vue
    - views/system/ClassifierView.vue
    - views/system/ConfigView.vue
    - views/system/CrossCameraView.vue
    - views/system/DegradationView.vue
    - views/system/ImagingView.vue
    - views/system/ModelStatusView.vue
    - views/system/ModulesView.vue
    - views/system/OverviewView.vue
    - views/system/RegionsView.vue
    - views/system/SegmenterView.vue
    - views/system/UsersView.vue

configs/
- default.yaml

scripts/
- activate_scene_profile.py
- attest_soak_report.py
- build_eval_set_from_inference_records.py
- build_opencv_cuda.sh
- build_scene_profile_baseline.py
- capture_baseline.py
- create_dev_video.py
- create_scene_profile.py
- evaluate_regression.py
- evaluate_video.py
- export_model.py
- extract_source_index.py
- generate_openapi_types.py
- generate_synthetic.py
- inventory_regression_corpus.py
- jetson_setup_nvidia_docker.sh
- prepare_gpu_validation.py
- run_regression_suite.py
- run_trusted_soak.py
- setup_remote_sshkey.py
- simulate_camera.py
- smoke_core_loop.py
- smoke_cuda.py
- smoke_dashboard_business_flow.py
- smoke_dashboard_routes.py
- smoke_physical_scenarios.py
- train_model.py
- gpu_validation_runtime.sh

tests/
- conftest.py
  - integration/__init__.py
  - integration/conftest.py
  - integration/test_alert_integration.py
  - integration/test_alert_model_version.py
  - integration/test_camera_manager.py
  - integration/test_camera_orchestrator.py
  - integration/test_dashboard_api.py
  - integration/test_pipeline_integration.py
  - integration/test_runtime_imports.py
  - lint/__init__.py
  - lint/test_dashboard_camera_runtime_boundary.py
  - lint/test_frontend_core_no_any.py
  - lint/test_mjpeg_dedicated_executor.py
  - lint/test_model_release_entrypoint_closure.py
  - lint/test_model_runtime_artifact_authority.py
  - lint/test_no_base_http_middleware.py
  - lint/test_no_hardcoded_audit_operator.py
  - lint/test_openapi_types_contract.py
  - lint/test_orm_auto_migrate.py
  - lint/test_gpu_validation_runtime_safety.py
  - regression/.gitkeep
    - regression/anomaly/.gitkeep
    - regression/normal/.gitkeep
- test_alert_replay_integration.py
- test_frame_quality.py
  - unit/test_activate_scene_profile.py
  - unit/test_adaptive_threshold.py
  - unit/test_alert_evidence.py
  - unit/test_alert_grader.py
  - unit/test_alert_grader_evidence.py
  - unit/test_alert_recording.py
  - unit/test_alert_ring_buffer.py
  - unit/test_alert_workflow.py
  - unit/test_alerts_route.py
  - unit/test_alerts_route_evidence.py
  - unit/test_alignment.py
  - unit/test_anomaly_postprocess.py
  - unit/test_api_contract.py
  - unit/test_attest_soak_report.py
  - unit/test_auth.py
  - unit/test_backbone_manager.py
  - unit/test_backpressure.py
  - unit/test_backup_route.py
  - unit/test_baseline.py
  - unit/test_baseline_job.py
  - unit/test_baseline_lifecycle.py
  - unit/test_batch_inference.py
  - unit/test_build_scene_profile_baseline.py
  - unit/test_calibration.py
  - unit/test_camera.py
  - unit/test_camera_file_api_flow.py
  - unit/test_camera_groups.py
  - unit/test_camera_rbac.py
  - unit/test_camera_runtime_planner.py
  - unit/test_candidate_model_evidence.py
  - unit/test_circuit_breaker.py
  - unit/test_classifier.py
  - unit/test_config.py
  - unit/test_config_api.py
  - unit/test_config_loader.py
  - unit/test_core_api_dtos.py
  - unit/test_core_loop_api.py
  - unit/test_core_smoke_script.py
  - unit/test_correlation.py
  - unit/test_corroboration_suppression.py
  - unit/test_create_scene_profile.py
  - unit/test_cuda_smoke_script.py
  - unit/test_dashboard.py
  - unit/test_dashboard_business_smoke_script.py
  - unit/test_dashboard_route_smoke_script.py
  - unit/test_dashboard_streaming_contract.py
  - unit/test_data_contracts.py
  - unit/test_database.py
  - unit/test_database_alert_evidence_migration.py
  - unit/test_dataset_merger.py
  - unit/test_dataset_selection.py
  - unit/test_degradation.py
  - unit/test_degradation_bridge.py
  - unit/test_degradation_manager.py
  - unit/test_detector_model_evidence.py
  - unit/test_dev_training.py
  - unit/test_dev_video_source.py
  - unit/test_diagnostics.py
  - unit/test_dispatcher.py
  - unit/test_drift.py
  - unit/test_ensemble.py
  - unit/test_error_channel.py
  - unit/test_event_bus.py
  - unit/test_fast_motion.py
  - unit/test_feedback.py
  - unit/test_feedback_queue.py
  - unit/test_feedback_record.py
  - unit/test_fp_integration.py
  - unit/test_frame_filter.py
  - unit/test_frame_quality_assessor.py
  - unit/test_go2rtc_manager.py
  - unit/test_health.py
  - unit/test_hybrid_pipeline.py
  - unit/test_imaging.py
  - unit/test_inference_buffer.py
  - unit/test_inference_queue.py
  - unit/test_inference_record.py
  - unit/test_inference_record_candidate.py
  - unit/test_inventory_regression_corpus.py
  - unit/test_job_executor.py
  - unit/test_labeling_lineage.py
  - unit/test_low_light.py
  - unit/test_metrics.py
  - unit/test_model_backend_policy.py
  - unit/test_model_compare.py
  - unit/test_model_diversity.py
  - unit/test_model_package.py
  - unit/test_model_registry.py
  - unit/test_model_release_runtime.py
  - unit/test_model_release_service.py
  - unit/test_model_runtime.py
  - unit/test_model_status.py
  - unit/test_models_business_flow.py
  - unit/test_models_status_route.py
  - unit/test_multiscale.py
  - unit/test_permissions.py
  - unit/test_persistent_change.py
  - unit/test_person_detector.py
  - unit/test_physical_scenarios_smoke_script.py
  - unit/test_physics_calibration.py
  - unit/test_physics_runtime_contract.py
  - unit/test_physics_speed.py
  - unit/test_physics_trajectory.py
  - unit/test_pipeline_core.py
  - unit/test_pipeline_mode_guard.py
  - unit/test_pipeline_modes.py
  - unit/test_pipeline_reliability.py
  - unit/test_pipeline_scene_guard_integration.py
  - unit/test_pipeline_shadow_runner.py
  - unit/test_pipeline_video_eval.py
  - unit/test_plume.py
  - unit/test_post_review.py
  - unit/test_precision_recall_metrics.py
  - unit/test_prefilter.py
  - unit/test_prepare_gpu_validation.py
  - unit/test_preview_gateway_contract.py
  - unit/test_release_api_policy.py
  - unit/test_release_evidence.py
  - unit/test_release_pipeline.py
  - unit/test_replay_route.py
  - unit/test_reports_route.py
  - unit/test_run_regression_suite.py
  - unit/test_run_trusted_soak.py
  - unit/test_runner_frame_identity.py
  - unit/test_runtime_interfaces.py
  - unit/test_samplers.py
  - unit/test_scene_guard.py
  - unit/test_scene_profile.py
  - unit/test_scheduled_retraining.py
  - unit/test_scheduler.py
  - unit/test_segmenter.py
  - unit/test_sensor_fusion.py
  - unit/test_sensors_route.py
  - unit/test_shadow_runner.py
  - unit/test_simplex.py
  - unit/test_ssim_fallback.py
  - unit/test_startup_security.py
  - unit/test_static_candidate_confirmation.py
  - unit/test_storage_recorder.py
  - unit/test_storyboard.py
  - unit/test_stream_registry_contract.py
  - unit/test_stream_websocket_proxy.py
  - unit/test_strict_video_decoder.py
  - unit/test_synthetic.py
  - unit/test_system_degradation_api.py
  - unit/test_task_pause.py
  - unit/test_temporal_tracker.py
  - unit/test_trainer.py
  - unit/test_trainer_export.py
  - unit/test_training_history.py
  - unit/test_training_job_wiring.py
  - unit/test_training_quality.py
  - unit/test_video_encoder.py
  - unit/test_video_eval.py
  - unit/test_warmup.py
  - unit/test_water_leak.py
  - unit/test_websocket.py
  - unit/test_zone_mask.py
  - unit/test_zone_rbac.py

docs/
- acceptance.md
- api-reference.md
- architecture.md
  - archive/architecture-map.html
  - audits/system-integration-audit-2026-05-26.md
  - design/argus_ux_v2_enhancements_5.md
  - design/camera-scene-guard-and-video-recognition-2026-07-13.md
  - design/recovery-plan-2026-06-12.md
  - enable_checklists/auto_retraining.md
  - enable_checklists/classifier_yolo_world.md
  - enable_checklists/cross_camera_corroboration.md
  - enable_checklists/segmenter_sam2.md
- index.md
  - modules/README.md
  - modules/capture-streaming.md
  - modules/dashboard-frontend.md
  - modules/detection-pipeline.md
  - modules/models-training.md
  - modules/runtime-configuration.md
  - modules/storage-operations.md
- requirements.md
  - runbooks/gpu-validation-runtime.md
- source-index.md

```

## Python 类/函数签名

### src/argus/__main__.py
- L51: `def _is_loopback_dashboard_host(host: str | None) -> bool`
- L64: `def _dashboard_exposed_without_auth(config: object) -> bool`
- L76: `def _warn_if_dashboard_exposed_without_auth(config: object) -> None`
- L93: `def _dashboard_thread_alive(dashboard_thread: threading.Thread | None) -> bool`
- L98: `def _should_continue_main_loop(running: bool, manager: CameraManager, dashboard_thread: threading.Thread | None) -> bool`
- L107: `def _apply_dev_video_source(config, cameras, video_path: Path, motion: str='settle')`
- L158: `def main()`

### src/argus/alerts/calibration.py
- L20: `class CalibrationResult`
- L32: `class ConformalCalibrator`
  - L43: `def calibrate(self, normal_scores: np.ndarray, target_fprs: dict[str, float] | None=None) -> CalibrationResult`
  - L119: `def save(self, result: CalibrationResult, path: Path, sorted_scores: np.ndarray | None=None) -> None`
  - L144: `def load(self, path: Path) -> CalibrationResult | None`
  - L159: `def load_sorted_scores(path: Path) -> np.ndarray | None`

### src/argus/alerts/dispatcher.py
- L38: `class AlertDispatcher`
  - L50: `def __init__(self, config: AlertConfig, database: Database, alerts_dir: str | Path='data/alerts', on_alert: Callable[..., bool | None] | None=None, audio_config: AudioAlertConfig | None=None)`
  - L99: `def _alert_to_dict(alert: Alert, snapshot_path: str | None=None, heatmap_path: str | None=None, evidence_unavailable: bool=False) -> dict`
  - L200: `def set_websocket_broadcaster(self, broadcaster: Callable[..., bool | None] | None) -> None`
  - L216: `def dispatch(self, alert: Alert) -> None`
  - L338: `def _dispatch_database(self, alert: Alert, snapshot_path: str | None, heatmap_path: str | None, evidence_unavailable: bool=False) -> None`
  - L375: `def _dispatch_database_strict(self, alert: Alert, snapshot_path: str | None, heatmap_path: str | None, evidence_unavailable: bool=False) -> None`
  - L468: `def _publish_persisted_alert(self, alert: Alert, snapshot_path: str | None, heatmap_path: str | None, evidence_unavailable: bool) -> None`
  - L479: `def _broadcaster_accepts_metadata(broadcaster: Callable[..., object]) -> bool`
  - L493: `def _drain_alert_outbox(self, limit: int=100) -> int`
  - L542: `def _publish_audio_alert(self, payload: dict) -> None`
  - L566: `def _dispatch_with_retry(self, queue: Queue, send_fn: Callable[[dict], None], channel: str, max_retries: int=3) -> None`
  - L644: `def _db_worker(self) -> None`
  - L684: `def _webhook_worker(self) -> None`
  - L705: `def _record_dispatch_failure(self, alert: Alert, reason: str) -> None`
  - L720: `def _save_snapshot(self, alert: Alert) -> str | None`
  - L743: `def _annotate_snapshot(frame: np.ndarray, heatmap: np.ndarray | None, score: float) -> np.ndarray`
  - L773: `def _save_heatmap(self, alert: Alert) -> str | None`
  - L794: `def _check_disk_space(self, min_free_mb: int=500) -> bool`
  - L823: `def flush_db_queue(self, timeout: float=5.0) -> None`
  - L831: `def get_circuit_breaker_status(self) -> dict`
  - L835: `def close(self) -> None`
- L873: `def _date_folder(timestamp: float) -> str`

### src/argus/alerts/evidence.py
- L30: `class DetectionEvidence`
  - L53: `def supported(self) -> bool`
  - L57: `def to_dict(self) -> dict[str, Any]`
  - L82: `def from_dict(cls, data: dict[str, Any]) -> 'DetectionEvidence'`
  - L109: `def evidence_summary(self) -> str`

### src/argus/alerts/feedback.py
- L36: `class FeedbackManager`
  - L44: `def __init__(self, database: Database, baselines_dir: str | Path='data/baselines', alerts_dir: str | Path='data/alerts', config: FeedbackConfig | None=None)`
  - L59: `def submit_feedback(self, alert_id: str, feedback_type: str, camera_id: str, zone_id: str='default', category: str | None=None, notes: str | None=None, submitted_by: str='operator', model_version_id: str | None=None, anomaly_score: float | None=None, snapshot_path: str | None=None) -> FeedbackRecord`
  - L131: `def submit_passive_feedback(self, camera_id: str, zone_id: str='all', source: str=FeedbackSource.DRIFT, notes: str | None=None, model_version_id: str | None=None) -> FeedbackRecord`
  - L169: `def get_pending_for_training(self, camera_id: str | None=None, feedback_type: str | None=None) -> list[FeedbackRecord]`
  - L179: `def mark_batch_processed(self, feedback_ids: list[str], model_version_id: str) -> int`
  - L192: `def collect_confirmed_for_validation(self, camera_id: str, validation_dir: str | Path | None=None) -> int`
  - L239: `def merge_fp_into_baseline(self, camera_id: str, zone_id: str='default', baseline_manager=None, max_fp_images: int | None=None) -> dict`
  - L346: `def get_feedback_stats(self, camera_id: str | None=None) -> dict`
  - L372: `def _copy_snapshot_with_meta(self, src: Path, dst: Path, meta: dict, log_event: str, feedback_id: str) -> None`
  - L394: `def _copy_to_baseline(self, record: FeedbackRecord) -> None`
  - L427: `def _copy_to_validation(self, record: FeedbackRecord) -> None`

### src/argus/alerts/grader.py
- L51: `class AlertCategory(str, Enum)`
- L64: `class Alert`
- L123: `class DetectionType(str, Enum)`
- L136: `class CusumSnapshot`
- L151: `class GradeResult`
  - L162: `def to_dict(self) -> dict`
- L175: `class _AnomalyTracker`
- L186: `class _SpatialAlert`
- L208: `class AlertGrader`
  - L219: `def __init__(self, config: AlertConfig, node_id: str='edge', calibration_path: Path | None=None, sensor_fusion: SensorFusion | None=None)`
  - L260: `def _normalize_incident_key(incident_key: str | None) -> str | None`
  - L268: `def _tracking_detection_type(cls, detection_type: str, incident_key: str | None) -> str`
  - L294: `def _tracker_key(cls, zone_key: str, detection_type: str, incident_key: str | None=None) -> str`
  - L322: `def _camera_suppression_key(cls, camera_id: str, zone_key: str, detection_type: str, incident_key: str | None=None) -> str`
  - L352: `def evaluate(self, camera_id: str, zone_id: str, zone_priority: ZonePriority, anomaly_score: float, frame_number: int, frame: np.ndarray | None=None, anomaly_map: np.ndarray | None=None, detection_type: str='anomaly', detected_objects: list[dict] | None=None, classification_label: str | None=None, classification_confidence: float | None=None, continuity_map: np.ndarray | None=None, *, incident_key: str | None=None) -> Alert | None`
  - L668: `def evaluate_with_evidence(self, camera_id: str, zone_id: str, zone_priority: ZonePriority, anomaly_score: float, frame_number: int, frame: np.ndarray | None=None, anomaly_map: np.ndarray | None=None, detection_type: str='anomaly', detected_objects: list[dict] | None=None, classification_label: str | None=None, classification_confidence: float | None=None, continuity_map: np.ndarray | None=None, *, incident_key: str | None=None) -> GradeResult`
  - L818: `def _early_warning_corroborated(ew_cfg, detection_type: str, detected_objects: list[dict] | None, classification_label: str | None, classification_confidence: float | None) -> bool`
  - L860: `def _fire_early_warning(self, *, camera_id: str, zone_id: str, zone_key: str, adjusted_score: float, severity: AlertSeverity, frame_number: int, frame: np.ndarray | None, anomaly_map: np.ndarray | None, detection_type: str, detected_objects: list[dict] | None, classification_label: str | None, classification_confidence: float | None, now: float, incident_key: str | None) -> Alert | None`
  - L981: `def _incident_bbox(detection_type: str, detected_objects: list[dict] | None) -> tuple[float, float, float, float] | None`
  - L1004: `def _bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float`
  - L1018: `def _suppression_reason(self, *, camera_id: str, zone_key: str, detection_type: str, detected_objects: list[dict] | None, now: float, incident_key: str | None) -> str | None`
  - L1107: `def _record_spatial_alert(self, *, camera_id: str, zone_key: str, detection_type: str, detected_objects: list[dict] | None, now: float, incident_key: str | None) -> None`
  - L1132: `def _assign_event_group(self, zone_key: str, alert_id: str, alert_time: float) -> tuple[str, int]`
  - L1155: `def _score_to_severity(self, score: float) -> AlertSeverity | None`
  - L1168: `def decay_unobserved(self, camera_id: str, zone_id: str, observed_incidents: set[tuple[str, str | None]], *, decay_detection_types: set[str] | None=None) -> None`
  - L1238: `def reset(self, zone_key: str | None=None) -> None`
  - L1264: `def reset_camera_suppression(self, camera_id: str) -> None`
  - L1279: `def _tracker_to_snapshot(tracker: _AnomalyTracker) -> CusumSnapshot | None`
  - L1290: `def get_cusum_state(self, zone_key: str) -> CusumSnapshot | None`
  - L1313: `def get_all_cusum_states(self) -> dict[str, CusumSnapshot]`
  - L1323: `def prune_stale_trackers(self, max_age_seconds: float=3600.0) -> int`
  - L1375: `def save_evidence_state(self, path: Path) -> None`
  - L1405: `def load_evidence_state(self, path: Path) -> int`

### src/argus/anomaly/_trainer_export.py
- L30: `def _list_images_for_calibration(directory: Path) -> list[Path]`
- L40: `def build_calibration_dataset(ov_model, val_dir: Path | None, max_images: int=100) -> list[np.ndarray]`
- L99: `def assert_openvino_static(export_path: Path) -> None`
- L130: `def quantize_int8(export_path: Path, val_dir: Path | None=None, calibration_images: int=100) -> bool`
- L230: `def export_model(engine, model, export_format: str, export_path: str, image_size: int=256, quantization: str='fp16', val_dir: Path | None=None, calibration_images: int=100) -> dict[str, str]`

### src/argus/anomaly/backbone_manager.py
- L20: `class BackboneManager`
  - L30: `def __init__(self)`
  - L38: `def get_instance(cls) -> BackboneManager`
  - L48: `def reset(cls) -> None`
  - L52: `def load(self, backbone_path: Path, backbone_version: str) -> bool`
  - L87: `def upgrade(self, new_backbone_path: Path, new_version: str) -> bool`
  - L105: `def get_backbone(self)`
  - L111: `def version(self) -> str | None`
  - L115: `def is_loaded(self) -> bool`
  - L118: `def validate_head_compatibility(self, head_backbone_ref: str | None) -> None`
- L138: `class HeadDetector`
  - L145: `def __init__(self, head_path: Path, camera_id: str, backbone_ref: str | None=None, threshold: float=0.5)`
  - L160: `def load(self) -> bool`
  - L197: `def hot_reload(self, new_head_path: Path, backbone_ref: str | None=None) -> bool`
  - L231: `def is_loaded(self) -> bool`
  - L235: `def camera_id(self) -> str`
  - L239: `def backbone_ref(self) -> str | None`

### src/argus/anomaly/backbone_trainer.py
- L28: `class BackboneTrainingResult`
- L42: `class _BaselineImageDataset(Dataset)`
  - L45: `def __init__(self, image_paths: list[Path], image_size: int=224)`
  - L49: `def __len__(self) -> int`
  - L52: `def __getitem__(self, idx: int) -> torch.Tensor | None`
- L75: `class _SSLHead(nn.Module)`
  - L81: `def __init__(self, in_dim: int=768, out_dim: int=256, hidden_dim: int=2048)`
  - L91: `def forward(self, x: torch.Tensor) -> torch.Tensor`
- L95: `def _collect_baseline_images(camera_ids: list[str], baselines_dir: Path, max_per_camera: int=5000, max_total: int=50000) -> list[Path]`
- L130: `def _compute_hash(path: Path) -> str`
- L137: `def _compute_dataset_hash(image_paths: list[Path]) -> str`
- L154: `class BackboneTrainer`
  - L157: `def __init__(self, output_dir: Path | str='data/backbones')`
  - L161: `def train(self, camera_ids: list[str], baselines_dir: Path, backbone_type: str='dinov2_vitb14', image_size: int=224, epochs: int=5, lr: float=1e-05, batch_size: int=16, progress_callback: callable | None=None) -> BackboneTrainingResult`

### src/argus/anomaly/baseline.py
- L24: `class BaselineManager`
  - L39: `def __init__(self, baselines_dir: str | Path, lifecycle=None)`
  - L45: `def _resolve_current_version(base: Path) -> Path`
  - L67: `def _create_version_dir(base: Path) -> tuple[Path, str]`
  - L82: `def get_baseline_dir(self, camera_id: str, zone_id: str='default') -> Path`
  - L86: `def create_new_version(self, camera_id: str, zone_id: str='default') -> Path`
  - L94: `def set_current_version(self, camera_id: str, zone_id: str, version: str) -> None`
  - L104: `def save_frame(self, frame: np.ndarray, output_dir: Path, index: int) -> Path`
  - L131: `def count_images(self, camera_id: str, zone_id: str='default') -> int`
  - L138: `def resolve_dataset_dirs(self, selection) -> list[Path]`
  - L163: `def count_images_multi(dirs: list[Path]) -> int`
  - L185: `def _safe_filename(filename: str) -> bool`
  - L195: `def _version_dir(self, camera_id: str, version: str, zone_id: str='default') -> Path`
  - L199: `def _is_active_version(self, camera_id: str, version: str, zone_id: str='default') -> bool`
  - L206: `def list_images(self, camera_id: str, version: str, zone_id: str='default') -> list[dict]`
  - L238: `def delete_image(self, camera_id: str, version: str, filename: str, zone_id: str='default') -> bool`
  - L313: `def add_image_from_bytes(self, camera_id: str, version: str, data: bytes, ext: str, zone_id: str='default') -> str`
  - L391: `def get_all_baselines(self) -> list[dict]`
  - L418: `def diversity_select(self, image_dir: Path, target_count: int, feature_size: tuple[int, int]=(64, 64)) -> list[Path]`
  - L476: `def cleanup_old_versions(self, camera_id: str, zone_id: str='default', keep: int=3) -> int`
  - L506: `def get_group_baseline_dir(self, group_id: str, zone_id: str='default') -> Path`
  - L512: `def create_group_version(self, group_id: str, zone_id: str='default') -> Path`
  - L523: `def merge_camera_baselines(self, group_id: str, camera_ids: list[str], zone_id: str='default', target_count: int | None=None) -> Path`

### src/argus/anomaly/baseline_lifecycle.py
- L30: `class BaselineLifecycleError(Exception)`
- L34: `class BaselineLifecycle`
  - L37: `def __init__(self, database: Database, audit: AuditLogger | None=None)`
  - L41: `def register_version(self, camera_id: str, zone_id: str, version: str, image_count: int=0, group_id: str | None=None) -> BaselineVersionRecord`
  - L83: `def get_version(self, camera_id: str, zone_id: str, version: str) -> BaselineVersionRecord | None`
  - L95: `def get_versions(self, camera_id: str, zone_id: str='default') -> list[BaselineVersionRecord]`
  - L109: `def get_active_version(self, camera_id: str, zone_id: str='default') -> BaselineVersionRecord | None`
  - L121: `def get_eligible_versions(self, camera_id: str, zone_id: str='default', *, since: datetime | None=None) -> list[BaselineVersionRecord]`
  - L153: `def get_trainable_versions(self, camera_id: str, zone_id: str='default') -> list[BaselineVersionRecord]`
  - L173: `def delete_version(self, camera_id: str, zone_id: str, version: str, user: str='operator', ip_address: str='') -> bool`
  - L219: `def _transition(self, camera_id: str, zone_id: str, version: str, target_state: str, user: str, detail: str='', ip_address: str='', extra_updates: dict | None=None) -> BaselineVersionRecord`
  - L283: `def verify(self, camera_id: str, zone_id: str, version: str, verified_by: str, verified_by_secondary: str | None=None, ip_address: str='') -> BaselineVersionRecord`
  - L311: `def activate(self, camera_id: str, zone_id: str, version: str, user: str, ip_address: str='') -> BaselineVersionRecord`
  - L339: `def retire(self, camera_id: str, zone_id: str, version: str, user: str, reason: str='', ip_address: str='') -> BaselineVersionRecord`
  - L359: `def is_version_tracked(self, camera_id: str, zone_id: str, version: str) -> bool`

### src/argus/anomaly/candidate_model_evidence.py
- L24: `class CandidateModelThresholds`
  - L32: `def complete(self) -> bool`
- L44: `class CandidateModelEvidence`
  - L69: `def to_dict(self) -> dict`
- L74: `class PreparedModelEvidenceMap`
- L87: `def prepare_model_evidence_map(*, model_anomaly_map: np.ndarray | None, model_score: float | None, output_space: str | None, output_comparable: bool, detection_failed: bool=False) -> PreparedModelEvidenceMap`
- L173: `def build_calibration_id(binding: Mapping[str, object]) -> str`
- L187: `def evaluate_candidate_model_evidence(*, model_anomaly_map: np.ndarray | None, bbox: BBox, frame_shape: tuple[int, ...], model_score: float | None, output_space: str | None, output_comparable: bool, calibration_id: str | None, expected_calibration_id: str | None, thresholds: CandidateModelThresholds, context_ratio: float=0.5, hot_pixel_threshold: float=0.5, detection_failed: bool=False, prepared_map: PreparedModelEvidenceMap | None=None, model_input_variant: str | None=None) -> CandidateModelEvidence`
- L329: `class SpatialTemporalDecision`
  - L340: `def to_dict(self) -> dict`
- L345: `class _SpatialState`
- L355: `class SpatialTemporalConfirmer`
  - L358: `def __init__(self, *, required_passes: int=2, min_gap_seconds: float=0.4, max_gap_seconds: float=2.0, iou_threshold: float=0.5) -> None`
  - L381: `def reset(self) -> None`
  - L385: `def observe(self, *, source_track_id: int, bbox: BBox, now: float, observation_number: int, evaluated: bool, passed: bool | None, status: str, observed_this_frame: bool=True) -> SpatialTemporalDecision`
  - L536: `def finish_frame(self, *, observed_track_ids: set[int], observation_number: int) -> None`
  - L552: `def snapshot(self) -> list[dict]`
  - L567: `def _match(self, source_track_id: int, bbox: BBox, *, observation_number: int) -> tuple[int | None, _SpatialState | None, float | None]`
- L599: `def _map_bbox_to_array(bbox: BBox, frame_shape: tuple[int, ...], map_shape: tuple[int, int]) -> BBox | None`
- L624: `def _bbox_iou(first: BBox, second: BBox) -> float`
- L638: `def _finite_float_or_none(value: float | None) -> float | None`

### src/argus/anomaly/classifier.py
- L40: `class OpenVocabClassifier`
  - L50: `def __init__(self, model_name: str='yolov8s-worldv2.pt', vocabulary: list[str] | None=None)`
  - L60: `def load(self) -> None`
  - L89: `def vocabulary(self) -> list[str]`
  - L93: `def update_vocabulary(self, new_vocab: list[str]) -> None`
  - L107: `def add_labels(self, labels: list[str]) -> None`
  - L117: `def save_vocabulary(self, path: str | Path) -> None`
  - L124: `def load_vocabulary(self, path: str | Path) -> None`
  - L137: `def classify(self, frame, bbox: tuple[int, int, int, int] | None=None) -> tuple[str, float] | None`

### src/argus/anomaly/dataset_selection.py
- L21: `class DatasetSelectionItem`
  - L29: `def relative_path(self) -> Path`
  - L33: `def to_dict(self) -> dict`
  - L42: `def from_dict(cls, raw: dict) -> 'DatasetSelectionItem'`
- L61: `class DatasetSelection`
  - L67: `def __post_init__(self) -> None`
  - L81: `def camera_id(self) -> str`
  - L84: `def iter_dirs(self, baselines_root: Path) -> Iterator[Path]`
  - L89: `def version_summary(self) -> str`
  - L96: `def to_json(self) -> str`
  - L105: `def from_json(cls, raw: str | None) -> 'DatasetSelection | None'`
  - L115: `def from_payload(cls, raw: dict) -> 'DatasetSelection'`

### src/argus/anomaly/detector.py
- L24: `def _torch_module_device(module: object) -> object | None`
- L40: `def _move_tensor_to_module(tensor: object, module: object) -> object`
- L50: `class AnomalyResult`
- L70: `class DetectorStatus`
- L84: `class AnomalibDetector`
  - L94: `def __init__(self, model_path: str | Path | None=None, threshold: float=0.7, image_size: tuple[int, int]=(256, 256), ssim_baseline_frames: int=15, ssim_sensitivity: float=50.0, ssim_midpoint: float=0.015, ssim_global_change_suppress_fraction: float=0.04, enable_calibration: bool=True, inference_backend: str='auto', camera_id: str='unknown')`
  - L144: `def _infer_model_image_size(self, model_path: Path) -> tuple[int, int] | None`
  - L203: `def _sync_image_size_from_model(self, model_path: Path) -> None`
  - L217: `def load(self) -> bool`
  - L355: `def _try_load_torch(self, device: str='cpu') -> bool`
  - L385: `def _load_inferencer_for_backend(self, model_path: Path)`
  - L428: `def _check_minmax_normalization(self) -> None`
  - L481: `def calibrate_raw_scores(self, baseline_dir: Path | None=None) -> None`
  - L533: `def _normalize_raw_score(self, raw_val: float) -> float`
  - L543: `def _load_calibration(self) -> None`
  - L582: `def _apply_calibration(self, raw_score: float) -> float`
  - L603: `def predict(self, frame: np.ndarray, fused_tensor: np.ndarray | None=None) -> AnomalyResult`
  - L629: `def predict_batch(self, frames: list[np.ndarray]) -> list[AnomalyResult]`
  - L644: `def _predict_anomalib(self, frame: np.ndarray) -> AnomalyResult`
  - L780: `def _safe_result(self) -> AnomalyResult`
  - L794: `def _predict_ssim_fallback(self, frame: np.ndarray) -> AnomalyResult`
  - L1055: `def get_status(self) -> DetectorStatus`
  - L1082: `def _detect_quantization(self) -> bool`
  - L1111: `def hot_reload(self, new_model_path: Path) -> bool`
  - L1154: `def hot_reload_with_warmup(self, new_model_path: Path, expected_hash: str | None=None, warmup_frames: list[np.ndarray] | None=None, warmup_count: int=10, max_latency_ms: float=500.0, baseline_dir: Path | None=None, callback: 'threading.Event | None'=None) -> dict`
  - L1269: `def is_loaded(self) -> bool`
  - L1273: `def is_calibrated(self) -> bool`
- L1279: `class _TileInfo`
- L1288: `class MultiScaleDetector`
  - L1307: `def __init__(self, base_detector: AnomalibDetector, tile_size: int=512, tile_overlap: float=0.25, pyramid_mode: bool=False, pyramid_sizes: list[int] | None=None)`
  - L1328: `def is_loaded(self) -> bool`
  - L1332: `def is_calibrated(self) -> bool`
  - L1336: `def status(self) -> ModelStatus`
  - L1339: `def load(self) -> bool`
  - L1342: `def get_status(self) -> DetectorStatus`
  - L1345: `def hot_reload(self, new_model_path: Path) -> bool`
  - L1348: `def calibrate_raw_scores(self, baseline_dir: Path | None=None) -> None`
  - L1351: `def predict(self, frame: np.ndarray, fused_tensor: np.ndarray | None=None) -> AnomalyResult`
  - L1475: `def _generate_tiles(self, frame_h: int, frame_w: int, tile_size: int | None=None, stride: int | None=None) -> list[_TileInfo]`
  - L1515: `def _merge_heatmaps(self, frame_h: int, frame_w: int, tile_results: list[tuple[_TileInfo, AnomalyResult]], global_result: AnomalyResult) -> np.ndarray | None`
  - L1569: `def _merge_model_heatmaps(self, frame_h: int, frame_w: int, tile_results: list[tuple[_TileInfo, AnomalyResult]], global_result: AnomalyResult, *, output_space: str) -> np.ndarray | None`
  - L1630: `def _nms_suppress(self, heatmap: np.ndarray, iou_threshold: float=0.5) -> np.ndarray`

### src/argus/anomaly/drift.py
- L22: `class DriftStatus`
- L34: `class DriftDetector`
  - L44: `def __init__(self, reference_scores: np.ndarray | None=None, window_size: int=5000, check_interval: int=500, ks_threshold: float=0.1, p_value_threshold: float=0.01, on_drift_callback: callable | None=None)`
  - L64: `def set_reference(self, scores: np.ndarray) -> None`
  - L69: `def update(self, score: float) -> None`
  - L78: `def _run_check(self) -> None`
  - L123: `def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]`
  - L131: `def get_status(self) -> DriftStatus`

### src/argus/anomaly/ensemble.py
- L24: `class EnsembleConfig`
- L40: `class EnsembleResult`
- L54: `class DetectorEnsemble`
  - L67: `def __init__(self, config: EnsembleConfig) -> None`
  - L76: `def load(self) -> bool`
  - L110: `def predict(self, frame: np.ndarray) -> EnsembleResult`
  - L188: `def _fuse_mean(self, results: list[AnomalyResult]) -> tuple[float, bool]`
  - L194: `def _fuse_max(self, results: list[AnomalyResult]) -> tuple[float, bool]`
  - L199: `def _fuse_weighted(self, results: list[AnomalyResult]) -> tuple[float, bool]`
  - L215: `def _fuse_vote(self, results: list[AnomalyResult]) -> tuple[float, bool]`
  - L233: `def _fuse_bayesian(self, results: list[AnomalyResult]) -> tuple[float, bool]`
  - L266: `def _update_dynamic_weights(self, results: list[AnomalyResult]) -> None`
  - L282: `def _combine_heatmaps(self, results: list[AnomalyResult]) -> np.ndarray | None`
  - L307: `def is_loaded(self) -> bool`
  - L312: `def model_count(self) -> int`
  - L316: `def get_status(self) -> dict`

### src/argus/anomaly/job_executor.py
- L65: `def validate_hyperparameters(params: dict) -> list[str]`
- L96: `class TrainingJobExecutor`
  - L99: `def __init__(self, database: Database, trainer: ModelTrainer, backbone_trainer: BackboneTrainer | None=None, training_validator: TrainingValidator | None=None, model_packager: ModelPackager | None=None, model_registry: ModelRegistry | None=None, baselines_dir: Path=Path('data/baselines'), model_packages_dir: Path=Path('data/model_packages'), camera_manager: 'CameraManager | None'=None, model_release_service: 'ModelReleaseService | None'=None)`
  - L128: `def _training_mode_guard(self, reason: str)`
  - L144: `def recover_stale_jobs(self, max_running_hours: float=6.0) -> int`
  - L173: `def process_queued_jobs(self) -> int`
  - L192: `def execute(self, job_id: str) -> None`
  - L201: `def _execute_job(self, job) -> None`
  - L250: `def _execute_backbone(self, job) -> None`
  - L312: `def _execute_head(self, job) -> None`
  - L548: `def _mark_labeling_entries_trained(self, job, params: dict, model_version_id: str | None) -> None`
- L585: `def _coerce_int_list(value: object) -> list[int]`

### src/argus/anomaly/metrics.py
- L18: `def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict`
- L32: `def compute_precision_recall_f1(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict`
- L61: `def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float`
- L97: `def compute_pr_curve(y_true: np.ndarray, y_scores: np.ndarray) -> dict`
- L136: `def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float`
- L155: `def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray, target: Literal['f1', 'fpr_0.1', 'fpr_0.01', 'fpr_0.001']='f1') -> float`
- L201: `def evaluate_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict`

### src/argus/anomaly/model_compare.py
- L24: `class ComparisonResult`
- L57: `class ModelComparator`
  - L70: `def compare(self, model_a_path: Path, model_b_path: Path, val_dir: Path, image_size: tuple[int, int]=(256, 256), positive_dir: Path | None=None, negative_dir: Path | None=None, threshold: float=0.7) -> ComparisonResult`
  - L171: `def _load_images(self, val_dir: Path, image_size: tuple[int, int]) -> list[np.ndarray]`
  - L185: `def _benchmark(self, detector: AnomalibDetector, images: list[np.ndarray]) -> tuple[list[float], float]`
  - L202: `def _compute_stats(self, scores: list[float]) -> dict[str, float]`
  - L212: `def _evaluate_labeled(self, detector: AnomalibDetector, positive_dir: Path, negative_dir: Path, threshold: float) -> dict | None`
  - L231: `def _pick_winner(self, stats_a: dict[str, float], stats_b: dict[str, float], latency_a: float, latency_b: float, labeled_a: dict | None=None, labeled_b: dict | None=None) -> tuple[str, str]`

### src/argus/anomaly/model_package.py
- L37: `def _sha256(path: Path) -> str`
- L46: `def _generate_model_card(camera_id: str | None, model_type: str | None, job_type: str, backbone_ref: str, dataset_ref: str, validation_report: dict | None, training_params: dict | None, created_at: str) -> str`
- L105: `class ModelPackager`
  - L108: `def __init__(self, signing_key: str='')`
  - L111: `def package(self, output_dir: Path, model_dir: Path, export_dir: Path | None, training_job: dict | None=None, validation_report: dict | None=None, calibration_path: Path | None=None, backbone_ref: str='pretrained', dataset_ref: str='', camera_id: str | None=None, model_type: str | None=None, training_params: dict | None=None, calibration_image_ids: list[str] | None=None) -> Path`
  - L184: `def _assemble(self, pkg_dir: Path, model_dir: Path, export_dir: Path | None, training_job: dict | None, validation_report: dict | None, calibration_path: Path | None, backbone_ref: str, dataset_ref: str, camera_id: str | None, model_type: str | None, training_params: dict | None, calibration_image_ids: list[str] | None) -> None`
  - L314: `def verify(self, package_dir: Path) -> bool`

### src/argus/anomaly/model_router.py
- L19: `class ModelRouter`
  - L22: `def __init__(self, session_factory)`
  - L25: `def get_model_for_camera(self, camera_id: str) -> ModelRecord | None`
  - L77: `def get_shadow_model_for_camera(self, camera_id: str) -> ModelRecord | None`
  - L90: `def get_model_path(self, camera_id: str) -> Path | None`
  - L97: `def has_model_records(self, camera_id: str) -> bool`
  - L106: `def is_canary(self, camera_id: str) -> bool`

### src/argus/anomaly/persistent_change.py
- L20: `class PersistentChangeCandidate`
  - L47: `def to_detected_object(self) -> dict`
  - L58: `def to_dict(self) -> dict`
- L76: `class PersistentChangeResult`
  - L89: `def evidence(self) -> dict`
- L105: `class _TrackState`
- L120: `class PersistentChangeScorer`
  - L123: `def __init__(self, *, process_width: int=640, reference_frames: int=15, diff_threshold: int=28, min_area_px: int=250, max_area_fraction: float=0.2, max_change_fraction: float=0.12, min_persistence_seconds: float=2.0, min_persistence_frames: int=3, learning_alpha: float=0.002, morph_kernel_size: int=5, match_radius_px: int=40, max_missed_frames: int=3, alert_score: float=0.82, max_candidates_per_frame: int=5) -> None`
  - L164: `def is_ready(self) -> bool`
  - L167: `def reset_tracks(self) -> None`
  - L170: `def reset(self, *, clear_reference: bool=False) -> None`
  - L176: `def set_reference(self, frame: np.ndarray) -> None`
  - L182: `def process(self, frame: np.ndarray, *, timestamp: float | None=None) -> PersistentChangeResult`
  - L249: `def _prepare_gray(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]`
  - L260: `def _extract_regions(self, mask: np.ndarray, diff: np.ndarray, frame_shape: tuple[int, ...], scale_x: float, scale_y: float) -> list[dict]`
  - L323: `def _update_tracks(self, regions: list[dict], timestamp: float) -> list[PersistentChangeCandidate]`
  - L376: `def _candidate_from_track(self, track: _TrackState) -> PersistentChangeCandidate`
  - L404: `def _update_reference(self, gray: np.ndarray, mask: np.ndarray | None) -> None`
  - L420: `def _build_anomaly_map(mask: np.ndarray, score: float) -> np.ndarray`

### src/argus/anomaly/plume.py
- L19: `class PlumeCandidate`
  - L40: `def to_detected_object(self) -> dict`
  - L55: `def to_dict(self) -> dict`
- L78: `class PlumeResult`
  - L92: `def evidence(self) -> dict`
- L109: `class _Region`
- L123: `class _TrackState`
- L144: `class PlumeHead`
  - L147: `def __init__(self, *, process_width: int=640, reference_frames: int=15, diff_threshold: int=16, brightening_threshold: int=12, darkening_threshold: int=12, max_saturation: int=150, min_area_px: int=180, max_area_fraction: float=0.25, max_change_fraction: float=0.18, min_growth_area_px: int=100, min_persistence_seconds: float=1.0, min_persistence_frames: int=3, max_edge_density: float=0.08, max_bbox_fill_fraction: float=0.9, learning_alpha: float=0.001, morph_kernel_size: int=7, match_radius_px: int=70, max_missed_frames: int=6, alert_score: float=0.83, max_candidates_per_frame: int=5) -> None`
  - L198: `def is_ready(self) -> bool`
  - L201: `def reset_tracks(self) -> None`
  - L204: `def reset(self, *, clear_reference: bool=False) -> None`
  - L210: `def set_reference(self, frame: np.ndarray) -> None`
  - L216: `def process(self, frame: np.ndarray, *, timestamp: float | None=None) -> PlumeResult`
  - L320: `def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]`
  - L341: `def _extract_regions(self, mask: np.ndarray, absdiff: np.ndarray, signed: np.ndarray, gray: np.ndarray, hsv: np.ndarray, frame_shape: tuple[int, ...], scale_x: float, scale_y: float) -> tuple[list[_Region], int]`
  - L402: `def _update_tracks(self, regions: list[_Region], timestamp: float) -> list[PlumeCandidate]`
  - L476: `def _candidate_from_track(self, track: _TrackState, timestamp: float) -> PlumeCandidate`
  - L515: `def _update_reference(self, gray: np.ndarray, mask: np.ndarray | None) -> None`
  - L531: `def _build_anomaly_map(mask: np.ndarray, frame_shape: tuple[int, ...], score: float) -> np.ndarray`
  - L540: `def _classify_plume(self, mean_signed_delta: float, mean_value: float) -> str`

### src/argus/anomaly/quality.py
- L27: `def _list_images(directory: Path) -> list[Path]`
- L42: `class ValidationResult`
- L55: `class TrainingValidator`
  - L64: `def validate_baseline(self, image_dir: Path) -> ValidationResult`
  - L125: `def _check_corruption(self, images: list[Path]) -> tuple[int, list[np.ndarray], list[float]]`
  - L144: `def _check_duplicates(self, frames: list[np.ndarray]) -> float`
- L189: `class SplitResult`
- L199: `class DatasetSplitter`
  - L202: `def split(self, image_dir: Path, output_dir: Path, val_ratio: float=0.2, seed: int=42) -> SplitResult`
- L250: `def recommend_threshold(scores: list[float]) -> float`
- L273: `class ModelQualityReport`
- L289: `def _compute_grade(score_max: float, score_std: float, threshold: float) -> str`
- L314: `class PostTrainingValidator`
  - L317: `def validate_model(self, model_path: Path, val_dir: Path, threshold: float=0.7, train_count: int=0) -> ModelQualityReport`
- L407: `class OutputValidationResult`
- L417: `class OutputValidator`
  - L423: `def validate_output(self, output_dir: Path) -> OutputValidationResult`

### src/argus/anomaly/segmenter.py
- L24: `class SegmentedObject`
- L35: `class SegmentationResult`
- L47: `def extract_peak_points(anomaly_map: np.ndarray, max_points: int=5, min_score: float=0.7, min_distance_px: int=30) -> list[tuple[int, int]]`
- L111: `class InstanceSegmenter`
  - L118: `def __init__(self, model_size: str='small', min_mask_area_px: int=100, timeout_seconds: float=10.0)`
  - L136: `def update_runtime_params(self, *, min_mask_area_px: int | None=None, timeout_seconds: float | None=None) -> None`
  - L164: `def load(self) -> None`
  - L213: `def segment(self, frame: np.ndarray, prompt_points: list[tuple[int, int]]) -> SegmentationResult`
  - L246: `def _segment_sam2_with_timeout(self, frame: np.ndarray, prompt_points: list[tuple[int, int]]) -> SegmentationResult`
  - L268: `def shutdown(self) -> None`
  - L278: `def _segment_sam2(self, frame: np.ndarray, prompt_points: list[tuple[int, int]]) -> SegmentationResult`
  - L322: `def _segment_contour_fallback(self, frame: np.ndarray, prompt_points: list[tuple[int, int]]) -> SegmentationResult`
  - L386: `def _mask_to_object(self, mask: np.ndarray, frame_h: int, frame_w: int, confidence: float) -> SegmentedObject | None`

### src/argus/anomaly/shadow_runner.py
- L24: `class ShadowRunner`
  - L27: `def __init__(self, shadow_model_path: Path, shadow_version_id: str, production_version_id: str | None, camera_id: str, session_factory, sample_rate: int=5, threshold: float=0.5, inference_backend: str='auto')`
  - L56: `def _ensure_loaded(self) -> bool`
  - L98: `def prepare(self) -> bool`
  - L103: `def run_shadow(self, frame: np.ndarray, production_score: float | None=None, production_alerted: bool=False) -> None`
  - L155: `def flush(self) -> None`
  - L164: `def _write_logs(self, logs: list[ShadowInferenceLog]) -> None`
  - L179: `def shadow_version_id(self) -> str`
  - L183: `def frame_count(self) -> int`

### src/argus/anomaly/static_candidate_confirmation.py
- L24: `class ConfirmationReason(str, Enum)`
- L37: `class StaticCandidateConfirmationConfig`
- L68: `class StaticCandidateMetrics`
- L91: `class StaticCandidateDecision`
- L103: `class StaticCandidateConfirmer`
  - L106: `def __init__(self, config: StaticCandidateConfirmationConfig | None=None) -> None`
  - L109: `def confirm(self, current_frame: NDArray[np.generic], reference_frame: NDArray[np.generic], bbox: BBox, *, sensitive_rois: Sequence[BBox] | None=None, external_anomaly_score: float | None=None) -> StaticCandidateDecision`
  - L283: `def _structure_metrics(self, reference: NDArray[np.float32], current: NDArray[np.float32]) -> tuple[float, float, float, float]`
  - L346: `def _is_global_illumination(self, metrics: StaticCandidateMetrics, required_structure_score: float) -> bool`
- L362: `def _to_gray_float(frame: NDArray[np.generic]) -> NDArray[np.float32]`
- L388: `def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox`
- L403: `def _intersection_over_candidate(candidate: BBox, roi: BBox) -> float`
- L415: `def _overlaps_sensitive_roi(candidate: BBox, rois: Sequence[BBox], width: int, height: int, threshold: float) -> bool`
- L440: `def _fit_global_photometric_model(reference: NDArray[np.float32], current: NDArray[np.float32]) -> tuple[float, float]`
- L470: `def _safe_correlation(first: NDArray[np.float32], second: NDArray[np.float32]) -> float`

### src/argus/anomaly/trainer.py
- L57: `class _ManifestSample`
- L62: `def _adjacent_baseline_manifest(directory: Path) -> Path`
- L66: `class _DatasetMerger`
  - L79: `def __init__(self, items: Sequence[tuple[str, Path]]) -> None`
  - L87: `def __enter__(self) -> Path`
  - L140: `def __exit__(self, exc_type, exc, tb) -> None`
- L145: `def _build_merger_items(selection: 'DatasetSelection', resolved_dirs: list[Path]) -> list[tuple[str, Path]]`
- L219: `def _resolve_dinomaly_backbone(backbone: str) -> str`
- L228: `class TrainingStatus(str, Enum)`
- L240: `class QualityReport`
- L250: `class TrainingResult`
- L269: `def _list_images(directory: Path) -> list[Path]`
- L274: `def _sha256_file(path: Path) -> str`
- L282: `def _is_sha256(value: object) -> bool`
- L290: `def _role_counts(value: object, *, field: str) -> dict[str, int]`
- L302: `def _scene_profile_manifest_samples(baseline_dir: Path, *, expected_camera_id: str | None) -> tuple[list[_ManifestSample], list[_ManifestSample]] | None`
- L520: `class ModelTrainer`
  - L527: `def __init__(self, baseline_manager: BaselineManager, models_dir: str | Path='data/models', exports_dir: str | Path='data/exports', model_registry=None, training_validator=None, validation_config: dict | None=None)`
  - L546: `def exports_dir(self) -> Path`
  - L550: `def status(self) -> TrainingStatus`
  - L554: `def last_result(self) -> TrainingResult | None`
  - L558: `def _free_gpu_for_training() -> None`
  - L597: `def _restore_gpu_after_training() -> None`
  - L611: `def _fail(self, start: float, **kwargs) -> TrainingResult`
  - L619: `def train(self, camera_id: str, zone_id: str='default', model_type: str='patchcore', image_size: int=256, export_format: str | None='openvino', quantization: str='fp16', calibration_images: int=100, progress_callback: callable | None=None, anomaly_config: object | None=None, resume_from: str | None=None, group_id: str | None=None, backbone_checkpoint: str | None=None, skip_baseline_validation: bool=False, baseline_dir_override: Path | None=None, image_count_override: int | None=None, baseline_versions_label: str | None=None) -> TrainingResult`
  - L1050: `def incremental_train(self, camera_id: str, zone_id: str='default', model_type: str='patchcore', labeled_entries: list | None=None, database: object | None=None, progress_callback: callable | None=None, anomaly_config: object | None=None) -> TrainingResult`
  - L1157: `def _validate_baseline_quality(self, baseline_dir: Path, min_images: int=MIN_BASELINE_IMAGES) -> dict`
  - L1236: `def _split_train_val(baseline_dir: Path, output_dir: Path, seed: int=42, *, expected_camera_id: str | None=None) -> tuple[Path, Path, int, int]`
  - L1319: `def _validate_on_val_set(self, val_dir: Path, *, detector=None) -> dict`
  - L1366: `def _generate_quality_report(val_stats: dict, threshold_recommended: float) -> QualityReport`
  - L1414: `def _recommend_threshold(val_stats: dict) -> float`
  - L1433: `def _validate_output(self, output_dir: Path, export_path: Path | None, val_dir: Path) -> tuple[dict, object | None]`
  - L1503: `def _sync_calibration_to_export(calibration_path: Path, export_path: Path | None) -> list[Path]`
  - L1540: `def compare_models(self, old_model_path: str | Path, new_model_path: str | Path, val_dir: str | Path) -> dict`
  - L1620: `def _find_best_model_file(model_dir: Path) -> Path | None`
  - L1631: `def _train_anomalib(self, data_dir: Path, output_dir: Path, model_type: str, image_size: int, anomaly_config: object | None=None, resume_from: str | None=None, backbone_checkpoint: str | None=None) -> tuple`
  - L1836: `def _find_checkpoint(model_dir: Path) -> Path | None`
  - L1846: `def _instantiate_model(model_type: str)`
  - L1876: `def reexport_model(self, model_dir: Path, export_format: str='openvino', quantization: str='fp16', model_type: str='patchcore', image_size: int=256, calibration_images: int=100, camera_id: str | None=None, zone_id: str | None=None) -> dict`
  - L1970: `def recalibrate_model(self, model_dir: Path, baseline_dir: Path, camera_id: str, zone_id: str='default') -> dict`

### src/argus/anomaly/training_validator.py
- L92: `class StepResult`
- L106: `class ValidationReport`
  - L133: `def to_dict(self) -> dict`
  - L162: `def to_json(self) -> str`
- L166: `def _compute_auroc(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> float`
- L195: `class TrainingValidator`
  - L198: `def __init__(self, database: Database | None=None, foe_objects_dir: Path | None=None)`
  - L206: `def validate(self, detector, camera_id: str, zone_id: str, val_dir: Path, baseline_dir: Path, threshold: float, auroc_threshold: float=0.99, recall_threshold: float=0.95, replay_days: int=30) -> ValidationReport`
  - L348: `def _score_images(detector, image_paths: list[Path]) -> list[float]`
  - L362: `def _validate_auroc(self, detector, val_dir: Path, baseline_dir: Path, auroc_threshold: float, synthetic_dir: Path | None=None) -> StepResult`
  - L433: `def _score_noise_injected(detector, val_images: list[Path], count: int=50) -> list[float]`
  - L456: `def _validate_synthetic_recall(self, detector, threshold: float, recall_threshold: float, synthetic_dir: Path | None=None) -> StepResult`
  - L520: `def _validate_historical_replay(self, detector, camera_id: str, threshold: float, replay_days: int) -> StepResult`
  - L660: `def _validate_real_labeled(self, detector, camera_id: str, threshold: float, min_f1: float=0.0) -> StepResult`

### src/argus/anomaly/water_leak.py
- L20: `class WaterLeakCandidate`
  - L39: `def to_detected_object(self) -> dict`
  - L54: `def to_dict(self) -> dict`
- L75: `class WaterLeakResult`
  - L88: `def evidence(self) -> dict`
- L104: `class _Region`
- L116: `class _TrackState`
- L135: `class WaterLeakHead`
  - L138: `def __init__(self, *, process_width: int=640, reference_frames: int=15, diff_threshold: int=22, darkening_threshold: int=14, brightening_threshold: int=18, specular_value_threshold: int=180, specular_saturation_max: int=80, min_area_px: int=120, max_area_fraction: float=0.12, max_change_fraction: float=0.08, min_growth_area_px: int=80, min_persistence_seconds: float=1.0, min_persistence_frames: int=3, min_drip_length_px: int=18, min_vertical_aspect: float=2.0, learning_alpha: float=0.001, morph_kernel_size: int=5, match_radius_px: int=50, max_missed_frames: int=5, alert_score: float=0.84, max_candidates_per_frame: int=5) -> None`
  - L193: `def is_ready(self) -> bool`
  - L196: `def reset_tracks(self) -> None`
  - L199: `def reset(self, *, clear_reference: bool=False) -> None`
  - L205: `def set_reference(self, frame: np.ndarray) -> None`
  - L211: `def process(self, frame: np.ndarray, *, timestamp: float | None=None) -> WaterLeakResult`
  - L300: `def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]`
  - L317: `def _extract_regions(self, mask: np.ndarray, absdiff: np.ndarray, signed: np.ndarray, frame_shape: tuple[int, ...], scale_x: float, scale_y: float) -> list[_Region]`
  - L369: `def _update_tracks(self, regions: list[_Region], timestamp: float) -> list[WaterLeakCandidate]`
  - L439: `def _candidate_from_track(self, track: _TrackState, timestamp: float) -> WaterLeakCandidate`
  - L474: `def _update_reference(self, gray: np.ndarray, mask: np.ndarray | None) -> None`
  - L489: `def _build_anomaly_map(self, mask: np.ndarray, frame_shape: tuple[int, ...], score: float) -> np.ndarray`

### src/argus/camera/models.py
- L16: `class DetectionInput`
- L27: `class PreviewInput`
- L36: `class SnapshotInput`
- L44: `class Go2RTCStreamSpec`
- L55: `class CameraRuntimePlan`

### src/argus/camera/orchestrator.py
- L13: `class CameraOrchestrator`
  - L16: `def __init__(self, camera_manager: Any, stream_registry: Any | None=None) -> None`
  - L20: `def start(self, camera_id: str) -> bool`
  - L28: `def stop(self, camera_id: str) -> None`
  - L32: `def restart(self, camera_id: str) -> bool`
  - L37: `def delete(self, camera_id: str) -> bool`
  - L50: `def reconcile_streams(self) -> dict[str, Any]`
  - L56: `def runtime_camera_config(self, camera_config: Any) -> Any`
  - L70: `def _ensure_registered(self, camera_config: Any) -> Any | None`
  - L91: `def _require_usb_redirect(camera_config: Any, resolution: Any | None) -> None`
  - L99: `def _camera_config(self, camera_id: str) -> Any | None`
  - L119: `def _camera_configs(self) -> list[Any]`

### src/argus/camera/planner.py
- L18: `class CameraRuntimePlanner`
  - L25: `def build(cls, config: CameraConfig, *, go2rtc_rtsp_host: str=DEFAULT_GO2RTC_RTSP_HOST, go2rtc_rtsp_port: int=DEFAULT_GO2RTC_RTSP_PORT) -> CameraRuntimePlan`
  - L69: `def _go2rtc_stream(cls, *, config: CameraConfig, camera_id: str, source: str, protocol: str, capture_script: str | None, rtsp_host: str, rtsp_port: int) -> Go2RTCStreamSpec | None`
  - L126: `def _detection_input(*, source: str, protocol: str, stream: Go2RTCStreamSpec | None) -> DetectionInput`
  - L153: `def _preview_input(*, camera_id: str, stream: Go2RTCStreamSpec | None) -> PreviewInput`
  - L171: `def _go2rtc_rtsp_url(camera_id: str, host: str, port: int) -> str`

### src/argus/capture/base.py
- L11: `class CaptureAdapter(Protocol)`
  - L19: `def state(self) -> CameraState`
  - L23: `def connect(self) -> bool`
  - L27: `def read(self) -> FrameData | None`
  - L31: `def read_latest(self) -> FrameData | None`
  - L35: `def start_capture_thread(self) -> None`
  - L39: `def request_reconnect(self) -> None`
  - L43: `def stop(self) -> None`

### src/argus/capture/baseline_job.py
- L41: `class BaselineCaptureJobConfig`
- L73: `class SamplerSelection`
- L79: `def run_baseline_capture_job(progress_callback: Callable[[int, str], None], *, job_config: BaselineCaptureJobConfig, camera_manager: Any, pause_event: Any, abort_event: Any, lifecycle: Any | None=None) -> dict`
- L483: `def _create_sampler(job_config: BaselineCaptureJobConfig) -> SamplerSelection`
- L552: `def _write_manifest(version_dir: Path) -> Path`
- L563: `def _write_stats(version_dir: Path, stats: CaptureStats, collected: int, start_time: float) -> Path`
- L603: `def _distribution(values: list[float]) -> dict`
- L616: `def _write_job_config(version_dir: Path, job_config: BaselineCaptureJobConfig) -> Path`
- L635: `def _write_capture_meta(version_dir: Path, job_config: BaselineCaptureJobConfig, stats: CaptureStats, *, error: str | None=None, status: str='ok') -> Path`

### src/argus/capture/camera.py
- L32: `class FrameData`
- L45: `class CameraState`
- L65: `class CameraCapture`
  - L72: `def __init__(self, camera_id: str, source: str, protocol: str='rtsp', fps_target: int=5, resolution: tuple[int, int]=(1920, 1080), reconnect_delay: float=5.0, max_reconnect_attempts: int=-1, usb_backend: str='auto', usb_pixel_format: str='auto', usb_min_runtime_fps: float=0.0, source_declared: str | None=None, source_runtime: str | None=None, stream_id: str | None=None)`
  - L116: `def state(self) -> CameraState`
  - L119: `def configure_frame_identity(self, *, source_declared: str | None=None, source_runtime: str | None=None, stream_id: str | None=None) -> None`
  - L133: `def _usb_backend_candidates(self) -> list[tuple[int | None, str]]`
  - L163: `def _normalise_fourcc(pixel_format: str | None) -> str | None`
  - L176: `def _decode_fourcc(value: float | int) -> str | None`
  - L185: `def _capture_get(self, prop: int) -> float | None`
  - L194: `def _measure_usb_read_fps(self, *, warmup_seconds: float=0.5, sample_seconds: float=2.0) -> float | None`
  - L231: `def _refresh_runtime_state(self, backend_name: str | None) -> None`
  - L278: `def runtime_status(self) -> dict`
  - L293: `def _open_capture(self) -> tuple[cv2.VideoCapture | None, str | None]`
  - L320: `def connect(self) -> bool`
  - L377: `def read(self) -> FrameData | None`
  - L456: `def start_capture_thread(self) -> None`
  - L477: `def _capture_loop(self) -> None`
  - L487: `def read_latest(self) -> FrameData | None`
  - L497: `def request_reconnect(self) -> None`
  - L515: `def _reconnect_background(self) -> None`
  - L558: `def reconnect(self) -> bool`
  - L599: `def release(self) -> None`
  - L606: `def stop(self) -> None`
  - L611: `def __enter__(self)`
  - L615: `def __exit__(self, *args)`

### src/argus/capture/factory.py
- L11: `class CaptureFactory`
  - L15: `def create(camera_config_or_detection_input: CameraConfig | CameraRuntimePlan | DetectionInput, camera_config: CameraConfig | None=None) -> CaptureAdapter`
  - L41: `def _from_config(camera_config: CameraConfig) -> CaptureAdapter`
  - L54: `def _from_detection_input(detection: DetectionInput, camera_config: CameraConfig) -> CaptureAdapter`
  - L72: `def _create_opencv(*, camera_config: CameraConfig, source: str, protocol: str) -> CaptureAdapter`
  - L97: `def _create_gige(*, camera_config: CameraConfig, source: str) -> CaptureAdapter`

### src/argus/capture/frame_buffer.py
- L24: `class LatestFrameBuffer`
  - L39: `def __init__(self) -> None`
  - L45: `def put(self, frame: FrameData) -> None`
  - L53: `def get(self, timeout: float=5.0) -> FrameData | None`
  - L67: `def frames_replaced(self) -> int`

### src/argus/capture/frame_filter.py
- L23: `class FilterResult`
- L36: `class FilterConfig`
- L64: `class FrameFilter`
  - L73: `def __init__(self, config: FilterConfig | None=None, person_detector: object | None=None) -> None`
  - L88: `def evaluate(self, frame: np.ndarray | None) -> FilterResult`
  - L123: `def reset(self) -> None`
  - L133: `def _check_entropy(self, gray: np.ndarray, result: FilterResult) -> bool`
  - L155: `def _check_exposure(self, gray: np.ndarray, result: FilterResult) -> bool`
  - L188: `def _check_blur(self, gray: np.ndarray, result: FilterResult) -> bool`
  - L209: `def _check_persons(self, frame: np.ndarray, result: FilterResult) -> None`
  - L231: `def _check_dedup(self, gray: np.ndarray, result: FilterResult) -> bool`
  - L252: `def _compute_ssim(img1: np.ndarray, img2: np.ndarray, **_kw) -> float`

### src/argus/capture/gige_capture.py
- L97: `def _ip_to_int(ip: str) -> int`
- L102: `class GigECapture`
  - L129: `def __init__(self, camera_id: str, source: str, fps_target: int=30, resolution: tuple[int, int]=(1280, 720), exposure: float=0, gain: float=0, pixel_format: str='Mono8', reconnect_delay: float=5.0, max_reconnect_attempts: int=-1, source_declared: str | None=None, source_runtime: str | None=None, stream_id: str | None=None)`
  - L184: `def state(self) -> CameraState`
  - L187: `def configure_frame_identity(self, *, source_declared: str | None=None, source_runtime: str | None=None, stream_id: str | None=None) -> None`
  - L205: `def connect(self) -> bool`
  - L220: `def _connect_impl(self) -> bool`
  - L297: `def _configure_camera(self, cam: MvCamera) -> None`
  - L342: `def read(self) -> FrameData | None`
  - L416: `def request_reconnect(self) -> None`
  - L430: `def _reconnect_background(self) -> None`
  - L489: `def reconnect(self) -> bool`
  - L546: `def start_capture_thread(self) -> None`
  - L561: `def _capture_loop(self) -> None`
  - L570: `def read_latest(self) -> FrameData | None`
  - L583: `def release(self) -> None`
  - L601: `def stop(self) -> None`
  - L606: `def __enter__(self)`
  - L610: `def __exit__(self, *args)`

### src/argus/capture/manager.py
- L52: `class CameraStatus`
- L65: `class _PreparedCameraRelease`
  - L89: `def commit(self)`
  - L156: `def abort(self)`
  - L199: `def finalize(self)`
  - L205: `def _release_lock(self) -> None`
- L214: `class CameraManager`
  - L227: `def __init__(self, cameras: list[CameraConfig], alert_config: AlertConfig, on_alert: Callable[[Alert], None] | None=None, on_status_change: Callable[[str, dict], None] | None=None, cross_camera_config: CrossCameraConfig | None=None, segmenter_config: SegmenterConfig | None=None, classifier_config: ClassifierConfig | None=None, physics_config: object | None=None, imaging_config: object | None=None, health_monitor: HealthMonitor | None=None, audit_logger: AuditLogger | None=None, record_store: InferenceRecordStore | None=None, database: Database | None=None, alert_recording_store: object | None=None, event_bus: object | None=None, sensor_fusion: object | None=None, inference_buffer: InferenceBuffer | None=None, source_resolver: Callable[[CameraConfig], CameraConfig] | None=None)`
  - L336: `def _get_shared_detector(self, anomaly_config: AnomalyConfig) -> object`
  - L383: `def _create_shadow_runner(self, cam_config: CameraConfig, shadow_record: ModelRecord | None=None, primary_record: ModelRecord | None=None) -> object | None`
  - L448: `def start_all(self) -> list[str]`
  - L477: `def stop_all(self) -> None`
  - L498: `def add_camera_config(self, cam_config: CameraConfig) -> None`
  - L503: `def remove_camera_config(self, camera_id: str) -> None`
  - L508: `def start_camera(self, camera_id: str) -> bool`
  - L519: `def stop_camera(self, camera_id: str) -> None`
  - L534: `def get_status(self) -> list[CameraStatus]`
  - L559: `def serialize_camera_status(self, status: CameraStatus) -> dict`
  - L604: `def get_camera_status_payload(self, camera_id: str) -> dict | None`
  - L614: `def get_runtime(self, camera_id: str) -> CameraRuntime | None`
  - L628: `def set_degradation_publisher(self, publisher: Callable[[str, dict], None] | None) -> dict[str, str]`
  - L648: `def get_pipeline(self, camera_id: str)`
  - L652: `def update_segmenter_params(self, *, max_points: int | None=None, min_anomaly_score: float | None=None, min_mask_area_px: int | None=None, timeout_seconds: float | None=None) -> int`
  - L694: `def update_cross_camera_pairs(self, pairs: list[CameraOverlapPair], corroboration_threshold: float | None=None) -> bool`
  - L726: `def update_classifier_vocabulary(self, vocabulary: list[str]) -> int`
  - L749: `def get_latest_frame(self, camera_id: str) -> np.ndarray | None`
  - L756: `def get_latest_frame_with_identity(self, camera_id: str) -> tuple[np.ndarray | None, FrameIdentity | None]`
  - L769: `def get_latest_frame_identity(self, camera_id: str) -> FrameIdentity | None`
  - L779: `def record_snapshot_access(self, camera_id: str, snapshot_id: str) -> FrameIdentity | None`
  - L793: `def record_preview_stream_access(self, camera_id: str) -> FrameIdentity | None`
  - L803: `def get_frame_identity_status(self, camera_id: str) -> dict | None`
  - L813: `def get_scene_guard_status(self, camera_id: str) -> dict | None`
  - L824: `def get_active_tracks_snapshot(self, camera_id: str) -> list[dict] | None`
  - L835: `def get_model_statuses_snapshot(self, camera_id: str | None=None) -> list[dict]`
  - L860: `def get_system_status_snapshots(self) -> list[dict]`
  - L883: `def get_runtime_component_summary(self) -> dict`
  - L911: `def update_pipeline_thresholds(self, *, camera_ids: list[str] | None=None, anomaly_threshold: float | None=None, severity_changed: bool=False, temporal_changed: bool=False, suppression_changed: bool=False) -> tuple[int, dict[str, bool]]`
  - L941: `def clear_camera_anomaly_lock(self, camera_id: str) -> bool`
  - L949: `def get_camera_zones_snapshot(self, camera_id: str) -> list | None`
  - L956: `def update_camera_zones(self, camera_id: str, zones: list) -> bool`
  - L964: `def apply_runtime_camera_config(self, camera_config: CameraConfig) -> bool`
  - L976: `def reload_module_for_cameras(self, camera_ids: list[str], key: str, value: bool) -> dict[str, int]`
  - L1006: `def get_runtime_model_state(self, camera_id: str) -> dict | None`
  - L1027: `def get_anomaly_backend_status(self, camera_id: str) -> dict | None`
  - L1038: `def get_anomaly_inference_handle(self, camera_id: str) -> object | None`
  - L1046: `def get_wall_status(self, camera_id: str) -> dict | None`
  - L1057: `def get_raw_frame(self, camera_id: str) -> np.ndarray | None`
  - L1064: `def reconnect_camera(self, camera_id: str) -> bool`
  - L1090: `def set_source_resolver(self, source_resolver: Callable[[CameraConfig], CameraConfig] | None) -> None`
  - L1097: `def get_camera_config(self, camera_id: str) -> CameraConfig | None`
  - L1101: `def is_anomaly_locked(self, camera_id: str) -> bool`
  - L1109: `def set_pipeline_mode(self, camera_id: str, mode: PipelineMode) -> bool`
  - L1117: `def get_pipeline_mode(self, camera_id: str) -> str | None`
  - L1124: `def list_camera_ids(self) -> list[str]`
  - L1128: `def get_all_pipeline_modes(self) -> dict[str, str]`
  - L1132: `def set_all_pipeline_modes(self, mode: PipelineMode) -> dict[str, str]`
  - L1144: `def get_learning_progress(self, camera_id: str) -> dict | None`
  - L1151: `def get_diagnostics(self, camera_id: str, n: int=50) -> list | None`
  - L1158: `def evaluate_threshold(self, camera_id: str, threshold: float) -> dict | None`
  - L1165: `def get_detector_status(self, camera_id: str) -> dict | None`
  - L1185: `def reload_model(self, camera_id: str, model_path: str, *, version_tag: str | None=None) -> bool`
  - L1218: `def prepare_model_release(self, plan)`
  - L1501: `def apply_model_release_state(self, camera_id: str, reason: str='release_transition') -> dict`
  - L1533: `def _broadcast_activation_failed(self, *, camera_id: str, attempted_version: str | None, current_version: str | None, reason: str, model_path: str | None=None) -> None`
  - L1559: `def get_latest_anomaly_map(self, camera_id: str) -> np.ndarray | None`
  - L1566: `def get_drift_status(self, camera_id: str) -> dict | None`
  - L1584: `def get_backpressure_stats(self) -> dict[str, dict]`
  - L1596: `def _broadcast_wall_update(self) -> None`
  - L1629: `def _broadcast_camera_and_wall(self, camera_id: str) -> None`
  - L1634: `def _notify_camera_status(self, camera_id: str) -> None`
  - L1646: `def _get_active_model_record(self, camera_id: str)`
  - L1659: `def _get_primary_model_record(self, camera_id: str)`
  - L1671: `def _get_shadow_model_record(self, camera_id: str)`
  - L1683: `def _has_registered_model_records(self, camera_id: str) -> bool`
  - L1695: `def _resolve_model_record_path(self, record, camera_id: str, backend: str='auto') -> Path | None`
  - L1809: `def _persisted_runtime_artifact_sha256(self, record) -> str | None`
  - L1914: `def _resolve_model_path(self, cam_config: CameraConfig) -> Path | None`
  - L1944: `def _resolve_authorized_model_identity(self, camera_id: str) -> ModelIdentity | None`
  - L1974: `def _model_version_id(self, cam_config: CameraConfig, model_path: Path | None=None) -> str`
  - L1996: `def _frame_identity_stream_id(declared_config: CameraConfig, runtime_config: CameraConfig) -> str | None`
  - L2008: `def get_runner_snapshot(self, camera_id: str) -> RunnerSnapshot | None`
  - L2015: `def get_model_slot_identity(self, camera_id: str) -> ModelIdentity | None`
  - L2023: `def alert_count(self) -> int`
  - L2027: `def is_running(self) -> bool`
  - L2030: `def _start_camera(self, cam_config: CameraConfig) -> bool`
  - L2222: `def _camera_loop(self, camera_id: str, runner: CameraInferenceRunner) -> None`
  - L2412: `def _anomaly_peak_location(anomaly_map: np.ndarray | None) -> tuple[int, int]`
  - L2429: `def _downgrade_severity(cls, severity: AlertSeverity, levels: int) -> AlertSeverity`
  - L2438: `def _zone_requires_corroboration(self, camera_id: str, zone_id: str) -> bool`
  - L2456: `def _start_watchdog(self) -> None`
  - L2468: `def _stop_watchdog(self) -> None`
  - L2475: `def _watchdog_loop(self) -> None`

### src/argus/capture/post_review.py
- L16: `def post_capture_review(version_dir: Path, camera_id: str, models_dir: Path, exports_dir: Path, flag_percentile: float=0.99, progress_callback: Callable[[int, str], None] | None=None) -> dict`
- L47: `def _find_model(camera_id: str, models_dir: Path, exports_dir: Path) -> Path | None`
- L74: `def _run_review(version_dir: Path, camera_id: str, models_dir: Path, exports_dir: Path, flag_percentile: float, progress_callback: Callable[[int, str], None] | None) -> dict`
- L192: `def _update_sidecar(frame_path: Path, score: float) -> None`

### src/argus/capture/quality.py
- L26: `class FrameQualityResult`
- L39: `class CaptureStats`
  - L53: `def total_rejected(self) -> int`
  - L62: `def brightness_range(self) -> tuple[float, float]`
  - L67: `def record_rejection(self, reason: str) -> None`
  - L72: `def to_dict(self) -> dict`
- L89: `class FrameQualityFilter`
  - L98: `def __init__(self, config: CaptureQualityConfig, *, enable_person_detection: bool=True, enable_duplicate_filter: bool=True) -> None`
  - L113: `def check(self, frame: np.ndarray, prev_frame: np.ndarray | None=None) -> FrameQualityResult`
  - L184: `def _compute_entropy(gray: np.ndarray) -> float`
  - L195: `def _compute_blur_score(gray: np.ndarray) -> float`
  - L199: `def _check_exposure(self, gray: np.ndarray, mean_val: float, std_val: float) -> bool`
  - L217: `def _check_blur(self, blur_score: float) -> bool`
  - L227: `def _compute_ssim(self, gray: np.ndarray, prev_frame: np.ndarray) -> float`
  - L240: `def _check_person(self, frame: np.ndarray) -> bool`

### src/argus/capture/samplers.py
- L32: `def get_active_sampler_unavailable_reason() -> str | None`
- L47: `def _resolve_model_image_size(model: Any, fallback: int) -> int`
- L59: `class BaseSampler(ABC)`
  - L63: `def should_accept(self, frame: np.ndarray) -> tuple[bool, dict]`
  - L71: `def get_sleep_interval(self) -> float`
  - L74: `def on_frame_saved(self, frame: np.ndarray, index: int) -> None`
- L78: `class UniformSampler(BaseSampler)`
  - L81: `def __init__(self, duration_hours: float, target_frames: int) -> None`
  - L85: `def should_accept(self, frame: np.ndarray) -> tuple[bool, dict]`
  - L88: `def get_sleep_interval(self) -> float`
- L92: `class ActiveSampler(BaseSampler)`
  - L95: `def __init__(self, diversity_threshold: float=0.3, backbone: str='dinov2_vits14', image_size: int=224, sleep_interval_seconds: float=1.0, cpu_threads: int=1) -> None`
  - L115: `def _ensure_model(self) -> None`
  - L165: `def _extract_features(self, frame: np.ndarray) -> np.ndarray`
  - L199: `def should_accept(self, frame: np.ndarray) -> tuple[bool, dict]`
  - L232: `def on_frame_saved(self, frame: np.ndarray, index: int) -> None`
  - L241: `def get_sleep_interval(self) -> float`
- L245: `class ScheduledSampler(BaseSampler)`
  - L248: `def __init__(self, schedule_periods: dict[str, tuple[int, int]], frames_per_period: int, inner_sampler: BaseSampler | None=None, duration_hours: float=72.0) -> None`
  - L263: `def _current_period(self) -> str | None`
  - L277: `def _period_duration_hours(self, name: str) -> float`
  - L285: `def _seconds_until_next_period(self) -> float`
  - L308: `def should_accept(self, frame: np.ndarray) -> tuple[bool, dict]`
  - L339: `def on_frame_saved(self, frame: np.ndarray, index: int) -> None`
  - L352: `def get_sleep_interval(self) -> float`

### src/argus/config/loader.py
- L32: `def _normalize_path_separators(data: Any) -> Any`
- L52: `def _apply_env_overrides(raw: dict) -> dict`
- L78: `def load_config(config_path: str | Path) -> ArgusConfig`
- L101: `def load_config_layered(*config_paths: str | Path, apply_env: bool=True) -> ArgusConfig`
- L135: `def _deep_merge(base: dict, override: dict) -> dict`
- L145: `def save_config(config: ArgusConfig, config_path: str | Path) -> None`
- L166: `def load_camera_configs(cameras_dir: str | Path) -> list[dict]`

### src/argus/config/schema.py
- L13: `def _default_foe_vocab() -> list[str]`
- L23: `class ZonePriority(str, Enum)`
- L29: `class AlertSeverity(str, Enum)`
- L36: `class ZoneConfig(BaseModel)`
- L54: `class MOG2Config(BaseModel)`
- L83: `class SimplexConfig(BaseModel)`
- L94: `class PersistentChangeConfig(BaseModel)`
- L147: `class WaterLeakConfig(BaseModel)`
- L180: `class PlumeConfig(BaseModel)`
- L212: `class USBCaptureConfig(BaseModel)`
- L244: `class FastMotionConfig(BaseModel)`
- L294: `class PersonFilterConfig(BaseModel)`
  - L344: `def validate_object_safety_alert_classes(self) -> PersonFilterConfig`
- L362: `class CaptureQualityConfig(BaseModel)`
- L402: `class EnsembleDetectionConfig(BaseModel)`
- L421: `class AnomalyConfig(BaseModel)`
- L588: `class ClassifierConfig(BaseModel)`
- L619: `class SegmenterConfig(BaseModel)`
- L646: `class DegradationConfig(BaseModel)`
- L663: `class DriftConfig(BaseModel)`
- L689: `class RingBufferConfig(BaseModel)`
- L724: `class LowLightConfig(BaseModel)`
- L766: `class EventCameraConfig(BaseModel)`
- L795: `class GigEConfig(BaseModel)`
- L819: `class AlignmentConfig(BaseModel)`
- L845: `class SceneGuardConfig(BaseModel)`
  - L935: `def validate_scene_guard_thresholds(self) -> SceneGuardConfig`
- L955: `class CandidateModelEvidenceConfig(BaseModel)`
  - L1010: `def validate_temporal_window(self) -> 'CandidateModelEvidenceConfig'`
  - L1018: `def thresholds_complete(self) -> bool`
- L1029: `class StaticCandidateConfirmationConfig(BaseModel)`
  - L1069: `def validate_model_required_enforcement(self) -> 'StaticCandidateConfirmationConfig'`
- L1090: `class CameraConfig(BaseModel)`
  - L1101: `def migrate_legacy_region_field(cls, data)`
  - L1118: `def validate_camera_id(cls, v: str) -> str`
  - L1202: `def validate_scene_guard_alignment_contract(self) -> CameraConfig`
- L1262: `class CameraGroupConfig(BaseModel)`
  - L1276: `def validate_group_id(cls, v: str) -> str`
- L1282: `class SeverityThresholds(BaseModel)`
  - L1294: `def check_ordering(self) -> SeverityThresholds`
- L1303: `class TemporalConfirmation(BaseModel)`
- L1347: `class SuppressionConfig(BaseModel)`
- L1364: `class EarlyWarningConfig(BaseModel)`
- L1393: `class WebhookConfig(BaseModel)`
- L1399: `class CalibrationConfig(BaseModel)`
- L1413: `class AlertConfig(BaseModel)`
- L1438: `class AuthConfig(BaseModel)`
- L1452: `class AudioAlertSeverityConfig(BaseModel)`
- L1466: `class AudioAlertConfig(BaseModel)`
- L1485: `class DebugConfig(BaseModel)`
- L1492: `class DashboardConfig(BaseModel)`
  - L1561: `def validate_go2rtc_webrtc_listen_host(cls, value: str) -> str`
- L1571: `class LoggingConfig(BaseModel)`
- L1580: `class StorageConfig(BaseModel)`
- L1596: `class ModelsConfig(BaseModel)`
- L1604: `class CameraOverlapConfig(BaseModel)`
- L1614: `class RetrainingConfig(BaseModel)`
  - L1682: `def validate_grade(cls, v: str) -> str`
- L1688: `class FeedbackConfig(BaseModel)`
- L1713: `class CrossCameraConfig(BaseModel)`
- L1732: `class BaselineCaptureConfig(BaseModel)`
- L1783: `class ImagingConfig(BaseModel)`
- L1826: `class PhysicsConfig(BaseModel)`
- L1892: `class ContinuousRecordingConfig(BaseModel)`
  - L1946: `def _validate_archive(self) -> ContinuousRecordingConfig`
- L1952: `class SensorFusionConfig(BaseModel)`
  - L1983: `def _validate_sensor_thresholds(self) -> SensorFusionConfig`
- L2002: `class ArgusConfig(BaseModel)`

### src/argus/contracts/validation.py
- L20: `class ContractViolation`
- L28: `def validate_anomaly_result(result: object) -> list[ContractViolation]`
- L68: `def validate_alert(alert: object) -> list[ContractViolation]`
- L100: `def validate_feedback(feedback: object) -> list[ContractViolation]`
- L136: `def validate_inference_record(record: object) -> list[ContractViolation]`
- L162: `def _log_violations(contract_name: str, violations: list[ContractViolation]) -> None`

### src/argus/core/active_learning.py
- L41: `class ActiveLearningConfig`
- L62: `class UncertainFrame`
- L73: `class ActiveLearningSampler`
  - L87: `def __init__(self, config: ActiveLearningConfig, event_bus: EventBus | None=None, database: object | None=None) -> None`
  - L122: `def binary_entropy(p: float) -> float`
  - L130: `def score_distribution_entropy(self, camera_id: str, score: float) -> float`
  - L158: `def _on_frame_analyzed(self, event: FrameAnalyzed) -> None`
  - L224: `def buffer_frame(self, camera_id: str, frame_number: int, frame: np.ndarray) -> None`
  - L237: `def _persist_to_db(self, uncertain: UncertainFrame) -> None`
  - L258: `def save_frame(self, camera_id: str, frame: np.ndarray, frame_number: int) -> str | None`
  - L281: `def get_queue(self, camera_id: str | None=None) -> list[UncertainFrame]`
  - L288: `def pop_labeled(self, camera_id: str, frame_number: int, label: str='anomaly') -> UncertainFrame | None`
  - L327: `def queue_size(self) -> int`
  - L331: `def labeled_count(self) -> int`
  - L334: `def get_stats(self) -> dict`

### src/argus/core/adaptive_threshold.py
- L24: `class ThresholdState`
- L36: `class AdaptiveThreshold`
  - L45: `def __init__(self, base_threshold: float=0.7, max_delta: float=0.1, window_size: int=300, update_interval: int=50, ewma_alpha: float=0.1)`
  - L66: `def record_score(self, score: float) -> None`
  - L76: `def get_threshold(self) -> float`
  - L80: `def _recompute(self) -> None`
  - L135: `def get_state(self) -> ThresholdState`
  - L147: `def reset(self) -> None`

### src/argus/core/alert_ring_buffer.py
- L33: `class FrameSnapshot`
- L55: `class RecordingStatus(str, Enum)`
- L64: `class SolidifiedRecording`
- L80: `class _RetentionPolicy`
- L94: `def compress_frame(frame: np.ndarray, quality: int=85, max_height: int=720) -> bytes`
- L107: `class _PendingCapture`
- L117: `class AlertFrameBuffer`
  - L129: `def __init__(self, fps: int=15, pre_seconds: int=60, post_seconds: int=30)`
  - L140: `def append(self, snapshot: FrameSnapshot) -> None`
  - L160: `def solidify(self, alert_id: str, camera_id: str, severity: str, trigger_timestamp: float, linked_alert_id: str | None=None) -> SolidifiedRecording | None`
  - L239: `def start_post_capture(self, alert_id: str, severity: str, trigger_timestamp: float) -> None`
  - L261: `def finish_post_capture(self, alert_id: str) -> list[FrameSnapshot]`
  - L291: `def get_pending_captures(self) -> list[str]`
  - L296: `def check_expired_captures(self) -> list[str]`
  - L306: `def frame_count(self) -> int`
  - L312: `def memory_estimate_mb(self) -> float`
  - L318: `def _find_closest_frame(self, timestamp: float) -> FrameSnapshot | None`
  - L326: `def _downsample(frames: list[FrameSnapshot], target_fps: int) -> list[FrameSnapshot]`

### src/argus/core/anomaly_postprocess.py
- L17: `class AnomalyRegion`
- L31: `class AnomalyMapProcessor`
  - L34: `def __init__(self, morphology_kernel: int=5, min_contour_area: int=100, gaussian_sigma: float=1.5)`
  - L44: `def process(self, anomaly_map: np.ndarray) -> np.ndarray`
  - L91: `def extract_regions(self, anomaly_map: np.ndarray, threshold: float=0.5) -> list[AnomalyRegion]`

### src/argus/core/circuit_breaker.py
- L26: `class CircuitState(str, Enum)`
- L33: `class CircuitBreakerConfig`
- L40: `class CircuitBreaker`
  - L43: `def __init__(self, config: CircuitBreakerConfig | None=None)`
  - L55: `def state(self) -> CircuitState`
  - L60: `def failure_count(self) -> int`
  - L64: `def allow_request(self) -> bool`
  - L82: `def record_success(self) -> None`
  - L93: `def record_failure(self) -> None`
  - L113: `def get_status(self) -> dict`
  - L123: `def _persist_state(self, *, force: bool=False) -> None`
  - L146: `def _load_state(self) -> None`

### src/argus/core/correlation.py
- L22: `class CameraOverlapPair`
- L31: `class CorrelationResult`
- L39: `class CrossCameraCorrelator`
  - L48: `def __init__(self, pairs: list[CameraOverlapPair], corroboration_threshold: float=0.3)`
  - L64: `def update(self, camera_id: str, anomaly_map: np.ndarray | None, timestamp: float) -> None`
  - L70: `def check(self, camera_id: str, anomaly_location: tuple[int, int], timestamp: float, max_age_seconds: float=5.0) -> CorrelationResult`
  - L117: `def update_pairs(self, pairs: list[CameraOverlapPair], corroboration_threshold: float | None=None) -> None`
  - L145: `def prune_stale(self, max_age: float=30.0) -> None`

### src/argus/core/degradation.py
- L31: `class DegradationState(str, Enum)`
- L41: `class LockState(str, Enum)`
- L50: `class DegradationEvent`
- L91: `class DegradationStateMachine`
  - L100: `def __init__(self, camera_id: str, audit_logger: AuditLogger | None=None, max_events: int=100)`
  - L113: `def state(self) -> DegradationState`
  - L117: `def transition(self, to: DegradationState, reason: str) -> bool`
  - L175: `def get_recent_events(self, n: int=20) -> list[DegradationEvent]`
  - L181: `def reset(self) -> None`
- L190: `class DegradationLevel(str, Enum)`
- L209: `class GlobalDegradationEvent`
- L269: `class GlobalDegradationManager`
  - L280: `def __init__(self, ws_manager: ConnectionManager | None=None, max_history: int=1000)`
  - L290: `def report(self, category: str, camera_id: str | None=None, **format_kwargs: Any) -> str`
  - L366: `def resolve(self, event_id: str) -> bool`
  - L396: `def resolve_by_category(self, category: str, camera_id: str | None=None) -> int`
  - L431: `def get_active(self) -> list[dict]`
  - L443: `def get_history(self, days: int=7) -> list[dict]`
  - L454: `def active_count(self) -> int`
  - L459: `def max_active_level(self) -> DegradationLevel | None`
  - L470: `def _event_to_dict(event: GlobalDegradationEvent) -> dict`

### src/argus/core/diagnostics.py
- L16: `class StageResult`
  - L25: `def to_dict(self) -> dict`
- L37: `class CandidateRecord`
  - L47: `def to_dict(self) -> dict`
- L60: `class FrameDiagnostics`
  - L80: `def add_candidate(self, *, candidate_type: str, decision: str, stage: str='', evidence: dict | None=None, suppression_reason: str | None=None, candidate_id: str | None=None) -> CandidateRecord`
  - L107: `def to_dict(self) -> dict`
- L132: `class FrameScoreRecord`
  - L140: `def to_dict(self) -> dict`
- L150: `class DiagnosticsBuffer`
  - L158: `def __init__(self, maxlen: int=1500, score_maxlen: int=300)`
  - L163: `def append(self, diag: FrameDiagnostics) -> None`
  - L168: `def append_score(self, record: FrameScoreRecord) -> None`
  - L173: `def get_recent(self, n: int=50) -> list[FrameDiagnostics]`
  - L179: `def get_scores(self) -> list[FrameScoreRecord]`
  - L184: `def evaluate_threshold(self, new_threshold: float) -> dict`
  - L220: `def __len__(self) -> int`

### src/argus/core/error_channel.py
- L24: `class ErrorChannel`
  - L40: `def __init__(self) -> None`
  - L44: `def set_publisher(self, publisher: Callable[[str, dict], None]) -> None`
  - L55: `def emit(self, severity: str, source: str, code: str, message: str, context: dict[str, Any] | None=None) -> None`
  - L89: `def _safe_publish(self, evt: dict[str, Any]) -> None`
- L106: `def get_error_channel() -> ErrorChannel`

### src/argus/core/event_bus.py
- L43: `class Event`
- L52: `class FrameAnalyzed(Event)`
- L65: `class AlertRaised(Event)`
- L78: `class AlertDispatched(Event)`
- L88: `class CameraConnected(Event)`
- L95: `class CameraDisconnected(Event)`
- L103: `class ModelLoaded(Event)`
- L113: `class DriftDetected(Event)`
- L122: `class PipelineStageCompleted(Event)`
- L132: `class RetrainingTriggered(Event)`
- L145: `class UncertainFrameDetected(Event)`
- L167: `class EventBus`
  - L174: `def __init__(self) -> None`
  - L179: `def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None`
  - L186: `def unsubscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None`
  - L196: `def has_subscribers(self, event_type: type) -> bool`
  - L200: `def publish(self, event: Event) -> None`
  - L220: `def clear(self) -> None`

### src/argus/core/frame_identity.py
- L12: `class FrameIdentity`
  - L26: `def with_updates(self, **updates: Any) -> 'FrameIdentity'`
  - L31: `def runtime_source_hash(self) -> str`
  - L35: `def to_dict(self) -> dict[str, Any]`
- L42: `def build_frame_identity(*, camera_id: str, source_declared: str, source_runtime: str, stream_id: str | None, frame_index: int, capture_wall_ts: float | None=None, capture_monotonic_ts: float | None=None) -> FrameIdentity`

### src/argus/core/frame_quality.py
- L19: `class QualityScore`
- L31: `class FrameQualityAssessor`
  - L38: `def __init__(self, blur_threshold: float=50.0, entropy_min: float=3.0, brightness_range: tuple[float, float]=(20.0, 240.0))`
  - L49: `def assess(self, frame: np.ndarray) -> QualityScore`
  - L144: `def confidence_multiplier(self, quality: QualityScore) -> float`

### src/argus/core/health.py
- L20: `class HealthStatus(str, Enum)`
- L27: `class CameraHealth`
- L40: `class SystemHealth`
- L51: `class HealthMonitor`
  - L58: `def __init__(self, on_change: Callable[[str, dict], None] | None=None, feedback_manager: object | None=None, degradation_manager: object | None=None)`
  - L74: `def update_camera(self, camera_id: str, connected: bool, frames_captured: int=0, avg_latency_ms: float=0.0, reconnect_count: int=0, error: str | None=None) -> None`
  - L137: `def record_alert(self) -> None`
  - L142: `def _notify_change(self) -> None`
  - L174: `def get_health(self) -> SystemHealth`
  - L197: `def get_camera_health(self, camera_id: str) -> dict | None`
  - L213: `def check_stale_cameras(self, max_stale_seconds: float=30.0) -> list[str]`

### src/argus/core/inference_queue.py
- L38: `class _InferenceRequest`
- L47: `class InferenceQueue`
  - L66: `def __init__(self, predict_fn: Callable[[np.ndarray], Any], max_batch_size: int=4, max_wait_ms: float=10.0, predict_batch_fn: Callable[[list[np.ndarray]], list[Any]] | None=None)`
  - L84: `def start(self) -> None`
  - L96: `def stop(self) -> None`
  - L103: `def submit(self, camera_id: str, frame: np.ndarray) -> Future`
  - L123: `def _worker_loop(self) -> None`
  - L166: `def _run_batch(self, batch: list[_InferenceRequest]) -> None`
  - L201: `def _run_sequential(self, batch: list[_InferenceRequest]) -> None`
  - L215: `def pending(self) -> int`
  - L220: `def stats(self) -> dict`

### src/argus/core/inference_record.py
- L14: `class PrefilterDecision(str, Enum)`
- L23: `class ConformalLevel(str, Enum)`
- L33: `class FinalDecision(str, Enum)`
- L54: `class InferenceRecord`
  - L78: `def to_dict(self) -> dict`
  - L99: `def from_dict(cls, data: dict) -> InferenceRecord`

### src/argus/core/metrics.py
- L36: `class _NoOpMetric`
  - L39: `def labels(self, **_kw: str) -> _NoOpMetric`
  - L42: `def inc(self, _amount: float=1) -> None`
  - L45: `def dec(self, _amount: float=1) -> None`
  - L48: `def set(self, _value: float) -> None`
  - L51: `def observe(self, _value: float) -> None`
  - L54: `def info(self, _val: dict) -> None`
- L61: `class MetricsRegistry`
  - L64: `def __init__(self) -> None`
  - L88: `def _init(self) -> None`
  - L97: `def _do_init(self) -> None`
  - L224: `def ensure_initialized(self) -> None`
  - L228: `def generate(self) -> bytes`
  - L235: `def timer(self, histogram: Histogram | _NoOpMetric, **labels: str) -> Generator[None, None, None]`

### src/argus/core/model_discovery.py
- L15: `def _patterns_for_backend(backend: InferenceBackend='auto') -> tuple[str, ...]`
- L23: `def _suffixes_for_backend(backend: InferenceBackend='auto') -> set[str]`
- L27: `def _latest_match(search_dir: Path, backend: InferenceBackend='auto') -> Path | None`
- L39: `def find_runtime_model_in_dir(base: Path, camera_id: str, backend: InferenceBackend='auto') -> Path | None`
- L64: `def find_runtime_model(camera_id: str, backend: InferenceBackend='auto') -> Path | None`
- L75: `def resolve_runtime_model_path(model_path: str | Path, camera_id: str, backend: InferenceBackend='auto') -> Path | None`
- L114: `def find_all_models(camera_id: str) -> list[Path]`

### src/argus/core/model_status.py
- L18: `class ModelStatus`
  - L55: `def mark_loaded(self, backend: str, model_path: str | None=None, image_size: tuple[int, int] | None=None) -> None`
  - L65: `def mark_load_failed(self, error: str) -> None`
  - L71: `def mark_inference_success(self) -> None`
  - L77: `def mark_inference_failure(self, error: str) -> None`
  - L85: `def set_extra(self, **kwargs) -> None`
  - L91: `def to_dict(self) -> dict`

### src/argus/core/pipeline.py
- L109: `def _runtime_package_version(distribution: str) -> str | None`
- L122: `def _get_default_inference_executor() -> ThreadPoolExecutor`
- L137: `def _updated_anomaly_result(result: object, **changes) -> AnomalyResult`
- L178: `def _model_artifact_sha256(model_path: Path | None) -> str | None`
- L188: `def _frame_content_sha256(frame: np.ndarray | None) -> str | None`
- L205: `class PreparedAnomalyModelRelease`
- L221: `class PipelineMode(str, Enum)`
- L234: `def _severity_rank(severity) -> int`
- L239: `class PipelineStats`
  - L254: `def __init__(self) -> None`
  - L272: `def update_fps(self) -> None`
  - L287: `def snapshot(self) -> dict`
- L303: `class DetectionPipeline`
  - L324: `def __init__(self, camera_config: CameraConfig, alert_config: AlertConfig, on_alert: Callable[[Alert], None] | None=None, shared_yolo_model: object | None=None, on_drift: Callable[[str, dict], None] | None=None, classifier_config: ClassifierConfig | None=None, segmenter_config: SegmenterConfig | None=None, physics_config: object | None=None, imaging_config: object | None=None, model_version_id: str | None=None, model_path: Path | None=None, auto_discover_model: bool=True, shared_anomaly_detector: object | None=None, shadow_runner: object | None=None, feedback_manager: object | None=None, recording_store: object | None=None, database: object | None=None, event_bus: EventBus | None=None, inference_executor: ThreadPoolExecutor | None=None, encode_executor: ThreadPoolExecutor | None=None, sensor_fusion: object | None=None, degradation_publisher: Callable[[str, dict], None] | None=None, capture: CaptureAdapter | None=None, frame_identity_declared_source: str | None=None, frame_identity_runtime_source: str | None=None, frame_identity_stream_id: str | None=None)`
  - L1155: `def set_nir_frame(self, nir_frame: np.ndarray) -> None`
  - L1164: `def _find_model_in_dir(base: Path, camera_id: str, backend: str='auto') -> Path | None`
  - L1177: `def _find_model(camera_id: str, backend: str='auto') -> Path | None`
  - L1186: `def _find_baseline_dir(camera_id: str) -> Path | None`
  - L1208: `def _initialize_scene_guard(self) -> None`
  - L1303: `def _resolve_scene_guard_reference(self) -> tuple[np.ndarray | None, str | None, str | None, dict, str | None]`
  - L1502: `def _validate_profile_active_baseline(self, profile) -> str | None`
  - L1530: `def _metadata_sensitive_rois(metadata: dict) -> list[tuple[int, int, int, int]]`
  - L1549: `def _scene_edge_signature(frame: np.ndarray, width: int=320) -> np.ndarray`
  - L1562: `def _scene_quick_evidence(self, aligned_frame: np.ndarray) -> tuple[float | None, float | None]`
  - L1619: `def set_scene_reference(self, frame: np.ndarray, *, source: str, profile_id: str | None=None, metadata: dict | None=None) -> dict`
  - L1682: `def get_scene_guard_status(self) -> dict`
  - L1819: `def _scene_policy_usable(self, result, *, checked: bool) -> bool`
  - L1873: `def _fill_scene_warp_border(self, aligned: np.ndarray, homography: np.ndarray | None, source_shape: tuple[int, ...]) -> np.ndarray`
  - L1900: `def _publish_scene_guard_availability(self, *, usable: bool, state: str, reason: str) -> None`
  - L1961: `def _apply_scene_guard(self, frame: np.ndarray, frame_data: FrameData, diag: FrameDiagnostics) -> tuple[np.ndarray, bool]`
  - L2227: `def _reset_scene_sensitive_state(self) -> None`
  - L2301: `def _reset_candidate_model_evidence_state(self) -> None`
  - L2312: `def _candidate_roi_anomaly_score(anomaly_map: np.ndarray | None, bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> float | None`
  - L2339: `def _confirm_persistent_change_candidates(self, *, current_frame: np.ndarray, persistent_result, anomaly_result: AnomalyResult, diag: FrameDiagnostics) -> list`
  - L2777: `def _candidate_model_evidence_binding(self, output_space: str | None, *, model_input_variant: str='scene_stable:unmasked') -> dict[str, object]`
  - L2867: `def _apply_candidate_anomaly_attribution(self, anomaly_result: AnomalyResult, *, diag: FrameDiagnostics, frame_shape: tuple[int, ...], simplex_detected: bool, persistent_candidates: list | None=None) -> AnomalyResult`
  - L3391: `def _candidate_model_input_contract(self) -> dict[str, object]`
  - L3434: `def _record_candidate_model_evidence_skip(self, *, persistent_result, diag: FrameDiagnostics, status: str, reason: str) -> None`
  - L3499: `def _try_create_ensemble(self, camera_config: CameraConfig, primary_model_path: Path | None) -> object | None`
  - L3622: `def _categorize_alert(self, alert, simplex_result, detection_result, frame_shape: tuple | None=None) -> str`
  - L3695: `def _person_overlaps_alert_zone(self, alert, detection_result, *, frame_shape: tuple | None=None) -> bool`
  - L3754: `def _predict_anomaly(self, frame: np.ndarray, fused_tensor: np.ndarray | None=None) -> AnomalyResult`
  - L3838: `def _enter_anomaly_degraded(self, reason: str, *, manual_recovery_hint: bool=False) -> None`
  - L3919: `def _record_anomaly_inference_ok(self) -> None`
  - L3925: `def _record_anomaly_inference_failure(self, reason: str) -> None`
  - L3944: `def is_anomaly_degraded(self) -> bool`
  - L3949: `def get_anomaly_degradation_reason(self) -> str | None`
  - L3954: `def get_anomaly_degradation_started_at(self) -> float | None`
  - L3959: `def set_degradation_publisher(self, publisher: Callable[[str, dict], None] | None) -> None`
  - L3971: `def initialize(self) -> bool`
  - L4104: `def _identity_for_frame_data(self, frame_data: FrameData, pipeline_ingest_ts: float) -> FrameIdentity`
  - L4125: `def _set_latest_frame_identity(self, identity: FrameIdentity) -> None`
  - L4129: `def process_frame(self, frame_data: FrameData) -> Alert | None`
  - L4217: `def _process_frame_inner(self, frame_data: FrameData, frame: np.ndarray, start: float, diag: FrameDiagnostics) -> Alert | None`
  - L7163: `def _handle_fast_motion_detection(self, *, frame_data: FrameData, frame: np.ndarray, start: float, diag: FrameDiagnostics, fast_result: FastMotionResult) -> Alert | None`
  - L7482: `def _handle_ring_buffer(self, frame_data: FrameData, frame: np.ndarray, anomaly_result: AnomalyResult, detection_result: ObjectDetectionResult, simplex_result: object | None, simplex_detected: bool, alert: Alert | None, active_tracks: list[dict] | None=None, candidate_records: list[dict] | None=None) -> None`
  - L7633: `def _persist_alert_recording(self, alert: Alert) -> None`
  - L7696: `def run_once(self) -> Alert | None`
  - L7711: `def _flush_expired_post_captures(self) -> None`
  - L7753: `def get_wall_status(self) -> dict`
  - L7767: `def get_active_tracks_snapshot(self) -> list[dict]`
  - L7776: `def get_model_statuses_snapshot(self) -> list[dict]`
  - L7803: `def get_system_status_snapshot(self) -> dict`
  - L7893: `def get_runtime_model_state(self) -> dict`
  - L7903: `def get_anomaly_backend_status(self) -> dict`
  - L7922: `def get_anomaly_inference_handle(self) -> object | None`
  - L7931: `def get_capture_status(self) -> dict`
  - L7982: `def _m_stages_fallback(self, stage_name: str) -> object`
  - L7990: `def get_latest_frame(self) -> np.ndarray | None`
  - L7995: `def get_latest_frame_with_identity(self) -> tuple[np.ndarray | None, FrameIdentity | None]`
  - L8009: `def get_latest_frame_identity(self) -> FrameIdentity | None`
  - L8016: `def mark_snapshot_identity(self, snapshot_id: str, *, timestamp: float | None=None) -> FrameIdentity | None`
  - L8034: `def mark_preview_stream_identity(self, *, timestamp: float | None=None) -> FrameIdentity | None`
  - L8053: `def get_frame_identity_status(self) -> dict`
  - L8086: `def get_raw_frame(self) -> np.ndarray | None`
  - L8091: `def get_latest_anomaly_map(self) -> np.ndarray | None`
  - L8096: `def clear_anomaly_lock(self) -> None`
  - L8106: `def lock_state(self) -> LockState`
  - L8116: `def last_heartbeat_time(self) -> float`
  - L8121: `def last_inference_record(self) -> InferenceRecord | None`
  - L8125: `def _set_frame_outcome(self, kind: FrameOutcomeKind, reason: str | None=None) -> None`
  - L8135: `def last_frame_outcome_kind(self) -> FrameOutcomeKind`
  - L8140: `def last_frame_outcome_reason(self) -> str | None`
  - L8145: `def last_detection_failed(self) -> bool`
  - L8150: `def segmenter_consecutive_failures(self) -> int`
  - L8154: `def get_cusum_states(self) -> dict[str, CusumSnapshot]`
  - L8158: `def set_event_trigger(self, callback: Callable | None) -> None`
  - L8168: `def mode(self) -> PipelineMode`
  - L8173: `def set_mode(self, mode: PipelineMode) -> PipelineMode`
  - L8207: `def _reset_fast_motion_after_detection_mode_entry(self, old: PipelineMode, new: PipelineMode, *, now: float | None=None) -> None`
  - L8233: `def get_learning_progress(self) -> dict`
  - L8246: `def update_zones(self, zones: list) -> None`
  - L8254: `def update_thresholds(self, anomaly_threshold: float | None=None, *, severity_changed: bool=False, temporal_changed: bool=False, suppression_changed: bool=False) -> dict[str, bool]`
  - L8299: `def get_diagnostics_buffer(self) -> DiagnosticsBuffer`
  - L8303: `def get_detector_status(self) -> DetectorStatus`
  - L8307: `def get_input_quality_status(self) -> dict`
  - L8350: `def prepare_anomaly_model_release(self, model_path: str | Path, *, version_tag: str, expected_artifact_sha256: str) -> PreparedAnomalyModelRelease`
  - L8458: `def commit_anomaly_model_release(self, prepared: PreparedAnomalyModelRelease) -> bool`
  - L8493: `def abort_anomaly_model_release(self, prepared: PreparedAnomalyModelRelease) -> bool`
  - L8515: `def reload_anomaly_model(self, model_path: str | Path, *, version_tag: str | None=None) -> bool`
  - L8601: `def reload_module(self, key: str, enabled: bool) -> bool`
  - L8661: `def _revalidate_scene_guard_reference(self) -> None`
  - L8696: `def _reject_scene_guard_reference(self, *, source: str | None, profile_id: str | None, metadata: dict, error: str) -> None`
  - L8751: `def set_model_version_id(self, model_version_id: str | None) -> None`
  - L8760: `def get_shadow_runner_version(self) -> str | None`
  - L8770: `def get_shadow_runner(self) -> object | None`
  - L8778: `def set_shadow_runner(self, shadow_runner: object | None) -> dict`
  - L8821: `def shutdown(self) -> None`
  - L8844: `def _decay_all_unobserved_alert_incidents(self, camera_id: str, *, decay_detection_types: set[str] | None=None) -> None`
  - L8867: `def _evaluate_zones(self, frame_data: FrameData, anomaly_result: AnomalyResult, frame: np.ndarray, detection_type: str='anomaly', detected_objects: list[dict] | None=None, classification_label: str | None=None, classification_confidence: float | None=None, appearance_result: AnomalyResult | None=None, incident_support_maps: dict[str, np.ndarray] | None=None, decay_detection_types: set[str] | None=None) -> Alert | None`
  - L9292: `def _object_detection_payload(self, obj: ObjectDetection) -> dict`
  - L9314: `def _independent_visual_payload(candidate) -> dict`
  - L9332: `def _simplex_visual_payload(region, score: float, index: int) -> dict`
  - L9361: `def _localized_score_map_from_objects(detected_objects: list[dict] | None, frame_shape: tuple[int, ...]) -> np.ndarray | None`
  - L9421: `def _confirmed_persistent_score_map(candidates: list, frame_shape: tuple[int, ...]) -> np.ndarray | None`
  - L9482: `def _independent_visual_evidence(detected_objects: list[dict] | None) -> tuple[float, DetectionType | None]`
  - L9512: `def _incident_identity(self, detection_type: DetectionType | str, detected_object: dict, *, zone_id: str | None=None, frame_number: int | None=None) -> str`
  - L9552: `def _finite_xyxy_bbox(raw_bbox: object) -> tuple[float, float, float, float] | None`
  - L9570: `def _incident_bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float`
  - L9584: `def _associate_runtime_incident(self, *, zone_id: str, detection_type: str, class_name: str, bbox: tuple[float, float, float, float], frame_token: int) -> int`
  - L9659: `def _map_has_positive_support(score_map: np.ndarray | None) -> bool`
  - L9669: `def _object_overlaps_score_support(detected_object: dict, score_map: np.ndarray, frame_shape: tuple[int, ...]) -> bool`
  - L9696: `def _appearance_map_components(score_map: np.ndarray | None, frame_shape: tuple[int, ...], *, max_components: int=32) -> list[tuple[float, np.ndarray, list[int]]]`
  - L9770: `def _merge_visual_maps_max(channel_maps) -> np.ndarray | None`
  - L9805: `def _object_safety_score(self, detected_objects: list[dict] | None, *, zone_mask: np.ndarray | None=None, require_contributed: bool=False) -> float`
  - L9857: `def _object_overlaps_zone_mask(detected_object: dict, zone_mask: np.ndarray) -> bool`
  - L9880: `def _detected_objects_for_zone(self, detected_objects: list[dict] | None, *, zone_mask: np.ndarray | None=None) -> list[dict]`
  - L9901: `def _get_zone_polygon_mask(self, zone_id: str, polygon: list[tuple[int, int]], shape: tuple[int, int], *, source_shape: tuple[int, int]) -> np.ndarray`
  - L9931: `def _compute_zone_score(self, anomaly_map: np.ndarray, zone_id: str, polygon: list[tuple[int, int]], frame_shape: tuple) -> float`
  - L9956: `def _mask_anomaly_map_to_zone(self, anomaly_map: np.ndarray, zone_id: str, polygon: list[tuple[int, int]], frame_shape: tuple) -> np.ndarray`
  - L9977: `def _update_lock_state(self, anomaly_result: AnomalyResult, *, allow_engage: bool=True) -> None`
  - L10027: `def _update_lock_state_time(self, now: float) -> None`
  - L10042: `def _update_latency(self, start: float) -> None`
  - L10052: `def _extract_anomaly_bbox(anomaly_map: np.ndarray | None, frame_shape: tuple) -> tuple[int, int, int, int] | None`
  - L10103: `def _adjust_alert_severity(alert: Alert, escalate: bool) -> Alert`
  - L10118: `def _feed_drift_score(self, score: float, camera_id: str) -> None`
  - L10213: `def get_drift_status(self) -> DriftStatus | None`

### src/argus/core/pipeline_mode_guard.py
- L28: `class PipelineModeGuard(AbstractContextManager['PipelineModeGuard'])`
  - L31: `def __init__(self, camera_manager: 'CameraManager', camera_id: str, target_mode: PipelineMode, fallback_mode: PipelineMode=PipelineMode.ACTIVE, reason: str='') -> None`
  - L47: `def __enter__(self) -> 'PipelineModeGuard'`
  - L73: `def __exit__(self, exc_type, exc, tb) -> None`
- L95: `class GlobalPipelineModeGuard(AbstractContextManager['GlobalPipelineModeGuard'])`
  - L98: `def __init__(self, camera_manager: 'CameraManager', target_mode: PipelineMode, fallback_mode: PipelineMode=PipelineMode.ACTIVE, reason: str='') -> None`
  - L112: `def __enter__(self) -> 'GlobalPipelineModeGuard'`
  - L131: `def __exit__(self, exc_type, exc, tb) -> None`

### src/argus/core/runner.py
- L45: `class RunnerSnapshot`
- L62: `class CameraInferenceRunner`
  - L70: `def __init__(self, pipeline: DetectionPipeline, health_monitor: HealthMonitor | None=None, version_tag: str='', audit_logger: AuditLogger | None=None, max_consecutive_failures: int=5, refuse_start_on_backbone_failure: bool=False, inference_buffer: InferenceBuffer | None=None, authorized_model_resolver: Callable[[str], ModelIdentity | None] | None=None, initial_model_path: str | Path | None=None)`
  - L101: `def pipeline(self) -> DetectionPipeline`
  - L106: `def camera_id(self) -> str`
  - L110: `def degradation_state(self) -> DegradationState`
  - L114: `def consecutive_failures(self) -> int`
  - L118: `def model_identity(self) -> ModelIdentity`
  - L122: `def set_version_tag(self, version_tag: str) -> None`
  - L134: `def initialize(self) -> bool`
  - L162: `def run_once(self) -> Alert | None`
  - L166: `def run_once_outcome(self) -> FrameOutcome`
  - L271: `def _classify_outcome(self, *, frames_before: int, alert: Alert | None, scene_guard_status: dict | None, analysis_performed: bool, explicit_kind: FrameOutcomeKind | None, explicit_reason: str | None) -> FrameOutcome`
  - L350: `def _pipeline_frame_outcome_kind(self) -> FrameOutcomeKind | None`
  - L365: `def _scene_guard_status(self) -> dict | None`
  - L380: `def _build_db_record(self, record: InferenceRecord, alert: Alert | None) -> DBInferenceRecord | None`
  - L434: `def _handle_failure_threshold(self) -> None`
  - L481: `def reload_model(self, model_path: str | Path, *, version_tag: str | None=None, artifact_sha256: str | None=None, scene_profile_id: str | None=None, release_revision: int | None=None) -> ModelReloadResult`
  - L508: `def adopt_prepared_model_identity(self, *, version: str | None, path: str | None, artifact_sha256: str | None=None, scene_profile_id: str | None=None, release_revision: int | None=None) -> ModelIdentity`
  - L534: `def _resolve_authorized_model(self) -> ModelIdentity`
  - L568: `def _sync_model_identity_from_pipeline(self) -> None`
  - L597: `def get_snapshot(self) -> RunnerSnapshot`
  - L660: `def set_record_store(self, store: InferenceRecordStore | None) -> None`

### src/argus/core/runtime.py
- L23: `class FrameOutcomeKind(str, Enum)`
- L34: `class FrameOutcome`
  - L51: `def frame_captured(self) -> bool`
- L56: `class ModelIdentity`
- L67: `class PreparedModelReload`
- L75: `class ModelReloadResult`
- L85: `class RuntimeModelSlot`
  - L93: `def __init__(self, *, version: str | None=None, path: str | None=None, artifact_sha256: str | None=None, scene_profile_id: str | None=None, release_revision: int | None=None) -> None`
  - L112: `def identity(self) -> ModelIdentity`
  - L116: `def adopt(self, *, version: str | None, path: str | None, artifact_sha256: str | None=None, scene_profile_id: str | None=None, release_revision: int | None=None) -> ModelIdentity`
  - L136: `def prepare(self, *, version: str | None, path: str, artifact_sha256: str | None=None, scene_profile_id: str | None=None, release_revision: int | None=None) -> PreparedModelReload`
  - L156: `def apply(self, prepared: PreparedModelReload, reloader: Callable[[str, str | None], bool]) -> ModelReloadResult`
  - L197: `def reload(self, *, version: str | None, path: str, artifact_sha256: str | None=None, scene_profile_id: str | None=None, release_revision: int | None=None, reloader: Callable[[str, str | None], bool]) -> ModelReloadResult`
- L221: `class CameraRuntime(Protocol)`
  - L225: `def camera_id(self) -> str`
  - L228: `def latest_frame(self) -> np.ndarray | None`
  - L231: `def snapshot(self) -> RunnerSnapshot | None`
  - L234: `def status(self) -> dict | None`
  - L237: `def scene_guard(self) -> dict | None`
  - L240: `def reconnect(self) -> bool`
- L244: `class CameraRuntimeAdapter`
  - L247: `def __init__(self, *, camera_id: str, latest_frame: Callable[[], np.ndarray | None], snapshot: Callable[[], RunnerSnapshot | None], status: Callable[[], dict | None], scene_guard: Callable[[], dict | None], reconnect: Callable[[], bool]) -> None`
  - L265: `def camera_id(self) -> str`
  - L268: `def latest_frame(self) -> np.ndarray | None`
  - L271: `def snapshot(self) -> RunnerSnapshot | None`
  - L274: `def status(self) -> dict | None`
  - L277: `def scene_guard(self) -> dict | None`
  - L280: `def reconnect(self) -> bool`

### src/argus/core/scheduler.py
- L31: `class TaskScheduler`
  - L39: `def __init__(self)`
  - L43: `def start(self) -> None`
  - L54: `def stop(self) -> None`
  - L60: `def add_interval_task(self, task_id: str, func: callable, hours: float=0, minutes: float=0, seconds: float=0) -> None`
  - L90: `def remove_task(self, task_id: str) -> None`
- L100: `def create_maintenance_tasks(scheduler: TaskScheduler, database: Database | None=None, health_monitor: HealthMonitor | None=None, alerts_dir: str | Path='data/alerts', retention_days: int=90, inference_records_dir: str | Path='data/inference_records', inference_retention_days: int=7) -> None`
- L245: `def create_job_processing_task(scheduler: TaskScheduler, job_executor, *, interval_seconds: float=60.0) -> None`
- L268: `def create_backbone_retraining_task(scheduler: TaskScheduler, config, database) -> None`
- L335: `def _resolve_dataset_selection_for_retrain(*, strategy: str, camera_id: str, zone_id: str, baseline_lifecycle, database, lookback_days: int | None)`
- L430: `def create_retraining_task(scheduler: TaskScheduler, config, camera_configs: list, trainer, model_registry, baseline_manager, feedback_manager=None, baseline_lifecycle=None, database=None) -> None`
- L736: `def _cleanup_empty_dirs(base_dir: Path) -> None`
- L748: `def _delete_recording_paths(recording_paths: list[str]) -> int`
- L775: `def _delete_old_alerts_with_recording_paths(database, days: int) -> tuple[int, list[str], list[str]]`

### src/argus/core/temporal_tracker.py
- L22: `class TrackedAnomaly`
- L46: `class TemporalAnalysis`
- L59: `class _TrackState`
- L85: `class TemporalAnomalyTracker`
  - L92: `def __init__(self, match_distance: float=50.0, max_gap_frames: int=5, stationary_threshold: float=10.0, fps: float=5.0, trajectory_history_length: int=300, stationary_suppress_enabled: bool=True, stationary_suppress_after_s: float=10.0)`
  - L115: `def update(self, regions: list, frame_number: int) -> TemporalAnalysis`
  - L120: `def _update_locked(self, regions: list, frame_number: int) -> TemporalAnalysis`
  - L285: `def get_severity_boost(self, track: TrackedAnomaly) -> float`
  - L305: `def reset(self) -> None`
  - L311: `def get_active_tracks_snapshot(self) -> list[dict]`
  - L332: `def get_trajectory_histories(self) -> list[list[tuple[float, float, float]]]`

### src/argus/core/video_encoder.py
- L35: `class _VideoWriterLike(Protocol)`
  - L38: `def write(self, frame: np.ndarray) -> None`
  - L41: `def isOpened(self) -> bool`
  - L44: `def release(self) -> None`
- L48: `class _FFmpegVideoWriter`
  - L55: `def __init__(self, path: Path, ffmpeg: str, fps: int, width: int, height: int, crf: int, preset: str) -> None`
  - L99: `def write(self, frame: np.ndarray) -> None`
  - L107: `def isOpened(self) -> bool`
  - L110: `def release(self) -> None`
- L123: `def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None`
- L129: `def jpeg_dimensions(jpeg_bytes: bytes) -> tuple[int, int] | None`
- L141: `def _create_video_writer(path: Path, fps: int, width: int, height: int, crf: int=23, preset: str='veryfast') -> _VideoWriterLike`
- L186: `def _remux_faststart(mp4_path: Path) -> None`
- L234: `class Mp4Encoder`
  - L248: `def __init__(self, output_path: Path, fps: int, width: int=1280, height: int=720, crf: int=23, preset: str='veryfast')`
  - L268: `def write_jpeg_frame(self, jpeg_bytes: bytes) -> None`
  - L282: `def finalize(self) -> int`
  - L300: `def frame_count(self) -> int`
- L304: `def concat_mp4(pre_path: Path, post_path: Path, output_path: Path, crf: int=23, preset: str='veryfast', fps: int | None=None) -> int`
- L358: `def extract_frame_jpeg(mp4_path: Path, index: int, jpeg_quality: int=85) -> bytes | None`
- L407: `def get_video_dimensions(mp4_path: Path) -> tuple[int, int] | None`
- L420: `def get_video_frame_count(mp4_path: Path) -> int | None`
- L437: `def repair_video_timestamps(mp4_path: Path, fps: int, crf: int=23, preset: str='veryfast') -> bool`

### src/argus/core/zone_mask.py
- L23: `class ZoneMaskEngine`
  - L32: `def __init__(self, zones: list[ZoneConfig], frame_height: int, frame_width: int)`
  - L44: `def apply(self, frame: np.ndarray) -> np.ndarray`
  - L52: `def update_zones(self, zones: list[ZoneConfig]) -> None`
  - L59: `def get_include_zones(self) -> list[ZoneConfig]`
  - L64: `def get_exclude_zones(self) -> list[ZoneConfig]`
  - L70: `def has_zones(self) -> bool`
  - L74: `def _build_mask(self, zones: list[ZoneConfig]) -> np.ndarray | None`

### src/argus/dashboard/api_contracts.py
- L47: `class ApiContractModel(BaseModel)`
- L57: `class ModelBatchInferenceRequest(ApiContractModel)`
- L62: `class ReleaseEvidenceSourcesRequest(ApiContractModel)`
- L72: `class ModelPromoteRequest(ApiContractModel)`
- L88: `class ModelRetireRequest(ApiContractModel)`
- L95: `class ModelReexportRequest(ApiContractModel)`
- L100: `class ModelAbLiveCompareRequest(ApiContractModel)`
- L105: `class BackboneUpgradeRequest(ApiContractModel)`
- L111: `class ReleaseReadinessGate(ApiContractModel)`
- L121: `class ReleaseReadinessResponse(ApiContractModel)`
- L145: `class CameraModeRequest(ApiContractModel)`
- L149: `class CameraModeResponse(ApiContractModel)`
- L155: `class CameraStatsResponse(ApiContractModel)`
- L164: `class CameraLifecycleResponse(ApiContractModel)`
- L181: `class CameraSummaryResponse(ApiContractModel)`
- L196: `class CameraListResponse(ApiContractModel)`
- L200: `class SceneGuardRecoveryProgress(ApiContractModel)`
- L206: `class SceneGuardMetrics(ApiContractModel)`
- L233: `class SceneGuardStatusResponse(ApiContractModel)`
- L261: `class CameraLifecycleStageResponse(ApiContractModel)`
- L268: `class CameraRuntimeResponse(ApiContractModel)`
- L276: `class CameraRunnerResponse(ApiContractModel)`
- L288: `class CameraSourceDiagnosticsResponse(ApiContractModel)`
- L308: `class CameraDetailResponse(ApiContractModel)`
- L348: `class AlertWorkflowTransitionRequest(ApiContractModel)`
- L357: `class AlertWorkflowTransitionResponse(ApiContractModel)`
- L365: `class AlertIdListRequest(ApiContractModel)`
- L369: `class AlertMutationResponse(ApiContractModel)`
- L374: `class AlertBulkMutationResponse(ApiContractModel)`
- L383: `class ReplayPinnedFrameResponse(ApiContractModel)`
- L389: `class ReplayMetadataResponse(ApiContractModel)`
- L411: `class ReplayOperatorActionResponse(ApiContractModel)`
- L417: `class ReplayKeyFrameResponse(ApiContractModel)`
- L423: `class ReplayTrajectoryPointResponse(ApiContractModel)`
- L430: `class ReplayPersonSignalResponse(ApiContractModel)`
- L436: `class ReplayClipResponse(ApiContractModel)`
- L443: `class ReplaySignalsResponse(ApiContractModel)`
- L461: `class ReplayPinFrameRequest(ApiContractModel)`
- L466: `class ReplayPinFrameResponse(ApiContractModel)`
- L470: `class ReplayClipRequest(ApiContractModel)`
  - L476: `def validate_range(self) -> 'ReplayClipRequest'`
- L482: `class ReplayClipMutationResponse(ApiContractModel)`
- L487: `class ReplayClipListResponse(ApiContractModel)`
- L495: `class TrajectoryMmPointResponse(ApiContractModel)`
- L501: `class TrajectoryFitPointResponse(TrajectoryMmPointResponse)`
- L506: `class TrajectoryFitResponse(ApiContractModel)`
- L517: `class TrajectoryPrimaryResponse(ApiContractModel)`
- L525: `class TrajectoryClassificationResponse(ApiContractModel)`
- L530: `class AlertTrajectoryResponse(ApiContractModel)`

### src/argus/dashboard/api_response.py
- L28: `class ApiEnvelope(BaseModel, Generic[T])`
- L46: `class ErrorCode`
- L104: `def error_code_for_status(status_code: int) -> int`
- L109: `class JsonApiEnvelopeMiddleware`
  - L122: `def __init__(self, app: ASGIApp, *, api_prefix: str='/api') -> None`
  - L127: `def _header_value(headers: list[tuple[bytes, bytes]], name: bytes) -> bytes`
  - L135: `def _is_json_content_type(cls, headers: list[tuple[bytes, bytes]]) -> bool`
  - L142: `def _is_envelope(payload: object) -> bool`
  - L152: `def _request_id(scope: Scope) -> str | None`
  - L160: `def _error_message(status_code: int, payload: object) -> str`
  - L180: `def _normalized_payload(cls, scope: Scope, status_code: int, payload: object) -> object`
  - L205: `def _replace_content_headers(headers: list[tuple[bytes, bytes]], content_length: int) -> list[tuple[bytes, bytes]]`
  - L220: `async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None`
- L278: `def install_api_envelope_openapi(app: FastAPI) -> None`
- L390: `def api_success(data: Any=None, msg: str='ok', status_code: int=200, headers: dict[str, str] | None=None) -> JSONResponse`
- L401: `def api_typed_success(data: Any, data_model: type[BaseModel], msg: str='ok', status_code: int=200, headers: dict[str, str] | None=None) -> JSONResponse`
- L433: `def api_error(code: int, msg: str, status_code: int=400, data: Any=None, headers: dict[str, str] | None=None) -> JSONResponse`
- L448: `def api_paginated(key: str, items: list, total: int, page: int, page_size: int) -> JSONResponse`
- L466: `def api_not_found(msg: str='资源不存在') -> JSONResponse`
- L470: `def api_unavailable(msg: str='服务不可用') -> JSONResponse`
- L474: `def api_forbidden(msg: str='权限不足') -> JSONResponse`
- L478: `def api_validation_error(msg: str) -> JSONResponse`
- L482: `def api_conflict(msg: str, data: Any=None) -> JSONResponse`
- L486: `def api_internal_error(msg: str='内部错误') -> JSONResponse`

### src/argus/dashboard/app.py
- L74: `def _request_id(request: Request) -> str | None`
- L78: `def _error_headers(request: Request, headers: dict[str, str] | None=None) -> dict[str, str]`
- L89: `def _http_error_payload(request: Request, exc: StarletteHTTPException) -> tuple[int, str, object]`
- L120: `def _mount_vue_spa(app: FastAPI, vue_dist: Path) -> None`
- L153: `def create_app(database: Database | None=None, camera_manager: CameraManager | None=None, health_monitor: HealthMonitor | None=None, alerts_dir: str | None=None, config: object | None=None, config_path: str | None=None, task_manager: object | None=None, go2rtc_instance: Go2RTCManager | None=None, stream_registry: StreamRegistry | None=None, camera_orchestrator: CameraOrchestrator | None=None, sensor_fusion: SensorFusion | None=None, baseline_manager: object | None=None, model_trainer: object | None=None) -> FastAPI`

### src/argus/dashboard/auth.py
- L54: `def hash_password(password: str) -> str`
- L61: `def verify_password(password: str, stored: str) -> bool`
- L74: `def create_session_token(username: str, role: str, secret: str) -> str`
- L81: `def verify_session_token(token: str, secret: str, max_age: int=28800) -> dict | None`
- L99: `def require_role(request: Request, *roles: str) -> bool`
- L107: `def current_username(request: Request) -> str`
- L151: `def has_permission(request: Request, permission: str) -> bool`
- L165: `def require_permission(request: Request, permission: str) -> bool`
- L175: `def get_denied_response(request: Request, permission: str) -> dict`
- L205: `class RequestIdMiddleware`
  - L219: `def __init__(self, app: ASGIApp)`
  - L223: `def _resolve_request_id(cls, scope: Scope) -> str`
  - L237: `async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None`
- L259: `class AuthMiddleware`
  - L270: `def __init__(self, app: ASGIApp, config: AuthConfig, session_secret: str, database: Database | None=None)`
  - L287: `async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None`
  - L368: `def _verify_api_token(self, token: str) -> bool`
- L378: `class RateLimitMiddleware`
  - L384: `def __init__(self, app: ASGIApp, max_requests_per_minute: int=60)`
  - L389: `async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None`
- L428: `def verify_token(token: str, config: AuthConfig) -> bool`
- L445: `class SecurityHeadersMiddleware`
  - L448: `def __init__(self, app: ASGIApp)`
  - L451: `async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None`
- L485: `async def login_page(request: Request)`
- L499: `async def login_handler(request: Request)`
- L552: `async def logout(request: Request)`

### src/argus/dashboard/degradation_bridge.py
- L12: `def publish_pipeline_degradation(ws_manager: object, degradation_manager: object | None, topic: str, payload: dict[str, Any]) -> None`

### src/argus/dashboard/forms.py
- L12: `async def parse_request_form(request: Request) -> FormData`
- L29: `def htmx_toast_headers(message: str, *, toast_type: str='success', **events: object) -> dict[str, str]`

### src/argus/dashboard/model_runtime.py
- L18: `def get_registry(request: Request) -> ModelRegistry | None`
- L29: `def find_registered_model_by_path(request: Request, model_path: str | Path, *, camera_id: str | None=None) -> ModelRecord | None`

### src/argus/dashboard/routes/alerts.py
- L52: `def _cleanup_alert_images(alert_id: str, image_paths: list[str], recording_store: AlertRecordingStore | None) -> None`
- L68: `def _generate_composite(snapshot_path: str, heatmap_path: str, alpha: float=0.4) -> bytes | None`
- L83: `def _is_safe_path(file_path: str, alerts_dir: Path) -> bool`
- L94: `def _get_recording_state(db, recording_store: AlertRecordingStore | None, alert_id: str) -> dict`
- L148: `def _safe_int(value) -> int | None`
- L157: `def _safe_float(value) -> float | None`
- L166: `def _candidate_events_from_signals(signals: dict) -> list[dict]`
- L199: `def _event_score(event: dict) -> float | None`
- L215: `def _compact_candidate_event(event: dict) -> dict`
- L274: `def _candidate_event_distance(event: dict, trigger_frame_index: int) -> int`
- L283: `def _get_candidate_evidence_summary(recording_store: AlertRecordingStore | None, alert_id: str) -> dict | None`
- L339: `async def alert_detail(request: Request, alert_id: str)`
- L401: `def export_evidence_package(request: Request, alert_id: str)`
- L493: `def alert_image(request: Request, alert_id: str, image_type: str)`
- L561: `async def alert_workflow_transition(request: Request, alert_id: str, body: AlertWorkflowTransitionRequest)`
- L663: `async def bulk_acknowledge(request: Request, body: AlertIdListRequest)`
- L702: `async def bulk_false_positive(request: Request, body: AlertIdListRequest)`
- L740: `async def bulk_delete(request: Request, body: AlertIdListRequest)`
- L780: `def get_alert_group(request: Request, event_group_id: str, limit: int=Query(50, ge=1, le=200))`
- L812: `def export_csv(request: Request, camera_id: str | None=Query(None), severity: str | None=Query(None))`
- L847: `def export_pdf_report(request: Request, camera_id: str | None=Query(None), severity: str | None=Query(None))`
- L911: `def acknowledge_alert(request: Request, alert_id: str)`
- L941: `def confirm_real_anomaly(request: Request, alert_id: str)`
- L978: `def feedback_stats(request: Request, camera_id: str | None=None)`
- L1004: `def mark_false_positive(request: Request, alert_id: str)`
- L1043: `def delete_alert(request: Request, alert_id: str)`
- L1076: `def _submit_workflow_feedback(request: Request, alert_id: str, workflow_status: str, category: str | None=None, notes: str | None=None) -> None`
- L1139: `def _add_fp_snapshot_to_baseline(request: Request, alert_id: str) -> None`
- L1209: `async def save_annotations(request: Request, alert_id: str)`
- L1239: `def alerts_json(request: Request, camera_id: str | None=Query(None), severity: str | None=Query(None), limit: int=Query(50, ge=1, le=500), offset: int=Query(0, ge=0))`

### src/argus/dashboard/routes/audit.py
- L28: `def _audit_entry_to_dict(entry) -> dict`
- L46: `async def audit_json(request: Request, user: str | None=Query(None), action: str | None=Query(None), page: int=Query(1, ge=1), page_size: int=Query(20, ge=1, le=100))`

### src/argus/dashboard/routes/backup.py
- L19: `def _deny_non_admin(request: Request)`
- L26: `async def backup_list_json(request: Request)`
- L37: `async def create_backup(request: Request)`
- L81: `async def restore_backup(request: Request)`
- L109: `def delete_backup(request: Request, backup_name: str)`

### src/argus/dashboard/routes/baseline.py
- L42: `def _camera_baseline_root(baselines_dir: Path, camera_id: str, zone_id: str=_DEFAULT_ZONE_ID) -> Path`
- L54: `def _iter_camera_baseline_versions(baselines_dir: Path, camera_id: str, zone_id: str=_DEFAULT_ZONE_ID)`
- L68: `def _resolve_baseline_version_dir(baselines_dir: Path, camera_id: str, version: str, zone_id: str=_DEFAULT_ZONE_ID) -> Path`
- L78: `def _read_capture_meta(version_dir: Path) -> dict`
- L91: `def _count_version_images(version_dir: Path) -> int`
- L96: `def _reset_current_marker_after_delete(base_dir: Path, deleted_version: str) -> None`
- L114: `def baseline_collections_json(request: Request)`
- L194: `def collections_activate(request: Request, camera_id: str, zone_id: str, version: str)`
- L216: `def collections_retire(request: Request, camera_id: str, zone_id: str, version: str)`
- L234: `def collections_delete(request: Request, camera_id: str, zone_id: str, version: str, confirm: bool=False)`
- L264: `def baseline_list_json(request: Request)`
- L302: `async def models_list_json(request: Request)`
- L338: `async def delete_model_by_path(request: Request)`
- L367: `async def training_history_json(request: Request)`
- L379: `async def training_history_metrics(request: Request, record_id: int)`
- L457: `async def delete_baseline_version(request: Request)`
- L514: `async def delete_baseline_version_post(request: Request)`
- L520: `async def deploy_model(request: Request)`
- L559: `async def start_capture_job(request: Request)`
- L643: `async def start_capture_legacy(request: Request)`
- L649: `async def pause_capture_job(task_id: str, request: Request)`
- L658: `async def resume_capture_job(task_id: str, request: Request)`
- L667: `async def abort_capture_job(task_id: str, request: Request)`
- L676: `async def get_capture_job_status(task_id: str, request: Request)`
- L695: `async def compare_models_route(request: Request)`
- L748: `def _train_model_task(progress_callback, *, baselines_dir, models_dir, camera_id, model_type, export_format, quantization='fp16', database_url=None, anomaly_config=None, resume_from=None, skip_baseline_validation=False)`
- L902: `def _get_baseline_manager(request: Request)`
- L915: `async def optimize_preview(request: Request, camera_id: str='', zone_id: str='default', target_ratio: float=0.2)`
- L943: `async def optimize_baseline_json(request: Request)`
- L1017: `async def camera_groups_json(request: Request)`
- L1048: `async def merge_group_baseline(request: Request)`
- L1096: `async def merge_false_positives(request: Request)`
- L1130: `def _get_lifecycle(request: Request)`
- L1136: `async def baseline_versions_json(request: Request)`
- L1152: `async def verify_baseline(request: Request)`
- L1182: `async def activate_baseline(request: Request)`
- L1208: `async def retire_baseline(request: Request)`
- L1252: `def _validate_filename_param(filename: str) -> bool`
- L1264: `async def list_version_images(camera_id: str, version: str, request: Request)`
- L1297: `async def get_version_image_content(camera_id: str, version: str, filename: str, request: Request)`
- L1345: `async def delete_version_image(camera_id: str, version: str, filename: str, request: Request)`
- L1400: `async def upload_version_image(camera_id: str, version: str, request: Request)`

### src/argus/dashboard/routes/calibration.py
- L19: `def _calibration_dir(request: Request) -> Path`
- L28: `async def get_calibration_status(request: Request, camera_id: str) -> JSONResponse`
- L50: `async def upload_calibration(request: Request, camera_id: str, file: UploadFile=File(...)) -> JSONResponse`
- L112: `async def compute_calibration(request: Request, camera_id: str) -> JSONResponse`
- L159: `async def verify_calibration(request: Request, camera_id: str) -> JSONResponse`

### src/argus/dashboard/routes/cameras.py
- L55: `def _preview_gateway(request: Request) -> PreviewGateway`
- L65: `def _active_orchestrator(request: Request)`
- L80: `def _ensure_go2rtc_stream(request: Request, cam_config)`
- L89: `def _find_camera_config(request: Request, camera_id: str)`
- L94: `def _manager_camera_status_payload(camera_manager, camera_id: str, *, status=None) -> dict | None`
- L132: `def _stream_resolution_for(request: Request, camera_id: str)`
- L140: `def _source_consistency_status(*, declared_source: str, runtime_source: str, go2rtc_stream: str | None) -> str`
- L153: `def _public_scene_guard_status(status: dict | None) -> dict | None`
- L230: `def _serialize_diagnostic_record(record) -> dict`
- L287: `def _recent_camera_diagnostics(camera_manager, camera_id: str, limit: int=20) -> list[dict]`
- L307: `def _manager_anomaly_backend_status(camera_manager, camera_id: str) -> tuple[str | None, str]`
- L336: `def _get_region_info(request: Request, region_id: int | None) -> dict`
- L372: `def _get_lifecycle_stages(request: Request, camera_id: str, *, cam_status=None) -> list[dict]`
- L445: `async def add_camera(request: Request)`
- L540: `async def delete_camera(request: Request, camera_id: str)`
- L602: `async def get_camera_config(request: Request, camera_id: str)`
- L626: `async def update_camera(request: Request, camera_id: str)`
- L720: `async def set_camera_mode(request: Request, camera_id: str, body: CameraModeRequest)`
- L763: `async def start_camera(request: Request, camera_id: str)`
- L798: `async def stop_camera(request: Request, camera_id: str)`
- L821: `async def test_camera_connection(request: Request, camera_id: str)`
- L846: `async def test_camera_connection_draft(request: Request)`
- L869: `async def usb_devices(request: Request)`
- L925: `def cameras_json(request: Request)`
- L986: `def camera_detail_json(request: Request, camera_id: str)`
- L1184: `def camera_diagnostics(request: Request, camera_id: str, limit: int=50)`
- L1204: `def camera_runner_snapshot(request: Request, camera_id: str)`
- L1243: `def camera_snapshot(request: Request, camera_id: str)`
- L1249: `async def camera_stream(request: Request, camera_id: str)`
- L1255: `async def camera_heatmap_stream(request: Request, camera_id: str)`
- L1263: `async def wall_status(request: Request)`

### src/argus/dashboard/routes/config.py
- L32: `def _deny_without_permission(request: Request, permission: str)`
- L38: `async def _parse_form_or_json(request: Request) -> dict`
- L50: `def _has_value(data: dict, key: str) -> bool`
- L54: `def _as_bool(value: object) -> bool`
- L60: `def _runtime_component_summary(camera_manager) -> dict`
- L84: `class ThresholdUpdateRequest(BaseModel)`
- L92: `class ModelReloadRequest(BaseModel)`
- L98: `async def update_detection_params(request: Request)`
- L195: `async def update_notifications(request: Request)`
- L224: `async def test_webhook(request: Request)`
- L260: `def storage_info_json(request: Request)`
- L290: `def cleanup_data(request: Request)`
- L320: `async def update_thresholds(request: Request, req: ThresholdUpdateRequest)`
- L341: `async def reload_model(request: Request, req: ModelReloadRequest)`
- L371: `async def clear_anomaly_lock(request: Request, camera_id: str)`
- L398: `async def restart_camera(request: Request, camera_id: str)`
- L417: `def reload_config(request: Request)`
- L456: `def save_config(request: Request)`
- L481: `async def get_audio_alerts(request: Request)`
- L498: `async def update_audio_alerts(request: Request)`
- L528: `async def get_module_states(request: Request)`
- L555: `async def get_classifier_config(request: Request)`
- L595: `async def get_segmenter_config(request: Request)`
- L634: `async def get_imaging_config(request: Request)`
- L665: `async def get_cross_camera_config(request: Request)`
- L706: `class CrossCameraPairEntry(BaseModel)`
- L712: `class CrossCameraUpdateRequest(BaseModel)`
- L720: `async def update_cross_camera_config(request: Request, req: CrossCameraUpdateRequest)`
- L839: `class SegmenterParamsRequest(BaseModel)`
- L847: `async def update_segmenter_params_route(request: Request, req: SegmenterParamsRequest)`
- L926: `class ClassifierVocabularyRequest(BaseModel)`
- L934: `async def update_classifier_vocabulary(request: Request, req: ClassifierVocabularyRequest)`
- L1033: `class ModuleToggleRequest(BaseModel)`
- L1039: `async def update_module_toggle(request: Request, req: ModuleToggleRequest)`

### src/argus/dashboard/routes/degradation.py
- L18: `def _get_degradation_manager(request: Request)`
- L24: `def active_degradations(request: Request)`
- L38: `def degradation_history(request: Request, days: int=Query(default=7, ge=1, le=90))`
- L54: `def degradation_summary(request: Request)`

### src/argus/dashboard/routes/labeling.py
- L31: `def labeling_queue(request: Request, camera_id: str | None=Query(None), status: str | None=Query(None), limit: int=Query(50, ge=1, le=200), offset: int=Query(0, ge=0)) -> JSONResponse`
- L56: `async def label_entry(request: Request, entry_id: int) -> JSONResponse`
- L102: `async def skip_entry(request: Request, entry_id: int) -> JSONResponse`
- L127: `def labeling_stats(request: Request, camera_id: str | None=Query(None)) -> JSONResponse`
- L145: `async def trigger_retrain(request: Request) -> JSONResponse`
- L235: `def labeling_image(request: Request, entry_id: int)`
- L261: `def _copy_to_baselines(camera_id: str, frame_path: str, zone_id: str='default') -> bool`

### src/argus/dashboard/routes/me.py
- L23: `async def get_me(request: Request) -> JSONResponse`

### src/argus/dashboard/routes/models.py
- L76: `def _deny_without_permission(request: Request, permission: str) -> JSONResponse | None`
- L83: `def _get_registry(request: Request) -> ModelRegistry | None`
- L88: `def _get_session_factory(request: Request)`
- L96: `def _get_release_anomaly_config(config: object | None) -> object | None`
- L108: `def _get_release_pipeline(request: Request) -> ReleasePipeline | None`
- L154: `def _get_release_service(request: Request) -> ModelReleaseService | None`
- L198: `def _expected_runtime_record(request: Request, model: ModelRecord) -> ModelRecord | None`
- L226: `def _model_runtime_state(request: Request, model: ModelRecord) -> str | None`
- L261: `def models_status(request: Request) -> JSONResponse`
- L287: `async def list_models(request: Request, camera_id: str | None=None)`
- L307: `async def activate_model(request: Request, version_id: str)`
- L332: `async def rollback_model(request: Request, version_id: str)`
- L359: `def _run_batch_inference(image_paths: list[str], detector: object) -> tuple[list[dict], int]`
- L400: `async def batch_inference(request: Request, body: ModelBatchInferenceRequest)`
- L445: `def get_release_operation(request: Request, operation_id: str)`
- L460: `def _public_runtime_method(manager: object | None, name: str)`
- L477: `def _append_readiness_gate(gates: list[dict], *, key: str, label: str, passed: bool, reason: str) -> None`
- L494: `def _safe_runtime_state(manager: object | None, camera_id: str) -> dict | None`
- L533: `def get_release_readiness(request: Request, version_id: str)`
- L826: `async def promote_model(request: Request, version_id: str, body: ModelPromoteRequest)`
- L938: `async def retire_model(request: Request, version_id: str, body: ModelRetireRequest | None=None)`
- L1002: `def delete_model(request: Request, version_id: str)`
- L1047: `def _get_trainer(request: Request)`
- L1070: `def _get_baseline_manager(request: Request)`
- L1083: `async def reexport_model(request: Request, version_id: str, body: ModelReexportRequest | None=None)`
- L1142: `async def recalibrate_model(request: Request, version_id: str)`
- L1202: `async def stage_history(request: Request, version_id: str)`
- L1216: `async def list_version_events(request: Request, camera_id: str | None=None, limit: int=50)`
- L1234: `async def shadow_report(request: Request, version_id: str, camera_id: str | None=None, days: int=7)`
- L1261: `def _query_shadow_logs(session, version_id: str, camera_id: str | None, days: int)`
- L1275: `async def ab_compare_scores(request: Request, version_id: str, camera_id: str | None=None, days: int=7, limit: int=500)`
- L1316: `async def ab_compare_distribution(request: Request, version_id: str, camera_id: str | None=None, days: int=7, bins: int=20)`
- L1367: `def _get_or_load_shadow_detector(shadow_model_path: Path, shadow_version_id: str, threshold: float)`
- L1391: `def _run_ab_heatmap(frame: np.ndarray, production_detector: object, shadow_model_path: Path, shadow_version_id: str, shadow_threshold: float) -> dict`
- L1441: `def _anomaly_map_to_jpeg(anomaly_map: np.ndarray, target_hw: tuple[int, int]) -> bytes`
- L1463: `async def ab_live_compare(request: Request, body: ModelAbLiveCompareRequest)`
- L1525: `async def backbone_status(request: Request)`
- L1537: `async def backbone_upgrade(request: Request, body: BackboneUpgradeRequest)`
- L1571: `def threshold_preview(request: Request, camera_id: str | None=None, threshold: float=0.5, days: int=7) -> JSONResponse`

### src/argus/dashboard/routes/physics.py
- L25: `async def get_alert_trajectory(request: Request, alert_id: str) -> JSONResponse`
- L113: `async def get_active_tracks(request: Request, camera_id: str) -> JSONResponse`
- L132: `async def get_alert_localization(request: Request, alert_id: str) -> JSONResponse`

### src/argus/dashboard/routes/recordings.py
- L18: `async def list_segments(request: Request, camera_id: str, start: str | None=Query(None, description='ISO date start filter'), end: str | None=Query(None, description='ISO date end filter')) -> JSONResponse`
- L49: `async def serve_segment_video(request: Request, camera_id: str, filename: str)`
- L85: `async def storage_stats(request: Request) -> JSONResponse`
- L100: `async def trigger_cleanup(request: Request) -> JSONResponse`

### src/argus/dashboard/routes/regions.py
- L24: `def _region_to_dict(region) -> dict`
- L35: `def _normalize_region_payload(body: dict) -> tuple[dict | None, str | None]`
- L51: `def _audit(request: Request, action: str, target_id: str, detail: str) -> None`
- L67: `async def regions_json(request: Request, name: str | None=Query(None), owner: str | None=Query(None), phone: str | None=Query(None))`
- L90: `async def create_region_json(request: Request)`
- L115: `async def update_region_json(request: Request, region_id: int)`
- L146: `async def delete_region_json(request: Request, region_id: int)`

### src/argus/dashboard/routes/replay.py
- L49: `def _parse_single_byte_range(range_header: str, file_size: int) -> tuple[int, int] | None`
- L105: `def _range_not_satisfiable(file_size: int)`
- L119: `def _get_recording_store(request: Request)`
- L124: `def _get_db(request: Request)`
- L133: `def replay_metadata(request: Request, alert_id: str)`
- L162: `def replay_signals(request: Request, alert_id: str)`
- L217: `def replay_video(request: Request, alert_id: str)`
- L277: `def replay_frame(request: Request, alert_id: str, index: int)`
- L301: `def replay_heatmap_frame(request: Request, alert_id: str, index: int)`
- L322: `def replay_reference(request: Request, alert_id: str, date: str=Query(default='', description='Reference date YYYY-MM-DD'), time_str: str=Query(default='', alias='time', description='Reference time HH:MM:SS'), frame_offset_seconds: float=Query(default=0.0, description='Seconds to shift the reference frame relative to trigger time'))`
- L465: `async def pin_frame(request: Request, alert_id: str, body: ReplayPinFrameRequest)`
- L488: `async def add_clip(request: Request, alert_id: str, body: ReplayClipRequest)`
- L530: `def delete_clip(request: Request, alert_id: str, index: int)`
- L546: `def _alert_trigger_timestamp(alert: AlertRecord) -> float`
- L562: `def _storyboard_item(primary_ts: float, alert: AlertRecord) -> dict`
- L577: `def replay_storyboard(request: Request, alert_id: str)`
- L659: `def _compute_key_frames(signals: dict) -> list[dict]`

### src/argus/dashboard/routes/reports.py
- L22: `async def reports_json(request: Request, days: int | None=Query(None, ge=7, le=365))`
- L52: `def _rate(count: int, total: int) -> float`
- L57: `def _compute_evidence_stats(alerts, db) -> dict`
- L90: `def _filter_alerts_by_days(alerts, days: int | None)`
- L99: `def _compute_severity_counts(alerts) -> dict[str, int]`
- L108: `def _compute_daily_trend(alerts, days: int)`
- L135: `def _compute_camera_dist(alerts)`
- L143: `def _compute_fp_trend(alerts, days: int)`
- L172: `async def daily_trend_json(request: Request, days: int=Query(30, ge=7, le=90))`
- L181: `async def severity_dist_json(request: Request, days: int | None=Query(None, ge=7, le=365))`
- L194: `async def camera_dist_json(request: Request, days: int | None=Query(None, ge=7, le=365))`
- L207: `async def fp_trend_json(request: Request, days: int=Query(30, ge=7, le=90))`
- L218: `def _generate_compliance_data(db, audit_logger, health_monitor, camera_manager, days: int) -> dict`
- L359: `def _render_csv(data: dict) -> str`
- L453: `def _render_pdf(data: dict) -> bytes`
- L625: `async def compliance_report(request: Request, days: int=Query(30, ge=7, le=365), format: str=Query('csv'))`

### src/argus/dashboard/routes/sensors.py
- L32: `class SensorSignalRequest(BaseModel)`
- L44: `class SensorReadingRequest(BaseModel)`
- L61: `class SensorSignalDeleteRequest(BaseModel)`
- L68: `def _get_fusion(request: Request)`
- L74: `async def set_signal(request: Request, payload: SensorSignalRequest) -> JSONResponse`
- L105: `async def set_reading(request: Request, payload: SensorReadingRequest) -> JSONResponse`
- L143: `async def list_signals(request: Request) -> JSONResponse`
- L156: `async def delete_signal(request: Request, payload: SensorSignalDeleteRequest) -> JSONResponse`

### src/argus/dashboard/routes/streaming.py
- L26: `def _preview_gateway(request: Request) -> PreviewGateway`
- L30: `def _find_camera_config(request: Request, camera_id: str)`
- L35: `def stream_info(request: Request, camera_id: str)`
- L46: `def streams_list(request: Request)`
- L53: `def register_stream(request: Request, camera_id: str)`

### src/argus/dashboard/routes/system.py
- L19: `def mode_summary(request: Request)`
- L45: `def health(request: Request)`
- L70: `def _system_snapshots(camera_manager) -> list[dict]`
- L83: `def _snapshot_section(camera_manager, section: str) -> list[dict]`
- L94: `async def drift_status(request: Request)`
- L105: `async def camera_health(request: Request)`
- L116: `async def ensemble_status(request: Request)`
- L127: `def prometheus_metrics()`
- L136: `def anomaly_degradation(request: Request)`

### src/argus/dashboard/routes/tasks.py
- L13: `def tasks_list_json(request: Request)`
- L36: `async def dismiss_task(request: Request, task_id: str)`

### src/argus/dashboard/routes/training_jobs.py
- L44: `def _deny_without_training_permission(request: Request) -> JSONResponse | None`
- L50: `def _get_db(request: Request)`
- L54: `def _write_audit(db, user: str, action: str, target_id: str, detail: str) -> None`
- L70: `def _get_pending_job(db, job_id: str)`
- L83: `async def list_training_jobs(request: Request, status: str | None=None, job_type: str | None=None, camera_id: str | None=None, limit: int=50)`
- L107: `async def get_training_job(request: Request, job_id: str)`
- L128: `async def get_training_job_source_alerts(request: Request, job_id: str)`
- L205: `async def create_training_job(request: Request)`
- L313: `async def confirm_training_job(request: Request, job_id: str)`
- L356: `async def reject_training_job(request: Request, job_id: str)`
- L397: `async def list_backbones(request: Request)`

### src/argus/dashboard/routes/users.py
- L20: `def _user_to_dict(user) -> dict`
- L32: `def _audit(request: Request, action: str, target_id: str, detail: str) -> None`
- L49: `async def users_json(request: Request, username: str | None=Query(None), display_name: str | None=Query(None), role: str | None=Query(None))`
- L82: `async def create_user_json(request: Request)`
- L119: `async def update_user_json(request: Request, username: str)`
- L163: `async def delete_user_json(request: Request, username: str)`
- L184: `async def toggle_active_json(request: Request, username: str)`

### src/argus/dashboard/routes/zones.py
- L23: `def _deny_without_zone_edit(request: Request)`
- L29: `def _persist_zones(request: Request, camera_id: str, zones: list) -> None`
- L54: `class ZoneCreateRequest(BaseModel)`
- L65: `async def create_zone(request: Request, zone_req: ZoneCreateRequest)`
- L126: `class ZoneBulkItem(BaseModel)`
- L135: `async def update_zones(request: Request, camera_id: str, payload: list[ZoneBulkItem])`
- L192: `async def delete_zone(request: Request, camera_id: str, zone_id: str)`
- L232: `def camera_snapshot(request: Request, camera_id: str)`

### src/argus/dashboard/tasks.py
- L22: `class TaskStatus(str, Enum)`
- L31: `def _set_event() -> threading.Event`
- L37: `class TaskAbortedError(Exception)`
- L42: `class TaskInfo`
- L59: `class TaskManager`
  - L68: `def __init__(self, max_concurrent: int=2, on_change: Callable[[str, dict], None] | None=None)`
  - L74: `def submit(self, task_type: str, func: Callable, camera_id: str='', **kwargs) -> str`
  - L151: `def _notify_task_change(self, info: TaskInfo) -> None`
  - L176: `def pause_task(self, task_id: str) -> bool`
  - L189: `def resume_task(self, task_id: str) -> bool`
  - L202: `def abort_task(self, task_id: str) -> bool`
  - L212: `def get_task(self, task_id: str) -> TaskInfo | None`
  - L216: `def get_all_tasks(self) -> list[TaskInfo]`
  - L220: `def get_active_tasks(self) -> list[TaskInfo]`
  - L229: `def dismiss(self, task_id: str) -> bool`
  - L237: `def cleanup_old(self, max_age_seconds: float=3600) -> int`

### src/argus/dashboard/websocket.py
- L69: `def allowed_topics_for_role(role: str) -> frozenset[str]`
- L83: `def make_ws_event(*, topic: str, event_type: str, data: dict[str, Any], resource_id: str, revision: int, event_id: str | None=None, occurred_at: str | None=None) -> dict[str, Any]`
- L106: `def _infer_resource_id(topic: str, data: dict[str, Any]) -> str`
- L122: `class _ClientConnection`
  - L127: `def __init__(self, client_id: str, websocket: WebSocket, subscriptions: set[str], role: str)`
- L141: `class ConnectionManager`
  - L149: `def __init__(self, heartbeat_seconds: int=30, max_connections: int=100)`
  - L170: `async def start(self) -> None`
  - L177: `async def stop(self) -> None`
  - L200: `async def accept(self, websocket: WebSocket, client_id: str, role: str='viewer') -> _ClientConnection`
  - L223: `async def disconnect(self, client_id: str) -> None`
  - L232: `def broadcast(self, topic: str, data: dict, *, event_type: str | None=None, resource_id: str | None=None, revision: int | None=None, event_id: str | None=None, occurred_at: str | None=None) -> bool`
  - L318: `async def handle_client(self, client: _ClientConnection) -> None`
  - L343: `async def _handle_subscribe(self, client: _ClientConnection, msg: dict[str, Any]) -> None`
  - L447: `async def _handle_unsubscribe(self, client: _ClientConnection, msg: dict[str, Any]) -> None`
  - L465: `async def _send_control(self, client: _ClientConnection, event_type: str, data: dict[str, Any]) -> None`
  - L484: `async def _dispatch_loop(self) -> None`
  - L508: `async def _heartbeat_loop(self) -> None`
  - L539: `async def _close_client(self, client: _ClientConnection) -> None`
  - L548: `def connection_count(self) -> int`
- L552: `def resolve_ws_identity(token: str, auth_config: AuthConfig, session_secret: str='') -> dict[str, str] | None`
- L578: `def verify_ws_token(token: str, auth_config: AuthConfig, session_secret: str='') -> bool`

### src/argus/imaging/acquisition.py
- L47: `class FrameMetadata`
- L72: `class IndustrialCameraCapture`
  - L83: `def __init__(self, source: int | str=0, backend: Literal['opencv', 'arena', 'spinnaker', 'metavision', 'dv']='opencv', is_polarization: bool=False) -> None`
  - L140: `def is_polarization_capable(self) -> bool`
  - L148: `def open(self) -> bool`
  - L160: `def close(self) -> None`
  - L170: `def is_opened(self) -> bool`
  - L182: `def grab_frame(self) -> tuple[np.ndarray | None, FrameMetadata]`
  - L199: `def set_nir_strobe(self, enabled: bool) -> None`
  - L220: `def _open_opencv(self) -> bool`
  - L238: `def _close_opencv(self) -> None`
  - L244: `def _grab_opencv(self) -> tuple[np.ndarray | None, FrameMetadata]`
  - L272: `def _read_exposure_us_opencv(self) -> float`
  - L286: `def _open_arena(self) -> bool`
  - L348: `def _arena_configure_polarization(self) -> None`
  - L367: `def _close_arena(self) -> None`
  - L378: `def _grab_arena(self) -> tuple[np.ndarray | None, FrameMetadata]`
  - L432: `def _arena_set_nir_strobe(self, enabled: bool) -> None`
  - L453: `def _open_spinnaker(self) -> bool`
  - L518: `def _spinnaker_configure_polarization(self) -> None`
  - L533: `def _close_spinnaker(self) -> None`
  - L550: `def _grab_spinnaker(self) -> tuple[np.ndarray | None, FrameMetadata]`
  - L612: `def _spinnaker_set_nir_strobe(self, enabled: bool) -> None`
  - L645: `def __enter__(self) -> IndustrialCameraCapture`
  - L649: `def __exit__(self, *_exc: object) -> None`

### src/argus/imaging/fusion.py
- L24: `class FusedFrame`
- L47: `class ModalityFusion`
  - L55: `def __init__(self, fusion_channels: int=3) -> None`
  - L65: `def fuse(self, visible: np.ndarray, dolp: Optional[np.ndarray]=None, nir: Optional[np.ndarray]=None) -> FusedFrame`
  - L117: `def _to_single_channel_uint8(img: np.ndarray) -> np.ndarray`
  - L138: `def _dolp_to_uint8(dolp: Optional[np.ndarray], h: int, w: int) -> np.ndarray`
  - L148: `def _adapt_channels(base: np.ndarray, target: int) -> np.ndarray`

### src/argus/imaging/polarization.py
- L36: `class PolarizationResult`
- L57: `class PolarizationProcessor`
  - L66: `def __init__(self, deglare_method: Literal['stokes', 'min_intensity']='stokes', dolp_threshold: float=0.3) -> None`
  - L88: `def process(self, raw_frame: np.ndarray) -> PolarizationResult`
  - L158: `def _deglare_stokes(s0: np.ndarray, dolp: np.ndarray) -> np.ndarray`
  - L166: `def _deglare_min_intensity(channels: np.ndarray) -> np.ndarray`
  - L177: `def _passthrough(raw_frame: np.ndarray) -> PolarizationResult`

### src/argus/person/detector.py
- L53: `def get_shared_yolo(model_name: str) -> Any`
- L78: `class ObjectDetection`
- L92: `class ObjectDetectionResult`
- L109: `class YOLOObjectDetector`
  - L120: `def __init__(self, model_name: str='yolo11n.pt', confidence: float=0.4, skip_frame_on_person: bool=False, mask_padding: int=20, shared_model: Any | None=None, classes_to_detect: list[int] | None=None, enable_tracking: bool=False, sahi_enabled: bool=False, sahi_slice_size: int=640, sahi_overlap_ratio: float=0.25, camera_id: str='unknown')`
  - L157: `def _ensure_model(self)`
  - L195: `def _detect_sahi(self, frame: np.ndarray) -> list[ObjectDetection]`
  - L238: `def detect(self, frame: np.ndarray) -> ObjectDetectionResult`
  - L330: `def detect_batch(self, frames: list[np.ndarray]) -> list[ObjectDetectionResult]`

### src/argus/physics/calibration.py
- L27: `class CalibrationData`
  - L44: `def to_dict(self) -> dict`
  - L56: `def from_dict(cls, data: dict) -> CalibrationData`
- L69: `class WorldPoint`
- L77: `class CameraCalibration`
  - L87: `def __init__(self, calibration_data: CalibrationData, pool_surface_z_mm: float=0.0) -> None`
  - L113: `def from_file(cls, path: str | Path, pool_surface_z_mm: float=0.0) -> CameraCalibration`
  - L125: `def data(self) -> CalibrationData`
  - L129: `def camera_position_world(self) -> WorldPoint`
  - L134: `def pixel_to_world(self, px: float, py: float, z_mm: float | None=None) -> WorldPoint`
  - L165: `def world_to_pixel(self, x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float]`
  - L175: `def pixel_distance_to_mm(self, px1: float, py1: float, px2: float, py2: float) -> float`
- L184: `class CalibrationTool`
  - L191: `def __init__(self, camera_id: str, board_size: tuple[int, int]=(9, 6), square_size_mm: float=80.0) -> None`
  - L204: `def add_frame(self, frame: np.ndarray) -> bool`
  - L234: `def frame_count(self) -> int`
  - L237: `def calibrate(self) -> CalibrationData`
  - L276: `def save(self, data: CalibrationData, path: str | Path) -> None`
  - L283: `def draw_corners(self, frame: np.ndarray) -> np.ndarray`

### src/argus/physics/multi_cam.py
- L24: `class CameraObservation`
- L35: `class TriangulationResult`
- L44: `class MultiCameraTriangulator`
  - L59: `def __init__(self, calibrations: dict[str, CameraCalibration], max_time_delta: float=0.05) -> None`
  - L71: `def triangulate(self, observations: list[CameraObservation]) -> TriangulationResult | None`
  - L99: `def refraction_correct(self, world_point: WorldPoint, entry_z_mm: float, n_water: float=1.33) -> WorldPoint`
  - L130: `def _triangulate_stereo(self, obs_a: CameraObservation, obs_b: CameraObservation) -> TriangulationResult | None`
  - L172: `def _triangulate_dlt(self, observations: list[CameraObservation]) -> TriangulationResult | None`
  - L221: `def _undistort(obs: CameraObservation, cal: CameraCalibration) -> np.ndarray`

### src/argus/physics/speed.py
- L24: `class SpeedEstimate`
- L37: `class PixelSpeedEstimator`
  - L54: `def __init__(self, fps: float=15.0, smoothing_window: int=5, pixel_scale_mm_per_px: float | None=None, calibration: object | None=None) -> None`
  - L70: `def estimate(self, track: TrackedAnomaly) -> SpeedEstimate`
  - L120: `def _smooth_velocity(self, history: list[tuple[float, float, float]]) -> tuple[float, float]`
  - L146: `def _detect_falling(self, vy_px_per_sec: float) -> bool`
  - L150: `def _check_freefall_acceleration(self, history: list[tuple[float, float, float]], pixel_scale: float | None) -> bool`

### src/argus/physics/trajectory.py
- L38: `class TrajectoryPoint`
- L48: `class TrajectoryFit`
- L66: `class TrajectoryAnalyzer`
  - L81: `def __init__(self, gravity_ms2: float=9.81, pool_surface_z_mm: float=0.0, min_points: int=5, use_drag_model: bool=True) -> None`
  - L97: `def fit_trajectory(self, points: list[TrajectoryPoint]) -> TrajectoryFit | None`
  - L140: `def estimate_origin(self, fit: TrajectoryFit) -> TrajectoryPoint`
  - L144: `def estimate_landing(self, fit: TrajectoryFit) -> TrajectoryPoint`
  - L152: `def _fit_freefall(self, ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> TrajectoryFit | None`
  - L215: `def _fit_projectile(self, ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> TrajectoryFit | None`
  - L307: `def _fit_drag_projectile(self, ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> TrajectoryFit | None`
- L398: `def trajectory_points_from_pixel_history(history: list[tuple[float, float, float]], calibration: CameraCalibration) -> list[TrajectoryPoint]`

### src/argus/prefilter/fast_motion.py
- L19: `class FastMotionCandidate`
  - L33: `def to_detected_object(self) -> dict`
- L57: `class FastMotionResult`
- L68: `class FastMotionDetector`
  - L71: `def __init__(self, *, process_width: int=960, diff_threshold: int=18, background_alpha: float=0.03, min_area_px: int=2, max_area_px: int=1500, min_streak_length_px: int=4, min_confidence: float=0.6, max_motion_fraction: float=0.015, max_streak_frame_fraction: float=0.25, streak_gap_px: int=4, max_candidates_per_frame: int=5, fps_hint: float | None=None, trajectory_history_length: int=16) -> None`
  - L108: `def process(self, frame: np.ndarray, *, timestamp: float | None=None) -> FastMotionResult`
  - L208: `def reset(self) -> None`
  - L215: `def _extract_candidates(self, *, motion: np.ndarray, mask: np.ndarray, raw_mask: np.ndarray, frame_shape: tuple[int, int], scale: float, timestamp: float) -> tuple[list[FastMotionCandidate], np.ndarray]`
  - L304: `def _assign_tracks(self, candidates: list[FastMotionCandidate], timestamp: float) -> None`
  - L363: `def _prune_inactive_track_histories(self, candidates: list[FastMotionCandidate]) -> None`
  - L377: `def _append_history(self, candidate: FastMotionCandidate, timestamp: float) -> None`

### src/argus/prefilter/mog2.py
- L23: `class PreFilterResult`
- L31: `class MOG2PreFilter`
  - L39: `def __init__(self, history: int=500, var_threshold: float=25.0, detect_shadows: bool=True, change_pct_threshold: float=0.005, learning_rate: float=-1, denoise: bool=True, enable_stabilization: bool=True)`
  - L92: `def _align_frame(self, frame: np.ndarray, gray_frame: np.ndarray | None=None) -> np.ndarray`
  - L166: `def process(self, frame: np.ndarray, learning_rate_override: float | None=None, gray_frame: np.ndarray | None=None) -> PreFilterResult`
  - L231: `def _suppress_chromaticity_shadows(frame: np.ndarray, mask: np.ndarray) -> np.ndarray`
  - L254: `def reset(self) -> None`

### src/argus/prefilter/simple_detector.py
- L31: `class StaticRegion`
- L45: `class SimplexResult`
- L53: `class SimplexDetector`
  - L69: `def __init__(self, diff_threshold: int=30, min_area_px: int=500, min_static_seconds: float=30.0, morph_kernel_size: int=5, match_radius_px: int=50)`
  - L87: `def set_reference(self, frame: np.ndarray) -> None`
  - L96: `def detect(self, frame: np.ndarray) -> SimplexResult`
  - L203: `def reset(self) -> None`

### src/argus/preprocessing/alignment.py
- L29: `class _AlignmentCfg(Protocol)`
- L36: `class PhaseCorrelator`
  - L49: `def __init__(self, max_shift_px: float=5.0, downsample: int=4, ref_update_interval_s: float=60.0) -> None`
  - L73: `def reset(self) -> None`
  - L80: `def set_reference(self, frame: np.ndarray, *, now: float | None=None, pinned: bool=False) -> None`
  - L104: `def align(self, frame: np.ndarray, *, now: float | None=None) -> tuple[np.ndarray, tuple[float, float]]`
  - L171: `def _downsample_gray(self, frame: np.ndarray) -> np.ndarray`
- L187: `def create_from_config(cfg: _AlignmentCfg) -> PhaseCorrelator`

### src/argus/preprocessing/scene_guard.py
- L26: `class SceneState(str, Enum)`
- L38: `class SceneGuardConfig`
  - L78: `def __post_init__(self) -> None`
- L131: `class SceneMatchMetrics`
  - L159: `def to_dict(self) -> dict[str, Any]`
- L169: `class SceneGuardResult`
  - L180: `def usable(self) -> bool`
- L185: `class SceneGuard`
  - L195: `def __init__(self, reference_frame: np.ndarray | None=None, *, config: SceneGuardConfig | None=None, process_width: int=960, feature_type: str='ORB', min_matches: int=24, min_inliers: int=12, min_inlier_ratio: float=0.4, max_reprojection_error_px: float=3.0, min_coverage_ratio: float=0.06, min_overlap_ratio: float=0.72, stable_max_shift_px: float=3.0, stable_max_corner_displacement_px: float=3.0, stable_max_rotation_deg: float=0.1, stable_max_scale_delta: float=0.003, alignable_max_shift_px: float=100.0, alignable_max_rotation_deg: float=12.0, alignable_max_scale_delta: float=0.18, consecutive_matches: int=3, return_aligned_frame: bool=False, **advanced: Any) -> None`
  - L258: `def _create_detector(config: SceneGuardConfig) -> tuple[cv2.Feature2D, int]`
  - L277: `def has_reference(self) -> bool`
  - L281: `def reference_frame(self) -> np.ndarray | None`
  - L288: `def state(self) -> SceneState`
  - L293: `def set_reference(self, reference_frame: np.ndarray) -> None`
  - L315: `def clear_reference(self) -> None`
  - L327: `def require_reconfirmation(self) -> None`
  - L343: `def evaluate(self, frame: np.ndarray, *, return_aligned: bool | None=None) -> SceneGuardResult`
  - L584: `def _result(self, raw_state: SceneState, reason: str, metrics: SceneMatchMetrics, *, aligned_frame: np.ndarray | None=None, homography: np.ndarray | None=None) -> SceneGuardResult`
  - L608: `def _debounce(self, candidate: SceneState) -> tuple[SceneState, int]`
  - L652: `def _find_homography(self, source: np.ndarray, destination: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, str]`
  - L678: `def _unreasonable_transform_reason(self, *, shift_px: float, rotation_deg: float, scale_delta: float, anisotropy: float, perspective: float, overlap_ratio: float) -> str | None`
  - L718: `def _transform_metrics(homography: np.ndarray, frame_shape: tuple[int, int], reference_shape: tuple[int, int]) -> dict[str, float]`
  - L770: `def _coverage(points: np.ndarray, shape: tuple[int, int]) -> float`
  - L779: `def _reprojection_error(source: np.ndarray, destination: np.ndarray, homography: np.ndarray) -> float`
  - L791: `def _overlap_ratio(homography: np.ndarray, frame_shape: tuple[int, int], reference_shape: tuple[int, int]) -> float`
  - L835: `def _points_to_full(points: np.ndarray, scale: tuple[float, float]) -> np.ndarray`
  - L845: `def _to_full_resolution_homography(homography: np.ndarray, frame_scale: tuple[float, float], reference_scale: tuple[float, float]) -> np.ndarray`
  - L857: `def _resize_for_processing(self, gray: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]`
  - L873: `def _extract_features(self, gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]`
  - L880: `def _blocked_reason(self, brightness: float, contrast: float, blur: float) -> str | None`
  - L897: `def _to_gray(frame: np.ndarray) -> np.ndarray`
  - L911: `def _validate_frame(frame: np.ndarray) -> None`

### src/argus/preprocessing/scene_profile.py
- L50: `def runtime_artifact_sha256(model_path: str | Path) -> str`
- L138: `def normalize_runtime_artifact_sha256(value: object) -> str`
- L146: `class SceneProfileError(RuntimeError)`
- L150: `class SceneProfileCorruptError(SceneProfileError)`
- L155: `class SceneProfile`
- L171: `class SceneProfileStore`
  - L184: `def __init__(self, root: str | Path, *, jpeg_quality: int=95) -> None`
  - L193: `def create_from_baseline(self, camera_id: str, profile_id: str, baseline_image: str | Path, *, metadata: Mapping[str, Any] | None=None, activate: bool=False) -> SceneProfile`
  - L291: `def load(self, camera_id: str, profile_id: str) -> SceneProfile`
  - L376: `def list_profiles(self, camera_id: str) -> list[SceneProfile]`
  - L398: `def set_active(self, camera_id: str, profile_id: str) -> SceneProfile`
  - L414: `def get_active_profile_id(self, camera_id: str) -> str | None`
  - L442: `def get_active(self, camera_id: str) -> SceneProfile | None`
  - L450: `def clear_active(self, camera_id: str) -> bool`
  - L464: `def _validate_identifier(value: str, label: str) -> str`
  - L473: `def _camera_dir(self, camera_id: str) -> Path`
  - L478: `def _profile_dir(self, camera_id: str, profile_id: str) -> Path`
  - L483: `def _assert_contained(self, path: Path) -> None`
  - L489: `def _reject_symlink(self, path: Path, label: str) -> None`
  - L495: `def _write_bytes(path: Path, content: bytes) -> None`
  - L502: `def _write_json(path: Path, payload: Mapping[str, Any]) -> None`
  - L512: `def _write_json_atomic(self, path: Path, payload: Mapping[str, Any]) -> None`
  - L522: `def _read_json(path: Path, label: str) -> dict[str, Any]`
  - L532: `def _is_sha256(value: object) -> bool`

### src/argus/runtime/dev_training.py
- L15: `def write_dev_openvino_ir(target_dir: Path, *, camera_id: str, zone_id: str, model_type: str, image_size: int, quantization: str) -> None`
- L71: `class DevFastModelTrainer`
  - L74: `def __init__(self, *, baseline_manager, models_dir: str | Path, exports_dir: str | Path) -> None`
  - L88: `def exports_dir(self) -> Path`
  - L92: `def status(self) -> TrainingStatus`
  - L96: `def last_result(self) -> TrainingResult | None`
  - L99: `def train(self, *, camera_id: str, zone_id: str='default', model_type: str='patchcore', image_size: int=256, export_format: str | None='openvino', quantization: str='fp16', **_: object) -> TrainingResult`
  - L163: `def reexport_model(self, *, model_dir: Path, export_format: str='openvino', quantization: str='fp16', model_type: str='patchcore') -> dict`
  - L192: `def _write_openvino_stub(target_dir: Path, *, camera_id: str, zone_id: str, model_type: str, image_size: int, quantization: str) -> None`

### src/argus/runtime/dev_video.py
- L13: `def _base_frame(width: int, height: int) -> np.ndarray`
- L32: `def _draw_foreign_object(frame: np.ndarray, x: int, y: int) -> None`
- L47: `def _draw_book(frame: np.ndarray, x: int, y: int, width: int, height: int) -> None`
- L83: `def _draw_projectile(frame: np.ndarray, x: int, y: int, length: int, thickness: int) -> None`
- L94: `def create_dev_video(output: Path, *, width: int=640, height: int=480, fps: int=10, seconds: int=20, anomaly_start_s: float=6.0, motion: str='settle') -> dict[str, int | str]`

### src/argus/runtime/logging_setup.py
- L37: `def setup_file_logging(config, log_level: int) -> None`
- L83: `def log_gpu_environment() -> None`

### src/argus/runtime/training_job_wiring.py
- L17: `def register_training_job_processing(scheduler: TaskScheduler, *, config, database: Database, baseline_manager, model_trainer, camera_manager: CameraManager | None=None, interval_seconds: float=60.0) -> None`

### src/argus/sensors/fusion.py
- L33: `class SensorFusionDecision`
  - L45: `def active(self) -> bool`
  - L48: `def to_dict(self) -> dict[str, object]`
- L62: `class SensorFusionHead`
  - L77: `def from_config(cls, config: object | None) -> 'SensorFusionHead'`
  - L99: `def evaluate(self, sensor_type: str, value: float, unit: str | None=None) -> SensorFusionDecision`
  - L172: `def _threshold_decision(self, *, sensor_type: str, value: float, unit: str, warning: float, danger: float, metric: str) -> SensorFusionDecision`
- L211: `class _SignalEntry`
- L217: `class SensorFusion`
  - L220: `def __init__(self, enabled: bool=False, default_valid_for_s: float=_DEFAULT_VALID_FOR_S, head: SensorFusionHead | None=None) -> None`
  - L234: `def set_signal(self, camera_id: str, zone_id: str, multiplier: float, valid_for_s: float | None=None, metadata: dict[str, object] | None=None) -> None`
  - L265: `def set_reading(self, camera_id: str, zone_id: str, sensor_type: str, value: float, *, unit: str | None=None, source: str | None=None, valid_for_s: float | None=None) -> SensorFusionDecision`
  - L299: `def get_multiplier(self, camera_id: str, zone_id: str, now: float | None=None) -> float`
  - L329: `def remove_signal(self, camera_id: str, zone_id: str) -> bool`
  - L334: `def active_signals(self, now: float | None=None) -> list[dict[str, object]]`
  - L361: `def clear(self, keys: Iterable[tuple[str, str]] | None=None) -> None`

### src/argus/storage/alert_recording.py
- L37: `class RecordingAvailability`
- L47: `def _find_video_file(rec_dir: Path) -> Path | None`
- L56: `def _json_safe(value: Any) -> Any`
- L77: `def _json_dumps(value: Any) -> str`
- L82: `def _aggregate_trajectories(frames: list[FrameSnapshot], frame_index_offset: int=0) -> dict[str, list[dict]]`
- L106: `def _aggregate_candidate_events(frames: list[FrameSnapshot], frame_index_offset: int=0) -> list[dict]`
- L123: `def _encode_heatmap_jpeg(heatmap_raw: np.ndarray, quality: int=60) -> bytes | None`
- L134: `def _detect_frame_dimensions(frames: list[FrameSnapshot]) -> tuple[int, int]`
- L144: `def _frame_identity_dict(snap: FrameSnapshot | None) -> dict | None`
- L150: `def _recording_frame_identity(recording: SolidifiedRecording) -> dict | None`
- L165: `class AlertRecordingStore`
  - L168: `def __init__(self, archive_dir: str='data/recordings', video_crf: int=23, video_preset: str='veryfast')`
  - L179: `def repair_all(self) -> int`
  - L222: `def save(self, recording: SolidifiedRecording) -> tuple[str, int]`
  - L330: `def append_post_frames(self, alert_id: str, post_frames: list[FrameSnapshot]) -> bool`
  - L481: `def load_metadata(self, alert_id: str) -> dict | None`
  - L491: `def load_signals(self, alert_id: str) -> dict | None`
  - L501: `def get_availability(self, alert_id: str) -> RecordingAvailability`
  - L549: `def load_frame(self, alert_id: str, index: int) -> bytes | None`
  - L559: `def load_heatmap_frame(self, alert_id: str, index: int) -> bytes | None`
  - L569: `def get_video_path(self, alert_id: str) -> Path | None`
  - L577: `def archive_dir(self) -> Path`
  - L581: `def pin_frame(self, alert_id: str, index: int, label: str) -> bool`
  - L596: `def list_clips(self, alert_id: str) -> list[dict] | None`
  - L615: `def add_clip(self, alert_id: str, start_index: int, end_index: int, label: str='') -> dict | None`
  - L648: `def delete_clip(self, alert_id: str, index: int) -> bool`
  - L665: `def has_recording(self, alert_id: str) -> bool`
  - L669: `def reindex_moov(self) -> int`
  - L765: `def delete_recording(self, alert_id: str) -> bool`
  - L774: `def cleanup_old(self, max_age_days: int=30) -> int`
  - L816: `def disk_usage(self) -> int`
  - L823: `def _get_video_file(rec_dir: Path) -> Path | None`
  - L827: `def _find_recording_dir(self, alert_id: str) -> Path | None`
  - L848: `def _build_signals(frames: list[FrameSnapshot]) -> dict`
  - L881: `def _archive_recording(rec_dir: Path) -> None`

### src/argus/storage/audit.py
- L16: `class AuditLogger`
  - L19: `def __init__(self, database: Database)`
  - L22: `def log(self, user: str, action: str, target_type: str, target_id: str='', detail: str='', ip_address: str='') -> None`
  - L48: `def get_logs(self, user: str | None=None, action: str | None=None, limit: int=100, offset: int=0) -> list[AuditLog]`
  - L65: `def count_logs(self, user: str | None=None, action: str | None=None) -> int`

### src/argus/storage/backup.py
- L19: `class BackupManager`
  - L22: `def __init__(self, database_url: str='sqlite:///data/db/argus.db', backup_dir: str | Path='data/backups', max_backups: int=10)`
  - L33: `def create_backup(self, include_models: bool=False) -> dict`
  - L88: `def restore_database(self, backup_name: str) -> bool`
  - L111: `def list_backups(self) -> list[dict]`
  - L143: `def delete_backup(self, backup_name: str) -> bool`
  - L152: `def _rotate_backups(self) -> None`

### src/argus/storage/continuous_recorder.py
- L29: `class _FFmpegWriter`
  - L32: `def __init__(self, path: Path, ffmpeg: str, width: int, height: int, fps: int, crf: int=26, preset: str='veryfast') -> None`
  - L62: `def write(self, frame: np.ndarray) -> None`
  - L69: `def isOpened(self) -> bool`
  - L72: `def release(self) -> None`
- L85: `class RecordingSegment`
- L96: `class ContinuousRecorder`
  - L117: `def __init__(self, camera_id: str, output_dir: Path, segment_duration_hours: float=4.0, encoding_crf: int=26, encoding_preset: str='veryfast', encoding_fps: int=10, resolution: tuple[int, int]=(1920, 1080)) -> None`
  - L150: `def start(self) -> None`
  - L167: `def stop(self) -> None`
  - L181: `def push_frame(self, frame: np.ndarray, timestamp: float) -> None`
  - L191: `def current_segment(self) -> RecordingSegment | None`
  - L196: `def is_recording(self) -> bool`
  - L203: `def _recording_loop(self) -> None`
  - L230: `def _ensure_segment(self, timestamp: float) -> None`
  - L236: `def _open_segment(self, timestamp: float) -> None`
  - L273: `def _open_ffmpeg_writer(self, seg_path: Path) -> _FFmpegWriter | None`
  - L292: `def _write_frame(self, frame: np.ndarray) -> None`
  - L302: `def _close_segment(self) -> None`
- L321: `class ContinuousRecordingManager`
  - L324: `def __init__(self, output_dir: Path, segment_duration_hours: float=4.0, encoding_crf: int=26, encoding_preset: str='veryfast', encoding_fps: int=10) -> None`
  - L339: `def start_camera(self, camera_id: str, resolution: tuple[int, int]=(1920, 1080)) -> None`
  - L356: `def stop_camera(self, camera_id: str) -> None`
  - L361: `def stop_all(self) -> None`
  - L365: `def push_frame(self, camera_id: str, frame: np.ndarray, timestamp: float) -> None`
  - L372: `def get_segments(self, camera_id: str, start: datetime | None=None, end: datetime | None=None) -> list[RecordingSegment]`
  - L404: `def get_storage_stats(self) -> dict`

### src/argus/storage/database.py
- L158: `class Database`
  - L165: `def __init__(self, database_url: str='sqlite:///data/db/argus.db')`
  - L170: `def initialize(self) -> None`
  - L215: `def _drop_deprecated_tables(self) -> None`
  - L237: `def _auto_migrate(self) -> None`
  - L315: `def _auto_migrate_indexes(self) -> None`
  - L346: `def _is_already_exists_error(err_msg: str) -> bool`
  - L367: `def get_session(self) -> Session`
  - L373: `def save_alert(self, alert_id: str, timestamp: datetime, camera_id: str, zone_id: str, severity: str, anomaly_score: float, snapshot_path: str | None=None, heatmap_path: str | None=None, detection_type: str | None=None, evidence_summary: str | None=None, evidence_json: dict | str | None=None, suppression_reason: str | None=None, event_group_id: str | None=None, event_group_count: int=1, speed_ms: float | None=None, speed_px_per_sec: float | None=None, trajectory_model: str | None=None, origin_x_mm: float | None=None, origin_y_mm: float | None=None, origin_z_mm: float | None=None, landing_x_mm: float | None=None, landing_y_mm: float | None=None, landing_z_mm: float | None=None, trajectories_json: str | None=None, classification_label: str | None=None, classification_confidence: float | None=None, corroborated: bool | None=None, correlation_partner: str | None=None, segmentation_count: int | None=None, segmentation_total_area_px: int | None=None, segmentation_objects: list[dict] | None=None, category: str | None=None, severity_adjusted_by_classifier: bool | None=None, trajectory_points: str | None=None, model_version_id: str | None=None, outbox_event: dict | None=None, _max_retries: int=3) -> AlertRecord`
  - L630: `def get_alerts(self, camera_id: str | None=None, severity: str | None=None, limit: int=100, offset: int=0) -> list[AlertRecord]`
  - L649: `def get_alert(self, alert_id: str) -> AlertRecord | None`
  - L656: `def _lookup_alert_commit_outcome(self, alert_id: str, event_id: str | None) -> tuple[AlertRecord | None, OutboxEvent | None]`
  - L679: `def _append_alert_outbox(session: Session, record: AlertRecord, event_type: str) -> OutboxEvent`
  - L703: `def get_pending_outbox_events(self, limit: int=100) -> list[OutboxEvent]`
  - L714: `def mark_outbox_published(self, event_id: str) -> bool`
  - L729: `def mark_outbox_failed(self, event_id: str, error: str) -> bool`
  - L742: `def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool`
  - L757: `def mark_false_positive(self, alert_id: str, notes: str | None=None) -> bool`
  - L773: `def update_alert_workflow(self, alert_id: str, workflow_status: str, notes: str | None=None, assigned_to: str | None=None) -> bool`
  - L805: `def get_alert_workflow_stats(self) -> dict[str, int]`
  - L817: `def get_alert_count(self, camera_id: str | None=None, severity: str | None=None, since: datetime | None=None, until: datetime | None=None) -> int`
  - L842: `def get_wall_status_batch(self, camera_ids: list[str], since: datetime, active_statuses: tuple[str, ...]=('new', 'acknowledged', 'investigating')) -> dict[str, dict]`
  - L911: `def delete_alert(self, alert_id: str) -> tuple[bool, list[str]]`
  - L943: `def delete_old_alerts(self, days: int=90, include_recording_paths: bool=False) -> tuple[int, list[str]] | tuple[int, list[str], list[str]]`
  - L1012: `def save_training_record(self, record: TrainingRecord) -> TrainingRecord`
  - L1021: `def get_training_history(self, camera_id: str | None=None, zone_id: str | None=None, limit: int=20) -> list[TrainingRecord]`
  - L1037: `def get_latest_training(self, camera_id: str, zone_id: str='default') -> TrainingRecord | None`
  - L1051: `def get_training_record(self, record_id: int) -> TrainingRecord | None`
  - L1060: `def save_training_job(self, record: TrainingJobRecord) -> TrainingJobRecord`
  - L1069: `def get_training_job(self, job_id: str) -> TrainingJobRecord | None`
  - L1076: `def list_training_jobs(self, status: str | None=None, job_type: str | None=None, camera_id: str | None=None, limit: int=50) -> list[TrainingJobRecord]`
  - L1095: `def update_training_job(self, job_id: str, *, expected_status: str | None=None, **kwargs) -> bool`
  - L1117: `def count_pending_jobs(self) -> int`
  - L1127: `def save_backbone(self, record: BackboneRecord) -> BackboneRecord`
  - L1136: `def get_active_backbone(self) -> BackboneRecord | None`
  - L1143: `def list_backbones(self, limit: int=20) -> list[BackboneRecord]`
  - L1150: `def activate_backbone(self, backbone_version_id: str) -> bool`
  - L1177: `def save_inference_batch(self, records: list[InferenceRecord]) -> int`
  - L1186: `def get_inference_stats(self, camera_id: str, model_version_id: str | None=None) -> dict`
  - L1209: `def delete_old_inference_records(self, days: int=30) -> int`
  - L1227: `def save_feedback(self, record: FeedbackRecord) -> FeedbackRecord`
  - L1240: `def get_feedback(self, feedback_id: str) -> FeedbackRecord | None`
  - L1247: `def get_pending_feedback(self, camera_id: str | None=None, feedback_type: str | None=None, limit: int=500) -> list[FeedbackRecord]`
  - L1267: `def mark_feedback_processed(self, feedback_ids: list[str], trained_into: str) -> int`
  - L1305: `def skip_feedback(self, feedback_ids: list[str], reason: str | None=None) -> int`
  - L1326: `def get_feedback_summary(self, camera_id: str | None=None) -> dict`
  - L1347: `def list_feedback(self, camera_id: str | None=None, feedback_type: str | None=None, status: str | None=None, limit: int=100, offset: int=0) -> list[FeedbackRecord]`
  - L1369: `def create_user(self, username: str, password_hash: str, role: str='viewer', display_name: str | None=None) -> User`
  - L1390: `def get_user(self, username: str) -> User | None`
  - L1395: `def get_all_users(self) -> list[User]`
  - L1400: `def update_user(self, username: str, **kwargs) -> bool`
  - L1412: `def delete_user(self, username: str) -> bool`
  - L1423: `def user_count(self) -> int`
  - L1430: `def create_region(self, name: str, owner: str, phone: str | None=None) -> Region`
  - L1445: `def get_region(self, region_id: int) -> Region | None`
  - L1450: `def get_region_by_name(self, name: str) -> Region | None`
  - L1455: `def get_regions(self, *, name: str | None=None, owner: str | None=None, phone: str | None=None) -> list[Region]`
  - L1473: `def update_region(self, region_id: int, **kwargs) -> bool`
  - L1485: `def delete_region(self, region_id: int) -> bool`
  - L1496: `def save_alert_recording(self, record: AlertRecordingRecord) -> AlertRecordingRecord`
  - L1508: `def get_alert_recording(self, alert_id: str) -> AlertRecordingRecord | None`
  - L1517: `def get_alert_recordings_batch(self, alert_ids: list[str]) -> dict[str, AlertRecordingRecord]`
  - L1532: `def update_alert_recording_status(self, alert_id: str, status: str, frame_count: int | None=None, end_timestamp: float | None=None, file_size_bytes: int | None=None) -> bool`
  - L1559: `def delete_alert_recording_metadata(self, alert_id: str) -> str | None`
  - L1575: `def cleanup_old_recordings(self, max_age_days: int=30) -> int`
  - L1591: `def save_labeling_entry(self, record: LabelingQueueRecord) -> LabelingQueueRecord`
  - L1604: `def get_labeling_queue(self, camera_id: str | None=None, status: str | None=None, limit: int=100, offset: int=0) -> list[LabelingQueueRecord]`
  - L1628: `def label_entry(self, entry_id: int, label: str, labeled_by: str) -> LabelingQueueRecord | None`
  - L1649: `def get_labeling_stats(self, camera_id: str | None=None) -> dict`
  - L1679: `def get_labeled_entries(self, camera_id: str | None=None, label: str | None=None, trained_into: str | None=None, limit: int=500) -> list[LabelingQueueRecord]`
  - L1705: `def mark_labeling_trained(self, entry_ids: list[int], trained_into: str) -> int`
  - L1734: `def close(self) -> None`

### src/argus/storage/inference_buffer.py
- L26: `class InferenceBuffer`
  - L38: `def __init__(self, database: Database, flush_seconds: int=60, max_size: int=1000)`
  - L54: `def start(self) -> None`
  - L71: `def stop(self) -> None`
  - L80: `def append(self, record: InferenceRecord) -> None`
  - L103: `def _flush_loop(self) -> None`
  - L109: `def _flush(self) -> None`
  - L146: `def pending_count(self) -> int`
  - L152: `def total_flushed(self) -> int`

### src/argus/storage/inference_store.py
- L25: `class InferenceRecordStore`
  - L35: `def __init__(self, base_dir: Path, max_queue: int=500)`
  - L43: `def start(self) -> None`
  - L56: `def stop(self) -> None`
  - L63: `def submit(self, record: InferenceRecord, raw_frame: np.ndarray | None=None) -> None`
  - L74: `def _writer_loop(self) -> None`
  - L100: `def _write_record(self, record: InferenceRecord, raw_frame: np.ndarray | None) -> None`

### src/argus/storage/model_registry.py
- L32: `class ModelStageMutationDisabledError(RuntimeError)`
- L36: `def _invalid_release_evidence(reason: str) -> ReleaseEvidenceValidation`
- L45: `def validate_persisted_release_evidence(record: ModelRecord) -> ReleaseEvidenceValidation`
- L106: `class ModelRegistry`
  - L111: `def __init__(self, session_factory)`
  - L114: `def register(self, model_path: str | Path, baseline_dir: str | Path, camera_id: str, model_type: str, training_params: dict | None=None, component_type: str='full', backbone_ref: str | None=None, registered_by: str='model_registry') -> str`
  - L200: `def _new_model_version_id(camera_id: str, model_type: str) -> str`
  - L206: `def get_active(self, camera_id: str) -> ModelRecord | None`
  - L236: `def get_runtime_approved_production(self, camera_id: str) -> list[ModelRecord]`
  - L266: `def get_by_stage(self, camera_id: str, stage: str) -> list[ModelRecord]`
  - L276: `def get_by_version_id(self, model_version_id: str) -> ModelRecord | None`
  - L285: `def promote(self, model_version_id: str, target_stage: str, triggered_by: str, reason: str | None=None, canary_camera_id: str | None=None, *, release_evidence: object | None=None) -> ModelRecord`
  - L302: `def activate(self, model_version_id: str, triggered_by: str='system') -> None`
  - L314: `def retire(self, model_version_id: str, reason: str | None=None, triggered_by: str='system') -> ModelRecord | None`
  - L327: `def delete_model(self, model_version_id: str) -> None`
  - L340: `def list_models(self, camera_id: str | None=None) -> list[ModelRecord]`
  - L348: `def rollback(self, camera_id: str, triggered_by: str='system') -> ModelRecord | None`
  - L356: `def compensate_failed_production_sync(self, promoted_version_id: str, previous_version_id: str | None, *, triggered_by: str, reason: str) -> ModelRecord | None`
  - L371: `def get_version_events(self, camera_id: str | None=None, model_version_id: str | None=None, limit: int=50) -> list[ModelVersionEvent]`
  - L392: `def _compute_dir_hash(path: Path) -> str`
  - L397: `def _get_git_hash() -> str | None`

### src/argus/storage/model_release_service.py
- L40: `class RuntimeReleaseIdentity`
- L50: `class RuntimeReleaseCommit`
- L59: `class RuntimeReleaseTransaction(Protocol)`
  - L67: `def commit(self) -> RuntimeReleaseCommit`
  - L70: `def abort(self) -> RuntimeReleaseCommit`
  - L73: `def finalize(self) -> RuntimeReleaseCommit`
- L79: `class RuntimeReleasePreparation`
- L93: `class _NoRuntimeReleaseTransaction`
  - L104: `def commit(self) -> RuntimeReleaseCommit`
  - L119: `def abort(self) -> RuntimeReleaseCommit`
  - L126: `def finalize(self) -> RuntimeReleaseCommit`
- L133: `class ModelReleaseService`
  - L139: `def __init__(self, session_factory, pipeline: ReleasePipeline, *, runtime_prepare: RuntimePrepare, event_publisher: EventPublisher | None=None) -> None`
  - L153: `def _lock_for_camera(cls, camera_id: str) -> threading.RLock`
  - L157: `def get_operation(self, operation_id: str) -> ModelReleaseOperation | None`
  - L165: `def get_latest_operation(self, model_version_id: str, *, target_stage: str | None=None) -> ModelReleaseOperation | None`
  - L190: `def request_transition(self, *, model_version_id: str, target_stage: str, requested_by: str, reason: str | None=None, canary_camera_id: str | None=None, release_evidence: object | None=None, expected_revision: int | None=None, idempotency_key: str | None=None) -> tuple[ModelReleaseOperation, ModelRecord]`
  - L416: `def _prepare_runtime(self, plan: ReleaseTransitionPlan) -> RuntimeReleasePreparation`
  - L461: `def _validate_runtime_preparation(plan: ReleaseTransitionPlan, prepared: RuntimeReleasePreparation) -> str | None`
  - L492: `def _return_idempotent(self, operation: ModelReleaseOperation, *, model_version_id: str, target_stage: str, requested_by: str, desired_state_json: str) -> tuple[ModelReleaseOperation, ModelRecord]`
  - L514: `def _validate_idempotent_command(operation: ModelReleaseOperation, *, model_version_id: str, target_stage: str, requested_by: str, desired_state_json: str) -> None`
  - L536: `def _canonical_desired_state(*, reason: str | None, canary_camera_id: str | None, release_evidence: object | None) -> str`
  - L554: `def _get_model(self, model_version_id: str) -> ModelRecord | None`
  - L562: `def _get_operation_by_idempotency(self, idempotency_key: str) -> ModelReleaseOperation | None`
  - L573: `def _create_operation(self, *, model: ModelRecord, target_stage: str, requested_by: str, reason: str | None, expected_revision: int, idempotency_key: str | None, desired_state_json: str) -> tuple[ModelReleaseOperation, bool]`
  - L615: `def _finish_failed_operation(self, operation_id: str, *, error: str, runtime_state: object=None) -> ModelReleaseOperation`
  - L648: `def _failed_result(self, operation_id: str, model_version_id: str) -> tuple[ModelReleaseOperation, ModelRecord]`
  - L660: `def _abort_runtime(transaction: RuntimeReleaseTransaction | None) -> object`
  - L683: `def _publish(self, operation: ModelReleaseOperation, record: ModelRecord) -> None`

### src/argus/storage/models.py
- L12: `class ModelStage(str, Enum)`
- L22: `class ReleaseOperationStatus(str, Enum)`
- L30: `class AlertWorkflowStatus(str, Enum)`
- L43: `class FeedbackType(str, Enum)`
- L51: `class FeedbackStatus(str, Enum)`
- L59: `class FeedbackSource(str, Enum)`
- L67: `class BaselineState(str, Enum)`
- L76: `class Base(DeclarativeBase)`
- L80: `class AlertRecord(Base)`
  - L156: `def to_dict(self) -> dict`
- L224: `class OutboxEvent(Base)`
  - L254: `def to_dict(self) -> dict`
- L275: `class BaselineRecord(Base)`
- L292: `class BaselineVersionRecord(Base)`
  - L325: `def to_dict(self) -> dict`
- L344: `class TrainingRecord(Base)`
  - L412: `def to_dict(self) -> dict`
- L457: `class AuditLog(Base)`
- L472: `class FeedbackRecord(Base)`
  - L535: `def to_dict(self) -> dict`
- L556: `class InferenceRecord(Base)`
- L588: `class ModelRecord(Base)`
  - L630: `def to_dict(self) -> dict`
- L674: `class ModelReleaseOperation(Base)`
  - L717: `def to_dict(self) -> dict`
- L752: `class TrainingJobStatus(str, Enum)`
- L764: `class TrainingJobType(str, Enum)`
- L771: `class TrainingTriggerType(str, Enum)`
- L779: `class TrainingJobRecord(Base)`
  - L830: `def to_dict(self) -> dict`
- L860: `class BackboneRecord(Base)`
  - L884: `def to_dict(self) -> dict`
- L899: `class User(Base)`
- L914: `class Region(Base)`
- L936: `class ModelVersionEvent(Base)`
  - L955: `def to_dict(self) -> dict`
- L971: `class AlertRecordingRecord(Base)`
  - L1006: `def to_dict(self) -> dict`
- L1027: `class LabelingQueueStatus(str, Enum)`
- L1035: `class LabelingQueueRecord(Base)`
  - L1076: `def to_dict(self) -> dict`
- L1095: `class ShadowInferenceLog(Base)`
- L1114: `class ContinuousRecordingSegment(Base)`

### src/argus/storage/release_evidence.py
- L75: `class ReleaseEvidenceValidation`
- L84: `def canonical_json(value: object) -> str`
- L100: `def canonical_sha256(value: object) -> str`
- L106: `def evidence_artifact_sha256(path: str | Path) -> str`
- L151: `def canonical_training_artifact_root(path: str | Path) -> Path`
- L174: `def registry_training_data_hash(path: str | Path) -> str`
- L202: `def _stream_sha256(path: Path) -> str`
- L210: `def _has_symlink_component(path: Path) -> bool`
- L227: `def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]`
- L239: `def _contained_relative_file(root: Path, raw_relative: object, *, label: str) -> tuple[Path, str]`
- L276: `def _strict_int(value: object, *, label: str, minimum: int=0) -> int`
- L282: `def _strict_finite_number(value: object, *, label: str, minimum: float | None=None) -> float`
- L299: `def _canonical_time_base(value: object, *, label: str) -> tuple[str, Fraction]`
- L312: `def _read_timestamp_jsonl(baseline_root: Path, evidence: object, *, source_index: int) -> list[Mapping[str, Any]]`
- L443: `def _validate_strict_decoder_source(baseline_root: Path, source: Mapping[str, Any], *, source_index: int, source_sha256: str) -> list[Mapping[str, Any]]`
- L638: `def _validate_baseline_manifest(baseline_root: Path, *, registry_data_hash: str) -> tuple[Mapping[str, Any], set[str]]`
- L1116: `def validate_baseline_manifest_source(baseline_root: str | Path, *, registry_data_hash: str | None=None) -> tuple[dict[str, Any], set[str], set[str]]`
- L1158: `def _validate_scene_profile_source(profile_root: Path, baseline_root: Path, baseline_manifest: Mapping[str, Any], *, model_version_id: str, camera_id: str, runtime_artifact_digest: str) -> dict[str, str]`
- L1293: `def _validate_regression_corpus(corpus_root: Path, suite: Mapping[str, Any], *, camera_id: str) -> tuple[set[str], set[str]]`
- L1436: `def _aware_datetime(value: object, *, label: str) -> datetime`
- L1448: `def _validate_soak_report_source(soak_report: Mapping[str, Any], *, expected_subject: Mapping[str, Any], telemetry_path: Path, trust_store_path: Path | None=None) -> tuple[bool, float]`
- L1591: `def soak_attestation_payload(soak_report: Mapping[str, Any]) -> bytes`
- L1599: `def sign_soak_report_hmac(soak_report: Mapping[str, Any], *, secret: bytes) -> str`
- L1616: `def _verify_soak_attestation(soak_report: Mapping[str, Any], *, trust_store_path: Path | None) -> None`
- L1696: `def _exact_mapping(value: object, *, path: str, keys: set[str], reasons: list[str]) -> Mapping[str, Any] | None`
- L1723: `def _required_string(value: object, *, path: str, reasons: list[str]) -> str | None`
- L1738: `def _required_sha256(value: object, *, path: str, reasons: list[str]) -> str | None`
- L1750: `def _required_true(value: object, *, path: str, reasons: list[str]) -> None`
- L1762: `def _finite_number(value: object, *, path: str, reasons: list[str]) -> float | None`
- L1778: `def validate_release_evidence(evidence: object, *, model_version_id: str, camera_id: str, registry_model_hash: str, runtime_artifact_sha256: str) -> ReleaseEvidenceValidation`
- L2150: `def build_release_evidence(regression_report: Mapping[str, Any], *, model_version_id: str, camera_id: str, registry_model_hash: str, config_sha256: str, scene_profile_sha256: str, baseline_sha256: str, corpus_sha256: str, operator_id: str, operator_approved: bool, approved_at: str, soak_passed: bool, soak_duration_hours: float, soak_report_sha256: str) -> dict[str, Any]`
- L2302: `def build_release_evidence_from_sources(sources: object, *, model_version_id: str, camera_id: str, registry_model_hash: str, registry_data_hash: str, runtime_artifact_digest: str, operator_id: str, operator_approved: bool, approved_at: str) -> dict[str, Any]`

### src/argus/storage/release_pipeline.py
- L57: `def _ensure_utc(dt: datetime) -> datetime`
- L62: `class BackboneIncompatibleError(Exception)`
- L66: `class StageTransitionError(Exception)`
- L71: `class ReleaseTransitionPlan`
- L94: `class ReleasePipeline`
  - L97: `def __init__(self, session_factory, *, min_shadow_days: int=3, min_canary_days: int=7, min_shadow_samples: int=0, max_shadow_alert_rate_delta: float=1.0, max_shadow_score_divergence: float=1.0, max_shadow_latency_ms: float=0.0, event_publisher: Callable[[str, dict], None] | None=None)`
  - L121: `def transition(self, model_version_id: str, target_stage: str, triggered_by: str, reason: str | None=None, canary_camera_id: str | None=None, *, now: datetime | None=None, release_evidence: object | None=None, expected_revision: int | None=None) -> ModelRecord`
  - L155: `def prepare_transition(self, *, model_version_id: str, target_stage: str, canary_camera_id: str | None=None, now: datetime | None=None, release_evidence: object | None=None, expected_revision: int | None=None) -> ReleaseTransitionPlan`
  - L260: `def commit_transition(self, plan: ReleaseTransitionPlan, *, triggered_by: str, reason: str | None=None, operation_id: str | None=None, runtime_state: object=None) -> ModelRecord`
  - L487: `def _publish_transition(self, plan: ReleaseTransitionPlan, *, triggered_by: str, reason: str | None) -> None`
  - L518: `def _validate_production_evidence(record: ModelRecord, evidence: object, *, artifact_sha256: str | None=None) -> ReleaseEvidenceValidation`
  - L551: `def get_shadow_stats(self, shadow_version_id: str, camera_id: str | None=None, days: int=7) -> dict`
  - L585: `def get_shadow_gate_report(self, shadow_version_id: str, camera_id: str | None=None, days: int=7) -> dict`
  - L647: `def validate_backbone_compatibility(head_record: ModelRecord, loaded_backbone_version: str) -> None`
  - L668: `def _validate_transition(self, current_stage: str, target_stage: str) -> None`
  - L678: `def _validate_min_stage_days(session: Session, record: ModelRecord, now: datetime, required_stage: str, min_days: int, stage_label: str) -> None`
  - L708: `def _validate_shadow_evidence(self, session: Session, record: ModelRecord, now: datetime) -> None`
  - L775: `def _shadow_stats_from_logs(logs: list[ShadowInferenceLog]) -> dict`
  - L818: `def _retire_current_production(session: Session, camera_id: str, exclude_version_id: str, triggered_by: str='system') -> None`
  - L846: `def _retire_current_canaries(session: Session, camera_id: str, exclude_version_id: str, triggered_by: str='system') -> None`

### src/argus/storage/retention.py
- L27: `class RetentionManager`
  - L55: `def __init__(self, continuous_recording_dir: Path, alert_recording_dir: Path, local_retention_days: int=180, archive_enabled: bool=False, archive_path: Path | None=None, archive_retention_days: int=360, alert_retention_days: int=180, cleanup_interval_hours: float=6.0, database: Database | None=None) -> None`
  - L84: `def start(self) -> None`
  - L101: `def stop(self) -> None`
  - L111: `def run_cleanup(self) -> dict`
  - L133: `def _emergency_cleanup_if_needed(self, min_free_gb: float=1.0) -> int`
  - L172: `def _cleanup_local(self) -> int`
  - L207: `def _archive_recordings(self) -> int`
  - L243: `def _cleanup_archive(self) -> int`
  - L250: `def _cleanup_alert_evidence(self) -> int`
  - L259: `def _delete_old_date_dirs(self, root: Path, cutoff: datetime) -> int`
  - L283: `def _delete_old_alert_recording_dirs(self, root: Path, cutoff: datetime) -> int`
  - L321: `def _iter_alert_recording_dirs(date_dir: Path)`
  - L334: `def _alert_id_from_recording_dir(rec_dir: Path) -> str | None`
  - L351: `def _delete_alert_recording_metadata(self, alert_id: str) -> bool`
  - L369: `def _remove_empty_ancestors(path: Path, stop_at: Path) -> None`
  - L389: `def _parse_date_dir(name: str) -> datetime | None`
  - L396: `def _run_loop(self) -> None`
  - L408: `def get_retention_stats(self) -> dict`

### src/argus/streaming/go2rtc_manager.py
- L37: `class CameraSourceResolution`
- L59: `def _normalise_usb_pixel_format(pixel_format: str | None) -> str | None`
- L72: `def usb_to_go2rtc_source(device_index: str | int, *, device_name: str | None=None, device_id: str | None=None, resolution: tuple[int, int] | list[int] | None=None, fps: int | float | None=None, pixel_format: str | None=None, video_codec: str='h264') -> str`
- L123: `def gige_to_go2rtc_source(script_path: str) -> str`
- L140: `def _find_go2rtc_binary() -> Path | None`
- L162: `class Go2RTCManager`
  - L183: `def __init__(self, *, api_port: int=_DEFAULT_API_PORT, rtsp_port: int=_DEFAULT_RTSP_PORT, webrtc_port: int=_DEFAULT_WEBRTC_PORT, webrtc_listen_host: str='0.0.0.0', binary_path: str | Path | None=None, config_dir: str | Path | None=None, replace_stale_process: bool=False) -> None`
  - L224: `def _write_config(self, streams: dict[str, str | list[str]] | None=None) -> Path`
  - L258: `def _kill_stale_process(self) -> None`
  - L311: `def start(self, initial_streams: dict[str, str] | None=None) -> None`
  - L370: `def stop(self) -> None`
  - L393: `def running(self) -> bool`
  - L396: `def _wait_for_ready(self) -> None`
  - L415: `def _wait_for_tcp_listener(label: str, port: int, deadline: float) -> None`
  - L439: `def register_camera(self, camera_id: str, source: str, protocol: str) -> str | None`
  - L465: `def add_stream(self, name: str, source: str) -> None`
  - L482: `def remove_stream(self, name: str) -> None`
  - L488: `def list_streams(self) -> dict[str, Any]`
  - L498: `def webrtc_url(self, stream_name: str) -> str`
  - L502: `def mse_url(self, stream_name: str) -> str`
  - L506: `def hls_url(self, stream_name: str) -> str`
  - L510: `def mjpeg_url(self, stream_name: str) -> str`
  - L514: `def stream_html_url(self, stream_name: str) -> str`
  - L522: `def sync_cameras(self, cameras: list[dict[str, Any]]) -> None`
  - L556: `def _drain_stdout(self) -> None`
  - L567: `def close(self) -> None`
  - L586: `def __del__(self) -> None`
- L598: `def _camera_attr(camera: Any, name: str, default: Any=None) -> Any`
- L604: `def _gige_capture_script(camera: Any) -> str | None`
- L611: `def _source_resolution(*, camera_id: str, source: str, protocol: str, rtsp_url: str | None) -> CameraSourceResolution`
- L635: `def register_go2rtc_stream(manager: Go2RTCManager, camera: Any) -> CameraSourceResolution | None`
- L644: `def _gige_initial_streams(cameras: list) -> dict[str, str]`
- L658: `def _manager_stream_registry(manager: Go2RTCManager)`
- L674: `def register_go2rtc_streams(manager: Go2RTCManager, cameras: list) -> dict[str, CameraSourceResolution]`
- L684: `def runtime_camera_config(camera: Any, resolution: CameraSourceResolution) -> Any`
- L691: `def start_and_register_cameras(manager: Go2RTCManager, cameras: list) -> dict[str, CameraSourceResolution]`

### src/argus/streaming/preview_gateway.py
- L27: `class Go2RTCProxyTarget`
- L35: `def _camera_attr(camera: Any, name: str, default: Any=None) -> Any`
- L41: `def _manager_running(manager: Any) -> bool`
- L45: `def _is_loopback_host(host: str | None) -> bool`
- L52: `def _encode_jpeg(frame: Any, quality: int) -> bytes | None`
- L59: `def _frame_identity_overlay_enabled(state: Any) -> bool`
- L68: `def _draw_frame_identity_overlay(frame: Any, identity: FrameIdentity | None) -> Any`
- L101: `def _frame_identity_headers(identity: FrameIdentity | None) -> dict[str, str]`
- L119: `def probe_source_blocking(source: str | int, timeout: float=5.0, protocol: str | None=None) -> dict[str, Any]`
- L172: `class PreviewGateway`
  - L175: `def __init__(self, state: Any, *, stream_executor: Executor | None=None, stream_semaphore: asyncio.Semaphore | None=None, max_stream_duration: float=30 * 60, source_probe: SourceProbe=probe_source_blocking) -> None`
  - L191: `def for_request(cls, request: Request, **kwargs: Any) -> PreviewGateway`
  - L194: `def find_camera_config(self, camera_id: str) -> Any | None`
  - L229: `def camera_manager(self) -> Any | None`
  - L232: `def stream_manager(self) -> Any | None`
  - L240: `def stream_manager_running(self) -> bool`
  - L243: `def snapshot_response(self, camera_id: str) -> Response`
  - L289: `def latest_frame_stream_response(self, request: Request, camera_id: str) -> Response`
  - L306: `def heatmap_stream_response(self, request: Request, camera_id: str) -> Response`
  - L336: `def streaming_info(self, request: Request, camera: Any) -> dict[str, Any]`
  - L354: `def resolve_proxy_target(self, camera: Any) -> Go2RTCProxyTarget | None`
  - L370: `def ensure_stream_registered(self, camera: Any) -> Any | None`
  - L399: `def list_registered_streams(self) -> tuple[bool, dict[str, Any]]`
  - L412: `def probe_running_camera_connection(self, camera_id: str) -> dict[str, Any] | None`
  - L454: `async def probe_source(self, source: str | int, timeout: float=5.0, protocol: str | None=None) -> dict[str, Any]`
  - L462: `def _mjpeg_response(self, request: Request, grab_fn: Callable[[], bytes | None]) -> Response`
  - L499: `def _stream_registry(self) -> Any | None`
  - L502: `def _external_go2rtc_proxy_target(self, camera: Any, camera_id: str) -> Go2RTCProxyTarget | None`
  - L541: `def _same_origin_streaming_info(request: Request, camera_id: str, fallback: str) -> dict[str, Any]`

### src/argus/streaming/stream_registry.py
- L31: `class StreamDeclaration`
  - L42: `def __post_init__(self) -> None`
  - L47: `def registration_signature(self) -> tuple[str, str, str | None]`
  - L51: `def supported_by_go2rtc(self) -> bool`
- L57: `def _attr(obj: Any, name: str, default: Any=None) -> Any`
- L63: `def _has_attr(obj: Any, name: str) -> bool`
- L69: `def _gige_capture_script(camera: Any) -> str | None`
- L81: `def _usb_go2rtc_source(camera: Any, source: str, protocol: str) -> str | None`
- L109: `def _is_iterable_plan(value: Any) -> bool`
- L116: `def _plan_items(plan: Any) -> list[Any]`
- L147: `def _looks_like_camera(value: Any) -> bool`
- L154: `def _declaration_from_camera(camera: Any, *, camera_id: str | None=None, source: str | None=None, protocol: str | None=None, stream_name: str | None=None, capture_script: str | None=None) -> StreamDeclaration | None`
- L211: `def _resolution_for(declaration: StreamDeclaration, rtsp_url: str | None) -> CameraSourceResolution`
- L233: `def runtime_camera_config(camera: Any, resolution: CameraSourceResolution) -> Any`
- L254: `class StreamRegistry`
  - L262: `def __init__(self, manager: Any) -> None`
  - L270: `def desired_streams(self) -> dict[str, StreamDeclaration]`
  - L274: `def resolutions(self) -> dict[str, CameraSourceResolution]`
  - L277: `def declare_plan(self, plan: Any) -> dict[str, StreamDeclaration]`
  - L288: `def declare_camera(self, camera: Any=None, *, camera_id: str | None=None, source: str | None=None, protocol: str | None=None, stream_name: str | None=None, capture_script: str | None=None) -> StreamDeclaration | None`
  - L321: `def ensure_registered(self, camera: Any=None, *, start_if_needed: bool=False, camera_id: str | None=None, source: str | None=None, protocol: str | None=None, stream_name: str | None=None, capture_script: str | None=None) -> CameraSourceResolution | None`
  - L351: `def undeclare_camera(self, camera_or_id: Any, *, remove_runtime: bool=True) -> bool`
  - L379: `def reconcile(self, plan: Any=None, *, start_if_needed: bool=True, remove_stale: bool=True) -> dict[str, CameraSourceResolution]`
  - L417: `def runtime_camera_config(self, camera: Any, resolution: CameraSourceResolution | None=None) -> Any`
  - L437: `def _ensure_started(self) -> bool`
  - L467: `def _restore_desired_registrations(self) -> None`
  - L473: `def _ensure_registered(self, declaration: StreamDeclaration) -> CameraSourceResolution`
  - L543: `def _initial_streams(self) -> dict[str, str]`
  - L557: `def _initial_source(declaration: StreamDeclaration) -> str`
  - L565: `def _requires_initial_config(declaration: StreamDeclaration) -> bool`
  - L574: `def _stream_name_for(self, camera_or_id: Any) -> str | None`
  - L587: `def _declaration_by_camera_id(self, camera_id: str) -> StreamDeclaration | None`
  - L593: `def _is_running(self) -> bool`
  - L596: `def _remove_stream(self, stream_name: str) -> None`
  - L602: `def _drop_resolution_for_stream(self, stream_name: str) -> None`

### src/argus/streaming/websocket_proxy.py
- L21: `def build_go2rtc_websocket_uri(target: Go2RTCProxyTarget, mode: str) -> str`
- L36: `async def bridge_websockets(browser: WebSocket, upstream: Any, *, max_message_bytes: int=_MAX_MESSAGE_BYTES) -> None`
- L90: `async def proxy_go2rtc_websocket(browser: WebSocket, target: Go2RTCProxyTarget, mode: str) -> None`

### src/argus/validation/pipeline_video_eval.py
- L38: `def _stream_sha256(path: Path) -> str`
- L47: `def _evidence_artifact_sha256_if_exists(path: Path) -> str | None`
- L55: `def _annotation_inventory(raw: Any, *, fps: float, parsed: dict[str, Any], clipped: dict[str, Any], total_frames: int) -> dict[str, Any]`
- L165: `class VirtualVideoClock`
  - L172: `def set(self, seconds: float) -> None`
  - L175: `def monotonic(self) -> float`
  - L181: `def time(self) -> float`
- L185: `class OfflineCapture`
  - L190: `def __init__(self, camera_id: str, source: str) -> None`
  - L196: `def state(self) -> CameraState`
  - L199: `def connect(self) -> bool`
  - L203: `def read(self) -> None`
  - L206: `def read_latest(self) -> None`
  - L209: `def start_capture_thread(self) -> None`
  - L212: `def request_reconnect(self) -> None`
  - L215: `def stop(self) -> None`
- L219: `def create_pipeline(*, config_path: Path, camera_id: str, video_path: Path, model_path: Path | None, model_version_id: str | None=None) -> DetectionPipeline`
- L282: `def _normalized_path(value: str | Path | None) -> str | None`
- L295: `def _value(source: Any, name: str, default: Any=None) -> Any`
- L301: `def _runtime_model_identity(pipeline: Any, *, requested_path: Path | None, requested_version_id: str | None) -> dict[str, Any]`
- L391: `def _candidate_confirmation_contract(pipeline: Any) -> dict[str, Any]`
- L415: `def _production_parity_contract(pipeline: Any) -> dict[str, Any]`
- L444: `def _active_profile_operator_authorization(pipeline: Any, configured_profile_id: str | None) -> dict[str, Any]`
- L515: `def _frame_label(frame_index: int, annotations: dict[str, Any] | None) -> int | None`
- L527: `def _clip_annotations(annotations: dict[str, Any] | None, *, evaluated_frames: int) -> dict[str, Any] | None`
- L555: `def _decision(label: int | None, predicted: int) -> str`
- L563: `def _serialize_alert(alert: Any, timestamp_sec: float) -> dict[str, Any]`
- L583: `def _summarize_scene_guard(records: list[dict[str, Any]], *, sample_period_seconds: float) -> dict[str, Any]`
- L645: `def _summarize_candidate_model_evidence(records: list[dict[str, Any]]) -> dict[str, Any]`
- L763: `def _summarize_prospective_candidate_fusion(records: list[dict[str, Any]]) -> dict[str, Any]`
- L922: `def _load_scene_reference(video_path: Path, *, reference_path: Path | None, frame_index: int) -> tuple[np.ndarray, dict[str, Any]]`
- L973: `def evaluate_pipeline_video(video_path: str | Path, *, config_path: str | Path, camera_id: str, annotation_path: str | Path | None=None, model_path: str | Path | None=None, model_version_id: str | None=None, sample_every: int=1, max_frames: int | None=None, warmup_seconds: float=30.0, scene_reference_path: str | Path | None=None, scene_reference_frame_index: int=0, scene_profile_id: str | None=None, pipeline_factory: Callable[..., Any]=create_pipeline) -> dict[str, Any]`

### src/argus/validation/recall_test.py
- L28: `def load_synthetic_pairs(synthetic_dir: Path) -> list[tuple[Path, Path]]`
- L43: `def evaluate_recall(detector, synthetic_dir: Path, threshold: float=0.5) -> dict`
- L100: `def _iter_images(directory: Path) -> Iterable[Path]`
- L117: `def _score_directory(detector, directory: Path, label: int) -> tuple[list[float], list[int]]`
- L135: `def evaluate_metrics(detector, positive_dir: Path, negative_dir: Path, threshold: float=0.7, min_samples_per_class: int=10) -> dict | None`
- L187: `def evaluate_real_labeled(detector, camera_id: str, data_root: Path=Path('data'), threshold: float=0.7, min_samples_per_class: int=10) -> dict | None`

### src/argus/validation/strict_video_decoder.py
- L37: `class StrictDecodeError(OSError)`
- L42: `class StrictFrameTimestamp`
  - L52: `def to_dict(self) -> dict[str, Any]`
- L57: `class StrictDecodedFrame`
- L66: `class _ProbeInventory`
- L78: `def _sha256(path: Path) -> str`
- L86: `def _required_binary(name: str) -> str`
- L95: `def _positive_fraction(raw: object, *, field: str) -> Fraction`
- L107: `def _optional_positive_rate(raw: object) -> float | None`
- L120: `def _strict_int(raw: object, *, field: str) -> int`
- L131: `def _optional_nonnegative_int(raw: object, *, field: str) -> int | None`
- L140: `def _finite_float(raw: object, *, field: str) -> float`
- L150: `def _parse_probe_payload(payload: Mapping[str, Any]) -> _ProbeInventory`
- L305: `class StrictVideoDecoder(Iterator[StrictDecodedFrame])`
  - L308: `def __init__(self, source: str | Path, *, expected_sha256: str) -> None`
  - L338: `def timestamps(self) -> tuple[StrictFrameTimestamp, ...]`
  - L342: `def nominal_frame_rate(self) -> float`
  - L346: `def declared_frame_count(self) -> int | None`
  - L350: `def probed_frame_count(self) -> int`
  - L353: `def _probe_source(self) -> _ProbeInventory`
  - L403: `def __enter__(self) -> StrictVideoDecoder`
  - L447: `def __iter__(self) -> StrictVideoDecoder`
  - L450: `def __next__(self) -> StrictDecodedFrame`
  - L498: `def _wait_and_capture_stderr(self) -> None`
  - L517: `def _finalize_clean_eof(self) -> None`
  - L542: `def _abort(self) -> None`
  - L552: `def __exit__(self, exc_type, exc, traceback) -> bool`
  - L566: `def evidence(self) -> dict[str, Any]`
- L621: `def write_frame_timestamp_jsonl(path: str | Path, timestamps: tuple[StrictFrameTimestamp, ...]) -> dict[str, Any]`

### src/argus/validation/synthetic.py
- L19: `def _random_perspective_transform(obj: np.ndarray, rng: np.random.Generator) -> np.ndarray`
- L50: `def _match_brightness(fg_region: np.ndarray, bg_region: np.ndarray, mask: np.ndarray) -> np.ndarray`
- L87: `def _blur_mask_edges(mask: np.ndarray, sigma: float=2.0) -> np.ndarray`
- L103: `def _extract_mask(obj: np.ndarray) -> np.ndarray`
- L116: `def generate_synthetic(baseline_dir: Path, objects_dir: Path, output_dir: Path, count: int=500, seed: int=42, min_scale: float=0.3, max_scale: float=2.0, blur_sigma: float=2.0, edge_margin: float=0.05, progress: bool=False) -> int`

### src/argus/validation/video_eval.py
- L25: `class VideoLabelRange`
  - L33: `def contains(self, frame_index: int) -> bool`
  - L36: `def to_dict(self) -> dict`
- L45: `def _range_frame_value(item: dict, frame_key: str, sec_key: str, fps: float) -> int | None`
- L53: `def _parse_ranges(items: list[dict], *, fps: float, label: int) -> list[VideoLabelRange]`
- L82: `def load_video_annotations(annotation_path: str | Path, *, fps: float) -> dict`
- L123: `def _ranges_from_dicts(items: list[dict]) -> list[VideoLabelRange]`
- L135: `def _label_for_frame(frame_index: int, *, positive_ranges: list[VideoLabelRange], ignore_ranges: list[VideoLabelRange], has_annotations: bool) -> int | None`
- L149: `def _extract_score(result: Any) -> float`
- L162: `def _decision(label: int | None, predicted: int) -> str`
- L174: `def _export_misclassified_frame(export_dir: Path, frame: np.ndarray, record: dict) -> str | None`
- L196: `def evaluate_video(detector, video_path: str | Path, *, annotation_path: str | Path | None=None, threshold: float=0.7, sample_every: int=1, max_frames: int | None=None, export_dir: str | Path | None=None, sweep: bool=False) -> dict`
- L335: `def write_video_evaluation_report(result: dict, output_path: str | Path) -> None`

## FastAPI 路由

### /api/alerts (src/argus/dashboard/routes/alerts.py)
- `GET /api/alerts/{alert_id}/detail` -> `alert_detail` (L338)
- `GET /api/alerts/{alert_id}/evidence.zip` -> `export_evidence_package` (L400)
- `GET /api/alerts/{alert_id}/image/{image_type}` -> `alert_image` (L492)
- `GET /api/alerts/group/{event_group_id}` -> `get_alert_group` (L779)
- `GET /api/alerts/export-csv` -> `export_csv` (L811)
- `GET /api/alerts/export-pdf` -> `export_pdf_report` (L846)
- `GET /api/alerts/feedback-stats` -> `feedback_stats` (L977)
- `POST /api/alerts/{alert_id}/annotations` -> `save_annotations` (L1208)
- `GET /api/alerts/json` -> `alerts_json` (L1238)

### /api/audit (src/argus/dashboard/routes/audit.py)
- `GET /api/audit/json` -> `audit_json` (L45)

### /api/backup (src/argus/dashboard/routes/backup.py)
- `GET /api/backup/list/json` -> `backup_list_json` (L25)
- `POST /api/backup/create` -> `create_backup` (L36)
- `POST /api/backup/restore` -> `restore_backup` (L80)
- `DELETE /api/backup/{backup_name}` -> `delete_backup` (L108)

### /api/baseline (src/argus/dashboard/routes/baseline.py)
- `GET /api/baseline/collections/json` -> `baseline_collections_json` (L113)
- `POST /api/baseline/collections/{camera_id}/{zone_id}/{version}/activate` -> `collections_activate` (L193)
- `POST /api/baseline/collections/{camera_id}/{zone_id}/{version}/retire` -> `collections_retire` (L215)
- `DELETE /api/baseline/collections/{camera_id}/{zone_id}/{version}` -> `collections_delete` (L233)
- `GET /api/baseline/list/json` -> `baseline_list_json` (L263)
- `GET /api/baseline/models/json` -> `models_list_json` (L301)
- `DELETE /api/baseline/models/by-path` -> `delete_model_by_path` (L337)
- `GET /api/baseline/training-history/json` -> `training_history_json` (L366)
- `GET /api/baseline/training-history/{record_id}/metrics` -> `training_history_metrics` (L378)
- `DELETE /api/baseline/version` -> `delete_baseline_version` (L456)
- `POST /api/baseline/version/delete` -> `delete_baseline_version_post` (L513)
- `POST /api/baseline/deploy` -> `deploy_model` (L519)
- `POST /api/baseline/job` -> `start_capture_job` (L558)
- `POST /api/baseline/capture` -> `start_capture_legacy` (L642)
- `POST /api/baseline/job/{task_id}/pause` -> `pause_capture_job` (L648)
- `POST /api/baseline/job/{task_id}/resume` -> `resume_capture_job` (L657)
- `POST /api/baseline/job/{task_id}/abort` -> `abort_capture_job` (L666)
- `GET /api/baseline/job/{task_id}` -> `get_capture_job_status` (L675)
- `POST /api/baseline/compare` -> `compare_models_route` (L694)
- `GET /api/baseline/optimize/preview` -> `optimize_preview` (L914)
- `POST /api/baseline/optimize/json` -> `optimize_baseline_json` (L942)
- `GET /api/baseline/groups/json` -> `camera_groups_json` (L1016)
- `POST /api/baseline/groups/merge` -> `merge_group_baseline` (L1047)
- `POST /api/baseline/merge-fp` -> `merge_false_positives` (L1095)
- `GET /api/baseline/versions/json` -> `baseline_versions_json` (L1135)
- `POST /api/baseline/verify` -> `verify_baseline` (L1151)
- `POST /api/baseline/activate-baseline` -> `activate_baseline` (L1181)
- `POST /api/baseline/retire` -> `retire_baseline` (L1207)
- `GET /api/baseline/{camera_id}/{version}/images` -> `list_version_images` (L1263)
- `GET /api/baseline/{camera_id}/{version}/images/{filename}/content` -> `get_version_image_content` (L1296)
- `DELETE /api/baseline/{camera_id}/{version}/images/{filename}` -> `delete_version_image` (L1344)
- `POST /api/baseline/{camera_id}/{version}/images` -> `upload_version_image` (L1399)

### /api/calibration (src/argus/dashboard/routes/calibration.py)
- `GET /api/calibration/{camera_id}` -> `get_calibration_status` (L27)
- `POST /api/calibration/{camera_id}/upload` -> `upload_calibration` (L49)
- `POST /api/calibration/{camera_id}/compute` -> `compute_calibration` (L111)
- `GET /api/calibration/{camera_id}/verify` -> `verify_calibration` (L158)

### /api/cameras (src/argus/dashboard/routes/cameras.py)
- `POST /api/cameras` -> `add_camera` (L444)
- `DELETE /api/cameras/{camera_id}` -> `delete_camera` (L539)
- `GET /api/cameras/{camera_id}/config` -> `get_camera_config` (L601)
- `PUT /api/cameras/{camera_id}` -> `update_camera` (L625)
- `POST /api/cameras/{camera_id}/test-connection` -> `test_camera_connection` (L820)
- `POST /api/cameras/test-connection-draft` -> `test_camera_connection_draft` (L845)
- `GET /api/cameras/usb-devices` -> `usb_devices` (L868)
- `GET /api/cameras/json` -> `cameras_json` (L924)
- `GET /api/cameras/{camera_id}/diagnostics` -> `camera_diagnostics` (L1183)
- `GET /api/cameras/{camera_id}/runner` -> `camera_runner_snapshot` (L1203)
- `GET /api/cameras/{camera_id}/snapshot` -> `camera_snapshot` (L1242)
- `GET /api/cameras/{camera_id}/stream` -> `camera_stream` (L1248)
- `GET /api/cameras/{camera_id}/heatmap-stream` -> `camera_heatmap_stream` (L1254)
- `GET /api/cameras/wall/status` -> `wall_status` (L1262)

### /api/config (src/argus/dashboard/routes/config.py)
- `POST /api/config/detection-params` -> `update_detection_params` (L97)
- `POST /api/config/notifications` -> `update_notifications` (L194)
- `POST /api/config/test-webhook` -> `test_webhook` (L223)
- `GET /api/config/storage/info` -> `storage_info_json` (L259)
- `POST /api/config/cleanup` -> `cleanup_data` (L289)
- `POST /api/config/thresholds` -> `update_thresholds` (L319)
- `POST /api/config/reload-model` -> `reload_model` (L340)
- `POST /api/config/clear-lock/{camera_id}` -> `clear_anomaly_lock` (L370)
- `POST /api/config/camera/{camera_id}/restart` -> `restart_camera` (L397)
- `POST /api/config/reload` -> `reload_config` (L416)
- `POST /api/config/save` -> `save_config` (L455)
- `GET /api/config/audio-alerts` -> `get_audio_alerts` (L480)
- `PUT /api/config/audio-alerts` -> `update_audio_alerts` (L497)
- `GET /api/config/modules` -> `get_module_states` (L527)
- `GET /api/config/classifier` -> `get_classifier_config` (L554)
- `GET /api/config/segmenter` -> `get_segmenter_config` (L594)
- `GET /api/config/imaging` -> `get_imaging_config` (L633)
- `GET /api/config/cross-camera` -> `get_cross_camera_config` (L664)
- `PUT /api/config/cross-camera/pairs` -> `update_cross_camera_config` (L719)
- `PUT /api/config/segmenter/params` -> `update_segmenter_params_route` (L846)
- `PUT /api/config/classifier/vocabulary` -> `update_classifier_vocabulary` (L933)
- `POST /api/config/modules` -> `update_module_toggle` (L1038)

### /api/degradation (src/argus/dashboard/routes/degradation.py)
- `GET /api/degradation/active` -> `active_degradations` (L23)
- `GET /api/degradation/history` -> `degradation_history` (L37)
- `GET /api/degradation/summary` -> `degradation_summary` (L53)

### /api/labeling (src/argus/dashboard/routes/labeling.py)
- `GET /api/labeling/queue` -> `labeling_queue` (L30)
- `POST /api/labeling/{entry_id}/label` -> `label_entry` (L55)
- `POST /api/labeling/{entry_id}/skip` -> `skip_entry` (L101)
- `GET /api/labeling/stats` -> `labeling_stats` (L126)
- `POST /api/labeling/trigger-retrain` -> `trigger_retrain` (L144)
- `GET /api/labeling/{entry_id}/image` -> `labeling_image` (L234)

### /api/me (src/argus/dashboard/routes/me.py)
- `GET /api/me` -> `get_me` (L22)

### /api/models (src/argus/dashboard/routes/models.py)
- `GET /api/models/status` -> `models_status` (L260)
- `GET /api/models/json` -> `list_models` (L286)
- `POST /api/models/{version_id}/activate` -> `activate_model` (L306)
- `POST /api/models/{version_id}/rollback` -> `rollback_model` (L331)
- `POST /api/models/batch-inference` -> `batch_inference` (L399)
- `GET /api/models/release-operations/{operation_id}` -> `get_release_operation` (L444)
- `POST /api/models/{version_id}/promote` -> `promote_model` (L825)
- `POST /api/models/{version_id}/retire` -> `retire_model` (L937)
- `DELETE /api/models/{version_id}` -> `delete_model` (L1001)
- `POST /api/models/{version_id}/reexport` -> `reexport_model` (L1082)
- `POST /api/models/{version_id}/recalibrate` -> `recalibrate_model` (L1141)
- `GET /api/models/{version_id}/stage-history` -> `stage_history` (L1201)
- `GET /api/models/events/list` -> `list_version_events` (L1215)
- `GET /api/models/{version_id}/shadow-report` -> `shadow_report` (L1233)
- `GET /api/models/{version_id}/ab-scores` -> `ab_compare_scores` (L1274)
- `GET /api/models/{version_id}/ab-distribution` -> `ab_compare_distribution` (L1315)
- `POST /api/models/ab-live-compare` -> `ab_live_compare` (L1462)
- `GET /api/models/backbone/status` -> `backbone_status` (L1524)
- `POST /api/models/backbone/upgrade` -> `backbone_upgrade` (L1536)
- `GET /api/models/threshold-preview` -> `threshold_preview` (L1570)

### /api/physics (src/argus/dashboard/routes/physics.py)
- `GET /api/physics/{camera_id}/active-tracks` -> `get_active_tracks` (L112)
- `GET /api/physics/{alert_id}/localization` -> `get_alert_localization` (L131)

### /api/recordings (src/argus/dashboard/routes/recordings.py)
- `GET /api/recordings/{camera_id}/segments` -> `list_segments` (L17)
- `GET /api/recordings/{camera_id}/segment/{filename}/video` -> `serve_segment_video` (L48)
- `GET /api/recordings/storage-stats` -> `storage_stats` (L84)
- `POST /api/recordings/cleanup` -> `trigger_cleanup` (L99)

### /api/regions (src/argus/dashboard/routes/regions.py)
- `GET /api/regions/json` -> `regions_json` (L66)
- `POST /api/regions/json` -> `create_region_json` (L89)
- `PUT /api/regions/{region_id}/json` -> `update_region_json` (L114)
- `DELETE /api/regions/{region_id}/json` -> `delete_region_json` (L145)

### /api/replay (src/argus/dashboard/routes/replay.py)
- `GET /api/replay/{alert_id}/video` -> `replay_video` (L216)
- `GET /api/replay/{alert_id}/frame/{index}` -> `replay_frame` (L276)
- `GET /api/replay/{alert_id}/heatmap/{index}` -> `replay_heatmap_frame` (L300)
- `GET /api/replay/{alert_id}/reference` -> `replay_reference` (L321)
- `GET /api/replay/storyboard/{alert_id}` -> `replay_storyboard` (L576)

### /api/reports (src/argus/dashboard/routes/reports.py)
- `GET /api/reports/json` -> `reports_json` (L21)
- `GET /api/reports/daily-trend/json` -> `daily_trend_json` (L171)
- `GET /api/reports/severity-dist/json` -> `severity_dist_json` (L180)
- `GET /api/reports/camera-dist/json` -> `camera_dist_json` (L193)
- `GET /api/reports/fp-trend/json` -> `fp_trend_json` (L206)
- `GET /api/reports/compliance` -> `compliance_report` (L624)

### /api/sensors (src/argus/dashboard/routes/sensors.py)
- `POST /api/sensors/signal` -> `set_signal` (L73)
- `POST /api/sensors/reading` -> `set_reading` (L104)
- `GET /api/sensors/signals` -> `list_signals` (L142)
- `DELETE /api/sensors/signal` -> `delete_signal` (L155)

### /api/streaming (src/argus/dashboard/routes/streaming.py)
- `GET /api/streaming/{camera_id}` -> `stream_info` (L34)
- `GET /api/streaming` -> `streams_list` (L45)
- `POST /api/streaming/{camera_id}/register` -> `register_stream` (L52)

### /api/system (src/argus/dashboard/routes/system.py)
- `GET /api/system/mode_summary` -> `mode_summary` (L18)
- `GET /api/system/health` -> `health` (L44)
- `GET /api/system/drift` -> `drift_status` (L93)
- `GET /api/system/camera-health` -> `camera_health` (L104)
- `GET /api/system/ensemble` -> `ensemble_status` (L115)
- `GET /api/system/metrics` -> `prometheus_metrics` (L126)
- `GET /api/system/anomaly-degradation` -> `anomaly_degradation` (L135)

### /api/tasks (src/argus/dashboard/routes/tasks.py)
- `GET /api/tasks/json` -> `tasks_list_json` (L12)
- `DELETE /api/tasks/{task_id}` -> `dismiss_task` (L35)

### /api/training-jobs (src/argus/dashboard/routes/training_jobs.py)
- `GET /api/training-jobs/json` -> `list_training_jobs` (L82)
- `GET /api/training-jobs/{job_id}` -> `get_training_job` (L106)
- `GET /api/training-jobs/{job_id}/source-alerts` -> `get_training_job_source_alerts` (L127)
- `POST /api/training-jobs/` -> `create_training_job` (L204)
- `POST /api/training-jobs/{job_id}/confirm` -> `confirm_training_job` (L312)
- `POST /api/training-jobs/{job_id}/reject` -> `reject_training_job` (L355)
- `GET /api/training-jobs/backbones/json` -> `list_backbones` (L396)

### /api/users (src/argus/dashboard/routes/users.py)
- `GET /api/users/json` -> `users_json` (L48)
- `POST /api/users/json` -> `create_user_json` (L81)
- `PUT /api/users/{username}/json` -> `update_user_json` (L118)
- `DELETE /api/users/{username}/json` -> `delete_user_json` (L162)
- `POST /api/users/{username}/toggle-active/json` -> `toggle_active_json` (L183)

### /api/zones (src/argus/dashboard/routes/zones.py)
- `POST /api/zones` -> `create_zone` (L64)
- `PUT /api/zones/{camera_id}` -> `update_zones` (L134)
- `DELETE /api/zones/{camera_id}/{zone_id}` -> `delete_zone` (L191)
- `GET /api/zones/snapshot/{camera_id}` -> `camera_snapshot` (L231)

## 前端路由

### web/src/router/index.ts
- `/` -> `overview` / 总览
- `cameras` -> `cameras` / 摄像头
- `cameras/:id` -> `camera-detail` / 摄像头详情
- `alerts` -> `alerts` / 告警
- `reports` -> `reports` / 报表
- `baseline` -> `models-baseline` / 基线管理
- `training` -> `models-training` / 训练与评估
- `registry` -> `models-registry` / 模型与发布
- `comparison` -> `models-comparison` / A/B 对比
- `labeling` -> `models-labeling` / 标注队列
- `threshold` -> `models-threshold` / 阈值预览
- `collections` -> `models-collections` / 采集集合
- `overview` -> `system-overview` / 系统概览
- `model-status` -> `system-model-status` / 模型状态
- `config` -> `system-config` / 配置管理
- `audit` -> `system-audit` / 审计日志
- `degradation` -> `system-degradation` / 降级事件
- `modules` -> `system-modules` / 功能模块
- `classifier` -> `system-classifier` / 分类器
- `segmenter` -> `system-segmenter` / 分割器
- `imaging` -> `system-imaging` / 多模态成像
- `cross-camera` -> `system-cross-camera` / 跨相机
- `users` -> `system-users` / 用户管理
- `regions` -> `system-regions` / 区域管理
- `replay/:alertId` -> `replay` / 录像回放
- `replay/:alertId/storyboard` -> `replay-storyboard` / 多机位回放
- `warehouse3D` -> `warehouse3D` / 3D 演示

## 前端 API 客户端

### web/src/api/alerts.ts
- `downloadEvidencePackage(alertId: string)`
- `getAlerts(params?: AlertQueryParams)`
- `getAlert(id: string)`
- `acknowledgeAlert(id: string)`
- `markFalsePositive(id: string)`
- `confirmRealAnomaly(id: string)`
- `getFeedbackStats(camera_id?: string)`
- `updateAlertWorkflow(id: string, data: AlertWorkflowUpdate)`
- `bulkAcknowledge(ids: string[])`
- `bulkFalsePositive(ids: string[])`
- `deleteAlert(id: string)`
- `bulkDeleteAlerts(ids: string[])`
- `saveAnnotations(alertId: string, annotations: AlertAnnotation[])`
- `getAlertGroup(eventGroupId: string)`
- `getModelVersionInfo(modelVersionId: string)`

### web/src/api/baselines.ts
- `getBaselines()`
- `getBaselineCollections()`
- `activateCollection(camera_id: string, zone_id: string, version: string,)`
- `retireCollection(camera_id: string, zone_id: string, version: string,)`
- `deleteCollection(camera_id: string, zone_id: string, version: string, opts?: { confirm?: boolean },)`
- `optimizeBaseline(data: { camera_id: string; zone_id?: string; target_ratio?: number })`
- `previewOptimize(params: { camera_id: string; zone_id?: string; target_ratio?: number })`
- `startCaptureJob(data: FormData)`
- `pauseCaptureJob(taskId: string)`
- `resumeCaptureJob(taskId: string)`
- `abortCaptureJob(taskId: string)`
- `mergeFalsePositives(data: { camera_id: string; zone_id?: string; max_fp_images?: number })`
- `getCameraGroups()`
- `mergeGroupBaseline(data: { group_id: string; zone_id?: string; target_count?: number })`
- `getBaselineVersions(params: { camera_id: string; zone_id?: string })`
- `verifyBaseline(data: { camera_id: string; zone_id?: string; version: string; verified_by: string; verified_by_secondary?: string })`
- `activateBaseline(data: { camera_id: string; zone_id?: string; version: string; user?: string })`
- `retireBaseline(data: { camera_id: string; zone_id?: string; version: string; user?: string; reason?: string })`
- `deleteBaselineVersion(data: { camera_id: string; zone_id?: string; version: string; user?: string })`
- `listBaselineImages(cameraId: string, version: string, zoneId: string = 'default',)`
- `deleteBaselineImage(cameraId: string, version: string, filename: string, zoneId: string = 'default',)`
- `uploadBaselineImage(cameraId: string, version: string, file: File, zoneId: string = 'default',)`
- `baselineImageContentUrl(cameraId: string, version: string, filename: string, zoneId: string = 'default',)`

### web/src/api/cameras.ts
- `getCameras()`
- `getCameraDetail(id: string)`
- `startCamera(id: string)`
- `stopCamera(id: string)`
- `setCameraMode(id: string, mode: PipelineMode)`
- `getUsbDevices()`
- `addCamera(data: FormData)`
- `updateCamera(id: string, data: FormData)`
- `getCameraConfig(id: string)`
- `deleteCamera(id: string)`
- `testCameraConnection(id: string)`
- `testCameraConnectionDraft(payload: { source: string | number; protocol?: string },)`
- `getSystemModeSummary()`
- `getWallStatus()`

### web/src/api/labeling.ts
- `getLabelingQueue(params?: { camera_id?: string status?: string limit?: number offset?: number })`
- `labelEntry(entryId: number, data: { label: 'normal' | 'anomaly'; labeled_by?: string })`
- `skipEntry(entryId: number)`
- `getLabelingStats(cameraId?: string)`
- `triggerRetrain(data?: { camera_id?: string; model_type?: string; triggered_by?: string })`
- `getLabelingImage(entryId: number)`

### web/src/api/models.ts
- `getModelRegistry(cameraId?: string)`
- `activateModel(versionId: string)`
- `rollbackModel(versionId: string)`
- `deleteModel(versionId: string)`
- `reexportModel(versionId: string, data: { export_format?: string; quantization?: string })`
- `recalibrateModel(versionId: string)`
- `promoteModel(versionId: string, data: ModelReleaseRequest)`
- `retireModel(versionId: string, data?: Omit<ModelReleaseRequest, 'target_stage' | 'canary_camera_id'>)`
- `getReleaseOperation(operationId: string)`
- `getReleaseReadiness(versionId: string)`
- `getStageHistory(versionId: string)`
- `getVersionEvents(params?: { camera_id?: string; limit?: number })`
- `getShadowReport(versionId: string, params?: { camera_id?: string; days?: number })`
- `getBackboneStatus()`
- `batchInference(cameraId: string, imagePaths: string[])`
- `getThresholdPreview(params: { camera_id?: string; threshold?: number; days?: number })`
- `getABScores(versionId: string, params?: { camera_id?: string; days?: number; limit?: number })`
- `getABDistribution(versionId: string, params?: { camera_id?: string; days?: number; bins?: number })`
- `runABLiveCompare(cameraId: string, shadowVersionId: string)`

### web/src/api/physics.ts
- `getAlertTrajectory(alertId: string)`
- `getActiveTracks(cameraId: string)`

### web/src/api/regions.ts
- `getRegions(params?: RegionQueryParams)`
- `createRegion(data: RegionPayload)`
- `updateRegion(regionId: number, data: RegionPayload)`
- `deleteRegion(regionId: number)`

### web/src/api/replay.ts
- `getReplayMetadata(alertId: string)`
- `getReplaySignals(alertId: string)`
- `getReplayVideoUrl(alertId: string)`
- `getReplayFrameUrl(alertId: string, index: number)`
- `getReplayHeatmapUrl(alertId: string, index: number)`
- `getReplayReference(alertId: string, params?: { date?: string; time?: string; frame_offset_seconds?: number },)`
- `pinReplayFrame(alertId: string, data: { index: number; label: string })`
- `addReplayClip(alertId: string, data: { start_index: number; end_index: number; label?: string },)`
- `deleteReplayClip(alertId: string, index: number)`
- `getStoryboard(alertId: string)`

### web/src/api/reports.ts
- `downloadComplianceReport(days = 30, format: 'csv' | 'pdf' = 'csv')`
- `getReportStats(days?: number)`
- `getDailyTrend(days = 30)`
- `getSeverityDist(days?: number)`
- `getCameraDist(days?: number)`
- `getFPTrend(days = 30)`

### web/src/api/system.ts
- `getHealth()`
- `getModelsStatus()`
- `reloadConfig()`
- `getStorageInfo()`
- `cleanupAlerts()`
- `createBackup()`
- `getDegradationActive()`
- `getDegradationSummary()`
- `getDegradationHistory(days?: number)`
- `getAnomalyDegradation()`
- `getAudioAlerts()`
- `updateAudioAlerts(data: AudioAlertsUpdate)`
- `getDriftStatus()`
- `getCameraHealth()`
- `getClassifierConfig()`
- `getImagingConfig()`
- `getSegmenterConfig()`
- `getCrossCameraConfig()`
- `updateCrossCameraConfig(payload: CrossCameraUpdatePayload)`
- `updateSegmenterParams(payload: SegmenterParamsUpdate)`
- `updateClassifierVocabulary(payload: ClassifierVocabularyUpdate)`
- `getModuleStates()`
- `toggleModule(key: string, value: boolean)`
- `saveConfig()`
- `updateThresholds(data: ThresholdUpdate)`
- `updateDetectionParams(data: DetectionParamsUpdate)`
- `reloadModel(data: { camera_id: string; model_path: string })`
- `clearLock(cameraId: string)`
- `restartCamera(cameraId: string)`
- `updateNotifications(data: NotificationUpdate)`
- `testWebhook()`
- `listBackups()`
- `restoreBackup(name: string)`
- `deleteBackup(name: string)`

### web/src/api/tasks.ts
- `getTasks()`
- `dismissTask(taskId: string)`

### web/src/api/training.ts
- `getTrainingHistory(params?: QueryParams)`
- `compareModels(data: { old_record_id: number; new_record_id: number })`
- `getTrainingMetrics(recordId: number)`
- `getTrainingJobs(params?: QueryParams)`
- `getTrainingJob(jobId: string)`
- `createTrainingJob(data: { job_type: string; camera_id?: string; zone_id?: string; model_type?: string; trigger_type?: string; triggered_by?: string; hyperparameters?: Record<string, unknown>; dataset_selection?: DatasetSelection; })`
- `confirmTrainingJob(jobId: string, data?: { confirmed_by?: string })`
- `rejectTrainingJob(jobId: string, data?: { rejected_by?: string; reason?: string })`
- `getBackbones()`

### web/src/api/users.ts
- `getUsers(params?: UserQueryParams)`
- `createUser(data: CreateUserPayload)`
- `updateUser(username: string, data: UpdateUserPayload)`
- `deleteUser(username: string)`
- `toggleUserActive(username: string)`
- `getAuditLogs(params?: { page?: number; page_size?: number; user?: string; action?: string })`

### web/src/api/zones.ts
- `createZone(data: ZoneCreateRequest)`
- `updateZones(cameraId: string, zones: CameraZone[])`
- `deleteZone(cameraId: string, zoneId: string)`
- `getZoneSnapshotUrl(cameraId: string)`
