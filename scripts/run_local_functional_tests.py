"""Run ordinary functionality without training, inference, or real media I/O.

The explicit suites are intentionally not the full detector test collection.
Mock trainers/runtimes are allowed; the real execution entry points fail closed.
"""

from __future__ import annotations

import argparse
import functools
import importlib.abc
import os
import shlex
import subprocess
import sys
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
UNIT_SUITES = {
    "platform": """
        analysis_task_storage analysis_tasks_api analysis_task_runtime frame_hub task_guidance
        analysis_task_manager analysis_task_adversarial task_platform_storage
        task_execution task_execution_manager task_adapters task_workflows task_collector task_event_collector task_platform_api task_preannotations task_automation
        task_deployments_api task_deployments task_deployment_adversarial
        gpu_smoke_contracts
        task_events task_release task_training task_lifecycle_api task_annotation_adversarial
        task_training_configuration task_training_runtime task_training_resources task_training_wiring compute_ownership
    """.split(),
    "core": """
        functional_test_guard
        auth permissions startup_security api_contract core_api_dtos config config_loader config_api
        database database_alert_evidence_migration alert_grader alert_grader_evidence
        alert_workflow alerts_route alerts_route_evidence
        backup_route reports_route feedback feedback_queue feedback_record labeling_lineage
        training_history training_route_artifacts scene_training_route_authority training_job_claims
        scene_workflow scene_workflow_routes scene_workflow_task_concurrency task_pause
        websocket error_channel event_bus metrics health health_readiness_api system_degradation_api
        camera_rbac camera_groups zone_rbac sensors_route physics_runtime_contract
        dashboard_streaming_contract preview_gateway_contract stream_registry_contract recordings_route
        stream_websocket_proxy model_registry model_registry_scope model_deletion model_deletion_events
        model_list_operations models_status_route model_release_service model_rollback release_pipeline release_execution
        release_slot_invariants release_api_policy release_evidence scene_training
        scene_training_execution data_contracts dataset_selection dataset_merger calibration
        circuit_breaker backpressure degradation degradation_bridge degradation_manager diagnostics
        inference_record inference_record_candidate model_status runtime_interfaces setup_remote_sshkey
    """.split(),
    "review": """
        cvat_integration fiftyone_integration fiftyone_scene_images scene_comparison_review
        scene_review_images scene_review_release_isolation video_review_contracts video_review_corpus
        video_review_cli lightglue_isolation_contract lightglue_report_adversarial
        attest_soak_report run_trusted_soak
    """.split(),
    "dashboard": """
        dashboard models_business_flow model_package
        baseline_lifecycle baseline_job training_job_wiring
        model_compare model_diversity replay_route
    """.split(),
    "auxiliary": """
        auxiliary_model_lifecycle auxiliary_pipeline_contract auxiliary_alert_policy
        auxiliary_runtime_adversarial auxiliary_camera_lifecycle
        auxiliary_classification_attribution auxiliary_pipeline_adversarial
    """.split(),
}


class NoModelEngines(importlib.abc.MetaPathFinder):
    """Never initialize heavyweight model SDKs from a local functional suite."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {
            "torch", "anomalib", "ultralytics", "openvino", "onnxruntime",
            "lightglue", "kornia", "fiftyone", "cvat_sdk", "sam2",
        }:
            raise ImportError(f"Local functional profile blocks model/external SDK: {fullname}")
        return None


class FunctionalGuard:
    def __init__(self):
        self.violations: list[str] = []
        self._simulated_process_frame = None
        self._simulated_training_prepare = None

    def simulate_training_prepare(self, worker, plan):
        """Exercise spawn bookkeeping only with an explicit fake context.

        Preparing now creates a worker process. A real multiprocessing context
        must not be used in this local profile, even before model authorization.
        The separate OS-lock/IPC subprocess tests never call this entry point.
        """
        from types import SimpleNamespace
        from unittest.mock import Mock
        if (worker is None or not isinstance(getattr(worker, "context", None), (SimpleNamespace, Mock))
                or self._simulated_training_prepare is None):
            return self.block("SpawnTrainingWorker.prepare")()
        return self._simulated_training_prepare(worker, plan)

    def simulate_frame(self, pipeline, frame):
        """Run orchestration only when every recognition head is a test double.

        The normal public process_frame entry point remains blocked. This
        narrow helper is used by the explicit auxiliary mock-contract suite.
        """
        from unittest.mock import Mock

        for name in (
            "_prefilter", "_object_detector", "_anomaly_detector", "_simplex",
            "_persistent_change", "_water_leak", "_plume", "_fast_motion",
            "_classifier", "_segmenter", "_scene_guard", "_low_light",
            "_physics_postproc", "_physics_tracker", "_ensemble_detector",
            "_imaging_processor", "_aligner", "_static_candidate_confirmer",
            "_frame_quality_assessor", "_modality_fusion", "_candidate_model_temporal",
        ):
            component = getattr(pipeline, name, None)
            if component is not None and not isinstance(component, Mock):
                return self.block(f"simulate_frame requires mock: {name}")()
        if self._simulated_process_frame is None:
            return self.block("simulate_frame is not configured")()
        return self._simulated_process_frame(pipeline, frame)

    def block(self, label):
        def forbidden(*args, **kwargs):
            self.violations.append(label)
            raise RuntimeError(f"Local functional profile forbids real execution: {label}")
        return forbidden

    def guard_process_launch(self, launch):
        """Block external media tools without blocking ordinary Python probes.

        OpenCV is not the only media path: on Windows the encoder prefers an
        FFmpeg subprocess. Check the executable as well as argv, including
        quoted Windows paths, before the process can open any source/output.
        """
        @functools.wraps(launch)
        def guarded(args, *positional, **kwargs):
            executable = kwargs.get("executable")
            if executable is None:
                if isinstance(args, os.PathLike):
                    executable = args
                elif isinstance(args, (str, bytes)):
                    command = os.fsdecode(args)
                    try:
                        tokens = shlex.split(command, posix=False)
                    except ValueError:
                        tokens = [command]
                    executable = tokens[0] if tokens else ""
                else:
                    executable = args[0] if args else ""
            name = PureWindowsPath(os.fsdecode(executable).strip('"\'')).name.lower()
            if name.startswith(("ffmpeg", "ffprobe")):
                return self.block(f"subprocess media tool: {name}")()
            return launch(args, *positional, **kwargs)
        return guarded

    def pytest_configure(self, config):
        import av
        import cv2

        from argus.anomaly.detector import AnomalibDetector, MultiScaleDetector
        from argus.anomaly.trainer import ModelTrainer
        from argus.core.pipeline import DetectionPipeline
        from argus.core import task_training
        from argus.runtime import task_training as runtime_task_training

        self._simulated_process_frame = DetectionPipeline.process_frame
        task_training.execute_training_plan = self.block("execute_training_plan")
        runtime_task_training.SpawnTrainingWorker.start = self.block("SpawnTrainingWorker.start")
        prepare = runtime_task_training.SpawnTrainingWorker.prepare
        self._simulated_training_prepare = getattr(prepare, "_argus_original_prepare", prepare)
        def guarded_prepare(worker, plan):
            return self.simulate_training_prepare(worker, plan)
        guarded_prepare._argus_original_prepare = self._simulated_training_prepare
        runtime_task_training.SpawnTrainingWorker.prepare = guarded_prepare

        for cls, methods in (
            (ModelTrainer, ("train", "export")),
            (AnomalibDetector, ("load", "predict", "predict_batch")),
            (MultiScaleDetector, ("load", "predict")),
            (DetectionPipeline, ("initialize", "process_frame")),
        ):
            for method in methods:
                if hasattr(cls, method):
                    setattr(cls, method, self.block(f"{cls.__name__}.{method}"))
        # Image encoding/decoding used by API fixtures is allowed; cameras and
        # source video readers/writers are not opened by this profile.
        cv2.VideoCapture = self.block("cv2.VideoCapture")
        cv2.VideoWriter = self.block("cv2.VideoWriter")
        av.open = self.block("av.open")
        subprocess.Popen = self.guard_process_launch(subprocess.Popen)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.violations:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            reporter.write_sep("!", "Prevented real training/inference/media execution")
            for label in sorted(set(self.violations)):
                reporter.write_line(label)
            session.exitstatus = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=[*UNIT_SUITES, "lint", "all"], default="core")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--maxfail", type=int, default=20)
    parser.add_argument("--filter", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.meta_path.insert(0, NoModelEngines())
    import pytest

    suites = UNIT_SUITES if args.suite == "all" else {args.suite: UNIT_SUITES.get(args.suite, [])}
    paths = [
        f"tests/unit/test_{name}.py"
        for names in suites.values() for name in names
    ]
    if args.suite in {"lint", "all"}:
        paths.append("tests/lint")
    if args.suite in {"review", "all"}:
        # Other tests in these files decode video or run feature matching.
        paths.extend(
            "tests/unit/test_prepare_gpu_validation.py::" + name
            for name in (
                "test_validation_config_is_isolated_authenticated_and_never_production",
                "test_venv_python_path_keeps_the_terminal_symlink",
                "test_validation_env_binds_exact_release_identity_and_token_file",
            )
        )
    if args.suite in {"dashboard", "all"}:
        paths.append("tests/unit/test_model_backend_policy.py::test_explicit_backend_model_discovery_filters_artifact_format")
    options = ["-p", "pytest_asyncio.plugin", "-v" if args.verbose else "-q", "--tb=short",
               "-o", "faulthandler_timeout=20", f"--maxfail={args.maxfail}"]
    if args.filter:
        options.extend(["-k", args.filter])
    if args.collect_only:
        options.append("--collect-only")
    return pytest.main([*options, *paths], plugins=[FunctionalGuard()])


if __name__ == "__main__":
    raise SystemExit(main())
