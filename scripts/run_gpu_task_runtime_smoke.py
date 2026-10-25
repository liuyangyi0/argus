"""Opt-in Windows/CUDA plumbing smoke; never a model-quality acceptance run.

This script creates its own ONNX graphs, synthetic source and SQLite database.
It does not open cameras, read videos, download weights, train or enable alerts.
Do not run it on a machine without an explicitly authorized NVIDIA GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_synthetic_model(path: Path, score: float) -> None:
    """A dynamic pixel-dependent MatMul, not a constant fake GPU graph."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = np.tile(np.array([[16 / 3, 16 / 3, 16 / 3, 16 / 3, score / 3]], dtype=np.float32), (3, 1))
    graph = helper.make_graph([
        helper.make_node("GlobalAveragePool", ["images"], ["pooled"], name="pixel_average"),
        helper.make_node("Flatten", ["pooled"], ["channels"], name="channels", axis=1),
        helper.make_node("MatMul", ["channels", "weights"], ["prediction"], name="pixel_dependent_matmul"),
        helper.make_node("Reshape", ["prediction", "shape"], ["output"], name="raw_yolo_shape"),
    ], "synthetic_cuda_smoke_not_a_trained_detector", [
        helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32]),
    ], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 1])], [
        numpy_helper.from_array(weights, "weights"),
        numpy_helper.from_array(np.array([1, 5, 1], dtype=np.int64), "shape"),
    ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="argus-gpu-smoke")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save_model(model, path, save_as_external_data=False)


def preflight_cuda(root: Path, model_path: Path) -> dict:
    import numpy as np
    import onnxruntime as ort
    try:
        import torch
    except ImportError:
        torch = None

    torch_evidence = {"installed": torch is not None, "cuda_tensor_checked": False}
    if torch is not None:
        require(torch.cuda.is_available(), "installed torch has no CUDA; CPU fallback is forbidden")
        actual = torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")
        torch.cuda.synchronize()
        require(actual.device.type == "cuda" and actual[0, 0].item() == 32, "torch CUDA MatMul failed")
        torch_evidence.update(cuda_tensor_checked=True, version=torch.__version__,
            gpu_name=torch.cuda.get_device_name(0), cuda_version=torch.version.cuda)
    require("CUDAExecutionProvider" in ort.get_available_providers(), "ORT CUDAExecutionProvider is unavailable")
    # A standalone ORT CUDA environment may use official NVIDIA site packages
    # instead of torch. The real spawn workers use the same production loader.
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(root / "cuda-node-placement")
    session = ort.InferenceSession(model_path.read_bytes(), sess_options=options,
        providers=["CUDAExecutionProvider"], enable_fallback=0, read_config_from_model=False)
    session.disable_fallback()
    require(session.get_providers()[0] == "CUDAExecutionProvider", "ORT did not select CUDA first")
    white = session.run(None, {"images": np.ones((1, 3, 32, 32), dtype=np.float32)})[0]
    black = session.run(None, {"images": np.zeros((1, 3, 32, 32), dtype=np.float32)})[0]
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    kernels = [event.get("args", {}) for event in events if event.get("args", {}).get("provider")]
    require(bool(kernels), "ORT profile contains no execution-provider evidence")
    require(not any(item["provider"] != "CUDAExecutionProvider" for item in kernels),
        "ORT profile shows non-CUDA node execution; fallback is forbidden")
    require(any(item.get("op_name") in {"MatMul", "Gemm", "FusedMatMul"} for item in kernels),
        "ORT profile does not prove CUDA MatMul/Gemm execution")
    require(math.isclose(float(white[0, 4, 0]), 0.9, abs_tol=1e-5)
        and float(black[0, 4, 0]) == 0.0, "ONNX output is not dependent on synthetic input pixels")
    return {"torch": torch_evidence, "ort_version": ort.__version__,
        "providers": session.get_providers(), "cpu_fallback_allowed": False,
        "profile": profile_path.name, "profile_kernel_providers": sorted({item["provider"] for item in kernels}),
        "profile_operations": sorted({item.get("op_name", "unknown") for item in kernels}),
        "dynamic_input_dependency_verified": True}


class SyntheticSource:
    """Controlled in-memory producer, not a camera or media decoder."""

    camera_id = "synthetic-gpu-smoke"

    def __init__(self):
        from argus.capture.frame_hub import FrameHub

        self.hub = FrameHub(self.camera_id)
        self.lock = threading.RLock()
        self.session = uuid4().hex
        self.sequence = 0
        self._last_published_at: float | None = None
        self.hub.activate(self.session, 1)

    @property
    def source_token(self):
        return self.hub.source_token

    def subscribe(self, name, **kwargs):
        return self.hub.subscribe(name, **kwargs)

    def commit_if_current(self, token, callback):
        with self.lock:
            if token != self.source_token:
                return False
            callback()
            return True

    def publish(self) -> int:
        import numpy as np

        from argus.capture.camera import FrameData
        from argus.core.frame_identity import build_frame_identity

        with self.lock:
            # Tests issue same-bundle commands much faster than a camera's
            # sampling interval. A single frame inside the 10-FPS task throttle
            # would correctly be skipped and falsely fail this plumbing smoke.
            # Keep actual monotonic capture times; do not invent future stamps.
            if self._last_published_at is not None:
                remaining = self._last_published_at + 0.2 - time.monotonic()
                while remaining > 0:
                    time.sleep(remaining)
                    remaining = self._last_published_at + 0.2 - time.monotonic()
            self.sequence += 1
            timestamp = time.monotonic()
            self._last_published_at = timestamp
            identity = build_frame_identity(camera_id=self.camera_id,
                source_declared="synthetic://gpu-task-smoke", source_runtime="synthetic://gpu-task-smoke",
                stream_id=None, frame_index=self.sequence, capture_monotonic_ts=timestamp,
                source_session_id=self.session, source_epoch=1,
                frame_index_provenance="synthetic_generator_sequence")
            frame = FrameData(np.full((32, 32, 3), 255, dtype=np.uint8), self.camera_id,
                timestamp, self.sequence, (32, 32), identity=identity)
            require(self.hub.publish(frame), "synthetic FrameHub rejected its current frame")
            return self.sequence


class SyntheticCameraManager:
    """Only supplies a source and task registry; all computation remains real."""

    def __init__(self, source: SyntheticSource):
        self.source = source
        self.tasks = {}
        self.lock = threading.RLock()

    def get_camera_source(self, camera_id):
        return self.source if camera_id == self.source.camera_id else None

    def register_analysis_task_runtime(self, runtime):
        with self.lock:
            require(runtime.task_id not in self.tasks, "runtime registration collision")
            self.tasks[runtime.task_id] = runtime

    def unregister_analysis_task_runtime(self, task_id, *, runtime):
        with self.lock:
            if self.tasks.get(task_id) is runtime:
                if not runtime.detach_if_clean():
                    return False
                del self.tasks[task_id]
            return True


class RuntimeSystem:
    def __init__(self, root: Path, camera: SyntheticCameraManager):
        from argus.capture.task_execution_manager import TaskExecutionManager
        from argus.runtime.task_deployments import TaskDeploymentCoordinator
        from argus.storage.analysis_tasks import AnalysisTaskService
        from argus.storage.database import Database
        from argus.storage.task_deployments import TaskDeploymentService

        self.db = Database(f"sqlite:///{root / 'smoke.db'}")
        self.db.initialize()
        self.tasks = AnalysisTaskService(self.db, models_root=root / "models")
        self.deployments = TaskDeploymentService(self.db, self.tasks)
        # No worker_factory override: exercise the production spawn worker.
        self.executor = TaskExecutionManager(self.tasks, camera, models_root=root / "models",
            max_workers=3, memory_budget_mb=4096)
        self.coordinator = TaskDeploymentCoordinator(self.deployments, self.executor, poll_interval=0.1)
        self.closed = False

    def start(self):
        require(self.coordinator.start(), "runtime coordinator could not acquire exclusive database ownership")

    def close(self):
        if self.closed:
            return
        self.coordinator.close(wait=False)
        workers_clean = self.executor.close()
        owner_clean = self.coordinator.join(15)
        require(workers_clean and owner_clean, "runtime cleanup is unconfirmed; ownership must not be discarded")
        self.db.close()
        self.closed = True

    def command(self, task_id: str, action: str, bundle_id: str | None = None) -> dict:
        view = self.deployments.get(task_id)
        operation = self.deployments.request(task_id, action=action, bundle_id=bundle_id,
            expected_deployment_revision=view["deployment_revision"], requested_by="gpu-smoke-human-operator",
            idempotency_key=f"smoke-{uuid4().hex}")
        self.coordinator.submit(operation["operation_id"])
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            current = self.deployments.get_operation(operation["operation_id"])
            if current["status"] == "applied":
                return current
            require(current["status"] not in {"failed", "superseded"},
                f"{action} failed: {current['status']} / {current.get('failure_code')}")
            time.sleep(0.1)
        raise RuntimeError(f"{action} did not apply before the 150-second smoke deadline")


def observe(system: RuntimeSystem, source: SyntheticSource, expected: dict[str, tuple[str, float]]) -> dict:
    first_frame = source.publish()
    deadline = time.monotonic() + 30
    next_publish = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        # A latest-frame subscriber may legitimately drop a result when the
        # nonblocking DB fence is busy. Feed a bounded real-time sequence, but
        # require all tasks to agree on one exact common original frame.
        candidates = {task_id: {item["source"]["frame_index"]: item
            for item in system.executor.results(task_id) if item["source"]["frame_index"] >= first_frame}
            for task_id in expected}
        common = set.intersection(*(set(items) for items in candidates.values()))
        found = {}
        if common:
            frame_index = max(common)
        else:
            if time.monotonic() >= next_publish:
                source.publish()
                next_publish = time.monotonic() + 0.5
            time.sleep(0.05)
            continue
        for task_id, (bundle_id, score) in expected.items():
            result = candidates[task_id][frame_index]
            require(result["task_id"] == task_id and result["bundle_id"] == bundle_id, "task/bundle result isolation failed")
            require(result["provider"] == "CUDAExecutionProvider", "a worker result did not execute with CUDA")
            require(result["production_allowed"] is False and result["alerts_emitted"] is False,
                "a synthetic shadow task unexpectedly enabled production or alerts")
            require(result["source"]["source_session_id"] == source.session
                and result["source"]["source_epoch"] == 1, "source result identity mismatch")
            require(result["source"]["pts"] is None and result["source"]["source_frame_index"] is None,
                "synthetic capture invented media decoder metadata")
            detections = result["data"]["detections"]
            require(len(detections) == 1 and detections[0]["label"] == "synthetic_object"
                and math.isclose(detections[0]["confidence"], score, abs_tol=1e-5), "synthetic CUDA observation mismatch")
            found[task_id] = result
        return found
    raise RuntimeError(f"workers did not share a synthetic frame >= {first_frame} before the 30-second deadline")


def worker_identity(system: RuntimeSystem, task_id: str) -> str:
    status = system.executor.status(task_id)
    require(status is not None and status["worker"]["provider"] == "CUDAExecutionProvider", "worker CUDA readiness missing")
    require(status["worker"]["state"] == "running", "worker process is not running")
    return status["worker"]["execution_id"]


def wait_for_recovery(system: RuntimeSystem, running: dict[str, tuple[str, float]], prior: dict[str, int]) -> None:
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        recovered = True
        for task_id, (bundle_id, _) in running.items():
            view = system.deployments.get(task_id)
            status = system.executor.status(task_id)
            operation = view.get("operation") or {}
            if not (view["deployment_revision"] > prior[task_id] and operation.get("status") == "applied"
                    and view["applied_bundle_id"] == bundle_id and status is not None
                    and status["bundle_id"] == bundle_id and status["worker"]["state"] == "running"):
                recovered = False
        if recovered:
            return
        time.sleep(0.1)
    raise RuntimeError("persisted applied bundles did not recover after database/coordinator restart")


def exercise(root: Path, report: dict) -> None:
    model_root = root / "models"
    model_root.mkdir()
    # Do not modify PATH or preload libraries for children. The actual spawn
    # workers must exercise production dependency discovery, not a test shim.
    report["worker_environment_modified"] = False
    for version, score in ((1, 0.9), (2, 0.75)):
        build_synthetic_model(model_root / f"synthetic-v{version}.onnx", score)
    report["cuda_preflight"] = preflight_cuda(root, model_root / "synthetic-v1.onnx")
    report["checks"].append({"name": "dynamic_matmul_cuda_profile", "status": "passed"})
    source = SyntheticSource()
    camera = SyntheticCameraManager(source)
    system = RuntimeSystem(root, camera)
    try:
        descriptor = {"name": "Synthetic hardware plumbing graph, not a trained detector",
            "adapter_id": "onnx_detection.v1", "output_schema": "yolov8_detection.v1",
            "labels": ["synthetic_object"], "input_size": [32, 32]}
        models = [system.tasks.register_task_model(f"synthetic-v{version}.onnx", descriptor,
            requested_by="gpu-smoke-human-operator", idempotency_key=f"synthetic-model-{version}") for version in (1, 2)]
        config = {"resources": {"device": "cuda", "memory_limit_mb": 1024,
            "load_timeout_seconds": 60.0, "timeout_seconds": 10.0},
            "sampling": {"fps_target": 10.0}, "scene": {"policy": "none"}}
        tasks = [system.tasks.create_task(source.camera_id, "object_detection.v1", f"Synthetic task {index}", config,
            requested_by="gpu-smoke-human-operator", idempotency_key=f"synthetic-task-{index}") for index in (1, 2)]
        a, b = (task["task_id"] for task in tasks)

        def freeze(task_id, model, key):
            return system.tasks.freeze_bundle(task_id, expected_revision=1,
                model_ids={"detector": model["model_id"]}, requested_by="gpu-smoke-human-operator", idempotency_key=key)["bundle_id"]

        a1, a2, b1 = freeze(a, models[0], "bundle-a1"), freeze(a, models[1], "bundle-a2"), freeze(b, models[0], "bundle-b1")
        report["task_ids"] = [a, b]
        report["bundles"] = {"a1": a1, "a2": a2, "b1": b1}
        system.start()
        system.command(a, "deploy", a1)
        system.command(b, "deploy", b1)
        initial = observe(system, source, {a: (a1, 0.9), b: (b1, 0.9)})
        a_worker, b_worker = worker_identity(system, a), worker_identity(system, b)
        require(a_worker != b_worker, "two tasks unexpectedly share a worker execution identity")
        children = {child.name: child.pid for child in multiprocessing.active_children()}
        require(all(children.get(f"task-worker-{task_id}") not in (None, os.getpid()) for task_id in (a, b)),
            "two independent real spawn child processes were not observed")
        require(children[f"task-worker-{a}"] != children[f"task-worker-{b}"], "tasks unexpectedly share a process")
        report["spawn_children"] = children
        report["checks"].append({"name": "two_real_cuda_tasks_same_frame_independent_results", "status": "passed",
            "capture_sequence": initial[a]["source"]["frame_index"]})

        system.command(a, "stop")
        require(system.executor.cleanup_confirmed(a), "stopped task A still owns resources")
        require(system.executor.results(a) == [], "stopped task A exposed old observations")
        observe(system, source, {b: (b1, 0.9)})
        require(worker_identity(system, b) == b_worker and source.source_token == (source.session, 1),
            "stopping A changed B's worker or source ownership")
        report["checks"].append({"name": "stop_one_task_preserves_peer_and_source", "status": "passed"})

        system.command(a, "deploy", a1)
        require(worker_identity(system, a) != a_worker, "redeploy after stop reused the cleaned worker identity")
        observe(system, source, {a: (a1, 0.9), b: (b1, 0.9)})
        system.command(a, "deploy", a1)
        observe(system, source, {a: (a1, 0.9), b: (b1, 0.9)})
        report["checks"].append({"name": "same_bundle_redeploy_stopped_and_running", "status": "passed"})

        system.command(a, "deploy", a2)
        observe(system, source, {a: (a2, 0.75), b: (b1, 0.9)})
        system.command(a, "rollback", a1)
        observe(system, source, {a: (a1, 0.9), b: (b1, 0.9)})
        require(worker_identity(system, b) == b_worker, "A upgrade/rollback restarted unrelated B")
        report["checks"].append({"name": "complete_bundle_upgrade_and_explicit_rollback", "status": "passed"})

        before = {task_id: system.deployments.get(task_id)["deployment_revision"] for task_id in (a, b)}
        system.close()
        require(not camera.tasks, "first coordinator shutdown leaked registered task runtimes")
        system = RuntimeSystem(root, camera)
        system.start()
        wait_for_recovery(system, {a: (a1, 0.9), b: (b1, 0.9)}, before)
        observe(system, source, {a: (a1, 0.9), b: (b1, 0.9)})
        report["checks"].append({"name": "database_and_coordinator_restart_recovers_authorized_applied_bundles", "status": "passed"})

        system.command(a, "stop")
        stopped = system.deployments.get(a)
        before = {b: system.deployments.get(b)["deployment_revision"]}
        system.close()
        system = RuntimeSystem(root, camera)
        system.start()
        wait_for_recovery(system, {b: (b1, 0.9)}, before)
        observe(system, source, {b: (b1, 0.9)})
        require(system.executor.status(a) is None and system.deployments.get(a)["desired_state"] == "stopped"
            and system.deployments.get(a)["deployment_revision"] == stopped["deployment_revision"],
            "persistently stopped task was resurrected during restart")
        report["checks"].append({"name": "durable_stop_survives_restart_without_resurrection", "status": "passed"})
        report["final_deployments"] = {task_id: system.deployments.get(task_id) for task_id in (a, b)}
    finally:
        system.close()
        require(not camera.tasks, "final shutdown leaked task registrations")
    report["checks"].append({"name": "workers_shared_memory_and_owner_cleanup", "status": "passed"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="explicit authorization to run real CUDA computation")
    parser.add_argument("--output-root", type=Path, required=True, help="new absolute isolated output directory; must not exist")
    args = parser.parse_args()
    if not args.execute:
        parser.error("real GPU smoke requires explicit --execute")
    if sys.platform != "win32" or os.name != "nt":
        parser.error("this smoke is restricted to the authorized Windows GPU test host")
    root = args.output_root
    if not root.is_absolute() or root.exists() or not root.parent.is_dir():
        parser.error("--output-root must be a new absolute directory beneath an existing parent")
    root = root.resolve()
    root.mkdir(exist_ok=False)
    report = {"schema_version": "argus.cuda-task-runtime-smoke.v1", "status": "failed",
        "started_at": datetime.now(timezone.utc).isoformat(), "hardware_plumbing_only": True,
        "quality_evaluated": False, "production_allowed": False, "training_performed": False,
        "camera_opened": False, "source_videos_read_or_modified": False, "checks": [],
        "planned_checks": ["dynamic_matmul_cuda_profile", "two_real_cuda_tasks_same_frame_independent_results",
            "stop_one_task_preserves_peer_and_source", "same_bundle_redeploy_stopped_and_running",
            "complete_bundle_upgrade_and_explicit_rollback",
            "database_and_coordinator_restart_recovers_authorized_applied_bundles",
            "durable_stop_survives_restart_without_resurrection", "workers_shared_memory_and_owner_cleanup"],
        "limitations": ["Synthetic graph and source do not measure detection quality, recall, FP/hour or live-camera throughput.",
            "SceneGuard and legacy anomaly equivalence are not exercised; production gates remain unchanged.",
            "Restart exercise recreates database connections and coordinator objects, not a machine crash or power failure."]}
    try:
        exercise(root, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", None), "message": str(exc)}
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        completed = {item["name"] for item in report["checks"] if item["status"] == "passed"}
        report["unverified_checks"] = [name for name in report["planned_checks"] if name not in completed]
        (root / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(root / "result.json"),
        "hardware_plumbing_only": True, "production_allowed": False}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
