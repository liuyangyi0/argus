"""Opt-in Windows/CUDA process-restart smoke with an isolated database/source.

Runs real production task workers, never cameras, source videos or training.
Windows jobs contain only smoke-owned controllers and descendants. Jobs stay
open during crash observation, so they cannot hide orphan-worker failures.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

from run_gpu_task_runtime_smoke import (
    RuntimeSystem, SyntheticCameraManager, SyntheticSource, build_synthetic_model,
    observe, require, wait_for_recovery, worker_identity,
)

CHECKS = ["controller_crash_workers_exit_without_test_job_kill",
    "new_controller_process_recovers_only_authorized_bundles", "crashed_gpu_worker_recovers_without_restarting_peer",
    "post_restart_upgrade_rollback_preserves_peer",
    "durable_stop_survives_real_process_restart", "owned_controllers_jobs_and_workers_cleaned"]


class OwnedWindowsJob:
    """Terminate only this test's descendants when its private job is closed."""

    def __init__(self, process):
        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                ("processes", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        self.api.IsProcessInJob.restype = wintypes.BOOL
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.WaitForSingleObject.restype = wintypes.DWORD
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.api.AssignProcessToJobObject(self.handle, wintypes.HANDLE(process.sentinel)):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            self.close()
            raise

    def capture_worker_handles(self, pids):
        handles = []
        try:
            for pid in pids:
                # IPC reports these PIDs from multiprocessing.active_children()
                # while the owned controller is alive. Hold handles to avoid
                # confusing PID reuse with an old worker still running.
                handle = self.api.OpenProcess(0x00101000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                handles.append(handle)
                contained = wintypes.BOOL()
                if not self.api.IsProcessInJob(handle, self.handle, ctypes.byref(contained)):
                    raise ctypes.WinError(ctypes.get_last_error())
                require(bool(contained.value), "reported worker is not in this smoke-owned job")
            return handles
        except Exception:
            for handle in handles:
                self.api.CloseHandle(handle)
            raise

    def close(self):
        if self.handle:
            if not self.api.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None


def controller(root_text, plan, running_keys, prior, connection, gate):
    system = None
    try:
        require(gate.wait(30), "test controller was not contained before startup")
        source = SyntheticSource()
        camera = SyntheticCameraManager(source)
        system = RuntimeSystem(Path(root_text), camera)
        system.start()
        expected = {plan[key]: (plan[f"{key}1"], 0.9) for key in running_keys}
        wait_for_recovery(system, expected, prior)
        observations = observe(system, source, expected)
        if "a" not in running_keys:
            require(system.executor.status(plan["a"]) is None
                and system.deployments.get(plan["a"])["desired_state"] == "stopped",
                "persisted stop was resurrected in a new controller process")
        children = {child.name: child.pid for child in multiprocessing.active_children()}
        pids = [children[f"task-worker-{plan[key]}"] for key in running_keys]
        require(len(set(pids)) == len(running_keys), "real workers are not process-isolated")
        connection.send({"kind": "ready", "pid": os.getpid(), "source_session": source.session,
            "worker_pids": pids, "worker_execution_ids": [worker_identity(system, plan[key]) for key in running_keys],
            "deployments": {plan[key]: system.deployments.get(plan[key]) for key in ("a", "b")},
            "observed_tasks": list(observations)})
        while connection.poll(240):
            command = connection.recv()
            if command == "finish":
                break
            b_identity = worker_identity(system, plan["b"])
            if command == "crash_a":
                prior_revision = system.deployments.get(plan["a"])["deployment_revision"]
                previous_identity = worker_identity(system, plan["a"])
                with system.executor._lock:
                    worker = system.executor._executions[plan["a"]].worker
                    process = worker._process
                require(process is not None and process.pid in {child.pid for child in multiprocessing.active_children()},
                    "failure injection does not own the target worker")
                process.terminate()
                process.join(10)
                require(not process.is_alive(), "owned worker did not terminate for failure injection")
                observe(system, source, {plan["b"]: (plan["b1"], 0.9)})
                wait_for_recovery(system, {plan["a"]: (plan["a1"], 0.9)}, {plan["a"]: prior_revision})
                observe(system, source, {plan["a"]: (plan["a1"], 0.9), plan["b"]: (plan["b1"], 0.9)})
                require(worker_identity(system, plan["a"]) != previous_identity, "dead worker execution identity was reused")
            elif command == "rollback":
                system.command(plan["a"], "deploy", plan["a2"])
                observe(system, source, {plan["a"]: (plan["a2"], 0.75), plan["b"]: (plan["b1"], 0.9)})
                receipt = system.command(plan["a"], "rollback", plan["a1"])
                observe(system, source, {plan["a"]: (plan["a1"], 0.9), plan["b"]: (plan["b1"], 0.9)})
                require(receipt["action"] == "rollback", "rollback receipt was not persisted")
            elif command == "stop_a":
                system.command(plan["a"], "stop")
                observe(system, source, {plan["b"]: (plan["b1"], 0.9)})
                require(system.executor.cleanup_confirmed(plan["a"]) and not system.executor.results(plan["a"]),
                    "stopped A retained resources or stale observations")
            else:
                raise RuntimeError("unknown fixed smoke command")
            require(worker_identity(system, plan["b"]) == b_identity, "command on A restarted unrelated B")
            connection.send({"kind": "ack", "command": command,
                "deployments": {plan[key]: system.deployments.get(plan[key]) for key in ("a", "b")}})
        system.close()
        system = None
        require(not camera.tasks, "controller shutdown leaked runtime registrations")
        connection.send({"kind": "closed"})
    except Exception as exc:
        connection.send({"kind": "failed", "error_type": type(exc).__name__,
            "error_code": getattr(exc, "code", None), "message": str(exc)})
        raise
    finally:
        if system is not None:
            system.close()
        connection.close()


def receive(connection, kind, timeout=180):
    require(connection.poll(timeout), f"controller did not return {kind} before its deadline")
    result = connection.recv()
    require(isinstance(result, dict) and result.get("kind") == kind,
        f"controller response failed: {result}")
    return result


def run(root, report):
    models = root / "models"
    models.mkdir()
    for version, score in ((1, 0.9), (2, 0.75)):
        build_synthetic_model(models / f"synthetic-v{version}.onnx", score)
    source = SyntheticSource()
    seed = RuntimeSystem(root, SyntheticCameraManager(source))
    plan = {}
    try:
        descriptor = {"name": "Synthetic restart validation, not trained detector", "adapter_id": "onnx_detection.v1",
            "output_schema": "yolov8_detection.v1", "labels": ["synthetic_object"], "input_size": [32, 32]}
        versions = [seed.tasks.register_task_model(f"synthetic-v{v}.onnx", descriptor,
            requested_by="gpu-smoke-operator", idempotency_key=f"model-{v}") for v in (1, 2)]
        for key in ("a", "b"):
            task = seed.tasks.create_task(source.camera_id, "object_detection.v1", f"Restart task {key}",
                {"resources": {"device": "cuda", "memory_limit_mb": 1024, "load_timeout_seconds": 60.0,
                    "timeout_seconds": 10.0}, "sampling": {"fps_target": 10}, "scene": {"policy": "none"}},
                requested_by="gpu-smoke-operator", idempotency_key=f"create-{key}")
            plan[key] = task["task_id"]
            for version in (1, 2) if key == "a" else (1,):
                plan[f"{key}{version}"] = seed.tasks.freeze_bundle(task["task_id"], expected_revision=1,
                    model_ids={"detector": versions[version - 1]["model_id"]}, requested_by="gpu-smoke-operator",
                    idempotency_key=f"freeze-{key}{version}")["bundle_id"]
            seed.deployments.request(task["task_id"], action="deploy", bundle_id=plan[f"{key}1"],
                expected_deployment_revision=0, requested_by="gpu-smoke-operator", idempotency_key=f"deploy-{key}")
    finally:
        seed.close()
    context = multiprocessing.get_context("spawn")
    owned = []

    def launch(keys, prior):
        parent, child = context.Pipe()
        gate = context.Event()
        process = context.Process(target=controller, args=(str(root), plan, keys, prior, child, gate),
            name="argus-owned-gpu-restart-controller")
        process.start()
        child.close()
        entry = {"process": process, "connection": parent, "job": None}
        owned.append(entry)
        entry["job"] = OwnedWindowsJob(process)
        gate.set()
        return entry, receive(parent, "ready")

    def finish(entry):
        entry["connection"].send("finish")
        receive(entry["connection"], "closed", 40)
        entry["process"].join(20)
        require(entry["process"].exitcode == 0, "controller process did not exit cleanly")
        entry["job"].close()
        entry["connection"].close()

    try:
        first, first_ready = launch(("a", "b"), {plan["a"]: 0, plan["b"]: 0})
        handles = first["job"].capture_worker_handles(first_ready["worker_pids"])
        try:
            # Intentionally no cooperative close: exercise an actual controller
            # crash. Its job remains open while waiting for production workers
            # to notice the broken pipe and exit themselves.
            first["process"].terminate()
            first["process"].join(10)
            require(not first["process"].is_alive(), "owned crashed controller did not exit")
            deadline = time.monotonic() + 20
            for handle in handles:
                wait_ms = max(0, int((deadline - time.monotonic()) * 1000))
                require(first["job"].api.WaitForSingleObject(handle, wait_ms) == 0,
                    "GPU worker survived its controller crash; job containment has not been closed yet")
        finally:
            for handle in handles:
                first["job"].api.CloseHandle(handle)
        first["job"].close()
        first["connection"].close()
        report["checks"].append({"name": "controller_crash_workers_exit_without_test_job_kill", "status": "passed"})
        prior = {task_id: view["deployment_revision"] for task_id, view in first_ready["deployments"].items()}
        second, second_ready = launch(("a", "b"), prior)
        require(second_ready["source_session"] != first_ready["source_session"], "new controller reused old source session")
        require(not set(second_ready["worker_execution_ids"]) & set(first_ready["worker_execution_ids"]),
            "new controller reused old worker execution identities")
        report["controllers"] = [first_ready, second_ready]
        report["checks"].append({"name": "new_controller_process_recovers_only_authorized_bundles", "status": "passed"})
        second["connection"].send("crash_a")
        receive(second["connection"], "ack")
        report["checks"].append({"name": "crashed_gpu_worker_recovers_without_restarting_peer", "status": "passed"})
        second["connection"].send("rollback")
        receive(second["connection"], "ack")
        report["checks"].append({"name": "post_restart_upgrade_rollback_preserves_peer", "status": "passed"})
        second["connection"].send("stop_a")
        stopped = receive(second["connection"], "ack")["deployments"]
        finish(second)
        third, third_ready = launch(("b",), {plan["b"]: stopped[plan["b"]]["deployment_revision"]})
        require(third_ready["deployments"][plan["a"]]["deployment_revision"] == stopped[plan["a"]]["deployment_revision"],
            "stopped task revision changed during restart")
        report["controllers"].append(third_ready)
        report["checks"].append({"name": "durable_stop_survives_real_process_restart", "status": "passed"})
        finish(third)
    finally:
        for entry in reversed(owned):
            if entry["job"] is not None:
                entry["job"].close()
            process = entry["process"]
            if process.is_alive():
                process.terminate()
            process.join(10)
            require(not process.is_alive(), "owned controller cleanup is unconfirmed")
            entry["connection"].close()
            process.close()
    report["checks"].append({"name": "owned_controllers_jobs_and_workers_cleaned", "status": "passed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute or sys.platform != "win32":
        parser.error("requires explicit --execute on the authorized Windows GPU host")
    root = args.output_root
    if not root.is_absolute() or root.exists() or not root.parent.is_dir():
        parser.error("requires a new absolute output directory beneath an existing parent")
    root = root.resolve()
    root.mkdir()
    report = {"schema_version": "argus.cuda-task-process-restart-smoke.v1", "status": "failed", "checks": [],
        "planned_checks": CHECKS,
        "started_at": datetime.now(timezone.utc).isoformat(), "hardware_plumbing_only": True,
        "production_allowed": False, "quality_evaluated": False, "training_performed": False,
        "source_videos_read_or_modified": False, "camera_opened": False,
        "limitations": ["Synthetic GPU graph, not business quality acceptance.",
            "Actual controller process termination and restart, not machine reboot or GPU reset."]}
    try:
        run(root, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", None), "message": str(exc)}
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        completed = {item["name"] for item in report["checks"] if item["status"] == "passed"}
        report["unverified_checks"] = [name for name in CHECKS if name not in completed]
        (root / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(root / "result.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
