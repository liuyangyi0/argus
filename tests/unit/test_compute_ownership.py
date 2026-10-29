"""Host compute handle exclusion; subprocesses only lock files, never engines."""

import gc
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import weakref

import pytest

from argus.runtime.compute_ownership import ComputeOwnership


class Owner:
    pass


def test_constructor_and_snapshot_are_inert_and_default_is_database_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    first, second = ComputeOwnership(), ComputeOwnership()
    assert first.path == second.path == tmp_path / ".argus" / "runtime" / "cuda0-compute.lock"
    assert first.snapshot() == {"held": False, "cleanup_pending": False}
    assert not first.path.exists() and not first.path.parent.exists()
    assert first.release(Owner()) is True
    assert not first.path.parent.exists()


def test_one_manager_can_reacquire_but_other_instances_and_identities_cannot(tmp_path):
    first, second = ComputeOwnership(tmp_path / "compute.lock"), ComputeOwnership(tmp_path / "compute.lock")
    owner, peer = Owner(), Owner()
    try:
        assert first.acquire(owner) is True
        assert first.acquire(owner) is True
        assert first.acquire(peer) is False
        assert second.acquire(owner) is False
        assert second.acquire(peer) is False
        assert first.release(peer) is False
        assert first.snapshot() == {"held": True, "cleanup_pending": False}
        assert first.release(owner) is True
        assert first.release(owner) is True
        assert second.acquire(peer) is True
    finally:
        assert first.release(owner)
        assert second.release(peer)


def test_alias_paths_share_canonical_owner_identity(tmp_path):
    path = tmp_path / "compute.lock"
    alias = tmp_path / "nonexistent" / ".." / path.name
    first, second = ComputeOwnership(path), ComputeOwnership(alias)
    assert first.key == second.key
    owner, peer = Owner(), Owner()
    try:
        assert first.acquire(owner)
        assert second.acquire(peer) is False
    finally:
        assert first.release(owner)
        assert second.release(peer)


@pytest.mark.skipif(os.name != "nt", reason="Windows case-insensitive path aliases")
def test_windows_case_aliases_share_process_local_registry(tmp_path):
    first = ComputeOwnership(tmp_path / "Compute.Lock")
    second = ComputeOwnership(tmp_path / "COMPUTE.LOCK")
    assert first.key == second.key
    owner, peer = Owner(), Owner()
    try:
        assert first.acquire(owner)
        assert second.acquire(peer) is False
    finally:
        assert first.release(owner)
        assert second.release(peer)


def test_registry_retains_owner_and_handle_when_external_references_disappear(tmp_path):
    owner = Owner()
    ownership = ComputeOwnership(tmp_path / "compute.lock")
    owner_ref, ownership_ref = weakref.ref(owner), weakref.ref(ownership)
    assert ownership.acquire(owner)
    del owner, ownership
    gc.collect()
    try:
        assert owner_ref() is not None and ownership_ref() is not None
        peer = ComputeOwnership(tmp_path / "compute.lock")
        assert peer.acquire(Owner()) is False
    finally:
        assert ownership_ref().release(owner_ref())


def test_unconfirmed_release_retains_owner_and_can_be_retried(tmp_path, monkeypatch):
    ownership, peer = ComputeOwnership(tmp_path / "compute.lock"), ComputeOwnership(tmp_path / "compute.lock")
    owner, other = Owner(), Owner()
    assert ownership.acquire(owner)
    close = ownership._close_handle
    monkeypatch.setattr(ownership, "_close_handle", lambda: False)
    try:
        assert ownership.release(owner) is False
        assert ownership.snapshot() == {"held": True, "cleanup_pending": True}
        assert peer.acquire(other) is False
        assert ownership.acquire(owner) is False  # Unknown release is not a fresh compute grant.
        monkeypatch.setattr(ownership, "_close_handle", close)
        assert ownership.release(owner)
        assert peer.acquire(other)
    finally:
        monkeypatch.setattr(ownership, "_close_handle", close)
        assert ownership.release(owner)
        assert peer.release(other)


def test_acquire_failure_after_lock_does_not_drop_unknown_handle(tmp_path, monkeypatch):
    from argus.runtime import compute_ownership

    ownership, peer = ComputeOwnership(tmp_path / "compute.lock"), ComputeOwnership(tmp_path / "compute.lock")
    owner, other = Owner(), Owner()
    lock_file, close = compute_ownership._lock_file, ownership._close_handle
    def locked_but_acknowledgement_lost(stream):
        lock_file(stream)
        raise OSError("synthetic lost lock acknowledgement")
    monkeypatch.setattr(compute_ownership, "_lock_file", locked_but_acknowledgement_lost)
    monkeypatch.setattr(ownership, "_close_handle", lambda: False)
    try:
        assert ownership.acquire(owner) is False
        assert ownership.snapshot()["cleanup_pending"] is True
        assert peer.acquire(other) is False
        monkeypatch.setattr(ownership, "_close_handle", close)
        assert ownership.release(owner)
        monkeypatch.setattr(compute_ownership, "_lock_file", lock_file)
        assert peer.acquire(other)
    finally:
        monkeypatch.setattr(compute_ownership, "_lock_file", lock_file)
        monkeypatch.setattr(ownership, "_close_handle", close)
        assert ownership.release(owner)
        assert peer.release(other)


def test_known_acquire_failure_closes_handle_and_allows_retry(tmp_path, monkeypatch):
    from argus.runtime import compute_ownership

    ownership, owner = ComputeOwnership(tmp_path / "compute.lock"), Owner()
    lock_file = compute_ownership._lock_file
    monkeypatch.setattr(compute_ownership, "_lock_file", lambda _stream: (_ for _ in ()).throw(OSError("synthetic busy")))
    assert ownership.acquire(owner) is False
    assert ownership.snapshot() == {"held": False, "cleanup_pending": False}
    monkeypatch.setattr(compute_ownership, "_lock_file", lock_file)
    try:
        assert ownership.acquire(owner)
    finally:
        assert ownership.release(owner)


_CHILD_ATTEMPT = """
import sys
from argus.runtime.compute_ownership import ComputeOwnership
ownership, owner = ComputeOwnership(sys.argv[1]), object()
acquired = ownership.acquire(owner)
print('acquired' if acquired else 'blocked', flush=True)
if acquired:
    ownership.release(owner)
"""

_CHILD_HOLD = """
import sys
from argus.runtime.compute_ownership import ComputeOwnership
ownership, owner = ComputeOwnership(sys.argv[1]), object()
assert ownership.acquire(owner)
print('acquired', flush=True)
sys.stdin.readline()
ownership.release(owner)
"""


def child_environment():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    return environment


def test_os_owner_excludes_another_python_process_and_releases_after_close(tmp_path):
    path, owner = tmp_path / "compute.lock", Owner()
    ownership = ComputeOwnership(path)
    assert ownership.acquire(owner)
    try:
        blocked = subprocess.run([sys.executable, "-c", _CHILD_ATTEMPT, str(path)],
            capture_output=True, text=True, timeout=10, env=child_environment(), check=True)
        assert blocked.stdout.strip() == "blocked"
        assert ownership.release(owner)
        acquired = subprocess.run([sys.executable, "-c", _CHILD_ATTEMPT, str(path)],
            capture_output=True, text=True, timeout=10, env=child_environment(), check=True)
        assert acquired.stdout.strip() == "acquired"
    finally:
        assert ownership.release(owner)


def test_os_automatically_releases_owner_after_process_death(tmp_path):
    path = tmp_path / "compute.lock"
    child = subprocess.Popen([sys.executable, "-c", _CHILD_HOLD, str(path)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=child_environment())
    lines = queue.Queue()
    reader = threading.Thread(target=lambda: lines.put(child.stdout.readline()), daemon=True)
    reader.start()
    ownership, owner = ComputeOwnership(path), Owner()
    try:
        assert lines.get(timeout=5).strip() == "acquired"
        assert ownership.acquire(owner) is False
        child.kill()
        child.communicate(timeout=5)
        assert ownership.acquire(owner) is True
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)
        reader.join(timeout=1)
        assert ownership.release(owner)


def test_none_is_not_an_owner_identity(tmp_path):
    with pytest.raises(ValueError):
        ComputeOwnership(tmp_path / "compute.lock").acquire(None)


def test_startup_identity_transfer_never_unlocks_or_admits_a_peer(tmp_path):
    ownership, startup, manager = ComputeOwnership(tmp_path / "transfer.lock"), Owner(), Owner()
    peer, other = ComputeOwnership(ownership.path), Owner()
    assert ownership.transfer(startup, manager) is False
    try:
        assert ownership.acquire(startup)
        assert ownership.transfer(other, manager) is False
        assert ownership.transfer(startup, manager) is True
        assert peer.acquire(other) is False
        assert ownership.release(startup) is False
        assert ownership.acquire(manager) is True
    finally:
        assert ownership.release(manager)
        assert peer.release(other)


def test_surviving_training_child_excludes_new_main_owner_after_parent_release(tmp_path):
    parent, owner = ComputeOwnership(tmp_path / "parent.lock"), Owner()
    child, child_owner = ComputeOwnership(parent.worker_path, guard_training_workers=False), Owner()
    peer, peer_owner = ComputeOwnership(parent.path), Owner()
    try:
        assert parent.acquire(owner)
        assert child.acquire(child_owner)
        assert parent.release(owner)
        assert peer.acquire(peer_owner) is False
        assert child.release(child_owner)
        assert peer.acquire(peer_owner) is True
    finally:
        assert parent.release(owner)
        assert child.release(child_owner)
        assert peer.release(peer_owner)


def test_unknown_secondary_probe_cleanup_retains_main_owner(tmp_path, monkeypatch):
    from argus.runtime import compute_ownership
    ownership, owner = ComputeOwnership(tmp_path / "probe.lock"), Owner()
    peer, other = ComputeOwnership(ownership.path), Owner()
    original_lock, original_close = compute_ownership._lock_file, ownership._close_handle
    calls = []
    def unknown_secondary_lock(stream):
        original_lock(stream)
        calls.append(stream)
        if len(calls) == 2:
            raise OSError("secondary lock acknowledgement lost")
    monkeypatch.setattr(compute_ownership, "_lock_file", unknown_secondary_lock)
    monkeypatch.setattr(ownership, "_close_handle", lambda: False)
    try:
        assert ownership.acquire(owner) is False
        assert ownership.snapshot()["cleanup_pending"] is True
        assert peer.acquire(other) is False
        monkeypatch.setattr(ownership, "_close_handle", original_close)
        assert ownership.release(owner)
        monkeypatch.setattr(compute_ownership, "_lock_file", original_lock)
        assert peer.acquire(other)
    finally:
        monkeypatch.setattr(ownership, "_close_handle", original_close)
        monkeypatch.setattr(compute_ownership, "_lock_file", original_lock)
        assert ownership.release(owner)
        assert peer.release(other)


def test_secondary_close_failure_keeps_real_main_handle_until_positive_retry(tmp_path, monkeypatch):
    ownership, owner = ComputeOwnership(tmp_path / "probe-close.lock"), Owner()
    peer, other = ComputeOwnership(ownership.path), Owner()
    original_open = Path.open
    wrapped = []
    class AmbiguousClose:
        fail = True
        def __init__(self, stream):
            self.stream = stream
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def close(self):
            if self.fail:
                raise OSError("synthetic close did not confirm release")
            self.stream.close()
    def open_stream(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == ownership.worker_path:
            stream = AmbiguousClose(stream)
            wrapped.append(stream)
        return stream
    monkeypatch.setattr(Path, "open", open_stream)
    try:
        assert ownership.acquire(owner) is False
        assert ownership.snapshot() == {"held": False, "cleanup_pending": True}
        assert ownership._stream.closed is False
        assert peer.acquire(other) is False
        wrapped[0].fail = False
        monkeypatch.setattr(Path, "open", original_open)
        assert ownership.release(owner) is True
        assert peer.acquire(other) is True
    finally:
        for stream in wrapped:
            stream.fail = False
        monkeypatch.setattr(Path, "open", original_open)
        assert ownership.release(owner)
        assert peer.release(other)


def test_secondary_guard_handle_does_not_outlive_its_python_process(tmp_path):
    parent = ComputeOwnership(tmp_path / "parent.lock")
    child = subprocess.Popen([sys.executable, "-c", _CHILD_HOLD.replace(
        "ComputeOwnership(sys.argv[1])", "ComputeOwnership(sys.argv[1], guard_training_workers=False)"),
        str(parent.worker_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=child_environment())
    lines = queue.Queue()
    reader = threading.Thread(target=lambda: lines.put(child.stdout.readline()), daemon=True)
    reader.start()
    owner = Owner()
    try:
        assert lines.get(timeout=5).strip() == "acquired"
        assert parent.acquire(owner) is False
        child.kill()
        child.communicate(timeout=5)
        assert parent.acquire(owner) is True
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)
        reader.join(timeout=1)
        assert parent.release(owner)


_CHILD_AUTHORIZED_GUARD = r"""
import importlib.abc
import json
import queue
import sys
import threading

engine_imports = []
class NoModelEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'anomalib', 'ultralytics', 'openvino', 'onnxruntime'}:
            engine_imports.append(fullname)
            raise ImportError('The handshake subprocess forbids model SDKs')
sys.meta_path.insert(0, NoModelEngines())

from argus.runtime.task_training import _training_child_authorized

eof = object()
messages = queue.Queue()
def read_input():
    for line in sys.stdin:
        try:
            messages.put(json.loads(line))
        except Exception:
            messages.put({'state': 'invalid'})
    messages.put(eof)
threading.Thread(target=read_input, daemon=True).start()

class Connection:
    pending = None
    def send(self, value):
        print(json.dumps(value), flush=True)
    def poll(self, timeout):
        try:
            self.pending = messages.get(timeout=timeout)
        except queue.Empty:
            return False
        return True
    def recv(self):
        value = self.pending
        self.pending = None
        if value is None:
            value = messages.get(timeout=5)
        if value is eof:
            raise EOFError
        return value

connection = Connection()
authorized = _training_child_authorized(connection, {'execution_id': sys.argv[2]}, sys.argv[1])
print(json.dumps({'authorized': authorized, 'engine_imports': engine_imports}), flush=True)
if authorized:
    # This subprocess intentionally has no execute/train/inference branch.
    # Retain the real secondary handle while simulating a still-live worker.
    assert connection.recv() == {'state': 'exit'}
"""


@pytest.mark.parametrize("authorization", ["matching", "wrong-execution", "eof"])
def test_real_handshake_and_secondary_handle_jointly_fence_new_compute_owner(tmp_path, authorization):
    parent, owner = ComputeOwnership(tmp_path / "handshake-main.lock"), Owner()
    peer, peer_owner = ComputeOwnership(parent.path), Owner()
    execution_id = "a" * 32
    assert parent.acquire(owner)
    child = subprocess.Popen([sys.executable, "-c", _CHILD_AUTHORIZED_GUARD,
        str(parent.worker_path), execution_id], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=child_environment())
    output = queue.Queue()
    def read_output():
        for line in child.stdout:
            try:
                output.put(json.loads(line))
            except ValueError:
                output.put({"unexpected_stdout": line})
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        assert output.get(timeout=15) == {"state": "ready", "execution_id": execution_id}
        # The parent can disappear after READY: the child's independent real
        # OS handle, not its PID or the parent's registry, fences new owners.
        assert parent.release(owner)
        assert peer.acquire(peer_owner) is False
        if authorization == "eof":
            child.stdin.close()
        else:
            command = {"state": "authorize", "execution_id": execution_id if authorization == "matching" else "b" * 32}
            child.stdin.write(json.dumps(command) + "\n")
            child.stdin.flush()
        assert output.get(timeout=10) == {"authorized": authorization == "matching", "engine_imports": []}
        if authorization == "matching":
            assert child.poll() is None
            assert peer.acquire(peer_owner) is False
            child.stdin.write(json.dumps({"state": "exit"}) + "\n")
            child.stdin.flush()
        assert child.wait(timeout=10) == 0
        assert peer.acquire(peer_owner) is True
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        reader.join(timeout=2)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()
        assert parent.release(owner)
        assert peer.release(peer_owner)
