"""An application binds derived training to one stable registry facade."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from argus.dashboard.model_runtime import get_registry
from argus.storage.model_registry import ModelRegistry


class DatabaseOwner:
    def get_session(self):
        raise AssertionError("registry construction must not open a session")


def test_parallel_requests_share_the_application_registry():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        database=DatabaseOwner(),
    )))
    with ThreadPoolExecutor(max_workers=8) as pool:
        registries = list(pool.map(lambda _: get_registry(request), range(20)))
    assert all(registry is registries[0] for registry in registries)


def test_replacing_database_never_reuses_old_registry():
    state = SimpleNamespace(database=DatabaseOwner())
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    first = get_registry(request)
    state.database = DatabaseOwner()
    second = get_registry(request)
    assert first is not second
    assert not first.uses_same_sessions_as(second)


def test_separate_facades_can_prove_the_same_database_owner():
    owner = DatabaseOwner()
    first = ModelRegistry(owner.get_session)
    second = ModelRegistry(owner.get_session)
    assert first.uses_same_sessions_as(second)
