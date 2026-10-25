from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, AuthConfig
from argus.dashboard.app import create_app
from argus.dashboard.auth import create_session_token, hash_password
from argus.storage.database import Database


class _BackupManager:
    def __init__(self) -> None:
        self.created: list[bool] = []
        self.restored: list[str] = []
        self.deleted: list[str] = []

    def list_backups(self) -> list[dict[str, object]]:
        return [{"name": "backup-20260525", "size_mb": 1.2}]

    def create_backup(self, *, include_models: bool = False) -> dict[str, object]:
        self.created.append(include_models)
        return {"name": "backup-20260525", "size_mb": 1.2}

    def restore_database(self, backup_name: str) -> bool:
        self.restored.append(backup_name)
        return True

    def delete_backup(self, backup_name: str) -> bool:
        self.deleted.append(backup_name)
        return True


@pytest.fixture
def database(tmp_path):
    db = Database(database_url=f"sqlite:///{tmp_path / 'backup-auth.db'}")
    db.initialize()
    yield db
    db.close()


def _client_for_role(role: str, tmp_path, database):
    config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="backup-test-token"))
    database.create_user(f"{role}-user", hash_password("backup-password"), role, None)
    app = create_app(database=database, config=config, alerts_dir=str(tmp_path / "alerts"))
    manager = _BackupManager()
    app.state.backup_manager = manager
    client = TestClient(app)
    token = create_session_token(f"{role}-user", role, app.state.session_secret)
    client.cookies.set("argus_session", token)
    return client, manager


def test_backup_mutations_require_admin_role(tmp_path, database):
    client, manager = _client_for_role("engineer", tmp_path, database)

    create_resp = client.post("/api/backup/create")
    restore_resp = client.post("/api/backup/restore", data={"backup_name": "backup-20260525"})
    delete_resp = client.delete("/api/backup/backup-20260525")

    assert create_resp.status_code == 403
    assert restore_resp.status_code == 403
    assert delete_resp.status_code == 403
    assert manager.created == []
    assert manager.restored == []
    assert manager.deleted == []


def test_admin_can_manage_backups(tmp_path, database):
    client, manager = _client_for_role("admin", tmp_path, database)

    create_resp = client.post("/api/backup/create")
    restore_resp = client.post("/api/backup/restore", data={"backup_name": "backup-20260525"})
    delete_resp = client.delete("/api/backup/backup-20260525")

    assert create_resp.status_code == 200
    assert restore_resp.status_code == 200
    assert delete_resp.status_code == 200
    assert manager.created == [False]
    assert manager.restored == ["backup-20260525"]
    assert manager.deleted == ["backup-20260525"]


def test_engineer_can_list_backups(tmp_path, database):
    client, _manager = _client_for_role("engineer", tmp_path, database)

    response = client.get("/api/backup/list/json")

    assert response.status_code == 200
    assert response.json()["data"]["backups"][0]["name"] == "backup-20260525"


@pytest.mark.parametrize("identity", ["api_token", "disabled", "inactive", "no_database"])
def test_restore_cannot_reinstate_personal_credentials_without_human_admin(
    tmp_path, database, identity,
):
    database.create_user("admin-user", hash_password("backup-password"), "admin", None)
    config = ArgusConfig(auth=AuthConfig(enabled=identity != "disabled", api_token="backup-test-token"))
    app = create_app(
        database=None if identity == "no_database" else database,
        config=config,
        alerts_dir=str(tmp_path / "alerts"),
    )
    manager = _BackupManager()
    app.state.backup_manager = manager
    client = TestClient(app)
    if identity == "api_token":
        client.headers["X-API-Token"] = "backup-test-token"
    else:
        client.cookies.set("argus_session", create_session_token(
            "admin-user", "admin", app.state.session_secret,
        ))
    if identity == "inactive":
        database.update_user("admin-user", active=False)

    response = client.post("/api/backup/restore", data={"backup_name": "backup-20260525"})

    assert response.status_code == (401 if identity == "inactive" else 403)
    assert manager.restored == []
    if identity in {"api_token", "disabled"}:
        # Creating/listing backups does not overwrite account authority.
        assert client.post("/api/backup/create").status_code == 200
        assert client.get("/api/backup/list/json").status_code == 200
