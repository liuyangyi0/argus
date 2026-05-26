from __future__ import annotations

from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, AuthConfig
from argus.dashboard.app import create_app
from argus.dashboard.auth import create_session_token


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


def _client_for_role(role: str, tmp_path):
    config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="backup-test-token"))
    app = create_app(config=config, alerts_dir=str(tmp_path / "alerts"))
    manager = _BackupManager()
    app.state.backup_manager = manager
    client = TestClient(app)
    token = create_session_token(f"{role}-user", role, app.state.session_secret)
    client.cookies.set("argus_session", token)
    return client, manager


def test_backup_mutations_require_admin_role(tmp_path):
    client, manager = _client_for_role("engineer", tmp_path)

    create_resp = client.post("/api/backup/create")
    restore_resp = client.post("/api/backup/restore", data={"backup_name": "backup-20260525"})
    delete_resp = client.delete("/api/backup/backup-20260525")

    assert create_resp.status_code == 403
    assert restore_resp.status_code == 403
    assert delete_resp.status_code == 403
    assert manager.created == []
    assert manager.restored == []
    assert manager.deleted == []


def test_admin_can_manage_backups(tmp_path):
    client, manager = _client_for_role("admin", tmp_path)

    create_resp = client.post("/api/backup/create")
    restore_resp = client.post("/api/backup/restore", data={"backup_name": "backup-20260525"})
    delete_resp = client.delete("/api/backup/backup-20260525")

    assert create_resp.status_code == 200
    assert restore_resp.status_code == 200
    assert delete_resp.status_code == 200
    assert manager.created == [False]
    assert manager.restored == ["backup-20260525"]
    assert manager.deleted == ["backup-20260525"]


def test_engineer_can_list_backups(tmp_path):
    client, _manager = _client_for_role("engineer", tmp_path)

    response = client.get("/api/backup/list/json")

    assert response.status_code == 200
    assert response.json()["data"]["backups"][0]["name"] == "backup-20260525"
