"""Pure ASGI exclusion for legacy ad-hoc compute in training maintenance."""

from argus.dashboard.api_response import api_conflict


class TrainingMaintenanceMiddleware:
    def __init__(self, app, *, enabled=False, compute_admission=None, maintenance_check=None,
                 require_compute_authority=False):
        self.app, self.enabled = app, enabled is True
        self.compute_admission = compute_admission
        self.maintenance_check = maintenance_check
        self.require_compute_authority = require_compute_authority

    @staticmethod
    def _executes_model(path):
        path = path.rstrip("/")
        if path in {
            "/api/models/batch-inference", "/api/models/ab-live-compare", "/api/models/backbone/upgrade",
            "/api/baseline/compare", "/api/baseline/job", "/api/baseline/capture",
        }:
            return True
        parts = path.strip("/").split("/")
        return len(parts) == 4 and parts[:2] == ["api", "models"] and parts[-1] in {"reexport", "recalibrate"}

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            path = scope.get("path", "")
            # These older workflows can construct a detector/trainer or load
            # auxiliary weights independently of managed task workers. Keep
            # new task data/approval/queue and camera source operations usable.
            if any(path == prefix or path.startswith(prefix + "/") for prefix in (
                "/api/models", "/api/baseline", "/api/config",
            )):
                maintenance = self.enabled
                if not maintenance and self.maintenance_check is not None:
                    try:
                        maintenance = self.maintenance_check() is not False
                    except Exception:
                        maintenance = True
                if maintenance:
                    response = api_conflict("独占训练维护模式禁止旧模型计算或配置热更新；请以普通配置重启后操作")
                elif self.require_compute_authority and self.compute_admission is None and self._executes_model(path):
                    response = api_conflict("模型计算缺少已装配的资源所有权，不能绕过主机计算互斥")
                elif self.compute_admission is not None and self.compute_admission() is not True:
                    response = api_conflict("另一 Argus 实例持有计算资源；当前实例不能加载或执行模型")
                else:
                    response = None
                if response is not None:
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)
