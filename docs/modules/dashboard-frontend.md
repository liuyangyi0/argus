# Dashboard 与前端

> 关键源码：`src/argus/dashboard/`、`web/src/router/index.ts`、`web/src/api/`、`web/src/views/`、`web/src/components/`

Dashboard 层由 FastAPI 后端和 Vue 3 前端组成。后端是 API/WebSocket/静态资源门面，核心业务状态主要来自 `app.state` 注入对象；前端消费 `/api/*` 和 `/ws`，不直接访问本地文件或后端内部对象。

## FastAPI 应用

`create_app()` 负责：

- 创建 `ConnectionManager`。
- 创建或复用 go2rtc、stream registry、camera orchestrator。
- 注入 `app.state`。
- 配置纯 ASGI 安全头、限流、认证中间件。
- 注册 `/api/*` routers。
- 注册 `/ws`。
- 如果 `web/dist/index.html` 存在，托管 Vue SPA 和 assets。

中间件必须保持纯 ASGI，不能使用 Starlette `BaseHTTPMiddleware`。

## 认证与权限

认证模块位于 `src/argus/dashboard/auth.py`：

- 支持密码 hash、session token、token 验证。
- `AuthMiddleware` 将认证信息写入 request scope/state。
- `require_role()` 和 `require_permission()` 供 route 使用。
- `RateLimitMiddleware` 限制请求速率。
- `SecurityHeadersMiddleware` 添加安全响应头。

前端使用 `/api/me` 获取当前身份。401 会触发整页跳转到后端 `/login`，因为当前没有 SPA login route。

## API 响应约定

`api_response.py` 提供统一 envelope：

```json
{
  "code": 0,
  "msg": "ok",
  "data": {}
}
```

前端 `web/src/api/client.ts` 的 `unwrap()` 会解包成功响应，非 0 code 或 HTTP 错误会转换为 `ApiError`。调用侧应使用 `utils/error.ts` 提取错误消息，避免重复 toast。

## WebSocket

WebSocket 入口是 `/ws`，鉴权 token 通过 query param 传入。`ConnectionManager` 使用 janus 队列桥接同步线程和 async WebSocket。

常见主题：

- `health`
- `cameras`
- `alerts`
- `tasks`
- `wall`
- `degradation`
- `heatmap`
- `audio_alert`
- `models`
- `model_release`
- `system_degradation`
- `system_errors`

`degradation` 偏全局操作员事件，`system_degradation` 偏每路 pipeline 的 anomaly fallback 状态。

`alerts` 主题的新告警 payload 会携带 P1 诊断字段：`detection_type`、`evidence_summary`、`evidence_json`、`suppression_reason` 和 `model_version_id`。这些字段用于实时告警列表和详情面板展示，前端仍按可选字段处理，以兼容旧数据库行和旧 replay 证据。

## HTTP API 分组

route 文件和前缀：

| 前缀 | 文件 | 职责 |
| --- | --- | --- |
| `/api/cameras` | `routes/cameras.py` | 摄像头配置、状态、模式、快照、MJPEG、视频墙 |
| `/api/alerts` | `routes/alerts.py` | 告警列表、详情、工作流、批量操作、导出、反馈 |
| `/api/zones` | `routes/zones.py` | 摄像头区域 CRUD 和区域快照 |
| `/api/config` | `routes/config.py` | 检测参数、模块开关、分类器/分割器/跨摄像头/音频配置 |
| `/api/system` | `routes/system.py` | 健康、漂移、ensemble、Prometheus、fallback 状态 |
| `/api/tasks` | `routes/tasks.py` | 前端可见长任务 |
| `/api/audit` | `routes/audit.py` | 审计日志 |
| `/api/backup` | `routes/backup.py` | 备份、恢复、删除 |
| `/api/users` | `routes/users.py` | 用户管理 |
| `/api/me` | `routes/me.py` | 当前用户 |
| `/api/regions` | `routes/regions.py` | 区域/责任人管理 |
| `/api/reports` | `routes/reports.py` | 报表与合规导出 |
| `/api/models` | `routes/models.py` | 模型注册、发布、A/B、阈值、backbone |
| `/api/replay` | `routes/replay.py` | replay metadata、video、frames、clips、storyboard |
| `/api/degradation` | `routes/degradation.py` | 全局降级事件 |
| `/api/training-jobs` | `routes/training_jobs.py` | 训练作业 |
| `/api/streaming` | `routes/streaming.py` | go2rtc stream 信息与注册 |
| `/api/labeling` | `routes/labeling.py` | 标注队列 |
| `/api/recordings` | `routes/recordings.py` | 持续录像片段 |
| `/api/calibration` | `routes/calibration.py` | 摄像头标定 |
| `/api/physics` | `routes/physics.py` | 轨迹、速度、定位 |
| `/api/sensors` | `routes/sensors.py` | 外部传感器融合信号 |
| `/api/baseline` | `routes/baseline.py` | 基线集合、版本、采集任务 |

完整端点表见 `docs/api-reference.md` 或自动生成的 `docs/source-index.md`。

## 告警证据展示

告警页和详情面板负责把后端证据字段翻译成操作员可读信息：

- `web/src/types/api.ts` 的告警类型包含 `evidence_summary`、`evidence_json` 和 `suppression_reason`，所有字段都是可选/可空。
- `AlertDetailPanel` 展示检测类型、模型版本、frame index、证据摘要和抑制原因；当 replay metadata 存在时，也展示 `candidate_evidence` 中的候选事件、候选 bbox、候选分数和 suppression reason。
- 摄像头详情和 replay signal tracks 使用 `candidate_records` / `candidate_events` 解释未触发最终告警的候选，不把这些候选误显示为已入库告警。

## 前端路由

当前主路由和重要子路由：

- `/overview`
- `/cameras`
- `/cameras/:id`
- `/alerts`
- `/reports`
- `/warehouse3D`
- `/models/baseline`
- `/models/training`
- `/models/registry`
- `/models/comparison`
- `/models/labeling`
- `/models/threshold`
- `/models/collections`
- `/system/overview`
- `/system/model-status`
- `/system/config`
- `/system/audit`
- `/system/degradation`
- `/system/modules`
- `/system/classifier`
- `/system/segmenter`
- `/system/imaging`
- `/system/cross-camera`
- `/system/users`
- `/system/regions`
- `/replay/:alertId`
- `/replay/:alertId/storyboard`

`/training` 只重定向到 `/models/training`。旧 `?tab=` query 会映射到新的 child route。

## 前端代码组织

- `web/src/api/*`：按业务域封装 HTTP 调用。
- `web/src/stores/*`：Pinia 状态，如 auth、alert、baseline、error、wall。
- `web/src/composables/*`：WebSocket、go2rtc、回放控制、模型状态、系统模式等组合式逻辑。
- `web/src/services/streaming/*`：播放器工厂和 stream manager。
- `web/src/views/models/*`：模型页子路由。
- `web/src/views/system/*`：系统页子路由。
- `web/src/components/models/*`、`components/system/*`、`components/alerts/*`、`components/replay/*`：高复用业务组件。

## 修改建议

- 新增后端 route 时同步 `web/src/api/` 封装和 `docs/api-reference.md`。
- 新增前端页面时优先使用现有 nested route 模式，不要把系统/模型子功能拆成无必要主路由。
- WebSocket 新主题需要同时补后端 broadcaster、前端 composable/store、降级/错误展示策略和测试。
- 告警证据字段要同步后端 route、`web/src/types/api.ts`、告警列表、`AlertDetailPanel` 和 replay 组件，保持旧数据可空兼容。
- 流媒体相关 UI 应优先使用 go2rtc 信息，只有不可用时回退 MJPEG。

## 相关测试

- `tests/integration/test_dashboard_api.py`
- `tests/unit/test_dashboard.py`
- `tests/unit/test_dashboard_streaming_contract.py`
- `tests/unit/test_websocket.py`
- `tests/unit/test_permissions.py`
- `web/src/router/__tests__/routes.test.ts`
- `web/src/layouts/__tests__/BasicLayoutNavigation.test.ts`
- `web/src/views/__tests__/*.test.ts`
- `web/src/components/*/__tests__/*.test.ts`
