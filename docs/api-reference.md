# Argus API 参考

> 最后更新：2026-06-12
> 自动端点清单来源：`docs/source-index.md`。后端装配入口：`src/argus/dashboard/app.py`。

本文档按业务域描述 Dashboard HTTP API 和 WebSocket。精确函数签名、源码行号和完整路由列表可从 `docs/source-index.md` 重新生成。

## 基础约定

- HTTP API 前缀：`/api`。
- WebSocket：`/ws?token=<session-token-or-api-token>`。
- 生产构建时 Vue SPA 由 FastAPI 托管；非 API 路径会 fallback 到 `web/dist/index.html`。
- 大多数 JSON API 使用统一 envelope：

```json
{
  "code": 0,
  "msg": "ok",
  "data": {}
}
```

流式响应、文件下载、图片、视频和 Prometheus metrics 不一定使用 envelope。

## 认证

- `/login`、`/logout` 由 `auth_router` 提供，服务端页面/表单方式处理。
- `/api/me` 返回当前用户信息。
- 401：前端会整页跳转 `/login?next=...`。
- 403：前端合并 1 秒内重复 toast。
- 敏感 API 使用角色或 permission 校验，典型角色为 admin、engineer、operator、viewer。

## WebSocket

连接入口：

```text
GET /ws?token=<token>
```

常见主题：

| 主题 | 含义 |
| --- | --- |
| `health` | 系统健康和运行时间 |
| `cameras` | 摄像头列表/单路状态变化 |
| `alerts` | 新告警和告警变更 |
| `tasks` | 长任务状态 |
| `wall` | 视频墙聚合状态 |
| `degradation` | 全局降级事件 |
| `system_degradation` | 每路 anomaly fallback/降级状态 |
| `system_errors` | ErrorChannel 系统错误 |
| `models` | 模型列表/状态变化 |
| `model_release` | 模型发布阶段变化 |
| `heatmap` | 热力图相关实时消息 |
| `audio_alert` | 音频告警触发 |

`alerts` 主题的新告警消息与告警 API 使用同一组 P1 诊断字段：`detection_type`、`evidence_summary`、`evidence_json`、`suppression_reason` 和 `model_version_id`。旧告警可能没有这些字段，客户端必须按可空字段处理。

## Cameras

前缀：`/api/cameras`

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/json` | 摄像头列表、运行状态和摘要 |
| `POST` | `` | 新增摄像头 |
| `GET` | `/{camera_id}/config` | 获取单路配置 |
| `PUT` | `/{camera_id}` | 更新单路配置 |
| `DELETE` | `/{camera_id}` | 删除摄像头 |
| `POST` | `/{camera_id}/start` | 启动单路 |
| `POST` | `/{camera_id}/stop` | 停止单路 |
| `POST` | `/{camera_id}/mode` | 切换 pipeline mode |
| `POST` | `/{camera_id}/test-connection` | 测试已保存摄像头连接 |
| `POST` | `/test-connection-draft` | 测试未保存草稿连接 |
| `GET` | `/usb-devices` | 枚举 USB 摄像头 |
| `GET` | `/{camera_id}/detail/json` | 单路详情、诊断和播放信息 |
| `GET` | `/{camera_id}/runner` | runner snapshot |
| `GET` | `/{camera_id}/snapshot` | JPEG 快照 |
| `GET` | `/{camera_id}/stream` | MJPEG 回退流 |
| `GET` | `/{camera_id}/heatmap-stream` | 热力图 MJPEG 流 |
| `GET` | `/wall/status` | 视频墙状态 |

## Streaming

前缀：`/api/streaming`

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `` | 列出 stream |
| `GET` | `/{camera_id}` | 获取单路 go2rtc/预览信息 |
| `POST` | `/{camera_id}/register` | 注册或刷新单路 stream |

## Alerts

前缀：`/api/alerts`

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/json` | 告警列表，支持分页/过滤 |
| `GET` | `/{alert_id}/detail` | 告警详情 |
| `GET` | `/{alert_id}/image/{image_type}` | 快照或热力图 |
| `GET` | `/{alert_id}/evidence.zip` | 证据包下载 |
| `POST` | `/{alert_id}/workflow` | 工作流状态转换 |
| `POST` | `/{alert_id}/acknowledge` | 确认收到 |
| `POST` | `/{alert_id}/confirm-anomaly` | 确认为真实异常 |
| `POST` | `/{alert_id}/false-positive` | 标记误报 |
| `DELETE` | `/{alert_id}` | 删除单条告警 |
| `POST` | `/{alert_id}/annotations` | 保存告警标注 |
| `POST` | `/bulk-acknowledge` | 批量确认 |
| `POST` | `/bulk-false-positive` | 批量标记误报 |
| `POST` | `/bulk-delete` | 批量删除 |
| `GET` | `/group/{event_group_id}` | 告警事件组 |
| `GET` | `/feedback-stats` | 反馈统计 |
| `GET` | `/export-csv` | CSV 导出 |
| `GET` | `/export-pdf` | PDF 导出 |

告警列表、详情和实时告警 payload 的核心诊断字段：

| 字段 | 含义 |
| --- | --- |
| `detection_type` | 触发来源，如 `anomaly`、`projectile`、`persistent_change`、`water_leak`、`plume` |
| `evidence_summary` | `DetectionEvidence.evidence_summary()` 生成的一行证据摘要 |
| `evidence_json` | `DetectionEvidence.to_dict()` 的 JSON 对象，包含 score、zone、bbox、候选、特征和 `debug_trace_id` 等 |
| `suppression_reason` | 抑制原因；最终已触发告警通常为空，候选诊断和旧数据可能有值 |
| `model_version_id` | 触发该告警的模型版本 lineage |
| `candidate_evidence` | 告警详情聚合 replay metadata 中的候选事件和每帧候选记录 |

## Replay

前缀：`/api/replay`

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/{alert_id}/metadata` | Replay 元数据 |
| `GET` | `/{alert_id}/signals` | 分数、检测、轨迹等时间线 |
| `GET` | `/{alert_id}/video` | 告警录像 |
| `GET` | `/{alert_id}/frame/{index}` | 单帧 JPEG |
| `GET` | `/{alert_id}/heatmap/{index}` | 单帧热力图 |
| `GET` | `/{alert_id}/reference` | 历史参考帧 |
| `POST` | `/{alert_id}/pin-frame` | 固定关键帧 |
| `POST` | `/{alert_id}/clips` | 新增片段 |
| `DELETE` | `/{alert_id}/clips/{index}` | 删除片段 |
| `GET` | `/storyboard/{alert_id}` | 多机位 story board |

`GET /api/replay/{alert_id}/signals` 返回的 signal timeline 包含 `candidate_records` 和 `candidate_events`。它们来自 `AlertFrameBuffer` 固化帧，用于解释非最终告警候选的 `decision`、`evidence` 和 `suppression_reason`。

## Baseline

前缀：`/api/baseline`

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/collections/json` | 基线集合 |
| `POST` | `/collections/{camera_id}/{zone_id}/{version}/activate` | 激活集合 |
| `POST` | `/collections/{camera_id}/{zone_id}/{version}/retire` | 退役集合 |
| `DELETE` | `/collections/{camera_id}/{zone_id}/{version}` | 删除集合 |
| `GET` | `/list/json` | 旧版基线列表 |
| `GET` | `/models/json` | 旧版模型列表 |
| `GET` | `/training-history/json` | 训练历史 |
| `GET` | `/training-history/{record_id}/metrics` | 训练指标 |
| `POST` | `/job` | 启动基线采集任务 |
| `POST` | `/job/{task_id}/pause` | 暂停采集 |
| `POST` | `/job/{task_id}/resume` | 恢复采集 |
| `POST` | `/job/{task_id}/abort` | 终止采集 |
| `GET` | `/job/{task_id}` | 采集任务状态 |
| `POST` | `/compare` | 模型/基线比较 |
| `GET` | `/optimize/preview` | 基线优化预览 |
| `POST` | `/optimize/json` | 执行基线优化 |
| `GET` | `/groups/json` | 摄像头组 |
| `POST` | `/groups/merge` | 组基线合并 |
| `POST` | `/merge-fp` | 合并误报样本 |
| `GET` | `/versions/json` | 基线版本 |
| `POST` | `/verify` | 验证基线 |
| `POST` | `/activate-baseline` | 激活基线 |
| `POST` | `/retire` | 退役基线 |
| `GET` | `/{camera_id}/{version}/images` | 版本图片列表 |
| `GET` | `/{camera_id}/{version}/images/{filename}/content` | 版本图片内容 |
| `DELETE` | `/{camera_id}/{version}/images/{filename}` | 删除版本图片 |
| `POST` | `/{camera_id}/{version}/images` | 上传版本图片 |

## Training Jobs

前缀：`/api/training-jobs`

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/json` | 训练作业列表 |
| `POST` | `/` | 创建训练作业 |
| `GET` | `/{job_id}` | 作业详情 |
| `GET` | `/{job_id}/source-alerts` | 训练来源告警 |
| `POST` | `/{job_id}/confirm` | 确认进入队列 |
| `POST` | `/{job_id}/reject` | 拒绝作业 |
| `GET` | `/backbones/json` | backbone 列表 |

## Models

前缀：`/api/models`

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/status` | 模型运行状态 |
| `GET` | `/json` | 模型注册表 |
| `POST` | `/{version_id}/activate` | 兼容激活入口 |
| `POST` | `/{version_id}/rollback` | 回滚 |
| `POST` | `/{version_id}/promote` | stage promote |
| `POST` | `/{version_id}/retire` | 退役 |
| `DELETE` | `/{version_id}` | 删除 |
| `POST` | `/{version_id}/reexport` | 重新导出 |
| `POST` | `/{version_id}/recalibrate` | 重新校准 |
| `GET` | `/{version_id}/stage-history` | 阶段历史 |
| `GET` | `/events/list` | 模型事件 |
| `GET` | `/{version_id}/shadow-report` | shadow 报告 |
| `GET` | `/{version_id}/ab-scores` | A/B 分数 |
| `GET` | `/{version_id}/ab-distribution` | A/B 分布 |
| `POST` | `/ab-live-compare` | 实时 A/B 对比 |
| `POST` | `/batch-inference` | 批量推理 |
| `GET` | `/backbone/status` | backbone 状态 |
| `POST` | `/backbone/upgrade` | backbone 升级 |
| `GET` | `/threshold-preview` | 阈值预览 |

## Config

前缀：`/api/config`

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/detection-params` | 更新检测参数 |
| `POST` | `/notifications` | 更新通知 |
| `POST` | `/test-webhook` | 测试 webhook |
| `GET` | `/storage/info` | 存储信息 |
| `POST` | `/cleanup` | 数据清理 |
| `POST` | `/thresholds` | 更新阈值 |
| `POST` | `/reload-model` | 重载模型 |
| `POST` | `/clear-lock/{camera_id}` | 清除异常锁 |
| `POST` | `/camera/{camera_id}/restart` | 重启摄像头 |
| `POST` | `/reload` | 重新加载配置 |
| `POST` | `/save` | 保存配置 |
| `GET` | `/audio-alerts` | 音频告警配置 |
| `PUT` | `/audio-alerts` | 更新音频告警 |
| `GET` | `/modules` | 模块状态 |
| `POST` | `/modules` | 切换模块 |
| `GET` | `/classifier` | 分类器配置 |
| `PUT` | `/classifier/vocabulary` | 更新分类器词表 |
| `GET` | `/segmenter` | 分割器配置 |
| `PUT` | `/segmenter/params` | 更新分割器参数 |
| `GET` | `/imaging` | 多模态成像配置 |
| `GET` | `/cross-camera` | 跨摄像头配置 |
| `PUT` | `/cross-camera/pairs` | 更新 overlap pairs |

## System / Ops

| 前缀 | 关键端点 | 用途 |
| --- | --- | --- |
| `/api/system` | `/health`、`/camera-health`、`/drift`、`/ensemble`、`/metrics`、`/anomaly-degradation` | 系统健康、指标和 fallback 状态 |
| `/api/tasks` | `/json`、`/{task_id}` DELETE | 长任务状态和清除 |
| `/api/degradation` | `/active`、`/history`、`/summary` | 全局降级事件 |
| `/api/audit` | `/json` | 审计日志 |
| `/api/backup` | `/list/json`、`/create`、`/restore`、`/{backup_name}` DELETE | 备份管理 |
| `/api/users` | `/json`、`/{username}/json`、`/{username}/toggle-active/json` | 用户管理 |
| `/api/regions` | `/json`、`/{region_id}/json` | 区域/责任人管理 |

## Reports / Physics / Sensors / Recordings / Zones / Labeling

| 前缀 | 关键端点 | 用途 |
| --- | --- | --- |
| `/api/reports` | `/json`、`/daily-trend/json`、`/severity-dist/json`、`/camera-dist/json`、`/fp-trend/json`、`/compliance` | 报表和合规导出 |
| `/api/physics` | `/{alert_id}/trajectory`、`/{camera_id}/active-tracks`、`/{alert_id}/localization` | 轨迹、速度和定位 |
| `/api/sensors` | `/signal`、`/signals` | 外部传感器信号 |
| `/api/recordings` | `/{camera_id}/segments`、`/{camera_id}/segment/{filename}/video`、`/storage-stats`、`/cleanup` | 持续录像片段 |
| `/api/zones` | `` POST、`/{camera_id}` PUT、`/{camera_id}/{zone_id}` DELETE、`/snapshot/{camera_id}` | 摄像头区域 |
| `/api/labeling` | `/queue`、`/{entry_id}/label`、`/{entry_id}/skip`、`/stats`、`/trigger-retrain`、`/{entry_id}/image` | 标注队列 |
| `/api/calibration` | `/{camera_id}`、`/{camera_id}/upload`、`/{camera_id}/compute`、`/{camera_id}/verify` | 摄像头标定 |

## 维护流程

新增或修改 API 时：

1. 更新 route 实现和测试。
2. 更新 `web/src/api/*.ts`。
3. 更新前端调用点和类型。
4. 运行 `python scripts/extract_source_index.py --output docs/source-index.md`。
5. 更新本文档的业务说明。
