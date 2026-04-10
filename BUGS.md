# Argus Bug 清单 — 冒烟测试 2026-04-10

## 测试环境
- Windows 11 Pro, Python 3.11.9, Node 22+
- USB 摄像头已连接 (camera_id: c)
- 后端运行正常，go2rtc 正常启动

---

## 已修复

### [已修复] P1 — 前端构建失败：33 个 TypeScript 编译错误
- 未使用的导入/变量 (18处)
- 类型不匹配 (12处)
- enum 语法不兼容 erasableSyntaxOnly (1处)
- **提交:** 3ff48fa

### [已修复] P2 — test_config 断言过时
- `test_load_default_config_file` 断言 `camera_id == "cam_01"` 但实际值已改为 `"c"`
- **提交:** 1f27ef0

---

## 待修复

### P2 — 前端 bundle 过大
- `api-GPQ5V0N3.js` 达 1.46MB (gzip 448KB)
- `dist-CBppCSgD.js` 达 572KB (gzip 191KB)
- 建议：动态 import() 拆分路由级组件，懒加载 ECharts

### P3 — Prometheus 指标全零
- `argus_inference_latency_seconds` 所有 bucket 为 0
- 可能原因：metrics 模块已注册但 pipeline 中未调用 observe()
- 需要确认 metrics.py 是否已集成到 pipeline.py

### P3 — 训练任务状态 "failed"
- 唯一的训练任务 `8b2dff72-8c3` 状态为 failed
- 需检查失败原因，确认训练流程是否正常

### P3 — 告警时间线返回空 cameras
- `/api/alerts/timeline` 返回 `"cameras": []`
- 有 21 条告警但时间线无数据，可能是日期过滤问题

---

## 冒烟测试通过项

### 后端 API（全部通过）
- [x] `/api/system/health` — 200, 返回正确状态
- [x] `/api/cameras/json` — 摄像头列表正常，stats 数据完整
- [x] `/api/cameras/c/detail/json` — 详情正常，stages 完整
- [x] `/api/cameras/c/snapshot` — 返回 image/jpeg
- [x] `/api/alerts/json` — 告警列表正常，21 条告警
- [x] `/api/alerts/export-csv` — 200
- [x] `/api/replay/{alert_id}/metadata` — 回放元数据正常
- [x] `/api/models/json` — 模型列表正常
- [x] `/api/models/backbone/status` — 骨干网络状态正常
- [x] `/api/models/threshold-preview` — 阈值预览正常
- [x] `/api/training-jobs/json` — 训练任务列表正常
- [x] `/api/baseline/list/json` — 基线列表正常
- [x] `/api/degradation/active` — 退化状态正常（空）
- [x] `/api/users/json` — 用户列表正常
- [x] `/api/audit/json` — 审计日志正常
- [x] `/api/labeling/queue` — 标注队列正常
- [x] `/api/labeling/stats` — 标注统计正常
- [x] `/api/tasks/json` — 任务列表正常
- [x] `/api/streaming` — go2rtc 流状态正常
- [x] `/api/system/metrics` — Prometheus 指标端点正常
- [x] `/api/config/audio-alerts` — 音频告警配置正常
- [x] `/api/reports/json` — 报表数据正常
- [x] 404 错误处理正确（非存在摄像头/告警返回 code 40400）

### 后端测试
- [x] 1027 passed / 0 failed（修复 test_config 后）

### 前端构建
- [x] `vue-tsc -b` 零错误
- [x] `vite build` 成功

---

## 待测试（需要浏览器）

以下功能需要浏览器交互验证，本次因 Chrome 扩展未连接而跳过：

- [ ] WebSocket 实时推送（告警、健康状态、任务进度）
- [ ] 摄像头实时流播放（go2rtc WebRTC/MSE）
- [ ] 告警回放播放器（视频 + 热力图）
- [ ] 区域编辑器交互（多边形绘制/删除）
- [ ] 标注覆盖层（画布标注）
- [ ] 图片对比滑块
- [ ] 训练任务创建完整流程
- [ ] 基线采集完整流程
- [ ] 亮色/暗色主题切换
- [ ] 各表格分页/筛选/排序
- [ ] 表单验证（摄像头添加、用户创建等）
