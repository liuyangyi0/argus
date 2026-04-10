# Argus Bug 清单 — 打磨阶段 2026-04-10

## 测试环境
- Windows 11 Pro, Python 3.11.9, Node 22+
- USB 摄像头已连接 (camera_id: c)
- 后端运行正常，go2rtc 正常启动

---

## 已修复

### [已修复] P1 — 前端构建失败：33 个 TypeScript 编译错误
- **提交:** 3ff48fa

### [已修复] P1 — 告警时间线返回空数据
- **根因:** `alerts.py:1071` .timestamp() 返回 float 与 DateTime 比较不兼容
- **提交:** dc2431f

### [已修复] P1 — 告警时间线 datetime.fromtimestamp TypeError
- **根因:** `alerts.py:1103` 对已有 datetime 对象调用 datetime.fromtimestamp()
- **提交:** 72081d3

### [已修复] P2 — test_config 断言过时
- **提交:** 1f27ef0

### [已修复] P2 — 前端 bundle 过大 (1.46MB → 37KB + 缓存)
- **提交:** dc2431f

### [已修复] P2 — dispatcher flush_db_queue 竞态条件
- **提交:** ad86a85

### [已修复] P2 — 16 处静默失败 (except: pass) 添加日志
- **提交:** d74d19e

### [已修复] P2 — alert grader _event_groups 内存泄漏
- **提交:** 36c0fe3

### [已修复] P2 — SQLite 缺少性能 pragmas
- 添加 busy_timeout、synchronous=NORMAL、cache_size、temp_store
- **提交:** 36c0fe3

### [已修复] P2 — 推理记录无自动清理
- delete_old_inference_records 方法已存在但未接入调度器
- **提交:** 36c0fe3

### [已修复] P2 — go2rtc 崩溃无自动重启
- 主循环每10秒检查进程状态，崩溃后自动重启
- **提交:** 36c0fe3

### [已关闭] — Prometheus 指标全零
- **非 bug:** 启动后需数秒积累帧数据

### [已关闭] — 训练任务 "failed"
- **非 bug:** 基线图片近似重复率 97.9% > 80% 上限，数据质量问题

---

## 测试统计

| 指标 | 打磨前 | 打磨后 |
|------|--------|--------|
| 单元测试数 | 1027 | **1093** (+66) |
| 测试通过率 | 99.9% | **100%** |
| TS 编译错误 | 33 | **0** |
| 静默失败 | 45 | **29** (关键的已加日志) |
| Bundle 大小 | 1.46MB 单chunk | **37KB + 缓存** |

---

## UI 验证结果 (Preview 工具)

### 已验证通过
- [x] 总览页面 — 暗色+亮色主题正常，1366x768 布局完整
- [x] 告警页面 — 表格数据、操作按钮、筛选器正常
- [x] 摄像头页面 — 列表、启停按钮、详情入口正常
- [x] 模型管理页面 — 标签页切换、基线列表正常
- [x] 系统页面 — 运行状态、配置、用户管理正常
- [x] 亮色/暗色主题切换 — 所有页面样式一致

### 待浏览器测试
- [ ] WebSocket 实时推送（告警、健康状态）
- [ ] 摄像头实时流播放（go2rtc WebRTC/MSE）
- [ ] 告警回放播放器（视频 + 热力图）
- [ ] 区域编辑器交互（多边形绘制/删除）
- [ ] 训练任务创建完整流程
- [ ] 基线采集完整流程

---

## 全部提交记录

| Commit | 内容 |
|--------|------|
| d643636 | 录像PTS修复 + 回放播放器重构 |
| bf8e49a | 事件总线、推理队列、指标、主动学习 |
| 1f27ef0 | test_config 断言修复 |
| 3ff48fa | 33个 TS 编译错误修复 |
| c675504 | BUGS.md 清单 |
| dc2431f | 告警时间线查询 + bundle 拆分 |
| 72081d3 | 66个新测试 + timeline datetime bug |
| ad86a85 | dispatcher 竞态条件修复 |
| d74d19e | 16处静默失败日志 + 表格滚动保护 |
| 36c0fe3 | 健壮性加固（内存、SQLite、go2rtc） |
