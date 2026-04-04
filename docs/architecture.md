# Argus 系统架构文档

> 版本 0.1.0 | 最后更新: 2026-04-04

## 1. 项目简介

Argus（阿耳戈斯）是面向核电站关键区域的异物检测（FOE）视觉系统。通过固定摄像头 + 深度学习异常检测，实现 7x24 小时不间断监控，自动发现并分级报警墙皮脱落、保温层掉落、遗留工具等异常物体。

**部署模式：** 边缘节点（Intel NUC / NVIDIA Jetson），每节点 2-4 路摄像头，核电站内网环境。

---

## 2. 需求概览

### 2.1 功能需求（FR-001 ~ FR-024）

| 编号 | 功能 | 状态 | 所在模块 |
|------|------|------|----------|
| FR-001 | 摄像头采集（RTSP/USB/文件，断线重连） | ✅ | `capture/camera.py` |
| FR-002 | MOG2 背景减除（辐射去噪、相位相关防抖） | ✅ | `prefilter/mog2.py` |
| FR-003 | YOLO 人员过滤（遮罩/跳帧，优雅降级） | ✅ | `person/detector.py` |
| FR-004 | Anomalib 异常检测（PatchCore/EfficientAD，SSIM 回退，模型热更新） | ✅ | `anomaly/detector.py` |
| FR-005 | 防背景吸收（心跳全帧检测 + 异常状态锁定） | ✅ | `core/pipeline.py` |
| FR-006 | 检测区域（Include Zone，多边形，多区域独立评估） | ✅ | `core/zone_mask.py` |
| FR-007 | 排除区域（Exclude Zone，优先级高于检测区域） | ✅ | `core/zone_mask.py` |
| FR-008 | 区域热更新（增删不需重启） | ✅ | `core/pipeline.py` |
| FR-009 | 四级报警分级（INFO/LOW/MEDIUM/HIGH） | ✅ | `alerts/grader.py` |
| FR-010 | 时间窗滤波（连续 N 帧确认 + 空间 IoU 连续性） | ✅ | `alerts/grader.py` |
| FR-011 | 重复抑制（同区域/同摄像头时间窗） | ✅ | `alerts/grader.py` |
| FR-012 | 报警分发（数据库 + Webhook + 邮件） | ✅ | `alerts/dispatcher.py` |
| FR-013 | 报警反馈（确认 + 标记误报） | ✅ | `alerts/feedback.py` |
| FR-014 | 多摄像头线程池管理 | ✅ | `capture/manager.py` |
| FR-015 | Dashboard 系统概览 | ✅ | `dashboard/routes/system.py` |
| FR-016 | 摄像头管理页面（添加/启停/统计） | ✅ | `dashboard/routes/cameras.py` |
| FR-017 | 告警中心（筛选/批量操作/CSV导出） | ✅ | `dashboard/routes/alerts.py` |
| FR-018 | 区域编辑器（Canvas 多边形绘制） | ✅ | `dashboard/routes/zones.py` + `static/js/zone_editor.js` |
| FR-019 | 系统设置（检测参数/通知/维护/日志） | ✅ | `dashboard/routes/config.py` |
| FR-020 | JSON API（health/alerts/cameras） | ✅ | `dashboard/routes/` |
| FR-021 | 基线管理（版本化存储/网页采集） | ✅ | `anomaly/baseline.py` + `dashboard/routes/baseline.py` |
| FR-022 | 模型训练（网页触发/后台执行） | ✅ | `anomaly/trainer.py` + `dashboard/routes/baseline.py` |
| FR-023 | 误报反馈闭环 | ✅ | `alerts/feedback.py` |
| FR-024 | Docker 容器化部署 | ✅ | `Dockerfile` + `docker-compose.yml` |

### 2.2 非功能需求

| 编号 | 需求 | 实现方式 |
|------|------|----------|
| NFR-001 | 帧到报警 <500ms | MOG2 跳帧节约 80% 算力，单帧检测 ~30ms（无多尺度）/ ~270ms（多尺度） |
| NFR-002 | 7x24 不间断运行 | 断线自动重连、优雅降级、进程守护（Docker restart: unless-stopped） |
| NFR-003 | 误报率 <1 次/天/摄像头 | 时间窗滤波 + 空间 IoU + 区域掩码 + 辐射去噪 + 人员过滤 |
| NFR-004 | 安全 | Docker 非 root、API 认证、路径遍历防护、限流、安全头 |
| NFR-005 | 可维护性 | Pydantic 类型安全、structlog 结构化日志、123 单元测试 |

### 2.3 未来扩展（未实现）

| 功能 | 说明 |
|------|------|
| Modbus/OPC-UA | 对接电厂 DCS 系统 |
| 多节点聚合 | PostgreSQL 中央数据库 |
| AnomalyDINO | 零样本异常检测 |
| WebSocket | 替代 HTMX 轮询的实时推送 |
| Prometheus | 可观测性指标导出 |

---

## 3. 系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Argus 边缘节点                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ 摄像头 1  │   │ 摄像头 2  │   │ 摄像头 3  │   │ 摄像头 4  │        │
│  │ (RTSP)   │   │ (RTSP)   │   │ (USB)    │   │ (File)   │        │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘        │
│       │              │              │              │               │
│       ▼              ▼              ▼              ▼               │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              CameraManager（线程池）                      │       │
│  │  每个摄像头一个独立 DetectionPipeline 线程                │       │
│  └─────────────────────────┬───────────────────────────────┘       │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              DetectionPipeline（每摄像头）                │       │
│  │                                                         │       │
│  │  Stage 0: ZoneMaskEngine（区域掩码）                     │       │
│  │       ↓                                                 │       │
│  │  Stage 1: MOG2PreFilter（背景减除 + 心跳旁路）            │       │
│  │       ↓ (有变化 or 心跳 or 锁定)                         │       │
│  │  Stage 2: YOLOPersonDetector（人员遮罩/跳帧）            │       │
│  │       ↓ (无人 or 已遮罩)                                │       │
│  │  Stage 3: AnomalibDetector（异常检测 + SSIM 回退）       │       │
│  │       ↓                                                 │       │
│  │  AlertGrader（分级 + 时间确认 + 去重抑制）               │       │
│  └─────────────────────────┬───────────────────────────────┘       │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              AlertDispatcher（多通道分发）                │       │
│  │                                                         │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │       │
│  │  │ SQLite   │  │ Webhook  │  │  邮件    │              │       │
│  │  │ 数据库   │  │ (后台线程) │  │(后台线程) │              │       │
│  │  └──────────┘  └──────────┘  └──────────┘              │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │         FastAPI Dashboard（HTMX 服务端渲染）              │       │
│  │                                                         │       │
│  │  总览 | 摄像头 | 基线与模型 | 检测区域 | 告警中心 | 系统设置 │       │
│  │                                                         │       │
│  │  + MJPEG 实时视频流                                     │       │
│  │  + 后台任务管理（基线采集/模型训练）                      │       │
│  │  + API 认证 + 限流 + 安全头                             │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────┐              │
│  │TaskScheduler│  │HealthMonitor │  │ TaskManager   │              │
│  │(定时维护)    │  │(健康监控)     │  │(后台任务队列)  │              │
│  └────────────┘  └──────────────┘  └───────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 线程模型

```
主线程 ─── main loop（健康检查 1s 轮询）
  │
  ├── 摄像头线程 1 ─── DetectionPipeline.run_once() 循环
  ├── 摄像头线程 2 ─── DetectionPipeline.run_once() 循环
  ├── ...
  ├── Dashboard 线程 ─── Uvicorn (FastAPI)
  ├── Scheduler 线程 ─── APScheduler（定时清理/磁盘检查）
  ├── Webhook 线程 ─── 后台 HTTP POST 队列
  └── Email 线程 ─── 后台 SMTP 发送队列
```

### 3.3 数据流

```
摄像头帧 → 区域掩码 → MOG2 判断变化？
                         │
                    ┌────┤
                    │    │
               无变化    有变化 / 心跳 / 锁定
                │        │
              跳过帧     YOLO 人员检测
                         │
                    ┌────┤
                    │    │
               有人跳帧  无人/已遮罩
                │        │
              跳过帧     Anomalib 异常检测
                         │
                    ┌────┤
                    │    │
               正常      异常 (score > threshold)
                │        │
              更新锁定   AlertGrader 评估
                状态      │
                    ┌────┤
                    │    │
               未达确认   达到确认条件
               阈值       (3帧连续 + IoU > 0.3)
                │         │
              等待下帧    检查去重抑制
                          │
                    ┌─────┤
                    │     │
               抑制窗口内  窗口外
                │         │
              丢弃       生成 Alert → 分发
```

---

## 4. 包结构

```
src/argus/
├── __init__.py          # 版本号 0.1.0
├── __main__.py          # CLI 入口，子系统编排
│
├── config/              # 配置系统
│   ├── schema.py        # 15 个 Pydantic 模型（全参数验证）
│   └── loader.py        # YAML 加载/保存（原子写入）
│
├── capture/             # 视频采集
│   ├── camera.py        # 单摄像头驱动（RTSP/USB/文件，自动重连）
│   └── manager.py       # 多摄像头线程池
│
├── prefilter/           # 预筛
│   └── mog2.py          # MOG2 背景减除 + 相位相关防抖
│
├── person/              # 人员过滤
│   └── detector.py      # YOLO11n 人员检测（遮罩/跳帧）
│
├── anomaly/             # 异常检测
│   ├── detector.py      # Anomalib 推理 + 多尺度 + SSIM 回退
│   ├── baseline.py      # 基线版本管理
│   └── trainer.py       # 模型训练编排
│
├── core/                # 核心编排
│   ├── pipeline.py      # 三阶段检测管线（每摄像头一个）
│   ├── zone_mask.py     # 多边形区域掩码引擎
│   ├── health.py        # 系统健康监控
│   └── scheduler.py     # 定时维护任务（APScheduler）
│
├── alerts/              # 告警系统
│   ├── grader.py        # 分级 + 时间确认 + 去重抑制
│   ├── dispatcher.py    # 多通道分发（DB/Webhook/邮件）
│   └── feedback.py      # 误报反馈闭环
│
├── storage/             # 持久化
│   ├── database.py      # SQLite + WAL 告警存储
│   └── models.py        # ORM 模型（AlertRecord, BaselineRecord）
│
└── dashboard/           # Web UI
    ├── app.py           # FastAPI 应用工厂
    ├── auth.py          # 认证/限流/安全头中间件
    ├── components.py    # 可复用 HTML 组件（中文）
    ├── tasks.py         # 后台任务管理器（线程池）
    ├── routes/
    │   ├── system.py    # 总览 + 健康 API
    │   ├── cameras.py   # 摄像头管理（添加/启停/流）
    │   ├── baseline.py  # 基线采集 + 模型训练 + 部署
    │   ├── zones.py     # 检测区域编辑器
    │   ├── alerts.py    # 告警中心（筛选/批量/导出）
    │   ├── config.py    # 系统设置（Tab 化编辑）
    │   └── tasks.py     # 后台任务进度 API
    └── static/
        ├── css/argus.css     # 样式表（暗色主题）
        └── js/
            ├── zone_editor.js  # Canvas 多边形绘制
            └── toast.js        # Toast 通知
```

---

## 5. 关键配置参数

### 5.1 配置层级

```yaml
ArgusConfig                    # 顶层
├── node_id: str               # 节点标识
├── cameras: list[CameraConfig]
│   ├── camera_id, name, source, protocol, fps_target, resolution
│   ├── mog2: MOG2Config       # 背景减除参数
│   ├── person_filter: PersonFilterConfig
│   └── anomaly: AnomalyConfig # 异常检测参数
├── alerts: AlertConfig
│   ├── severity_thresholds    # 四级阈值（必须递增）
│   ├── temporal               # 时间窗确认
│   ├── suppression            # 去重抑制
│   ├── webhook                # Webhook 推送
│   └── email                  # 邮件告警
├── auth: AuthConfig           # API 认证
├── dashboard: DashboardConfig # Web UI 配置
├── storage: StorageConfig     # 存储路径和保留策略
├── models: ModelsConfig       # 模型文件路径
└── logging: LoggingConfig     # 日志轮转
```

### 5.2 核心参数默认值

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| MOG2 历史帧数 | 500 | 10-5000 | 背景建模所用帧数 |
| 变化检测阈值 | 0.5% | 0.01%-50% | 低于此比例视为无变化 |
| 心跳帧间隔 | 150 | 10-3000 | 强制全帧检测间隔 |
| 锁定触发分数 | 0.85 | 0.5-0.99 | 异常锁定触发阈值 |
| 锁定解除帧数 | 10 | 1-100 | 连续正常帧数解除锁定 |
| 确认连续帧数 | 3 | 1-30 | 告警触发所需连续帧 |
| 确认最大间隔 | 10s | 1-120s | 帧间最大时间间隔 |
| 空间重叠(IoU) | 0.3 | 0-1.0 | 连续帧异常区域重叠度 |
| 同区域抑制 | 300s | 10-3600s | 同区域告警去重窗口 |
| 告警保留 | 90 天 | 7-3650 天 | 自动清理周期 |
| 日志文件大小 | 50 MB | 1-500 MB | 单个日志文件上限 |

---

## 6. 代码统计

| 类别 | 文件数 | 代码行数 |
|------|--------|----------|
| 核心源码 (`src/argus/`) | 25 | ~4,600 |
| Dashboard 路由 | 7 | ~2,700 |
| Dashboard 静态资源 | 3 | ~540 |
| 单元测试 | 13 | ~1,700 |
| 工具脚本 | 4 | ~380 |
| 配置/文档 | 5 | ~350 |
| **合计** | **57** | **~10,270** |

测试覆盖：**123 个测试函数**，覆盖所有核心模块。

---

## 7. 技术栈

| 层 | 技术 | 版本 |
|---|------|------|
| 视频采集 | OpenCV | >= 4.10 |
| 人员检测 | Ultralytics YOLO11n | >= 8.3 |
| 异常检测 | Anomalib (PatchCore/EfficientAD) | >= 1.2 |
| 推理加速 | OpenVINO | >= 2024.0 |
| Web 框架 | FastAPI + HTMX | >= 0.115 |
| 前端渲染 | 服务端 HTML + HTMX 局部刷新 | 2.0.4 |
| 数据库 | SQLAlchemy + SQLite (WAL) | >= 2.0 |
| 配置 | Pydantic v2 + YAML | >= 2.0 |
| 日志 | structlog（JSON 文件 + 控制台） | >= 24.0 |
| 容器 | Docker + docker-compose | - |
| 语言 | Python | >= 3.11 |

---

## 8. 部署架构

```
docker-compose.yml
├── argus-edge (单容器)
│   ├── Port: 8080 (Dashboard)
│   ├── Volumes:
│   │   ├── ./data/baselines → /app/data/baselines
│   │   ├── ./data/models    → /app/data/models
│   │   ├── ./data/exports   → /app/data/exports
│   │   ├── ./data/db        → /app/data/db
│   │   ├── ./data/alerts    → /app/data/alerts
│   │   └── ./data/logs      → /app/data/logs
│   ├── Resources: 4 CPU, 4GB RAM
│   ├── Healthcheck: GET /api/system/health (30s interval)
│   └── Restart: unless-stopped
```

---

## 9. 完整工作流

```
1. 部署系统
   docker compose up -d
   浏览器打开 http://localhost:8080

2. 添加摄像头
   摄像头页面 → 添加摄像头 → 输入 RTSP 地址 → 启动

3. 配置检测区域
   检测区域页面 → 选择摄像头 → Canvas 绘制多边形 → 保存

4. 采集基线
   基线与模型页面 → 基线采集 Tab → 选择摄像头 → 设定帧数 → 开始采集

5. 训练模型
   基线与模型页面 → 模型训练 Tab → 选择摄像头+模型类型 → 开始训练

6. 部署模型
   基线与模型页面 → 模型管理 Tab → 选择模型 → 部署（热加载）

7. 实时监控
   总览页面 → 查看系统状态/摄像头状态/最近告警

8. 告警处理
   告警中心 → 筛选/查看详情 → 确认 or 标记误报

9. 误报反馈
   标记误报的快照自动进入基线目录 → 重新训练模型改进精度

10. 系统维护
    系统设置 → 调整参数/清理旧数据/查看日志/保存配置
```
