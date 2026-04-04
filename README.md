# Argus — 核电站异物视觉检测系统

Nuclear Power Plant Foreign Object Exclusion (FOE) Visual Detection System

## 项目背景

核电站关键区域（反应堆厂房、安全壳内、管廊等）严禁出现异物（墙皮脱落、保温层掉落、遗留工具、杂物等）。本系统通过固定摄像头实时监控，利用深度学习异常检测算法自动发现异常并分级报警。

## 系统架构

```
摄像头(RTSP/USB) → 帧队列
    │
    ▼
[Stage 0] 区域掩码 ── exclude 区域像素置零
    │
    ▼
[Stage 1] MOG2 预筛 ── 无变化 → 跳过（心跳每30s强制全帧检测）
    │ 有变化 / 心跳 / 异常锁定
    ▼
[Stage 2] YOLO 人员过滤 ── 遮罩/跳过人员区域
    │
    ▼
[Stage 3] Anomalib 异常检测 → 异常分数 + 热力图
    │
    ▼
[报警分级] 分数阈值 + 时间窗滤波(连续3帧) + 去重
    │
    ├→ Dashboard (http://localhost:8080)
    ├→ Webhook (HTTP POST)
    ├→ SQLite 持久化
    └→ 快照 + 热力图保存
```

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# 2. 安装依赖
pip install -e ".[dev]"

# 3. 运行（Dashboard 自动在 :8080 启动）
python -m argus --config configs/default.yaml

# 4. 运行测试
python -m pytest tests/unit/ -v
```

## 核心功能清单

### 已实现

| # | 功能 | 模块 | 状态 |
|---|------|------|------|
| **检测管线** | | | |
| 1 | 多协议摄像头采集（RTSP/USB/文件） | `capture/camera.py` | ✅ |
| 2 | 断线自动重连（指数退避） | `capture/camera.py` | ✅ |
| 3 | 帧读取超时（5s，防冻结流卡死） | `capture/camera.py` | ✅ |
| 4 | 帧率看门狗（30s 无帧自动重连） | `capture/manager.py` | ✅ |
| 5 | MOG2 背景减除预筛（节约~80%算力） | `prefilter/mog2.py` | ✅ |
| 6 | 辐射椒盐噪声中值滤波 | `prefilter/mog2.py` | ✅ |
| 7 | YOLO 人员检测过滤（遮罩/跳帧模式） | `person/detector.py` | ✅ |
| 8 | YOLO 不可用时优雅降级（不崩溃） | `person/detector.py` | ✅ |
| 9 | Anomalib 异常检测（PatchCore/EfficientAD） | `anomaly/detector.py` | ✅ |
| 10 | 冷启动 SSIM 回退（无模型可用时） | `anomaly/detector.py` | ✅ |
| 11 | 模型热更新（原子替换，不中断推理） | `anomaly/detector.py` | ✅ |
| **防背景吸收** | | | |
| 12 | 心跳全帧检测（每30s跳过MOG2强制检测） | `core/pipeline.py` | ✅ |
| 13 | 异常状态锁定（高置信度异常持续监控） | `core/pipeline.py` | ✅ |
| **区域管理** | | | |
| 14 | include/exclude 多边形区域掩码 | `core/zone_mask.py` | ✅ |
| 15 | 区域热更新（不重启管线） | `core/zone_mask.py` | ✅ |
| 16 | 多区域独立检测和报警 | `core/pipeline.py` | ✅ |
| **报警系统** | | | |
| 17 | 四级报警分级（INFO/LOW/MEDIUM/HIGH） | `alerts/grader.py` | ✅ |
| 18 | 时间窗滤波（连续N帧确认，降低误报） | `alerts/grader.py` | ✅ |
| 19 | 重复报警抑制（同区域去重） | `alerts/grader.py` | ✅ |
| 20 | 区域优先级乘数（关键区域放大分数） | `alerts/grader.py` | ✅ |
| 21 | 报警 debug 日志（帮助调试报警逻辑） | `alerts/grader.py` | ✅ |
| 22 | 多通道分发（数据库 + Webhook） | `alerts/dispatcher.py` | ✅ |
| 23 | 快照和热力图自动保存 | `alerts/dispatcher.py` | ✅ |
| **多摄像头** | | | |
| 24 | 线程池管理（每摄像头独立线程） | `capture/manager.py` | ✅ |
| 25 | 独立启停单个摄像头 | `capture/manager.py` | ✅ |
| **数据持久化** | | | |
| 26 | SQLite + WAL 模式 | `storage/database.py` | ✅ |
| 27 | 写入重试（3次，防瞬态失败丢数据） | `storage/database.py` | ✅ |
| 28 | 报警确认/标记误报 | `storage/database.py` | ✅ |
| 29 | 分页查询 + 多条件过滤 | `storage/database.py` | ✅ |
| **误报反馈** | | | |
| 30 | 误报标记导出到基线目录 | `alerts/feedback.py` | ✅ |
| 31 | 误报率统计 | `alerts/feedback.py` | ✅ |
| **基线管理** | | | |
| 32 | 版本化基线存储（v001, v002...） | `anomaly/baseline.py` | ✅ |
| 33 | 基线采集脚本 | `scripts/capture_baseline.py` | ✅ |
| 34 | 旧版本自动清理 | `anomaly/baseline.py` | ✅ |
| **模型训练** | | | |
| 35 | Anomalib 训练脚本（PatchCore/EfficientAD） | `scripts/train_model.py` | ✅ |
| 36 | 模型导出脚本（OpenVINO/ONNX） | `scripts/export_model.py` | ✅ |
| 37 | 训练编排（验证→训练→导出） | `anomaly/trainer.py` | ✅ |
| **Dashboard 仪表盘** | | | |
| 38 | 系统概览（指标卡片 + 摄像头网格 + 最近报警） | `dashboard/routes/system.py` | ✅ |
| 39 | 摄像头状态（帧数、跳过率、延迟、报警数） | `dashboard/routes/cameras.py` | ✅ |
| 40 | 报警列表（分页 + 严重级别筛选） | `dashboard/routes/alerts.py` | ✅ |
| 41 | 一键确认/标记误报 | `dashboard/routes/alerts.py` | ✅ |
| 42 | JSON API（供外部集成） | `dashboard/routes/alerts.py` | ✅ |
| 43 | 区域编辑器（Canvas 画多边形，即时生效） | `dashboard/routes/zones.py` | ✅ |
| 44 | 配置控制台（阈值调整、摄像头重启、清除锁定） | `dashboard/routes/config.py` | ✅ |
| 45 | HTMX 自动刷新（3-5s 轮询） | `dashboard/app.py` | ✅ |
| **系统监控** | | | |
| 46 | 健康状态（HEALTHY/DEGRADED/UNHEALTHY） | `core/health.py` | ✅ |
| 47 | 定时任务调度（APScheduler） | `core/scheduler.py` | ✅ |
| **部署** | | | |
| 48 | Docker 多阶段构建 | `Dockerfile` | ✅ |
| 49 | docker-compose（数据卷持久化、资源限制） | `docker-compose.yml` | ✅ |
| 50 | Pydantic 类型安全配置 + YAML | `config/schema.py` | ✅ |
| **测试** | | | |
| 51 | 83 个单元测试（全绿） | `tests/unit/` | ✅ |

### 未实现（Phase 5 / 后续）

| # | 功能 | 说明 |
|---|------|------|
| 1 | Modbus/OPC-UA 接口 | 对接电厂 DCS 系统，需现场调试 |
| 2 | 多节点聚合 | 中央服务器聚合多个边缘节点，需 PostgreSQL |
| 3 | AnomalyDINO 零样本模式 | 需 anomalib >= 2.0，零标注即可检测 |
| 4 | WebSocket 实时视频流 | Dashboard 实时画面（当前是截图刷新） |
| 5 | 邮件/SMS 通知 | SMTP 集成 |
| 6 | 审计日志 | 合规记录（谁在什么时间做了什么操作） |
| 7 | Prometheus 指标导出 | 推理延迟、误报率趋势等可观测性 |
| 8 | 自动基线定时采集 | 定时从摄像头采集新基线 |
| 9 | 相位相关图像对齐 | 消除相机微震动影响 |

## 项目结构

```
argus/
├── pyproject.toml             # 项目依赖
├── configs/default.yaml       # 默认配置
├── Dockerfile                 # 容器化部署
├── docker-compose.yml         # 一键部署
├── src/argus/
│   ├── __main__.py            # CLI 入口
│   ├── config/                # Pydantic 配置系统
│   ├── capture/               # 摄像头采集 + 多摄像头管理
│   ├── prefilter/             # MOG2 背景减除 + 去噪
│   ├── person/                # YOLO 人员检测
│   ├── anomaly/               # Anomalib 异常检测 + 基线 + 训练
│   ├── core/                  # 管线编排 + 区域掩码 + 健康监控
│   ├── alerts/                # 报警分级 + 分发 + 误报反馈
│   ├── storage/               # 数据库 ORM
│   └── dashboard/             # FastAPI + HTMX 仪表盘
├── scripts/                   # 基线采集、模型训练、模拟工具
├── tests/unit/                # 83 个单元测试
└── data/                      # 运行时数据（git-ignored）
```

## 技术栈

| 层 | 技术 |
|----|------|
| 预筛 | OpenCV MOG2 + 中值滤波 |
| 人员过滤 | Ultralytics YOLO11n |
| 异常检测 | Anomalib PatchCore / EfficientAD |
| 推理加速 | OpenVINO / ONNX Runtime |
| Web | FastAPI + HTMX + Alpine.js |
| 数据库 | SQLAlchemy + SQLite (WAL) |
| 配置 | Pydantic + YAML |
| 容器化 | Docker + docker-compose |
| 测试 | pytest |

## 配置示例

```yaml
cameras:
  - camera_id: cam_01
    name: "反应堆厂房北侧"
    source: "rtsp://192.168.1.100:554/stream1"
    protocol: rtsp
    fps_target: 5
    zones:
      - zone_id: main_area
        name: "主检测区域"
        polygon: [[100,100], [1800,100], [1800,900], [100,900]]
        zone_type: include
        priority: critical
      - zone_id: fan_area
        name: "排风扇（排除）"
        polygon: [[50,50], [150,50], [150,150], [50,150]]
        zone_type: exclude
    mog2:
      denoise: true
      heartbeat_frames: 150
```
