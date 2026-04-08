# Argus — 视觉异常检测系统

基于固定摄像头与深度学习的智能视觉异常检测系统，实现 7×24 小时不间断监控。

## 项目简介

Argus 通过实时视频流分析，自动检测监控区域内出现的异常物体或场景变化（遗留物品、结构损伤、环境异常等），并进行四级分级报警与多通道推送。系统采用边缘部署架构，单节点支持 2-4 路摄像头，支持 CPU-only 推理与内网离线环境。

**核心特性：**
- **少样本异常检测** — 仅需少量"正常"图像即可训练，无需异常样本标注（Dinomaly2 最少 8 张）
- **双通道安全架构** — 深度学习（Anomalib）+ 传统 CV（Simplex 帧差）并行运行，单通道故障不影响检测
- **多阶段检测管线** — 区域掩码 → MOG2 预筛 → YOLO 人员过滤 → 双通道异常检测 → CUSUM 时序确认 → 分级报警
- **全链路可观测** — 告警回放取证、降级检测、模型漂移监控、审计日志
- **模型全生命周期** — 基线采集 → 训练 → 影子部署 → 金丝雀 → 生产上线，完整 MLOps 流程

## 系统架构

```
摄像头(RTSP/USB/File) → CameraManager（线程池）
    │
    ▼
[Stage 0] ZoneMask ── include/exclude 多边形区域掩码
    │
    ▼
[Stage 1] MOG2 预筛 ── 无变化 → 跳过（心跳强制全帧 + 异常锁定旁路）
    │ 有变化
    ▼
[Stage 2] YOLO 多类目标检测 ── 人员遮罩/跳帧 + BoT-SORT 追踪
    │
    ▼
[Stage 3] 双通道并行检测
    ├── Anomalib（Dinomaly2/PatchCore/EfficientAD）→ 异常分数 + 热力图
    └── Simplex 安全通道（帧差 + 静态参考）→ 独立分数
    │ 融合
    ▼
[Stage 4] OVD 分类（YOLO-World，可选）+ SAM 2 分割（可选）
    │
    ▼
[后处理] AnomalyPostprocess + TemporalTracker + DriftDetector(KS)
    │
    ▼
[报警] AlertGrader（自适应阈值 + CUSUM 证据累积 + 去重抑制）
    │
    ▼
AlertDispatcher → SQLite + Webhook + 邮件 + WebSocket 实时推送
```

## 快速开始

```bash
# 安装（Python 3.11+）
pip install -e ".[dev]"

# 运行（Dashboard 自动在 :8080 启动）
python -m argus --config configs/default.yaml

# 运行测试
python -m pytest tests/ -v
```

## 功能概览

### 检测管线

| 功能 | 说明 |
|------|------|
| 多协议摄像头采集 | RTSP / USB / 视频文件，断线自动重连（指数退避） |
| MOG2 背景减除预筛 | 节约 ~80% 算力，含噪声抑制与相位相关防抖 |
| YOLO 多类目标检测 | YOLO11n 80 类 COCO + BoT-SORT 追踪，人员遮罩/跳帧 |
| Anomalib 异常检测 | Dinomaly2 / PatchCore / EfficientAD，SSIM 冷启动回退 |
| Simplex 安全通道 | 纯 OpenCV 帧差检测器，与 ML 通道并行，安全兜底 |
| 防背景吸收 | 心跳全帧检测 + 异常状态锁定，防止 MOG2 学习吞帧 |
| OVD 开放词汇分类 | YOLO-World 对异常区域分类（可选） |
| SAM 2 实例分割 | 异常区域精细分割（可选） |

### 报警系统

| 功能 | 说明 |
|------|------|
| 四级分级报警 | INFO / LOW / MEDIUM / HIGH，区域优先级乘数 |
| CUSUM 时序确认 | 软证据累积替代硬性连续帧计数，降低误报 |
| 自适应阈值 | 基于滑动窗口分数分布动态调整 |
| 多通道分发 | SQLite + Webhook + 邮件 + WebSocket，含断路器 |
| 告警工作流 | NEW → ACKNOWLEDGED → INVESTIGATING → RESOLVED → CLOSED |
| 告警回放取证 | 环形缓冲区录制（前 60s + 后 30s），多轨信号时间线 |

### 模型管理

| 功能 | 说明 |
|------|------|
| 基线生命周期 | 版本化存储，质量过滤，激活/退役 |
| 模型训练 | 训练前验证 → 训练 → 质量报告 → A/B 对比 → 导出 |
| 发布管线 | shadow（≥3 天）→ canary（≥7 天）→ production |
| DINOv2 骨干训练 | 共享骨干 SSL 微调，升级后自动触发下游重训练 |
| INT8 量化 | NNCF PTQ 量化导出，降低推理延迟 |
| Conformal 校准 | 分布无关的 FPR 保证 |

### Dashboard 仪表盘

Vue 3 + Ant Design Vue SPA，暗色主题，WebSocket 实时更新。

| 页面 | 说明 |
|------|------|
| 系统总览 | 状态卡片、摄像头网格、最近告警 |
| 摄像头管理 | 添加/启停/统计，go2rtc 视频流 + MJPEG 回退 |
| 告警中心 | 筛选/批量操作/CSV 导出/工作流状态/音频告警 |
| 回放取证 | 多轨时间线、逐帧检视、信号历史、参考帧对比 |
| 模型管理 | 基线管理/训练/模型列表/批量推理/骨干网络/事件日志 |
| 降级监控 | 全局降级条、降级事件历史、影响评估 |
| 检测调试 | 帧级诊断、灵敏度预览、学习模式 |
| 系统设置 | 阈值调整、通知配置、存储维护 |
| 用户/审计/报表/备份 | RBAC 权限、操作审计、统计报表、数据库备份恢复 |

### 系统监控

| 功能 | 说明 |
|------|------|
| 摄像头健康 | 画面冻结、镜头污染、位移、闪光干扰、漂移检测 |
| 降级检测 | 5 种降级场景自动识别，状态机追踪 |
| KS 漂移监控 | 异常分数分布偏移检测 |
| 跨摄像头关联 | 多视角异常时空关联，提升置信度 |

## 项目结构

```
src/argus/                   # 后端（103 个 Python 模块，12 个子包）
├── config/                  # Pydantic 配置系统
├── capture/                 # 摄像头采集 + 基线采集 + 帧采样
├── prefilter/               # MOG2 背景减除 + Simplex 安全通道
├── person/                  # YOLO 多类目标检测
├── anomaly/                 # 异常检测 + 基线 + 训练 + 骨干 + 影子部署
├── core/                    # 管线编排 + 区域掩码 + 健康 + 降级 + 环形缓冲
├── alerts/                  # 报警分级 + 分发 + 反馈 + 校准
├── storage/                 # SQLite ORM（14 模型）+ 模型注册 + 发布管线
├── streaming/               # go2rtc 流媒体代理（WebRTC/MSE/HLS）
├── contracts/               # 数据契约验证
├── validation/              # 合成异常 + Recall 评估
└── dashboard/               # FastAPI + 17 个路由模块

web/                         # 前端（Vue 3 + Ant Design Vue，32 个文件）
├── components/              # 7 通用组件 + 10 模型管理子组件
├── composables/             # WebSocket / go2rtc / 模型状态
└── views/                   # 6 页面（总览/摄像头/告警/模型/系统/详情）

tests/                       # 935 个测试函数，72 个测试文件
configs/default.yaml         # 默认配置
scripts/                     # 基线采集、模型训练、导出、模拟工具
```

## 技术栈

| 层 | 技术 | 版本 |
|----|------|------|
| 视频采集 | OpenCV | >= 4.10 |
| 目标检测 | Ultralytics YOLO11n + BoT-SORT | >= 8.3 |
| 异常检测 | Anomalib（Dinomaly2/PatchCore/EfficientAD），DINOv2 backbone | >= 2.0 |
| 推理加速 | OpenVINO | >= 2024.0 |
| Web 框架 | FastAPI + WebSocket | >= 0.115 |
| 前端 | Vue 3 + Ant Design Vue + Vite | Vue 3.5 |
| 视频流 | go2rtc（WebRTC/MSE/HLS/MJPEG） | 内置 |
| 数据库 | SQLAlchemy + SQLite（WAL） | >= 2.0 |
| 配置 | Pydantic v2 + YAML | >= 2.0 |
| 日志 | structlog（JSON + 控制台） | >= 24.0 |
| 容器化 | Docker + docker-compose | - |
| 语言 | Python 3.11+ / TypeScript | - |

## 配置示例

```yaml
cameras:
  - camera_id: cam_01
    name: "厂房北侧监控"
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
    simplex:
      enabled: true
      diff_threshold: 30
```

## 部署

```bash
# Docker 一键部署
docker-compose up -d

# 资源需求：4 CPU / 4GB RAM
# Dashboard: http://localhost:8080
# 健康检查: GET /api/system/health
```

## 许可证

MIT
