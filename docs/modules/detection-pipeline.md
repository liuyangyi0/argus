# 检测管线

> 关键源码：`src/argus/core/pipeline.py`、`src/argus/core/runner.py`、`src/argus/capture/manager.py`、`src/argus/alerts/grader.py`、`src/argus/alerts/evidence.py`、`src/argus/prefilter/`、`src/argus/person/detector.py`

检测管线的核心单位是“单路摄像头”。`CameraManager` 负责多路调度，`CameraInferenceRunner` 负责单路状态聚合，`DetectionPipeline` 负责帧级处理和告警候选生成。

## 对象关系

```text
CameraManager
  -> CameraInferenceRunner
     -> DetectionPipeline
        -> CameraCapture
        -> ZoneMaskEngine
        -> FrameAlignment
        -> FastMotionDetector
        -> MOG2PreFilter
        -> YOLOObjectDetector
        -> AnomalibDetector / DetectorEnsemble
        -> SimplexDetector
        -> AlertGrader
        -> AlertFrameBuffer
```

`CameraInferenceRunner` 是外部查询单路推理状态的统一入口，它把 pipeline、健康状态、降级状态、模型版本 tag、CUSUM 状态和统计信息聚合成 `RunnerSnapshot`。

## Pipeline 模式

`PipelineMode` 当前包含：

- `active`：正常检测并产生告警。
- `maintenance`：维护模式，检测和告警继续，但 MOG2 背景更新逻辑会受控。
- `learning`：完整管线运行，但告警被抑制，用于学习/校准。
- `collection`：采集模式，跳过 YOLO 和 anomaly，保留原始帧流。
- `training`：训练中模式，跳过 YOLO 和 anomaly，为 GPU/CPU 释放资源。

训练作业通过 `GlobalPipelineModeGuard` 临时把所有摄像头切到 `training`，结束后恢复之前模式。

## 帧处理阶段

单帧主链路大致如下：

1. 从 `CameraCapture` 读取 `FrameData`。
2. 更新 ring buffer，用于告警前后录像固化。
3. 应用 include/exclude zone mask。
4. 可选相位相关对齐，减轻固定摄像头微振导致的误报。
5. fast motion 检测快速运动小目标，可直接形成 projectile 类告警。
6. MOG2 预筛，低变化帧跳过重模型；heartbeat 和锁定区域可绕过 MOG2。
7. YOLO 人员/对象过滤，按配置决定只标注、跳过或参与分类。
8. 异常检测主通道，默认基于 anomalib/OpenVINO 导出模型；可接 ensemble 或 SSIM fallback。
9. Simplex 安全通道并行产生低依赖候选。
10. 可选分类器、分割器、物理轨迹、传感器融合增强告警信息。
11. `AlertGrader` 做时序证据、阈值、区域优先级、类别和抑制判断。
12. 告警成立后固化录像、保存推理记录、发布事件和回调 `AlertDispatcher`。

## MOG2 与反吸收

MOG2 的目标是降低重模型吞吐压力，不是最终判定器。为避免静态异物被背景模型吸收，管线包含：

- heartbeat 全帧旁路。
- 异常区域锁定。
- MOG2 背景更新冻结/维护模式控制。
- lock clear frames 机制。

清除锁定状态通过配置 API 或 pipeline 方法触发，避免长时间锁住已处理区域。

## 告警判定

`AlertGrader` 将多类信号合成为最终告警：

- anomaly score。
- Simplex 检测结果。
- fast motion/projectile 候选。
- CUSUM 时序证据。
- zone priority 和 severity multiplier。
- same-zone / same-camera suppression。
- early warning 规则。
- 可选分类器/分割器/跨摄像头 corroboration。

`Alert` 会携带 camera、zone、severity、score、bbox、category、detection type、model version、物理/分类/分割增强字段等信息。进入 `AlertDispatcher` 后才进行落库、图像保存、Webhook/邮件/WebSocket 分发。

### P1 诊断证据

P1 候选诊断把“为什么报警/为什么没报警”收敛为稳定结构：

- `DetectionEvidence` 是告警和非告警候选共用的 JSON-safe 证据对象，包含 `detection_type`、`score`、`camera_id`、`zone_id`、`zone_type`、`bbox`、`frame_number`、`severity`、`evidence_json`、`features`、`candidates`、`model_version_id`、`debug_trace_id`、`stable_for_sec` 和 `suppression_reason`。
- `AlertGrader.evaluate_with_evidence()` 返回 `GradeResult`，同时给出 `should_alert`、最终 `severity`、`DetectionEvidence`、`suppression_reason` 和可选 `Alert`。旧 `evaluate()` 只保留兼容入口。
- 告警成立时，`DetectionEvidence.evidence_summary()` 写入 `Alert.evidence_summary`，`DetectionEvidence.to_dict()` 写入 `Alert.evidence_json`，并随 `AlertDispatcher` 进入数据库、WebSocket 和 API。
- 告警被抑制时，不生成 `AlertRecord`；抑制原因保留在 `GradeResult.suppression_reason`、`FrameDiagnostics.candidate_records` 和 replay signals 中，用于解释阈值不足、学习模式、重复抑制等非告警结果。

## 降级与恢复

降级分两层：

- `DetectionPipeline` 内部记录 anomaly head 连续失败、segmenter 失败、低光照/输入质量受限等运行态。
- `CameraInferenceRunner` 使用 `DegradationStateMachine` 将连续失败映射为 `nominal`、`degraded_segmenter`、`degraded_detector`、`backbone_failed`、`restarting` 等状态。

全局可见降级事件由 `GlobalDegradationManager` 和 Dashboard WebSocket 主题广播给前端。降级是用户可见状态，不应只写日志。

## 诊断与记录

管线会维护：

- `PipelineStats`：捕获帧、跳过帧、分析帧、heartbeat、告警、平均延迟、FPS。
- `DiagnosticsBuffer`：阶段耗时、得分、异常区域、跳过原因等。
- `CandidateRecord`：每帧候选级证据，记录 `candidate_type`、`decision`、`evidence` 和 `suppression_reason`。
- `InferenceRecord`：非 NORMAL 帧的审计记录，进入文件记录和 DB write-behind buffer；`INFO_CANDIDATE` 和 `SKIPPED_HEARTBEAT` 默认持久化，`DEBUG_NORMAL_SAMPLE` 不默认持久化。
- `AlertFrameBuffer`：告警前后帧、热力图和视频片段固化。
- Prometheus metrics：推理延迟、处理帧、跳过帧、anomaly score、pipeline stage seconds。

## 修改建议

- 改 `process_frame()` 或 `_process_frame_inner()` 时，优先补 targeted unit test，避免只跑端到端 smoke。
- 新增 CPU 密集阶段时使用明确的 executor，不要抢占默认 API 线程池。
- 新增告警字段时同步检查 `Alert`、`AlertRecord`、`Database.save_alert()`、API 序列化和前端类型。
- 新增模式时同步 `PipelineMode`、模式切换 API、前端展示和 smoke 语义。

## 相关测试

- `tests/unit/test_pipeline_core.py`
- `tests/unit/test_pipeline_modes.py`
- `tests/unit/test_pipeline_mode_guard.py`
- `tests/unit/test_pipeline_reliability.py`
- `tests/unit/test_hybrid_pipeline.py`
- `tests/unit/test_temporal_tracker.py`
- `tests/unit/test_alert_grader.py`
- `tests/unit/test_alert_grader_evidence.py`
- `tests/unit/test_alert_evidence.py`
- `tests/unit/test_inference_record_candidate.py`
- `tests/unit/test_anomaly_postprocess.py`
- `tests/unit/test_frame_filter.py`
- `tests/unit/test_low_light.py`
