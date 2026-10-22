# Argus 检测质量收敛计划（2026-06-12）

> 状态：执行中。原计划保留在本文，当前现场记录和恢复入口见
> `data/regression/HANDOFF.md`。

## 当前执行状态（2026-06-13）

| 阶段 | 状态 | 证据 |
| --- | --- | --- |
| Phase 0 冻结+定标 | 完成 | `2110d43 Freeze detection quality baseline`；`docs/acceptance.md` 已建立初版验收数字 |
| Phase 1 回归库首批素材+工具 | 首批完成，覆盖仍需扩展 | `ca96c94 Add real video regression suite`；`75150f1 Document regression corpus handoff` |
| Phase 2 收敛检测链 | 未开始 | 下一步选择真实发布模型/训练 PatchCore，并用回归库前后对比 |
| Phase 3 真实闭环验收 | 未开始 | 等 Phase 2 指标逼近验收后执行 USB business smoke、现场演练和 soak |
| Phase 4 防回归固化 | 未开始 | 等回归库指标稳定后接 nightly/CI 守门 |

已录制并同步首批真实素材：

- 远端源：`gpu-validation:/srv/argus-regression-capture`
- 本地忽略视频目录：`data/regression/clips/`
- 索引入口：`data/regression/manifest.yaml`
- 交接入口：`data/regression/HANDOFF.md`
- 负样本：`neg_scene_001_30min_20260613_095718`，30 分钟，54000 帧
- 静态异物：`pos_static_foreign_10events_wide_20260613_103042`，10 个粗窗口，35999 帧
- 高速小物体：`pos_projectile_10events_wide_20260613_105301`，10 个粗窗口，23398 帧
- 后续采集默认帧率：OBSBOT Meet 2 `/dev/video0` 使用 MJPEG 1920x1080 60 FPS；已验证 120 FPS 会降到 60、4K 限 30、YUYV 限 30。本批 2026-06-13 素材仍是 1920x1080 30 FPS。

首版基线（SSIM fallback，`sample_every=30`，约 1 FPS）：

| 指标 | 目标 | 实际 | 状态 |
| --- | ---: | ---: | --- |
| False positives/hour | <= 2.00 | 11.5044 | fail |
| Event recall | >= 0.95 | 0.5000 | fail |
| P95 detection delay | <= 10.00s | 43.0000s | fail |

重要限制：

- 这只是首批回归素材，不等于完整 Phase 1 覆盖；后续仍需更多场景、更多静态异物位置/大小/光照组合。
- 正样本时间窗是现场粗标注，操作者确认存在秒级误差；做严格帧级守门前必须目视复核并收紧边界。
- 高速小物体需求仍是决策点；若产品范围不保留，应从 go/no-go 指标中拆出或降级。
- 当前基线是 SSIM fallback 抽样，不是主模型通道基线；Phase 2 必须用真实发布模型或新训练 PatchCore 重新跑。
- 通道归因目前没有接入 pipeline inference records，因此 fallback 报表中的 FP attribution 全部落在 `anomaly`。

下一步从这里继续：

1. 选择真实部署模型路径或用首批负样本训练并发布 PatchCore。
2. 目视复核两段正样本 labels，收紧 `positive_ranges`。
3. 用 `scripts/run_regression_suite.py --manifest data/regression/manifest.yaml --model <model>` 重新跑主模型基线。
4. 决定是否保留高速小物体验收项；如保留，继续补更多速度/亮暗/拖影样本。
5. 新录素材统一使用 MJPEG 1920x1080 60 FPS，除非专门测试 4K 或低码率模式。

## 0. 问题定性

链路层面（采集 → 告警 → 回放 → 训练 → 发布）已经通了，全量测试与 smoke 全绿。
不达预期的是**真实场景下的检测质量**，根因有三个：

1. **没有量化的验收标准。** 现有文档只描述能力范围，没有“误报 ≤ N/小时、召回 ≥ Y%”这类可判定的指标。任何修改都无法客观判定“变好还是变坏”。
2. **没有真实素材回归库。** 最近 40+ 提交全是按单一场景打补丁（低光抑制、反光守门、SSIM 校准守门……），修一个场景可能弄坏另一个，且无法察觉。
3. **主模型通道未真正上岗。** 审计中“异常检测主通道已验证”的证据来自 SSIM fallback。真实部署实际依赖 SSIM fallback + simplex + persistent_change + fast_motion 等多条启发式通道并行，每条十几个手调阈值，对光照/反光/振动天生脆弱，靠调参不可能收敛。

已有但未发挥作用的零件：`src/argus/preprocessing/alignment.py`（配准）、
`src/argus/anomaly/ensemble.py`（双模型）、`scripts/evaluate_video.py` /
`scripts/evaluate_regression.py`（离线评估 + `--sweep` + `--min-f1` 守门）。

**计划的核心原则：先建度量，再改代码；凡是不能在回归库上整体提升的改动一律不合入；不再新增场景特判。**

## Phase 0 — 冻结现场，定义“预期”（0.5–1 天）

1. 处理当前工作区 65 个文件、约 6000 行的未提交改动：按主题拆分提交或回退，
   回到可复现的基线。后续所有质量数字都要能对应到一个 commit。
2. 新建 `docs/acceptance.md`，写下可判定的验收标准（数字先拍初版，Phase 1 拿到基线数据后修订）。建议模板：

   | 指标 | 目标 | 测量方式 |
   | --- | --- | --- |
   | 负样本误报率 | ≤ 2 次/小时（每路） | 回归库负样本素材 + evaluate_video.py |
   | 静态异物召回（≥ 目标最小尺寸） | ≥ 95%，触发延迟 ≤ 10s | 回归库正样本素材 |
   | 高速小物体召回（如保留此需求） | ≥ 80% | 专项正样本素材 |
   | 72h soak | 无崩溃、无内存增长、单路故障不拖全局 | 长跑 + 健康指标 |

3. 同时明确**目标部署的物理边界**：最低照度、摄像头是否固定、目标物体最小像素尺寸。低于边界的场景明确写为“不承诺”。

产出：一个干净的 git 基线 + `docs/acceptance.md`。

## Phase 1 — 真实素材回归库（2–4 天，需现场配合，关键路径）

这是整个计划的咽喉。没有素材，一切修改仍是盲调。

1. 用现场摄像头（OBSBOT 或同款）录制：
   - **负样本**：每个目标场景 ≥ 30 分钟连续素材，必须覆盖：开/关灯、人员走动、
     反光、低光、轻微抖动、日光变化。这些正是历史误报来源。
   - **正样本**：异物放置（不同大小/位置/光照）≥ 10 段；如保留高速小物体需求，
     螺丝/小物体飞过（不同速度/亮暗/拖影）≥ 10 段。
2. 用 `evaluate_video.py` 的 annotations 格式（inclusive frame ranges）标注正样本。
3. 素材入库 `data/regression/`（或 NAS + 清单文件），写 README 说明每段覆盖什么场景。
4. 写一个一键脚本（`scripts/run_regression_suite.py`）：对全库逐段跑
   `evaluate_video.py`，汇总输出每段的 FP/FN/分数分布，以及**按检测通道归因**
   （每个误报来自 fast_motion / simplex / persistent_change / ssim_fallback /
   anomaly 哪条通道）。inference records 已有逐帧记录，归因是小开发量。
5. 跑出**当前代码的基线报表** —— 这是第一份“现在到底差多少”的客观答案，
   也用来修订 Phase 0 的验收数字。

产出：回归库 + 一键评估 + 当前基线数字。

## Phase 2 — 收敛检测链（1–2 周）

合入纪律：每个改动必须附回归库前后对比，整体不提升不合入；禁止新增场景特判。

按优先级：

1. **让主模型通道真正上岗。** 用 Phase 1 现场的负样本素材采集基线、训练
   PatchCore（必要时 ensemble），通过 release pipeline 发布到目标摄像头。
   SSIM fallback 回归其设计定位：模型不可用时的降级备胎，而不是常态主力。
2. **按归因报表做通道减法。** 误报贡献最大的启发式通道：优先考虑默认关闭或
   降权（如 fast_motion 在不需要高速小物体的场景默认关），其次才是调参。
   多条并行启发式通道是误报的结构性来源，方向是减通道，不是加守门。
3. **阈值用数据选。** 所有 threshold 用 `evaluate_regression.py --sweep` 在
   回归库上扫出来，替换手拍值；结果记录进 acceptance.md。
4. **清理历史特判。** 此前为单一场景加的抑制逻辑逐个回测：在回归库上无正贡献的
   删除（每删一个跑一次全库确认无回归）。

产出：回归库指标达到（或逼近）acceptance.md；若逼近不了，进入“调整预期”决策（见 §风险）。

## Phase 3 — 真实闭环验收（3–5 天，关闭审计遗留项）

对应 `docs/audits/system-integration-audit-2026-05-26.md` 的“仍需补充的验证”：

1. 真实 USB 完整 business smoke：
   `python scripts/smoke_dashboard_business_flow.py --preflight --camera-source 0 --camera-protocol usb --require-go2rtc --browser required`
2. 现场实物演练：按正样本场景实际放置/抛掷物体，确认告警、Replay、Reports 闭环，
   semantics（detection_type、bbox）正确。
3. 24–72h soak：记录 FP/小时、内存曲线、降级与恢复事件；如条件允许加第二路摄像头
   验证单路故障隔离。
4. 对照 acceptance.md 出具 go / no-go 结论，更新审计文档。

## Phase 4 — 防回归固化（1–2 天）

1. 回归库（或抽样子集）进 nightly CI：`evaluate_regression.py --min-f1` 作守门，
   指标跌破即红。
2. 流程约定写进 CLAUDE.md / requirements.md：检测链改动必须附回归库前后数字。
3. 把审计文档的“已验证/部分验证”矩阵改为引用回归库报表数字，不再用“链路通过”
   代替质量结论。

## 风险与决策点

- **素材是关键路径。** Phase 1 拿不到现场素材，后面全部失效。退而求其次可先用
  现有 inference records（`build_eval_set_from_inference_records.py`）凑第一版
  负样本库，但正样本必须真实录制。
- **预期对齐是计划的另一半。** 如果 Phase 2 结束时数字仍达不到验收，正确动作不是
  继续改代码，而是回到 acceptance.md 缩范围：例如只承诺静态异物、放弃高速小物体，
  或对部署提出最低照度/固定光照要求。“能实现”=能力与承诺对齐。
- **避免再次发散。** 计划执行期间冻结新功能（前端、报表、3D 等一律不动），
  所有人力集中在度量与检测质量上。

## 里程碑总览

| 阶段 | 工期 | 出口判据 |
| --- | --- | --- |
| P0 冻结+定标 | 0.5–1 天 | 干净 git 基线；acceptance.md 初版 |
| P1 回归库 | 2–4 天（含现场） | 一键评估可跑；当前基线数字出炉 |
| P2 收敛 | 1–2 周 | 回归库指标达标或触发缩范围决策 |
| P3 真实验收 | 3–5 天 | 真实 USB 闭环 + soak 通过，go/no-go |
| P4 固化 | 1–2 天 | nightly 守门上线 |
