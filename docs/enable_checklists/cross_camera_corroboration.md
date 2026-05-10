# 跨摄像头关联（Cross-Camera Corroboration）启用 Checklist

> 适用版本：当前 main 分支实现（`src/argus/core/correlation.py` + `src/argus/capture/manager.py`）。
> 模块默认 **关闭**，本 checklist 帮助新人按步骤启用并自检。
> 受众：现场部署/调试工程师。

---

## 1. 模块定位

跨摄像头关联（以下简称「关联模块」）的目的是 **降低单机位假阳性**：当两路摄像头视场重叠时，A 相机看到的异常如果在 B 相机对应位置 **不能被同样的高分热力图佐证**，那这个异常多半是 A 相机镜头脏污、单机位反光、局部抖动等假阳性。模块在告警发出之前进行二次检验，按配置选择「降级 severity」或「直接丢弃」。

与单机位告警的差异：

- 单机位告警链路 = MOG2 → Anomaly model → CUSUM 时间确认 → suppression → 出告警。
- 启用关联后，在 `CameraManager._alert_handler` 内追加一次 `CrossCameraCorrelator.check()`，结果写回 `Alert.corroborated` / `Alert.correlation_partner`。
- 关联模块 **不会让告警提前触发**，只会在已经触发的告警上加一层"二次确认"。

---

## 2. 前置条件

| 项 | 要求 |
|---|---|
| 摄像头数量 | 至少 **2 路**。仅有 1 路时禁止开启（无意义且会被代码当作"无 partner → 默认 corroborated=True"）。|
| 视场关系 | **必须有重叠区域**。模块用单应矩阵（homography, 3×3）把 A 的像素点投影到 B 的像素坐标，再在 B 的热力图上以投影点为中心、半径 25 像素的方框内取 max 分数。无重叠 → 投影落在画外 → 永远 uncorroborated。|
| 时间同步 | 两路摄像头时间偏差应小于 `cross_camera.max_age_seconds`（默认 5 秒）。生产环境建议接 NTP，把节点之间偏差压到 **< 200ms**。摄像头之间帧到达间隔 + 推理延迟一般 < 1s，留 4s 富余即可，不要把 max_age 调到 < 2s。|
| 标定 | **不强制**做 physics calibration。关联模块只需要每对摄像头的 3×3 单应矩阵，可以用一组人工选取的同名点（≥4 对）通过 `cv2.findHomography` 离线算出。如果项目里同时启了 `physics.triangulation_enabled`，那 `physics/multi_cam.py` 的标定与本模块**互不干扰**，单应矩阵仍要单独提供。|
| 热力图来源 | 配对的两路摄像头都必须运行异常检测管线，即 pipeline 必须能产出 `get_latest_anomaly_map()`（SSIM fallback 模式也算）。否则 partner 侧无热力图 → corroborated=False。|

什么场景下不要开：

- 只有 1 路摄像头。
- 摄像头视场完全独立（不同走廊、不同房间）。这种情况应当用 zone-priority 而不是关联。
- 摄像头帧率差异巨大（一路 30fps、一路 1fps）：低帧率那一侧热力图常常超过 max_age，关联永远拿不到 partner 数据，等同于始终 uncorroborated → 全部告警被降级或丢弃。

---

## 3. 配置项

源定义：`src/argus/config/schema.py` 中的 `CrossCameraConfig` 与 `CameraOverlapConfig`。
默认值：`configs/default.yaml` 第 246–251 行。

```yaml
cross_camera:
  enabled: false                       # 开关
  overlap_pairs: []                    # 摄像头对列表
  corroboration_threshold: 0.3         # partner 热力图最小分数
  max_age_seconds: 5.0                 # partner 数据最大时延
  uncorroborated_severity_downgrade: 1 # 未佐证时降几级
```

字段详解：

| 字段 | 类型 / 范围 | 默认 | 说明 / 调优建议 |
|---|---|---|---|
| `enabled` | bool | `false` | 总开关。改动需重启进程（`config.py` 的 `_restart_required_keys` 包含 `cross_camera.enabled`）。|
| `overlap_pairs` | list of `{camera_a, camera_b, homography}` | `[]` | 每对摄像头一个条目。`homography` 是 3×3 浮点矩阵，把 `camera_a` 像素点投影到 `camera_b` 像素坐标。模块构造时同时算出反向矩阵 `H_inv`，所以无需手动提供 B→A 那一份。|
| `corroboration_threshold` | float | `0.3` | 取值 0.1–0.9。partner 热力图在投影点 ±25px 区域内的 max 分数 ≥ 该值即视为佐证。**调高** → 更严格，更多告警被降级；**调低** → 更宽松，假阳性回升。建议从默认 0.3 起步，结合误报回放调整。|
| `max_age_seconds` | float | `5.0` | 取值 1.0–30.0。partner 热力图比当前告警时间戳老超过该值就忽略。摄像头同步差时建议放大；流式延迟低时可压到 2–3s 减少滞后告警。|
| `uncorroborated_severity_downgrade` | int 0–2 | `1` | 未佐证时把告警降几级（HIGH → MEDIUM → LOW → INFO）。设 `0` 关闭降级（仅打标，不改行为）；设 `2` 大幅压制；与 zone 内 `require_corroboration` 配合使用时，强制 zone 走"丢弃"，本字段对该 zone 不生效。|

补充：单个 zone 可以在 `cameras[].zones[]` 配 `require_corroboration: true`（`ZoneConfig.require_corroboration`），未佐证时该 zone 的告警**直接丢弃**，不走降级。该开关源自 schema.py 第 47–51 行。

---

## 4. 启用步骤

> 任意一步出错都不要继续，先按"自检命令"段落确认状态。

1. **盘点摄像头对**：确认哪些摄像头视场重叠，记录每对的同名点（≥4 对）。
2. **算单应矩阵**：用 OpenCV 离线算，例如：

   ```python
   import cv2, numpy as np
   src = np.array([[x1a,y1a],[x2a,y2a],[x3a,y3a],[x4a,y4a]], dtype=np.float32)
   dst = np.array([[x1b,y1b],[x2b,y2b],[x3b,y3b],[x4b,y4b]], dtype=np.float32)
   H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
   print(H.tolist())
   ```

3. **写入 `configs/default.yaml`**（或当前生效的 config 文件），把 `cross_camera.enabled` 置为 `true`，按下方格式追加 `overlap_pairs`：

   ```yaml
   cross_camera:
     enabled: true
     overlap_pairs:
       - camera_a: cam-01
         camera_b: cam-02
         homography:
           - [1.02, 0.01, 12.5]
           - [-0.01, 1.01, -8.3]
           - [0.0001, 0.0, 1.0]
     corroboration_threshold: 0.3
     max_age_seconds: 5.0
     uncorroborated_severity_downgrade: 1
   ```

4. **可选**：对必须强佐证的 zone，在 `cameras[].zones[]` 加 `require_corroboration: true`。
5. **重启后端**：`python -m argus --config configs/default.yaml`。`enabled` 切换属于"重启生效"集合。
6. **确认日志**：启动时应看到 `manager.cross_camera_enabled` 一条记录，附带 `pairs=N` 和 `threshold=...`。
7. **运行时调参**（可选）：scalar 字段（threshold / max_age / downgrade）和 `overlap_pairs` 都支持热更新，走系统设置 → "跨相机" 标签页或者直接调 `PUT /api/config/cross-camera/pairs`，无需重启。**唯一的例外是 `enabled` 自身**，从关闭切到开启必须重启。

---

## 5. 告警变化

### 5.1 数据库字段（`alerts` 表）

源：`src/argus/storage/models.py` 第 115–120 行 + `src/argus/storage/database.py` 自动迁移条目第 116–118 行：

```python
("alerts", "corroborated", "BOOLEAN"),
("alerts", "correlation_partner", "VARCHAR(50)"),
```

写入时机（`src/argus/capture/manager.py:775-820`）：

- `corroborated = None`：模块未开 / 该相机不在任何 pair 中 → 字段保持 NULL。
- `corroborated = True`：partner 在投影点附近找到 ≥ threshold 的高分。
- `corroborated = False`：模块运行了但 partner 分数不达标，或 partner 数据过期。
- `correlation_partner`：写入对面摄像头 ID（即使 corroborated=False，也会回填"本来该问的那个 partner"，方便排查）。

### 5.2 严重度调整

- `corroborated == False` 时：
  - 若 zone 设了 `require_corroboration: true` → 告警 **直接丢弃**，记录 `alerts.suppressed_uncorroborated` 日志。
  - 否则按 `uncorroborated_severity_downgrade` 降级（`HIGH → MEDIUM → LOW → INFO`，不会降到 None）。降级后写 `manager.alert_uncorroborated` 日志。
- `corroborated == True / None` 时：告警按原 severity 走，不动。

降级映射顺序见 `CameraManager._SEVERITY_ORDER`（`INFO < LOW < MEDIUM < HIGH`），降到 INFO 后不再继续降。

---

## 6. UI 入口

- 设置面板：路由 `/system`（`web/src/views/System.vue`），标签页 **"跨相机"**，组件 `CrossCameraPanel.vue`。可视化展示当前 `enabled` / threshold / max_age / pairs，并支持表内增删 pair、保存后热推送。
- API：
  - `GET /api/config/cross-camera`：读取当前配置 + 运行时计数（`total_pipelines`、`correlator_present`）。
  - `PUT /api/config/cross-camera/pairs`：更新 pairs 与三个 scalar，由 `update_cross_camera_config` 处理。
- 告警侧：
  - `GET /api/alerts/...` 返回字段已包含 `corroborated` / `correlation_partner`（`src/argus/dashboard/routes/alerts.py:114-115`）。
  - 告警详情面板 `web/src/components/alerts/AlertDetailPanel.vue` 读取 `correlation_partner`，存在则会渲染"多机位回放"入口（同时受 `event_group_count` 影响）。
  - 列表页本身没有专门列展示 corroborated，但筛选/排序可以基于这个字段扩展（当前未实现）。

---

## 7. 性能影响

> 本节数字标 **TBD** 处需要在目标硬件上 benchmark；不要在没测过的环境拍脑袋。

- **算法复杂度**：`CrossCameraCorrelator.check()` 对每条告警遍历 `self._pairs[camera_id]`（即该相机的所有 partner），每个 partner 只做一次 `cv2.perspectiveTransform`（单点）+ 一次 `partner_map[y1:y2, x1:x2].max()`（51×51 区域）。复杂度 O(P) 每告警，P=该相机的 partner 数。**不是 O(N²)**，没有空间索引（也不需要），实际上线上 P ≤ 4 即可覆盖典型场景。
- **热力图缓存**：`_recent_maps` 每个相机保留 **最新 1 张**（`(timestamp, anomaly_map)`），不是滑窗。内存占用 ≈ `相机数 × 单张 anomaly_map 字节数`。256×256 fp32 ≈ 256KB，1024×1024 ≈ 4MB，N=20 路相机最多 80MB。`prune_stale(max_age=30)` 可手动清理，模块自身不定时跑。
- **更新路径**：`CameraManager._loop` 每帧都会调 `correlator.update(camera_id, anomaly_map, time.time())`（`manager.py:963-969`），更新走 `_maps_lock` 单线程互斥，开销 ≈ 一次字典写入，纳秒级。
- **主链路延迟**：关联只发生在 **告警触发瞬间**（每相机典型每分钟 ≤ 几次），不在帧链路中。单次 `check` 在 1024×1024 热力图上预计 < 1ms（TBD：用 `time.perf_counter()` 包裹 `correlator.check` 实测）。
- **对比开销 / 收益**：当假阳性占告警总量 > 30% 时通常正收益，否则收益微弱。请先量化误报率，再决定是否启用。

测试方法（TBD 数据收集）：

```bash
# 1. 计时 check() 的纯算力开销
python -c "import time, numpy as np; from argus.core.correlation import CrossCameraCorrelator, CameraOverlapPair; c=CrossCameraCorrelator([CameraOverlapPair('a','b',[[1,0,0],[0,1,0],[0,0,1]])]); m=np.random.rand(1024,1024).astype(np.float32); c.update('b', m, time.time()); t=time.perf_counter(); [c.check('a',(512,512), time.time()) for _ in range(10000)]; print((time.perf_counter()-t)/10000*1000, 'ms/call')"

# 2. 内存：监控 process RSS 在多相机场景下的稳态值
```

---

## 8. 自检命令

最小可复现：开启 2 路摄像头 + 同步触发同位置告警。

1. **配置最小化双相机** + identity homography（视场近似重叠时可用）：

   ```yaml
   cross_camera:
     enabled: true
     overlap_pairs:
       - camera_a: cam-01
         camera_b: cam-02
         homography: [[1,0,0],[0,1,0],[0,0,1]]
   ```

2. **重启**：`python -m argus --config configs/default.yaml`。

3. **确认初始化日志**（`structlog` JSON）：

   ```
   manager.cross_camera_enabled  pairs=1 threshold=0.3
   ```

   未出现 → 检查 `enabled` 是否为 `true`、`overlap_pairs` 是否非空。

4. **触发同位置告警**（手动遮挡两台摄像头同一点 / 同时往两个画面里扔同一个异物），观察日志：

   - 佐证成功 → 看不到 `manager.alert_uncorroborated`，告警照常出。
   - 仅遮挡 cam-01 → 出现：

     ```
     manager.alert_uncorroborated  camera_id=cam-01  partner=cam-02  severity=...
     ```

   - 若 zone 设了 `require_corroboration: true`：

     ```
     alerts.suppressed_uncorroborated  alert_id=ALT-... camera_id=cam-01
     ```

5. **数据库验证**（默认 SQLite at `data/db/argus.db`）：

   ```bash
   sqlite3 data/db/argus.db "SELECT alert_id, camera_id, severity, corroborated, correlation_partner FROM alerts ORDER BY timestamp DESC LIMIT 10;"
   ```

   字段应能看到 `corroborated` 是 0/1/NULL，`correlation_partner` 是相机 ID 或 NULL。

6. **API 验证**：

   ```bash
   curl http://localhost:8080/api/config/cross-camera | python -m json.tool
   ```

   `runtime.correlator_present` 应为 `true`，pairs 数量正确。

7. **单元测试**（无需现场摄像头）：

   ```bash
   python -m pytest tests/unit/test_correlation.py -v
   python -m pytest tests/unit/test_corroboration_suppression.py -v
   ```

   绿灯说明本机模块代码自身工作正常。

期望日志关键字总览：

| 关键字 | 含义 |
|---|---|
| `manager.cross_camera_enabled` | 启动时模块已激活 |
| `manager.cross_camera_created_runtime` | 运行时通过 API 新建 correlator |
| `correlator.pairs_updated` | 热更新 pair 成功 |
| `manager.alert_uncorroborated` | 告警未佐证，已降级 |
| `alerts.suppressed_uncorroborated` | zone 强佐证，告警丢弃 |
| `manager.cross_camera_pairs_push_failed` | 热更新失败，看异常栈 |

---

## 9. 常见问题

**Q1：开了之后告警全降级 / 全丢，怎么办？**
先查 partner 侧 pipeline 是否真的在产 `anomaly_map`（SSIM fallback 也算），然后看 `max_age_seconds` 是不是太小、相机间时间是否同步。临时把 `corroboration_threshold` 调到 0.1，确认是配置问题还是阈值过严。

**Q2：单应矩阵该怎么估？精度要求多高？**
`cv2.findHomography(method=cv2.RANSAC)` 用 4–8 对同名点即可。投影点附近会取 ±25px 的 max，所以**单应矩阵在投影点处的误差 ≤ 25 像素就够用**，不必追求亚像素精度。两路相机分辨率不同也没关系，homography 自动吸收尺度差。

**Q3：3 台及以上摄像头怎么配？**
按"两两组合"列条目即可：cam-01↔cam-02, cam-01↔cam-03, cam-02↔cam-03。模块自动用反向矩阵补充对侧映射，所以每对只写一次。partner 数 P 增长缓慢（典型现场 P≤4），性能无压力。

**Q4：关联模块和 `physics.triangulation_enabled` 是同一个东西吗？**
**不是**。`physics/multi_cam.py` 是 3D 三角化，依赖 `data/calibration/` 下的内/外参，目的是算物体的真实世界坐标。本关联模块只用 2D 单应做"佐证 / 不佐证"的二元判断，互不依赖。

**Q5：可以只佐证特定 zone，其他 zone 不管吗？**
当前实现是 **相机级**：只要 `camera_a / camera_b` 在 `overlap_pairs` 里，该相机所有 zone 触发的告警都会进入关联检查。zone 粒度的差异控制只能通过 `ZoneConfig.require_corroboration`（决定不佐证时降级 vs. 丢弃），不能让某个 zone 完全跳过关联。
