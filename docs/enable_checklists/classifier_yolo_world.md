# 启用 Checklist — YOLO-World 开放词汇分类器（D1）

> 适用对象：第一次接手 Argus、需要把开放词汇分类器从 "代码在仓库里但从来没人跑过" 推到 "在线上跑得通" 的工程师。
> 涉及文件：`src/argus/anomaly/classifier.py`、`src/argus/config/schema.py`、`src/argus/core/pipeline.py`、`src/argus/capture/manager.py`、`src/argus/dashboard/routes/config.py`、`web/src/components/system/ClassifierPanel.vue`、`configs/default.yaml`。

---

## 1. 模块定位

OpenVocabClassifier（代号 D1）是 Argus 异常告警链路上的 **可选语义分类层**。当主 anomaly 检测器（PatchCore/PaDiM 等）输出 `anomaly_score >= min_anomaly_score_to_classify` 的告警候选时，pipeline 会把异常区域裁剪后送进 [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) 这个开放词汇检测模型，给区域贴一个语义标签（例如 `wrench`、`shadow`、`crane`），随后：

- 命中 `high_risk_labels` → 告警严重度 +1 级（`pipeline.classifier_escalated`）。
- 命中 `low_risk_labels` → 告警严重度 −1 级（`pipeline.classifier_suppressed`）。
- 命中 `suppress_labels` → 直接抑制告警（`pipeline.classifier_label_suppressed`），不再下发。
- 还会把 `(label, confidence)` 注入 `EarlyWarning`，作为单帧快路径的二次佐证（`require_detection_or_classifier=true` 时必备其一）。

**为什么默认关闭**：
1. `ClassifierConfig.enabled` 在 `src/argus/config/schema.py:310` 的 schema 默认值是 `False`（仓库当前 `configs/default.yaml:208` 把它打开了，但 schema 默认仍然是 `False`，意味着新建的最小化 yaml 不会自动启用）。
2. 它依赖 `ultralytics>=8.3.0` 的 YOLOWorld 类，并且要在第一次 `load()` 时联网或本地命中 `yolov8s-worldv2.pt` 权重，对离线/受限环境是个隐患。
3. 在每个被裁剪出的异常 bbox 上多做一次 YOLO-World 推理，CPU 单核场景会显著拖慢主链路（见 §6），生产侧不见得能 always-on。

---

## 2. 依赖清单

### 2.1 Python 包

| 包 | 最低版本 | 说明 |
|---|---|---|
| `ultralytics` | `>=8.3.0` | 必须含 `YOLOWorld` 类（在 `src/argus/anomaly/classifier.py:65` 直接 `from ultralytics import YOLOWorld`）。`pyproject.toml:8` 已锁定。 |
| `torch` | 任意支持 CUDA 的版本 | 仅当希望走 GPU。`classifier.py:69` 通过 `torch.cuda.is_available()` 判断；ImportError 时会自动回退 CPU。 |
| `opencv-python` | `>=4.10.0` | bbox 裁剪用 numpy slicing，但 frame 在上游已经是 BGR ndarray。 |

`pip install -e ".[dev]"` 已经覆盖以上全部，无需额外 `pip install`。

### 2.2 模型权重

| 文件 | 期望路径 | 大小（参考） | 来源 |
|---|---|---|---|
| `yolov8s-worldv2.pt` | 工作目录（`configs/default.yaml:209` 写的是裸文件名，ultralytics 会按当前工作目录 + `~/.config/Ultralytics/` 顺序查找） | 约 50 MB（YOLOv8s-world v2 官方权重） | 第一次 `OpenVocabClassifier.load()` 会触发 ultralytics 自动从其 GitHub release 下载；离线环境需手动放置。 |

**离线部署提示**：
- 提前下载 `yolov8s-worldv2.pt` 放到 Argus 启动时的工作目录（一般是仓库根目录）。当前仓库根目录已存在 `yolov8s-worldv2.pt`（见 `git status` untracked 列表），证明此模式可用。
- 也可以把 `model_name` 改成绝对路径，例如 `model_name: D:\models\yolov8s-worldv2.pt`，避免依赖工作目录。

### 2.3 硬件要求

| 场景 | 推荐配置 |
|---|---|
| CPU only | 至少 4 物理核 / 8GB 内存。延迟会显著高于 GPU（见 §6 TBD）。 |
| GPU | 任意支持 CUDA 11.8+ 的 NVIDIA 卡，YOLOv8s-world FP32 显存约 1.5–2 GB。`classifier.py:71` 写死 `cuda:0`，多卡环境需要自行改。 |

> 没有 `torch` 或 `torch.cuda.is_available()` 为 False 时，`classifier.py:73-75` 会直接 fallback 到 CPU，不报错。

---

## 3. 配置项

所有字段定义在 `src/argus/config/schema.py:307-335`（`ClassifierConfig`），yaml 入口在 `configs/default.yaml:207-238`。

| 字段 | 默认值（schema） | 默认值（default.yaml） | 合法范围 | 调整建议 |
|---|---|---|---|---|
| `classifier.enabled` | `False` | `true` | bool | 关闭即整个分类层不接入 pipeline；可在 UI 热切换（见 §5）。 |
| `classifier.model_name` | `yolov8s-worldv2.pt` | 同左 | 任意 ultralytics YOLOWorld 兼容权重路径或文件名 | 想要更高精度可换 `yolov8l-worldv2.pt`（更大、更慢）。 |
| `classifier.vocabulary` | `FOE_VOCAB` 全集（`classifier.py:20-37`，约 50 项） | yaml 里只放了 17 项核心词 | 非空字符串列表 | 词表越大，YOLO-World 推理越慢；建议保留实际场景里出现的标签即可。 |
| `classifier.min_anomaly_score_to_classify` | `0.5` | `0.5` | `[0.0, 1.0]` | 调高以减少分类调用次数。低于该分数的候选直接跳过分类，pipeline.py:1581 处判断。 |
| `classifier.high_risk_labels` | 18 个核工业 FOE 高风险词 | yaml 里 5 项（`wrench` 等） | 必须是 `vocabulary` 的子集 | 命中后告警 +1 级。**API 层有子集校验**（`config.py:1075`），手改 yaml 不校验，自己注意。 |
| `classifier.low_risk_labels` | `["insect", "shadow", "reflection"]` | 同左 | 必须是 `vocabulary` 的子集 | 命中后告警 −1 级。 |
| `classifier.suppress_labels` | `["crane", "overhead_bridge", "scaffold"]` | （yaml 中未显式列出，使用 schema 默认） | 必须是 `vocabulary` 的子集 | 命中后整条告警直接吞掉。 |
| `classifier.custom_vocabulary_path` | `None` | （未设） | JSON 文件路径 | 设置后 pipeline 启动时会 `OpenVocabClassifier.load_vocabulary()`（`pipeline.py:711-715`）覆盖 yaml 中的 `vocabulary`。适合多套自定义 FOE 列表。 |

### 跨配置依赖

- **告警链路使用**：`alert.early_warning.require_detection_or_classifier`（`schema.py:746`，default.yaml:120）。当此项为 `true`（默认）且 `classifier.enabled=false` 时，单帧快路径只能靠 YOLO 检测器佐证；若 YOLO 检测器也没装，该路径几乎不会触发。
- **置信度门槛**：`alert.early_warning.classifier_min_confidence`（默认 `0.9`）。分类标签的置信度必须 ≥ 此值才会被视为有效佐证。YOLO-World 在 OOD 场景置信度普遍偏低，必要时降到 `0.5–0.7` 验证再回调。
- **运行时下发**：`PUT /api/config/classifier/vocabulary`（`config.py:1021`）会把新词表 push 到所有 live pipeline 的分类器（`camera_manager.update_classifier_vocabulary()`，`capture/manager.py:445`），但 **不会写回 yaml**。要持久化必须再走 `POST /api/config/save`。

---

## 4. 启用步骤

按编号顺序执行：

1. **确认依赖**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -c "from ultralytics import YOLOWorld; print(YOLOWorld.__module__)"
   ```
   期望输出 `ultralytics.models.yolo.world.model`（或类似），无 ImportError。

2. **准备权重**
   ```powershell
   # 仓库根目录
   Test-Path .\yolov8s-worldv2.pt
   ```
   返回 `False` 时，要么联网让 `OpenVocabClassifier.load()` 自动下载，要么手动从 ultralytics release 下载放在仓库根目录。

3. **编辑配置**
   打开 `configs/default.yaml`，确保 `classifier.enabled: true`（仓库现状已经是 `true`），并按需调整 `vocabulary` / `high_risk_labels` / `low_risk_labels` / `suppress_labels`。如果使用自定义词表 JSON：
   ```yaml
   classifier:
     enabled: true
     custom_vocabulary_path: data/classifier/foe_site_a.json
   ```

4. **启动后端**
   ```powershell
   python -m argus --config configs/default.yaml
   ```

5. **观察启动日志**（结构化字段，使用 structlog）
   - `pipeline.classifier_configured`（`pipeline.py:527`）：每个相机 pipeline 装载分类器配置时打印。
   - `classifier.loaded`（`classifier.py:78`）：模型权重首次加载完成。第一次会延迟数秒（取决于 I/O 与设备）。
   - 如果看到 `classifier.yoloworld_not_available`（`classifier.py:80`）→ `ultralytics` 缺失或版本过低；回到第 1 步。
   - 如果看到 `classifier.load_failed`（`classifier.py:82`）→ 通常是权重文件找不到或 GPU 初始化失败。

6. **触发一次推理**
   - 让任一相机出现 anomaly 候选（可用回放或人为遮挡）。
   - debug 日志中会出现 `pipeline.classified label=... confidence=...`（`pipeline.py:1599`）。

7. **可选：通过 UI 二次确认**
   - 浏览器进 Argus，选 **System → 分类器** Tab（路由见 §5）。`runtime.pipelines_loaded` 应当等于在线相机数。

---

## 5. UI 入口

- 前端路由：`#/system`，对应组件 `web/src/views/System.vue`，分类器面板在 Tab key=`classifier`，显示文案 **"分类器"**（`System.vue:64`）。
- 面板组件：`web/src/components/system/ClassifierPanel.vue`，提供：
  - 启用/关闭开关 → 调 `POST /api/config/modules`，body `{"key":"classifier.enabled","value":true|false}`（`config.py:1126`）。
  - 词表编辑（含 high/low/suppress 桶分配）→ `PUT /api/config/classifier/vocabulary`（`config.py:1021`）。
- 相关 API（用 `curl` 也能验证）：
  ```bash
  curl -s http://127.0.0.1:8000/api/config/classifier
  # -> { enabled, model_name, vocabulary[], runtime: { total_pipelines, pipelines_attached, pipelines_loaded } }
  ```

热切换语义：`classifier.enabled` 是 `_hot_reloadable_keys` 之一（`config.py:1158`），从 OFF→ON 会触发 `pipeline.classifier_hot_loaded`（`pipeline.py:2465`）。OFF 时直接置空 `_classifier`（`pipeline.py:2470`）。**不需要重启进程**。

---

## 6. 性能影响

> 仓库内没有正式 benchmark。下表数字为 ultralytics 官方 README 提供的量级参考 + 代码路径推断；正式上线前请用 §7 的自检脚本在目标硬件上自测。

| 指标 | 估计 | 怎么测 |
|---|---|---|
| 单帧分类延迟（GPU, 640×640 输入） | TBD（参考量级 5–15 ms） | 见 §7 自检脚本，输出 `cls_duration_ms`。 |
| 单帧分类延迟（CPU, 640×640 输入） | TBD（参考量级 80–250 ms，与 CPU 主频/词表大小相关） | 同上。 |
| GPU 显存峰值 | TBD（参考量级 1.5–2 GB FP32） | 启动后 `nvidia-smi` 取 Argus 进程 used 列。 |
| CPU 模式 RSS 增量 | TBD（参考量级 +600–900 MB） | `ps -o rss` 对比启用前后 Argus 主进程。 |
| 对主异常检测吞吐影响 | 仅 `anomaly_score >= min_anomaly_score_to_classify` 的帧会触发分类（`pipeline.py:1581`）；正常无告警时 0 开销。 | 在 grafana / `pipeline.classified` debug 日志中统计触发频率 × 单次延迟。 |

**关键观察**：分类调用在主 pipeline 协程里同步执行（`pipeline.py:1586` 直接调用 `self._classifier.classify(...)`，没有 `run_in_executor`）。CPU 模式 + 高频告警场景下，可能阻塞该相机协程，进而拖慢同 pipeline 后续帧。GPU 场景一般无问题。

---

## 7. 自检命令

### 7.1 离线 smoke test（不需要启动整个 Argus）

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import time, numpy as np; from argus.anomaly.classifier import OpenVocabClassifier; clf = OpenVocabClassifier(); clf.load(); frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8); t0 = time.perf_counter(); r = clf.classify(frame); print('result=', r, 'latency_ms=', round((time.perf_counter()-t0)*1000, 1))"
```

期望：
- 终端打印 `classifier.loaded model=yolov8s-worldv2.pt vocab_size=... device=cpu|cuda:0`（structlog 默认走 stdout）。
- `result=` 后面是 `('xxx', 0.xx)` 或 `None`（随机噪声常常分不出东西，None 也算正常）。
- `latency_ms=` 是真实的单次推理延迟，可以直接当 §6 表格里 TBD 的实测值。

### 7.2 配置层检查（无需模型权重）

```powershell
python -c "from argus.config.loader import load_config; c = load_config('configs/default.yaml'); print('enabled=', c.classifier.enabled, 'min_score=', c.classifier.min_anomaly_score_to_classify, 'vocab=', len(c.classifier.vocabulary))"
```

期望：`enabled= True min_score= 0.5 vocab= 17`（与当前 default.yaml 一致）。

### 7.3 在线状态自检

启动 `python -m argus --config configs/default.yaml` 后：

```powershell
curl http://127.0.0.1:8000/api/config/classifier
```

期望 JSON 字段（节选）：
```json
{ "enabled": true, "runtime": { "total_pipelines": N, "pipelines_attached": N, "pipelines_loaded": N } }
```
`pipelines_loaded` 在第一次实际触发分类后才会变成非零（懒加载，`classifier.py:60-62`）。

### 7.4 期望日志关键字（structlog event 名）

| 阶段 | event | 出现位置 |
|---|---|---|
| 配置接入 | `pipeline.classifier_configured` | `pipeline.py:527` |
| 模型加载 | `classifier.loaded` | `classifier.py:78` |
| 单帧推理结果 | `pipeline.classified` (debug) | `pipeline.py:1599` |
| 升级告警 | `pipeline.classifier_escalated` | `pipeline.py:1704` |
| 降级告警 | `pipeline.classifier_suppressed` | `pipeline.py:1714` |
| 抑制告警 | `pipeline.classifier_label_suppressed` (debug) | `pipeline.py:1617` |
| 词表热更新 | `classifier.vocabulary_updated` | `classifier.py:103` |

### 7.5 失败时的回退表现

- `ultralytics` 不可用或权重加载失败：`classifier.py:80,82` 打 warn/error，`self._model` 保持 None。`classify()` 直接 return None（`classifier.py:149-150`），pipeline 跳过分类逻辑，告警仍按原始 severity 出。**不会抛出阻塞主链路**。
- 单次推理异常：`pipeline.py:1604-1610` 捕获并打 `pipeline.classifier_error`，告警走未分类路径。
- bbox 越界：`classifier.py:158-159` `crop.size == 0` 直接 return None。

---

## 8. 常见问题

**Q1：日志里只看到 `classifier.yoloworld_not_available`，但 `pip show ultralytics` 显示已安装。**
A：很可能 `ultralytics<8.3.0`，没有 `YOLOWorld` 符号。运行 `python -c "from ultralytics import YOLOWorld"`，如果报 ImportError 就 `pip install -U "ultralytics>=8.3.0"`。

**Q2：第一次启动卡在 `classifier.loaded` 之前很久。**
A：首次会从 ultralytics GitHub release 下载 `yolov8s-worldv2.pt`。生产环境若不允许出网，请提前把权重放进仓库根目录（参考 §2.2）。

**Q3：通过 UI 改了词表，但重启后又变回去了。**
A：`PUT /api/config/classifier/vocabulary` 仅更新内存（`config.py:1087`）。要持久化必须再调 `POST /api/config/save` 或手改 yaml。这是已知设计，UI 上有 "未保存" 提示。

**Q4：高风险词命中了，但严重度没动。**
A：检查 `alert.severity` 是否已经处于上限/下限。`_adjust_alert_severity()` 在边界处不抛错也不变化。同时确认 `classifier_min_confidence`（`schema.py:751`）— 单帧快路径要求置信度 ≥ 0.9，普通告警链路不强制，但低置信度结果可能被其他过滤层吞掉。

**Q5：CPU 模式下相机 FPS 暴跌。**
A：参考 §6。短期方案：把 `min_anomaly_score_to_classify` 拉高到 0.7 以上，让分类只在很确信的告警上跑；中期方案：换 GPU；长期方案：把 `classify()` 调用挪到独立 executor（当前未做，需 PR）。

---

> 维护者：第一次按本文档启用后，请补 §6 的 TBD 实测数据 + 在 §8 添加你踩到的新坑。
