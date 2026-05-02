# SAM2 分割器启用 Checklist

> 适用版本：argus 0.1.0（FastAPI 内部 0.2.0）。
> 模块代号：D2（Instance Segmentation）。
> 目标读者：第一次启用 SAM2 分割的运维 / 工程师，按编号步骤可机械执行。

---

## 1. 模块定位

SAM2 分割器（`src/argus/anomaly/segmenter.py` 中的 `InstanceSegmenter`）在异常检测命中告警之后被调用，输入是当帧 BGR 图像 + 异常热力图上的峰值点（`extract_peak_points`），输出是每个目标的二值 mask、bbox、面积和 centroid。最终结果不会回写主管线分数，只是把 `segmentation_count`、`segmentation_total_area_px`、`segmentation_objects`（JSON 列表）挂到 `Alert` 对象上，由告警分发器写入数据库 `alerts` 表。

为什么默认关闭：

- 依赖体积大（PyTorch + sam2 包 + Hiera 权重，几百 MB 到 GB 级）。
- GPU 推理才有意义，CPU 上单帧延迟会拖慢主链路。
- 它不参与 severity 判定，对核心告警链路是"增强信息"而非必要环节，所以默认 `segmenter.enabled=false`。

相对纯 anomaly heatmap 的额外价值：

- 把"哪里异常"从一团热力区收敛成"具体物体的轮廓 / 面积 / bbox"，前端 `AlertDetailPanel.vue` 会显示"X 个对象"并可叠加 mask。
- 对告警分级和回放更有解释性，便于人工复核异物。
- 失败时自动回退到基于 OpenCV 阈值 + 轮廓的 fallback（即使 sam2 没装、模型加载失败、超时也不会让管线崩）。

---

## 2. 依赖清单

### 2.1 Python 包

| 包 | 版本 | 来源 | 备注 |
| --- | --- | --- | --- |
| `sam2` | `>=1.0` | `pyproject.toml` 的 optional extra `segment` | 不在默认依赖里 |
| `torch` | 与 sam2 兼容（一般 ≥ 2.3） | sam2 自带依赖会拉，但建议先按本机 CUDA 版本手动装 | 决定 GPU/CPU 推理路径 |
| `opencv-python` | `>=4.10.0` | 已是默认依赖 | 用于 BGR↔RGB、轮廓 fallback |
| `numpy` | 项目默认 | 已是默认依赖 | — |
| `structlog` | 项目默认 | 已是默认依赖 | 输出 `segmenter.*` 日志 |

注意：`pyproject.toml` 里只是声明 `segment = ["sam2>=1.0"]`，并不是安装就开启，**还要在 yaml 配置里把 `segmenter.enabled` 置 true**。

### 2.2 模型权重

`InstanceSegmenter.load()` 通过 `SAM2ImagePredictor.from_pretrained(model_id)` 走的是 Hugging Face Hub，仓库里**不会**自带 `.pt` 文件。`model_size` 与 HF repo id 的对应关系直接写死在源码里：

| `model_size` | HF repo id | 大致体积 |
| --- | --- | --- |
| `tiny` | `facebook/sam2-hiera-tiny` | ~150 MB |
| `small` | `facebook/sam2-hiera-small` | ~180 MB |
| `base_plus` | `facebook/sam2-hiera-base-plus` | ~320 MB |
| `large` | `facebook/sam2-hiera-large` | ~900 MB |

下载方式（任选其一）：

1. **联网自动下载**：第一次 `load()` 时自动从 HF Hub 拉取并缓存到 `~/.cache/huggingface/hub`。  
2. **离线机器**：先在有网机器执行  
   ```bash
   huggingface-cli download facebook/sam2-hiera-small --local-dir ~/.cache/huggingface/hub/models--facebook--sam2-hiera-small
   ```
   再把整个 `~/.cache/huggingface/hub` 拷贝到目标机器的同位置。也可设置 `HF_HOME` 自定义缓存目录。
3. **私有镜像**：通过 `HF_ENDPOINT=https://hf-mirror.com` 等环境变量切换镜像。

> 体积只是估算，以 HF 仓库实际显示为准。

### 2.3 硬件要求

- **GPU 强烈推荐**：`load()` 里通过 `torch.cuda.is_available()` 选 `cuda`，否则掉到 `cpu`。CPU 上 SAM2 单帧十几秒级别，没意义。
- **显存（推理时）**：
  - tiny / small：≥ 4 GB（与主链路 backbone、YOLO 等模型共用同一块卡时建议 ≥ 6 GB）。
  - base_plus / large：≥ 8 GB / ≥ 12 GB（具体 TBD，见第 6 节实测方法）。
- **CPU**：fallback 路径走 OpenCV 轮廓，不挑 CPU。
- **磁盘**：HF 缓存目录至少预留 1.5 GB。

---

## 3. 配置项

### 3.1 `configs/default.yaml` 中的 `segmenter:` 段

```yaml
segmenter:
  enabled: false
  model_size: small
  max_points: 5
  min_anomaly_score: 0.7
  min_mask_area_px: 100
  timeout_seconds: 10.0
```

字段语义（来源：`src/argus/config/schema.py::SegmenterConfig`）：

| 字段 | 默认值 | 合法范围 | 调整建议 |
| --- | --- | --- | --- |
| `enabled` | `false` | `true` / `false` | 启用本模块的总开关。 |
| `model_size` | `small` | `tiny` / `small` / `base_plus` / `large`（regex 校验） | GPU 显存紧张就 `tiny` 或 `small`；显存充裕、要更准确边界用 `base_plus` 或 `large`。**运行时不可热切换**（注释写明改 model_size 必须重启管线）。 |
| `max_points` | `5` | `1` ~ `20` | 单帧最多分割多少个异常峰。复杂场景多目标可调到 `8~10`，单目标场景 `3` 足矣。 |
| `min_anomaly_score` | `0.7` | `0.1` ~ `0.99` | 触发分割的最小异常分。低于此分的告警不调用 SAM2，节省算力。 |
| `min_mask_area_px` | `100` | `10` ~ `100000` | 小于该面积的 mask 直接丢弃，过滤噪点。**支持运行时热更新**（`update_runtime_params`）。 |
| `timeout_seconds` | `10.0` | `1.0` ~ `60.0` | 单帧 SAM2 推理超时，超时返回空结果不阻塞主链路。**支持运行时热更新**。 |

### 3.2 跨配置依赖

- 必须先有 anomaly 检测产出 `anomaly_result.anomaly_map`（管线代码 `pipeline.py` 1720 行附近的判断），否则没有峰值可分割。换言之，至少要有一个开启了异常检测（baseline + scorer）的摄像头。
- 触发分割还有两个前置条件，源自 `pipeline.py`：
  1. 该帧已生成 `Alert` 对象（即 grader 已判定为告警）。
  2. `anomaly_result.anomaly_score >= segmenter.min_anomaly_score`。
- `classifier`、`cross_camera`、`retraining` 与本模块互相独立，不需要先开。
- 数据库迁移：`segmentation_count` / `segmentation_total_area_px` / `segmentation_objects` 三列由 `src/argus/storage/database.py` 的 `_AUTO_MIGRATIONS` 自动 ALTER TABLE 添加，老库重启后台即会补列，不需要手工 SQL。

---

## 4. 启用步骤

> 假设当前已经在 `.venv` 里能正常 `python -m argus --config configs/default.yaml` 启动后端。

1. **确认或安装 PyTorch（按本机 CUDA 版本）**  
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   如果输出 `False` 但本机有 GPU，先按 [pytorch.org](https://pytorch.org) 选择匹配 CUDA 版本的 wheel 装 `torch`。

2. **安装 segment extra**  
   ```powershell
   pip install -e ".[segment]"
   ```
   该 extra 只声明 `sam2>=1.0`；如果 sam2 反向依赖会重装 torch，请上一步先锁好 torch 版本。

3. **预下载模型权重（联网机器）**  
   ```powershell
   python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-small')"
   ```
   等待 HF Hub 下载完成。离线机器请按第 2.2 节拷贝缓存目录。

4. **修改配置**  
   编辑 `configs/default.yaml` 或自己的 yaml，把  
   ```yaml
   segmenter:
     enabled: true
     model_size: small
   ```
   写进去。其他字段保持默认即可，调优放后面。

5. **重启后端**  
   ```powershell
   python -m argus --config configs/default.yaml
   ```
   注意：现有运行中的 `Pipeline` 实例**不会**因为热切 `segmenter.enabled=true` 就回填一个 segmenter（见 `dashboard/routes/config.py::get_segmenter_config` 注释和 `pipeline.py` 2474 行附近的 `_segmenter_config is None` 判断）。**要让所有摄像头生效必须重启服务**。

6. **观察启动日志**  
   预期看到：
   - `segmenter.sam2_loaded` 关键字（`logger.info`），并带 `device=cuda` 或 `cpu`。
   - 紧接着 `pipeline.segmenter_configured`。
   - 失败回退会出现 `segmenter.sam2_not_installed` 或 `segmenter.sam2_load_failed`。

7. **触发一次告警验证**  
   走正常异常触发流程，等命中一次告警后查看数据库或前端：`alerts.segmentation_count > 0` 即说明流程贯通。

---

## 5. UI 入口

- **配置面板**：前端路由 `/system`（`web/src/views/System.vue`），里面的 `分割器` Tab 由 `web/src/components/system/SegmenterPanel.vue` 渲染。可视化展示 `enabled`、`model_size`、`max_points`、`min_anomaly_score`、`min_mask_area_px`、`timeout_seconds`，并提供 toggle + 4 个运行时可调字段的编辑保存。后端 API：`GET/POST /api/config/segmenter`。
- **告警详情**：前端 `web/src/components/alerts/AlertDetailPanel.vue` 在告警详情面板里读取 `selectedAlert.segmentation_count`，显示"X 个对象"。
- **告警分级**：分割结果**不参与** severity 判定（`src/argus/alerts/grader.py` 中 severity 由分数 + classifier 决定，segmentation_* 只作为附加字段挂在 `Alert` 上）。`category`、`severity_adjusted_by_classifier` 仍由 classifier 路径写入，分割器不会改动告警等级。

---

## 6. 性能影响

- **执行时机**：分割发生在 grader 已经生成告警之后（`pipeline.py` 1720 行的 `if alert is not None` 分支），但仍在主帧处理协程里。也就是说，**它是同步管线的一部分**，会被计入该帧的总耗时（管线对每个 stage 都记 `StageResult`，本 stage 名为 `segmenter`）。  
  保护机制：超时通过专用 `ThreadPoolExecutor`（`thread_name_prefix="sam2"`）+ `future.result(timeout=timeout_seconds)` 兜底，超时返回空结果继续；任何异常都进 `except` 不阻塞下一帧。
- **单帧分割延迟**：取决于 `model_size`、显存、prompt 点数：
  - `tiny` / `small` + 单 GPU + ≤5 个点：经验 100~400 ms 量级（**TBD，本仓库未提供基准；测量见第 7 节自检命令**）。
  - `base_plus` / `large` + 多点：可能逼近 `timeout_seconds=10`，建议把 `max_points` 调小或上更大显存。
  - CPU 路径：秒级，强烈建议直接走 fallback 或者根本别开。
- **GPU 显存**：
  - `tiny` / `small`：约 1.5–3 GB（SAM2 image predictor 本身），**TBD，请用第 7 节命令实测**。
  - `base_plus` / `large`：4 GB+，最大 8 GB+。
  - 与 backbone（PaDiM/PatchCore/EfficientAD 等）共卡时，加起来不要超过物理显存的 80%。
- **对主链路吞吐影响**：分割会把异常告警帧的处理时间拉长（典型 +200~500 ms），未告警帧不受影响。如果系统经常打满帧率，建议提高 `min_anomaly_score`（少分割）或减小 `max_points`。

---

## 7. 自检命令

### 7.1 仅校验包和模型加载

```powershell
.\.venv\Scripts\Activate.ps1
python -c "from argus.anomaly.segmenter import InstanceSegmenter; s = InstanceSegmenter(model_size='small'); s.load(); print('sam2_available =', s._sam2_available)"
```
- 期望：打印 `sam2_available = True`，stderr/日志含 `segmenter.sam2_loaded` + `device=cuda`（或 `cpu`）。
- 失败信号：`sam2_available = False` 或 `segmenter.sam2_not_installed` / `segmenter.sam2_load_failed`，多半是 sam2 未装或模型未下完。

### 7.2 端到端跑一遍 fake heatmap

```powershell
python -c "import numpy as np; from argus.anomaly.segmenter import InstanceSegmenter, extract_peak_points; frame = np.zeros((480, 640, 3), dtype=np.uint8); frame[100:200, 200:300] = 255; hm = np.zeros((480, 640), dtype=np.float32); hm[150, 250] = 0.9; pts = extract_peak_points(hm, max_points=3, min_score=0.7); seg = InstanceSegmenter(model_size='small'); seg.load(); r = seg.segment(frame, pts); print('num_objects=', r.num_objects, 'total_area_px=', r.total_area_px)"
```
- 期望：`num_objects >= 1`，`total_area_px > 0`。

### 7.3 单元测试

```powershell
python -m pytest tests/unit/test_segmenter.py -v
```
- 期望全部通过；这套测试同时覆盖了 SAM2 路径（mock）和 contour fallback。

### 7.4 预期日志关键字（`structlog`）

| 关键字 | 含义 |
| --- | --- |
| `segmenter.sam2_loaded` | SAM2 模型加载成功，包含 `device`、`model_id` |
| `segmenter.sam2_not_installed` | sam2 包没装，自动走 fallback |
| `segmenter.sam2_load_failed` | sam2 装了但权重加载失败（多半是网络 / 缓存） |
| `segmenter.timeout` | 单帧推理超过 `timeout_seconds` |
| `segmenter.segment_error` | 推理异常（非超时） |
| `pipeline.segmenter_configured` | 启动时管线挂上了 segmenter |
| `pipeline.segmented` | DEBUG 级，告警命中并出了分割结果 |
| `pipeline.segmenter_error` | 管线层 catch 的异常，会累计 `_segmenter_consecutive_failures` |

### 7.5 数据库字段写入时机

- 模型字段（`src/argus/storage/models.py`）：`segmentation_count`、`segmentation_total_area_px`、`segmentation_objects`（JSON 文本，不存 mask 像素，只有 bbox/area/centroid/conf）。
- 写入路径：`pipeline.py` 把字段挂到 `alert` 对象 → `argus/alerts/dispatcher.py` 取出 → `argus/storage/database.py::save_alert` 把 `segmentation_objects` 用 `json.dumps` 编码后入库。
- **只有当 `seg_result.num_objects > 0` 时才会写**（见 `pipeline.py` 1759 行）。否则仍是 `None`，前端 `AlertDetailPanel` 也会隐藏分割面板。
- 自动迁移：旧库启动时由 `database.py::_AUTO_MIGRATIONS` 自动 `ALTER TABLE alerts ADD COLUMN`，无需手工 SQL。

校验 SQL（任意 SQLite 工具）：

```sql
SELECT alert_id, segmentation_count, segmentation_total_area_px,
       json_array_length(segmentation_objects) AS obj_count
FROM alerts
WHERE segmentation_count IS NOT NULL
ORDER BY timestamp DESC
LIMIT 10;
```

---

## 8. 常见问题（FAQ）

**Q1. 装好了 sam2，前端 toggle 也开了，为什么数据库里 `segmentation_count` 还是 NULL？**  
最常见原因是热切了开关但**没重启服务**：现有 `Pipeline` 实例只在创建时根据 `segmenter_config.enabled` 决定是否实例化 `InstanceSegmenter`。前端面板会显示这种情况下的"需要重启"提示。重启后端即可。

**Q2. 启动看到 `segmenter.sam2_load_failed` 怎么办？**  
看 `error` 字段：通常是 HF 拉权重失败（无网 / 代理）或 PyTorch 与 sam2 版本不兼容。处理：手动 `huggingface-cli download` 把权重放到缓存目录（见 2.2），或检查 `python -c "import torch; print(torch.__version__)"` 与 sam2 README 的兼容矩阵。失败不会让服务崩，但会回退到 contour fallback，分割质量大打折扣。

**Q3. 想关掉 SAM2 但保留 fallback 怎么做？**  
把 `segmenter.enabled` 置 `false` 即可——fallback 路径只在 `enabled=true` 且 `_sam2_available=False` 时启用，单纯关 enabled 不会留下 fallback。如果只想要 fallback，可以保持 `enabled=true` 且**不安装 sam2**：`InstanceSegmenter.load()` 会捕获 `ImportError` 自动走 contour 路径（这是设计意图，但生产环境不推荐，因为分割质量明显差）。

**Q4. `timeout_seconds` 经常触发警告，是不是 GPU 不够？**  
先看 `segmenter.sam2_loaded` 里的 `device`：如果是 `cpu`，无解，必须上 GPU。如果已经是 `cuda`：(1) 把 `model_size` 从 `large` 换 `small`；(2) 把 `max_points` 调小到 `3`；(3) 把 `min_anomaly_score` 提到 `0.85` 以上，减少触发频率。`timeout_seconds` 和 `min_mask_area_px` 都支持热更新，可以在 `/system` → `分割器` Tab 直接调整保存。

**Q5. 我能不能不写数据库，只在前端展示分割？**  
不能。当前实现里 `pipeline.py` 把字段挂在 `alert` 上，`dispatcher.py` 通过 `save_alert` 写库，前端 `AlertDetailPanel` 是从告警接口（`/api/alerts/...`）读 `segmentation_count`/`segmentation_objects` 渲染的，没有数据库就没有展示。`segmentation_objects` 只存 bbox / area / centroid / confidence，不会存 mask 像素，体积可控。
