# Argus — Development Notes

## Python Environment

- 当前工作区使用仓库内 `.venv`。
- `.venv` 基于系统 Python 3.11.9 创建，底层解释器为 `C:\Users\here\AppData\Local\Programs\Python\Python311\python.exe`。
- 不使用 conda；如需重新初始化环境，优先使用 `python -m venv .venv`。
- 首次拉起环境建议执行 `pip install -e ".[dev]"`。

## Shell Environment

- 当前用户经常在 PowerShell 中激活 `.venv\Scripts\Activate.ps1`。
- 如果使用 Git Bash，也可以继续使用 Unix 风格路径，但以 `.venv` 为准，不再假设“没有 venv”。
- 不需要 `conda activate`。

## 常用命令

```powershell
# 激活环境
.\.venv\Scripts\Activate.ps1

# 安装开发依赖
pip install -e ".[dev]"

# 启动后端
python -m argus --config configs/default.yaml

# 运行全量测试
python -m pytest tests/ -v

# 运行前端开发环境
cd web
npm install
npm run dev

# 构建前端
cd web
npm run build
```

## Project Structure

- 后端源码：`src/argus/`
- 前端源码：`web/src/`
- 默认配置：`configs/default.yaml`
- 测试：`tests/unit/` 与 `tests/integration/`
- 运行数据：`data/`

## 当前实现速记

- FastAPI 应用在 `src/argus/dashboard/app.py` 中装配 API、WebSocket、中间件和前端静态资源。
- 前端当前是 7 个主路由（Overview / Cameras / Alerts / Reports / Models / System / Replay），很多管理能力集中在模型页和系统页的标签中。
- go2rtc 是当前流媒体链路的重要组成部分，USB 摄像头路径尤其依赖它的重定向逻辑。
- 默认配置中分类器、分割器、跨摄像头和自动重训练通常关闭。

## Testing

- 测试框架：pytest + pytest-asyncio。
- 运行方式：`python -m pytest tests/ -v`。
- 修改共享检测链、Dashboard API、模型管理或存储逻辑时，优先补或跑对应的 targeted tests。

## 踩坑记录

每条雷区都对应一个 lint/集成测试守门，CI 失败时直接跳到对应测试看断言信息。

### ❌ 禁止使用 Starlette `BaseHTTPMiddleware`

**绝对不要** 在本项目中使用 `starlette.middleware.base.BaseHTTPMiddleware`。

**原因**：它会缓冲流式响应，MJPEG 等长连接场景下容易阻塞其他请求。

**正确做法**：所有 middleware 必须实现为纯 ASGI middleware，直接透传 `scope`、`receive`、`send`。

**守门测试**：`tests/lint/test_no_base_http_middleware.py`

### ❌ ORM 新增字段必须同步更新自动迁移逻辑

在 `src/argus/storage/models.py` 中新增字段后，必须同时检查 `src/argus/storage/database.py` 中的 `_AUTO_MIGRATIONS` 列表（模块级常量），否则已有数据库可能出现缺列错误。

**守门测试**：`tests/lint/test_orm_auto_migrate.py` —— 同时校验 `_AUTO_MIGRATIONS` 中的每条都对应一个真实 ORM 列、跑空 DB 做 round-trip、丢失列后能恢复。

### ❌ 新摄像头协议必须同步接入 go2rtc 注册路径

新增摄像头协议时，必须同时检查 go2rtc 注册逻辑和应用启动阶段，否则浏览器侧会回退到低效的视频路径。

**守门测试**：尚未自动化（计划中），新协议 PR 需手动确认 `__main__.py` 的 `start_and_register_cameras()` 已覆盖。

### ❌ MJPEG 编码必须使用专用线程池

`cv2.imencode()` 等 CPU 密集操作不要丢到默认线程池；应继续使用摄像头路由中的 `_STREAM_EXECUTOR` 专用线程池，避免拖慢普通 API。

**守门测试**：`tests/lint/test_mjpeg_dedicated_executor.py` —— 校验 `_STREAM_EXECUTOR` 存在 + 禁止 `asyncio.to_thread(cv2.imencode, ...)` 反模式。

### ❌ OpenVINO IR 必须是静态形状

anomalib 默认导出可能产生动态 batch/spatial 维度的 IR，CPU 推理时直接 `Broadcast Check failed, Value -1 not in range`。`ModelTrainer.export()` 必须传 `input_size=(H, W)` + `onnx_kwargs={'dynamo': False, 'dynamic_axes': {}}`。

**守门测试**：`tests/unit/test_trainer_export.py::TestAssertOpenvinoStatic` —— `_assert_openvino_static()` 在导出后立刻校验 `partial_shape.is_static`，失败时点名 trainer 而不是把锅甩给后续推理。
