# Argus — Development Notes

## Python Environment

- **Python**: 3.11.9, 系统级安装 `C:\Users\here\AppData\Local\Programs\Python\Python311\python.exe`，**没有 venv/conda**
- **pip**: 24.0，路径 `C:\Users\here\AppData\Local\Programs\Python\Python311\Scripts\pip`
- **Package**: `argus` 已 editable install (`pip install -e .`)，指向当前工作目录
- **Worktree 注意**: git worktree 会自动被识别为 editable install 的源路径，**不需要重新 `pip install -e .`**。新增的 `.py` 文件会被直接导入，无需任何额外操作。如果遇到 `ModuleNotFoundError`，先重试一次（可能是 `__pycache__` 缓存），不要急着跑 `pip install`。
- **推理运行时**: PyTorch 2.11.0+cpu（无 CUDA/GPU）, OpenVINO 2026.0.0
- **无 pre-commit hooks**，无 ruff/black/mypy 全局安装（ruff 仅在 `[dev]` 可选依赖中）

## Shell Environment

- Shell: Git Bash on Windows (`/usr/bin/bash`)
- 使用 Unix 风格路径（`/c/Users/...`），不用 Windows 反斜杠
- `python` / `pip` 直接在 PATH 中，无需激活任何环境
- 不需要 `source activate`、`conda activate`、`.venv\Scripts\activate` 等操作

## 常用命令

```bash
# 运行全量测试（~30秒）
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_frame_quality.py -v

# 启动应用
python -m argus

# 代码格式检查（需先 pip install -e ".[dev]"）
ruff check src/ tests/

# 安装开发依赖（仅首次）
pip install -e ".[dev]"
```

## Project Structure

- 源码: `src/argus/`（src layout）
- 测试: `tests/` (根目录) + `tests/unit/`
- 配置: `configs/default.yaml`
- 构建: `pyproject.toml` (hatchling backend)
- pytest 配置在 `pyproject.toml` 的 `[tool.pytest.ini_options]`
- ruff 配置在 `pyproject.toml` 的 `[tool.ruff]`，line-length=100, target py311

## Testing

- 框架: pytest + pytest-asyncio（asyncio_mode = "auto"）
- 已知预存失败: `test_dashboard.py::TestCompositeGeneration::test_generate_composite_missing_file` — ultralytics monkey-patch `cv2.imread` 导致，与业务逻辑无关
- 运行全量测试约 30 秒

## 踩坑记录

### ❌ 禁止使用 Starlette `BaseHTTPMiddleware`

**绝对不要** 在本项目中使用 `starlette.middleware.base.BaseHTTPMiddleware`。

**原因**: `BaseHTTPMiddleware` 会将 `StreamingResponse`（如 MJPEG 视频流）的每一帧通过内部 asyncio queue 中转，阻塞事件循环直到整个流结束（最长 30 分钟）。多层 `BaseHTTPMiddleware` 叠加后，所有其他 HTTP 请求完全无法处理，导致前端切页后全部请求超时。

**正确做法**: 所有 middleware 必须实现为**纯 ASGI middleware**（`__call__(self, scope, receive, send)`），直接传递 `send`/`receive` 给下游 app，不拦截不缓冲 response body。

**参考**:
- https://github.com/encode/starlette/discussions/1729
- https://github.com/encode/starlette/issues/1012
- 本项目修复提交: `3d1ed38`

```python
# ✅ 正确: 纯 ASGI middleware
class MyMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # 在这里做检查、修改 scope 等
            pass
        await self.app(scope, receive, send)  # 直接传递, 不缓冲

# ❌ 错误: BaseHTTPMiddleware 会缓冲 StreamingResponse body
class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)  # 内部通过 queue 中转 body
        return response
```

### ❌ ORM 新增字段必须同步添加 `_auto_migrate`

在 `src/argus/storage/models.py` 中给任何表新增 `mapped_column` 后，**必须** 同时在 `src/argus/storage/database.py` 的 `_auto_migrate()` 方法中添加对应的 `ALTER TABLE ADD COLUMN` 条目，否则已有数据库会触发 `sqlite3.OperationalError: no such column` → HTTP 500。

### ❌ 所有摄像头协议都必须注册到 go2rtc

新增摄像头协议时（如 USB、ONVIF），**必须** 在 `go2rtc_manager.py` 的 `sync_cameras()` 和 `app.py` 的 lifespan 中同步注册。未注册的摄像头会回退到 MJPEG `<img>` 标签，每个占一个 HTTP 长连接（最长 30 分钟），多个摄像头直接吃满浏览器的 6 连接/origin 上限，导致所有后续请求排队超时。使用 `usb_to_go2rtc_source()` 将 USB 设备索引转为 `ffmpeg:device?video=N#video=h264` 格式。

### ❌ MJPEG 流编码必须使用专用线程池

`cv2.imencode()` 等 CPU 密集操作 **禁止** 使用 `asyncio.to_thread()`（会占满默认线程池），必须使用 `loop.run_in_executor(_STREAM_EXECUTOR, ...)` 指定专用 `ThreadPoolExecutor`，避免阻塞普通 API 请求。见 `src/argus/dashboard/routes/cameras.py` 中的 `_STREAM_EXECUTOR`。
