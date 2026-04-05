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
