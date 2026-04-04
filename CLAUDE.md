# Argus — Development Notes

## Python Environment

- **Python**: `C:\Users\here\AppData\Local\Programs\Python\Python311\python.exe` (3.11, system-level, no venv)
- **Package**: `argus` is installed as **editable** (`pip install -e .`)，指向当前工作目录
- **Worktree 注意**: git worktree 会自动被识别为 editable install 的源路径，**不需要重新 `pip install -e .`**。新增的 `.py` 文件会被直接导入，无需任何额外操作。如果遇到 `ModuleNotFoundError`，先重试一次（可能是 `__pycache__` 缓存），不要急着跑 `pip install`。

## 常用命令

```bash
# 运行测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_frame_quality.py -v

# 启动应用
python -m argus

# 安装开发依赖（仅首次）
pip install -e ".[dev]"
```

## Shell Environment

- Shell: Git Bash on Windows (`/usr/bin/bash`)
- 使用 Unix 风格路径（`/c/Users/...`），不用 Windows 反斜杠
- `python` / `pip` 在 PATH 中，直接调用即可
- 不需要 `source activate` 或 conda 等虚拟环境操作

## Project Structure

- 源码: `src/argus/`
- 测试: `tests/` (根目录) + `tests/unit/`
- 配置: `configs/default.yaml`
- 构建: `pyproject.toml` (hatchling backend)

## Testing

- 框架: pytest + pytest-asyncio
- 已知预存失败: `test_dashboard.py::TestCompositeGeneration::test_generate_composite_missing_file` — ultralytics monkey-patch `cv2.imread` 导致，与业务逻辑无关
- 运行全量测试约 30 秒
