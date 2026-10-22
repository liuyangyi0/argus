"""Generate a compact source index for documentation work.

The index intentionally avoids importing project modules. Python signatures are
read through ``ast`` and frontend/API hints are extracted with lightweight
regular expressions, so the script is safe to run without cameras, models, or a
configured database.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT = Path("docs/source-index.md")
TREE_ROOTS = [
    Path("src/argus"),
    Path("web/src"),
    Path("configs"),
    Path("scripts"),
    Path("tests"),
    Path("docs"),
]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".benchmarks",
    ".idea",
    "node_modules",
    "dist",
    "__pycache__",
    "openvino_cache",
    "data",
}


@dataclass(frozen=True)
class PythonMember:
    kind: str
    name: str
    signature: str
    line: int
    children: tuple["PythonMember", ...] = ()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_tree_files(root: Path, max_depth: int) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if len(rel_parts) > max_depth:
            continue
        if path.is_file():
            files.append(path)
    return files


def _render_tree(repo_root: Path, max_depth: int) -> list[str]:
    lines: list[str] = []
    for tree_root in TREE_ROOTS:
        abs_root = repo_root / tree_root
        files = _iter_tree_files(abs_root, max_depth=max_depth)
        if not files:
            continue
        lines.append(f"{tree_root.as_posix()}/")
        for file_path in files:
            rel = file_path.relative_to(abs_root)
            indent = "  " * (len(rel.parts) - 1)
            lines.append(f"{indent}- {rel.as_posix()}")
        lines.append("")
    return lines


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({args}){returns}"


def _class_signature(node: ast.ClassDef) -> str:
    bases = [ast.unparse(base) for base in node.bases]
    suffix = f"({', '.join(bases)})" if bases else ""
    return f"class {node.name}{suffix}"


def _method_members(node: ast.ClassDef) -> tuple[PythonMember, ...]:
    members: list[PythonMember] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.append(
                PythonMember(
                    kind="method",
                    name=child.name,
                    signature=_signature(child),
                    line=child.lineno,
                )
            )
    return tuple(members)


def _python_members(path: Path) -> tuple[PythonMember, ...]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8-sig")
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError:
        return ()

    members: list[PythonMember] = []
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            members.append(
                PythonMember(
                    kind="class",
                    name=node.name,
                    signature=_class_signature(node),
                    line=node.lineno,
                    children=_method_members(node),
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.append(
                PythonMember(
                    kind="function",
                    name=node.name,
                    signature=_signature(node),
                    line=node.lineno,
                )
            )
    return tuple(members)


def _render_python_signatures(repo_root: Path) -> list[str]:
    lines: list[str] = []
    python_files = sorted((repo_root / "src/argus").rglob("*.py"))
    for path in python_files:
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        members = _python_members(path)
        if not members:
            continue
        lines.append(f"### {_relative(path, repo_root)}")
        for member in members:
            lines.append(f"- L{member.line}: `{member.signature}`")
            for child in member.children:
                lines.append(f"  - L{child.line}: `{child.signature}`")
        lines.append("")
    return lines


def _route_prefixes(repo_root: Path) -> dict[str, str]:
    app_path = repo_root / "src/argus/dashboard/app.py"
    if not app_path.exists():
        return {}
    text = app_path.read_text(encoding="utf-8")
    imports = dict(
        re.findall(
            r"from argus\.dashboard\.routes\.([a-z_]+) import router as ([a-z_]+_router)",
            text,
        )
    )
    alias_to_module = {alias: module for module, alias in imports.items()}
    prefixes: dict[str, str] = {}
    for alias, prefix in re.findall(
        r"app\.include_router\(([a-z_]+_router), prefix=\"([^\"]+)\"",
        text,
    ):
        module = alias_to_module.get(alias)
        if module:
            prefixes[module] = prefix
    return prefixes


def _route_functions(path: Path) -> list[tuple[str, str, int, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    routes: list[tuple[str, str, int, str]] = []
    pending: tuple[str, str, int] | None = None
    for idx, line in enumerate(lines, start=1):
        match = re.match(
            r"\s*@router\.(get|post|put|patch|delete)\(\s*\"([^\"]*)\"",
            line,
        )
        if match:
            pending = (match.group(1).upper(), match.group(2), idx)
            continue
        if pending is not None:
            func = re.match(r"\s*(?:async\s+def|def)\s+([a-zA-Z_][\w]*)\(", line)
            if func:
                method, route, line_no = pending
                routes.append((method, route, line_no, func.group(1)))
                pending = None
    return routes


def _render_api_routes(repo_root: Path) -> list[str]:
    prefixes = _route_prefixes(repo_root)
    route_dir = repo_root / "src/argus/dashboard/routes"
    lines: list[str] = []
    for path in sorted(route_dir.glob("*.py")):
        module = path.stem
        routes = _route_functions(path)
        if not routes:
            continue
        prefix = prefixes.get(module, f"/api/{module.replace('_', '-')}")
        lines.append(f"### {prefix} ({_relative(path, repo_root)})")
        for method, route, line_no, func_name in routes:
            full_path = f"{prefix}{route}"
            lines.append(f"- `{method} {full_path}` -> `{func_name}` (L{line_no})")
        lines.append("")
    return lines


def _render_frontend_routes(repo_root: Path) -> list[str]:
    router_path = repo_root / "web/src/router/index.ts"
    if not router_path.exists():
        return []
    text = router_path.read_text(encoding="utf-8")
    lines = ["### web/src/router/index.ts"]
    for match in re.finditer(
        r"path:\s*'([^']+)'.{0,180}?name:\s*'([^']+)'.{0,240}?title:\s*'([^']+)'",
        text,
        flags=re.DOTALL,
    ):
        route_path, name, title = match.groups()
        lines.append(f"- `{route_path}` -> `{name}` / {title}")
    lines.append("")
    return lines


def _render_frontend_api_clients(repo_root: Path) -> list[str]:
    api_dir = repo_root / "web/src/api"
    if not api_dir.exists():
        return []
    lines: list[str] = []
    for path in sorted(api_dir.glob("*.ts")):
        text = path.read_text(encoding="utf-8")
        functions = re.findall(
            r"export\s+(?:async\s+)?function\s+([a-zA-Z_][\w]*)\s*\(([^)]*)\)",
            text,
        )
        consts = re.findall(
            r"export\s+const\s+([a-zA-Z_][\w]*)\s*=\s*(?:async\s*)?\(([^)]*)\)",
            text,
        )
        if not functions and not consts:
            continue
        lines.append(f"### {_relative(path, repo_root)}")
        for name, args in functions + consts:
            compact_args = " ".join(args.split())
            lines.append(f"- `{name}({compact_args})`")
        lines.append("")
    return lines


def generate(repo_root: Path, max_tree_depth: int) -> str:
    sections: list[str] = [
        "# Argus 源码索引",
        "",
        "> 由 `python scripts/extract_source_index.py --output docs/source-index.md` 生成。",
        "> 该文件用于压缩阅读上下文，写架构或模块文档前可先查看这里，再按需打开源码。",
        "",
        "## 目录树",
        "",
        "```text",
        *_render_tree(repo_root, max_depth=max_tree_depth),
        "```",
        "",
        "## Python 类/函数签名",
        "",
        *_render_python_signatures(repo_root),
        "## FastAPI 路由",
        "",
        *_render_api_routes(repo_root),
        "## 前端路由",
        "",
        *_render_frontend_routes(repo_root),
        "## 前端 API 客户端",
        "",
        *_render_frontend_api_clients(repo_root),
    ]
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tree-depth", type=int, default=3)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate(repo_root, args.max_tree_depth), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
