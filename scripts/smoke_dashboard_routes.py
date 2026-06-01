"""Smoke the built Dashboard served by ``python -m argus``.

This is a fast local-development guard for the user-visible entry points in the
core loop. It starts Argus with ``configs/default.yaml --dev-video``, waits for
the generated file camera to produce frames, then checks that the main SPA
routes and key camera media endpoints are reachable through the same HTTP
server a user opens in the browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import httpx


CORE_DASHBOARD_ROUTES = [
    "/cameras",
    "/alerts",
    "/replay/ALT-smoke",
    "/replay/ALT-smoke/storyboard",
    "/models/baseline",
    "/models/training",
    "/models/registry",
    "/system/overview",
    "/system/config",
    "/system/degradation",
    "/reports",
]

BROWSER_ROUTE_EXPECTATIONS = {
    "/cameras": ["新增摄像头", "摄像头 ID"],
    "/alerts": ["告警中心", "总计"],
    "/replay/ALT-smoke": ["录像回放", "ALT-smoke"],
    "/replay/ALT-smoke/storyboard": ["多机位回放", "ALT-smoke"],
    "/models/baseline": ["基线数据"],
    "/models/training": ["待确认训练任务", "新建训练任务"],
    "/models/registry": ["模型总数", "模型版本"],
    "/system/overview": ["降级监控", "模型运行状态"],
    "/system/config": ["系统配置", "检测参数"],
    "/system/degradation": ["降级事件历史"],
    "/reports": ["报表统计", "告警总数"],
}


class DashboardSmokeFailure(RuntimeError):
    """Raised when the Dashboard route smoke cannot prove the route contract."""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, lines: int = 40) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _cleanup_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _find_headless_browser(explicit_path: str | None = None) -> str | None:
    if explicit_path:
        found = shutil.which(explicit_path) or explicit_path
        return found if Path(found).exists() else None

    names = ["chrome", "chrome.exe", "msedge", "msedge.exe", "chromium", "chromium-browser"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found

    candidates: list[Path] = []
    for env_name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        root = os.environ.get(env_name)
        if not root:
            continue
        candidates.extend([
            Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        ])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _dump_dom_with_browser(
    *,
    browser_path: str,
    url: str,
    user_data_dir: Path,
    timeout_s: float,
    virtual_time_ms: int,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(2):
        attempt_dir = user_data_dir / f"cdp-{attempt}-{time.monotonic_ns()}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        try:
            return _dump_dom_with_cdp(
                browser_path=browser_path,
                url=url,
                user_data_dir=attempt_dir,
                timeout_s=timeout_s,
                virtual_time_ms=virtual_time_ms,
            )
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.5)

    assert last_exc is not None
    raise DashboardSmokeFailure(
        f"browser CDP DOM dump failed for {url}: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def _dump_dom_with_cdp(
    *,
    browser_path: str,
    url: str,
    user_data_dir: Path,
    timeout_s: float,
    virtual_time_ms: int,
) -> str:
    debug_port = _free_port()
    cmd = [
        browser_path,
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-background-networking",
        "--disable-extensions",
        "--no-first-run",
        "--no-default-browser-check",
        "--blink-settings=imagesEnabled=false",
        "--remote-debugging-address=127.0.0.1",
        f"--remote-debugging-port={debug_port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={user_data_dir}",
        "about:blank",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        page_ws_url = _wait_for_cdp_page(debug_port, timeout_s=min(timeout_s, 30.0))
        return asyncio.run(_read_dom_via_cdp(
            page_ws_url,
            url=url,
            timeout_s=timeout_s,
            virtual_time_ms=virtual_time_ms,
        ))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def _wait_for_cdp_page(debug_port: int, *, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(  # noqa: S310 - local Chrome DevTools endpoint
                f"http://127.0.0.1:{debug_port}/json",
                timeout=1,
            ) as response:
                pages = json.loads(response.read().decode("utf-8", errors="replace"))
            for page in pages:
                if page.get("type") == "page" and page.get("webSocketDebuggerUrl"):
                    return str(page["webSocketDebuggerUrl"])
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise TimeoutError(f"Chrome DevTools page target not ready: {last_error}")


async def _read_dom_via_cdp(
    page_ws_url: str,
    *,
    url: str,
    timeout_s: float,
    virtual_time_ms: int,
) -> str:
    import websockets

    next_id = 0

    async with websockets.connect(page_ws_url, max_size=16 * 1024 * 1024) as ws:
        async def command(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
            nonlocal next_id
            next_id += 1
            command_id = next_id
            await ws.send(json.dumps({
                "id": command_id,
                "method": method,
                "params": params or {},
            }))
            while True:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_s))
                if message.get("id") == command_id:
                    if "error" in message:
                        raise RuntimeError(message["error"])
                    return message.get("result") or {}

        await command("Page.enable")
        await command("Runtime.enable")
        await command("Network.enable")
        await command("Network.setBlockedURLs", {
            "urls": [
                "*/api/cameras/*/stream*",
                "*/api/replay/*/stream*",
                "*/api/streaming/*",
            ],
        })
        await command("Page.navigate", {"url": url})

        load_deadline = time.monotonic() + min(timeout_s, 15.0)
        while time.monotonic() < load_deadline:
            try:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
            except asyncio.TimeoutError:
                break
            if message.get("method") == "Page.loadEventFired":
                break

        await asyncio.sleep(max(0.5, min(virtual_time_ms / 1000.0, 12.0)))
        try:
            await command("Page.stopLoading")
        except Exception:
            pass
        result = await command(
            "Runtime.evaluate",
            {
                "expression": "document.documentElement.outerHTML",
                "returnByValue": True,
            },
        )
        value = (result.get("result") or {}).get("value")
        if not isinstance(value, str) or not value:
            raise RuntimeError("CDP returned an empty DOM")
        return value


def _check_browser_dom_routes(args: argparse.Namespace, base_url: str, work_dir: Path) -> dict[str, Any]:
    if args.browser == "off":
        return {"status": "off", "routes_checked": []}

    browser_path = _find_headless_browser(args.browser_path)
    if not browser_path:
        if args.browser == "required":
            raise DashboardSmokeFailure(
                "headless browser required but Chrome/Edge/Chromium was not found"
            )
        return {"status": "skipped", "reason": "Chrome/Edge/Chromium not found", "routes_checked": []}

    checked: list[dict[str, Any]] = []
    for route, markers in BROWSER_ROUTE_EXPECTATIONS.items():
        profile_dir = work_dir / "browser-profiles" / route.strip("/").replace("/", "_")
        profile_dir.mkdir(parents=True, exist_ok=True)
        dom = _dump_dom_with_browser(
            browser_path=browser_path,
            url=f"{base_url}{route}",
            user_data_dir=profile_dir,
            timeout_s=args.browser_timeout,
            virtual_time_ms=args.browser_virtual_time_ms,
        )
        missing = [marker for marker in markers if marker not in dom]
        if missing:
            snippet = " ".join(dom.split())[:500]
            raise DashboardSmokeFailure(
                f"browser DOM missing markers for {route}: {missing}; snippet={snippet!r}"
            )
        if "/login" in dom and "登录" in dom:
            raise DashboardSmokeFailure(f"browser DOM reached login page for {route}")
        checked.append({
            "route": route,
            "markers": markers,
            "dom_bytes": len(dom.encode("utf-8", errors="replace")),
        })

    return {
        "status": "checked",
        "browser": browser_path,
        "routes_checked": checked,
    }


def _wait_for_camera(
    client: httpx.Client,
    *,
    camera_id: str,
    timeout_s: float,
    min_frames: int,
    process: subprocess.Popen,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise DashboardSmokeFailure(
                f"argus exited early with code {process.returncode}"
            )
        try:
            response = client.get("/api/cameras/json", timeout=3)
            response.raise_for_status()
            payload = response.json()
            rows = (payload.get("data") or {}).get("cameras") or []
            row = next(
                (item for item in rows if item.get("camera_id") == camera_id),
                None,
            )
            stats = row.get("stats") if row else {}
            if (
                row
                and row.get("connected")
                and row.get("running")
                and (stats or {}).get("frames_captured", 0) >= min_frames
            ):
                return row
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    suffix = f" Last error: {last_error}" if last_error else ""
    raise DashboardSmokeFailure(
        f"Timed out waiting for camera {camera_id} to produce frames.{suffix}"
    )


def run_dashboard_route_smoke(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    work_dir = Path(tempfile.mkdtemp(prefix="argus-dashboard-smoke-"))
    stdout_path = work_dir / "argus.stdout.log"
    stderr_path = work_dir / "argus.stderr.log"
    video_path = work_dir / "dev_camera.avi"
    port = args.port or _free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["ARGUS__DASHBOARD__PORT"] = str(port)
    cmd = [
        sys.executable,
        "-m",
        "argus",
        "--config",
        str(args.config),
        "--dev-video",
        "--dev-video-path",
        str(video_path),
    ]
    if args.camera:
        cmd.extend(["--camera", args.camera])

    proc: subprocess.Popen | None = None
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            with httpx.Client(base_url=base_url, follow_redirects=False) as client:
                camera_row = _wait_for_camera(
                    client,
                    camera_id=args.camera_id,
                    timeout_s=args.timeout,
                    min_frames=args.min_frames,
                    process=proc,
                )

                snapshot = client.get(f"/api/cameras/{args.camera_id}/snapshot")
                if snapshot.status_code != 200:
                    raise DashboardSmokeFailure(
                        f"snapshot failed: HTTP {snapshot.status_code}"
                    )
                content_type = snapshot.headers.get("content-type", "")
                if "image/jpeg" not in content_type:
                    raise DashboardSmokeFailure(
                        f"snapshot content-type mismatch: {content_type}"
                    )

                streaming = client.get(f"/api/streaming/{args.camera_id}")
                if streaming.status_code != 200:
                    raise DashboardSmokeFailure(
                        f"streaming info failed: HTTP {streaming.status_code}"
                    )
                streaming_data = (streaming.json().get("data") or {})
                expected_fallback = f"/api/cameras/{args.camera_id}/stream"
                if streaming_data.get("fallback") != expected_fallback:
                    raise DashboardSmokeFailure(
                        f"streaming fallback mismatch: {streaming_data}"
                    )

                checked_routes: list[str] = []
                for route in CORE_DASHBOARD_ROUTES:
                    page = client.get(route, headers={"accept": "text/html"})
                    if page.status_code != 200 or '<div id="app"' not in page.text:
                        raise DashboardSmokeFailure(
                            f"SPA route failed: {route} HTTP {page.status_code}"
                        )
                    checked_routes.append(route)

                api_miss = client.get("/api/not-a-route")
                if api_miss.status_code != 404:
                    raise DashboardSmokeFailure(
                        f"unknown API returned HTTP {api_miss.status_code}, expected 404"
                    )

                browser_result = _check_browser_dom_routes(args, base_url, work_dir)

        return {
            "ok": True,
            "base_url": base_url,
            "camera": {
                "camera_id": camera_row.get("camera_id"),
                "connected": camera_row.get("connected"),
                "running": camera_row.get("running"),
                "frames_captured": (camera_row.get("stats") or {}).get("frames_captured"),
            },
            "snapshot": {
                "status": snapshot.status_code,
                "content_type": content_type,
                "bytes": len(snapshot.content),
            },
            "streaming": {
                "go2rtc": streaming_data.get("go2rtc"),
                "fallback": streaming_data.get("fallback"),
            },
            "routes_checked": checked_routes,
            "unknown_api_status": api_miss.status_code,
            "browser": browser_result,
            "work_dir": str(work_dir),
        }
    except Exception as exc:
        if isinstance(exc, DashboardSmokeFailure):
            error = str(exc)
        else:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "ok": False,
            "error": error,
            "base_url": base_url,
            "work_dir": str(work_dir),
            "stdout_tail": _tail(stdout_path),
            "stderr_tail": _tail(stderr_path),
        }
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if not args.keep_work_dir:
            _cleanup_dir(work_dir)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke built Dashboard routes")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Argus config to launch (default: configs/default.yaml)",
    )
    parser.add_argument("--port", type=int, default=0, help="Dashboard port; 0 picks a free port")
    parser.add_argument("--camera", default=None, help="Optional --camera argument for argus")
    parser.add_argument("--camera-id", default="c", help="Camera ID expected in the Dashboard")
    parser.add_argument("--timeout", type=float, default=75.0, help="Seconds to wait for camera frames")
    parser.add_argument("--min-frames", type=int, default=5, help="Minimum frames before route checks")
    parser.add_argument(
        "--browser",
        choices=["auto", "required", "off"],
        default="auto",
        help="Headless browser DOM smoke mode (default: auto)",
    )
    parser.add_argument("--browser-path", default=None, help="Explicit Chrome/Edge/Chromium executable")
    parser.add_argument(
        "--browser-timeout",
        type=float,
        default=20.0,
        help="Seconds allowed for each browser DOM dump",
    )
    parser.add_argument(
        "--browser-virtual-time-ms",
        type=int,
        default=6000,
        help="Chrome virtual time budget per route, in milliseconds",
    )
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep temporary logs/video")
    args = parser.parse_args(argv)
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    if args.browser_timeout <= 0:
        parser.error("--browser-timeout must be positive")
    if args.browser_virtual_time_ms <= 0:
        parser.error("--browser-virtual-time-ms must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    result = run_dashboard_route_smoke(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
