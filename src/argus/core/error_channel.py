"""统一的运行时错误事件总线。

让所有后台关键失败事件同时推到 WebSocket,前端能聚合显示。
本身是被动通道(receiver injection),不做日志/指标存储 —— 那些由 structlog 已经做了。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


# severity 取值约定(与前端对齐)
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITY_CRITICAL = "critical"


class ErrorChannel:
    """轻量错误事件分发器。

    生命周期:
        1. 进程启动早期 emit() 会把事件缓冲到内存(因为 ws_manager 还没就绪)
        2. dashboard 启动后 set_publisher(ws_manager.broadcast) 一次性 flush 缓冲
        3. 后续 emit() 直接走 publisher

    设计特性:
        - publisher 失败永不上抛(用 try/except 兜)
        - 缓冲上限 200 条,溢出丢最旧
        - 不做去重 / 不做节流(由前端 store 决定如何展示)
    """

    _BUFFER_LIMIT = 200

    def __init__(self) -> None:
        self._publisher: Callable[[str, dict], None] | None = None
        self._buffer: list[dict[str, Any]] = []

    def set_publisher(self, publisher: Callable[[str, dict], None]) -> None:
        """注入 ws_manager.broadcast(topic, payload) 类似的可调用对象。

        注入后立即 flush 启动期累积的事件。重复注入会替换旧 publisher。
        """
        self._publisher = publisher
        if self._buffer:
            for evt in self._buffer:
                self._safe_publish(evt)
            self._buffer.clear()

    def emit(
        self,
        severity: str,
        source: str,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """记录一个错误事件并尝试推送到前端。

        Args:
            severity: info / warning / error / critical
            source: 'dispatcher' / 'pipeline' / 'release_pipeline' / 'gige_capture' / ...
            code: 短机器码,如 'db_overflow' / 'reconnect_exhausted'
            message: 人读消息(中文 ok)
            context: 额外上下文(alert_id / camera_id / model_version_id 等)
        """
        evt = {
            "type": "error_event",
            "severity": severity,
            "source": source,
            "code": code,
            "message": message,
            "context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._publisher is None:
            if len(self._buffer) >= self._BUFFER_LIMIT:
                # 溢出丢最旧:保留最新的事件,因为运维更关心当下发生了什么。
                self._buffer.pop(0)
            self._buffer.append(evt)
            return
        self._safe_publish(evt)

    def _safe_publish(self, evt: dict[str, Any]) -> None:
        try:
            self._publisher("system_errors", evt)
        except Exception as e:
            # 不能再调 logger.error(可能形成"广播错误失败"的死循环),用 debug 即可
            logger.debug(
                "error_channel.publish_failed",
                error_type=type(e).__name__,
                error=str(e),
                evt_code=evt.get("code"),
            )


# 全局单例(模块级)
_channel = ErrorChannel()


def get_error_channel() -> ErrorChannel:
    """获取全局 ErrorChannel 单例。"""
    return _channel
