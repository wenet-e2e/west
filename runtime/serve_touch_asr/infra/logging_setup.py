# Copyright (c) 2026 Pengshen Zhang
"""Logging Setup: RealtimeASR 日志初始化。

- setup_logging: 配置控制台和 RotatingFileHandler
- _THIRD_PARTY_LEVELS: 收敛 websockets/asyncio/http 等依赖日志噪声
- _HandshakeNoiseFilter: 丢弃对端探活/断连造成的 "opening handshake failed" 噪声
- 模块内用 _configured 防止重复添加 handler
"""
import asyncio
import logging
import logging.handlers
from pathlib import Path

_configured = False

_THIRD_PARTY_LEVELS = {
    "websockets": logging.ERROR,
    "asyncio": logging.WARNING,
    "urllib3": logging.WARNING,
    "vllm": logging.WARNING,
    "filelock": logging.WARNING,
}

# 握手阶段对端断开/超时属于良性噪声（探活、端口扫描、客户端中途断连），
# 仅丢弃这类异常对应的 ERROR 日志，真实握手错误仍会保留。
_BENIGN_HANDSHAKE_EXC = (
    ConnectionError,
    OSError,
    asyncio.TimeoutError,
)


class _HandshakeNoiseFilter(logging.Filter):
    """过滤 websockets "opening handshake failed" 的良性断连噪声。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() != "opening handshake failed":
            return True
        exc = record.exc_info[1] if record.exc_info else None
        if exc is None:
            return True
        try:
            from websockets.exceptions import ConnectionClosed
        except Exception:
            ConnectionClosed = ()  # noqa: N806
        if isinstance(exc, ConnectionClosed):
            return False
        if isinstance(exc, _BENIGN_HANDSHAKE_EXC):
            return False
        # 兜底：链式异常里若包含良性断连，也一并丢弃。
        cause = exc.__cause__ or exc.__context__
        if cause is not None and isinstance(cause, _BENIGN_HANDSHAKE_EXC):
            return False
        return True


def setup_logging(
    log_dir: Path,
    console_level: str = "DEBUG",
    file_level: str = "INFO",
    max_bytes: int = 200 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """一次性配置 root + RealtimeASR logger。

    幂等：多次调用只生效一次（防 reload 叠加 handler）。
    """
    global _configured
    if _configured:
        return logging.getLogger("RealtimeASR")
    _configured = True

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, console_level, logging.DEBUG),
        format=fmt,
        datefmt=datefmt,
    )

    logger = logging.getLogger("RealtimeASR")
    logger.setLevel(getattr(logging, console_level, logging.DEBUG))

    for name, level in _THIRD_PARTY_LEVELS.items():
        logging.getLogger(name).setLevel(level)

    # websockets 在 conn_handler 里以 ERROR 打印握手失败，单纯调级别压不住，
    # 这里挂过滤器精确丢弃对端探活/断连噪声。
    _ws_server_logger = logging.getLogger("websockets.server")
    if not any(
        isinstance(f, _HandshakeNoiseFilter)
        for f in _ws_server_logger.filters
    ):
        _ws_server_logger.addFilter(_HandshakeNoiseFilter())

    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        str(log_dir / "server.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(getattr(logging, file_level, logging.INFO))
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(fh)

    return logger
