# Copyright (c) 2026 Pengshen Zhang
"""Logging Setup: RealtimeASR 日志初始化。

- setup_logging: 配置控制台和 RotatingFileHandler
- _THIRD_PARTY_LEVELS: 收敛 websockets/asyncio/http 等依赖日志噪声
- 模块内用 _configured 防止重复添加 handler
"""
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
