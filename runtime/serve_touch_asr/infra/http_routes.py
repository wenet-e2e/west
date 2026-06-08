# Copyright (c) 2026 Pengshen Zhang
"""HTTP Routes: WebSocket 服务旁路 HTTP 入口。

- health_request_handler: 处理 /health、/ready 和根路径静态页面
- build_response: 兼容 websockets.http11.Response 的响应封装
- 暴露 engine ready/error 状态，不参与 realtime 协议处理
"""
import json
import logging
from pathlib import Path

from engine.runtime import ServiceRuntime
from websockets.datastructures import Headers as WsHeaders
from websockets.http11 import Response as WsResponse


async def health_request_handler(
    connection,
    request,
    *,
    service_runtime: ServiceRuntime,
    web_page: Path,
    logger: logging.Logger,
):
    """
    websockets 12+ 新 API:
    process_request(connection, request) -> WsResponse | None

    拦截 HTTP GET /health 请求，返回模型就绪状态。
    同时拦截所有非 WebSocket upgrade 请求（如浏览器刷新时发出的普通 HTTP GET、
    前端健康轮询），主动返回 HTTP 响应，避免 websockets 抛出 InvalidUpgrade 错误日志。

      返回 None        -> 继续走 WebSocket 升级握手
      返回 WsResponse  -> 直接作为 HTTP 响应发回，跳过升级
    """
    path = request.path

    if path == "/health":
        body = json.dumps({
            "ready": service_runtime.engine_state.ready,
            "phase": service_runtime.engine_state.phase.value,
            "message": service_runtime.engine_state.message,
            "error": service_runtime.engine_state.error,
            "stage": service_runtime.engine_state.message,
        }).encode()
        return WsResponse(
            status_code=200,
            reason_phrase="OK",
            headers=WsHeaders([
                ("Content-Type", "application/json"),
                ("Access-Control-Allow-Origin", "*"),
                ("Content-Length", str(len(body))),
            ]),
            body=body,
        )

    if path == "/" or path == "/index.html":
        try:
            body = web_page.read_bytes()
            return WsResponse(
                status_code=200,
                reason_phrase="OK",
                headers=WsHeaders([
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(body))),
                ]),
                body=body,
            )
        except Exception as e:
            logger.error(f"Failed to serve index.html: {e}")

    # 检查是否为合法的 WebSocket 升级请求。
    # 浏览器刷新页面时会先发普通 HTTP GET（Connection: keep-alive），
    # 直接让 websockets 处理会抛出 InvalidUpgrade 并记录无意义的 ERROR 日志。
    conn_hdr = request.headers.get("Connection", "")
    upgrade_hdr = request.headers.get("Upgrade", "")
    if "upgrade" not in conn_hdr.lower() or upgrade_hdr.lower() != "websocket":
        logger.debug(
            f"Non-WebSocket HTTP {getattr(request, 'method', 'GET')} "
            f"to {path} (Connection: {conn_hdr!r}) - returning 426")
        body = b"This endpoint requires a WebSocket connection.\n"
        return WsResponse(
            status_code=426,
            reason_phrase="Upgrade Required",
            headers=WsHeaders([
                ("Content-Type", "text/plain"),
                ("Upgrade", "websocket"),
                ("Content-Length", str(len(body))),
            ]),
            body=body,
        )

    return None
