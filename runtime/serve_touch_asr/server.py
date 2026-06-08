# Copyright (c) 2026 Pengshen Zhang
"""Realtime ASR Server: 入口、路由与连接编排。

- main/run_server: 解析启动参数，绑定 HTTP/WebSocket 端口
- realtime_handler: 创建单连接 session/context/sender
- process_client_message: 处理 realtime 协议消息和 session.update
- engine 启动任务、健康检查路由、SSL 上下文在这里完成装配
"""
import argparse
import asyncio
import functools
import json
import logging
import os
import traceback
import uuid
from pathlib import Path

import websockets
import yaml
from engine.config import InferenceConfig, InferenceConfigCache, StartupConfig
from engine.runtime import ServerSettings, ServiceRuntime
from engine.startup import init_engine
from infra import http_routes
from infra.logging_setup import setup_logging
from infra.ssl_context import create_ssl_context
from model.history_rollback import HistoryRollbackConfig
from model.vad import SileroVadProvider
from session.inference_loop import run_inference_loop
from session.sender import RealtimeSender
from session.session import RealtimeSession
from session.state import ConnectionContext, ConnectionLoopState
from session.vad_loop import run_vad_loop

# logger 在 run_server 中通过 settings.log_dir 完成初始化
logger: logging.Logger = logging.getLogger("RealtimeASR")


def _on_config_reload(cfg: InferenceConfig) -> None:
    new_level = cfg.log_level
    numeric = getattr(logging, new_level, logging.INFO)
    if logger.level != numeric:
        logger.setLevel(numeric)
        logger.info(f"Log level changed to {new_level}")
    logger.info("Inference config reloaded")


def _build_service_runtime(
    settings: ServerSettings,
) -> ServiceRuntime:
    """根据 ServerSettings 构造进程级 ServiceRuntime。"""
    return ServiceRuntime(
        inference_cfg=InferenceConfigCache(
            path=settings.inference_config_path,
            on_reload=_on_config_reload,
        ),
        vad_provider=SileroVadProvider(
            silero_repo=settings.silero_repo,
        ),
    )


async def realtime_handler(websocket, service_runtime: ServiceRuntime):
    """
    OpenAI Realtime 兼容 WebSocket 端点。
    采用"双协程"并发模式，解耦"接收音频"与"模型推理"。
    """
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    session = RealtimeSession(
        session_id,
        default_history_rollback_config=(
            service_runtime.settings.default_history_rollback_config),
        save_audio=service_runtime.settings.save_audio,
        audio_save_dir=service_runtime.settings.audio_save_dir,
    )
    logger.info(f"[{session_id}] Client connected.")

    # 等待引擎就绪 (兜底逻辑，防止未经过 /health 检查的客户端直连导致音频积压)
    if not service_runtime.engine_state.ready:
        logger.warning(
            f"[{session_id}] Engine not ready at WebSocket connect, waiting...")
        while not service_runtime.engine_state.ready:
            await asyncio.sleep(1)
        logger.info(f"[{session_id}] Engine ready, proceeding.")

    cfg = service_runtime.inference_cfg.load()

    async with session.lock:
        session.asr.history_reset_chunk_num = cfg.history_reset_chunk_num
        session.asr.min_history_chars = cfg.min_history_chars

    realtime_sender = RealtimeSender(websocket)
    await realtime_sender.session_created(session, session_id, service_runtime)

    # 跨协程共享状态容器 (asyncio 单线程模型无需加锁)
    state = ConnectionLoopState()
    ctx = ConnectionContext(
        session_id=session_id,
        websocket=websocket,
        realtime_session=session,
        connection_loop_state=state,
        service_runtime=service_runtime,
        realtime_sender=realtime_sender,
    )

    inference_task = asyncio.create_task(run_inference_loop(ctx))
    vad_task = asyncio.create_task(run_vad_loop(ctx))

    try:
        async for raw_msg in websocket:
            # Handle binary messages (audio data directly)
            if isinstance(raw_msg, bytes):
                await session.append_audio_bytes(raw_msg)
                continue

            try:
                msg = json.loads(raw_msg)
            except Exception:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "input_audio_buffer.commit":
                async with session.lock:
                    buf_bytes = len(session.asr.buffer)
                    buf_duration = buf_bytes / (session.sr * 2)
                    session.commit_pending_count += 1
                    session.is_committing = True
                logger.info(
                    f"[{session_id}] Received COMMIT. "
                    f"buffer={buf_bytes}B ({buf_duration:.2f}s), "
                    f"chunk_id={session.asr.chunk_id}, "
                    f"accumulated_text=\"{session.asr.accumulated_text[:60]}\"")

            elif msg_type == "session.update":
                sess_cfg = msg.get("session", {})
                extra_cfg = sess_cfg.get("extra", {})
                updated_fields = []

                # --- OpenAI 标准字段 ---
                # instructions → prompt
                instructions_val = sess_cfg.get("instructions")
                if instructions_val is not None and isinstance(
                        instructions_val, str) and len(instructions_val) < 500:
                    async with session.lock:
                        session.asr.prompt = instructions_val
                    updated_fields.append(
                        f"instructions=\"{session.asr.prompt[:40]}\"")

                # audio.input.turn_detection
                audio_cfg = sess_cfg.get("audio", {})
                input_cfg = audio_cfg.get("input", {})
                td_obj = input_cfg.get("turn_detection")
                if td_obj is not None and isinstance(td_obj, dict):
                    td_type = td_obj.get("type")
                    if td_type in ("server_vad", "none"):
                        async with session.lock:
                            session.turn_detection = td_type
                        state.vad_speech_active = False  # 切换模式时重置，避免状态残留
                        updated_fields.append(f"turn_detection={td_type}")

                # --- extra 自定义扩展字段 ---
                history_rollback_dict = extra_cfg.get("history_rollback")
                if history_rollback_dict is not None:
                    new_rb = HistoryRollbackConfig.from_dict(
                        history_rollback_dict)
                    async with session.lock:
                        session.asr.history_rollback_config = new_rb
                    updated_fields.append(
                        f"history_rollback={new_rb.strategy}/{new_rb.value}")

                chunk_ms_val = extra_cfg.get("chunk_ms")
                if chunk_ms_val is not None:
                    async with session.lock:
                        session.asr.chunk_ms = max(
                            200, min(int(chunk_ms_val), 10000))
                    updated_fields.append(f"chunk_ms={session.asr.chunk_ms}")

                use_history_val = extra_cfg.get("use_history")
                if use_history_val is not None:
                    async with session.lock:
                        session.asr.use_history = bool(use_history_val)
                    updated_fields.append(
                        f"use_history={session.asr.use_history}")

                if updated_fields:
                    logger.info(
                        f"[{session_id}] Session updated: "
                        f"{', '.join(updated_fields)}")
                await ctx.realtime_sender.session_updated(
                    session, session_id, service_runtime)

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[{session_id}] WebSocket disconnected.")
    except Exception:
        logger.error(f"[{session_id}] Error: {traceback.format_exc()}")
    finally:
        state.keep_processing = False
        inference_task.cancel()
        vad_task.cancel()
        await asyncio.gather(
            inference_task,
            vad_task,
            return_exceptions=True,
        )
        logger.info(f"[{session_id}] Cleaned up.")


async def ws_router(websocket, service_runtime: ServiceRuntime):
    """根据请求路径分发到对应 handler，兼容前端 ws://host:port/v1/realtime"""
    path = websocket.request.path if hasattr(
        websocket, 'request') else getattr(
        websocket, 'path', '/')
    if path == "/v1/realtime" or path == "/":
        await realtime_handler(websocket, service_runtime)
    else:
        await websocket.close(4004, f"Unknown path: {path}")


async def run_server(settings: ServerSettings):
    global logger
    logger = setup_logging(settings.log_dir)

    service_runtime = _build_service_runtime(settings)
    service_runtime.settings.model_name = settings.model
    service_runtime.settings.save_audio = settings.save_audio
    service_runtime.settings.audio_save_dir = settings.audio_save_dir
    service_runtime.settings.default_history_rollback_config = (
        HistoryRollbackConfig(
            settings.history_rollback_strategy,
            settings.history_rollback_value))
    if (service_runtime.settings.default_history_rollback_config.strategy
            != "none"):
        logger.info(
            f"Default history_rollback: "
            f"strategy={settings.history_rollback_strategy}, "
            f"value={settings.history_rollback_value}")
    if service_runtime.settings.save_audio:
        logger.info(
            f"Audio saving is ENABLED. "
            f"Audios will be saved to "
            f"./{service_runtime.settings.audio_save_dir}/")

    ssl_context = create_ssl_context(
        use_ssl=settings.use_ssl, logger=logger)

    ws_scheme = "wss" if settings.use_ssl else "ws"
    http_protocol = "https" if settings.use_ssl else "http"
    logger.info(
        f"正在初始化模型，完成后启动 WebSocket 服务，"
        f"监听 {settings.host}:{settings.port} ...")

    try:
        await init_engine(
            service_runtime=service_runtime,
            logger=logger,
            model=settings.model,
            tensor_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            model_type_arg=settings.model_type,
        )
    except Exception as exc:
        service_runtime.engine_state.set_failed(str(exc))
        logger.error(f"模型初始化失败:\n{traceback.format_exc()}")
        raise

    process_request = functools.partial(
        http_routes.health_request_handler,
        service_runtime=service_runtime,
        web_page=settings.web_page,
        logger=logger,
    )

    handler = functools.partial(
        ws_router, service_runtime=service_runtime)

    async with websockets.serve(
        handler,
        settings.host,
        settings.port,
        process_request=process_request,
        ssl=ssl_context,
        open_timeout=60,
        close_timeout=10,
        ping_interval=20,
        ping_timeout=20,
        max_size=50 * 1024 * 1024,
    ):
        logger.info(
            f"WebSocket 服务已启动: "
            f"{ws_scheme}://{settings.host}:{settings.port}/v1/realtime  "
            f"健康检查与前端: "
            f"{http_protocol}://{settings.host}:{settings.port}/")
        await asyncio.Future()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Qwen3-Omni Realtime WebSocket ASR')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--gpu-ids', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--silero-model-path', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument(
        '--web-page', type=str, default='./index.html',
        help='Path to the debug web UI HTML file')
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = yaml.safe_load(
        config_path.read_text(encoding="utf-8")) or {}
    startup = StartupConfig.model_validate(raw.get("startup", {}))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    asyncio.run(run_server(ServerSettings(
        model=args.model,
        host=startup.host,
        port=startup.port,
        tensor_parallel_size=startup.tensor_parallel_size,
        gpu_memory_utilization=startup.gpu_memory_utilization,
        save_audio=startup.save_audio,
        history_rollback_strategy=startup.history_rollback_strategy,
        history_rollback_value=startup.history_rollback_value,
        audio_save_dir=startup.audio_save_dir,
        model_type=startup.model_type,
        use_ssl=startup.ssl.enabled,
        inference_config_path=config_path,
        silero_repo=Path(args.silero_model_path),
        log_dir=Path(args.log_dir),
        web_page=Path(args.web_page),
    )))
