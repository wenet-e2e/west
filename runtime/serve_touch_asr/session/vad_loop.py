# Copyright (c) 2026 Pengshen Zhang
"""VAD Loop: server_vad 话轮检测循环。

- vad_worker: 消费连接音频流，驱动单连接 VadDetector
- speech_started/speech_stopped: 根据 VAD 状态发送 realtime 事件
- 自动 commit 话轮并唤醒 inference_worker
- 只负责话轮边界，不直接执行 ASR 推理
"""
import asyncio
import logging

logger = logging.getLogger("RealtimeASR")


async def run_vad_loop(ctx) -> None:
    """后台 VAD 监控协程：每 50ms 检查一次新增音频，流式 VAD 检测。"""
    session = ctx.realtime_session
    app = ctx.service_runtime
    state = ctx.connection_loop_state
    session_id = ctx.session_id
    realtime_sender = ctx.realtime_sender

    item_id_counter = 0

    while state.keep_processing:
        await asyncio.sleep(0.05)

        # 读取当前 turn_detection 类型（session 覆盖 > yaml 默认）
        if session.turn_detection is not None:
            td_type = session.turn_detection
        else:
            td_type = app.inference_cfg.load().turn_detection.type

        if td_type != "server_vad":
            if state.vad_detector is not None:
                state.vad_detector = None
            continue

        async with session.lock:
            if session.is_committing:
                continue

        if state.vad_detector is None:
            td_cfg = app.inference_cfg.load().turn_detection
            state.vad_detector = app.vad_provider.create_detector(
                session_id, td_cfg)
            state.vad_speech_active = False
            async with session.lock:
                state.vad_base_byte = session.vad_read_pos
            logger.info(
                f"[{session_id}] VAD monitor: created "
                f"{state.vad_detector.engine_name} "
                f"detector (base_byte={state.vad_base_byte})")

        new_audio = await session.drain_new_audio()
        if new_audio is None or len(new_audio) == 0:
            continue

        try:
            events = state.vad_detector.feed(new_audio)
        except Exception as e:
            logger.error(f"[{session_id}] VAD monitor feed error: {e}")
            continue

        for evt in events:
            if "start" in evt and not state.vad_speech_active:
                state.vad_speech_active = True
                audio_start_ms = int(evt["start"] / 16.0)
                speech_start_sample = evt["start"]
                async with session.lock:
                    session.speech_start_byte = state.vad_base_byte + int(
                        speech_start_sample) * 2
                    session.asr.chunk_id = 0
                item_id_counter += 1
                current_item_id = f"{session_id}_item_{item_id_counter}"
                logger.info(
                    f"[{session_id}] VAD: speech_started at {audio_start_ms}ms "
                    f"(byte_offset={session.speech_start_byte})")
                await realtime_sender.speech_started(
                    audio_start_ms=audio_start_ms,
                    item_id=current_item_id,
                )

            elif "end" in evt and state.vad_speech_active:
                state.vad_speech_active = False
                audio_end_ms = int(evt["end"] / 16.0)
                logger.info(
                    f"[{session_id}] VAD: speech_stopped at {audio_end_ms}ms")
                item_id = f"{session_id}_item_{item_id_counter}"
                await realtime_sender.speech_stopped(
                    audio_end_ms=audio_end_ms,
                    item_id=item_id,
                )

                async with session.lock:
                    buf_bytes = len(session.asr.buffer)
                    buf_duration = buf_bytes / (session.sr * 2)
                    session.commit_pending_count += 1
                    session.is_committing = True
                logger.info(
                    f"[{session_id}] VAD auto-commit: "
                    f"buffer={buf_bytes}B ({buf_duration:.2f}s)")

                await realtime_sender.input_audio_committed(item_id=item_id)

                state.vad_detector.reset()
                async with session.lock:
                    state.vad_base_byte = session.vad_read_pos
