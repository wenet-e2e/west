# Copyright (c) 2026 Pengshen Zhang
"""Inference Loop: 单连接推理调度与增量事件生成。

- begin_inference_turn: 计算 history_rollback 后的前缀基线
- maybe_emit_history_reset: 向前端发送回退光标 delta
- inference_worker: 根据 commit/VAD/chunk 条件触发模型推理
- 负责循环调度和事件发送，不直接实现模型调用细节
"""
import asyncio
import importlib
import logging
import time
import traceback
from typing import TYPE_CHECKING

from model import model_inference
from model.vad import detect_leading_silence

if TYPE_CHECKING:
    from engine.runtime import ServiceRuntime
    from session.session import RealtimeSession
    from session.state import ConnectionLoopState

logger = logging.getLogger("RealtimeASR")


async def begin_inference_turn(
    *,
    session: "RealtimeSession",
    service_runtime: "ServiceRuntime",
    realtime_sender,
) -> str:
    """推理轮次开始时调用：计算并返回 turn_base_text（history_rollback 后的前缀基线）。"""
    async with session.lock:
        old_confirmed = session.asr.confirmed_text
        history_rollback_cfg = session.asr.history_rollback_config

    if history_rollback_cfg.strategy == "none":
        base = old_confirmed
    else:
        _proc = service_runtime.engine_state.processor
        base = history_rollback_cfg.apply(
            old_confirmed,
            tokenizer=_proc.tokenizer if _proc else None,
        )

    if len(base) < len(old_confirmed):
        await realtime_sender.delta(
            cursor=len(base),
            delta="",
            is_final=False,
        )
        async with session.lock:
            session.asr.confirmed_text = base

    return base


async def send_streaming_delta(
    *,
    session: "RealtimeSession",
    realtime_sender,
    turn_base: str,
    model_full_text: str,
    is_final: bool = False,
    chunk_id: int = 0,
):
    """在一轮推理中推送 streaming 增量。"""
    if is_final:
        target_text = model_full_text
    else:
        target_text = turn_base + model_full_text

    async with session.lock:
        prev_confirmed = session.asr.confirmed_text

    cursor = len(turn_base)
    delta_to_send = target_text[cursor:]

    if delta_to_send or cursor < len(prev_confirmed) or is_final:
        async with session.lock:
            language = session.asr.language

        await realtime_sender.delta(
            cursor=cursor,
            delta=delta_to_send,
            is_final=is_final,
            chunk_id=chunk_id,
            language=language,
        )

    async with session.lock:
        session.asr.confirmed_text = target_text

    return target_text


async def _reset_vad_state(
    *,
    session: "RealtimeSession",
    state: "ConnectionLoopState",
) -> None:
    state.vad_speech_active = False
    if state.vad_detector is not None:
        state.vad_detector.reset()
        async with session.lock:
            state.vad_base_byte = session.vad_read_pos


async def run_inference_loop(ctx) -> None:
    """后台消费者：定时查看 buffer 是否达到 chunk 阈值或被 commit。"""
    session = ctx.realtime_session
    service_runtime = ctx.service_runtime
    state = ctx.connection_loop_state
    session_id = ctx.session_id
    realtime_sender = ctx.realtime_sender

    infer_count = 0

    while state.keep_processing:
        try:
            await asyncio.sleep(0.1)

            importlib.reload(model_inference)
            cfg = service_runtime.inference_cfg.load()
            chunk_ms = (session.asr.chunk_ms
                        if session.asr.chunk_ms is not None
                        else cfg.chunk_ms)
            prompt = (session.asr.prompt
                      if session.asr.prompt is not None
                      else cfg.prompt)

            async with session.lock:
                is_committing = session.is_committing

            logger.debug(
                f"[{session_id}] POLL is_committing={is_committing} "
                f"buf={len(session.asr.buffer)}B "
                f"last_chunk={session.asr.chunk_id} "
                f"accumulated_len={len(session.asr.accumulated_text)}")

            # VAD 未检测到语音时跳过推理
            td_type = (session.turn_detection
                       or cfg.turn_detection.type)
            is_vad_mode = td_type == "server_vad"
            if (is_vad_mode and not is_committing
                    and not state.vad_speech_active):
                continue

            # [server_vad] 若从未检测到语音，直接空提交
            if is_vad_mode and is_committing:
                async with session.lock:
                    never_had_speech = (
                        session.speech_start_byte == 0
                        and session.asr.accumulated_text == "")
                    snapshot_len = len(session.asr.buffer)
                if never_had_speech:
                    logger.info(
                        f"[{session_id}] COMMIT skipped "
                        "(VAD never detected speech), "
                        "sending empty result")
                    turn_base = await begin_inference_turn(
                        session=session,
                        service_runtime=service_runtime,
                        realtime_sender=realtime_sender,
                    )
                    await send_streaming_delta(
                        session=session,
                        realtime_sender=realtime_sender,
                        turn_base=turn_base,
                        model_full_text="",
                        is_final=True,
                        chunk_id=session.asr.chunk_id,
                    )
                    await realtime_sender.completed(
                        transcript="", language=None)
                    await realtime_sender.done()
                    await session.reset(keep_buffer_from=snapshot_len)
                    infer_count = 0
                    await _reset_vad_state(session=session, state=state)
                    continue

            audio_numpy, extracted_buf_len = (
                await session.take_audio_up_to_chunk(
                    chunk_ms=chunk_ms,
                    force_all=is_committing,
                    use_speech_offset=is_vad_mode))

            if audio_numpy is not None:
                logger.debug(
                    f"[{session_id}] EXTRACT force={is_committing} "
                    f"audio={len(audio_numpy) / 16000:.2f}s "
                    f"chunk_id={session.asr.chunk_id}")
            else:
                logger.debug(
                    f"[{session_id}] EXTRACT force={is_committing} "
                    f"audio=None buf={len(session.asr.buffer)}B")

            if audio_numpy is not None and len(audio_numpy) > 0:
                audio_sec = len(audio_numpy) / 16000
                chunk_id = session.asr.chunk_id

                # ---- 前导静音跳过（none 模式，仅在首次检测到语音前生效） ----
                if (not is_vad_mode and not is_committing
                        and not session.speech_detected):
                    ss_cfg = cfg.silence_skip
                    if ss_cfg.enabled:
                        decision = detect_leading_silence(
                            audio_numpy, ss_cfg)
                        if decision.is_silence:
                            logger.info(
                                f"[{session_id}] SKIP chunk={chunk_id} "
                                f"audio={audio_sec:.2f}s "
                                f"rms={decision.rms:.5f} "
                                f"voiced={decision.voiced_frames}/"
                                f"{decision.total_frames}frames "
                                f"(thresh={decision.threshold}) "
                                "— 前导静音，跳过推理")
                            async with session.lock:
                                session.asr.chunk_id -= 1
                            continue
                        else:
                            async with session.lock:
                                session.speech_detected = True
                            logger.info(
                                f"[{session_id}] VOICE_DETECTED "
                                f"chunk={chunk_id} "
                                f"audio={audio_sec:.2f}s "
                                f"rms={decision.rms:.5f} "
                                f"voiced={decision.voiced_frames}/"
                                f"{decision.total_frames}frames "
                                f"(thresh={decision.threshold}) "
                                "— 检测到语音，开始推理")

                infer_count += 1
                logger.info(
                    f"[{session_id}] >>> #{infer_count} "
                    f"audio={audio_sec:.2f}s chunk={chunk_id} "
                    f"force={is_committing}")

                turn_base = await begin_inference_turn(
                    session=session,
                    service_runtime=service_runtime,
                    realtime_sender=realtime_sender,
                )
                # use_history=False 或 history_reset 生效时，模型看全量音频，
                # 输出即完整文本，不需要拼 turn_base（否则会重复），cursor=0 覆盖
                if not session.asr.use_history:
                    turn_base = ""
                elif chunk_id <= session.asr.history_reset_chunk_num:
                    turn_base = ""
                final_text = ""
                turn_t0 = time.time()

                es = service_runtime.engine_state
                infer_gen = model_inference.run_model_inference(
                    audio_numpy, prompt, session,
                    es.engine, es.processor, es.process_mm_info,
                    infer_tag=f"#{infer_count}/chunk={chunk_id}",
                    model_type=service_runtime.settings.model_type,
                    cfg=cfg,
                )
                async for current_full_text in infer_gen:
                    if current_full_text:
                        final_text = current_full_text
                        await send_streaming_delta(
                            session=session,
                            realtime_sender=realtime_sender,
                            turn_base=turn_base,
                            model_full_text=current_full_text,
                            is_final=False,
                            chunk_id=chunk_id,
                        )

                turn_cost = time.time() - turn_t0
                logger.info(
                    f"[{session_id}] <<< #{infer_count} "
                    f"\"{final_text[:80]}\" chunk={chunk_id} "
                    f"cost={turn_cost:.3f}s")

                if not is_committing:
                    logger.info(
                        f"[{session_id}] INTERMEDIATE_DONE "
                        f"text=\"{final_text[:40]}\" "
                        f"chunk={chunk_id} infers={infer_count}")

                if is_committing:
                    logger.info(
                        f"[{session_id}] COMMIT_INFERRED "
                        f"accumulated=\"{session.asr.accumulated_text[:60]}\" "
                        f"chunk={chunk_id} infers={infer_count}")
                    async with session.lock:
                        accumulated = session.asr.accumulated_text
                    target = await send_streaming_delta(
                        session=session,
                        realtime_sender=realtime_sender,
                        turn_base=turn_base,
                        model_full_text=accumulated,
                        is_final=True,
                        chunk_id=chunk_id,
                    )
                    async with session.lock:
                        language = session.asr.language
                    await realtime_sender.completed(
                        transcript=target,
                        language=language,
                    )
                    await realtime_sender.done()

                    logger.info(
                        f"[{session_id}] COMMIT transcript=\"{target[:100]}\" "
                        f"chunk={chunk_id} infers={infer_count}")

                    await session.reset(keep_buffer_from=extracted_buf_len)
                    infer_count = 0
                    await _reset_vad_state(session=session, state=state)

            elif is_committing:
                logger.warning(
                    f"[{session_id}] COMMIT_CACHED "
                    f"accumulated=\"{session.asr.accumulated_text[:60]}\" "
                    f"buf={len(session.asr.buffer)}B "
                    f"last_chunk={session.asr.chunk_id} "
                    f"infers={infer_count}")
                async with session.lock:
                    accumulated = session.asr.accumulated_text
                turn_base = await begin_inference_turn(
                    session=session,
                    service_runtime=service_runtime,
                    realtime_sender=realtime_sender,
                )
                if not session.asr.use_history:
                    turn_base = ""
                elif chunk_id <= session.asr.history_reset_chunk_num:
                    turn_base = ""
                target = await send_streaming_delta(
                    session=session,
                    realtime_sender=realtime_sender,
                    turn_base=turn_base,
                    model_full_text=accumulated,
                    is_final=True,
                    chunk_id=session.asr.chunk_id,
                )
                async with session.lock:
                    language = session.asr.language

                await realtime_sender.completed(
                    transcript=target,
                    language=language,
                )
                await realtime_sender.done()

                await session.reset(keep_buffer_from=extracted_buf_len)
                infer_count = 0
                await _reset_vad_state(session=session, state=state)
        except Exception:
            logger.error(
                f"[{session_id}] inference_loop error: "
                f"{traceback.format_exc()}")
            await asyncio.sleep(1)  # 避免死循环狂刷日志
