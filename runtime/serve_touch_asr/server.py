# Copyright (c) 2026 Pengshen Zhang
# ==============================================
# Qwen3-Omni Realtime ASR Service
#
# Features:
# - OpenAI Realtime API (WebSocket) 兼容 (部分核心事件)
# - 支持 Manual (手动截断) 与 VAD (服务端自动截断) 双模式
# - 完美支持多并发 (每个 WebSocket 连接独立 Session，无全局锁竞争)
# - 内部实现基于字符/词维度的 History Rollback 上下文回退
# - 纯 websockets 库实现，无 FastAPI/Uvicorn 依赖
# ==============================================
import argparse
import asyncio
import base64
import importlib
import json
import logging
import os
import time
import traceback
import uuid
from typing import Optional

import inference_logic
import numpy as np
import soundfile as sf
import websockets
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.async_omni import AsyncOmni
from websockets.datastructures import Headers as WsHeaders
from websockets.http11 import Response as WsResponse

# ==============================================
# vLLM-Omni & Model Imports
# ==============================================
os.environ['VLLM_USE_V1'] = '0'
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')


# ==============================================
# 推理逻辑模块 (支持 importlib.reload 热更新)
# ==============================================

# ==============================================
# Global Engine & Logger
# ==============================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RealtimeASR")
logger.setLevel(logging.INFO)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

engine: Optional['AsyncOmni'] = None
processor: Optional['Qwen3OmniMoeProcessor'] = None
model_name: str = "qwen3-omni"
save_audio: bool = False
audio_save_dir: str = "saved_audios"
default_rollback_config = inference_logic.RollbackConfig("none", 0.0)

# 模型加载状态追踪（供 WebSocket 连接时判断是否就绪）
engine_ready = False
engine_loading_stage: str = "等待启动"

# ==============================================
# Multi-Concurrency Audio Session
# ==============================================


class AudioSession:
    """
    独立于每个 WebSocket 连接的音频会话状态。
    内部封装音频 buffer 和并发锁，实现 Session 级别的状态隔离。
    """

    def __init__(self, session_id: str, sr: int = 16000):
        self.session_id = session_id
        self.sr = sr
        self.buffer = bytearray()
        self.last_triggered_chunk_id = 0
        self.accumulated_text = ""
        self.confirmed_text = ""
        self.rollback_config = inference_logic.RollbackConfig(
            default_rollback_config.strategy,
            default_rollback_config.value,
        )
        self.is_committing = False
        self.commit_pending_count = 0
        self.lock = asyncio.Lock()

        # client 可控参数（None = 使用 inference_config.yaml 默认值）
        self.chunk_ms: Optional[int] = None
        self.prompt: Optional[str] = None
        self.use_history: bool = True

        # 前导静音跳过标志
        self.speech_detected = False

        # VAD 话轮检测状态
        self.turn_detection: Optional[str] = None   # "server_vad" | "none"
        self.vad_read_pos: int = 0                  # 已读取的缓冲区位置（字节）
        self.speech_start_byte: int = 0             # 本轮语音起始位置（字节）

    async def append_audio(self, b64_audio: str):
        try:
            delta_bytes = base64.b64decode(b64_audio)

            if len(delta_bytes) == 0:
                return
            if len(delta_bytes) % 2 != 0:
                logger.warning(
                    f"[{self.session_id}] Audio packet size {len(delta_bytes)}B "
                    f"is not even — not valid PCM16. Dropping packet.")
                return

            async with self.lock:
                is_first = len(self.buffer) == 0
                self.buffer.extend(delta_bytes)
                buf_bytes = len(self.buffer)
                buf_duration = buf_bytes / (self.sr * 2)

                if is_first:
                    samples = len(delta_bytes) // 2
                    pkt_ms = samples / self.sr * 1000
                    peek = np.frombuffer(delta_bytes, dtype=np.int16)
                    abs_max = int(np.abs(peek).max())
                    logger.info(
                        f"[{self.session_id}] FirstAudioPacket: "
                        f"{len(delta_bytes)}B ({samples} samples, {pkt_ms:.0f}ms) "
                        f"peak={abs_max} {'(silence?)' if abs_max < 10 else ''}")

                if buf_bytes % (self.sr * 2) == 0:
                    logger.debug(
                        f"[{self.session_id}] chunk={self.last_triggered_chunk_id} "
                        f"AudioBuffer: append={len(delta_bytes)}B "
                        f"total={buf_bytes}B ({buf_duration:.1f}s)")
        except Exception as e:
            logger.error(f"[{self.session_id}] Audio decode error: {e}")

    async def extract_for_inference(
            self,
            chunk_ms: int,
            force_all: bool = False,
            is_pcm16: bool = True,
            use_speech_offset: bool = False):
        async with self.lock:
            bytes_per_sample = 2 if is_pcm16 else 4
            chunk_bytes = int(self.sr * bytes_per_sample * (chunk_ms / 1000))

            # 应用 VAD 语音起始偏移
            offset = self.speech_start_byte if use_speech_offset else 0
            effective_buf = self.buffer[offset:]
            available_chunks = len(effective_buf) // chunk_bytes

            buffer_len_at_extract = len(self.buffer)

            target_pcm_bytes = None
            if force_all and len(effective_buf) > 0:
                target_pcm_bytes = bytes(effective_buf)
            else:
                next_chunk_id = self.last_triggered_chunk_id + 1
                if available_chunks >= next_chunk_id:
                    self.last_triggered_chunk_id = next_chunk_id
                    valid_len = next_chunk_id * chunk_bytes
                    target_pcm_bytes = bytes(effective_buf[:valid_len])

            if target_pcm_bytes:
                valid_bytes_len = (
                    len(target_pcm_bytes) // bytes_per_sample) * bytes_per_sample
                target_pcm_bytes = target_pcm_bytes[:valid_bytes_len]

                if is_pcm16:
                    arr = np.frombuffer(
                        target_pcm_bytes,
                        dtype=np.int16).astype(
                        np.float32) / 32768.0
                else:
                    arr = np.frombuffer(target_pcm_bytes, dtype=np.float32)

                audio_sec = len(arr) / self.sr
                logger.debug(
                    f"[{self.session_id}] chunk={self.last_triggered_chunk_id} "
                    f"ExtractAudio: {audio_sec:.2f}s ({len(arr)} samples) "
                    f"force={force_all} offset={offset}B range=[{arr.min():.4f}, {arr.max():.4f}]")
                return arr, buffer_len_at_extract
            return None, buffer_len_at_extract

    async def extract_new_audio_for_vad(self) -> Optional[np.ndarray]:
        """提取 VAD 还未读过的新增音频片段 (float32, 16kHz)。"""
        async with self.lock:
            if self.vad_read_pos >= len(self.buffer):
                return None
            new_bytes = bytes(self.buffer[self.vad_read_pos:])
            self.vad_read_pos = len(self.buffer)
        valid_len = (len(new_bytes) // 2) * 2
        if valid_len == 0:
            return None
        arr = np.frombuffer(new_bytes[:valid_len],
                            dtype=np.int16).astype(np.float32) / 32768.0
        return arr

    async def reset(self, keep_buffer_from: int = 0):
        async with self.lock:
            if save_audio and len(self.buffer) > 0:
                try:
                    os.makedirs(audio_save_dir, exist_ok=True)
                    offset = self.speech_start_byte if self.speech_start_byte > 0 else 0
                    if keep_buffer_from > 0:
                        save_buf = self.buffer[offset:keep_buffer_from]
                    else:
                        save_buf = self.buffer[offset:]
                    valid_bytes_len = (len(save_buf) // 2) * 2
                    raw_bytes = bytes(save_buf[:valid_bytes_len])
                    if len(raw_bytes) > 0:
                        audio_arr = np.frombuffer(
                            raw_bytes, dtype=np.int16).astype(
                            np.float32) / 32768.0
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(
                            audio_save_dir, f"{self.session_id}_{timestamp}.wav")

                        audio_dur = len(audio_arr) / self.sr
                        transcript_snap = self.accumulated_text[:80]
                        sid_snap = self.session_id

                        def _save_task():
                            sf.write(save_path, audio_arr, 16000)
                            logger.info(
                                f"[{sid_snap}] Audio saved to {save_path} "
                                f"({audio_dur:.2f}s) text=\"{transcript_snap}\"")

                        asyncio.get_event_loop().run_in_executor(None, _save_task)
                except Exception as e:
                    logger.error(
                        f"[{self.session_id}] Failed to dispatch audio save task: {e}")

            if keep_buffer_from > 0 and keep_buffer_from < len(self.buffer):
                self.buffer = bytearray(self.buffer[keep_buffer_from:])
            else:
                self.buffer = bytearray()

            self.last_triggered_chunk_id = 0
            if self.commit_pending_count > 0:
                self.commit_pending_count -= 1
            self.is_committing = (self.commit_pending_count > 0)
            self.accumulated_text = ""
            self.confirmed_text = ""
            self.speech_detected = False
            self.vad_read_pos = 0
            self.speech_start_byte = 0

# ==============================================
# WebSocket 工具函数
# ==============================================


async def ws_send(ws, obj: dict):
    await ws.send(json.dumps(obj, ensure_ascii=False))


def _build_session_snapshot(session: 'AudioSession', session_id: str) -> dict:
    """构建符合 OpenAI Realtime 协议的 session 对象快照。"""
    cfg = inference_logic.load_inference_config()
    td_cfg = cfg.get("turn_detection", {})
    eff_td = session.turn_detection or td_cfg.get("type", "none")
    eff_prompt = session.prompt or cfg.get("prompt", "请转录这段音频。")
    return {
        "id": session_id,
        "object": "realtime.session",
        "model": model_name,
        "instructions": eff_prompt,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 16000},
                "turn_detection": {
                    "type": eff_td,
                    "threshold": td_cfg.get("threshold", 0.5),
                    "prefix_padding_ms": td_cfg.get("prefix_padding_ms", 300),
                    "silence_duration_ms": td_cfg.get("silence_duration_ms", 700),
                },
            },
        },
        "extra": {
            "chunk_ms": session.chunk_ms or cfg.get("chunk_ms", 1000),
            "use_history": session.use_history,
            "rollback": {
                "enabled": session.rollback_config.strategy != "none",
                "strategy": session.rollback_config.strategy,
                "value": session.rollback_config.value,
            },
        },
    }


# ==============================================
# WebSocket Connection Handler
# ==============================================
async def realtime_handler(websocket):
    """
    OpenAI Realtime 兼容 WebSocket 端点。
    采用"双协程"并发模式，解耦"接收音频"与"模型推理"。
    """
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    session = AudioSession(session_id)
    logger.info(f"[{session_id}] Client connected.")

    # 等待引擎就绪 (兜底逻辑，防止未经过 /health 检查的客户端直连导致音频积压)
    if not engine_ready:
        logger.warning(
            f"[{session_id}] Engine not ready at WebSocket connect, waiting...")
        while not engine_ready:
            await asyncio.sleep(1)
        logger.info(f"[{session_id}] Engine ready, proceeding.")

    cfg = inference_logic.load_inference_config()
    td_cfg = cfg.get("turn_detection", {})
    await ws_send(websocket, {
        "type": "session.created",
        "session": {
            "id": session_id,
            "object": "realtime.session",
            "model": model_name,
            "instructions": cfg.get("prompt", "请转录这段音频。"),
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "turn_detection": {
                        "type": td_cfg.get("type", "none"),
                        "threshold": td_cfg.get("threshold", 0.5),
                        "prefix_padding_ms": td_cfg.get("prefix_padding_ms", 300),
                        "silence_duration_ms": td_cfg.get("silence_duration_ms", 700),
                    },
                },
            },
            "extra": {
                "chunk_ms": cfg.get("chunk_ms", 1000),
                "use_history": True,
                "rollback": {
                    "enabled": default_rollback_config.strategy != "none",
                    "strategy": default_rollback_config.strategy,
                    "value": default_rollback_config.value,
                },
            },
        },
    })

    # 跨协程共享状态容器 (asyncio 单线程模型无需加锁)
    state = {
        "keep_processing": True,
        "vad_speech_active": False,   # server_vad: VAD 检测到语音才允许定时推理
        "streaming_vad": None,        # StreamingVAD 实例
        "vad_base_byte": 0,           # VAD 实例创建/重置时，buffer 的基础偏移量
    }

    async def begin_inference_turn() -> str:
        """推理轮次开始时调用：计算并返回 turn_base_text（rollback 后的前缀基线）。

        前端此时显示的文本 = session.confirmed_text（上一轮的 target_text）。
        本轮所有 streaming delta 的 cursor 都锚定在 turn_base_text 的末尾。
        """
        async with session.lock:
            old_confirmed = session.confirmed_text
            rollback_cfg = session.rollback_config

        if rollback_cfg.strategy == "none":
            base = old_confirmed
        else:
            base = rollback_cfg.apply(old_confirmed)

        if len(base) < len(old_confirmed):
            await ws_send(websocket, {
                "type": "conversation.item.input_audio_transcription.delta",
                "cursor": len(base),
                "delta": "",
                "is_final": False,
            })
            async with session.lock:
                session.confirmed_text = base

        return base

    async def send_streaming_delta(
        turn_base: str, model_full_text: str,
        is_final: bool = False, chunk_id: int = 0,
    ):
        """在一轮推理中推送 streaming 增量。

        turn_base:        本轮开始时的前缀基线（begin_inference_turn 返回值）
        model_full_text:  模型本次推理的累积完整输出（不含历史前缀）
        is_final:         是否为整个会话最终结果
        chunk_id:         当前触发推理的 chunk 序号

        前端维护 current_turn_text，收到时执行:
          current_turn_text = current_turn_text.slice(0, cursor) + delta
        """
        target_text = turn_base + model_full_text if not is_final else model_full_text

        async with session.lock:
            prev_confirmed = session.confirmed_text

        cursor = len(turn_base)
        delta_to_send = target_text[cursor:]

        if delta_to_send or cursor < len(prev_confirmed) or is_final:
            await ws_send(websocket, {
                "type": "conversation.item.input_audio_transcription.delta",
                "cursor": cursor,
                "delta": delta_to_send,
                "is_final": is_final,
                "chunk_id": chunk_id,
            })

        async with session.lock:
            session.confirmed_text = target_text

        return target_text

    async def inference_processor():
        """后台消费者：定时查看 buffer 是否达到 chunk 阈值或被 commit"""
        infer_count = 0

        while state["keep_processing"]:
            await asyncio.sleep(0.1)

            importlib.reload(inference_logic)
            cfg = inference_logic.load_inference_config()
            chunk_ms = session.chunk_ms if session.chunk_ms is not None else cfg.get(
                "chunk_ms", 1000)
            prompt = session.prompt if session.prompt is not None else cfg.get(
                "prompt", "请转录这段音频。")

            async with session.lock:
                is_committing = session.is_committing

            # VAD 未检测到语音时跳过推理
            td_type = session.turn_detection or cfg.get(
                "turn_detection", {}).get("type", "none")
            is_vad_mode = td_type == "server_vad"
            if is_vad_mode and not is_committing and not state["vad_speech_active"]:
                continue

            # [server_vad] 若从未检测到语音，直接空提交
            if is_vad_mode and is_committing:
                async with session.lock:
                    never_had_speech = (
                        session.speech_start_byte == 0 and session.accumulated_text == "")
                    snapshot_len = len(session.buffer)
                if never_had_speech:
                    logger.info(
                        f"[{session_id}] COMMIT skipped (VAD never detected speech), sending empty result")
                    turn_base = await begin_inference_turn()
                    await send_streaming_delta(turn_base, "", is_final=True, chunk_id=session.last_triggered_chunk_id)
                    await ws_send(websocket, {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": ""
                    })
                    await ws_send(websocket, {"type": "response.done"})
                    await session.reset(keep_buffer_from=snapshot_len)
                    infer_count = 0
                    state["vad_speech_active"] = False
                    if state["streaming_vad"] is not None:
                        state["streaming_vad"].reset()
                        async with session.lock:
                            state["vad_base_byte"] = session.vad_read_pos
                    continue

            audio_numpy, extracted_buf_len = await session.extract_for_inference(
                chunk_ms=chunk_ms, force_all=is_committing, use_speech_offset=is_vad_mode)

            if audio_numpy is not None and len(audio_numpy) > 0:
                audio_sec = len(audio_numpy) / 16000
                chunk_id = session.last_triggered_chunk_id

                # ---- 前导静音跳过（none 模式，仅在首次检测到语音前生效） ----
                if not is_vad_mode and not is_committing and not session.speech_detected:
                    ss_cfg = cfg.get("silence_skip", {})
                    if ss_cfg.get("enabled", False):
                        frame_samples = int(
                            16000 * ss_cfg.get("frame_ms", 20) / 1000)
                        rms_thresh = ss_cfg.get("rms_threshold", 0.003)
                        min_voiced = ss_cfg.get("min_voiced_frames", 1)

                        total_frames = 0
                        voiced_frames = 0
                        for i in range(0, len(audio_numpy), frame_samples):
                            frame = audio_numpy[i:i + frame_samples]
                            if len(frame) == 0:
                                break
                            total_frames += 1
                            frame_rms = float(np.sqrt(np.mean(frame ** 2)))
                            if frame_rms >= rms_thresh:
                                voiced_frames += 1

                        chunk_rms = float(np.sqrt(np.mean(audio_numpy ** 2)))

                        if voiced_frames < min_voiced:
                            logger.info(
                                f"[{session_id}] SKIP chunk={chunk_id} "
                                f"audio={audio_sec:.2f}s rms={chunk_rms:.5f} "
                                f"voiced={voiced_frames}/{total_frames}frames "
                                f"(thresh={rms_thresh}) — 前导静音，跳过推理")
                            async with session.lock:
                                session.last_triggered_chunk_id -= 1
                            continue
                        else:
                            async with session.lock:
                                session.speech_detected = True
                            logger.info(
                                f"[{session_id}] VOICE_DETECTED chunk={chunk_id} "
                                f"audio={audio_sec:.2f}s rms={chunk_rms:.5f} "
                                f"voiced={voiced_frames}/{total_frames}frames "
                                f"(thresh={rms_thresh}) — 检测到语音，开始推理")

                infer_count += 1
                logger.info(
                    f"[{session_id}] >>> #{infer_count} "
                    f"audio={audio_sec:.2f}s chunk={chunk_id} force={is_committing}")

                turn_base = await begin_inference_turn()
                final_text = ""
                turn_t0 = time.time()

                async for current_full_text in inference_logic.run_omni_inference(
                    audio_numpy, prompt, session,
                    engine, processor, process_mm_info,
                    infer_tag=f"#{infer_count}/chunk={chunk_id}",
                ):
                    if current_full_text:
                        final_text = current_full_text
                        await send_streaming_delta(turn_base, current_full_text, is_final=False, chunk_id=chunk_id)

                turn_cost = time.time() - turn_t0
                logger.info(
                    f"[{session_id}] <<< #{infer_count} "
                    f"\"{final_text[:80]}\" chunk={chunk_id} cost={turn_cost:.3f}s")

                if is_committing:
                    async with session.lock:
                        accumulated = session.accumulated_text
                    target = await send_streaming_delta(turn_base, accumulated, is_final=True, chunk_id=chunk_id)
                    await ws_send(websocket, {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": target
                    })
                    await ws_send(websocket, {"type": "response.done"})

                    logger.info(
                        f"[{session_id}] COMMIT transcript=\"{target[:100]}\" "
                        f"chunk={chunk_id} infers={infer_count}")

                    await session.reset(keep_buffer_from=extracted_buf_len)
                    infer_count = 0
                    state["vad_speech_active"] = False
                    if state["streaming_vad"] is not None:
                        state["streaming_vad"].reset()
                        async with session.lock:
                            state["vad_base_byte"] = session.vad_read_pos

            elif is_committing:
                # 收到 commit 但 buffer 中没有有效音频，直接使用积累的历史文本完成这一轮
                async with session.lock:
                    accumulated = session.accumulated_text
                turn_base = await begin_inference_turn()
                target = await send_streaming_delta(
                    turn_base, accumulated, is_final=True,
                    chunk_id=session.last_triggered_chunk_id
                )
                await ws_send(websocket, {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": target
                })
                await ws_send(websocket, {"type": "response.done"})

                logger.info(
                    f"[{session_id}] COMMIT (empty audio) transcript=\"{target[:100]}\" "
                    f"chunk={session.last_triggered_chunk_id} infers={infer_count}")

                await session.reset(keep_buffer_from=extracted_buf_len)
                infer_count = 0
                state["vad_speech_active"] = False
                if state["streaming_vad"] is not None:
                    state["streaming_vad"].reset()
                    async with session.lock:
                        state["vad_base_byte"] = session.vad_read_pos

    # ==============================================
    # Server VAD 后台协程：流式检测 speech_start / speech_end
    # ==============================================
    async def vad_monitor():
        """后台 VAD 监控协程：每 50ms 检查一次新增音频，流式 VAD 检测。"""
        item_id_counter = 0

        while state["keep_processing"]:
            await asyncio.sleep(0.05)

            # 读取当前 turn_detection 类型（session 覆盖 > yaml 默认）
            if session.turn_detection is not None:
                td_type = session.turn_detection
            else:
                td_type = inference_logic.load_inference_config().get(
                    "turn_detection", {}).get("type", "none")

            if td_type != "server_vad":
                if state["streaming_vad"] is not None:
                    state["streaming_vad"] = None
                continue

            async with session.lock:
                if session.is_committing:
                    continue

            if state["streaming_vad"] is None:
                importlib.reload(inference_logic)
                td_cfg = dict(
                    inference_logic.load_inference_config().get(
                        "turn_detection", {}))
                state["streaming_vad"] = inference_logic.create_streaming_vad(
                    session_id, td_cfg)
                state["vad_speech_active"] = False
                async with session.lock:
                    state["vad_base_byte"] = session.vad_read_pos
                logger.info(
                    f"[{session_id}] VAD monitor: created {state['streaming_vad'].engine_name} "
                    f"iterator (base_byte={state['vad_base_byte']})")

            new_audio = await session.extract_new_audio_for_vad()
            if new_audio is None or len(new_audio) == 0:
                continue

            try:
                events = state["streaming_vad"].feed(new_audio)
            except Exception as e:
                logger.error(f"[{session_id}] VAD monitor feed error: {e}")
                continue

            for evt in events:
                if "start" in evt and not state["vad_speech_active"]:
                    state["vad_speech_active"] = True
                    audio_start_ms = int(evt["start"] / 16.0)
                    speech_start_sample = evt["start"]
                    async with session.lock:
                        session.speech_start_byte = state["vad_base_byte"] + int(
                            speech_start_sample) * 2
                        session.last_triggered_chunk_id = 0
                    item_id_counter += 1
                    current_item_id = f"{session_id}_item_{item_id_counter}"
                    logger.info(
                        f"[{session_id}] VAD: speech_started at {audio_start_ms}ms "
                        f"(byte_offset={session.speech_start_byte})")
                    await ws_send(websocket, {
                        "type": "input_audio_buffer.speech_started",
                        "audio_start_ms": audio_start_ms,
                        "item_id": current_item_id,
                    })

                elif "end" in evt and state["vad_speech_active"]:
                    state["vad_speech_active"] = False
                    audio_end_ms = int(evt["end"] / 16.0)
                    logger.info(
                        f"[{session_id}] VAD: speech_stopped at {audio_end_ms}ms")
                    await ws_send(websocket, {
                        "type": "input_audio_buffer.speech_stopped",
                        "audio_end_ms": audio_end_ms,
                        "item_id": f"{session_id}_item_{item_id_counter}",
                    })

                    async with session.lock:
                        buf_bytes = len(session.buffer)
                        buf_duration = buf_bytes / (session.sr * 2)
                        session.commit_pending_count += 1
                        session.is_committing = True
                    logger.info(
                        f"[{session_id}] VAD auto-commit: "
                        f"buffer={buf_bytes}B ({buf_duration:.2f}s)")

                    await ws_send(websocket, {
                        "type": "input_audio_buffer.committed",
                        "item_id": f"{session_id}_item_{item_id_counter}",
                    })

                    state["streaming_vad"].reset()
                    async with session.lock:
                        state["vad_base_byte"] = session.vad_read_pos

    processor_task = asyncio.create_task(inference_processor())
    vad_monitor_task = asyncio.create_task(vad_monitor())

    try:
        async for raw_msg in websocket:
            try:
                msg = json.loads(raw_msg)
            except Exception:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "input_audio_buffer.append":
                b64_audio = msg.get("audio", "")
                await session.append_audio(b64_audio)

            elif msg_type == "input_audio_buffer.commit":
                async with session.lock:
                    buf_bytes = len(session.buffer)
                    buf_duration = buf_bytes / (session.sr * 2)
                    session.commit_pending_count += 1
                    session.is_committing = True
                logger.info(
                    f"[{session_id}] Received COMMIT. "
                    f"buffer={buf_bytes}B ({buf_duration:.2f}s), "
                    f"chunk_id={session.last_triggered_chunk_id}, "
                    f"accumulated_text=\"{session.accumulated_text[:60]}\"")

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
                        session.prompt = instructions_val
                    updated_fields.append(
                        f"instructions=\"{session.prompt[:40]}\"")

                # audio.input.turn_detection
                audio_cfg = sess_cfg.get("audio", {})
                input_cfg = audio_cfg.get("input", {})
                td_obj = input_cfg.get("turn_detection")
                if td_obj is not None and isinstance(td_obj, dict):
                    td_type = td_obj.get("type")
                    if td_type in ("server_vad", "none"):
                        async with session.lock:
                            session.turn_detection = td_type
                        state["vad_speech_active"] = False  # 切换模式时重置，避免状态残留
                        updated_fields.append(f"turn_detection={td_type}")

                # --- extra 自定义扩展字段 ---
                rollback_dict = extra_cfg.get("rollback")
                if rollback_dict is not None:
                    new_rb = inference_logic.RollbackConfig.from_dict(
                        rollback_dict)
                    async with session.lock:
                        session.rollback_config = new_rb
                    updated_fields.append(
                        f"rollback={new_rb.strategy}/{new_rb.value}")

                chunk_ms_val = extra_cfg.get("chunk_ms")
                if chunk_ms_val is not None:
                    async with session.lock:
                        session.chunk_ms = max(
                            200, min(int(chunk_ms_val), 10000))
                    updated_fields.append(f"chunk_ms={session.chunk_ms}")

                use_history_val = extra_cfg.get("use_history")
                if use_history_val is not None:
                    async with session.lock:
                        session.use_history = bool(use_history_val)
                    updated_fields.append(f"use_history={session.use_history}")

                if updated_fields:
                    logger.info(
                        f"[{session_id}] Session updated: {', '.join(updated_fields)}")
                await ws_send(websocket, {
                    "type": "session.updated",
                    "session": _build_session_snapshot(session, session_id),
                })

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[{session_id}] WebSocket disconnected.")
    except Exception:
        logger.error(f"[{session_id}] Error: {traceback.format_exc()}")
    finally:
        state["keep_processing"] = False
        processor_task.cancel()
        vad_monitor_task.cancel()
        logger.info(f"[{session_id}] Cleaned up.")

# ==============================================
# WebSocket 路由分发 (支持路径匹配)
# ==============================================


async def health_request_handler(connection, request):
    """
    websockets 12+ 新 API: process_request(connection, request) -> WsResponse | None

    拦截 HTTP GET /health 请求，返回模型就绪状态。
    同时拦截所有非 WebSocket upgrade 请求（如浏览器刷新时发出的普通 HTTP GET、
    前端健康轮询），主动返回 HTTP 响应，避免 websockets 抛出 InvalidUpgrade 错误日志。

      返回 None        → 继续走 WebSocket 升级握手
      返回 WsResponse  → 直接作为 HTTP 响应发回，跳过升级
    """
    path = request.path

    if path == "/health":
        body = json.dumps({
            "ready": engine_ready,
            "stage": engine_loading_stage,
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

    # 检查是否为合法的 WebSocket 升级请求。
    # 浏览器刷新页面时会先发普通 HTTP GET（Connection: keep-alive），
    # 直接让 websockets 处理会抛出 InvalidUpgrade 并记录无意义的 ERROR 日志。
    conn_hdr = request.headers.get("Connection", "")
    upgrade_hdr = request.headers.get("Upgrade", "")
    if "upgrade" not in conn_hdr.lower() or upgrade_hdr.lower() != "websocket":
        logger.debug(
            f"Non-WebSocket HTTP {request.method if hasattr(request, 'method') else 'GET'} "
            f"to {path} (Connection: {conn_hdr!r}) — returning 426")
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


async def ws_router(websocket):
    """根据请求路径分发到对应 handler，兼容前端 ws://host:port/v1/realtime"""
    path = websocket.request.path if hasattr(
        websocket, 'request') else getattr(
        websocket, 'path', '/')
    if path == "/v1/realtime" or path == "/":
        await realtime_handler(websocket)
    else:
        await websocket.close(4004, f"Unknown path: {path}")

# ==============================================
# Engine Initialization & Entrypoint
# ==============================================


async def init_engine(
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.75):
    global engine, processor, model_name, engine_ready, engine_loading_stage
    model_name = model

    logger.info("========== 模型初始化开始 (过程可能需要数分钟，请耐心等待...) ==========")

    engine_loading_stage = "1/3 加载 Processor"
    logger.info(f"-> {engine_loading_stage}: {model}")
    start_time = time.time()

    processor = Qwen3OmniMoeProcessor.from_pretrained(model)
    logger.info(f"<- 1/3 Processor 加载完成，耗时: {time.time() - start_time:.2f}s")

    stage_configs_path = os.path.join(
        os.path.dirname(__file__),
        "conf",
        "qwen3_omni_moe_asr_only.yaml")
    if not os.path.isfile(stage_configs_path):
        stage_configs_path = None

    engine_kwargs = dict(model=model, trust_remote_code=True)
    if stage_configs_path:
        engine_kwargs["stage_configs_path"] = stage_configs_path

    engine_loading_stage = "2/3 初始化推理引擎 (加载模型权重，耗时最久)"
    logger.info(
        f"-> {engine_loading_stage}，准备分配 {tensor_parallel_size} 张 GPU ...")
    vllm_start_time = time.time()

    engine = AsyncOmni(**engine_kwargs)

    # ==========================
    # 解决 vLLM 的“假就绪”问题：
    # AsyncOmni 实例化很快，但底层多进程 Worker 可能还在加载极大的 MoE 权重和 profiling。
    # 必须执行一次假数据预热，等待预热完全走通，才能对前端开放服务。
    # ==========================
    engine_loading_stage = "3/4 执行引擎预热 (分配 KV Cache，约需1-2分钟)"
    logger.info(f"-> {engine_loading_stage} ...")
    warmup_t0 = time.time()
    try:
        # 构造一条极短的假文本消息进行预热
        messages = [{"role": "user", "content": "你好"}]
        prompt_formatted = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        vllm_inputs = {"prompt": prompt_formatted}
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        sampling_params_list = [
            sp] * (len(engine.stage_list) if hasattr(engine, 'stage_list') else 1)

        async for _ in engine.generate(
            prompt=vllm_inputs,
            request_id="warmup-1",
            sampling_params_list=sampling_params_list
        ):
            pass
        logger.info(f"<- 3/4 引擎预热成功！耗时: {time.time() - warmup_t0:.2f}s")
    except Exception as e:
        logger.error(f"引擎预热失败，服务可能不可用: {e}")

    logger.info(
        f"<- 2/3 vLLM 引擎加载完成，含预热耗时: {time.time() - vllm_start_time:.2f}s")
    logger.info(f"-> 4/4 模型初始化全部完成！总耗时: {time.time() - start_time:.2f}s")
    logger.info("========== 服务就绪 ==========")

    engine_loading_stage = "就绪"
    engine_ready = True


async def run_server(
        model: str,
        host: str,
        port: int,
        tp: int,
        gpu_mem: float,
        save_audio_flag: bool,
        rollback_strategy: str = "none",
        rollback_value: float = 0.0,
        audio_dir: str = "saved_audios"):
    global save_audio, default_rollback_config, model_name, audio_save_dir
    model_name = model
    save_audio = save_audio_flag
    audio_save_dir = audio_dir
    default_rollback_config = inference_logic.RollbackConfig(
        rollback_strategy, rollback_value)
    if default_rollback_config.strategy != "none":
        logger.info(
            f"Default rollback: strategy={rollback_strategy}, value={rollback_value}")
    if save_audio:
        logger.info(
            f"Audio saving is ENABLED. Audios will be saved to ./{audio_save_dir}/")

    # 先启动 WebSocket 服务（接受连接），模型在后台并行加载
    logger.info(f"正在启动 WebSocket 服务，监听 {host}:{port} （模型将在后台加载）...")

    async def _safe_init():
        try:
            await init_engine(model, tp, gpu_mem)
        except Exception:
            logger.error(f"模型初始化失败:\n{traceback.format_exc()}")
    asyncio.create_task(_safe_init())

    async with websockets.serve(ws_router, host, port, process_request=health_request_handler):
        logger.info(
            f"WebSocket 服务已启动: ws://{host}:{port}/v1/realtime  健康检查: http://{host}:{port}/health")
        await asyncio.Future()  # 永久运行，直到被 Ctrl+C 中断

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Qwen3-Omni Realtime WebSocket ASR')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='qwen3-omni',
        help='Model path')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.75)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument(
        '--save-audio',
        action='store_true',
        help='Save received audio to disk as wav files')
    parser.add_argument(
        '--audio-save-dir',
        type=str,
        default='saved_audios',
        help='Directory to save audio files when --save-audio is enabled')
    parser.add_argument(
        '--rollback-strategy',
        type=str,
        default='none',
        choices=[
            'none',
            'ratio',
            'chars',
            'words'],
        help='Default rollback strategy for incremental inference')
    parser.add_argument(
        '--rollback-value',
        type=float,
        default=0.0,
        help='Rollback value: ratio(0~1) for "ratio", char count for "chars", word count for "words"')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    asyncio.run(run_server(
        args.model, args.host, args.port, args.tensor_parallel_size,
        args.gpu_memory_utilization, args.save_audio,
        args.rollback_strategy, args.rollback_value,
        args.audio_save_dir,
    ))
