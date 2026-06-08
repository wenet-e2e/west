# Copyright (c) 2026 Pengshen Zhang
"""Realtime Session: 单个 WebSocket 连接的会话生命周期。

- 持有 session_id、并发锁、保存音频开关
- 持有 commit/turn_detection/VAD 偏移等服务层并发协调字段
- 持有算法层状态 asr: StreamingAsrState
- 提供 append_audio_bytes / take_audio_up_to_chunk / drain_new_audio / reset
"""
import asyncio
import logging
import os
import time
from typing import Optional

import numpy as np
import soundfile as sf
from model.history_rollback import HistoryRollbackConfig
from session.state import StreamingAsrState

logger = logging.getLogger("RealtimeASR")


class RealtimeSession:
    """单 WebSocket 连接的会话上下文。"""

    def __init__(
        self,
        session_id: str,
        sr: int = 16000,
        *,
        default_history_rollback_config: HistoryRollbackConfig,
        save_audio: bool = False,
        audio_save_dir: str = "saved_audios",
    ):
        self.session_id = session_id
        self.sr = sr

        # 算法层状态
        self.asr = StreamingAsrState(default_history_rollback_config)

        # 服务层并发协调
        self.lock = asyncio.Lock()

        # 落盘配置
        self.save_audio = save_audio
        self.audio_save_dir = audio_save_dir

        # commit 控制
        self.is_committing = False
        self.commit_pending_count = 0

        # 前导静音跳过标志
        self.speech_detected = False

        # VAD 话轮检测状态
        self.turn_detection: Optional[str] = None
        self.vad_read_pos: int = 0
        self.speech_start_byte: int = 0

    # ---- 音频缓冲方法 ----

    async def append_audio_bytes(self, delta_bytes: bytes):
        try:
            if len(delta_bytes) == 0:
                return
            if len(delta_bytes) % 2 != 0:
                logger.warning(
                    f"[{self.session_id}] Audio packet size "
                    f"{len(delta_bytes)}B is not even - not valid PCM16.")
                return

            async with self.lock:
                is_first = len(self.asr.buffer) == 0
                self.asr.buffer.extend(delta_bytes)
                buf_bytes = len(self.asr.buffer)
                buf_duration = buf_bytes / (self.sr * 2)

                if is_first:
                    samples = len(delta_bytes) // 2
                    pkt_ms = samples / self.sr * 1000
                    peek = np.frombuffer(delta_bytes, dtype=np.int16)
                    abs_max = int(np.abs(peek).max())
                    logger.info(
                        f"[{self.session_id}] FirstAudioPacket: "
                        f"{len(delta_bytes)}B ({samples} samples, "
                        f"{pkt_ms:.0f}ms) peak={abs_max} "
                        f"{'(silence?)' if abs_max < 10 else ''}")

                if buf_bytes % (self.sr * 2) == 0:
                    logger.debug(
                        f"[{self.session_id}] "
                        f"chunk={self.asr.chunk_id} "
                        f"AudioBuffer: append={len(delta_bytes)}B "
                        f"total={buf_bytes}B ({buf_duration:.1f}s)")
        except Exception as e:
            logger.error(
                f"[{self.session_id}] Audio decode error: {e}")

    async def take_audio_up_to_chunk(
            self,
            chunk_ms: int,
            force_all: bool = False,
            is_pcm16: bool = True,
            use_speech_offset: bool = False):
        async with self.lock:
            bytes_per_sample = 2 if is_pcm16 else 4
            chunk_bytes = int(
                self.sr * bytes_per_sample * (chunk_ms / 1000))

            offset = self.speech_start_byte if use_speech_offset else 0
            buffer = self.asr.buffer
            effective_buf = buffer[offset:]
            available_chunks = len(effective_buf) // chunk_bytes

            buffer_len_at_extract = len(buffer)

            target_pcm_bytes = None
            if force_all and len(effective_buf) > 0:
                target_pcm_bytes = bytes(effective_buf)
            else:
                next_chunk_id = self.asr.chunk_id + 1
                if available_chunks >= next_chunk_id:
                    self.asr.chunk_id = next_chunk_id
                    valid_len = next_chunk_id * chunk_bytes
                    target_pcm_bytes = bytes(effective_buf[:valid_len])

            if target_pcm_bytes:
                valid_bytes_len = (
                    len(target_pcm_bytes)
                    // bytes_per_sample) * bytes_per_sample
                target_pcm_bytes = target_pcm_bytes[:valid_bytes_len]

                if is_pcm16:
                    arr = np.frombuffer(
                        target_pcm_bytes,
                        dtype=np.int16).astype(
                        np.float32) / 32768.0
                else:
                    arr = np.frombuffer(
                        target_pcm_bytes, dtype=np.float32)

                audio_sec = len(arr) / self.sr
                logger.debug(
                    f"[{self.session_id}] "
                    f"chunk={self.asr.chunk_id} "
                    f"ExtractAudio: {audio_sec:.2f}s ({len(arr)} samples) "
                    f"force={force_all} offset={offset}B "
                    f"range=[{arr.min():.4f}, {arr.max():.4f}]")
                return arr, buffer_len_at_extract
            return None, buffer_len_at_extract

    async def drain_new_audio(self) -> Optional[np.ndarray]:
        """提取 VAD 还未读过的新增音频片段 (float32, 16kHz)。"""
        async with self.lock:
            buffer = self.asr.buffer
            if self.vad_read_pos >= len(buffer):
                return None
            new_bytes = bytes(buffer[self.vad_read_pos:])
            self.vad_read_pos = len(buffer)
        valid_len = (len(new_bytes) // 2) * 2
        if valid_len == 0:
            return None
        arr = np.frombuffer(
            new_bytes[:valid_len],
            dtype=np.int16).astype(np.float32) / 32768.0
        return arr

    async def reset(self, keep_buffer_from: int = 0):
        async with self.lock:
            buffer = self.asr.buffer
            if self.save_audio and len(buffer) > 0:
                try:
                    date_dir = os.path.join(
                        self.audio_save_dir,
                        time.strftime("%Y%m%d"))
                    os.makedirs(date_dir, exist_ok=True)
                    offset = (self.speech_start_byte
                              if self.speech_start_byte > 0 else 0)
                    if keep_buffer_from > 0:
                        save_buf = buffer[offset:keep_buffer_from]
                    else:
                        save_buf = buffer[offset:]
                    valid_bytes_len = (len(save_buf) // 2) * 2
                    raw_bytes = bytes(save_buf[:valid_bytes_len])
                    if len(raw_bytes) > 0:
                        audio_arr = np.frombuffer(
                            raw_bytes, dtype=np.int16).astype(
                            np.float32) / 32768.0
                        timestamp = time.strftime("%H%M%S")
                        save_path = os.path.join(
                            date_dir,
                            f"{self.session_id}_{timestamp}.wav")

                        audio_dur = len(audio_arr) / self.sr
                        transcript_snap = self.asr.accumulated_text[:80]
                        sid_snap = self.session_id

                        def _save_task():
                            sf.write(save_path, audio_arr, 16000)
                            logger.info(
                                f"[{sid_snap}] Audio saved to "
                                f"{save_path} ({audio_dur:.2f}s) "
                                f"text=\"{transcript_snap}\"")

                        asyncio.get_running_loop().run_in_executor(
                            None, _save_task)
                except Exception as e:
                    logger.error(
                        f"[{self.session_id}] Failed to dispatch "
                        f"audio save task: {e}")

            if keep_buffer_from > 0 and keep_buffer_from < len(buffer):
                self.asr.buffer = bytearray(
                    buffer[keep_buffer_from:])
            else:
                self.asr.buffer = bytearray()

            self.asr.chunk_id = 0
            if self.commit_pending_count > 0:
                self.commit_pending_count -= 1
            self.is_committing = (self.commit_pending_count > 0)
            self.asr.accumulated_text = ""
            self.asr.confirmed_text = ""
            self.asr.language = None
            self.speech_detected = False
            self.vad_read_pos = 0
            self.speech_start_byte = 0
