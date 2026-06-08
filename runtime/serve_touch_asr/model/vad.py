# Copyright (c) 2026 Pengshen Zhang
"""VAD: Provider/Detector 分层架构 + 前导静音跳过。

- SileroVadProvider: 加载 Silero model，创建 per-session detector
- VadDetector: 单 session 的流式 VAD 检测器 (feed/reset)
- detect_leading_silence: 基于 RMS/帧数判断前导静音（none 模式）
- Provider 是进程级资源，Detector 是连接级状态，零业务全局状态
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, List

import numpy as np
import torch
from engine.config import SilenceSkipConfig, TurnDetectionConfig

logger = logging.getLogger("RealtimeASR")


def _prepend_silero_repo_to_syspath(silero_repo: Path) -> None:
    silero_src = str(silero_repo / "src")
    if silero_src not in sys.path:
        sys.path.insert(0, silero_src)


# ── Provider ─────────────────────────────────────────

@dataclass
class SileroVadProvider:
    """Silero VAD 模型 Provider。

    - silero_repo / sr 由外部注入（服务启动时确定）
    - load() 显式加载 JIT model，不在构造器中偷偷做 I/O
    - create_detector() 创建 per-session VadDetector
    """
    silero_repo: Path
    sr: int = 16000

    _model: Any = field(init=False, default=None, repr=False)
    _vad_iterator_cls: Any = field(
        init=False, default=None, repr=False)

    SILERO_WINDOW_SAMPLES: ClassVar[int] = 512

    def load(self) -> None:
        """显式加载 Silero JIT model。服务启动时调用一次。"""
        _prepend_silero_repo_to_syspath(self.silero_repo)
        if os.environ.get("SILERO_REPO") is None:
            os.environ["SILERO_REPO"] = str(self.silero_repo)

        from silero_vad.utils_vad import VADIterator  # noqa: E402
        from silero_vad.utils_vad import init_jit_model

        model_path = str(
            self.silero_repo / "src" / "silero_vad"
            / "data" / "silero_vad.jit")
        self._model = init_jit_model(model_path)
        self._vad_iterator_cls = VADIterator
        logger.info(f"Silero VAD loaded from {model_path}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def create_detector(
        self,
        session_id: str,
        td_cfg: TurnDetectionConfig,
    ) -> "VadDetector":
        """创建 per-session VAD detector。"""
        if not self.is_loaded:
            raise RuntimeError(
                "SileroVadProvider not loaded. "
                "Call load() before create_detector().")

        iterator = self._vad_iterator_cls(
            self._model,
            threshold=td_cfg.threshold,
            sampling_rate=self.sr,
            min_silence_duration_ms=td_cfg.silence_duration_ms,
            speech_pad_ms=td_cfg.prefix_padding_ms,
        )
        detector = VadDetector(
            session_id=session_id,
            iterator=iterator,
            sr=self.sr,
            window=self.SILERO_WINDOW_SAMPLES,
        )
        logger.info(
            f"[{session_id}] VadDetector created "
            f"(threshold={td_cfg.threshold}, "
            f"silence={td_cfg.silence_duration_ms}ms)")
        return detector


# ── Detector ─────────────────────────────────────────

class VadDetector:
    """单 session 的 VAD 流式检测器。只做 feed/reset。"""

    def __init__(
        self,
        session_id: str,
        iterator: Any,
        sr: int,
        window: int,
    ):
        self.session_id = session_id
        self._iterator = iterator
        self._sr = sr
        self._window = window
        self._total_samples = 0

    def feed(self, audio_chunk: np.ndarray) -> List[dict]:
        """传入增量音频 (float32, 16kHz)，返回事件列表。"""
        events: List[dict] = []
        w = self._window
        for i in range(0, len(audio_chunk), w):
            seg = audio_chunk[i:i + w]
            if len(seg) < w:
                seg = np.pad(seg, (0, w - len(seg)))
            t = torch.from_numpy(seg)
            result = self._iterator(t)
            if result is not None:
                if 'start' in result:
                    events.append({"start": result['start']})
                elif 'end' in result:
                    events.append({"end": result['end']})
        self._total_samples += len(audio_chunk)
        return events

    def reset(self) -> None:
        """重置 VAD 状态（话轮结束后调用）。"""
        self._total_samples = 0
        if self._iterator is not None:
            try:
                self._iterator.reset_states()
            except Exception:
                pass

    @property
    def engine_name(self) -> str:
        return "silero"


# ── Leading Silence Detection ────────────────────────

@dataclass(frozen=True)
class VadSilenceDecision:
    is_silence: bool
    rms: float
    voiced_frames: int
    total_frames: int
    threshold: float


def detect_leading_silence(
    audio: np.ndarray,
    cfg: SilenceSkipConfig,
    sr: int = 16000,
) -> VadSilenceDecision:
    """判断一个候选 chunk 是否仍是前导静音。"""
    if audio is None or len(audio) == 0:
        return VadSilenceDecision(True, 0.0, 0, 0, cfg.rms_threshold)

    frame_samples = max(1, int(sr * cfg.frame_ms / 1000))
    total_frames = 0
    voiced_frames = 0

    for i in range(0, len(audio), frame_samples):
        frame = audio[i:i + frame_samples]
        if len(frame) == 0:
            break
        total_frames += 1
        frame_rms = float(np.sqrt(np.mean(frame ** 2)))
        if frame_rms >= cfg.rms_threshold:
            voiced_frames += 1

    chunk_rms = float(np.sqrt(np.mean(audio ** 2)))
    return VadSilenceDecision(
        is_silence=voiced_frames < cfg.min_voiced_frames,
        rms=chunk_rms,
        voiced_frames=voiced_frames,
        total_frames=total_frames,
        threshold=cfg.rms_threshold,
    )
