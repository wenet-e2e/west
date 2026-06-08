# Copyright (c) 2026 Pengshen Zhang
"""Session State: 单连接内所有状态容器。

- VadPhase / VadState: VAD 检测阶段状态机
- ConnectionLoopState: inference/VAD 后台协程共享的运行状态
- StreamingAsrState: 算法层会话状态（累计音频 + 识别文本 + chunk 进度）
- ConnectionContext: 单连接 worker 的依赖聚合
"""
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Optional

from model.history_rollback import HistoryRollbackConfig

# ── VAD State ────────────────────────────────────────


class VadPhase(str, Enum):
    DISABLED = "disabled"
    IDLE = "idle"
    SPEAKING = "speaking"


@dataclass
class VadState:
    """单个连接内 server_vad 检测器和语音阶段状态。"""

    phase: VadPhase = VadPhase.DISABLED
    detector: Any = None
    base_byte: int = 0

    @property
    def speech_active(self) -> bool:
        return self.phase == VadPhase.SPEAKING

    def attach_detector(self, detector: Any, base_byte: int) -> None:
        self.detector = detector
        self.base_byte = base_byte
        self.phase = VadPhase.IDLE

    def mark_started(self) -> None:
        if self.detector is not None:
            self.phase = VadPhase.SPEAKING

    def mark_stopped(self) -> None:
        if self.detector is not None:
            self.phase = VadPhase.IDLE

    def disable(self) -> None:
        self.detector = None
        self.phase = VadPhase.DISABLED


# ── Connection Loop State ────────────────────────────

@dataclass
class ConnectionLoopState:
    """单个 WebSocket 连接内 inference/VAD 后台协程共享的运行状态。"""

    keep_processing: bool = True
    vad: VadState = field(default_factory=VadState)

    @property
    def vad_speech_active(self) -> bool:
        return self.vad.speech_active

    @vad_speech_active.setter
    def vad_speech_active(self, value: bool) -> None:
        if value:
            self.vad.mark_started()
        else:
            self.vad.mark_stopped()

    @property
    def vad_detector(self) -> Any:
        return self.vad.detector

    @vad_detector.setter
    def vad_detector(self, value: Any) -> None:
        if value is None:
            self.vad.disable()
        else:
            self.vad.attach_detector(value, self.vad.base_byte)

    @property
    def vad_base_byte(self) -> int:
        return self.vad.base_byte

    @vad_base_byte.setter
    def vad_base_byte(self, value: int) -> None:
        self.vad.base_byte = value


# ── Streaming ASR State ──────────────────────────────

@dataclass
class StreamingAsrState:
    """单路音频流的 ASR 推理状态。

    账本：累计音频 + 累计识别文本 + chunk 进度。
    """

    default_history_rollback_config: InitVar[HistoryRollbackConfig]

    buffer: bytearray = field(default_factory=bytearray)
    chunk_id: int = 0
    accumulated_text: str = ""
    confirmed_text: str = ""
    language: Optional[str] = None
    history_rollback_config: HistoryRollbackConfig = field(
        init=False)
    chunk_ms: Optional[int] = None
    prompt: Optional[str] = None
    use_history: bool = True
    history_reset_chunk_num: int = 0
    min_history_chars: int = 0

    def __post_init__(
        self,
        default_history_rollback_config: HistoryRollbackConfig,
    ):
        self.history_rollback_config = HistoryRollbackConfig(
            default_history_rollback_config.strategy,
            default_history_rollback_config.value,
        )


# ── Connection Context ───────────────────────────────

@dataclass
class ConnectionContext:
    """单个 WebSocket 连接处理所需依赖的轻量容器。"""

    session_id: str
    websocket: Any
    realtime_session: Any  # session.session.RealtimeSession
    connection_loop_state: "ConnectionLoopState"
    service_runtime: Any   # engine.runtime.ServiceRuntime
    realtime_sender: Any   # session.sender.RealtimeSender
