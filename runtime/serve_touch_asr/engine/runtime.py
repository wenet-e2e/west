# Copyright (c) 2026 Pengshen Zhang
"""Engine Runtime: 进程级共享状态容器 + 启动参数。

- ServerSettings: CLI 启动参数（frozen dataclass）
- ServiceSettings: 运行时可变配置（model_name, save_audio 等）
- EnginePhase / EngineRuntime: 引擎生命周期状态
- ServiceRuntime: 聚合模型生命周期、ready/error 状态和配置缓存
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from engine.config import InferenceConfigCache
from model.history_rollback import HistoryRollbackConfig
from model.vad import SileroVadProvider

# ── Server Settings (CLI 启动参数) ────────────────────


@dataclass(frozen=True)
class ServerSettings:
    model: str
    host: str
    port: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    save_audio: bool
    history_rollback_strategy: str = "none"
    history_rollback_value: float = 0.0
    audio_save_dir: str = "saved_audios"
    model_type: str = "qwen3-omni"
    use_ssl: bool = False
    inference_config_path: Path = Path("conf/inference_config.yaml")
    silero_repo: Path = Path("/path/to/silero-vad")
    log_dir: Path = Path("logs")
    web_page: Path = Path("./index.html")


# ── Engine Lifecycle ─────────────────────────────────

class EnginePhase(str, Enum):
    NOT_STARTED = "not_started"
    LOADING_PROCESSOR = "loading_processor"
    LOADING_ENGINE = "loading_engine"
    WARMING_UP = "warming_up"
    READY = "ready"
    FAILED = "failed"


@dataclass
class EngineRuntime:
    """vLLM 引擎生命周期状态（运行中可变）。"""
    engine: Optional[Any] = None
    processor: Optional[Any] = None
    process_mm_info: Optional[Any] = None
    phase: EnginePhase = EnginePhase.NOT_STARTED
    message: str = "等待启动"
    error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.phase == EnginePhase.READY

    @ready.setter
    def ready(self, value: bool) -> None:
        self.phase = EnginePhase.READY if value else EnginePhase.NOT_STARTED
        if value:
            self.error = None

    def set_phase(self, phase: EnginePhase, message: str) -> None:
        self.phase = phase
        self.message = message
        if phase != EnginePhase.FAILED:
            self.error = None

    def set_failed(self, error: str) -> None:
        self.phase = EnginePhase.FAILED
        self.message = "加载失败"
        self.error = error


# ── Service Settings (运行时可变) ─────────────────────

@dataclass
class ServiceSettings:
    """启动参数（init 后基本不变）。"""
    model_name: str = "qwen3-omni"
    model_type: str = "qwen3-omni"
    save_audio: bool = False
    audio_save_dir: str = "saved_audios"
    default_history_rollback_config: HistoryRollbackConfig = field(
        default_factory=HistoryRollbackConfig)


# ── Service Runtime (进程级顶层容器) ──────────────────

class ServiceRuntime:
    """进程级顶层容器，拥有所有单例。"""

    def __init__(
        self,
        service_settings: Optional[ServiceSettings] = None,
        engine_runtime: Optional[EngineRuntime] = None,
        inference_config_cache: Optional[InferenceConfigCache] = None,
        vad_provider: Optional[SileroVadProvider] = None,
        *,
        settings: Optional[ServiceSettings] = None,
        engine_state: Optional[EngineRuntime] = None,
        inference_cfg: Optional[InferenceConfigCache] = None,
    ):
        self.service_settings = (
            service_settings or settings or ServiceSettings())
        self.engine_runtime = (
            engine_runtime or engine_state or EngineRuntime())
        self.inference_config_cache = (
            inference_config_cache or inference_cfg)
        self.vad_provider = vad_provider

    @property
    def settings(self) -> ServiceSettings:
        return self.service_settings

    @settings.setter
    def settings(self, value: ServiceSettings) -> None:
        self.service_settings = value

    @property
    def engine_state(self) -> EngineRuntime:
        return self.engine_runtime

    @engine_state.setter
    def engine_state(self, value: EngineRuntime) -> None:
        self.engine_runtime = value

    @property
    def inference_cfg(self) -> Optional[InferenceConfigCache]:
        return self.inference_config_cache

    @inference_cfg.setter
    def inference_cfg(self, value: InferenceConfigCache) -> None:
        self.inference_config_cache = value
