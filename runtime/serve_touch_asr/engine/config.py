# Copyright (c) 2026 Pengshen Zhang
"""Inference Config: YAML 热配置加载与校验。

- Pydantic models: 校验 prompt、sampling_params、VAD、history_rollback 等配置
- InferenceConfigCache: 按 mtime/hash 缓存并热加载 inference_config.yaml
- ServerConfig: 合并启动参数和热配置，提供推理循环读取入口
- 只处理配置解析，不直接修改 session 或 engine 状态
"""
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                      field_validator)

logger = logging.getLogger("RealtimeASR")


class SamplingParamsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    temperature: float = Field(default=0.01, ge=0.0)
    top_p: float = Field(default=0.1, ge=0.0, le=1.0)
    top_k: int = Field(default=1, ge=-1)
    max_tokens: int = Field(default=512, gt=0)
    seed: int = 42
    repetition_penalty: float = Field(default=1.05, gt=0.0)


class SilenceSkipConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = True
    frame_ms: int = Field(default=20, gt=0)
    rms_threshold: float = Field(default=0.003, ge=0.0)
    min_voiced_frames: int = Field(default=1, ge=0)


class TurnDetectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: str = "none"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    silence_duration_ms: int = Field(default=700, ge=0)
    prefix_padding_ms: int = Field(default=300, ge=0)


class HistoryRollbackConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    strategy: str = "none"
    value: float = 0.0

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        allowed = {
            "none", "ratio", "chars", "words", "tokens"}
        if v not in allowed:
            raise ValueError(
                f"unsupported history_rollback strategy: {v}")
        return v


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    log_level: str = "INFO"
    prompt: str = "请转录这段音频。"
    chunk_ms: int = Field(default=1000, gt=0)
    sampling_params: SamplingParamsConfig = Field(
        default_factory=SamplingParamsConfig)
    silence_skip: SilenceSkipConfig = Field(
        default_factory=SilenceSkipConfig)
    turn_detection: TurnDetectionConfig = Field(
        default_factory=TurnDetectionConfig)
    history_rollback: HistoryRollbackConfigModel = Field(
        default_factory=HistoryRollbackConfigModel)
    history_reset_chunk_num: int = Field(default=0, ge=0)
    min_history_chars: int = Field(default=0, ge=0)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"unsupported log_level: {v}")
        return level


class SslConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    enabled: bool = False
    cert_file: str = "cert.pem"
    key_file: str = "key.pem"
    auto_generate: bool = True


class StartupConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str = "qwen3-omni"
    model_type: str = "qwen3-omni"
    host: str = "0.0.0.0"
    port: int = Field(default=8001, gt=0)
    gpu_ids: str = "0"
    tensor_parallel_size: int = Field(default=1, gt=0)
    gpu_memory_utilization: float = Field(default=0.75, gt=0.0, le=1.0)
    save_audio: bool = False
    audio_save_dir: str = "saved_audios"
    history_rollback_strategy: str = "none"
    history_rollback_value: float = 0.0
    ssl: SslConfig = Field(default_factory=SslConfig)


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    startup: StartupConfig = Field(default_factory=StartupConfig)
    runtime: InferenceConfig = Field(default_factory=InferenceConfig)


def _stable_fingerprint(raw_cfg: Any) -> str:
    payload = json.dumps(
        raw_cfg or {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_server_config(raw_cfg: Any) -> tuple[dict, Optional[dict]]:
    if not isinstance(raw_cfg, dict):
        return {}, None
    runtime_raw = raw_cfg.get("runtime")
    if isinstance(runtime_raw, dict):
        startup_raw = raw_cfg.get("startup")
        return runtime_raw, startup_raw if isinstance(startup_raw, dict) else {}
    return raw_cfg, None


@dataclass
class InferenceConfigCache:
    """YAML 配置热加载缓存。

    - path 由外部注入，不硬编码
    - SHA-256 内容指纹检测变更（非 mtime）
    - min_check_interval_sec 节流，避免高频 I/O
    - on_reload 回调在锁外执行
    - 返回 frozen Pydantic 对象
    """
    path: Path
    on_reload: Optional[Callable[["InferenceConfig"], None]] = field(
        default=None, repr=False)
    min_check_interval_sec: float = 10.0

    cache: "InferenceConfig" = field(
        default_factory=InferenceConfig, repr=False)
    _fingerprint: str = field(default="", repr=False)
    _startup_fingerprint: str = field(default="", repr=False)
    _last_check_ts: float = field(default=0.0, repr=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False)

    def load(self) -> "InferenceConfig":
        """读取配置。节流 + 指纹检测 + 锁外回调。"""
        now = time.monotonic()
        if (now - self._last_check_ts) < self.min_check_interval_sec:
            return self.cache

        callback = None
        snapshot = None
        with self._lock:
            self._last_check_ts = now
            try:
                raw_cfg = yaml.safe_load(
                    self.path.read_text(encoding="utf-8")) or {}
                runtime_raw, startup_raw = _split_server_config(raw_cfg)
                fp = _stable_fingerprint(runtime_raw)
                startup_fp = (
                    _stable_fingerprint(startup_raw)
                    if startup_raw is not None else "")
                if (
                    startup_raw is not None
                    and self._startup_fingerprint
                    and startup_fp != self._startup_fingerprint
                ):
                    logger.warning(
                        "startup config changed; restart required")
                self._startup_fingerprint = startup_fp
                if fp != self._fingerprint:
                    parsed = InferenceConfig.model_validate(
                        runtime_raw)
                    self.cache = parsed
                    self._fingerprint = fp
                    callback = self.on_reload
                    snapshot = parsed
            except FileNotFoundError:
                pass
            except (yaml.YAMLError, ValidationError) as exc:
                logger.warning(
                    f"Config reload failed, keeping previous:"
                    f" {exc}")
        if callback and snapshot is not None:
            callback(snapshot)
        return self.cache
